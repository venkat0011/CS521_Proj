"""Compare a hand-written schedule and all-rule controls on one small CNN.

Single process deliberately reuses TASO's operator-cost cache for comparable
search decisions. Timings here are warm-cache smoke observations, not a fair
performance ranking. No LLM is called.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import traceback


def small_cnn(ts):
    import numpy as np
    rng = np.random.default_rng(521)
    graph = ts.new_graph()
    x = graph.new_input(dims=(1, 8, 16, 16))
    def conv(tensor, kernel):
        dims = (8, 8, kernel, kernel)
        w = graph.new_weight(dims=dims, data=rng.standard_normal(dims).astype('float32'))
        return graph.conv2d(input=tensor, weight=w, strides=(1, 1), padding='SAME')
    left = graph.relu(conv(x, 3))
    right = graph.relu(conv(x, 1))
    merged = graph.add(left, right)
    graph.relu(conv(merged, 3))
    return graph


def observe(graph):
    return {'operators': dict(Counter(graph.get_operator_type(op) for op in graph.get_operator_list())),
            'estimated_ms': graph.cost()}


def aggregate(trace):
    counts = {}
    for phase in trace['phases']:
        for rule in phase['rules']:
            entry = counts.setdefault(rule['rule_index'], {})
            for key, value in rule.items():
                if key not in ('rule_index', 'seconds'):
                    entry[key] = entry.get(key, 0) + value
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--strategy', type=Path, default=Path(__file__).resolve().parents[1] / 'strategies/manual_conv.json')
    parser.add_argument('--alpha', type=float, default=1.5,
                        help='Exploratory threshold held fixed across variants; not a tuned value')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'alpha': args.alpha,
              'notes': ['No LLM calls; skipped rules are not replayed on expanded graphs.',
                        'All variants use the same cached operator costs; warm-cache timings are not a speedup result.',
                        'Final latency observations do not verify numerical equivalence.']}
    try:
        import taso as ts
        from taso.strategy import compile_strategy
        import os
        os.environ['TASO_HOME'] = str(Path(__file__).resolve().parents[1] / 'TASO')
        manual = json.loads(args.strategy.read_text())
        budgets, compiled_rules = compile_strategy(manual, args.alpha)
        total = sum(budgets)
        if total < 2:
            raise ValueError('This comparison needs a budget of at least 2')
        def all_rules(parts, name):
            return {'schema_version': 1, 'name': name, 'phases': [
                {'name': f'phase_{i}', 'rule_ids': 'all', 'expansion_budget': n}
                for i, n in enumerate(parts)]}
        variants = [
            ('legacy_api', None),
            ('all_rules', all_rules([total], 'all_rules')),
            ('split_all_rules', all_rules([total // 2, total - total // 2], 'split_all_rules')),
            ('manual', manual),
        ]
        results = {}
        for name, strategy in variants:
            graph = small_cnn(ts)
            before = observe(graph)
            path = args.out / f'{name}_trace.json'
            path.unlink(missing_ok=True)
            started = time.perf_counter()
            if strategy is None:
                # Legacy TASO expands budget+1 graphs; use B-1 for an exact B comparison.
                result = ts.optimize(graph, alpha=args.alpha, budget=total - 1, print_subst=True)
            else:
                result = ts.optimize_with_strategy(graph, strategy, alpha=args.alpha,
                                                   print_subst=True, trace_file=str(path))
            elapsed = time.perf_counter() - started
            # Input graph may have been deleted by optimize; only access result now.
            observed = observe(result)
            observed.update(before=before, optimize_seconds=elapsed, measured_ms=result.run_time())
            if path.exists():
                observed['trace'] = json.loads(path.read_text())
            results[name] = observed
        baseline = results['all_rules']
        split = results['split_all_rules']
        tolerance = max(1e-8, abs(baseline['estimated_ms']) * 1e-6)
        checks = {
            'legacy_api_same_cost': abs(results['legacy_api']['estimated_ms'] - baseline['estimated_ms']) <= tolerance,
            'split_all_rules_same_cost': abs(split['estimated_ms'] - baseline['estimated_ms']) <= tolerance,
            'split_all_rules_same_expansions': split['trace']['expansions'] == baseline['trace']['expansions'],
            'split_all_rules_same_rule_counters': aggregate(split['trace']) == aggregate(baseline['trace']),
            'budgets_respected': all(r['trace']['expansions'] <= total for r in results.values() if 'trace' in r),
            'manual_rule_filter_respected': all(
                r['attempts'] == 0 for r in results['manual']['trace']['phases'][0]['rules']
                if compiled_rules[0] and r['rule_index'] not in compiled_rules[0]),
            'loaded_full_catalog': baseline['trace']['rule_count'] == 151,
            'assertions_active_rule_matching': any(r['full_matches'] for p in baseline['trace']['phases'] for r in p['rules']),
        }
        report.update(status='completed' if all(checks.values()) else 'check_failed',
                      total_expansion_budget=total, checks=checks, variants=results)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    (args.out / 'manual_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    # Full traces stay in files; keep terminal output short enough to share.
    printable = dict(report)
    if 'variants' in printable:
        printable['variants'] = {name: {k: v for k, v in values.items() if k != 'trace'}
                                 for name, values in report['variants'].items()}
        for name, values in report['variants'].items():
            if 'trace' in values:
                printable['variants'][name]['expansions'] = values['trace']['expansions']
                printable['variants'][name]['phase_expansions'] = [p['expansions'] for p in values['trace']['phases']]
                printable['variants'][name]['admitted'] = sum(r['admitted'] for p in values['trace']['phases'] for r in p['rules'])
    print(json.dumps(printable, indent=2), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

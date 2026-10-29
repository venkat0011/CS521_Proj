"""Isolated GPU worker: prepare one frozen prompt, or evaluate a paired trial."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'llm_context'))
sys.path.insert(0, str(HERE.parent / 'baseline'))
from networks import build_network


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def all_rules(budget):
    return {'schema_version': 1, 'name': 'all_rules', 'phases': [
        {'name': 'all', 'rule_ids': 'all', 'expansion_budget': budget}]}


def observe(graph):
    return {'operators': dict(Counter(graph.get_operator_type(op) for op in graph.get_operator_list())),
            'estimated_ms': graph.cost()}


def environment(ts):
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                      '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = 'unavailable'
    return {'platform': platform.platform(), 'python': sys.version,
            'taso_module': ts.__file__, 'gpu': gpu}


def optimize(ts, graph, strategy, alpha, trace_path, budget):
    started = time.perf_counter()
    result = ts.optimize_with_strategy(graph, strategy, alpha=alpha,
                                      trace_file=str(trace_path), print_subst=False)
    elapsed = time.perf_counter() - started
    trace = json.loads(trace_path.read_text())
    if not 0 < trace['expansions'] <= budget:
        raise RuntimeError('Native executor violated expansion budget')
    if trace['rule_count'] != 151:
        raise RuntimeError(f"Expected 151 rules, loaded {trace['rule_count']}")
    counters = Counter()
    for phase in trace['phases']:
        for rule in phase['rules']:
            counters.update({k: v for k, v in rule.items() if k not in ('rule_index', 'seconds')})
    observation = observe(result)
    observation.update(search_seconds=elapsed, expansions=trace['expansions'],
                       stop_reason=trace['stop_reason'], queue_remaining=trace['queue_remaining'],
                       counters=dict(counters), trace_file=trace_path.name)
    return result, observation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'evaluate'], required=True)
    parser.add_argument('--network', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, help='Frozen network preparation directory (evaluate only)')
    parser.add_argument('--trial', type=int, default=1)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    report = {'status': 'started', 'stage': args.stage, 'network': args.network,
              'budget': cfg['budget'], 'alpha': cfg['alpha']}
    save(args.out / 'worker_result.json', report)
    try:
        os.environ['TASO_HOME'] = str(args.repo.resolve())
        import taso as ts
        report['environment'] = environment(ts)
        baseline = all_rules(cfg['budget'])
        build = lambda: build_network(args.network, ts, args.repo, cfg['weight_seed'])
        if args.stage == 'prepare':
            from run_llm_strategy import ensure_context
            from llm_strategy import request_strategy
            from graph_context import graph_context
            graph, source, info = build()
            report['network_info'] = info
            save(args.out / 'graph.json', graph_context(graph))
            (args.out / 'model_source.py').write_text(source)
            context = ensure_context(args.repo.resolve())
            static = json.loads(context.read_text())
            save(args.out / 'catalog.json', static['catalog'])
            request_strategy(graph, None, context, args.out, cfg['budget'], cfg['alpha'],
                             cfg['max_phases'], source, dry_run=True)
            # A complete baseline preflight happens before any paid call for this network.
            result, observed = optimize(ts, graph, baseline, cfg['alpha'],
                                         args.out / 'preflight_trace.json', cfg['budget'])
            observed['measured_ms'] = result.run_time()
            report['baseline_preflight'] = observed
        else:
            from llm_strategy import validate_strategy
            if args.prepared is None:
                raise ValueError('--prepared is required for evaluation')
            prepared = args.prepared
            catalog = json.loads((prepared / 'catalog.json').read_text())
            strategy = validate_strategy((args.out / 'strategy.json').read_text(), catalog,
                                         cfg['budget'], cfg['max_phases'], cfg['alpha'])
            policies = {'baseline': baseline, 'llm': strategy}
            order = ['baseline', 'llm'] if args.trial % 2 else ['llm', 'baseline']
            report.update(evaluation_order=order, warmup_searches_per_policy=1,
                          notes=['Both policies receive one discarded full-budget search before timed searches.',
                                 'Operator-cost cache is shared within this paired trial, fresh in each trial process.',
                                 'One run_time observation per variant; upstream averages repeated GPU executions.',
                                 'This does not check numerical equivalence.'])
            warmup_started = time.perf_counter()
            for name in order:
                graph, _, _ = build()
                optimize(ts, graph, policies[name], cfg['alpha'],
                         args.out / f'{name}_warmup_trace.json', cfg['budget'])
            report['warmup_search_seconds'] = time.perf_counter() - warmup_started
            results, variants = {}, {}
            report['variants'] = variants
            for name in order:
                graph, _, info = build()
                before = observe(graph)
                results[name], variants[name] = optimize(
                    ts, graph, policies[name], cfg['alpha'],
                    args.out / f'{name}_trace.json', cfg['budget'])
                variants[name]['before'] = before
                report['network_info'] = info
                save(args.out / 'worker_result.json', report)
            # Map each final graph only once: upstream run_time retains its allocations.
            for name in order:
                variants[name]['measured_ms'] = results[name].run_time()
                save(args.out / 'worker_result.json', report)
            report.update(network_info=info, strategy=strategy, variants=variants)
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    save(args.out / 'worker_result.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

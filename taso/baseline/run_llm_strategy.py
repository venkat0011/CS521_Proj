"""Generate one complete OpenRouter strategy, then compare it to all-rule TASO.

Use --dry-run to save the actual prompt without making an API request.
Use --strategy-file to rerun a saved strategy without another model call.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CONTEXT = PROJECT / 'taso' / 'llm_context'
sys.path.insert(0, str(CONTEXT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=PROJECT / 'taso/TASO')
    parser.add_argument('--out', type=Path, default=HERE / 'results')
    parser.add_argument('--budget', type=int, default=24)
    parser.add_argument('--alpha', type=float, default=1.5)
    parser.add_argument('--max-phases', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--model', help='OpenRouter model ID; defaults to OPENROUTER_MODEL or DeepSeek V4 Pro')
    parser.add_argument('--model-source', type=Path, help='Optional real model source; default is the TASO graph builder')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--strategy-file', type=Path, help='Reuse a saved strategy; makes no API call')
    return parser.parse_args(argv)


def ensure_context(repo):
    from prepare_context import prepare_context
    path = CONTEXT / 'prepared/static_context.json'
    if path.exists():
        saved = json.loads(path.read_text())
        hashes = saved.get('source_inputs_sha256', {})
        if hashes and all((repo / name).is_file() and
                          hashlib.sha256((repo / name).read_bytes()).hexdigest() == value
                          for name, value in hashes.items()):
            return path
    prepare_context(repo, path.parent)
    return path


def main():
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.out.resolve() / 'llm' / run_id
    out.mkdir(parents=True, exist_ok=True)
    for stale in ('summary.json', 'baseline_trace.json', 'llm_trace.json'):
        (out / stale).unlink(missing_ok=True)
    report = {'status': 'started', 'mode': 'one_shot_graph_specific',
              'model_calls': 0, 'budget': args.budget, 'alpha': args.alpha,
              'notes': ['No online feedback or GPU measurements are supplied to the model.',
                        'Same-process operator-cost caching makes these warm-cache smoke observations.',
                        'This run is not offline cross-workload strategy synthesis or a numerical equivalence check.']}
    client = None
    try:
        if not args.dry_run and not args.strategy_file and not os.environ.get('OPENROUTER_API_KEY', '').strip():
            raise ValueError('Set OPENROUTER_API_KEY in the terminal before requesting a model strategy.')
        import taso as ts
        from llm_strategy import request_strategy, validate_strategy
        from openrouter_client import OpenRouterClient
        from run_manual_strategy import small_cnn, observe
        os.environ['TASO_HOME'] = str(args.repo.resolve())
        context_path = ensure_context(args.repo.resolve())
        graph = small_cnn(ts)
        report['initial_graph'] = observe(graph)
        if args.strategy_file:
            if args.dry_run:
                raise ValueError('Choose either --dry-run or --strategy-file')
            catalog = json.loads(context_path.read_text())['catalog']
            strategy = validate_strategy(args.strategy_file.read_text(), catalog,
                                         args.budget, args.max_phases, args.alpha)
            (out / 'strategy.json').write_text(json.dumps(strategy, indent=2) + '\n')
        else:
            source = args.model_source.read_text() if args.model_source else (
                'Framework: TASO graph construction (not PyTorch). Actual function used:\n' + inspect.getsource(small_cnn))
            client = OpenRouterClient(args.model, save_to=out, max_tokens=args.max_tokens)
            # A model failure/invalid result stops execution; no hidden repair calls.
            strategy = request_strategy(graph, client, context_path, out, args.budget,
                                        args.alpha, args.max_phases, source, args.dry_run)
            report['inference'] = client.metadata
            if args.dry_run:
                report.update(status='prepared', prompt_file=str(out / 'request.txt'))
            else:
                report['model_calls'] = 1
        if not args.dry_run:
            baseline = {'schema_version': 1, 'name': 'all_rules', 'phases': [
                {'name': 'all', 'rule_ids': 'all', 'expansion_budget': args.budget}]}
            variants = {}
            for name, policy in (('baseline', baseline), ('llm', strategy)):
                current = graph if name == 'baseline' else small_cnn(ts)
                trace_path = out / f'{name}_trace.json'
                started = time.perf_counter()
                result = ts.optimize_with_strategy(current, policy, alpha=args.alpha,
                                                    trace_file=str(trace_path), print_subst=True)
                elapsed = time.perf_counter() - started
                observation = observe(result)
                trace = json.loads(trace_path.read_text())
                observation.update(search_seconds=elapsed, measured_ms=result.run_time(),
                                   expansions=trace['expansions'],
                                   admitted=sum(r['admitted'] for p in trace['phases'] for r in p['rules']))
                variants[name] = observation
            report.update(status='completed', strategy=strategy, variants=variants)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    if client is not None:
        report['model_calls'] = client.request_attempts
        report['inference'] = client.metadata
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print(f'Files saved to: {out}', flush=True)
    return 0 if report['status'] in ('completed', 'prepared') else 1


if __name__ == '__main__':
    raise SystemExit(main())

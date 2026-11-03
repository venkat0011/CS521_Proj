"""Paired checkpoint evaluation inside one isolated GPU worker process."""
import json
import time

from checkpoint_strategy import CheckpointStrategist
from worker import all_rules, observe, optimize, save, search_observation


def evaluate_checkpoint_trial(ts, build, cfg, args, report):
    if args.prepared is None or not args.model:
        raise ValueError('Checkpoint evaluation requires --prepared and --model')
    if not callable(getattr(ts, 'optimize_with_checkpoints', None)):
        raise RuntimeError('Rebuild TASO bindings: optimize_with_checkpoints is missing')
    budget, interval = cfg['budget'], cfg['checkpoint_interval']
    baseline = all_rules(budget)
    order = ['baseline', 'llm'] if args.trial % 2 else ['llm', 'baseline']
    controller = CheckpointStrategist(args.model, cfg,
        args.prepared / 'static_context.json',
        (args.prepared / 'model_source.py').read_text(), args.out)
    report.update(search_mode='checkpoint', checkpoint_interval=interval,
        max_model_calls=(budget - 1) // interval + 1,
        evaluation_order=order, warmup_searches_per_policy=1,
        notes=[
            'Two discarded all-rule warm-up searches, one per policy slot; no model calls during warm-up.',
            'Checkpoint policy decisions use actual timed-search feedback, never a replayed warm-up schedule.',
            'Queue, visited hashes, best graph and cost cache persist across in-memory callbacks.',
            'Queue exhaustion stops early; no graph replay, fallback policy or extra early API call.',
            'search_seconds excludes checkpoint callback time; optimization_wall_seconds includes callbacks.',
            'Inference latency and usage are summed over every reached checkpoint request.',
            'Feedback contains estimated costs and rewrite statistics, not GPU runtime measurements.',
            'One final run_time observation per variant; this does not check numerical equivalence.',
        ])
    try:
        started = time.perf_counter()
        for name in order:
            graph, _, _ = build()
            optimize(ts, graph, baseline, cfg['alpha'], args.out / f'{name}_warmup_trace.json', budget)
        report['warmup_search_seconds'] = time.perf_counter() - started
        variants, results = {}, {}
        report['variants'] = variants
        for name in order:
            graph, _, info = build()
            before = observe(graph)
            trace_path = args.out / f'{name}_trace.json'
            if name == 'baseline':
                results[name], variants[name] = optimize(ts, graph, baseline, cfg['alpha'], trace_path, budget)
                variants[name]['optimization_wall_seconds'] = variants[name]['search_seconds']
            else:
                started = time.perf_counter()
                results[name] = ts.optimize_with_checkpoints(graph, controller, budget=budget,
                    checkpoint_interval=interval, alpha=cfg['alpha'], trace_file=str(trace_path))
                elapsed = time.perf_counter() - started
                trace = json.loads(trace_path.read_text())
                callback_seconds = trace['checkpoint_callback_seconds']
                variants[name] = search_observation(results[name], trace_path, budget,
                                                     max(0, elapsed - callback_seconds))
                variants[name].update(optimization_wall_seconds=elapsed,
                                      checkpoint_callback_seconds=callback_seconds)
            variants[name]['before'] = before
            report['network_info'] = info
            report.update(model_calls=controller.model_calls, inference=controller.aggregate(),
                          checkpoint_decisions=controller.records)
            save(args.out / 'worker_result.json', report)
        for name in order:
            variants[name]['measured_ms'] = results[name].run_time()
            save(args.out / 'worker_result.json', report)
    finally:
        report.update(model_calls=controller.model_calls, inference=controller.aggregate(),
                      checkpoint_decisions=controller.records)

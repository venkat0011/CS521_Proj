"""Zero-API GPU timing diagnostic: baseline batch first, checkpoint batch second."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'benchmarks'))
sys.path.insert(0, str(HERE.parent / 'llm_context'))
from worker import all_rules, environment, optimize, save, search_observation
from networks import NETWORKS, build_network


def summarize(values):
    if len(values) < 4 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Timing requires at least four positive finite samples')
    median = statistics.median(values)
    half = len(values) // 2
    return dict(n=len(values), median_ms=median, min_ms=min(values), max_ms=max(values),
                range_pct=100 * (max(values) - min(values)) / median,
                cv_pct=100 * statistics.stdev(values) / statistics.mean(values),
                drift_pct=100 * (statistics.median(values[-half:]) /
                                statistics.median(values[:half]) - 1))


def source_fingerprint():
    root = HERE.parents[1]
    paths = [HERE / 'gpu_timing_smoke.py', HERE / 'run_gpu_smoke.py',
             HERE.parent / 'run_experiment.py', HERE.parent / 'benchmarks/networks.py',
             HERE.parent / 'llm_context/graph_context.py',
             HERE.parent / 'TASO/src/core/ops.cc',
             HERE.parent / 'TASO/src/cudnn/ops_cudnn.cu',
             HERE.parent / 'TASO/python/taso/_cython/core.pyx',
             HERE.parent / 'TASO/python/taso/_cython/CCore.pxd',
             HERE.parent / 'TASO/include/taso/ops.h',
             HERE.parent / 'TASO/graph_subst.pb']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def replay_policies(directory, budget, interval):
    from llm_strategy import validate_strategy
    from experiments.run_llm_strategy import ensure_context
    records = json.loads((directory / 'checkpoints.json').read_text())
    catalog = json.loads(ensure_context(HERE.parent / 'TASO').read_text())['catalog']
    policies = {}
    for record in records:
        if 'strategy' not in record:
            raise ValueError(f'Incomplete saved decision in {directory}')
        consumed = int(record['expansions'])
        chunk = min(interval, budget - consumed)
        if consumed < 0 or chunk <= 0 or consumed % interval:
            raise ValueError('Replay schedule does not match the requested budget/interval')
        if consumed in policies:
            raise ValueError('Duplicate replay boundary')
        policies[consumed] = validate_strategy(json.dumps(record['strategy']), catalog,
                                               chunk, 1, 1.05)
    if set(policies) != set(range(0, budget, interval)):
        raise ValueError('Replay needs every decision for the requested budget/interval')
    return policies


def measure(graph, args, out, report):
    if not callable(getattr(graph, 'benchmark_runtime', None)):
        raise RuntimeError('Rebuild TASO: benchmark_runtime binding is missing')
    rows, passes = [], []
    # Same live returned graph in both passes; remapping occurs only between passes.
    for mapping in range(2):
        telemetry_before = telemetry()
        started = time.perf_counter()
        flat = list(graph.benchmark_runtime(args.blocks))
        wall_seconds = time.perf_counter() - started
        if len(flat) != 2 * args.blocks:
            raise RuntimeError('Unexpected native diagnostic sample count')
        gpu, wall = flat[::2], flat[1::2]
        gs, ws = summarize(gpu), summarize(wall)
        rows.extend(dict(mapping=mapping + 1, block=i + 1, gpu_ms=g, host_wall_ms=w)
                    for i, (g, w) in enumerate(zip(gpu, wall)))
        flags = []
        if gs['range_pct'] > args.spread_threshold_pct:
            flags.append('wide_sample_range')
        if abs(gs['drift_pct']) > args.spread_threshold_pct:
            flags.append('within_mapping_drift')
        # CPU includes launch and synchronization overhead, so equality is not expected.
        if any(g > w * 1.05 for g, w in zip(gpu, wall)):
            flags.append('gpu_clock_exceeds_host_clock_by_over_5pct')
        passes.append(dict(mapping=mapping + 1, gpu=gs, host_wall=ws, flags=flags,
                           measurement_wall_seconds=wall_seconds,
                           telemetry_before=telemetry_before, telemetry_after=telemetry()))
        report['timing'] = passes
        save(out / 'worker_result.json', report)
        with (out / 'timing_samples.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['mapping', 'block', 'gpu_ms', 'host_wall_ms'])
            writer.writeheader(); writer.writerows(rows)
    report['remapping_median_change_pct'] = 100 * (
        passes[1]['gpu']['median_ms'] / passes[0]['gpu']['median_ms'] - 1)
    report['timing_concerns'] = any(p['flags'] for p in passes) or abs(
        report['remapping_median_change_pct']) > args.spread_threshold_pct


def telemetry():
    try:
        return subprocess.check_output(['nvidia-smi',
            '--query-gpu=timestamp,name,uuid,temperature.gpu,pstate,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader'], text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f'unavailable: {exc}'


def worker(args):
    out = args.out
    report = dict(status='started', variant=args.variant, attempt=args.attempt,
                  model_calls=0, budget=args.budget, checkpoint_interval=args.interval,
                  replay_trial=str(args.replay_trial) if args.replay_trial else None,
                  policy='saved_checkpoint_replay' if args.replay_trial else 'all_rules_control',
                  numerical_equivalence_checked=False, full_candidate_identity_recorded=False,
                  timing_protocol='2 mappings; fixed 0.125 inputs; 100 warm-up and 100 timed executions per block; same buffers within each mapping')
    save(out / 'worker_result.json', report)
    try:
        os.environ['TASO_HOME'] = str(args.repo)
        import taso as ts
        from graph_context import graph_context
        report['environment'] = environment(ts)
        graph, _, info = build_network(args.network, ts, args.repo, args.seed)
        if not callable(getattr(graph, 'benchmark_runtime', None)):
            raise RuntimeError('Rebuild TASO: benchmark_runtime binding is missing')
        report['network_info'] = info
        save(out / 'initial_graph_structure.json', graph_context(graph))
        # Match the historical discarded search, with no model calls.
        optimize(ts, graph, all_rules(args.budget), 1.05, out / 'warmup_trace.json', args.budget)
        graph, _, _ = build_network(args.network, ts, args.repo, args.seed)
        trace = out / 'search_trace.json'
        if args.variant == 'baseline':
            result, observed = optimize(ts, graph, all_rules(args.budget), 1.05, trace, args.budget)
        else:
            policies = replay_policies(args.replay_trial, args.budget, args.interval) if args.replay_trial else None
            decisions = []
            def select(current, progress):
                consumed = progress['expansions']
                policy = policies[consumed] if policies is not None else all_rules(
                    min(args.interval, args.budget - consumed))
                save(out / f'checkpoint_{consumed:04d}_structure.json', graph_context(current))
                decisions.append(dict(expansions=consumed, progress=progress, strategy=policy))
                save(out / 'checkpoints.json', decisions)
                return policy
            started = time.perf_counter()
            result = ts.optimize_with_checkpoints(graph, select, budget=args.budget,
                         checkpoint_interval=args.interval, alpha=1.05, trace_file=str(trace))
            observed = search_observation(result, trace, args.budget, time.perf_counter() - started)
        report['search'] = observed
        save(out / 'final_graph_structure.json', graph_context(result))
        # GUID-bearing getter output is a diagnostic snapshot, not canonical graph identity.
        report['graph_structure_sha256'] = hashlib.sha256(
            (out / 'final_graph_structure.json').read_bytes()).hexdigest()
        report['graph_structure_hash_scope'] = 'Exact file checksum only; not graph-equivalence evidence, no weight serialization'
        measure(result, args, out, report)
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    save(out / 'worker_result.json', report)
    return 0 if report['status'] == 'completed' else 1


def execute_batch(args):
    args.out.mkdir(parents=True, exist_ok=True)
    replay = [p.resolve() for p in args.replay_trial]
    replay_hashes = {str(p / 'checkpoints.json'): hashlib.sha256(
        (p / 'checkpoints.json').read_bytes()).hexdigest() for p in replay}
    config = dict(network=args.network, budget=args.budget, interval=args.interval,
                  seed=args.seed, blocks=args.blocks, required_successes=3, max_attempts=5,
                  order='complete baseline batch, then checkpoint batch',
                  replay_trials=[str(p) for p in replay], replay_sha256=replay_hashes,
                  spread_threshold_pct=args.spread_threshold_pct, source_sha256=source_fingerprint())
    config_path = args.out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Saved configuration or sources differ; use a new output folder')
    save(config_path, config)
    summary = dict(status='started', model_calls=0, configuration=config, variants={})
    for variant in ('baseline', 'checkpoint'):
        reports = []
        for attempt in range(1, 6):
            if sum(r['status'] == 'completed' for r in reports) >= 3:
                break
            out = args.out / variant / f'attempt_{attempt:02d}'
            path = out / 'worker_result.json'
            if out.exists():
                if path.exists():
                    report = json.loads(path.read_text())
                    if report['status'] == 'started':
                        report.update(status='failed', error='Interrupted attempt; retained without rerun')
                        save(path, report)
                else:
                    report = dict(status='failed', error='Interrupted before worker result; retained without rerun')
                    save(path, report)
            else:
                out.mkdir(parents=True)
                save(path, dict(status='started', variant=variant, attempt=attempt))
                command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                    '--variant', variant, '--attempt', str(attempt), '--out', str(out),
                    '--repo', str(args.repo), '--network', args.network, '--budget', str(args.budget),
                    '--interval', str(args.interval), '--seed', str(args.seed), '--blocks', str(args.blocks),
                    '--spread-threshold-pct', str(args.spread_threshold_pct)]
                if variant == 'checkpoint' and replay:
                    command += ['--replay-trial', str(replay[(attempt - 1) % len(replay)])]
                print(f'{variant}: attempt {attempt}/5; {sum(r["status"] == "completed" for r in reports)}/3 successful', flush=True)
                env = dict(os.environ)
                env.pop('OPENROUTER_API_KEY', None)
                with (out / 'worker.log').open('w') as log:
                    try:
                        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   timeout=args.worker_timeout)
                        report = json.loads(path.read_text())
                        if completed.returncode or report['status'] != 'completed':
                            report.update(status='failed', error=report.get('error', f'Worker exit {completed.returncode}'))
                    except subprocess.TimeoutExpired:
                        report = json.loads(path.read_text())
                        report.update(status='failed', error='Worker timeout')
                save(path, report)
            reports.append(report)
            print(f'{variant} attempt {attempt}: {report["status"]}; timing concerns={report.get("timing_concerns", "unavailable")}', flush=True)
            summary['variants'][variant] = dict(successful=sum(r['status'] == 'completed' for r in reports), attempts=reports)
            save(args.out / 'summary.json', summary)
        if sum(r['status'] == 'completed' for r in reports) < 3:
            summary.update(status='insufficient_successes', failed_variant=variant)
            save(args.out / 'summary.json', summary)
            return 1
    summary['status'] = 'completed'
    summary['interpretation'] = ('Execution success is not timing stability. Unstable finite timings count toward the three successes. '
        'The groups use separate processes/cost caches and are not paired counterfactual searches. '
        'Default checkpoint mode is an all-rules control, not Kimi. Replay executes saved policies, not new adaptive decisions. '
        'This diagnoses repeatability and two-clock agreement, not absolute accuracy, numerical equivalence or global optimality.')
    save(args.out / 'summary.json', summary)
    lines = ['GPU timing smoke completed: baseline first, checkpoint second; zero model calls.']
    for name, data in summary['variants'].items():
        lines.append(f'{name}: {data["successful"]} successes / {len(data["attempts"])} attempts')
        for i, report in enumerate(data['attempts'], 1):
            if report['status'] == 'completed':
                medians = [round(p['gpu']['median_ms'], 6) for p in report['timing']]
                ranges = [round(p['gpu']['range_pct'], 2) for p in report['timing']]
                lines.append(f'  attempt {i}: GPU medians {medians} ms; ranges {ranges}%; concerns={report["timing_concerns"]}')
    lines.append(summary['interpretation'])
    (args.out / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--network', choices=NETWORKS, default='squeezenet')
    p.add_argument('--budget', type=int, default=100)
    p.add_argument('--interval', type=int, default=25)
    p.add_argument('--seed', type=int, default=521)
    p.add_argument('--blocks', type=int, default=20)
    p.add_argument('--spread-threshold-pct', type=float, default=10)
    p.add_argument('--worker-timeout', type=int, default=1800)
    p.add_argument('--replay-trial', type=Path, action='append', default=[])
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--variant', choices=['baseline', 'checkpoint'])
    p.add_argument('--attempt', type=int, default=1)
    args = p.parse_args(argv)
    if args.budget < 1 or args.interval < 1 or not 4 <= args.blocks <= 200:
        p.error('Positive budget/interval and 4..200 timing blocks required')
    if not math.isfinite(args.spread_threshold_pct) or args.spread_threshold_pct <= 0 or args.worker_timeout < 1:
        p.error('Positive finite threshold and positive timeout required')
    args.repo, args.out = args.repo.resolve(), args.out.resolve()
    if args.worker:
        args.replay_trial = args.replay_trial[0] if args.replay_trial else None
        return worker(args)
    return execute_batch(args)


if __name__ == '__main__':
    raise SystemExit(main())

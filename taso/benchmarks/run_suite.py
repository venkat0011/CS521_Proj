#!/usr/bin/env python3
"""Five independent one-shot strategies per LLM/network, with paired TASO controls.

--dry-run prepares prompts and runs baseline preflights without API calls.
--plan-only inspects the configuration without TASO/GPU imports.
--batch-dir resumes a specific batch without replacing failed model calls.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'llm_context'))
from model_settings import resolve_settings
from networks import NETWORKS
from results import write_exports
from worker import save


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=HERE.parent / 'TASO')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--out', type=Path, default=HERE.parent / 'baseline/results')
    parser.add_argument('--batch-dir', type=Path, help='Stable results folder; resume the same batch here')
    parser.add_argument('--model', action='append', help='Repeat to compare multiple OpenRouter model IDs')
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--summarize', action='store_true', help='Rebuild tables/plots from saved trial files')
    parser.add_argument('--retry-evaluation', action='store_true',
                        help='Retry failed GPU evaluations using their existing strategies; no new model calls')
    return parser.parse_args(argv)


def validate_config(cfg):
    if cfg.get('schema_version') != 1:
        raise ValueError('Unsupported configuration schema')
    for key, choices in [('networks', NETWORKS), ('models', None)]:
        values = cfg[key]
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f'{key} must be a nonempty list of names')
        if len(set(values)) != len(values) or (choices and set(values) - set(choices)):
            raise ValueError(f'Duplicate or unknown {key}')
    for key in ('trials', 'budget', 'max_phases', 'max_tokens', 'worker_timeout_seconds'):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 2**31 - 1:
            raise ValueError(f'{key} must be a positive integer <= INT_MAX')
    if type(cfg['weight_seed']) is not int or not 0 <= cfg['weight_seed'] < 2**32:
        raise ValueError('weight_seed must fit NumPy seed range')
    for key, low, high in [('alpha', 1, float('inf')), ('temperature', 0, 2)]:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Invalid {key}')
    effort = cfg.get('reasoning_effort')
    if effort is not None and not isinstance(effort, str):
        raise ValueError('reasoning_effort must be a string or null')
    overrides = cfg.get('reasoning_effort_overrides', {})
    if not isinstance(overrides, dict) or any(
            not isinstance(model, str) or (value is not None and not isinstance(value, str))
            for model, value in overrides.items()):
        raise ValueError('reasoning_effort_overrides must map model IDs to strings or null')
    return cfg


def model_directory(network, model):
    return network + '__' + hashlib.sha256(model.encode()).hexdigest()[:16]


def source_hashes(repo):
    files = sorted((HERE).glob('*.py')) + [HERE / 'config.json', HERE.parent / 'baseline/run_llm_strategy.py']
    files += sorted((HERE.parent / 'llm_context').glob('*.py')) + [HERE.parent / 'llm_context/prompt_template.txt']
    hashes = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    for folder in ('examples', 'src/core', 'include/taso', 'python/taso'):
        for path in sorted((repo / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.pyx', '.pxd', '.cc', '.h', '.proto'):
                hashes['taso/TASO/' + str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes['taso/TASO/graph_subst.pb'] = hashlib.sha256((repo / 'graph_subst.pb').read_bytes()).hexdigest()
    return hashes


def run_worker(stage, network, directory, batch, cfg, repo, trial=1):
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(HERE / 'worker.py'), '--stage', stage,
               '--network', network, '--repo', str(repo), '--out', str(directory),
               '--config', str(batch / 'config.json'), '--trial', str(trial),
               '--prepared', str(batch / 'prepared' / network)]
    print(f'[{network}] {stage}; log: {directory / "worker.log"}', flush=True)
    code, error = None, None
    with (directory / 'worker.log').open('w') as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=cfg['worker_timeout_seconds']).returncode
        except subprocess.TimeoutExpired:
            error = f"Worker exceeded safety timeout of {cfg['worker_timeout_seconds']} seconds"
    path = directory / 'worker_result.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    if code != 0 or report.get('status') != 'completed':
        report.update(status='failed', error=error or report.get('error') or
                      f'GPU worker exited with code {code}; inspect worker.log')
    save(path, report)
    return report


def trial_reports(batch, cfg):
    reports = []
    for network in cfg['networks']:
        for model in cfg['models']:
            for trial in range(1, cfg['trials'] + 1):
                path = batch / 'trials' / model_directory(network, model) / f'trial_{trial:02}' / 'trial.json'
                reports.append(json.loads(path.read_text()) if path.exists() else {
                    'network': network, 'model': model, 'trial': trial, 'status': 'pending',
                    'budget': cfg['budget'], 'alpha': cfg['alpha'], 'model_calls': 0})
    return reports


def run_trial(network, model, trial, batch, cfg, repo, retry_evaluation=False):
    from llm_strategy import validate_strategy
    from openrouter_client import OpenRouterClient
    out = batch / 'trials' / model_directory(network, model) / f'trial_{trial:02}'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'trial.json'
    report = json.loads(path.read_text()) if path.exists() else {
        'network': network, 'model': model, 'trial': trial, 'budget': cfg['budget'],
        'alpha': cfg['alpha'], 'status': 'pending', 'model_calls': 0}
    status = report['status']
    if status == 'failed_preflight':
        # Preparation has now passed and no model call was previously attempted.
        status = report['status'] = 'pending'
    if status == 'completed' or (status.startswith('failed') and not
                               (retry_evaluation and status == 'failed_evaluation')):
        return report
    prepared = batch / 'prepared' / network
    catalog = json.loads((prepared / 'catalog.json').read_text())
    generation_path = out / 'generation.json'
    client = None
    try:
        if (out / 'strategy.json').exists():
            validate_strategy((out / 'strategy.json').read_text(), catalog,
                              cfg['budget'], cfg['max_phases'], cfg['alpha'])
            if (out / 'inference_metadata.json').exists():
                report['inference'] = json.loads((out / 'inference_metadata.json').read_text())
            report['model_calls'] = 1
        else:
            if generation_path.exists():
                raise RuntimeError('A previous generation attempt has no saved valid strategy. '
                                   'It is retained as a failed trial; no replacement API call is made.')
            if not os.environ.get('OPENROUTER_API_KEY', '').strip():
                raise ValueError('Set OPENROUTER_API_KEY before running the suite')
            messages = json.loads((prepared / 'request.json').read_text())
            report['prompt_sha256'] = json.loads((prepared / 'request_metadata.json').read_text())['request_sha256']
            report.update(status='generating', model_calls=1)
            save(path, report)
            # Persist intent BEFORE contacting the API to prevent duplicate charges after interruption.
            save(generation_path, {'status': 'started', 'model': model})
            client = OpenRouterClient(model, save_to=out, max_tokens=cfg['max_tokens'],
                                      temperature=cfg['temperature'],
                                      reasoning_effort=cfg.get('reasoning_effort'),
                                      reasoning_effort_overrides=cfg.get('reasoning_effort_overrides'))
            raw = client(messages)
            save(out / 'response.json', {'raw_response': raw})
            strategy = validate_strategy(raw, catalog, cfg['budget'], cfg['max_phases'], cfg['alpha'])
            save(out / 'strategy.json', strategy)
            save(generation_path, {'status': 'completed', 'model_calls': client.request_attempts})
            report['inference'] = client.metadata
        report['status'] = 'evaluating'
        report.pop('error', None)
        save(path, report)
        worker = run_worker('evaluate', network, out, batch, cfg, repo, trial)
        report.update(evaluation=worker, status='completed' if worker['status'] == 'completed' else 'failed_evaluation')
        if worker['status'] != 'completed':
            report['error'] = worker['error']
    except Exception as exc:
        evaluating = report.get('status') == 'evaluating'
        report.update(status='failed_evaluation' if evaluating else 'failed_generation',
                      error=f'{type(exc).__name__}: {exc}')
        if client is not None:
            report.update(model_calls=client.request_attempts, inference=client.metadata)
            save(generation_path, {'status': 'failed', 'error': report['error'], 'model_calls': client.request_attempts})
        (out / 'error.log').write_text(traceback.format_exc())
    if report.get('inference') is None and (out / 'inference_metadata.json').exists():
        report['inference'] = json.loads((out / 'inference_metadata.json').read_text())
    save(path, report)
    print(f"[{network}] {model} trial {trial}/{cfg['trials']}: {report['status']}", flush=True)
    return report


def main(argv=None):
    args = parse_args(argv)
    if args.summarize:
        if args.batch_dir is None:
            raise ValueError('--summarize requires --batch-dir')
        cfg = validate_config(json.loads((args.batch_dir / 'config.json').read_text()))
        write_exports(args.batch_dir, trial_reports(args.batch_dir, cfg))
        print(f'Updated distributions: {args.batch_dir}')
        return 0
    cfg = json.loads(args.config.read_text())
    if args.model:
        cfg['models'] = args.model
    if args.max_tokens is not None:
        cfg['max_tokens'] = args.max_tokens
    validate_config(cfg)
    plan = {'networks': cfg['networks'], 'models': cfg['models'], 'trials_per_pair': cfg['trials'],
            'max_model_calls': len(cfg['networks']) * len(cfg['models']) * cfg['trials'],
            'budget': cfg['budget'], 'alpha': cfg['alpha'],
            'model_settings': {model: resolve_settings(
                model, cfg['temperature'], cfg.get('reasoning_effort'),
                cfg.get('reasoning_effort_overrides')) for model in cfg['models']}}
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return 0
    repo = args.repo.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    batch = (args.batch_dir or args.out / 'suite' / stamp).resolve()
    batch.mkdir(parents=True, exist_ok=True)
    contract = {'config': cfg, 'source_sha256': source_hashes(repo)}
    manifest_path = batch / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['contract'] != contract:
            raise ValueError('Batch configuration or sources changed. Use a new --batch-dir.')
    else:
        save(manifest_path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'contract': contract})
        save(batch / 'config.json', cfg)
    print(f'Batch: {batch}', flush=True)
    # Check credentials once, before recording individual attempts as failures.
    if not args.dry_run and not os.environ.get('OPENROUTER_API_KEY', '').strip():
        raise ValueError('Set OPENROUTER_API_KEY, or use --dry-run for baseline preflights only')
    write_exports(batch, trial_reports(batch, cfg), plots=False)
    for network in cfg['networks']:
        prepared = batch / 'prepared' / network
        path = prepared / 'worker_result.json'
        preflight = json.loads(path.read_text()) if path.exists() else None
        if not preflight or preflight.get('status') != 'completed':
            preflight = run_worker('prepare', network, prepared, batch, cfg, repo)
        if preflight['status'] != 'completed':
            for model in cfg['models']:
                for trial in range(1, cfg['trials'] + 1):
                    out = batch / 'trials' / model_directory(network, model) / f'trial_{trial:02}'
                    out.mkdir(parents=True, exist_ok=True)
                    if not (out / 'trial.json').exists():
                        save(out / 'trial.json', {'network': network, 'model': model, 'trial': trial,
                             'budget': cfg['budget'], 'alpha': cfg['alpha'], 'status': 'failed_preflight',
                             'model_calls': 0, 'error': preflight['error']})
            continue
        if args.dry_run:
            continue
        for model in cfg['models']:
            for trial in range(1, cfg['trials'] + 1):
                run_trial(network, model, trial, batch, cfg, repo, args.retry_evaluation)
                write_exports(batch, trial_reports(batch, cfg), plots=False)
    reports = trial_reports(batch, cfg)
    write_exports(batch, reports)
    print(f'Raw trials: {batch / "trials.csv"}\nSummary: {batch / "summary.md"}', flush=True)
    preflights_ok = all((batch / 'prepared' / n / 'worker_result.json').exists() and
                        json.loads((batch / 'prepared' / n / 'worker_result.json').read_text()).get('status') == 'completed'
                        for n in cfg['networks'])
    return 0 if (preflights_ok and (args.dry_run or all(r['status'] == 'completed' for r in reports))) else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Recover saved numeric-index strategies without another LLM request.

Run only after the suite has stopped. Uses each network's frozen catalog,
keeps the raw API response, and records the original validation failure.
This script imports the pure Python strategy validator, never the GPU runtime.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


def unique_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def convert(raw, catalog, cfg, compile_strategy):
    strategy = json.loads(raw, object_pairs_hook=unique_keys)
    converted = copy.deepcopy(strategy)
    index_to_id = {}
    for rule in catalog['rules']:
        index = rule['taso_xfer_index']
        if type(index) is not int or index < 0 or index in index_to_id:
            raise ValueError('Catalog has invalid or ambiguous native indices')
        index_to_id[index] = rule['rule_id']
    changes = []
    for phase_index, phase in enumerate(converted['phases']):
        ids = phase.get('rule_ids')
        if isinstance(ids, list):
            for position, value in enumerate(ids):
                if type(value) is int:
                    if value not in index_to_id:
                        raise ValueError(f'Unknown native rule index: {value}')
                    ids[position] = index_to_id[value]
                    changes.append({'phase': phase_index, 'position': position,
                                    'native_index': value, 'rule_id': ids[position]})
    if not changes:
        raise ValueError(
            'No numeric native indices found; no conversion performed')
    budgets, compiled_ids = compile_strategy(converted, cfg['alpha'])
    if sum(budgets) != cfg['budget'] or len(budgets) > cfg['max_phases']:
        raise ValueError('Strategy violates the frozen budget or phase limit')
    known = set(index_to_id.values())
    for phase in converted['phases']:
        if phase['rule_ids'] != 'all' and set(phase['rule_ids']) - known:
            raise ValueError('Strategy contains unknown catalog IDs')
    # Prove that conversion selects exactly the original native indices.
    for before, after, compiled in zip(strategy['phases'], converted['phases'], compiled_ids):
        if isinstance(before['rule_ids'], list):
            for value, index in zip(before['rule_ids'], compiled):
                if type(value) is int and value != index:
                    raise ValueError(
                        'Catalog and executor index mappings disagree')
    return converted, changes


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def recover_batch(batch, project):
    batch, project = Path(batch).resolve(), Path(project).resolve()
    cfg = json.loads((batch / 'config.json').read_text())
    spec = importlib.util.spec_from_file_location('saved_strategy_validator',
                                                  project / 'taso/TASO/python/taso/strategy.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    recovered = []
    for trial_path in sorted((batch / 'trials').glob('*/trial_*/trial.json')):
        report = json.loads(trial_path.read_text())
        if report.get('status') != 'failed_generation' or 'rule_ids' not in report.get('error', ''):
            continue
        directory = trial_path.parent
        strategy_path = directory / 'strategy.json'
        if strategy_path.exists():
            print(f'SKIP {directory}: existing strategy preserved')
            continue
        try:
            api = json.loads((directory / 'api_response.json').read_text())
            choice = api['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Response did not finish normally')
            raw = choice['message']['content']
            response_path = directory / 'response.json'
            if response_path.exists() and json.loads(response_path.read_text())['raw_response'] != raw:
                raise ValueError('Saved response files disagree')
            catalog = json.loads(
                (batch / 'prepared' / report['network'] / 'catalog.json').read_text())
            strategy, changes = convert(
                raw, catalog, cfg, validator.compile_strategy)
            generation = directory / 'generation.json'
            original_generation = json.loads(
                generation.read_text()) if generation.exists() else None
            audit = {'conversion': 'catalog_native_index_to_rule_id',
                     'original_format_valid': False, 'api_response_id': api.get('id'),
                     'original_trial_record': report,
                     'original_generation_record': original_generation,
                     'changes': changes,
                     'strategy_sha256': hashlib.sha256(json.dumps(strategy, sort_keys=True).encode()).hexdigest(),
                     'notes': ['No additional LLM request; rule selections and phase budgets are preserved.',
                               'Original API/response/error files are retained.',
                               'Report strict format failures separately from recovered strategy performance.']}
            save(directory / 'rule_id_recovery.json', audit)
            save(strategy_path, strategy)
            save(generation, {'status': 'completed', 'model_calls': report.get('model_calls', 1),
                              'recovered_format': True, 'audit_file': 'rule_id_recovery.json'})
            updated = dict(report)
            updated.update(status='strategy_ready', format_recovery={
                'original_format_valid': False, 'converted_ids': len(changes),
                'audit_file': 'rule_id_recovery.json'})
            updated.pop('error', None)
            save(trial_path, updated)
            recovered.append(str(directory))
            print(
                f'RECOVERED {directory}: mapped {len(changes)} IDs; ready for GPU evaluation')
        except (ValueError, KeyError, TypeError, OSError) as error:
            print(f'SKIP {directory}: {type(error).__name__}: {error}')
    print(
        f'Recovered {len(recovered)} trial(s). No API calls or GPU evaluations were made.')
    return recovered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path,
                        help='Default: detect the existing four_networks_100 batch')
    parser.add_argument('--project-root', type=Path,
                        default=Path('/content/CS521_Proj'))
    parser.add_argument('--resume', action='store_true',
                        help='Authenticate and resume evaluation with the saved strategies')
    args = parser.parse_args()
    project = args.project_root.resolve()
    batch = args.batch_dir
    if batch is None:
        candidates = [project / 'taso/baseline/results/four_networks_100',
                      project / 'taso/baseline/taso/baseline/results/four_networks_100']
        candidates = [p for p in candidates if (p / 'manifest.json').exists()]
        if len(candidates) != 1:
            raise SystemExit(
                'Could not identify exactly one existing batch; pass --batch-dir explicitly.')
        batch = candidates[0]
    batch = batch.resolve()
    if args.resume:
        key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        if not key:
            raise SystemExit(
                'Set OPENROUTER_API_KEY in this notebook/terminal before using --resume.')
        request = urllib.request.Request('https://openrouter.ai/api/v1/key',
                                         headers={'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f'Authentication: {response.status}', flush=True)
    recover_batch(batch, project)
    if args.resume:
        command = [sys.executable, str(project / 'taso/baseline/setup_and_run.py'),
                   '--experiment', 'suite', '--batch-dir', str(batch)]
        with subprocess.Popen(command, cwd=project, env=os.environ.copy(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
            code = process.wait()
        if code:
            for path in sorted((batch / 'trials').glob('*/trial_*/trial.json')):
                trial = json.loads(path.read_text())
                if trial.get('error'):
                    print(f'{path}: {trial["error"]}', flush=True)
        raise SystemExit(code)

"""Import historical checkpoint reports read-only, with protocol/source provenance."""
import copy
import hashlib
import json
import math
from pathlib import Path

# Filled with the reviewed pre-GPU source hashes when this update is packaged.
COMPATIBLE_PLAIN_SOURCE_HASHES = {'taso/TASO/include/taso/ops.h': '72ffea67e32c9e0b8f0f8bf16033ac0802da25bed0d36d73ecd44c8e14e359da', 'taso/TASO/src/core/ops.cc': 'ae04e2b0f515635e2fc0c1a7b40dc98316ef18f65f424346e99cb5c4b250de49', 'taso/TASO/python/taso/_cython/CCore.pxd': '0e1fab84dc6b5dced60539a9afaa1d8ac65ef5ab0ae9a31dd6388e409be67c10', 'taso/TASO/python/taso/_cython/core.pyx': '4da1d1f6cfb6b4c056d0e98d325e3408df28f770ff4f002111e03ed79cab4251', 'taso/llm_context/checkpoint_strategy.py': '7ee471c85575f519958214ffb9abc836c710643c0b400dcf6950976ea0fe9301', 'taso/llm_context/compact_checkpoint_prompt.py': 'b5e11cd1f75b8bd0d8e8adea4c0783a92c9ec9b3b9b1edb8111af470011b1792', 'taso/benchmarks/checkpoint_worker.py': 'bb515472d3909b0f85af7ccf16bd630daf5e55cd16c57189d3848730b2c95909', 'taso/benchmarks/run_suite.py': '7e6d4476c5bcb2e8c750c6275eb7fe1b778ad231dfb6e5d3860660babd7af81b', 'taso/benchmarks/results.py': 'fad44b2fee7e916f48aa0ec9da8b6efa08d620b4b81835355e39bb694a615241', 'taso/run_experiment.py': 'b6be3c4d29d5f0efb7f17f8613039c825c02d09f48d915995f1c41a851ccaf29', 'taso/verify_bindings.py': '13cae4a6c58bc9859b5274861a23a23c2061f3dcc4785fde5e781cf0589aaf90'}
PROTOCOL_KEYS = ('schema_version', 'budget', 'alpha', 'weight_seed', 'trials', 'max_phases',
    'max_tokens', 'temperature', 'reasoning_effort', 'reasoning_effort_overrides',
    'search_mode', 'checkpoint_interval', 'completion_policy', 'max_retries',
    'request_timeout_seconds', 'worker_timeout_seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_checkpoint_reports(source, cfg, current_hashes):
    from run_suite import model_directory
    source = Path(source).resolve()
    manifest_path, config_path = source/'manifest.json', source/'config.json'
    manifest, old = json.loads(manifest_path.read_text()), json.loads(config_path.read_text())
    if old.get('search_mode') != 'checkpoint' or old.get('gpu_feedback', False):
        raise ValueError('Reuse source must be a plain-checkpoint batch without GPU feedback')
    if manifest.get('contract', {}).get('config') != old:
        raise ValueError('Reuse source config does not match its frozen manifest')
    differing = [key for key in PROTOCOL_KEYS if old.get(key) != cfg.get(key)]
    if differing:
        raise ValueError('Reuse protocol differs: '+', '.join(differing)+
                         '. Use the saved source config and the same checkpoint interval.')
    if set(cfg['models']) - set(old['models']) or set(cfg['networks']) - set(old['networks']):
        raise ValueError('Reuse source does not cover the requested models/networks')
    old_hashes = manifest['contract']['source_sha256']
    required = ('taso/TASO/src/core/ops.cc','taso/TASO/src/core/substitution.cc',
        'taso/TASO/include/taso/ops.h','taso/TASO/graph_subst.pb','taso/benchmarks/networks.py',
        'taso/llm_context/model_settings.py','taso/llm_context/checkpoint_strategy.py',
        'taso/llm_context/compact_checkpoint_prompt.py','taso/llm_context/checkpoint_compact_prompt_template.txt')
    if any(name not in old_hashes for name in required):
        raise ValueError('Reuse manifest lacks required search/rule/model/prompt source hashes')
    differences = [name for name, value in old_hashes.items() if current_hashes.get(name) != value]
    unknown = [name for name in differences if COMPATIBLE_PLAIN_SOURCE_HASHES.get(name) != old_hashes[name]]
    if unknown:
        raise ValueError('Unreviewed source changes prevent result reuse: '+', '.join(unknown))
    reports, provenance = [], []
    for network in cfg['networks']:
        for model in cfg['models']:
            for trial in range(1,cfg['trials']+1):
                path = source/'trials'/model_directory(network,model)/f'trial_{trial:02}'/'trial.json'
                if path.exists():
                    report = json.loads(path.read_text())
                    if (report.get('network'),report.get('model'),report.get('trial')) != (network,model,trial):
                        raise ValueError('Reuse trial identity does not match its path: '+str(path))
                    provenance.append({'trial':str(path),'sha256':sha(path)})
                else:
                    report = {'network':network,'model':model,'trial':trial,'status':'not_available',
                              'model_calls':0,'error':'Historical checkpoint trial was not present; no replacement calls made'}
                if report.get('status') == 'completed':
                    if report.get('budget') != cfg['budget'] or report.get('alpha') != cfg['alpha']:
                        raise ValueError('Reuse trial budget/alpha differs from its frozen config: '+str(path))
                    variants = (report.get('evaluation') or {}).get('variants',{})
                    for name in ('baseline','llm'):
                        value = variants.get(name,{}).get('measured_ms')
                        if not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
                            raise ValueError('Completed reuse trial has no valid final GPU measurement: '+str(path))
                elif report.get('status') in ('pending','started','evaluating','generating'):
                    report = copy.deepcopy(report)
                    report['status'] = 'not_available'
                    report['error'] = 'Historical trial unfinished; retained without replacement model calls'
                report = copy.deepcopy(report)
                report.update(search_mode='checkpoint',checkpoint_interval=cfg['checkpoint_interval'],
                    feedback_mode='checkpoint',gpu_feedback=False,reused_result=True,
                    reuse_source_trial=str(path),reuse_source_batch=str(source),
                    reuse_source_manifest_sha256=sha(manifest_path))
                reports.append(report)
    return {'schema_version':1,'source_batch':str(source),'source_manifest_sha256':sha(manifest_path),
            'reviewed_source_differences':differences,'trial_provenance':provenance,'reports':reports,
            'note':'Historical observations; no source files, reports, GPU timings or model calls are rerun or overwritten.'}

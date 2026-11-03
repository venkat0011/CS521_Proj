"""Checkpoint/GPU-feedback conditions, interleaved trials and read-only historical reuse."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from results import write_exports
from worker import save
from reuse_results import import_checkpoint_reports


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def combined_reports(batch, stored):
    from run_suite import trial_reports
    reports=[]
    if stored.get('reuse_checkpoint_batch'):
        snapshot=json.loads((batch/'reused_checkpoint_reports.json').read_text())
        manifest=json.loads((batch/'manifest.json').read_text())
        if digest(snapshot) != manifest['reuse_snapshot_sha256']:
            raise ValueError('Frozen reuse snapshot changed')
        reports.extend(snapshot['reports'])
    for mode,cfg in stored['arms'].items():
        for report in trial_reports(batch/mode,cfg):
            report=copy.deepcopy(report)
            report.update(feedback_mode=mode,gpu_feedback=cfg['gpu_feedback'],
                          checkpoint_interval=cfg['checkpoint_interval'],reused_result=False)
            reports.append(report)
    return reports


def summarize_comparison(batch, plots=True):
    batch=Path(batch)
    stored=json.loads((batch/'config.json').read_text())
    return write_exports(batch,combined_reports(batch,stored),plots=plots,
                         min_successful_trials=stored['minimum_successful_trials'])


def run_comparison(args, base):
    from run_suite import validate_config,source_hashes,run_worker,run_trial,model_directory
    if args.summarize:
        if args.batch_dir is None: raise ValueError('--summarize requires --batch-dir')
        summarize_comparison(args.batch_dir)
        return 0
    if base.get('search_mode') != 'checkpoint':
        raise ValueError('Use checkpoint mode/--checkpoint-interval for GPU feedback comparison')
    if args.retry_evaluation:
        raise ValueError('In-memory checkpoint trials cannot use --retry-evaluation')
    if args.reuse_checkpoint_batch is not None and args.gpu_feedback == 'off':
        raise ValueError('Use --gpu-feedback on or both with historical checkpoint reuse')
    mode=args.gpu_feedback or 'on'
    threshold=args.min_successful_trials if args.min_successful_trials is not None else min(3,base['trials'])
    if not 1 <= threshold <= base['trials']: raise ValueError('--min-successful-trials must be between 1 and trials')
    modes=['checkpoint','checkpoint_gpu'] if mode=='both' else ['checkpoint_gpu']
    if args.reuse_checkpoint_batch is not None and 'checkpoint' in modes:
        modes.remove('checkpoint')  # all historical slots, including failures, are reused without calls
    arms={name:validate_config(dict(base,gpu_feedback=name=='checkpoint_gpu',
                                  gpu_feedback_repeats=base.get('gpu_feedback_repeats',3))) for name in modes}
    per_decision=base.get('max_retries',3)+1 if base.get('completion_policy') else 1
    decisions=(base['budget']-1)//base['checkpoint_interval']+1
    slots=len(base['networks'])*len(base['models'])*base['trials']
    plan={'feedback_modes':(['checkpoint'] if args.reuse_checkpoint_batch else [])+modes,
          'gpu_feedback_target':'current estimated-best graph','checkpoint_interval':base['checkpoint_interval'],
          'budget':base['budget'],'trials_per_group':base['trials'],'minimum_successful_trials':threshold,
          'new_trial_slots':slots*len(arms),'max_new_model_calls':slots*len(arms)*decisions*per_decision,
          'reuse_checkpoint_batch':str(args.reuse_checkpoint_batch) if args.reuse_checkpoint_batch else None,
          'reuse_model_calls':0,'provider_routing':'openrouter_auto'}
    print(json.dumps(plan,indent=2),flush=True)
    if args.plan_only:return 0
    repo=args.repo.resolve()
    batch=(args.batch_dir or args.out/'gpu_comparison'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    reuse=args.reuse_checkpoint_batch.resolve() if args.reuse_checkpoint_batch else None
    if reuse==batch:raise ValueError('Comparison output must differ from the historical batch')
    hashes=source_hashes(repo)
    stored={'comparison_schema_version':1,'base_config':base,'arms':arms,
            'reuse_checkpoint_batch':str(reuse) if reuse else None,'minimum_successful_trials':threshold}
    contract={'config':stored,'source_sha256':hashes}
    manifest_path=batch/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['contract']!=contract:raise ValueError('Comparison config/sources changed; use a new batch directory')
    else:
        # Validate the historical protocol before creating output or contacting a model.
        snapshot=import_checkpoint_reports(reuse,base,hashes) if reuse else None
        batch.mkdir(parents=True,exist_ok=True)
        manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'contract':contract}
        if snapshot is not None:
            manifest['reuse_snapshot_sha256']=digest(snapshot)
            save(batch/'reused_checkpoint_reports.json',snapshot)
        save(batch/'config.json',stored)
        save(manifest_path,manifest)
    for name,cfg in arms.items():
        arm=batch/name
        arm.mkdir(parents=True,exist_ok=True)
        expected={'contract':{'config':cfg,'source_sha256':hashes},'comparison_parent':str(batch)}
        if (arm/'manifest.json').exists():
            if json.loads((arm/'manifest.json').read_text())!=expected or json.loads((arm/'config.json').read_text())!=cfg:
                raise ValueError('Comparison condition manifest/config changed: '+name)
        else:
            save(arm/'config.json',cfg)
            save(arm/'manifest.json',expected)
    summarize_comparison(batch,plots=False)
    if not args.dry_run and not os.environ.get('OPENROUTER_API_KEY','').strip():
        raise ValueError('Set OPENROUTER_API_KEY in this terminal')
    all_ready=True
    for network in base['networks']:
        ready={}
        for name,cfg in arms.items():
            arm=batch/name
            prepared=arm/'prepared'/network
            path=prepared/'worker_result.json'
            preflight=json.loads(path.read_text()) if path.exists() else None
            if not preflight or preflight.get('status')!='completed':
                preflight=run_worker('prepare',network,prepared,arm,cfg,repo)
            ready[name]=preflight.get('status')=='completed'
            all_ready=all_ready and ready[name]
            if not ready[name]:
                for model in cfg['models']:
                    for trial in range(1,cfg['trials']+1):
                        out=arm/'trials'/model_directory(network,model)/f'trial_{trial:02}'
                        out.mkdir(parents=True,exist_ok=True)
                        if not (out/'trial.json').exists():
                            save(out/'trial.json',{'network':network,'model':model,'trial':trial,
                                'status':'failed_preflight','model_calls':0,'budget':cfg['budget'],
                                'alpha':cfg['alpha'],'error':preflight.get('error')})
        if args.dry_run:continue
        for model in base['models']:
            for trial in range(1,base['trials']+1):
                order=modes if trial%2 else list(reversed(modes))
                for name in order:
                    if ready[name]:run_trial(network,model,trial,batch/name,arms[name],repo)
                    summarize_comparison(batch,plots=False)
    stats=summarize_comparison(batch)
    print(f'Combined trials: {batch / "trials.csv"}\nAlias: {batch / "trial.csv"}\nSummary: {batch / "summary.md"}',flush=True)
    if args.dry_run:return 0 if all_ready else 1
    return 0 if stats['groups'] and all(g['meets_minimum'] for g in stats['groups']) else 1

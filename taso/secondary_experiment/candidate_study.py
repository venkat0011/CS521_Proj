"""Run reference TASO batch first, then one live Kimi CP25 batch; three successes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'benchmarks'))
from worker import save


def configuration(args):
    cfg=json.loads((HERE.parent/'benchmarks/checkpoint_continuation.json').read_text())
    cfg.update(model='moonshotai/kimi-k3.1-pro',models=['moonshotai/kimi-k3.1-pro'],networks=list(getattr(args,'networks',['squeezenet','resnext50','inceptionv3'])),
               budget=100,checkpoint_interval=25,gpu_feedback=False,reference_budget=args.reference_budget,
               reference_points=sorted({100,250,args.reference_budget}),blocks=args.blocks,
               weight_initialization=getattr(args,'weight_initialization','original'),input_seeds=[1,2,3],rtol=1e-3,atol=1e-6,required_successes=3,max_attempts=5)
    cfg['source_sha256']={str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [HERE/'candidate_study.py',HERE/'candidate_worker.py',HERE/'candidate_identity.py',
                  HERE/'restore_graph.py',HERE/'study_networks.py',HERE/'graph_archive.py',HERE/'analyze_candidates.py',HERE/'gpu_timing_smoke.py',
                  HERE.parent/'TASO/include/taso/ops.h',HERE.parent/'TASO/include/taso/candidate_audit.h',
                  HERE.parent/'TASO/include/taso/candidate_snapshot.h',HERE.parent/'TASO/src/core/ops.cc',
                  HERE.parent/'TASO/src/core/substitution.cc',HERE.parent/'TASO/src/cudnn/ops_cudnn.cu',
                  HERE.parent/'TASO/python/taso/_cython/core.pyx',HERE.parent/'TASO/python/taso/_cython/CCore.pxd',
                  HERE.parent/'benchmarks/networks.py',HERE.parent/'llm_context/checkpoint_strategy.py',
                  HERE.parent/'llm_context/checkpoint_compact_prompt_template.txt',HERE.parent/'TASO/graph_subst.pb']}
    extra=[HERE/'run_candidate_study.py',HERE.parent/'run_experiment.py',
           HERE.parent/'TASO/examples/resnext50.py',HERE.parent/'TASO/examples/inceptionv3.py']
    extra.extend((HERE.parent/'llm_context').glob('*.py'))
    extra.extend(p for p in (HERE.parent/'TASO').rglob('*') if p.is_file() and p.suffix in ('.h','.cc','.cu','.pxd','.pyx'))
    for p in extra:cfg['source_sha256'][str(p.relative_to(HERE.parents[1]))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return cfg


def run_group(root,mode,network,cfg,path,reference_dirs):
    reports=[]
    for attempt in range(1,6):
        if sum(r['status']=='completed' for r in reports)>=3:break
        directory=root/mode/network/f'attempt_{attempt:02d}';report_path=directory/'worker_result.json'
        if directory.exists():
            record=json.loads(report_path.read_text()) if report_path.exists() else dict(status='failed',error='Interrupted before worker initialization')
            if record['status']=='started':record.update(status='failed',error='Interrupted worker; preserved without repeating calls')
        else:
            directory.mkdir(parents=True);record=dict(status='started');save(report_path,record)
            command=[sys.executable,str(HERE/'candidate_worker.py'),'--config',str(path),'--out',str(directory),
                     '--mode',mode,'--network',network,'--attempt',str(attempt)]
            if mode=='kimi':command+=['--baseline',str(reference_dirs[(attempt-1)%len(reference_dirs)])]
            env=dict(os.environ)
            if mode=='baseline':env.pop('OPENROUTER_API_KEY',None)
            print(f'{mode}/{network}: attempt {attempt}/5; {sum(r["status"]=="completed" for r in reports)}/3 completed',flush=True)
            with (directory/'worker.log').open('w') as log:
                try:
                    process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cfg['worker_timeout_seconds'])
                    record=json.loads(report_path.read_text())
                    if process.returncode or record['status']!='completed':record.update(status='failed',error=record.get('error',f'Worker exit {process.returncode}'))
                except subprocess.TimeoutExpired:
                    record=json.loads(report_path.read_text());record.update(status='failed',error='Worker timeout')
        record.update(directory=str(directory),network=network);save(report_path,record);reports.append(record)
        print(f'{mode}/{network} attempt {attempt}: {record["status"]}; numerically_valid={record.get("numerically_valid","unknown")}',flush=True)
        if record['status']=='failed':
            print('  '+record.get('error','Worker failed without a Python error; inspect worker.log'),flush=True)
            if record.get('fatal_preflight_error'):
                print('  Stopping this DNN group: repeating this preflight error is not useful.',flush=True)
                break
    return dict(successful=sum(r['status']=='completed' for r in reports),attempts=reports)


def run(args):
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True);cfg=configuration(args)
    path=root/'config.json'
    if path.exists() and json.loads(path.read_text())!=cfg:
        raise ValueError('Configuration/source changed; use a new study directory')
    save(path,cfg)
    summary=dict(status='started',variants={},networks={n:dict(variants={}) for n in cfg['networks']},
                 mode='live Kimi CP25 without GPU feedback',
                 success_definition='Execution completed; noisy timings and numerical disagreements are retained, not replaced')
    references={n:[] for n in cfg['networks']}
    for mode in ('baseline','kimi'):
        if mode=='kimi' and args.preflight_only:
            summary['status']='baseline_preflight_completed';save(root/'summary.json',summary);return 0
        if mode=='kimi' and not os.environ.get('OPENROUTER_API_KEY'):
            summary['status']='waiting_for_api_key';save(root/'summary.json',summary)
            print('All baseline batches complete. Set OPENROUTER_API_KEY and rerun unchanged to start Kimi.',flush=True);return 1
        incomplete=[];flat=[]
        for network in cfg['networks']:
            group=run_group(root,mode,network,cfg,path,references[network])
            summary['networks'][network]['variants'][mode]=group;flat.extend(group['attempts'])
            summary['variants'][mode]=dict(successful=sum(r['status']=='completed' for r in flat),attempts=flat)
            if mode=='baseline':references[network]=[Path(r['directory']) for r in group['attempts'] if r['status']=='completed']
            if group['successful']<3:incomplete.append(network)
            save(root/'summary.json',summary)
        if incomplete:
            summary.update(status='insufficient_successes',failed_variant=mode,failed_networks=incomplete)
            save(root/'summary.json',summary);return 1
    summary['status']='completed';save(root/'summary.json',summary)
    from analyze_candidates import analyze
    analyze(root)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-budget',type=int,default=500)
    p.add_argument('--weight-initialization',choices=['original','fan_in'],default='original')
    p.add_argument('--networks',nargs='+',choices=['squeezenet','resnext50','inceptionv3'],default=['squeezenet','resnext50','inceptionv3']);
    p.add_argument('--blocks',type=int,default=20);p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(argv)
    if len(set(args.networks))!=len(args.networks):p.error('Duplicate DNN names are not allowed')
    if args.reference_budget<250 or args.reference_budget%25 or not 4<=args.blocks<=200:
        p.error('Reference budget must be a multiple of 25 and at least 250; blocks must be 4..200')
    return run(args)

if __name__=='__main__':raise SystemExit(main())

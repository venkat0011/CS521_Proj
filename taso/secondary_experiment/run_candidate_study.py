#!/usr/bin/env python3
"""Build and run the secondary candidate identity / checkpoint / budget study."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from run_experiment import Runner


class CandidateRunner(Runner):
    def experiment_command(self,root,args,python,results):
        self.stage='secondary_candidate_study'
        command=[str(python),str(HERE/'candidate_study.py'),'--out',str(results),
                 '--reference-budget',str(args.reference_budget),'--blocks',str(args.blocks)]
        command.extend(['--networks']+args.networks)
        if args.preflight_only:command.append('--preflight-only')
        return command


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path)
    p.add_argument('--networks',nargs='+',choices=['squeezenet','resnext50','inceptionv3'],default=['squeezenet','resnext50','inceptionv3'])
    p.add_argument('--reference-budget',type=int,default=500)
    p.add_argument('--blocks',type=int,default=20)
    p.add_argument('--preflight-only',action='store_true',help='Run three baseline references per DNN; zero model calls')
    p.add_argument('--jobs',choices=['1','2','4'],default='2');args=p.parse_args(argv)
    if len(set(args.networks))!=len(args.networks):p.error('Duplicate DNN names are not allowed')
    if args.reference_budget<250 or args.reference_budget%25 or not 4<=args.blocks<=200:
        p.error('Reference budget: multiple of 25, at least 250; timing blocks: 4..200')
    args.experiment='baseline';root=HERE.parents[1]
    results=(args.out or root/'taso/results/secondary_experiment/candidate_study_3dnns').resolve()
    results.mkdir(parents=True,exist_ok=True);runner=CandidateRunner(results)
    status=dict(status='started',experiment='secondary_candidate_study',started_utc=datetime.now(timezone.utc).isoformat())
    try:
        runner.build(root,args);status.update(status='completed',stage=runner.stage)
    except Exception as exc:
        status.update(status='failed',stage=runner.stage,error=str(exc));runner.write(traceback.format_exc())
    finally:
        (results/'setup_status.json').write_text(json.dumps(status,indent=2)+'\n');runner.log.close()
    return 0 if status['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())

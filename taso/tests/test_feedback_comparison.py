"""Condition separation, read-only cache reuse and comparison resume without paid calls."""
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import run_suite
import results
import reuse_results
from worker import save as write


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    write(path,value)


class FeedbackComparisonTests(unittest.TestCase):
    def cfg(self):
        cfg=json.loads((ROOT/'benchmarks/checkpoint_continuation.json').read_text())
        return dict(cfg,networks=['squeezenet'],models=['google/gemini-3.8-flash'],checkpoint_interval=10)

    def report(self,cfg,trial,completed=True):
        return {'network':'squeezenet','model':cfg['models'][0],'trial':trial,'budget':cfg['budget'],
            'alpha':cfg['alpha'],'search_mode':'checkpoint','checkpoint_interval':cfg['checkpoint_interval'],
            'status':'completed' if completed else 'failed_evaluation','model_calls':10,
            'inference':{'inference_seconds':1.,'usage':{'cost':.1,'total_tokens':100}},
            'evaluation':{'variants':{'baseline':{'measured_ms':1.,'estimated_ms':1.,'search_seconds':.1},
                                       'llm':{'measured_ms':.9,'estimated_ms':.8,'search_seconds':.2}}}}

    def cache(self,path,cfg):
        hashes=run_suite.source_hashes(ROOT/'TASO')
        hashes.update(reuse_results.COMPATIBLE_PLAIN_SOURCE_HASHES)
        save(path/'config.json',cfg)
        save(path/'manifest.json',{'contract':{'config':cfg,'source_sha256':hashes}})
        for trial in range(1,cfg['trials']+1):
            report=self.report(cfg,trial,trial<=3)
            save(path/'trials'/run_suite.model_directory('squeezenet',cfg['models'][0])/f'trial_{trial:02}'/'trial.json',report)

    def test_reuse_has_no_plain_workers_or_calls_and_resume_skips_completed_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);old=directory/'old';new=directory/'new';cfg=self.cfg()
            self.cache(old,cfg);save(directory/'config.json',cfg)
            before={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}
            workers=[];trials=[]
            def fake_worker(stage,network,out,batch,arm,repo,*args,**kwargs):
                workers.append((stage,arm['gpu_feedback']))
                save(out/'worker_result.json',{'status':'completed'})
                return {'status':'completed'}
            def fake_trial(network,model,trial,batch,arm,repo,*args):
                self.assertTrue(arm['gpu_feedback'],'Historical condition was rerun')
                trials.append((network,model,trial))
                report=self.report(arm,trial,trial<=3)
                report.update(feedback_mode='checkpoint_gpu',gpu_feedback=True)
                save(batch/'trials'/run_suite.model_directory(network,model)/f'trial_{trial:02}'/'trial.json',report)
                return report
            command=['--config',str(directory/'config.json'),'--repo',str(ROOT/'TASO'),
                     '--gpu-feedback','both','--reuse-checkpoint-batch',str(old),'--batch-dir',str(new)]
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'fixture'}),patch.object(run_suite,'run_worker',side_effect=fake_worker),patch.object(
                    run_suite,'run_trial',side_effect=fake_trial),patch.object(results,'plot_distributions'),patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected API request')):
                self.assertEqual(run_suite.main(command),0)
            self.assertEqual(workers,[('prepare',True)])
            self.assertEqual(len(trials),5)
            self.assertEqual(before,{str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()})
            with (new/'trials.csv').open() as file:rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),10)
            historical=[r for r in rows if r['feedback_mode']=='checkpoint']
            self.assertTrue(all(r['model_calls_this_run']=='0' and r['api_cost_this_run']=='0' for r in historical))
            self.assertTrue(all(r['model_calls']=='10' for r in historical))
            self.assertEqual((new/'trials.csv').read_bytes(),(new/'trial.csv').read_bytes())
            stats=json.loads((new/'summary.json').read_text())
            self.assertEqual(len(stats['groups']),2)
            self.assertTrue(all(g['completed']==3 and g['planned_trials']==5 and g['meets_minimum'] for g in stats['groups']))
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'fixture'}),patch.object(run_suite,'run_worker',side_effect=AssertionError('Resume started worker')),patch.object(
                    results,'plot_distributions'),patch('urllib.request.urlopen',side_effect=AssertionError('Resume made model call')):
                # Resume using the saved comparison config, as with ordinary batches.
                self.assertEqual(run_suite.main(['--config',str(new/'config.json'),'--batch-dir',str(new),'--repo',str(ROOT/'TASO')]),0)

    def test_protocol_and_unreviewed_sources_fail_before_any_paid_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old';cfg=self.cfg();self.cache(old,cfg)
            hashes=run_suite.source_hashes(ROOT/'TASO')
            for change in ({'checkpoint_interval':25},{'budget':50},{'temperature':.5}):
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Reuse protocol differs'):
                    reuse_results.import_checkpoint_reports(old,dict(cfg,**change),hashes)
            manifest=json.loads((old/'manifest.json').read_text())
            manifest['contract']['source_sha256']['taso/benchmarks/networks.py']='0'*64
            save(old/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'Unreviewed source'):
                reuse_results.import_checkpoint_reports(old,cfg,hashes)

    def test_modes_and_intervals_never_mix_distributions(self):
        cfg=self.cfg();reports=[]
        for mode in ('checkpoint','checkpoint_gpu'):
            for interval in (10,25):
                report=self.report(cfg,1)
                report.update(feedback_mode=mode,checkpoint_interval=interval)
                reports.append(report)
        with tempfile.TemporaryDirectory() as tmp:
            stats=results.write_exports(tmp,reports,plots=False,min_successful_trials=1)
            self.assertEqual(len(stats['groups']),4)
            self.assertTrue(all(g['metrics']['llm_over_baseline_measured_ms']['n']==1 for g in stats['groups']))

    def test_both_modes_interleave_and_reuse_mismatch_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);cfg=dict(self.cfg(),trials=2);save(directory/'config.json',cfg)
            order=[]
            def prepare(stage,network,out,*args,**kwargs):
                save(out/'worker_result.json',{'status':'completed'});return {'status':'completed'}
            def trial(network,model,n,batch,arm,repo):
                order.append((n,arm['gpu_feedback']))
                save(batch/'trials'/run_suite.model_directory(network,model)/f'trial_{n:02}'/'trial.json',self.report(arm,n))
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'fixture'}),patch.object(run_suite,'run_worker',side_effect=prepare),patch.object(
                    run_suite,'run_trial',side_effect=trial),patch.object(results,'plot_distributions'):
                self.assertEqual(run_suite.main(['--config',str(directory/'config.json'),'--gpu-feedback','both',
                    '--batch-dir',str(directory/'both'),'--repo',str(ROOT/'TASO')]),0)
            self.assertEqual(order,[(1,False),(1,True),(2,True),(2,False)])
            old=directory/'old';self.cache(old,dict(cfg,checkpoint_interval=25))
            with patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected API')):
                with self.assertRaisesRegex(ValueError,'checkpoint_interval'):
                    run_suite.main(['--config',str(directory/'config.json'),'--gpu-feedback','on',
                        '--reuse-checkpoint-batch',str(old),'--batch-dir',str(directory/'bad'),'--repo',str(ROOT/'TASO')])
            self.assertFalse((directory/'bad').exists())

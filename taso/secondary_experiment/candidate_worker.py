"""One isolated, fully recorded TASO reference or live Kimi CP25 search."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'benchmarks'))
sys.path.insert(0, str(HERE.parent / 'llm_context'))
sys.path.insert(0, str(HERE.parent))
from worker import all_rules, environment, optimize, save, search_observation
from networks import build_network
from gpu_timing_smoke import measure
from graph_archive import save_snapshot, externalize_node


def safe_outputs(values):
    return [v if math.isfinite(v) else None for v in values]


def compare_outputs(reference, values, rtol, atol):
    finite = len(reference) == len(values) and bool(values) and all(math.isfinite(v) for v in values + reference)
    errors = [abs(a-b) for a,b in zip(reference,values)] if finite else []
    return dict(passed=finite and all(e <= atol + rtol * abs(a) for a,e in zip(reference,errors)),
                finite=finite, size=len(values), max_abs_error=max(errors) if errors else None,
                max_scaled_error=max((e / (atol + rtol*abs(a)) for a,e in zip(reference,errors)), default=None))


class TimingInput:
    def __init__(self,graph,scale):self.graph,self.scale=graph,scale
    def benchmark_runtime(self,blocks):return self.graph.benchmark_runtime_inputs(blocks,1,self.scale)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--network',choices=['squeezenet','resnext50','inceptionv3'],required=True)
    p.add_argument('--mode',choices=['baseline','kimi'],required=True);p.add_argument('--attempt',type=int,required=True)
    args=p.parse_args(argv);cfg=json.loads(args.config.read_text());out=args.out
    weight_store=args.config.resolve().parent/'weights'
    report=dict(status='started',mode=args.mode,attempt=args.attempt,model_calls=0,
                observations=[],gpu_feedback=False,network=args.network)
    save(out/'worker_result.json',report)
    controller=None
    try:
        if args.network not in cfg['networks']:raise ValueError('DNN is outside the saved study configuration')
        repo=HERE.parent/'TASO';os.environ['TASO_HOME']=str(repo);os.environ.pop('TASO_CANDIDATE_AUDIT',None)
        import taso as ts
        graph,source,info=build_network(args.network,ts,repo,cfg['weight_seed'])
        for api in ('audit_snapshot','audit_clone','audit_preprocess','audit_outputs','benchmark_runtime_inputs'):
            if not callable(getattr(graph,api,None)):raise RuntimeError('Rebuild TASO: missing '+api)
        report.update(environment=environment(ts),network_info=info,
                      numerical_scope='Three finite random input seeds; outputs calibrated on original graph; no universal proof',
                      timing_input_seed=1)
        budget=cfg['reference_budget'] if args.mode=='baseline' else cfg['budget']
        # Only an all-rule discarded warm-up; no model requests in this search.
        optimize(ts,graph,all_rules(cfg['budget']),cfg['alpha'],out/'warmup_trace.json',cfg['budget'])
        graph,_,_=build_network(args.network,ts,repo,cfg['weight_seed'])
        original=graph.audit_clone()
        # Calibrate scale on the original only, before any model request. Keep weights unchanged.
        scale=1e-30
        probe=list(original.audit_outputs(1,scale))
        if not probe or any(not math.isfinite(v) for v in probe) or max(abs(v) for v in probe)==0:
            raise RuntimeError('Original output cannot calibrate nonvacuous numerical checks')
        scale=struct.unpack('f',struct.pack('f',max(1e-35,min(1.,scale/max(abs(v) for v in probe)))))[0]
        from restore_graph import terminal_shapes
        reference_shapes=terminal_shapes(json.loads(original.audit_snapshot()))
        references={seed:list(original.audit_outputs(seed,scale)) for seed in cfg['input_seeds']}
        if any(not v or any(not math.isfinite(x) for x in v) or max(abs(x) for x in v)<100*cfg['atol'] for v in references.values()):
            raise RuntimeError('Original numerical reference is nonfinite or too small for the configured tolerance')
        save(out/'numerical_reference.json',dict(scale=scale,seeds=references,rtol=cfg['rtol'],atol=cfg['atol'],
             scope='Finite random inputs and unchanged seeded weights; agreement is empirical, not a proof for every input'))
        if args.mode=='kimi':
            if args.baseline is None or json.loads((args.baseline/'worker_result.json').read_text()).get('network')!=args.network:
                raise ValueError('The recorded TASO reference must use the same DNN')
            from prepare_context import prepare_context
            from checkpoint_strategy import CheckpointStrategist
            prepared=out/'prepared';prepare_context(repo,prepared)
            controller=CheckpointStrategist(cfg['model'],cfg,prepared/'static_context.json',source,out)
        retained=[]
        points=set(cfg['reference_points']+[0]) if args.mode=='baseline' else set(range(0,cfg['budget'],25))
        def select(current,progress):
            consumed=progress['expansions']
            if consumed in points:
                retained.append((consumed,current.audit_clone(),'checkpoint_search'))
            if controller is not None:return controller(current,progress)
            return all_rules(min(25,budget-consumed))
        os.environ['TASO_CANDIDATE_AUDIT']='1'
        trace=out/'search_trace.json';started=time.perf_counter()
        result=ts.optimize_with_checkpoints(graph,select,budget=budget,checkpoint_interval=25,
                                           alpha=cfg['alpha'],trace_file=str(trace))
        os.environ.pop('TASO_CANDIDATE_AUDIT',None)
        report['search']=search_observation(result,trace,budget,time.perf_counter()-started)
        report['search']['timing_scope']='Includes audit I/O and callbacks; do not compare this with historical search-speed results'
        retained.append((report['search']['expansions'],result,'final_execution'))
        if controller is not None:
            report.update(model_calls=controller.model_calls,inference=controller.aggregate())
        report['timing_input_scale']=scale
        restored_reference=None
        if args.mode=='kimi':
            if args.baseline is None:raise ValueError('A recorded TASO reference is required')
            from restore_graph import restore
            reference_report=json.loads((args.baseline/'worker_result.json').read_text())
            choices=[o for o in reference_report['observations'] if o['expansion']==cfg['budget']]
            if not choices:raise ValueError('TASO reference did not reach the requested 100-expansion boundary')
            path=args.baseline/choices[0]['directory']/'execution_graph.json'
            restored_reference,restoration=restore(ts,path)
            report['reference_restoration']=dict(source=str(path),**restoration)
            for seed,reference in references.items():
                if not compare_outputs(reference,list(restored_reference.audit_outputs(seed,scale)),cfg['rtol'],cfg['atol'])['passed']:
                    raise RuntimeError('Restored TASO graph fails numerical validation')
        if restored_reference is not None:
            reference_dir=out/'restored_taso_reference';reference_dir.mkdir()
            reference_timing=dict(numerically_valid=True)
            measure(TimingInput(restored_reference,scale),argparse.Namespace(blocks=cfg['blocks'],spread_threshold_pct=10.),reference_dir,reference_timing)
            report['restored_reference_timing']=reference_timing
        for number,(expansion,snapshot,role) in enumerate(retained):
            directory=out/'observations'/f'{number:02d}_expansion_{expansion:04d}';directory.mkdir(parents=True)
            raw=json.loads(snapshot.audit_snapshot());save_snapshot(directory/'search_graph.json',raw,weight_store)
            execution=snapshot.audit_preprocess()
            execution_snapshot=json.loads(execution.audit_snapshot())
            save_snapshot(directory/'execution_graph.json',execution_snapshot,weight_store)
            observation=dict(expansion=expansion,role=role,native_graph_id=raw['native_graph_id'],
                             estimated_search_ms=snapshot.cost(),estimated_execution_ms=execution.cost(),directory=str(directory.relative_to(out)),
                             numerical_checks={},numerically_valid=True)
            observation['output_shapes']=terminal_shapes(execution_snapshot)
            observation['output_shapes_match']=observation['output_shapes']==reference_shapes
            observation['numerically_valid'] &= observation['output_shapes_match']
            values_by_seed={}
            for seed,reference in references.items():
                values=list(execution.audit_outputs(seed,scale));values_by_seed[seed]=safe_outputs(values)
                check=compare_outputs(reference,values,cfg['rtol'],cfg['atol'])
                observation['numerical_checks'][str(seed)]=check
                observation['numerically_valid'] &= check['passed']
            save(directory/'outputs.json',values_by_seed)
            if args.mode=='baseline' and expansion in cfg['reference_points']:
                from restore_graph import restore
                restored,restoration=restore(ts,directory/'execution_graph.json')
                observation['archive_round_trip']=restoration
                for seed,reference in references.items():
                    if not compare_outputs(reference,list(restored.audit_outputs(seed,scale)),cfg['rtol'],cfg['atol'])['passed']:
                        raise RuntimeError('Recorded reference fails round-trip numerical validation')
            timing_args=argparse.Namespace(blocks=cfg['blocks'],spread_threshold_pct=10.)
            measure(TimingInput(execution,scale),timing_args,directory,observation)
            report['observations'].append(observation);save(out/'worker_result.json',report)
        if args.mode=='baseline' and not any(o['expansion']==cfg['budget'] for o in report['observations']):
            raise RuntimeError('Reference search did not reach the comparison budget; no Kimi calls should be made')
        # Lossless compression only after the complete journal has closed.
        journal=Path(str(trace)+'.candidates.jsonl')
        with journal.open('rt') as src,gzip.open(str(journal)+'.gz','wt',compresslevel=1) as dst:
            for line in src:
                if '"event":"node"' in line:
                    row=json.loads(line);externalize_node(row['node'],weight_store)
                    dst.write(json.dumps(row,separators=(',',':'))+'\n')
                else:dst.write(line)
        journal.unlink();report['journal']=journal.name+'.gz'
        report.update(status='completed',numerically_valid=all(o['numerically_valid'] for o in report['observations']))
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
    finally:
        os.environ.pop('TASO_CANDIDATE_AUDIT',None)
        if controller is not None:report.update(model_calls=controller.model_calls,inference=controller.aggregate())
        save(out/'worker_result.json',report)
    return 0 if report['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())

"""Identity, provenance, stopping rules, numerical checks and native audit serialization."""
import argparse
from collections import Counter
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'secondary_experiment'))
from candidate_identity import prepare, node_label, equivalent, Journal
from candidate_worker import compare_outputs, TimingInput, calibrate_outputs
import candidate_study
from restore_graph import restore


def node(guid,kind=8,attrs=None,weight=None):
    n=dict(guid=guid,type=kind,attributes=attrs if attrs is not None else [0],
           inputs=[[23333,1,2,1,0]],outputs=[[23333,1,2,1,0]],complete=True)
    if weight is not None:n['weight_hex']=struct.pack('ff',*weight).hex()
    return n


def graph(nodes,edges):
    return prepare(dict(nodes=[n['guid'] for n in nodes],edges=edges),
                   {n['guid']:dict(label=node_label(n),complete=n['complete']) for n in nodes})


class CandidateTests(unittest.TestCase):
    def test_guid_order_independent_but_weights_ports_attributes_and_sharing_matter(self):
        a=graph([node(1,0,[]),node(2,1,[],[1,2]),node(3,3,[1,1,0,2])],[(1,0,3,0),(2,0,3,1)])
        b=graph([node(40,3,[1,1,0,2]),node(20,1,[],[1,2]),node(10,0,[])],[(20,0,40,1),(10,0,40,0)])
        self.assertTrue(equivalent(a,b))
        changed=graph([node(40,3,[1,1,0,2]),node(20,1,[],[1,3]),node(10,0,[])],[(20,0,40,1),(10,0,40,0)])
        self.assertFalse(equivalent(a,changed))
        changed=graph([node(40,3,[1,1,0,0]),node(20,1,[],[1,2]),node(10,0,[])],[(20,0,40,1),(10,0,40,0)])
        self.assertFalse(equivalent(a,changed))
        changed=graph([node(40,3,[1,1,0,2]),node(20,1,[],[1,2]),node(10,0,[])],[(20,0,40,0),(10,0,40,1)])
        self.assertFalse(equivalent(a,changed))
        self.assertFalse(equivalent(graph([node(1),node(2)],[(1,0,2,0),(1,0,2,1)]),
                                    graph([node(1),node(2),node(3)],[(1,0,3,0),(2,0,3,1)])))

    def test_equal_refinement_fingerprints_are_not_treated_as_proof(self):
        nodes=[node(i) for i in range(6)]
        a=graph(nodes,[(i,0,(i+1)%6,0) for i in range(6)])
        b=graph(nodes,[(i,0,(i//3)*3+(i+1)%3,0) for i in range(6)])
        self.assertEqual(a['token'],b['token'])
        self.assertFalse(equivalent(a,b))
        self.assertIsNone(equivalent(a,a,max_steps=0))
        bad=copy.deepcopy(a);bad['complete']=False
        self.assertIsNone(equivalent(a,bad))

    def test_missing_weight_bytes_and_unrecorded_edges_fail_closed(self):
        with self.assertRaises(ValueError):node_label(node(1,1,[]))
        with self.assertRaises(ValueError):graph([node(1)],[(1,0,2,0)])

    def test_numeric_agreement_rejects_nan_wrong_size_and_wrong_result(self):
        self.assertTrue(compare_outputs([1.],[1.0005],1e-3,1e-6)['passed'])
        for values in ([2.],[],[float('nan')],[float('inf')]):
            self.assertFalse(compare_outputs([1.],values,1e-3,1e-6)['passed'])
        calls=[]
        timer=TimingInput(types.SimpleNamespace(benchmark_runtime_inputs=lambda *args:calls.append(args)),.25)
        timer.benchmark_runtime(20);self.assertEqual(calls,[(20,1,.25)])

    def test_archive_roundtrip_uses_exact_comparison(self):
        saved=dict(nodes=[node(1,0,[]),node(2)],edges=[[1,0,2,0]])
        class FakeGraph:
            def __init__(self):self.nodes=[];self.edges=[]
            def audit_leaf(self,kind,metadata,data):
                leaf=node(10,kind,[]);leaf['inputs']=[metadata];leaf['outputs']=[metadata];self.nodes.append(leaf);return 10
            def relu(self,x,inplace=False):self.nodes.append(node(20,8,[int(inplace)]));self.edges.append([x,0,20,0]);return 20
            def audit_preprocess(self):return self
            def audit_snapshot(self):return json.dumps(dict(nodes=self.nodes,edges=self.edges))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'numpy':types.ModuleType('numpy')}):
            path=Path(tmp)/'graph.json';path.write_text(json.dumps(saved))
            restored,report=restore(types.SimpleNamespace(new_graph=FakeGraph),path)
            self.assertTrue(report['exact_graph_restoration'])
            saved['nodes'][1]['outputs'][0][2]=99;path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):restore(types.SimpleNamespace(new_graph=FakeGraph),path)

    def test_three_successes_keep_bad_quality_and_failures_then_resume_without_calls(self):
        calls=[]
        def run(command,**kwargs):
            mode=command[command.index('--mode')+1];attempt=int(command[command.index('--attempt')+1])
            directory=Path(command[command.index('--out')+1]);calls.append((mode,attempt))
            failed=mode=='kimi' and attempt==1
            (directory/'worker_result.json').write_text(json.dumps(dict(status='failed' if failed else 'completed',numerically_valid=False)))
            return types.SimpleNamespace(returncode=int(failed))
        with tempfile.TemporaryDirectory() as tmp,patch.object(candidate_study.subprocess,'run',run),\
             patch.dict(os.environ,{'OPENROUTER_API_KEY':'fixture'}),\
             patch('analyze_candidates.analyze',return_value={}),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(out=Path(tmp),reference_budget=500,blocks=4,networks=['squeezenet'],preflight_only=False)
            self.assertEqual(candidate_study.run(args),0)
            self.assertEqual(calls,[('baseline',i) for i in range(1,4)]+[('kimi',i) for i in range(1,5)])
            self.assertEqual(candidate_study.run(args),0);self.assertEqual(len(calls),7)

    def test_preflight_is_baseline_only_and_failure_cap_is_five(self):
        calls=[]
        def run(command,**kwargs):
            mode=command[command.index('--mode')+1];calls.append(mode)
            directory=Path(command[command.index('--out')+1])
            (directory/'worker_result.json').write_text(json.dumps(dict(status='completed')))
            return types.SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp,patch.object(candidate_study.subprocess,'run',run),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(out=Path(tmp),reference_budget=500,blocks=4,networks=['squeezenet'],preflight_only=True)
            self.assertEqual(candidate_study.run(args),0);self.assertEqual(calls,['baseline']*3)
        calls=[]
        def fail(command,**kwargs):
            directory=Path(command[command.index('--out')+1]);calls.append(directory)
            (directory/'worker_result.json').write_text(json.dumps(dict(status='failed')))
            return types.SimpleNamespace(returncode=1)
        with tempfile.TemporaryDirectory() as tmp,patch.object(candidate_study.subprocess,'run',fail),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(out=Path(tmp),reference_budget=500,blocks=4,networks=['squeezenet'],preflight_only=False)
            self.assertEqual(candidate_study.run(args),1);self.assertEqual(len(calls),5)

    def test_end_to_end_analysis_uses_exact_matches_and_reconciles_counters(self):
        from analyze_candidates import compare_trial
        def write_trial(path,offset):
            path.mkdir();(path/'prepared').mkdir()
            (path/'prepared/static_context.json').write_text(json.dumps(dict(catalog=dict(rules=[dict(taso_xfer_index=0,rule_id='builtin:000')]))))
            rows=[dict(event='node',node=node(offset+1,0,[])),dict(event='node',node=node(offset+2))]
            rows += [dict(event='graph',id=1,nodes=[offset+1],edges=[],estimated_ms=2.),
                     dict(event='initial',graph=1,parent=0,expansion=0,phase=0,rule=-1),
                     dict(event='checkpoint',graph=1,parent=0,expansion=0,phase=0,rule=-1),
                     dict(event='expanded',graph=1,parent=0,expansion=0,phase=0,rule=-1),
                     dict(event='graph',id=2,nodes=[offset+1,offset+2],edges=[[offset+1,0,offset+2,0]],estimated_ms=1.)]
            rows += [dict(event=e,graph=2,parent=1,expansion=0,phase=0,rule=0) for e in ('generated','admitted')]
            rows += [dict(event='returned_search',graph=2,parent=0,expansion=1,phase=0,rule=-1)]
            (path/'journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            observation=dict(expansion=1,role='final_execution',numerically_valid=True,estimated_search_ms=1.,
                             timing=[dict(gpu=dict(median_ms=1.)),dict(gpu=dict(median_ms=1.01))],timing_concerns=False)
            report=dict(journal='journal.jsonl',search=dict(expansions=1,counters=dict(generated=1,admitted=1)),
                        numerical_checks={},numerically_valid=True,observations=[observation],
                        reference_restoration=dict(exact_graph_restoration=True,exact_backend_plan_restoration=True),
                        restored_reference_timing=dict(timing=[dict(gpu=dict(median_ms=2.)),dict(gpu=dict(median_ms=2.01))],timing_concerns=False))
            (path/'worker_result.json').write_text(json.dumps(report))
            (path/'search_trace.json').write_text(json.dumps(dict(phases=[dict(rules=[dict(rule_index=0,attempts=1)])])))
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);write_trial(root/'baseline',0);write_trial(root/'kimi',100)
            result=compare_trial(root/'kimi',root/'baseline',dict(reference_points=[1,2],budget=1))
            self.assertEqual(result['overlap']['generated']['1'],{'matched':1})
            self.assertFalse(result['checkpoints'][-1]['same_graph_as_previous'])
            self.assertEqual(result['checkpoints'][-1]['lineage'][0]['rule_id'],'builtin:000')
            self.assertTrue(result['gpu_evidence']['every_guided_mapping_median_below_every_reference_mapping_median'])
            path=root/'kimi/worker_result.json';report=json.loads(path.read_text());report['search']['counters']['generated']=99
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):compare_trial(root/'kimi',root/'baseline',dict(reference_points=[1,2],budget=1))

    def test_all_three_networks_finish_baselines_before_kimi_with_matching_references(self):
        networks=['squeezenet','resnext50','inceptionv3'];calls=[]
        def run(command,**kwargs):
            mode=command[command.index('--mode')+1];network=command[command.index('--network')+1]
            directory=Path(command[command.index('--out')+1]);calls.append((mode,network))
            if mode=='kimi':self.assertEqual(Path(command[command.index('--baseline')+1]).parent.name,network)
            (directory/'worker_result.json').write_text(json.dumps(dict(status='completed',network=network)))
            return types.SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp,patch.object(candidate_study.subprocess,'run',run),\
             patch.dict(os.environ,{'OPENROUTER_API_KEY':'fixture'}),\
             patch('analyze_candidates.analyze',return_value={}),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(out=Path(tmp),reference_budget=500,blocks=4,networks=networks,preflight_only=False)
            self.assertEqual(candidate_study.run(args),0)
            self.assertEqual(calls,[(m,n) for m in ('baseline','kimi') for n in networks for _ in range(3)])
            summary=json.loads((Path(tmp)/'summary.json').read_text())
            for n in networks:
                self.assertEqual(summary['networks'][n]['variants']['kimi']['successful'],3)
            self.assertEqual(candidate_study.run(args),0);self.assertEqual(len(calls),18)

    def test_any_failed_dnn_preflight_blocks_all_paid_groups(self):
        networks=['squeezenet','resnext50','inceptionv3'];calls=[]
        def run(command,**kwargs):
            mode=command[command.index('--mode')+1];network=command[command.index('--network')+1]
            calls.append((mode,network));directory=Path(command[command.index('--out')+1])
            failed=network=='resnext50'
            (directory/'worker_result.json').write_text(json.dumps(dict(status='failed' if failed else 'completed')))
            return types.SimpleNamespace(returncode=int(failed))
        with tempfile.TemporaryDirectory() as tmp,patch.object(candidate_study.subprocess,'run',run),\
             patch.dict(os.environ,{'OPENROUTER_API_KEY':'fixture'}),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(out=Path(tmp),reference_budget=500,blocks=4,networks=networks,preflight_only=False)
            self.assertEqual(candidate_study.run(args),1)
            self.assertEqual(len(calls),11);self.assertTrue(all(m=='baseline' for m,n in calls))
            self.assertEqual(json.loads((Path(tmp)/'summary.json').read_text())['failed_networks'],['resnext50'])

    def test_weight_archives_deduplicate_verify_and_restore_exact_bytes(self):
        from graph_archive import save_snapshot,load_snapshot,weight_digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=root/'weights';first=root/'first.json';second=root/'second.json'
            initial=dict(nodes=[node(1,1,[],[1,2])],edges=[])
            save_snapshot(first,copy.deepcopy(initial),store);save_snapshot(second,copy.deepcopy(initial),store)
            self.assertEqual(len(list(store.glob('*.bin.gz'))),1)
            archived=json.loads(first.read_text());self.assertNotIn('weight_hex',archived['nodes'][0])
            self.assertEqual(load_snapshot(first)['nodes'][0]['weight_hex'],initial['nodes'][0]['weight_hex'])
            reference=archived['nodes'][0]['weight_blob'];self.assertEqual(weight_digest(reference),reference['sha256'])
            self.assertEqual(node_label(archived['nodes'][0]),node_label(initial['nodes'][0]))
            import gzip
            with gzip.open(reference['path'],'wb') as f:f.write(b'changed')
            with self.assertRaises(ValueError):load_snapshot(first)

    def test_reference_probe_diagnostics_distinguish_nan_infinity_and_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'calibration_report.json'
            bad=types.SimpleNamespace(audit_outputs=lambda seed,scale:[float('nan'),float('inf'),0.])
            with self.assertRaisesRegex(RuntimeError,'calibration_report.json'):
                calibrate_outputs(bad,'original',path)
            probes=json.loads(path.read_text())['probes']
            self.assertEqual(len(probes),5)
            self.assertEqual(probes[0]['nan_count'],1);self.assertEqual(probes[0]['infinity_count'],1)
            self.assertEqual(probes[0]['zero_count'],1)
            good=types.SimpleNamespace(audit_outputs=lambda seed,scale:[2.*scale,-scale])
            self.assertAlmostEqual(calibrate_outputs(good,'fan_in',path),.5)

    def test_leaf_restoration_preserves_stride_and_split_descriptors(self):
        layout=[23333,2,2,3,1,2,1,1,2,1,2]
        leaf=node(1,0,[]);leaf['inputs']=[layout];leaf['outputs']=[layout]
        saved=dict(nodes=[leaf],edges=[])
        class Fake:
            def audit_leaf(self,kind,metadata,data):
                self.leaf=node(100,kind,[]);self.leaf['inputs']=[metadata];self.leaf['outputs']=[metadata];return 100
            def audit_preprocess(self):return self
            def audit_snapshot(self):return json.dumps(dict(nodes=[self.leaf],edges=[]))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'numpy':types.ModuleType('numpy')}):
            path=Path(tmp)/'graph.json';path.write_text(json.dumps(saved))
            rebuilt,report=restore(types.SimpleNamespace(new_graph=Fake),path)
            self.assertTrue(report['exact_graph_restoration'])
            self.assertEqual(rebuilt.leaf['outputs'][0],layout)

    def test_fatal_preflight_stops_repeating_a_deterministic_error(self):
        calls=[]
        def run(command,**kwargs):
            directory=Path(command[command.index('--out')+1]);calls.append(directory)
            (directory/'worker_result.json').write_text(json.dumps(dict(status='failed',error='bad archive',fatal_preflight_error=True)))
            return types.SimpleNamespace(returncode=1)
        with tempfile.TemporaryDirectory() as tmp,patch.object(candidate_study.subprocess,'run',run),contextlib.redirect_stdout(io.StringIO()):
            args=argparse.Namespace(out=Path(tmp),reference_budget=500,blocks=4,networks=['squeezenet'],preflight_only=True)
            self.assertEqual(candidate_study.run(args),1);self.assertEqual(len(calls),1)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required')
    def test_actual_native_leaf_metadata_parser_preserves_layout(self):
        source=(ROOT/'TASO/src/core/ops.cc').read_text()
        body=source[source.index('TensorHandle Graph::audit_leaf('):source.index('TensorHandle Graph::audit_pool(')]
        program=r'''
#include "taso/ops.h"
#include <stdexcept>
using namespace taso;
namespace taso {
Op::Op():guid(0),ptr(nullptr){}
Graph::Graph():totalCost(-1){model=nullptr;}
TensorHandle Graph::input_wrapper(const TensorHandle value){return new Tensor(*value);}
TensorHandle Graph::new_weight(const Tensor& value){return new Tensor(value);}
}
'''+body+r'''
int main(){
 Graph graph;vector<int> layout{23333,2,2,3,1,2,1,1,2,1,2};vector<float> values{1,2,3,4,5,6};
 TensorHandle weight=graph.audit_leaf(OP_WEIGHT,layout,values);
 int keys[Tensor::MAX_KEY_LENGTH],length=0;weight->serialize(keys,length);
 assert(vector<int>(keys,keys+length)==layout && static_cast<float*>(weight->data_ptr)[2]==3);
 TensorHandle input=graph.audit_leaf(OP_INPUT,layout,{});length=0;input->serialize(keys,length);
 assert(vector<int>(keys,keys+length)==layout && input->data_ptr==nullptr);
 try{graph.audit_leaf(OP_WEIGHT,layout,{1});assert(false);}catch(const runtime_error&){}
 layout.pop_back();try{graph.audit_leaf(OP_INPUT,layout,{});assert(false);}catch(const runtime_error&){}
 delete input;delete weight;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'test.cc';path.write_text(program)
            result=subprocess.run(['c++','-std=c++11','-I',str(ROOT/'TASO/include'),str(path),'-o',str(root/'test')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            subprocess.run([str(root/'test')],check=True)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required')
    def test_actual_native_snapshot_weight_bytes_and_journal_provenance(self):
        code=r'''
#include "taso/ops.h"
#include "taso/candidate_audit.h"
#include "taso/candidate_snapshot.h"
#include <cstring>
using namespace taso;
namespace taso {
Model::Model() {}
bool Model::copy_memory(float* dst,const float* src,size_t n){memcpy(dst,src,n);return true;}
Op::Op():guid(0),ptr(nullptr){}
Graph::Graph():totalCost(1){}
float Graph::total_cost(){return totalCost;}
OpBase::OpBase(Model* m,OpType t):numInputs(1),numOutputs(1),model(m),type(t),runtime(0){}
bool OpBase::get_input_parameter(TNParameter,DIMParameter,int*){return false;}
bool OpBase::get_int_parameter(PMParameter,int*){return false;}
bool OpBase::get_float_parameter(PMParameter,float*){return false;}
}
class FakeOp:public OpBase {
public:FakeOp(Model* m,OpType t,float* data):OpBase(m,t){
 for(int i=0;i<2;i++){Tensor& x=i?outputs[0]:inputs[0];x.numDim=1;x.dim[0]=2;x.stride[0]=1;x.data_ptr=data;}}
 void forward(bool){}void map(){}void unmap(){}
 void collect_costs(float&,float&,float&,int&){}
};
int main(int argc,char** argv){
 assert(argc==2);Model model;float values[]={1,2};FakeOp input(&model,OP_INPUT,nullptr),weight(&model,OP_WEIGHT,values);
 Graph a,b;a.model=b.model=&model;a.inEdges[Op(100,&input)];a.inEdges[Op(101,&weight)];b.inEdges=a.inEdges;
 setenv("TASO_CANDIDATE_AUDIT","1",1);
 {CandidateAudit audit(argv[1]);audit.capture("initial",&a);audit.expansion=24;audit.phase=0;audit.rule=13;
 audit.capture("generated",&b,&a);audit.ref("admitted",&b,&a);audit.expansion=25;audit.ref("expanded",&b);
 assert(a.audit_id==1 && b.audit_id==2);}
 assert(CandidateAudit::active()==nullptr);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);cpp=folder/'test.cc';cpp.write_text(code)
            proc=subprocess.run(['c++','-std=c++11','-I',str(ROOT/'TASO/include'),str(cpp),'-o',str(folder/'test')],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            subprocess.run([str(folder/'test'),str(folder/'trace.json')],check=True)
            journal=Journal(folder/'trace.json.candidates.jsonl')
            self.assertTrue(equivalent(journal.get(1),journal.get(2)))
            self.assertEqual(journal.ids('generated'),{2})
            self.assertEqual(journal.lineage(2)[0]['rule'],13)
            self.assertEqual(journal.lineage(2)[0]['parent'],1)
            self.assertEqual(journal.lineage(2)[0]['expansion'],24)

if __name__=='__main__':unittest.main()

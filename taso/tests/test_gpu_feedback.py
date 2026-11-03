"""Real feedback packaging and production native snapshot ownership without CUDA/API calls."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'llm_context'))
sys.path.insert(0,str(ROOT/'benchmarks'))
import checkpoint_strategy
from gpu_feedback import measure_best_graph


class GpuFeedbackTests(unittest.TestCase):
    def test_median_change_and_invalid_measurements(self):
        graph=types.SimpleNamespace(run_time_for_feedback=iter([1.,3.,2.,1.,1.5,2.]).__next__)
        initial=measure_best_graph(graph,0,[])
        next_one=measure_best_graph(graph,10,[initial])
        self.assertEqual(initial['median_ms'],2.)
        self.assertEqual(next_one['median_ms'],1.5)
        self.assertEqual(next_one['improvement_since_previous_ms'],.5)
        self.assertAlmostEqual(next_one['speedup_vs_initial'],2/1.5)
        for value in (0.,-1.,float('nan')):
            with self.assertRaises(RuntimeError):
                measure_best_graph(types.SimpleNamespace(run_time_for_feedback=lambda:value),0,[])
        with self.assertRaisesRegex(RuntimeError,'Rebuild TASO'):
            measure_best_graph(object(),0,[])

    def test_enabled_feedback_reaches_prompt_and_disabled_mode_never_probes(self):
        import test_checkpoints as fixture
        spec=importlib.util.spec_from_file_location('taso.strategy',ROOT/'TASO/python/taso/strategy.py')
        strategy=importlib.util.module_from_spec(spec);spec.loader.exec_module(strategy)
        fake=types.ModuleType('taso');fake.operator_attrs={}
        from prepare_context import prepare_context
        cfg=json.loads((ROOT/'benchmarks/checkpoint_continuation.json').read_text())
        cfg.update(budget=20,checkpoint_interval=10)
        for enabled in (False,True):
            with self.subTest(enabled=enabled),tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'taso':fake,'taso.strategy':strategy}):
                directory=Path(tmp);prepare_context(ROOT/'TASO',directory)
                probes=[]
                def probe(graph):
                    probes.append(graph.value)
                    if not enabled:raise AssertionError('GPU probe in disabled mode')
                    return graph.value/10
                sent=[]
                class Client:
                    def __init__(self,*args,**kwargs):self.request_attempts=0;self.metadata=None
                    def __call__(self,messages):
                        sent.append(copy.deepcopy(messages));self.request_attempts=1
                        self.metadata={'usage':{'prompt_tokens':10,'completion_tokens':5,'cost':.01}}
                        return json.dumps(fixture.strategy(10))
                controller=checkpoint_strategy.CheckpointStrategist('google/gemini-3.8-flash',
                    dict(cfg,gpu_feedback=enabled),directory/'static_context.json','model',directory,client_factory=Client)
                with patch.object(fixture.Graph,'run_time_for_feedback',probe,create=True):
                    fixture.optimize_with_checkpoints(fixture.Graph(),controller,budget=20,checkpoint_interval=10)
                self.assertEqual(len(sent),2)
                self.assertEqual(len(probes),6 if enabled else 0)
                text=sent[1][1]['content']
                if enabled:
                    self.assertIn('"median_ms":0.99',text)
                    self.assertIn('improvement_since_previous_ms',text)
                    self.assertNotIn('All cost feedback is estimated;',text)
                    self.assertNotIn('Do not claim measured GPU improvements.',text)
                    self.assertEqual(len(controller.gpu_history),2)
                else:
                    self.assertNotIn('"gpu_feedback":',text)
                    self.assertIn('All cost feedback is estimated;',text)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required')
    def test_production_snapshot_folds_constants_and_releases_temporaries(self):
        source=(ROOT/'TASO/src/core/ops.cc').read_text()
        body=source[source.index('float Graph::run(void)'):source.index('void Graph::print_costs(void)')]
        fixture=(ROOT/'tests/gpu_runtime_fixture.h').read_text()
        program=fixture+'\n'+body+r'''
int main() {
 Model model; Graph graph;graph.model=&model;
 Tensor input,w1,w2;w1.data_ptr=(float*)model.allocate_memory(4);w2.data_ptr=(float*)model.allocate_memory(4);
 NoOp* ip=new NoOp(&model,input,OP_INPUT);NoOp* a=new NoOp(&model,w1,OP_WEIGHT);NoOp* b=new NoOp(&model,w2,OP_WEIGHT);
 Enlarge* en=new Enlarge(&model,w1,w2);Constant* cn=new Constant(&model,1,w1.dim,OP_CONSTANT_ONE);
 Mul* mul=new Mul(&model,w1,w2);Conv2D* conv=new Conv2D(&model,input,w1,1,1,0,0);
 Activation* act=new Activation(&model,input,OP_RELU,true);
 Op ghost{1,nullptr},i{100,ip},x{101,a},y{102,b},e{103,en},c{104,cn},m{105,mul},v{106,conv},r{107,act};
 graph.edge(ghost,i);graph.edge(ghost,x);graph.edge(ghost,y);graph.edge(x,e,0);graph.edge(y,e,1);
 graph.inEdges[c];graph.edge(c,m,0);graph.edge(e,m,1);graph.edge(i,v,0);graph.edge(m,v,1);graph.edge(v,r);
 const auto in_size=graph.inEdges.size(),out_size=graph.outEdges.size();const int live=model.live;
 for(int n=0;n<100;n++) {
  assert(graph.run_time_for_feedback()==.5f);
  assert(model.blocks.size()==2 && model.live==live && model.guid==500);
  assert(graph.inEdges.size()==in_size && graph.outEdges.size()==out_size);
  assert(conv->inputs[0].data_ptr==nullptr && conv->outputs[0].data_ptr==nullptr);
 }
 assert(model.forwards[OP_ENLARGE]==100 && model.forwards[OP_MUL]==100 && model.forwards[OP_CONSTANT_ONE]==100);
 model.fail_measure=true;try {graph.run_time_for_feedback();assert(false);}catch(const runtime_error&){}
 assert(model.blocks.size()==2 && model.live==live);
 act->type=OP_ANY;model.fail_measure=false;try {graph.run_time_for_feedback();assert(false);}catch(const runtime_error&){}
 assert(model.blocks.size()==2 && model.live==live);
 delete ip;delete a;delete b;delete en;delete cn;delete mul;delete conv;delete act;
 model.release_memory(w1.data_ptr);model.release_memory(w2.data_ptr);assert(model.live==0 && model.blocks.empty());
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);cpp=directory/'test.cc';cpp.write_text(program)
            subprocess.run(['c++','-std=c++11',str(cpp),'-o',str(directory/'check')],check=True,capture_output=True,text=True)
            subprocess.run([str(directory/'check')],check=True,capture_output=True,text=True)

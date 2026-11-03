"""Execute the actual C++ search-loop source with a synthetic graph backend."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SearchEngineTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'A C++ compiler is required')
    def test_production_loop_budget_frontier_and_cleanup(self):
        source = (ROOT / 'TASO/src/core/ops.cc').read_text()
        start = source.index('// Shared trace writer')
        stop = source.index('Graph* Graph::preprocess_weights(', start)
        fixture = (ROOT / 'tests/search_engine_fixture.h').read_text()
        program = fixture + '\n' + source[start:stop] + r'''
struct PolicyState {
  vector<int> snapshots;
  vector<string> progress;
  int fail_at=-1;
};
int online(void* raw,const string& progress,vector<int>& rules,void* context) {
  Graph* graph=static_cast<Graph*>(raw);
  PolicyState& state=*static_cast<PolicyState*>(context);
  state.snapshots.push_back(graph->id);
  state.progress.push_back(progress);
  if (graph->id==state.fail_at) return -1;
  rules=GraphXfer::chain ? vector<int>{17} :
        (state.snapshots.size()==1 ? vector<int>{1} : vector<int>{0});
  return 0;
}
int main(int argc,char** argv) {
  assert(argc==2);
  string out=argv[1];
  Graph* result=(new Graph)->optimize_scheduled(1.5,{2},{{}},false,out+"/single.json");
  assert(result->id==3 && result->cost==5 && Graph::live==1);
  delete result;
  assert(Graph::live==0);
  result=(new Graph)->optimize_scheduled(1.5,{1,1},{{},{}},false,out+"/split.json");
  assert(result->id==3 && Graph::live==1);
  delete result;
  result=(new Graph)->optimize_scheduled(1.5,{1,1},{{1},{0}},false,out+"/manual.json");
  assert(result->id==4 && result->cost==4 && Graph::live==1);
  delete result;
  result=(new Graph)->optimize(1.5,1,false);
  assert(result->id==3 && Graph::live==1);
  delete result;
  result=(new Graph)->optimize_scheduled(1.5,{10},{{2}},false,out+"/empty.json");
  assert(result->id==0 && Graph::live==1);
  delete result;
  assert(Graph::live==0);
  PolicyState state;
  result=(new Graph)->optimize_checkpointed(1.5,2,1,online,&state,false,out+"/online.json");
  assert((state.snapshots==vector<int>{0,2}));
  assert(result->id==4 && Graph::live==1);
  delete result;
  state=PolicyState();
  result=(new Graph)->optimize_checkpointed(1.5,100,25,online,&state,false,out+"/online_empty.json");
  assert((state.snapshots==vector<int>{0}));
  assert(result->id==2 && Graph::live==1);
  delete result;
  GraphXfer::chain=true;
  for (int interval : {25,10,5,100,200}) {
    state=PolicyState();
    result=(new Graph)->optimize_checkpointed(1.5,100,interval,online,&state,false,
        out+"/interval_"+to_string(interval)+".json");
    assert(result->id==100 && Graph::live==1);
    for (size_t i=0; i<state.snapshots.size(); ++i)
      assert(state.snapshots[i]==(int)i*interval);
    assert(state.snapshots.size()==(size_t)((99/interval)+1));
    for (size_t i=0; i<state.progress.size(); ++i)
      ofstream(out+"/progress_"+to_string(interval)+"_"+to_string(i)+".json") << state.progress[i];
    delete result;
  }
  state=PolicyState();
  result=(new Graph)->optimize_checkpointed(1.5,63,25,online,&state,false,out+"/remainder.json");
  assert(result->id==63);
  delete result;
  state=PolicyState(); state.fail_at=25;
  try {
    (new Graph)->optimize_checkpointed(1.5,100,25,online,&state,false,"");
    assert(false);
  } catch (const runtime_error&) {}
  assert(Graph::live==0);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cpp = folder / 'fixture.cc'
            binary = folder / 'fixture'
            cpp.write_text(program)
            subprocess.run(['c++', '-std=c++11', '-I', str(ROOT / 'TASO/include'), str(cpp), '-o', str(binary)], check=True, capture_output=True)
            env = dict(os.environ, TASO_HOME=str(ROOT / 'TASO'))
            subprocess.run([str(binary), str(folder)], env=env, check=True, capture_output=True)
            single = json.loads((folder / 'single.json').read_text())
            split = json.loads((folder / 'split.json').read_text())
            manual = json.loads((folder / 'manual.json').read_text())
            empty = json.loads((folder / 'empty.json').read_text())
            for interval in (25, 10, 5, 100, 200):
                trace = json.loads((folder / f'interval_{interval}.json').read_text())
                self.assertEqual(trace['expansions'], 100)
                self.assertEqual(trace['stop_reason'], 'budget')
                self.assertAlmostEqual(trace['best_search_cost_ms'], 9)
                count = (99 // interval) + 1
                for index in range(count):
                    progress = json.loads((folder / f'progress_{interval}_{index}.json').read_text())
                    self.assertEqual(progress['expansions'], index * interval)
                    self.assertAlmostEqual(progress['best_search_cost_ms'], 10 - index * interval * .01, places=4)
            remainder = json.loads((folder / 'remainder.json').read_text())
            self.assertEqual([p['expansions'] for p in remainder['phases']], [25, 25, 13])
            online = json.loads((folder / 'online.json').read_text())
            self.assertEqual(online['best_search_cost_ms'], 4)
            online_empty = json.loads((folder / 'online_empty.json').read_text())
            self.assertEqual(online_empty['expansions'], 2)
            self.assertEqual(online_empty['stop_reason'], 'queue_empty')
            self.assertEqual(single['expansions'], 2)
            self.assertEqual([p['expansions'] for p in split['phases']], [1, 1])
            self.assertEqual(single['best_search_cost_ms'], split['best_search_cost_ms'])
            self.assertEqual(manual['best_search_cost_ms'], 4)
            self.assertEqual(empty['stop_reason'], 'queue_empty')
            self.assertEqual(empty['expansions'], 1)
            self.assertEqual(manual['phases'][0]['rules'][0]['attempts'], 0)
            self.assertEqual(manual['phases'][0]['rules'][1]['attempts'], 1)


if __name__ == '__main__':
    unittest.main()

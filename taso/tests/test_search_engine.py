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
        start = source.index('Graph* Graph::optimize(float')
        stop = source.index('Graph* Graph::preprocess_weights(', start)
        fixture = (ROOT / 'tests/search_engine_fixture.h').read_text()
        program = fixture + '\n' + source[start:stop] + r'''
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

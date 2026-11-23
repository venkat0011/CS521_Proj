"""Diagnostic orchestration, non-selection of stable results, and native ownership."""
import argparse
import contextlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'secondary_experiment'))
import gpu_timing_smoke as smoke


class SmokeTests(unittest.TestCase):
    def test_statistics_detect_drift_without_filtering_samples(self):
        stats = smoke.summarize([1., 1., 2., 2.])
        self.assertEqual(stats['n'], 4)
        self.assertEqual(stats['drift_pct'], 100.)
        for values in ([1, 0, 1, 1], [1, 1, float('nan'), 1], [1, 1]):
            with self.assertRaises(ValueError):
                smoke.summarize(values)

    def args(self, path):
        return argparse.Namespace(out=path, repo=ROOT / 'TASO', network='squeezenet',
            budget=100, interval=25, seed=521, blocks=4, replay_trial=[],
            spread_threshold_pct=10., worker_timeout=30)

    def test_baseline_batch_then_checkpoint_stop_three_include_unstable_resume(self):
        launched = []
        def run(command, **kwargs):
            variant = command[command.index('--variant') + 1]
            attempt = int(command[command.index('--attempt') + 1])
            out = Path(command[command.index('--out') + 1])
            launched.append((variant, attempt))
            # Failed first checkpoint is retained; no selective exclusion of unstable runs.
            failed = variant == 'checkpoint' and attempt == 1
            report = dict(status='failed' if failed else 'completed', timing_concerns=True,
                timing=[dict(gpu=dict(median_ms=1., range_pct=50.)) for _ in range(2)])
            (out / 'worker_result.json').write_text(json.dumps(report))
            return types.SimpleNamespace(returncode=int(failed))
        with tempfile.TemporaryDirectory() as tmp, patch.object(smoke.subprocess, 'run', run), contextlib.redirect_stdout(io.StringIO()):
            args = self.args(Path(tmp))
            self.assertEqual(smoke.execute_batch(args), 0)
            self.assertEqual(launched, [('baseline', n) for n in range(1, 4)] +
                             [('checkpoint', n) for n in range(1, 5)])
            self.assertEqual(smoke.execute_batch(args), 0)
            self.assertEqual(len(launched), 7)
            summary = json.loads((args.out / 'summary.json').read_text())
            self.assertEqual(summary['variants']['checkpoint']['successful'], 3)
            self.assertEqual(summary['model_calls'], 0)

    def test_five_attempt_limit_no_checkpoint_after_failed_baseline_batch(self):
        launched = []
        def run(command, **kwargs):
            out = Path(command[command.index('--out') + 1]); launched.append(out)
            (out / 'worker_result.json').write_text(json.dumps(dict(status='failed', error='GPU failure')))
            return types.SimpleNamespace(returncode=1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(smoke.subprocess, 'run', run), contextlib.redirect_stdout(io.StringIO()):
            args = self.args(Path(tmp))
            self.assertEqual(smoke.execute_batch(args), 1)
            self.assertEqual(len(launched), 5)
            self.assertFalse((args.out / 'checkpoint').exists())

    def test_measure_retains_every_sample_and_two_mapping_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp)); report = {}
            graph = types.SimpleNamespace(benchmark_runtime=lambda blocks: [1., 1.1, 1., 1.1, 2., 2.1, 2., 2.1])
            with patch.object(smoke, 'telemetry', return_value='fixture'):
                smoke.measure(graph, args, args.out, report)
            self.assertTrue(report['timing_concerns'])
            self.assertEqual(len((args.out / 'timing_samples.csv').read_text().splitlines()), 9)
            self.assertEqual(len(report['timing']), 2)

    def test_replay_requires_complete_budget_and_boundary_alignment(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('taso.strategy', ROOT / 'TASO/python/taso/strategy.py')
        strategy = importlib.util.module_from_spec(spec); spec.loader.exec_module(strategy)
        records = [dict(expansions=e, strategy=smoke.all_rules(25)) for e in (0, 25, 50, 75)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'taso.strategy': strategy}):
            directory = Path(tmp)
            path = directory / 'checkpoints.json'
            path.write_text(json.dumps(records))
            self.assertEqual(set(smoke.replay_policies(directory, 100, 25)), {0, 25, 50, 75})
            path.write_text(json.dumps(records[:-1]))
            with self.assertRaises(ValueError):
                smoke.replay_policies(directory, 100, 25)

    def test_setup_dispatches_new_diagnostic_and_replay_paths(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('smoke_setup_test', ROOT / 'secondary_experiment/run_gpu_smoke.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        args = module.parse_args(['--replay-trial', '/tmp/replay'])
        runner = object.__new__(module.SecondaryRunner)
        command = runner.experiment_command(ROOT.parent, args, Path('/tmp/python'), Path('/tmp/out'))
        self.assertIn('gpu_timing_smoke.py', str(command))
        self.assertIn('--replay-trial', command)
        self.assertEqual(command[command.index('--budget') + 1], '100')

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_native_fixed_mapping_cleanup_and_block_bounds(self):
        source = (ROOT / 'TASO/src/core/ops.cc').read_text()
        body = source[source.index('float Graph::run(void)'):source.index('void Graph::print_costs(void)')]
        fixture = (ROOT / 'tests/gpu_runtime_fixture.h').read_text()
        main = r'''
int main() {
 Model model;Graph graph;graph.model=&model;
 Tensor input,weight;weight.data_ptr=(float*)model.allocate_memory(4);
 NoOp ip(&model,input,OP_INPUT), w(&model,weight,OP_WEIGHT);
 Conv2D conv(&model,input,weight,1,1,0,0);
 Op ghost{1,nullptr},i{100,&ip},x{101,&w},v{102,&conv};
 graph.edge(ghost,i);graph.edge(ghost,x);graph.edge(i,v,0);graph.edge(x,v,1);
 int live=model.live;
 for(int n=0;n<20;n++) {
  auto samples=graph.benchmark_runtime(4);assert(samples.size()==8);
  assert(samples[0]==.5f && model.blocks.size()==1 && model.live==live);
 }
 // Exactly 4 measured forwards per mapping in the fixture, no remapping per block.
 assert(model.forwards[OP_CONV2D]==80);
 for(int bad:{0,3,201}) {try {graph.benchmark_runtime(bad);assert(false);}catch(const runtime_error&){} }
 model.fail_measure=true;
 try {graph.benchmark_runtime(4);assert(false);}catch(const runtime_error&){}
 assert(model.blocks.size()==1 && model.live==live);
 model.release_memory(weight.data_ptr);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); cpp = directory / 'test.cc'; cpp.write_text(fixture + body + main)
            subprocess.run(['c++', '-std=c++11', '-DUSE_CUDNN', str(cpp), '-o', str(directory / 'check')],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(directory / 'check')], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()

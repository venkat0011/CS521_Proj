"""Neutral experiment entry point, compatible aliases and output-root regression."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
import run_experiment as runner
import run_suite
import worker
from experiments import run_llm_strategy, run_manual_strategy


class LayoutTests(unittest.TestCase):
    def test_old_entry_points_delegate_to_canonical_implementations(self):
        for old_name, expected in [('setup_and_run', runner),
                                   ('run_manual_strategy', run_manual_strategy),
                                   ('run_llm_strategy', run_llm_strategy)]:
            with self.subTest(entry=old_name):
                spec = importlib.util.spec_from_file_location('legacy_' + old_name, ROOT / 'baseline' / (old_name + '.py'))
                old = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(old)
                self.assertIs(old.main, expected.main)
                self.assertIs(old.parse_args, expected.parse_args)
        self.assertEqual(run_manual_strategy.parse_args([]).out, ROOT / 'results/manual')
        self.assertEqual(run_llm_strategy.parse_args([]).out, ROOT / 'results')

    def test_launcher_root_defaults_and_dispatch_after_moving_entry_point(self):
        for experiment in ('baseline', 'manual', 'llm', 'suite'):
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp).resolve()
                args = runner.parse_args(['--experiment', experiment])
                calls = []
                class FakeRunner:
                    def __init__(self, results):
                        self.results, self.stage, self.log = results, experiment, io.StringIO()
                    def build(self, root, parsed):
                        calls.append((root, runner.experiment_command(root, parsed, Path('/tmp/python'), self.results)))
                    def write(self, text): pass
                with patch.object(runner, '__file__', str(project / 'taso/run_experiment.py')), patch.object(
                        runner, 'parse_args', return_value=args), patch.object(runner, 'Runner', FakeRunner):
                    self.assertEqual(runner.main(), 0)
                expected = project / ('taso/baseline/results' if experiment == 'baseline' else
                                      'taso/results/manual' if experiment == 'manual' else 'taso/results')
                self.assertEqual(json.loads((expected / 'setup_status.json').read_text())['status'], 'completed')
                self.assertEqual(calls[0][0], project)
                script = {'baseline': 'baseline/run_baseline.py', 'manual': 'experiments/run_manual_strategy.py',
                          'llm': 'experiments/run_llm_strategy.py', 'suite': 'benchmarks/run_suite.py'}[experiment]
                self.assertEqual(calls[0][1][1], project / 'taso' / script)

    def test_default_suite_batches_are_grouped_by_mode_outside_baseline(self):
        for mode in ('one_shot', 'checkpoint'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                folder = Path(tmp)
                cfg = json.loads((ROOT / 'benchmarks/config.json').read_text())
                cfg.update(networks=['squeezenet'], models=['google/gemini-3.8-flash'], trials=1)
                if mode == 'checkpoint':
                    cfg.update(search_mode='checkpoint', checkpoint_interval=25)
                worker.save(folder / 'config.json', cfg)
                def prepare(stage, network, directory, *args):
                    directory.mkdir(parents=True, exist_ok=True)
                    report = {'status': 'completed'}
                    worker.save(directory / 'worker_result.json', report)
                    return report
                with patch.object(run_suite, 'source_hashes', return_value={}), patch.object(
                        run_suite, 'run_worker', side_effect=prepare):
                    self.assertEqual(run_suite.main(['--config', str(folder / 'config.json'),
                        '--out', str(folder / 'results'), '--dry-run']), 0)
                batches = list((folder / 'results' / mode).iterdir())
                self.assertEqual(len(batches), 1)
                self.assertTrue((batches[0] / 'manifest.json').exists())
                self.assertFalse((folder / 'results/suite').exists())
        self.assertEqual(run_suite.parse_args([]).out, ROOT / 'results')


if __name__ == '__main__':
    unittest.main()

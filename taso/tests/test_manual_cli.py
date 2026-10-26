"""Regression for the setup runner's manual-experiment arguments."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('manual', ROOT / 'baseline/run_manual_strategy.py')
manual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manual)
llm_spec = importlib.util.spec_from_file_location('llm_experiment', ROOT / 'baseline/run_llm_strategy.py')
llm_experiment = importlib.util.module_from_spec(llm_spec)
llm_spec.loader.exec_module(llm_experiment)


class ManualCliTests(unittest.TestCase):
    def test_accepts_arguments_passed_by_setup_runner(self):
        repo = ROOT / 'TASO'
        out = ROOT / 'baseline/results'
        args = manual.parse_args(['--repo', str(repo), '--out', str(out)])
        self.assertEqual(args.repo, repo)
        self.assertEqual(args.out, out)

    def test_llm_accepts_setup_runner_arguments(self):
        args = llm_experiment.parse_args(['--repo', str(ROOT / 'TASO'), '--out', '/tmp/output',
                                          '--model', 'deepseek/deepseek-v4-pro', '--max-tokens', '8192', '--dry-run'])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.repo, ROOT / 'TASO')
        self.assertEqual(args.max_tokens, 8192)

    def test_default_source_matches_sibling_checkout(self):
        self.assertEqual(manual.parse_args([]).repo, ROOT / 'TASO')


if __name__ == '__main__':
    unittest.main()

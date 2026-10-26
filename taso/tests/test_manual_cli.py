"""Regression for the setup runner's manual-experiment arguments."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('manual', ROOT / 'baseline/run_manual_strategy.py')
manual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manual)


class ManualCliTests(unittest.TestCase):
    def test_accepts_arguments_passed_by_setup_runner(self):
        repo = ROOT / 'TASO'
        out = ROOT / 'baseline/results'
        args = manual.parse_args(['--repo', str(repo), '--out', str(out)])
        self.assertEqual(args.repo, repo)
        self.assertEqual(args.out, out)

    def test_default_source_matches_sibling_checkout(self):
        self.assertEqual(manual.parse_args([]).repo, ROOT / 'TASO')


if __name__ == '__main__':
    unittest.main()

"""Tensat setup script: command list, defaults and the stop point in TASO's Runner."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_and_run as tensat_setup

BINARY = Path('/tmp/tensat')
STATS = Path('/tmp/stats.txt')
COMMON = [BINARY, '-r', 'converted.txt', '-t', 'converted_multi.txt', '-u', '-s', 'none',
          '--n_iter', '3', '--iter_multi', '1', '--n_sec', '600', '--n_nodes', '50000',
          '--all_weight_only']


def stop_at_bindings(self, root, args):
    self.install_bindings(root, root / 'taso/.venv/bin/python', root / 'taso/TASO')


def finish_without_bindings(self, root, args):
    return None


class SetupTests(unittest.TestCase):
    def test_default_command_is_squeezenet_with_greedy_extraction(self):
        command = tensat_setup.tensat_command(BINARY, tensat_setup.parse_args([]), STATS)
        self.assertEqual(command, COMMON + ['-e', 'greedy', '-d', 'squeezenet', '-o', STATS])

    def test_ilp_command_adds_upstream_ilp_flags(self):
        args = tensat_setup.parse_args(['--extract', 'ilp', '--model', 'resnext50'])
        command = tensat_setup.tensat_command(BINARY, args, STATS)
        self.assertEqual(command, COMMON + ['-e', 'ilp', '-d', 'resnext50', '-o', STATS,
                                            '--no_order', '--no_cycle', '--ilp_time_sec', '0'])

    def test_rule_files_named_in_command_exist(self):
        command = tensat_setup.tensat_command(BINARY, tensat_setup.parse_args([]), STATS)
        for flag in ('-r', '-t'):
            rule_file = command[command.index(flag) + 1]
            self.assertTrue((tensat_setup.TENSAT / rule_file).is_file(), rule_file)

    def test_taso_build_stops_before_python_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = tensat_setup.TensatRunner(root)
            try:
                with patch.object(tensat_setup.Runner, 'build', stop_at_bindings):
                    self.assertEqual(runner.build_taso_runtime(root, '2'), root / 'taso/.venv/native')
                with patch.object(tensat_setup.Runner, 'build', finish_without_bindings):
                    with self.assertRaises(RuntimeError):
                        runner.build_taso_runtime(root, '2')
            finally:
                runner.log.close()


if __name__ == '__main__':
    unittest.main()

"""Installed-extension verification and forced rebuild contract without a GPU."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
import run_experiment
import verify_bindings
import worker
from test_operator_introspection import namespace


class BindingInstallTests(unittest.TestCase):
    def modules(self, prefix, table=None, path=None):
        path = path or prefix / 'lib/python3.11/site-packages/taso/core.fixture.so'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture-extension')
        core = types.ModuleType('taso.core')
        core.__file__ = str(path)
        core.op_table = dict(namespace['op_table']) if table is None else table
        taso = types.ModuleType('taso')
        taso.__file__ = str(path.parent / '__init__.py')
        taso.optimize_with_checkpoints = lambda *args: None
        return patch.dict(sys.modules, {'taso': taso, 'taso.core': core})

    def test_installed_table_matches_source_and_records_actual_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            with self.modules(prefix):
                report = verify_bindings.inspect_binding(ROOT / 'TASO', prefix)
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['type_20'], 'Enlarge')
            self.assertEqual(report['declared_operator_count'], 62)
            self.assertEqual(report['installed_mapping_count'], 62)
            self.assertEqual(report['compiled_binding_sha256'], hashlib.sha256(b'fixture-extension').hexdigest())
            self.assertTrue(Path(report['compiled_binding']).is_relative_to(prefix.resolve()))

    def test_old_compiled_table_is_rejected_despite_correct_source(self):
        table = dict(namespace['op_table'])
        table.pop(20)
        table.pop(21)
        with tempfile.TemporaryDirectory() as tmp, self.modules(Path(tmp), table):
            report = verify_bindings.inspect_binding(ROOT / 'TASO', tmp)
        self.assertEqual(report['status'], 'failed')
        self.assertIsNone(report['type_20'])
        self.assertEqual({m['value'] for m in report['mismatches']}, {20, 21})
        self.assertIn('Installed binding differs', report['errors'][0])

    def test_another_environment_cannot_shadow_the_selected_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'venv'
            with self.modules(prefix, path=Path(tmp) / 'other/core.fixture.so'):
                report = verify_bindings.inspect_binding(ROOT / 'TASO', prefix)
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(any('outside' in error for error in report['errors']))

    def test_old_checkout_source_is_reported_separately_from_old_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            repo = folder / 'TASO'
            header = repo / 'include/taso/ops.h'
            pyx = repo / 'python/taso/_cython/core.pyx'
            header.parent.mkdir(parents=True)
            pyx.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / 'TASO/include/taso/ops.h', header)
            pyx.write_text((ROOT / 'TASO/python/taso/_cython/core.pyx').read_text().replace(
                'op_table[OP_ENLARGE] = "Enlarge"\n', ''))
            with self.modules(folder / 'venv'):
                report = verify_bindings.inspect_binding(repo, folder / 'venv')
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(any('Checkout source' in error and 'OP_ENLARGE' in error for error in report['errors']))

    def test_setup_forces_generation_compilation_copy_and_checks_installed_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            runner = run_experiment.Runner(folder)
            root, python = folder / 'project', Path('/tmp/python')
            try:
                with patch.object(runner, 'run') as run:
                    runner.install_bindings(root, python, root / 'taso/TASO')
                self.assertEqual(runner.env['TASO_FORCE_CYTHON'], '1')
                commands = run.call_args_list
                self.assertEqual(commands[0].args[0], [python, 'setup.py', 'build_ext', '--force', 'install', '--force'])
                self.assertEqual(commands[1].args[0][1], root / 'taso/verify_bindings.py')
                self.assertEqual(commands[1].args[0][-1], folder / 'bindings_check.json')
            finally:
                runner.log.close()

    def test_stale_worker_binding_stops_before_graph_build_or_api_calls(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            folder = Path(tmp)
            cfg = json.loads((ROOT / 'benchmarks/checkpoint_smoke.json').read_text())
            worker.save(folder / 'config.json', cfg)
            fake = types.ModuleType('taso')
            with patch.dict(sys.modules, {'taso': fake}), patch.object(worker, 'environment', return_value={}), patch.object(
                    worker, 'inspect_binding', return_value={'status': 'failed', 'errors': ['Stale installed extension']}), patch.object(
                    worker, 'build_network') as build:
                code = worker.main(['--stage', 'prepare', '--network', 'squeezenet', '--out', str(folder),
                    '--repo', str(ROOT / 'TASO'), '--config', str(folder / 'config.json')])
                build.assert_not_called()
            self.assertEqual(code, 1)
            report = json.loads((folder / 'worker_result.json').read_text())
            self.assertIn('Installed TASO binding verification failed', report['error'])
            self.assertEqual(json.loads((folder / 'bindings_check.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()

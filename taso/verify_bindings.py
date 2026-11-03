"""Verify the INSTALLED TASO extension against this checkout; no GPU search/API calls."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'llm_context'))
from prepare_context import enum_names


def inspect_binding(repo, expected_prefix=None):
    repo = Path(repo).resolve()
    prefix = Path(expected_prefix or sys.prefix).resolve()
    pyx = repo / 'python/taso/_cython/core.pyx'
    source = pyx.read_text()
    enum = enum_names((repo / 'include/taso/ops.h').read_text(), 'OpType')
    values = {name: value for value, name in enum.items()}
    block = source[source.index('op_table = dict()'):source.index('cdef class PyGraph:')]
    expected = {}
    for node in ast.parse(block).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == 'op_table' and isinstance(target.slice, ast.Name)):
                expected[values[target.slice.id]] = node.value.value
    errors = []
    missing = [name for value, name in enum.items() if value not in expected]
    if missing:
        errors.append('Checkout source is missing operator mappings: ' + ', '.join(missing) + '. Pull the operator fix first.')
    report = {'status': 'failed', 'python': sys.executable, 'python_prefix': str(prefix),
              'source_pyx': str(pyx), 'source_pyx_sha256': hashlib.sha256(pyx.read_bytes()).hexdigest(),
              'declared_operator_count': len(enum), 'source_mapping_count': len(expected),
              'errors': errors}
    try:
        taso = importlib.import_module('taso')
        core = importlib.import_module('taso.core')
        path = Path(core.__file__).resolve()
        table = getattr(core, 'op_table', {})
        mismatches = [{'value': value, 'symbol': enum[value], 'expected': name,
                       'installed': table.get(value)} for value, name in expected.items()
                      if table.get(value) != name]
        report.update(taso_package=getattr(taso, '__file__', None), compiled_binding=str(path),
                      compiled_binding_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
                      installed_mapping_count=len(table), type_19=table.get(19), type_20=table.get(20),
                      type_21=table.get(21), mismatches=mismatches,
                      checkpoint_api=callable(getattr(taso, 'optimize_with_checkpoints', None)))
        report['gpu_feedback_api'] = callable(getattr(getattr(core, 'PyGraph', None), 'run_time_for_feedback', None))
        if mismatches:
            errors.append('Installed binding differs from checkout source: force-regenerate, rebuild and reinstall TASO.')
        if not path.is_relative_to(prefix):
            errors.append('TASO extension is being loaded outside the selected Python environment.')
        if not report['checkpoint_api']:
            errors.append('Installed TASO package is missing optimize_with_checkpoints; its Python files are stale.')
    except Exception as exc:
        errors.append(f'Cannot import installed TASO: {type(exc).__name__}: {exc}')
    report['status'] = 'passed' if not errors else 'failed'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=HERE / 'TASO')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    report = inspect_binding(args.repo)
    rendered = json.dumps(report, indent=2) + '\n'
    print(rendered, flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

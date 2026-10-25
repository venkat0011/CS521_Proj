#!/usr/bin/env python3
"""Create an isolated, versioned Python environment for TASO or Tensat.

Run: python3 setup_env.py taso
Python environments are recreated on each machine, not copied between hosts.
Tensat's environment is a Python scaffold; its native dependencies are pending.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def create_environment(project, run=None, env=None):
    if project not in ('taso', 'tensat'):
        raise ValueError('Choose taso or tensat')
    env = dict(os.environ) if env is None else env
    if run is None:
        def run(command):
            print('+', ' '.join(map(str, command)), flush=True)
            subprocess.run(list(map(str, command)), env=env, check=True)
    folder = ROOT / project
    version = (folder / '.python-version').read_text().strip()
    target = folder / '.venv'
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if target.exists():
        if not python.exists():
            raise RuntimeError(f'Incomplete environment at {target}; rename it before retrying.')
        installed = subprocess.check_output([str(python), '-c', 'import platform; print(platform.python_version())'], text=True).strip()
        if installed != version:
            raise RuntimeError(f'{target} uses Python {installed}; expected {version}. Rename the old environment before retrying.')
        print(f'Reusing {target} (Python {version})', flush=True)
        return python
    uv = shutil.which('uv', path=env.get('PATH'))
    if uv:
        command = [uv]
    else:
        # Bootstrap uv locally, without changing Colab's global packages.
        tools = ROOT / '.tools' / 'uv'
        run([sys.executable, '-m', 'pip', 'install', '--target', tools, 'uv>=0.9,<1'])
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(tools), env.get('PYTHONPATH')]))
        command = [sys.executable, '-m', 'uv']
    run(command + ['python', 'install', version])
    run(command + ['venv', '--managed-python', '--python', version, '--seed', target])
    actual = subprocess.check_output([str(python), '-c', 'import platform; print(platform.python_version())'], text=True).strip()
    if actual != version:
        raise RuntimeError(f'Environment has Python {actual}, expected {version}')
    print(f'Created {target} (Python {actual})', flush=True)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', choices=['taso', 'tensat'])
    args = parser.parse_args()
    python = create_environment(args.project)
    print(f'Python executable: {python}')
    print(f'Activate in a Linux/Colab terminal: source {python.parent / "activate"}')
    if args.project == 'tensat':
        print('Python scaffold only; Tensat itself has not been cloned or installed.')


if __name__ == '__main__':
    main()

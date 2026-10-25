#!/usr/bin/env python3
"""Build vanilla TASO and run its tiny baseline from a Linux GPU terminal.

Usage: python3 taso/baseline/setup_and_run.py
The first CUDA-12 attempt uses a private venv with cuDNN 8.9.7.29. This
legacy build has not yet been verified in Colab; failures are saved by stage.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import traceback


class Runner:
    def __init__(self, results):
        self.results = results
        self.stage = 'preflight'
        self.env = dict(os.environ)
        self.log = (results / 'setup.log').open('w', buffering=1)

    def write(self, text):
        print(text, flush=True)
        self.log.write(text + '\n')

    def run(self, command, cwd=None):
        command = [str(c) for c in command]
        self.write(f'\n[{self.stage}] {shlex.join(command)}')
        with subprocess.Popen(command, cwd=cwd, env=self.env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                self.write(line.rstrip('\n'))
            code = process.wait()
        if code:
            raise RuntimeError(f'{self.stage}: command exited with code {code}')

    def build(self, root, args):
        if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
            raise RuntimeError('Use a Linux x86_64 NVIDIA GPU runtime, such as Colab. This setup does not support macOS.')
        if not shutil.which('nvidia-smi'):
            raise RuntimeError('nvidia-smi is missing. Select/connect a GPU runtime.')
        self.run(['nvidia-smi'])
        nvcc = shutil.which('nvcc')
        if not nvcc:
            raise RuntimeError('CUDA development toolkit missing: nvcc not found.')
        version = subprocess.check_output([nvcc, '--version'], text=True)
        self.write(version)
        if not re.search(r'release\s+12\.', version):
            raise RuntimeError('This initial dependency set targets CUDA 12. Share the nvcc output to adapt other versions.')
        cuda = Path(nvcc).resolve().parents[1]
        source = root / 'taso' / 'TASO'
        if not (source / 'graph_subst.pb').is_file():
            raise RuntimeError(f'TASO source or rewrite file missing: {source}')
        self.stage = 'system_dependencies'
        required = ['g++', 'make', 'protoc']
        if not all(shutil.which(tool) for tool in required) or not Path('/usr/include/google/protobuf/message.h').exists():
            if not shutil.which('apt-get'):
                raise RuntimeError('Need g++, make, protoc and protobuf development headers.')
            prefix = [] if os.geteuid() == 0 else ['sudo']
            self.run(prefix + ['apt-get', 'update', '-qq'])
            self.run(prefix + ['apt-get', 'install', '-y', 'build-essential', 'protobuf-compiler',
                               'libprotobuf-dev', 'python3-venv', 'python3-dev'])
        self.stage = 'python_environment'
        sys.path.insert(0, str(root))
        from setup_env import create_environment
        python = create_environment('taso', run=self.run, env=self.env)
        environment = python.parent.parent
        self.env['PATH'] = str(environment / 'bin') + os.pathsep + self.env.get('PATH', '')
        self.run([python, '-m', 'pip', 'install', '-r', root / 'taso' / 'requirements-build.txt'])
        self.run([python, '-c', 'import sys; print("TASO Python:", sys.version)'])
        # Keep the notebook's/global Python environment untouched. CUDA toolkit
        # libraries already present on the host supply cuBLAS dependencies.
        self.run([python, '-m', 'pip', 'install', '--no-deps', 'nvidia-cudnn-cu12==8.9.7.29'])
        with (self.results / 'python-packages.txt').open('w') as packages:
            subprocess.run([str(python), '-m', 'pip', 'freeze', '--all'], env=self.env, stdout=packages, check=True)
        site_packages = Path(subprocess.check_output(
            [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
        cudnn = site_packages / 'nvidia' / 'cudnn'
        header = cudnn / 'include' / 'cudnn.h'
        library = cudnn / 'lib' / 'libcudnn.so.8'
        if not header.exists() or not library.exists():
            raise RuntimeError('Installed cuDNN package lacks expected development headers/libraries.')
        native = environment / 'native'
        self.env['TASO_HOME'] = str(source)
        for key, paths in {
            'CPATH': [cudnn / 'include', cuda / 'include'],
            'LIBRARY_PATH': [native / 'lib', cudnn / 'lib', cuda / 'lib64'],
            'LD_LIBRARY_PATH': [native / 'lib', cudnn / 'lib', cuda / 'lib64'],
        }.items():
            self.env[key] = os.pathsep.join([str(p) for p in paths] + [self.env.get(key, '')])
        self.stage = 'compile_runtime'
        build = root / 'taso' / 'baseline' / 'build'
        command = [environment / 'bin' / 'cmake', '-S', source, '-B', build,
                   '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_INSTALL_PREFIX={native}',
                   f'-DUSE_CUDA={cuda}', '-DUSE_DNNL=OFF', '-DBUILD_CPP_EXAMPLES=OFF',
                   f'-DCUDA_CUDNN_LIBRARY={library}', f'-DCMAKE_INCLUDE_PATH={cudnn / "include"}',
                   f'-DCMAKE_CUDA_COMPILER={nvcc}']
        self.run(command)
        self.run([environment / 'bin' / 'cmake', '--build', build, '-j', args.jobs])
        self.run([environment / 'bin' / 'cmake', '--install', build])
        self.stage = 'install_python_bindings'
        self.run([python, 'setup.py', 'install'], cwd=source / 'python')
        self.stage = 'import_check'
        self.run([python, '-c', 'import taso; print("TASO package:", taso.__file__); assert callable(taso.new_graph)'], cwd=root / 'taso' / 'baseline')
        self.stage = 'baseline'
        self.run([python, root / 'taso' / 'baseline' / 'run_baseline.py',
                  '--repo', source, '--out', self.results], cwd=root / 'taso' / 'baseline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Results directory; defaults to baseline/results')
    parser.add_argument('--jobs', default='2', choices=['1', '2', '4'], help='Parallel compilation jobs')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    results = (args.out or root / 'taso' / 'baseline' / 'results').resolve()
    results.mkdir(parents=True, exist_ok=True)
    # Avoid reporting a previous successful run if the current setup fails early.
    (results / 'baseline.json').unlink(missing_ok=True)
    runner = Runner(results)
    report = {'status': 'started', 'platform': platform.platform(), 'python': sys.version,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        runner.build(root, args)
        report.update(status='completed', stage=runner.stage)
    except Exception as exc:
        report.update(status='failed', stage=runner.stage, error=f'{type(exc).__name__}: {exc}')
        runner.write(traceback.format_exc())
    finally:
        (results / 'setup_status.json').write_text(json.dumps(report, indent=2) + '\n')
        runner.write('\n' + json.dumps(report, indent=2))
        runner.write(f'\nShare setup_status.json and the last lines of setup.log. Results: {results}')
        runner.log.close()
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

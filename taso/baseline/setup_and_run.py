#!/usr/bin/env python3
"""Build TASO and run a baseline, strategy comparison or three-network suite.

Usage: python3 taso/baseline/setup_and_run.py
Uses a private venv with CUDA 12 and cuDNN 8.9.7.29. Failures are saved by stage.
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
import tempfile
import urllib.request


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

    def find_cuda12(self):
        candidates = []
        if self.env.get('CUDA_HOME'):
            candidates.append(Path(self.env['CUDA_HOME']) / 'bin' / 'nvcc')
        candidates.append(Path('/usr/local/cuda-12.6/bin/nvcc'))
        current = shutil.which('nvcc', path=self.env.get('PATH'))
        if current:
            candidates.append(Path(current))
        candidates.extend(sorted(Path('/usr/local').glob('cuda-12.*/bin/nvcc'), reverse=True))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            version = subprocess.check_output([str(candidate), '--version'], text=True, env=self.env)
            if re.search(r'release\s+12\.', version):
                self.write(version)
                return candidate.resolve()
        return None

    def configure_cuda(self):
        self.stage = 'cuda_toolkit'
        nvcc = self.find_cuda12()
        if nvcc is None:
            release = {}
            for line in Path('/etc/os-release').read_text().splitlines():
                if '=' in line:
                    key, value = line.split('=', 1)
                    release[key] = value.strip('"')
            if release.get('ID') != 'ubuntu' or release.get('VERSION_ID') not in ('22.04', '24.04'):
                raise RuntimeError('Automatic CUDA 12.6 installation supports Ubuntu 22.04/24.04. Share setup.log for other environments.')
            if not shutil.which('apt-get'):
                raise RuntimeError('apt-get is required for automatic CUDA installation.')
            admin = [] if os.geteuid() == 0 else ['sudo']
            if admin and not shutil.which('sudo'):
                raise RuntimeError('CUDA installation requires root or sudo access.')
            distro = 'ubuntu' + release['VERSION_ID'].replace('.', '')
            url = f'https://developer.download.nvidia.com/compute/cuda/repos/{distro}/x86_64/cuda-keyring_1.1-1_all.deb'
            self.write('No CUDA 12 toolkit found; installing CUDA 12.6 alongside existing toolkits. GPU driver is retained.')
            with tempfile.TemporaryDirectory(dir=self.results) as temporary:
                keyring = Path(temporary) / 'cuda-keyring.deb'
                urllib.request.urlretrieve(url, keyring)
                self.run(admin + ['dpkg', '-i', str(keyring)])
            self.run(admin + ['apt-get', 'update', '-qq'])
            self.run(admin + ['apt-get', 'install', '-y', '--no-install-recommends', 'cuda-toolkit-12-6'])
            nvcc = self.find_cuda12()
            if nvcc is None:
                raise RuntimeError('CUDA installation finished but a CUDA 12 nvcc was not found.')
        cuda = nvcc.parents[1]
        self.env['CUDA_HOME'] = str(cuda)
        self.env['CUDACXX'] = str(nvcc)
        self.env['PATH'] = str(cuda / 'bin') + os.pathsep + self.env.get('PATH', '')
        self.env['LD_LIBRARY_PATH'] = str(cuda / 'lib64') + os.pathsep + self.env.get('LD_LIBRARY_PATH', '')
        self.write(f'Selected CUDA toolkit: {cuda}')
        return cuda, nvcc

    def build(self, root, args):
        if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
            raise RuntimeError('Use a Linux x86_64 NVIDIA GPU runtime, such as Colab. This setup does not support macOS.')
        if not shutil.which('nvidia-smi'):
            raise RuntimeError('nvidia-smi is missing. Select/connect a GPU runtime.')
        self.run(['nvidia-smi'])
        cuda, nvcc = self.configure_cuda()
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
        if args.experiment == 'suite':
            self.run([python, '-m', 'pip', 'install', '-r', root / 'taso/benchmarks/requirements.txt'])
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
        self.stage = args.experiment
        command = experiment_command(root, args, python, self.results)
        self.run(command, cwd=root / 'taso' / 'baseline')


def experiment_command(root, args, python, results):
    scripts = {'baseline': 'baseline/run_baseline.py', 'manual': 'baseline/run_manual_strategy.py',
               'llm': 'baseline/run_llm_strategy.py', 'suite': 'benchmarks/run_suite.py'}
    command = [python, root / 'taso' / scripts[args.experiment],
               '--repo', root / 'taso/TASO', '--out', results]
    if args.experiment in ('llm', 'suite'):
        if args.max_tokens is not None or args.experiment == 'llm':
            command.extend(['--max-tokens', str(args.max_tokens if args.max_tokens is not None else 8192)])
        if args.model:
            command.extend(['--model', args.model])
        if args.dry_run:
            command.append('--dry-run')
    if args.strategy_file:
        command.extend(['--strategy-file', args.strategy_file.resolve()])
    if args.experiment == 'suite':
        if args.suite_config:
            command.extend(['--config', args.suite_config.resolve()])
        if args.batch_dir:
            command.extend(['--batch-dir', args.batch_dir.resolve()])
        if args.retry_evaluation:
            command.append('--retry-evaluation')
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['baseline', 'manual', 'llm', 'suite'], default='baseline')
    parser.add_argument('--dry-run', action='store_true', help='LLM/suite: prepare prompts without calling provider')
    parser.add_argument('--model', help='LLM experiment: OpenRouter model ID')
    parser.add_argument('--max-tokens', type=int, help='Output token limit; llm default 8192, suite default 32768')
    parser.add_argument('--strategy-file', type=Path, help='LLM experiment: reuse a saved strategy without an API call')
    parser.add_argument('--out', type=Path, help='Results directory; defaults to baseline/results')
    parser.add_argument('--jobs', default='2', choices=['1', '2', '4'], help='Parallel compilation jobs')
    parser.add_argument('--suite-config', type=Path, help='Suite experiment configuration JSON')
    parser.add_argument('--batch-dir', type=Path, help='Suite results folder; reuse to resume the same batch')
    parser.add_argument('--retry-evaluation', action='store_true', help='Suite: retry GPU evaluation using saved strategies')
    args = parser.parse_args(argv)
    if args.experiment not in ('llm', 'suite') and (args.dry_run or args.model or args.max_tokens):
        parser.error('--dry-run, --model and --max-tokens require --experiment llm or suite')
    if args.strategy_file and args.experiment != 'llm':
        parser.error('--strategy-file requires --experiment llm')
    if args.experiment != 'suite' and (args.suite_config or args.batch_dir or args.retry_evaluation):
        parser.error('--suite-config, --batch-dir and --retry-evaluation require --experiment suite')
    return args


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    results = (args.out or root / 'taso' / 'baseline' / 'results').resolve()
    results.mkdir(parents=True, exist_ok=True)
    # Avoid reporting a previous successful run if the current setup fails early.
    for filename in ('baseline.json', 'manual_summary.json'):
        (results / filename).unlink(missing_ok=True)
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

#!/usr/bin/env python3
"""Build Tensat against this project's TASO runtime and run one optimisation.

Usage: python3 tensat/setup_and_run.py                  (SqueezeNet, greedy extraction)
       python3 tensat/setup_and_run.py --extract ilp
Needs a Linux x86_64 NVIDIA GPU runtime, such as Colab. Failures are saved by stage.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'taso'))
from run_experiment import Runner
from setup_env import create_environment

ROOT = Path(__file__).resolve().parents[1]
TENSAT = ROOT / 'tensat' / 'tensat'
TOOLS = ROOT / '.tools'
MODELS = ('squeezenet', 'resnext50', 'inceptionv3')
RUSTUP_URL = 'https://sh.rustup.rs'
# bindgen 0.54 predates newer libclang releases; 14 is the Ubuntu 22.04 default.
LLVM = Path('/usr/lib/llvm-14')
CLANG_PACKAGES = ['clang-14', 'libclang-14-dev']
# E-graph limits used by upstream run_exp_main.sh.
NODE_LIMIT = 50000
ITER_MULTI = 1


class NativeBuildDone(Exception):
    """Leaves Runner.build once libtaso_runtime is installed."""


class TensatRunner(Runner):
    """TASO's Runner, stopped after the native build.

    taso/run_experiment.py is hashed by the TASO suite, so it is reused unchanged.
    Runner.build installs libtaso_runtime and then calls install_bindings; that
    call is the first step Tensat does not need, so it is where the build stops.
    """

    def install_bindings(self, root, python, source):
        self.native = python.parent.parent / 'native'
        raise NativeBuildDone

    def build_taso_runtime(self, root, jobs):
        try:
            self.build(root, argparse.Namespace(experiment='baseline', jobs=jobs))
        except NativeBuildDone:
            return self.native
        raise RuntimeError('TASO build did not reach the binding install step.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=MODELS, default='squeezenet')
    parser.add_argument('--extract', choices=['greedy', 'ilp'], default='greedy', help='Extraction method')
    parser.add_argument('--n-iter', type=int, default=3, help='Maximum equality-saturation iterations')
    parser.add_argument('--n-sec', type=int, default=600, help='Time limit for equality saturation, in seconds')
    parser.add_argument('--jobs', default='2', choices=['1', '2', '4'], help='Parallel TASO compilation jobs')
    parser.add_argument('--out', type=Path, help='Output folder; defaults to tensat/results')
    return parser.parse_args(argv)


def tensat_command(binary, args, stats_file):
    """Flags follow upstream greedy_vs_ilp.sh (greedy) and run_exp_main.sh (ILP).

    Rule files are relative paths, so the command must run inside tensat/tensat.
    """
    command = [binary, '-r', 'converted.txt', '-t', 'converted_multi.txt', '-u', '-s', 'none',
               '--n_iter', str(args.n_iter), '--iter_multi', str(ITER_MULTI),
               '--n_sec', str(args.n_sec), '--n_nodes', str(NODE_LIMIT), '--all_weight_only',
               '-e', args.extract, '-d', args.model, '-o', stats_file]
    if args.extract == 'ilp':
        command += ['--no_order', '--no_cycle', '--ilp_time_sec', '0']
    return command


def install_clang(runner):
    """libclang and the clang binary are needed by bindgen to read the TASO header."""
    runner.stage = 'clang'
    if (LLVM / 'bin' / 'clang').exists() and (LLVM / 'lib' / 'libclang.so').exists():
        return
    admin = [] if os.geteuid() == 0 else ['sudo']
    runner.run(admin + ['apt-get', 'update', '-qq'])
    runner.run(admin + ['apt-get', 'install', '-y', '--no-install-recommends'] + CLANG_PACKAGES)


def install_rust(runner):
    """Private rustup under .tools, at the version in tensat/rust-toolchain."""
    runner.stage = 'rust_toolchain'
    cargo_home = TOOLS / 'cargo'
    runner.env['RUSTUP_HOME'] = str(TOOLS / 'rustup')
    runner.env['CARGO_HOME'] = str(cargo_home)
    runner.env['PATH'] = str(cargo_home / 'bin') + os.pathsep + runner.env.get('PATH', '')
    version = (ROOT / 'tensat' / 'rust-toolchain').read_text().strip()
    rustup = cargo_home / 'bin' / 'rustup'
    if not rustup.exists():
        with tempfile.TemporaryDirectory(dir=runner.results) as temporary:
            installer = Path(temporary) / 'rustup-init.sh'
            urllib.request.urlretrieve(RUSTUP_URL, installer)
            runner.run(['sh', installer, '-y', '--no-modify-path', '--profile', 'minimal',
                        '--default-toolchain', version])
    runner.run([rustup, 'toolchain', 'install', version, '--profile', 'minimal'])
    cargo = cargo_home / 'bin' / 'cargo'
    runner.run([cargo, '--version'])
    return cargo


def install_extractor_python(runner):
    """Tensat runs `python extractor/extract.py`, so this venv must come first on PATH."""
    runner.stage = 'extractor_python'
    python = create_environment('tensat', run=runner.run, env=runner.env)
    runner.run([python, '-m', 'pip', 'install', '-r', ROOT / 'tensat' / 'requirements.txt'])
    runner.run([python, '-m', 'pip', 'freeze'])
    runner.env['PATH'] = str(python.parent) + os.pathsep + runner.env.get('PATH', '')


def build_tensat(runner, cargo, native):
    runner.stage = 'build_tensat'
    runner.env.update(TASO_LIB_DIR=str(native / 'lib'), TASO_INCLUDE_DIR=str(native / 'include'),
                      LIBCLANG_PATH=str(LLVM / 'lib'), CLANG_PATH=str(LLVM / 'bin' / 'clang'))
    # --locked makes cargo fail instead of rewriting Cargo.lock, so every build uses the
    # committed crate versions.
    runner.run([cargo, 'build', '--release', '--locked'], cwd=TENSAT)
    return TENSAT / 'target' / 'release' / 'tensat'


def run_tensat(runner, binary, args):
    runner.stage = 'run_tensat'
    stats_file = runner.results / f'{args.model}_{args.extract}_stats.txt'
    # Tensat appends to the stats file and reads tmp/solved.json after the ILP
    # solver. Remove both first, so a failed run cannot be read as a result.
    stats_file.unlink(missing_ok=True)
    (TENSAT / 'tmp' / 'solved.json').unlink(missing_ok=True)
    # No terminal input: on an error, extract.py opens a debugger and would wait forever.
    without_input = ['sh', '-c', 'exec "$@" < /dev/null', 'tensat']
    runner.run(without_input + tensat_command(binary, args, stats_file), cwd=TENSAT)
    result = json.loads(stats_file.read_text().splitlines()[-1])
    runner.write('Tensat result: ' + json.dumps(result))
    return result


def setup_and_run(runner, args):
    native = runner.build_taso_runtime(ROOT, args.jobs)
    install_clang(runner)
    cargo = install_rust(runner)
    install_extractor_python(runner)
    binary = build_tensat(runner, cargo, native)
    return run_tensat(runner, binary, args)


def main():
    args = parse_args()
    results = (args.out or ROOT / 'tensat' / 'results').resolve()
    results.mkdir(parents=True, exist_ok=True)
    runner = TensatRunner(results)
    report = {'status': 'started', 'platform': platform.platform(), 'python': sys.version,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'model': args.model, 'extract': args.extract}
    try:
        report['result'] = setup_and_run(runner, args)
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

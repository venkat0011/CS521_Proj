#!/usr/bin/env python3
"""Set up and run the secondary experiment's zero-API GPU timing smoke test.

Colab: python3 taso/secondary_experiment/run_gpu_smoke.py --out /path/to/results
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_experiment import Runner


class SecondaryRunner(Runner):
    """Reuse installation/build steps; dispatch only this separate experiment."""
    def experiment_command(self, root, args, python, results):
        self.stage = 'secondary_gpu_timing_smoke'
        command = [str(python), str(root / 'taso/secondary_experiment/gpu_timing_smoke.py'),
                   '--repo', str(root / 'taso/TASO'), '--out', str(results),
                   '--network', args.network, '--budget', str(args.budget),
                   '--blocks', str(args.blocks)]
        for trial in args.replay_trial:
            command.extend(['--replay-trial', str(trial.resolve())])
        return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Results directory; reuse unchanged to resume')
    parser.add_argument('--network', choices=['squeezenet', 'resnext50', 'inceptionv3'], default='squeezenet')
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--blocks', type=int, default=20)
    parser.add_argument('--replay-trial', type=Path, action='append', default=[],
                        help='Saved Kimi CP25 trial directory; repeat for multiple schedules; zero API calls')
    parser.add_argument('--jobs', choices=['1', '2', '4'], default='2', help='Compilation jobs')
    args = parser.parse_args(argv)
    if args.budget < 1 or not 4 <= args.blocks <= 200:
        parser.error('A positive budget and 4..200 timing blocks are required')
    # Select baseline build dependencies only; dispatch uses SecondaryRunner above.
    args.experiment = 'baseline'
    return args


def default_results(root, args):
    return (args.out if args.out is not None else
            root / 'taso/results/secondary_experiment/gpu_timing_smoke').resolve()


def main(argv=None):
    args = parse_args(argv)
    root = HERE.parents[1]
    results = default_results(root, args)
    results.mkdir(parents=True, exist_ok=True)
    runner = SecondaryRunner(results)
    report = dict(status='started', experiment='secondary_gpu_timing_smoke',
                  platform=platform.platform(), python=sys.version, model_calls=0,
                  started_utc=datetime.now(timezone.utc).isoformat())
    try:
        runner.build(root, args)
        report.update(status='completed', stage=runner.stage)
    except Exception as exc:
        report.update(status='failed', stage=runner.stage, error=f'{type(exc).__name__}: {exc}')
        runner.write(traceback.format_exc())
    finally:
        (results / 'setup_status.json').write_text(json.dumps(report, indent=2) + '\n')
        runner.write('\n' + json.dumps(report, indent=2))
        runner.write(f'Results: {results}')
        runner.log.close()
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

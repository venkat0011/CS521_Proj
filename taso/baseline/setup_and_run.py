"""Compatibility entry point. Canonical implementation: run_experiment.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import Runner, experiment_command, parse_args, default_results, main


if __name__ == '__main__':
    raise SystemExit(main())

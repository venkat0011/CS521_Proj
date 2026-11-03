"""Compatibility entry point. Canonical implementation: experiments/run_manual_strategy.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_manual_strategy import small_cnn, observe, aggregate, parse_args, main


if __name__ == '__main__':
    raise SystemExit(main())

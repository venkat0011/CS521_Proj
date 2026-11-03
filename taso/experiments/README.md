# Strategy smoke experiments

- `run_manual_strategy.py`: hand-written rule schedule on a small branched CNN.
- `run_llm_strategy.py`: one-shot LLM generation and evaluation on that small CNN.

The shared launcher is `../run_experiment.py`. Full three-network/five-trial
one-shot and checkpoint studies live in `../benchmarks/`. Each paired experiment
includes an all-rules TASO `baseline` control; the LLM methods are experimental
policies. New outputs are written under `taso/results/`.

The old files in `../baseline/` are compatibility aliases. Existing results are
left in place; explicit output/batch paths are honored.

# CS521 Project

Investigating LLM-guided tensor graph search using TASO and, later, Tensat.

The [three-network, multi-model experiment](taso/benchmarks/README.md) compares TASO and LLM
schedules with 100 graph expansions and five independent LLM trials per network.
It saves every trial, full search traces, distribution statistics and dot/box plots.

## Layout

- `taso/TASO/`: vendored upstream TASO source, tracked as regular files.
- `taso/llm_context/`: one-time rule/source preparation and runtime graph/LLM interfaces.
- `taso/run_experiment.py`: shared build/setup and experiment launcher.
- `taso/baseline/`: vanilla TASO smoke test and compatibility entry points.
- `taso/experiments/`: manual-strategy and one-shot small-CNN smoke scripts.
- `taso/benchmarks/`: paired one-shot/checkpoint experiment runners and configurations.
- `taso/results/`: new experiment outputs, grouped by mode.
- A separate Tensat folder will be added later.

## Start with vanilla TASO

In a Colab GPU terminal:

```sh
git clone https://github.com/venkat0011/CS521_Proj.git
cd CS521_Proj
python3 taso/baseline/setup_and_run.py
```

No notebook file is required. If already cloned, run `git pull --ff-only`
inside the checkout before the Python command.
See `taso/baseline/README.md` for supported environment versions and result paths.
The script selects or installs CUDA 12.6 when needed, then creates the TASO
Python environment and builds/runs TASO. No separate CUDA setup command is needed.
The GPU build has not yet been verified; failures are saved with their stage
and full log. No successful baseline performance results have been obtained yet.

After installation, the smoke test constructs Input -> Conv -> ReLU and records
search output, graph costs, operator counts and execution timings. Internal
profiling and LLM-controlled search are subsequent steps.

## Upstream attribution

TASO: https://github.com/jiazhihao/TASO

Vendored from commit `f11782c4db180ad7d1e7ff7f4bb82e574ac3b6b8`.
The original Apache 2.0 license is retained at `taso/TASO/LICENSE`.
The upstream Git metadata is not part of this repository.

## Separate Python environments

`python3 setup_env.py taso` creates `taso/.venv`; `python3 setup_env.py tensat`
creates `tensat/.venv`. Each folder has its own `.python-version`. Both start
with Python 3.11.17; their versions can change independently. uv downloads
the interpreter instead of relying on Colab's default version.

The TASO runner creates its environment automatically. Tensat currently has
only a Python scaffold, not an installed optimizer. Recreate environments on
each machine rather than copying them. CUDA/compilers remain host dependencies.

## First manually scheduled search

```sh
git pull --ff-only
python3 taso/run_experiment.py --experiment manual
```

This rebuilds TASO, then compares all-rule controls with the hand-written
`taso/strategies/manual_conv.json` policy on a small branched CNN. It runs no
LLM calls. See `taso/strategies/README.md` for execution semantics and limits.
The runtime now keeps assertions enabled: upstream uses them for required
parsing and matching operations, so prior Release smoke results are not a
valid search-quality reference.

## First LLM-generated strategy

```sh
python3 taso/run_experiment.py --experiment llm --dry-run
```

This saves the actual prompt without an API call. With `OPENROUTER_API_KEY`
set in the terminal, omit `--dry-run` to generate one strategy using DeepSeek
V4 Pro through OpenRouter and compare it with all-rule search. See
`taso/llm_context/README.md` for key entry, saved outputs and strategy reuse.

Online feedback: [configurable expansion checkpoints](taso/benchmarks/CHECKPOINTS.md)
(`--checkpoint-interval 25`, or 10/5). Omitting the option retains one-shot mode.

## Experiment naming and outputs

A baseline is the all-rules TASO control. One-shot and checkpoint runs are LLM
experiments, each with its own baseline comparison. New commands use the neutral
`taso/run_experiment.py` entry point:

```sh
python3 taso/run_experiment.py --experiment suite \
  --suite-config taso/benchmarks/checkpoint_smoke.json --checkpoint-interval 25 \
  --batch-dir /content/CS521_Proj/taso/results/checkpoint/checkpoint25_smoke_v2 \
  --dry-run
```

New automatic batches go under `taso/results/one_shot/<run>/` or
`taso/results/checkpoint/<run>/`. A supplied `--batch-dir` is used exactly as
provided. Vanilla smoke output remains in `taso/baseline/results/`; manual smoke
output goes to `taso/results/manual/`, and the small-CNN one-shot script writes
`taso/results/one_shot_smoke/<run>/`.

Old `taso/baseline/setup_and_run.py` commands remain compatibility aliases.
Existing results are not moved or renamed. Use a new batch after this source
layout change; old saved batches can still be summarized using their original
`--batch-dir` with `run_suite.py --summarize`.

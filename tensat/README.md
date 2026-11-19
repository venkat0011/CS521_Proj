# Tensat on this project's TASO runtime

Vanilla [Tensat](https://github.com/uwplse/tensat) (equality saturation in place of
TASO's backtracking search), built against the same `taso/TASO` runtime the TASO
experiments use. There is no checkpoint or LLM control here yet.

## Run

In a Colab GPU terminal, from the project root:

```sh
python3 tensat/setup_and_run.py                  # SqueezeNet, greedy extraction
python3 tensat/setup_and_run.py --extract ilp    # SqueezeNet, ILP extraction
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `squeezenet` | `squeezenet`, `resnext50` or `inceptionv3` |
| `--extract` | `greedy` | `greedy` or `ilp` |
| `--n-iter` | `3` | Maximum equality-saturation iterations |
| `--n-sec` | `600` | Time limit for equality saturation, in seconds |
| `--jobs` | `2` | Parallel TASO compilation jobs |
| `--out` | `tensat/results` | Output folder |

The defaults are a smoke test. Upstream's experiments use 15 iterations and a
3600-second limit (`tensat/tensat/run_exp_main.sh`).

## What the script does

Each stage name appears in `setup_status.json` if that stage fails.

| Stage | Action |
|---|---|
| `cuda_toolkit` to `compile_runtime` | Reuses `taso/run_experiment.py` to select or install CUDA 12.6, install cuDNN and build `libtaso_runtime` into `taso/.venv/native` |
| `clang` | Installs clang 14 and libclang 14 with apt, if missing |
| `rust_toolchain` | Installs rustup under `.tools/` and the Rust version in `tensat/rust-toolchain` |
| `extractor_python` | Creates `tensat/.venv` and installs `tensat/requirements.txt` (ILP solver) |
| `build_tensat` | `cargo build --release --locked` in `tensat/tensat` |
| `run_tensat` | Runs one optimisation and copies its result line into `setup_status.json` |

`taso/run_experiment.py` is not edited, because the TASO suite hashes it for each
batch. The script reuses its `Runner` through a subclass that stops after the
native build.

## Output

All files are under `tensat/results/` (not tracked by git):

- `setup_status.json`: status, failing stage if any, and the result.
- `setup.log`: full output of every command.
- `<model>_<extract>_stats.txt`: the JSON line Tensat writes.

Result fields: `original` and `optimized` are measured graph runtimes in
milliseconds; `saturation` and `extraction` are seconds; `nodes`, `classes` and
`iter` describe the e-graph.

## Verified so far

Environment of both runs (2026-11-18): Colab, Tesla T4, Ubuntu 24.04, CUDA
toolkit 12.6, cuDNN 8.9.7.29, Rust 1.70.0, clang 14, ortools 9.8.3296.

| Model | Extraction | Iterations (limit 3) | E-graph nodes / classes | Original (ms) | Optimized (ms) | Status |
|---|---|---|---|---|---|---|
| SqueezeNet | greedy | 3, limit reached | 1072 / 582 | 1.0998 | 1.0298 | Completed, one run |
| SqueezeNet | ILP | 2, saturated | 288 / 199 | 1.0786 | 0.8450 | Completed, one run; solver reported an optimal solution in 0.22 s |
| ResNeXt-50 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Inception-v3 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

These are smoke-test numbers from single runs, not experiment results. The two
SqueezeNet rows are not a like-for-like comparison of extraction methods: the
ILP command also passes `--no_cycle`, as upstream does, which changes the
e-graph that is built.

After both builds, `git status` showed `tensat/tensat/Cargo.lock` unchanged, so
cargo used the pinned crate versions as they are.

## Source and changes

| Folder | Upstream | Commit | License |
|---|---|---|---|
| `tensat/tensat/` | https://github.com/uwplse/tensat | `4fd1ad4a5082155e6bfb1685037f7d7fa07e5e0a` | MIT, `tensat/tensat/LICENSE` |
| `tensat/egg/` | https://github.com/yycdavid/egg | `12cc1ee7731d37fe91901c81f59678fa1d08a2bb` | MIT, `tensat/egg/LICENSE` |

Both are vendored as regular files; the upstream Git metadata is not included.

Upstream Tensat expects its own TASO fork (https://github.com/yycdavid/taso).
This project links Tensat to `taso/TASO` instead, which is left unchanged. The
changes are all in the vendored Tensat source:

| File | Change | Reason |
|---|---|---|
| `src/model.rs`, `src/optimize.rs` | The two batchnorm calls pass epsilon `-1.0` | Our TASO takes an epsilon; a negative value selects cuDNN's minimum, which the fork hard-codes |
| `src/main.rs` | `-x` / `--export_model` exits at start-up with a message | Our TASO has no `Graph::export_to_file_raw` |
| `build.rs` | Reads `TASO_LIB_DIR` and `TASO_INCLUDE_DIR`; the defaults are upstream's Docker paths | The TASO build installs to `taso/.venv/native` |
| `wrapper.h` | Includes `taso/ops.h` | The old relative path only resolved in upstream's Docker layout |
| `Cargo.lock` | Package name `tamago` changed to `tensat` | Upstream's lock file was stale and cargo would not accept it |

## Differences from the Tensat paper's setup

- **Timing:** the final graph runtime uses this project's TASO protocol (100
  warm-up runs and 100 timed runs). The fork uses a two-minute warm-up and 500
  runs. Numbers are comparable with the TASO experiments here, not with the paper.
- **Model export:** not supported.
- **Scope:** the source comparison between the fork and `taso/TASO` found no
  difference that affects SqueezeNet, ResNeXt-50 or Inception-v3. Element-wise
  and matmul handling does differ for graphs with matmul or transpose, such as
  BERT and NasRNN, so those are outside what was checked.
- **Environment:** no Docker; CUDA 12 and cuDNN 8 in place of CUDA 10 and cuDNN 7.

## Tests

No GPU needed:

```sh
python3 -B -m unittest discover -s tensat/tests -p 'test_*.py'
```

## Not done yet

- Checkpoint hook and LLM rule selection in Tensat's search loop.
- Runs at upstream's experiment settings.

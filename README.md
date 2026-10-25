# CS521 Project

Investigating LLM-guided tensor graph search using TASO and, later, Tensat.

## Layout

- `taso/TASO/`: vendored upstream TASO source, tracked as regular files.
- `taso/llm_context/`: one-time rule/source preparation and runtime graph/LLM interfaces.
- `taso/baseline/`: tiny vanilla TASO test and Python terminal setup script.
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

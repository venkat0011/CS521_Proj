# Vanilla TASO baseline: terminal workflow

Run in the Colab terminal after connecting a GPU runtime:

```sh
git clone https://github.com/venkat0011/CS521_Proj.git
cd CS521_Proj
python3 taso/baseline/setup_and_run.py
```

If the repository is already cloned, enter it and run `git pull --ff-only`
before the Python command. No .ipynb file or file upload is required.

The runner checks Linux/NVIDIA GPU/CUDA, installs missing system build tools,
creates `.venv-taso`, installs the initial dependency set in that isolated
Python environment, builds TASO's C++ library, installs the Python bindings,
and runs `run_baseline.py` on Input -> Conv -> ReLU.

This initial setup targets Linux x86_64, Python 3.9–3.12 and CUDA 12 with
NVIDIA cuDNN 8.9.7.29 (https://pypi.org/project/nvidia-cudnn-cu12/8.9.7.29/).
It does not change TASO's search source or replace the global Python packages.
Other CUDA/Python environments stop with an explanatory diagnostic. The actual
Colab GPU build is not yet verified; legacy compiler/library compatibility may
still require changes. Dependency setup downloads several large packages.

Results are written to `taso/baseline/results/`:

- `setup.log`: streamed dependency/build/search output.
- `setup_status.json`: success or exact failing stage.
- `baseline.json`: graph costs and timings, only if the baseline reaches its
  Python report-writing path. A failed or aborted native process may not create it.

Paste the result JSON or the last error lines into the project chat. No
performance or bottleneck claims have been established yet. The tiny test is
a smoke run, not a numerical equivalence check or controlled speedup study.

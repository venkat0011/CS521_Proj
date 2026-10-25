# Vanilla TASO baseline: terminal workflow

Run in the Colab terminal after connecting a GPU runtime:

```sh
git clone https://github.com/venkat0011/CS521_Proj.git
cd CS521_Proj
python3 taso/baseline/setup_and_run.py
```

If the repository is already cloned, enter it and run `git pull --ff-only`
before the Python command. No .ipynb file or file upload is required.

The runner checks Linux/NVIDIA GPU, selects an existing CUDA 12 toolkit or
installs CUDA 12.6 alongside newer toolkits, installs missing system build tools,
creates `taso/.venv` using the Python version pinned in `taso/.python-version`, installs the initial dependency set in that isolated
Python environment, builds TASO's C++ library, installs the Python bindings,
and runs `run_baseline.py` on Input -> Conv -> ReLU.

This initial setup targets Linux x86_64, managed Python 3.11.17 and CUDA 12 with
NVIDIA cuDNN 8.9.7.29 (https://pypi.org/project/nvidia-cudnn-cu12/8.9.7.29/).
It does not change TASO's search source or replace the global Python packages.
Automatic toolkit installation supports Ubuntu 22.04/24.04 x86_64 with root
or sudo access. Other hosts need a preinstalled CUDA 12 toolkit. The NVIDIA
driver is retained; CUDA selection applies to the runner and its child processes.
TASO's Python build uses CUDA_HOME so it selects matching development headers. The actual
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

## Python environment

The runner downloads the pinned Python interpreter using uv if necessary;
it does not inherit Colab's default Python version. To create just the Python
environment, without building TASO:

```sh
python3 setup_env.py taso
source taso/.venv/bin/activate
python --version
```

The separate Tensat scaffold is created with `python3 setup_env.py tensat`.
Never copy a Mac virtual environment to Colab; recreate it from the checked-in
version and requirements files. A venv isolates Python packages, not the host
GPU driver, CUDA toolkit or system compiler. Colab resets can remove the
installed environments, so rerun setup when needed. `python-packages.txt`
records resolved package versions; this is not yet a complete validated lock.

An older `.venv-taso` may remain locally; the runner now uses `taso/.venv`.

No separate CUDA-install script or environment-activation step is required:

```sh
git pull --ff-only
python3 taso/baseline/setup_and_run.py
```

Order: GPU check -> select/install CUDA -> system build tools -> managed Python
and venv -> packages -> TASO native library -> Python bindings -> baseline.
CUDA 12 is the initial tested-for setup target, not a proven upper bound of
TASO compatibility. Full GPU execution is still awaiting verification.

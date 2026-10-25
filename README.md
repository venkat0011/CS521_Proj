# CS521 Project

Investigating LLM-guided tensor graph search using TASO and, later, Tensat.

## Layout

- `taso/TASO/`: vendored upstream TASO source, tracked as regular files.
- `taso/llm_context/`: one-time rule/source preparation and runtime graph/LLM interfaces.
- `taso/baseline/`: tiny vanilla TASO test and Colab setup notebook.
- A separate Tensat folder will be added later.

## Start with vanilla TASO

Open [the baseline notebook in Colab](https://colab.research.google.com/github/venkat0011/CS521_Proj/blob/main/taso/baseline/vanilla_taso_colab.ipynb).
Select a GPU runtime, then run cells in order. The notebook clones this repository;
there is no source ZIP upload step.

The Colab installation is not yet verified. The legacy TASO checkout requires
compatible CUDA/cuDNN libraries, and the notebook may stop at the dependency
check. Save the first failure output for diagnosis. No successful baseline
performance results or bottleneck measurements have been obtained yet.

After installation, the smoke test constructs Input -> Conv -> ReLU and records
search output, graph costs, operator counts and execution timings. Internal
profiling and LLM-controlled search are subsequent steps.

## Upstream attribution

TASO: https://github.com/jiazhihao/TASO

Vendored from commit `f11782c4db180ad7d1e7ff7f4bb82e574ac3b6b8`.
The original Apache 2.0 license is retained at `taso/TASO/LICENSE`.
The upstream Git metadata is not part of this repository.

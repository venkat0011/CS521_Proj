# Vanilla TASO baseline

Local execution was attempted and failed: TASO is not installed. The host is
Apple Silicon macOS, CUDA tools are absent, and Docker is not running.
`results/baseline.json` records the actual failed attempt. No performance
numbers were obtained and no bottleneck has been established.

For Colab, open `vanilla_taso_colab.ipynb` in a GPU runtime and run the clone cell to obtain this repository. The notebook is UNTESTED on Colab.
It inspects dependencies before attempting a build and may stop for missing
or incompatible legacy cuDNN dependencies. Do not interpret a prepared notebook
as a successful installation. Keep the first failure output for diagnosis.

The tiny test constructs Conv -> ReLU, prints TASO substitutions, and saves
operator counts, estimated costs, measured execution times and optimization
wall time. These are smoke-test observations, not controlled speedup results
or numerical equivalence checks. Internal profiling is the next step after
successful execution. The TASO source itself has not been changed.

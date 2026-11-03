# Optional measured GPU feedback

`--gpu-feedback off|on|both` controls the checkpoint condition. With no option,
the existing suite behavior remains available. `off` disables checkpoint GPU
observations; TASO still uses its existing backend cost measurements and final
GPU evaluation. `on` runs the GPU-feedback condition. `both` interleaves the two
conditions and reports them separately in one comparison directory.

The total graph-expansion budget remains 100. Interval 10 makes decisions at
0,10,...,90; interval 25 uses 0,25,50,75. There is no model call after the final
budget. Reaching an empty queue can still end search earlier. All 151 rules,
the current reasoning/temperature settings, automatic OpenRouter routing and
the shared three-follow-up allowance per decision are retained.

## What the LLM sees

At each reached boundary, measure the graph currently best by TASO estimated
cost. Take the median of three native timing observations; each observation uses
upstream's 100 warm-up and 100 timed GPU executions. The prompt receives those
samples, median milliseconds, the previous and initial median, change since the
previous checkpoint, and speedup against the initial observation. The repeat
count is configurable with `gpu_feedback_repeats` in a suite JSON (1..10).

The prompt version is `checkpoint_compact_gpu_v1`. It retains the full compact
catalog/graph and explicitly distinguishes GPU observations from estimated cost.
Only this one best graph is observed; candidates/frontier graphs are not GPU
ranked. The LLM selects rules using feedback, while TASO's queue ordering,
admission threshold, incumbent and returned graph continue to use estimated cost.
A previously GPU-faster graph is not automatically retained as a new incumbent.

GPU observations require fresh LLM decisions because the inputs differ from the
plain condition. Cached plain-condition results can be reused without further
calls; their strategies are not substituted into the GPU-feedback arm.

## Timing snapshot and native build

The new `run_time_for_feedback` binding copies the live graph topology and creates
fresh execution operators. Weight-only operations (including constants, scalar
multiply, Enlarge/MergeGConv and weight fusion helpers) execute once outside the
inference timer. Only the runtime inference portion is timed. Temporary operators,
descriptors, buffers and inputs are released after every observation, including
on C++ timing errors. Original search topology, operator pointers, weights, GUIDs
and the priority queue are not replaced or rewritten by the measurement.

This avoids running `preprocess_weights` on the borrowed search graph and avoids
the old timing path's accumulation of mapped execution buffers across callbacks.
The ordinary final `run_time` path remains separate. Snapshot feedback is an
inference-latency observation; final `measured_ms` remains the primary endpoint
GPU quality measurement. Neither is a numerical-equivalence check.

The setup launcher rebuilds native TASO and Cython bindings. A GPU preflight runs
the real snapshot method before paid calls for each network. Missing bindings,
unsupported snapshot operators, or nonfinite/nonpositive timings fail explicitly
rather than silently reverting to estimated-only feedback. Use the update after
the current batch finishes; the new source fingerprint requires a new batch.

## Small Colab preflight

From `/content/CS521_Proj`, with Drive mounted:

```bash
python3 taso/run_experiment.py --experiment suite \
  --suite-config taso/benchmarks/gpu_feedback_smoke.json \
  --gpu-feedback both \
  --batch-dir "/content/drive/MyDrive/CS521 Result/gpu_feedback_smoke" \
  --dry-run
```

This uses SqueezeNet, Flash, B=20 and interval 10. There are zero provider calls.
It exercises GPU snapshots with all-rule preflight policies, which do not predict
later LLM search trajectories. Removing `--dry-run` runs one trial per condition,
with at most 16 total API attempts under the retry allowance.

## Interval 10: run both conditions

```bash
python3 taso/run_experiment.py --experiment suite \
  --suite-config taso/benchmarks/gpu_feedback_comparison.json \
  --gpu-feedback both \
  --batch-dir "/content/drive/MyDrive/CS521 Result/checkpoint_10_comparison"
```

The preset uses all three networks, four LLMs and five trial slots per group,
B=100 and interval 10. There are 120 paired TASO/LLM trial slots across two
conditions, up to 1200 initial policy calls, and at most 4800 API attempts if
every decision uses all follow-ups. Odd trials run plain then GPU-feedback;
even trials reverse that order. Each condition has an isolated worker/cost cache
and its own paired TASO control. Reuse the same unchanged output directory to
resume, rather than creating a new directory for each resume.

Inspect the plan without building TASO, importing GPU libraries or contacting a
provider by running `taso/benchmarks/run_suite.py --config
taso/benchmarks/gpu_feedback_comparison.json --gpu-feedback both --plan-only`.

## Interval 25: reuse the completed plain run

The source folder must contain `manifest.json`, `config.json` and `trials/`.
For the current Drive layout:

```bash
OLD_BATCH="/content/drive/MyDrive/CS521 Result/checkpoint_25"
python3 taso/run_experiment.py --experiment suite \
  --suite-config "$OLD_BATCH/config.json" \
  --gpu-feedback on \
  --reuse-checkpoint-batch "$OLD_BATCH" \
  --batch-dir "/content/drive/MyDrive/CS521 Result/checkpoint_25_gpu"
```

Use a different output directory. The source is read-only: no basic-condition
worker, timing, model call or replacement trial is launched. Failed/missing old
slots remain failed/unavailable. The combined CSV includes old observations and
new GPU-feedback trials. Imported usage remains historical, with
`model_calls_this_run=0` and `api_cost_this_run=0`. The old Drive symlink is used
as-is; it does not need to be removed or recreated.

Reuse verifies the saved config/manifest, budget, alpha, interval, trial count,
weight seed, model/network coverage and model/retry settings. A 25-interval batch
cannot serve as the plain control for interval 10. Available source hashes must
match either current sources or the explicitly reviewed pre-GPU hashes recorded
in `reuse_results.py`; unknown changes are rejected before model work. Required
rule/search/model/prompt hashes cannot be absent. Old manifests did not cover all
GPU backend files; new manifests include CUDA/DNNL source hashes as well. This
checks the recorded evidence and reviewed update, not unrecorded historical state.

`reused_checkpoint_reports.json` freezes the imported report contents, source
paths and per-file hashes. Resume uses that frozen snapshot. It does not reread
and silently change the historical observations. Imported GPU measurements may
come from another device/session; CSV GPU metadata and reuse provenance identify
that distinction. The source batch is not retimed to create a contemporaneous
paired comparison across conditions.

## Resume and reports

Resume a comparison with the same code and saved configuration:

```bash
BATCH="/content/drive/MyDrive/CS521 Result/checkpoint_10_comparison"
python3 taso/run_experiment.py --experiment suite \
  --suite-config "$BATCH/config.json" --batch-dir "$BATCH"
```

Completed and failed slots are skipped. A worker's native queue still cannot
resume after it exits; an interrupted slot is retained as failed. Historical
imports and completed model decisions are not repurchased automatically.

The comparison directory contains condition subfolders, provenance and combined
exports. `trial.csv` and `trials.csv` have identical contents. Key fields include
`feedback_mode` (`checkpoint`/`checkpoint_gpu`), `checkpoint_interval`, reused
status, GPU identity, estimated and final measured latency, search/optimizer wall
time, GPU-feedback overhead/sample count, retries, tokens and historical/new cost.
Conditions and intervals have separate distributions. `model_calls.csv` has every
API attempt; `gpu_feedback.csv` has every supplied GPU observation and samples.

`optimization_wall_seconds` includes GPU observation overhead inside callbacks;
`search_seconds` excludes callbacks. `gpu_feedback_seconds` reports complete
snapshot observation wall time, including mapping/constant preparation/cleanup;
the supplied median latency excludes that preparation and cleanup. Do not add
observation or inference time again to optimizer wall time. Measurements can warm
the GPU and affect subsequent timing; median sampling reduces noise but does not
establish a causal improvement by itself.

The comparison's default success threshold is three completed trials per
network/LLM/condition (or the available slot count in smaller smoke configs).
Override it with `--min-successful-trials`. All five planned attempts remain in
reports; state, for example, "3 successful of 5 attempted", rather than relabeling
the experiment as only three attempts. The summary records `meets_minimum`, full
individual values and distributions. The CSV keeps failures and their costs even
when the minimum is satisfied. No extra trials are silently purchased to reach
the threshold.

# Secondary experiment: Kimi CP25 candidate identity and quality

Scope: one LLM (Kimi K3.1 Pro), one guided variant (live CP25, no GPU feedback),
one DNN (SqueezeNet). ResNeXt-50 and Inception-v3 are outside the current study scope. The existing GPU smoke protocol is retained; its findings
are recorded in `GPU_SMOKE_FINDINGS.md`.

## Run the zero-API preflight first

From `/content/CS521_Proj` after syncing these changes:

```bash
python3 taso/secondary_experiment/run_candidate_study.py \
  --preflight-only \
  --networks squeezenet --weight-initialization fan_in \
  --out "/content/drive/MyDrive/CS521 Result/secondary_experiment/candidate_study_squeezenet_cp25_v1"
```

This rebuilds TASO, then runs the baseline reference batch for SqueezeNet only. It checks full
candidate recording, saved graph restoration, finite output comparisons and GPU
measurement before spending on Kimi. Send `summary.json` and any failed attempt's
`worker.log` if the preflight fails. Baseline work is reused when continuing with
the same sources, options and directory.

Once the preflight is validated, make the key available in the same terminal and
run the same command without `--preflight-only`:

```bash
python3 taso/secondary_experiment/run_candidate_study.py \
  --networks squeezenet --weight-initialization fan_in \
  --out "/content/drive/MyDrive/CS521 Result/secondary_experiment/candidate_study_squeezenet_cp25_v1"
```

**This full run makes new paid Kimi calls.** A complete 100-expansion trial has
four decisions at 0,25,50,75: normally 12 initial calls across three successful SqueezeNet guided trials.
Each decision retains the prior maximum of three follow-ups. At most five trial
attempts are launched (at most 80 requests with every follow-up used at all four
checkpoints of all five trials). Failed and interrupted trials
are retained and are not silently repurchased on resume.

## Order and budgets

- Complete the SqueezeNet baseline batch first: three successes, with at most
  five attempts. No Kimi calls start unless the baseline reaches its required success count.
- Each reference searches to 500 expansions and retains estimated-best snapshots
  at 0,100,250,500. Prefix observations come from one continuous native search,
  not independent reruns. Queue, visited graphs and costs persist.
- Then complete three live Kimi CP25 trials for SqueezeNet at budget 100, with at most
  five attempts. Save the estimated-best graph at 0,25,50,75 and the returned graph.
- No measured GPU values or numerical validation results are sent to Kimi. All
  checkpoint diagnostic execution happens after search is finished.
- A noisy result or a numerically disagreeing guided graph is retained; trials
  are not repeated until a favorable result is obtained. Invalid outputs are
  excluded from valid GPU improvement evidence.

`--reference-budget 1000` adds a more expensive reference endpoint; use a new
output folder when changing this option. The default fivefold budget increase
requires no extra LLM calls. Different workers have independent profiling caches;
references are comparative observations, not controlled counterfactual pairs.

## What is recorded

The opt-in native journal records **every generated candidate**, including
cycle/cost/size rejections, duplicates, admissions and expansions. Each generation
has a unique observation ID, parent graph ID, native rule index, phase and
expansion. It also records checkpoint and final search/execution graphs.

Operator descriptors include supported operator attributes, all tensor shape,
stride and split metadata, ordered port connections, and exact float32 leaf-weight
bytes. Backend convolution algorithms are recorded separately. Unsupported types
are marked incomplete, and cannot establish equality. The implementation targets the three existing single-input, single-output
inference graphs. Unsupported restored operators fail explicitly rather than
being approximated.

Comparison ignores GUIDs and memory addresses. An invariant refinement hash only
indexes possible matches; an exact labeled directed multigraph comparison checks
connectivity, shared nodes, port numbers, descriptors and weights before declaring
equality. A comparison exceeding its work limit is `unknown`, not a match or a
proven difference. Both raw candidate-record counts and unique established graph
classes are reported. Native search hashing/deduplication is left unchanged.

At checkpoints, the observed graph is the **current estimated-best graph**, not
the next expanded graph or the entire frontier. The native journal separately
tracks the complete generated/admitted/expanded trajectory. The final execution
graph can differ structurally from its search representation because native
weight preprocessing folds constant-only transformations; both stages are saved.

## Numerical and GPU checks

Use three deterministic random input seeds with unchanged original seeded weights
for each DNN. Finite, nontrivial reference outputs must pass in the zero-API
preflight; if a deeper graph overflows, it is reported and paid work is blocked.
Weights are not silently rescaled.
Inputs are scaled using only the original graph so its output is finite and not
trivially zero; scale and reference values are saved. This is needed because these
are untrained positive random weights. The exact same seed and calibrated scale
are used for candidate timing. Output shapes and finite values must agree within
`atol=1e-6, rtol=1e-3`. This is empirical agreement on three inputs, not universal
mathematical equivalence. The earlier timing-only smoke used input 0.125; keep
these workload differences explicit when comparing timing numbers.

Retained graphs are converted with native weight preprocessing before evaluation.
Snapshots and complete weights are saved before and after it. The folded TASO
100-expansion reference is reconstructed from its archive and must pass exact
graph and numerical checks; baseline preflight exercises this round trip. The
restored reference is measured first in the Kimi worker, followed by the retained
Kimi snapshots. CUDA algorithm restoration is checked separately and any mismatch
is reported. All graph reconstruction/diagnostic measurement occurs after Kimi's
search, so larger-budget reference profiling is not injected into its guidance.

Every fixed graph receives the existing two mappings and 20 timing blocks per
mapping. No extra clock warm-up or outlier filtering is introduced. GPU/host clocks,
raw samples, variation and telemetry remain available. A consistent direction
across mapping medians is reported descriptively, not as statistical significance
or proof that graph changes caused the observed speedup.

## Reports

`candidate_analysis.json` and `candidate_report.txt` answer:

1. How many guided candidates match reference generated/admitted/expanded graphs
   at each budget, with raw records distinguished from unique graph classes?
2. Did the estimated-best graph actually change at each checkpoint?
3. Which rule/parent sequence produced it, and was it generated before the last
   search block (hence already waiting in the search)?
4. Did the baseline winner appear anywhere in guided candidate generation?
5. Did valid graphs found at larger budgets improve estimated or recorded GPU
   quality, and did Kimi's final timing retain a consistent direction against the
   restored baseline within its diagnostic measurements?

Trials are separated as `baseline/<network>/attempt_NN` and
`kimi/<network>/attempt_NN`; candidate comparisons always use the same DNN.

Retain `worker.log`, `search_trace.json`, compressed `.candidates.jsonl.gz` journals,
checkpoints, reasoning/usage, observations, archived graphs/weights, numeric
outputs and raw timing CSVs. Exact weight bytes are stored once by SHA-256 in shared `weights/*.bin.gz`
blobs, and snapshots/journals reference them. Checksums are verified before
identity claims or restoration; these are lossless storage changes, not changes
to weight values. Keep the entire study directory, including `weights/`. If
relocating the study, set `TASO_AUDIT_ROOT` to its new root for blob lookup.

Full graph journaling can be large and CPU analysis
can take time. Audit I/O affects search wall time and potentially profiling
conditions, so these runs are not a search-speed benchmark.

If analysis was interrupted after searches completed, rerun without provider work:

```bash
/content/CS521_Proj/taso/.venv/bin/python \
  taso/secondary_experiment/analyze_candidates.py \
  "/content/drive/MyDrive/CS521 Result/secondary_experiment/candidate_study_squeezenet_cp25_v1"
```

## Limits on conclusions

Larger budgets yield **best-found references**, not a global optimum. TASO still
uses estimated-cost ranking, alpha admission and size limits; even queue exhaustion
is not an exhaustive proof over every possible equivalent graph. Exact structural
identity does not mean two different structures are inequivalent functions.
Numerical agreement, complete lineage and repeated timing strengthen the argument,
but hardware variability, fixed ordering and backend plan differences remain
material limits on causal claims about GPU speedups.

### Diagnosing the first preflight failures

The first GPU run failed before Kimi was started. SqueezeNet failed exact archive restoration; the old loader discarded tensor stride/split metadata and recreated pooling weights. Restoration now preserves these values and sharing; equality checks remain strict. A mismatch writes an adjacent `.restore_failure.json` diagnostic. GPU confirmation is still required.

ResNeXt-50 and Inception-v3 could not establish a finite, nonzero original output. The original error did not identify NaN, infinity, zero or empty outputs. `calibration_report.json` now records those counts for each input scale. No numerical disagreement is relabelled as a pass. Deterministic reference/restoration errors stop the affected group after its first failed attempt, rather than repeating five times.

The default `--weight-initialization original` keeps historical initialization. The explicit `--weight-initialization fan_in` option creates a new secondary workload with float32 weights sampled uniformly from [-sqrt(6/fan_in), sqrt(6/fan_in)]. This is a workload change, not a repair to historical measurements. Both TASO and Kimi use the same seed and weights, all three architectures retain the same shapes, and the executed construction is recorded and supplied to Kimi. Numerical calibration must still pass.

After updating/rebuilding, use a new output folder and run the zero-model-call preflight:

```bash
python3 taso/secondary_experiment/run_candidate_study.py --preflight-only --weight-initialization fan_in --out "/content/drive/MyDrive/CS521 Result/secondary_experiment/candidate_study_3dnns_v2"
```

Only after all three DNN baseline groups complete should the same command be repeated without `--preflight-only`, retaining the initialization and folder. Existing failed artifacts and the GPU smoke protocol stay unchanged.

### Exact native archive import (v3)

The v2 GPU preflight still failed. A CPU regression now links the actual TASO graph and operator constructors and reproduces a descriptor mismatch: weight replacement updates cached operator inputs without recomputing its output annotations. Rebuilding through public factories derives different annotations. The secondary loader now imports private native operators, preserves all archived input/output descriptors, copy flags and convolution algorithms, and checks exact graph identity. It validates ranks/shapes and rejects unsupported operators; it does not rerun preprocessing or change the original search algorithm.

Numerical archive fidelity is now compared against the saved candidate's own outputs, using the same input scale/seeds. Candidate agreement with the original DNN is a separate recorded result. A faithfully restored but numerically invalid candidate remains invalid and is retained as an experiment result. A finite candidate whose output changes on reconstruction still fails the preflight, with per-seed errors in `round_trip_checks.json`. Nonfinite candidate outputs cannot establish numerical round-trip agreement and are labelled uncheckable. GPU comparisons require both guided and reference graphs to pass original-network numerical checks.

Use a new `candidate_study_3dnns_v3` output directory after committing/updating Colab. The runner rebuilds the native bindings. Keep `--preflight-only --weight-initialization fan_in` for the first run; no model calls are made. Local native regressions cannot substitute for verification on the actual GPU.

The CUDA audit also found that `Conv2D::map()` allocated a bias buffer without initializing it, although every forward path adds it to the output. The buffer now starts at zero for these bias-free convolutions. The shared cost-measurement bias buffer is also zeroed before algorithm selection/timing. A regression executes the production mapping function against deliberately poisoned allocations and checks zero bias for fused and unfused paths. These fixes make future numerical checks meaningful; they do not retroactively establish correctness of historical outputs or GPU speedups.

## Current scope: SqueezeNet only

The user narrowed the secondary study to SqueezeNet on 24 November 2026, retaining one LLM (Kimi), CP25 without GPU feedback, and baseline budgets 100/250/500. Both entry points now default to SqueezeNet. Explicit multi-DNN options remain available, but larger networks are outside this study's scope. Earlier all-three-DNN notes below/above describe prior preflight work.

The supplied `summary-2.json` reports three completed SqueezeNet baseline trials, all numerically valid, with exact restoration at 100/250/500 expansions and zero model calls. All SqueezeNet observation records still flag timing concerns; these data do not establish a reliable GPU speedup. Preserve the existing run folder. A network-list/source change requires a new output directory under the runner's provenance checks; do not overwrite or relabel the existing three-DNN config.

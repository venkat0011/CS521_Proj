# Secondary experiment: GPU timing smoke test

For the candidate identity, checkpoint and larger-budget study on
SqueezeNet only (Kimi CP25, without GPU feedback), see
[CANDIDATE_STUDY.md](CANDIDATE_STUDY.md). Completed timing observations are
recorded in [GPU_SMOKE_FINDINGS.md](GPU_SMOKE_FINDINGS.md).

The separate `run_gpu_smoke.py` script handles setup and rebuilding;
`gpu_timing_smoke.py` contains the diagnostic runner and isolated workers.
Default local outputs go to `taso/results/secondary_experiment/gpu_timing_smoke`.

This first follow-up checks timing repeatability on a fixed returned graph. It makes
zero model/API calls. It does not yet test numerical correctness, exact candidate
identity, or optimality. A larger search budget supplies a stronger reference,
not a proof of a global optimum.

## Run in Colab

After pushing these changes and pulling them into the Colab checkout, run from
`/content/CS521_Proj`:

```bash
python3 taso/secondary_experiment/run_gpu_smoke.py \
  --out "/content/drive/MyDrive/CS521 Result/secondary_experiment/gpu_timing_smoke_v1"
```

The launcher sets up the environment and rebuilds the native library and Python
binding. Defaults: SqueezeNet, budget 100, alpha 1.05, CP25, weight seed 521.
Use a new output directory for a different configuration/source version. Running
the same unchanged command resumes without repeating finished/failed attempts.

The runner completes the baseline batch before starting the checkpoint batch.
Each batch stops as soon as three executions succeed, with at most five attempts
per batch. There is no alternating order and no concurrent GPU work. Workers have
fresh processes and cost caches; these are independent groups, not paired searches.
Failures, timeouts and interruptions remain recorded. If the baseline cannot get
three successes in five attempts, checkpoint work does not start.

**A noisy finite timing is a successful execution and is retained.** We do not
keep retrying until three stable-looking results happen. With no failures there
are three baseline workers and three checkpoint workers (three runs per variant).

## What the checkpoint stage means

By default it is a CP25 **all-rule control**, using the checkpoint interface but
not Kimi. This tests the measurement machinery without buying decisions. It must
not be reported as a Kimi performance result.

To measure graphs produced by saved Kimi CP25 schedules, provide a directory
containing that historical trial's `checkpoints.json`:

```bash
python3 taso/secondary_experiment/run_gpu_smoke.py \
  --replay-trial "/content/drive/MyDrive/CS521 Result/checkpoint_25/trials/REPLACE_WITH_KIMI_SQUEEZENET_DIRECTORY/trial_01" \
  --out "/content/drive/MyDrive/CS521 Result/secondary_experiment/gpu_timing_kimi_replay_v1"
```

The path above contains a placeholder: replace it with a real trial path. Repeat
`--replay-trial` for multiple historical strategies. They are assigned in
supplied order, cycling if necessary, with every assignment saved. No provider
is contacted. Replaying a policy executes a new search and does not reconstruct
the original graph or adapt its decisions to the new feedback. Policies must
contain every required boundary and match budget 100/interval 25. Use fresh
adaptive Kimi decisions only after this timing smoke has been assessed.

## What each worker measures

1. A discarded all-rule warm-up search, then a fresh measured search.
2. The returned graph stays fixed throughout GPU measurement; no new search is
   run between its timing samples.
3. A snapshot is mapped once. Constant-only transformations run once outside the
   inference timer. Runtime input tensors are initialized to the same finite
   value, 0.125, rather than left uninitialized.
4. Twenty blocks reuse those mapped execution operators/buffers. Every block
   performs 100 warm-up executions, synchronizes, then times 100 executions.
5. Each block records CUDA-event milliseconds per execution and synchronized
   host-clock milliseconds per execution. Host time includes launch and event
   overhead, so these clocks need not match exactly.
6. The snapshot is released, the **same unchanged returned graph** is remapped,
   and twenty more blocks are measured. This separates within-mapping variation
   from remapping-associated changes. Cleanup occurs even on C++ exceptions.

`--blocks 4` reduces the number of blocks; the default is 20. Measurements
exclude mapping and constant preparation; complete measurement wall time is saved
separately. GPU identity, clocks, utilization and temperature are sampled before
and after each mapping where supported by nvidia-smi.

## Read the results

Open `report.txt` and `summary.json` in the output root. Each attempt also saves:

- `timing_samples.csv`: every sample, in chronological mapping/block order.
- `worker_result.json`: medians, ranges, coefficient of variation, first-half to
  last-half median drift, remapping changes and available GPU telemetry.
- `search_trace.json`, warm-up trace, graph structure snapshots and checkpoint
  decisions for contextual inspection.
- `worker.log`: native output and any error/traceback.

A 10% relative range/drift threshold flags a diagnostic concern; it is not a
statistical significance threshold and does not determine execution success.
GPU-event time exceeding host time by over 5% is separately flagged for review.
The two clocks time the same kernels, not independent implementations. Agreement
and low variability are useful evidence, not proof of absolute timing accuracy.

The structural JSON uses incomplete public getters and contains GUIDs. Its file
checksum is **not a canonical graph fingerprint**, contains no weights and cannot
prove graph equivalence. The next candidate-tracking experiment needs full graph
identity/provenance and numerical validation. Avoid interpreting the smoke's
across-group median differences as confirmed speedups: graph choices and GPU
conditions can differ, and fixed baseline-first ordering still has time drift.

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

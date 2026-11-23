# Recorded GPU timing findings — retain as an unresolved limitation

The completed smoke run used SqueezeNet, budget 100, all rules in both baseline
and CP25 control, zero LLM calls, baseline batch first and checkpoint batch second.
Each variant completed three of three attempts. Each returned graph was kept
fixed for two mappings; each mapping retained execution buffers across 20 blocks,
with 100 warm-up and 100 timed executions per block. Input values were fixed at
0.125. GPU events and synchronized host time were both recorded. Numerical output
agreement was not checked by this smoke test.

Source results: `CS521 Result/secondary_experiment/gpu_timing_smoke_v1` on Drive.
No additional warm-up, clock locking, outlier removal or changes to this smoke
protocol have been implemented. The new candidate study is separate.

## Established observations

- Nine of twelve mapping groups exceeded a 10% max-minus-min range relative to
  their own sample median. The threshold was diagnostic, not significance.
- GPU-event and host-clock medians differed by approximately 0.011–0.027%; both
  clocks captured the excursions. This does not prove isolated kernel latency.
- Baseline attempt 1 had eight initially slow blocks. Its first-five median was
  1.249 ms, versus 0.864 ms for the last five: roughly 31% lower. Before/after
  telemetry showed compute clocks increasing from 585 to 1,590 MHz. This supports
  clock ramp-up as a possible contributor, without linking individual blocks to
  clocks because telemetry was sampled only before and after each mapping.
- Five of six first mappings had an elevated first block. Later isolated spikes
  also occurred, so removing the first sample would not resolve all variation.
- The first baseline remapping changed the median by -8.07%; the other five
  changes were +0.35% to +1.34%. No general second-mapping speed advantage appeared.
- Independent baseline searches also returned different graph compositions and
  generated 10,762–19,269 candidates. Between-search differences therefore cannot
  all be classified as timing noise; within-mapping changes occur on a fixed graph.

Downloaded CSV mapping, established by matching both mapping medians:

| Download name | Recorded attempt |
|---|---|
| timing_samples.csv | Checkpoint control 1 |
| timing_samples-2.csv | Checkpoint control 2 |
| timing_samples-3.csv | Checkpoint control 3 |
| timing_samples-4.csv | Baseline 1 |
| timing_samples-5.csv | Baseline 2 |
| timing_samples-6.csv | Baseline 3 |

## Interpretation carried into the candidate study

These observations establish timing variability without graph changes. They do
not identify a broken timer, prove clock changes caused every spike, or establish
which historical GPU speedups were real. Exact graph identity, larger search
budgets and numerical checks address different questions; none alone makes a
latency difference reproducible. Keep raw samples, variability, fixed-order bias,
backend algorithm differences and independent-search conditions explicit.

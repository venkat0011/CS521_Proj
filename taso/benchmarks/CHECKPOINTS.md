# Online checkpoints by graph-expansion budget

The suite supports one-shot and checkpoint guidance. Without
`--checkpoint-interval`, the existing one-shot path is used. Supplying the flag
without a value selects 25; explicit values such as 10 and 5 are also supported.
The total budget stays in the suite configuration (`budget: 100`).

| Interval | Policy calls at expansions | Maximum calls per trial |
|---|---|---:|
| 25 | 0, 25, 50, 75 | 4 |
| 10 | 0, 10, ..., 90 | 10 |
| 5 | 0, 5, ..., 95 | 20 |

There is no call at the final budget. If the budget is not divisible by the
interval, the final chunk receives the remaining expansions. A larger interval
than the budget makes a single call. Candidate counts are feedback, not triggers.

## Run

For the first GPU check, `checkpoint_smoke.json` selects SqueezeNet, Flash and one
trial: at most **four paid calls**. Use the same fresh batch directory for its
dry run and paid run:

```bash
python3 taso/run_experiment.py --experiment suite \
  --suite-config taso/benchmarks/checkpoint_smoke.json --checkpoint-interval 25 \
  --batch-dir /content/CS521_Proj/taso/results/checkpoint/checkpoint25_smoke_v2 \
  --dry-run
```

After the dry run succeeds, remove `--dry-run`. The following examples use the
full five-trial, three-network configuration instead of this smoke preset.

From the project root, inspect the plan without a GPU or model call:

```bash
python3 taso/benchmarks/run_suite.py --plan-only \
  --model google/gemini-3.8-flash --checkpoint-interval 25
```

This is at most 60 calls: three networks × five trials × four calls per trial.
The full four-model batch is at most 240 calls. The setup runner rebuilds the
modified native runtime and Python bindings. In a Colab GPU terminal:

```bash
python3 taso/run_experiment.py --experiment suite \
  --model google/gemini-3.8-flash --checkpoint-interval 25 \
  --batch-dir /content/CS521_Proj/taso/results/checkpoint/checkpoint25_flash_v2 \
  --dry-run
```

The dry run executes the actual native checkpoint hook with a fixed all-rules
policy and saves the graph/progress/request at each reached preflight boundary.
No API client is constructed. These are all-rule preflight observations, not the
feedback the future adaptive search is guaranteed to produce. Check
`prepared/<network>/checkpoint_preflight/`. The initial prompt is also copied to
`prepared/<network>/request.json`.

With `OPENROUTER_API_KEY` set, remove `--dry-run` to run the trial slots:

```bash
python3 taso/run_experiment.py --experiment suite \
  --model google/gemini-3.8-flash --checkpoint-interval 25 \
  --batch-dir /content/CS521_Proj/taso/results/checkpoint/checkpoint25_flash_v2
```

For 10 or 5, change the option AND choose a new batch directory. To run all four
configured models, omit `--model`. For a smaller smoke run, copy `config.json`,
set one network and `trials: 1`, and pass it through `--suite-config`.

## Search and strategy semantics

The existing C++ search loop is shared by both modes. Its priority queue, visited
hashes, current best graph, rewrite definitions, alpha and cost cache stay alive
throughout a checkpoint run. A synchronous callback asks Python for a strategy
at each reached budget boundary; search then continues inside that same loop.
There is no graph export/import or fresh optimizer call between intervals.

Each checkpoint response uses the existing strategy JSON schema, restricted to
**one phase for the next interval**. Its budget must equal that interval's exact
remaining chunk. The one-shot path still permits its existing multi-phase plans.
An interval of 100 therefore means one callback and one phase, not a recreation
of every possible multi-phase one-shot strategy.

Feedback includes the current best found graph, remaining budget, queue size,
estimated improvement, cumulative rewrite counters, active-rule statistics for
the last interval and previous policy decisions. The best found graph can be an
unexpanded queue candidate, so it need not be the next graph TASO pops. It is not
a complete view of the frontier. The rule catalog, model construction source
and selected search source remain in the prompt. No GPU runtime measurements are
provided to the LLM in this version.

A skipped rule is not replayed on an already expanded graph. If the queue empties,
search stops early and saves `stop_reason: queue_empty`. There are no extra early
model calls, automatic rule fallbacks or re-expanded graphs. This preserves the
rule-filtering behavior of the existing one-shot experiment.

These checkpoints are **in memory only**. Completed trial slots can be skipped
on batch resume, but an interrupted trial cannot resume its queue after the GPU
process exits. It remains a failed trial; the runner makes no replacement paid
calls. `--retry-evaluation` is not supported for checkpoint batches. Responses,
usage, intent markers and completed decisions remain available for inspection.

## Results and timing

Each trial saves `checkpoints/expansion_XXXX/` with the live graph, full progress,
LLM feedback, exact request, raw API response, validated strategy and inference
metadata. `checkpoints.json` records all reached decisions. API cost, token counts
and inference latency are summed across the actual requests in `trials.csv`.
Interrupted requests are counted as attempts; their provider usage may be unknown.

Both final graphs are measured on the GPU. Final `measured_ms` is the primary
latency-quality observation; `estimated_ms` remains TASO's search surrogate.

- `search_seconds`: optimizer time excluding time inside the policy callbacks.
- `checkpoint_callback_seconds`: callback time, including graph/prompt preparation,
  API waiting, validation and saving artifacts.
- `optimization_wall_seconds`: actual optimizer wall time including callbacks.
  CSV names are `llm_optimization_wall_seconds` and `baseline_optimization_wall_seconds`.
- `inference_seconds`: summed API request latency, a component of callback time.
  Do not add it again to `optimization_wall_seconds`.
- `llm_inference_plus_search_seconds`: retained legacy metric; it omits other
  callback overhead and is not the complete online optimization wall time.

Warm-up, model construction and final GPU measurements are separate from optimizer
wall time. Each checkpoint trial performs two discarded all-rule warm-up searches,
one for each policy slot, without model calls. Timed baseline/LLM order alternates
by trial as before. Earlier one-shot batches warm their selected policies instead;
record that protocol difference when comparing those batches. Shared operator
caching and GPU timing noise remain relevant.

## Implementation

The shared launcher is `taso/run_experiment.py`; `baseline` inside a paired
trial still means its all-rules TASO control. The checkpoint method itself lives
in the experiment/benchmark code.

- `TASO/src/core/ops.cc`: shared search loop, optional callback, per-chunk trace.
- `TASO/python/taso/`: binding plus `optimize_with_checkpoints` entry point.
- `llm_context/checkpoint_strategy.py`: next-interval prompt, validation and API records.
- `benchmarks/checkpoint_worker.py`: paired online evaluation in one GPU process.
- `benchmarks/run_suite.py`: optional interval, call-count plan and trial scheduling.

Local checks exercise the actual C++ loop against a synthetic backend, generate
and syntax-check the Cython bridge, and test the full Python dry-run/trial/resume
path using mocked model calls. GPU execution must still be verified in Colab.

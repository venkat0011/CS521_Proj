# First manual TASO strategy

Run after pulling the update in a Linux GPU environment:

```sh
python3 taso/baseline/setup_and_run.py --experiment manual
```

This rebuilds the native library and Python extension before running the demo.
The default `--experiment baseline` still runs the original tiny smoke case.
No provider credentials or LLM calls are used.

## Network and policy

The demo has two branches, Conv(3x3)->ReLU and Conv(1x1)->ReLU, sharing an
input. Their results are added, then passed through Conv(3x3)->ReLU. Weights
use a fixed random seed. Input shape is [1,8,16,16]; convolutions have 8 output
channels. This is an executor demonstration, not a representative benchmark.

`manual_conv.json` requests two expansions using the four registered Conv/ReLU
fusion rules, then up to 22 expansions with all 151 registered rule instances.
Each phase shares the same cost-ordered candidate queue, visited hashes,
best graph, and cost threshold. There is no search restart at phase boundaries.

A skipped graph/rule application is pruned on that expansion. Later phases do
not replay expanded graphs. If the queue empties early, search ends; later
phases do not restart the input graph. Explicit rule order is preserved but
reordering the same complete set alone is not expected to change exploration.

The exploratory alpha=1.5 and total budget=24 are identical across variants.
These are demonstration settings, not tuned values. The scheduler counts
exact graph expansions. Existing `optimize(..., budget=N)` retains its N+1
expansion convention; the comparison passes 23 to that legacy API.

## What the experiment checks

1. Legacy optimize and a single all-rules phase have the same estimated result.
2. Splitting all-rules into two phases preserves cost, expansions and counters.
3. The manual schedule respects its budget and enabled rule set.
4. Full rule loading and actual rule matching are active.

All controls share the refactored core loop. They are consistency checks,
not an independent comparison against a separately compiled upstream binary.
All runs reuse the same Model operator-cost cache to keep estimated costs
consistent. Optimization times are therefore warm-cache observations and
must not be used as a speedup claim. No numerical equivalence check is added.
GPU measurements happen only after the returned graph is selected.

## Outputs

`taso/baseline/results/manual_summary.json` contains comparisons and check
results. `all_rules_trace.json`, `split_all_rules_trace.json`, and
`manual_trace.json` record phase expansions and per-rule attempts, recursive
match checks, complete matches, generated/admitted graphs and rejection counts.
Copy the terminal summary back to the project chat. A native assertion failure
may interrupt a child process before it writes a summary; then share setup.log.

## Important build correction

Upstream TASO has necessary side effects inside assert expressions, including
protobuf parsing and reading operator constraints. Release builds define
NDEBUG and can silently skip those operations. The runtime target now compiles
with -UNDEBUG. This can change behavior from the previous smoke run; that run
is not a valid search-quality baseline. Rebuild before comparing strategies.

## Local verification

Without a GPU:

```sh
python3 -B -m unittest discover -s taso/tests -p 'test_*.py'
c++ -std=c++11 -I taso/TASO/include taso/tests/test_schedule.cc -o /tmp/taso_schedule_test
/tmp/taso_schedule_test
```

The local test suite also executes the production C++ search-loop source
against a synthetic graph backend, covering final-budget candidate selection,
rule filtering, queue exhaustion and graph cleanup. It does not execute real
TASO graph matching or GPU kernels.

Core syntax checks may use `taso/tests/stubs/rules.pb.h`, which supplies only
protobuf declarations. That is not a substitute for a real protobuf/CUDA build.

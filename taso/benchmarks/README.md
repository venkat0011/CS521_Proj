# Three-network, five-trial TASO experiment

The default configuration compares all-rule TASO with a one-shot LLM schedule
on SqueezeNet, ResNeXt-50 and Inception-v3. Every search has an upper
bound of **100 whole-graph expansions**, and each LLM/network pair gets **five
independent API requests**. Phases are preplanned; there is no online feedback.
The default configuration uses four LLMs via OpenRouter: **60 requests** for a
complete three-network batch (four LLMs × three networks × five trials). Failures occupy their original trial slots.

## Model parameters

`config.json` requests temperature `0.2` and reasoning effort `medium`, with
explicit `high` reasoning overrides for DeepSeek and Kimi:

| Category | OpenRouter model ID | Temperature sent | Reasoning sent |
|---|---|---|---|
| Open-weight | `deepseek/deepseek-v4-pro` | `0.2` | `high` |
| Open-weight | `moonshotai/kimi-k3.1-pro` | omitted (provider default) | `high` |
| Proprietary | `openai/gpt-5.3-codex` | omitted (provider default) | `medium` |
| Proprietary | `google/gemini-3.8-flash` | `0.2` | `medium` |

The default batch excludes Claude Opus and Gemini Argon. Their profiles remain
available for explicitly selected runs. Flash's temperature parameter currently
routes to compatible Google AI Studio endpoints under strict parameter routing;
endpoints that do not advertise temperature support are excluded.
[Flash endpoint capabilities](https://openrouter.ai/api/v1/models/google/gemini-3.8-flash/endpoints)
and [Google's thinking levels](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
were verified for this update.

Capabilities were checked on 2026-10-31 against OpenRouter's
[model catalog](https://openrouter.ai/api/v1/models) and per-model endpoint lists.
The fixed profiles are in `llm_context/model_settings.py`; verify and update them
before adding a different model. Unknown models or unsupported effort settings
fail before GPU preparation or paid calls. The adapter uses OpenRouter's unified
[`reasoning.effort`](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
field and retains strict provider parameter routing. Provider availability can
still change after this capability snapshot.

The plan, `inference_metadata.json`, full trial records and `trials.csv` distinguish
requested from sent temperature/reasoning and record omissions and overrides.
An omitted temperature is **not** a claim that the provider used `0.2`. Reasoning
labels are provider controls; equal labels do not imply equal compute across LLMs.
`max_tokens` defaults to **32768**, including reasoning and visible output.
Use `--max-tokens provider` to omit the application's cap; provider defaults and
model limits still apply. There is no automatic retry or increase in this limit.
[Output limits and usage reports](../llm_context/TOKEN_LIMITS.md) describe the
per-request `model_calls.csv` export, including failed checkpoint calls.

Standalone legacy DeepSeek calls retain their existing default reasoning.
The new suite explicitly requests `high` for DeepSeek; its profile does not
support `medium`. Earlier provider-default DeepSeek results remain available in
their original batches. This new configuration starts fresh trial slots for all
selected models; it does not import old trials or guarantee the old default effort.

The default one-shot mode makes one strategy request per trial.
The separate [checkpoint continuation preset](../llm_context/CONTINUATION.md)
starts with model-specific maximum allowances and bounds follow-ups at three per
checkpoint. It records all attempts and leaves the default configuration unchanged.
[Online checkpoint mode](CHECKPOINTS.md) is enabled with `--checkpoint-interval`
and supports intervals such as 25, 10 and 5 expansions. Checkpoint requests use
the [compact representation](../llm_context/COMPACTION.md).

## Run in the Colab GPU terminal

From the project root, first prepare the three actual prompts and exercise each
network with all-rule TASO. This step makes no model calls:

```bash
python3 taso/run_experiment.py --experiment suite --dry-run \
  --batch-dir taso/results/one_shot/budget_models_100_v2
```

The first invocation builds the updated TASO bindings and installs the plotting
dependency. Check the `prepared/<network>/worker_result.json` files. Then run:

```bash
export OPENROUTER_API_KEY="your-key"
python3 taso/run_experiment.py --experiment suite \
  --batch-dir taso/results/one_shot/budget_models_100_v2
```

Keep the key in your terminal environment. It is never written to result files.
The second command reuses the prepared prompts and successful preflights.
Running it again resumes unfinished work and skips completed or failed model
calls. A failed GPU evaluation can explicitly be retried without another API call:

```bash
python3 taso/run_experiment.py --experiment suite --retry-evaluation \
  --batch-dir taso/results/one_shot/budget_models_100_v2
```

Use a new batch directory for this update; do not resume an old model batch.
Configuration and source hashes are frozen for a batch. If either changes, use
a new results directory. An interrupted API request without a saved valid
strategy becomes a failed trial on resume; its request is never silently repeated.
For larger models, native assertions, timeouts and GPU out-of-memory errors are
saved as failures. GPU workers run in separate processes so one failed network
does not terminate the entire batch. The one-hour worker timeout is a safety
limit, not an alternative search budget.

To inspect the plan locally without a GPU or API call:

```bash
python3 taso/benchmarks/run_suite.py --plan-only
```

To run only Flash (15 calls across three networks), use a separate directory:

```bash
python3 taso/run_experiment.py --experiment suite \
  --model google/gemini-3.8-flash \
  --batch-dir taso/results/one_shot/flash_100_v2
```

To add LLMs, copy `config.json` and edit `models`, then pass
`--suite-config path/to/config.json` to the setup runner. All pairs still receive
five trial slots. Other options, including `max_tokens`, are saved in the batch.
The default output limit is 32768 because the earlier DeepSeek smoke run needed
more than 8192 tokens. Context is the existing full rule/source prompt; the large
prompt and every provider's reported token usage/cost remain visible.

## Raw results and distribution

The batch directory contains:

| File | Contents |
|---|---|
| `trials.csv` | One row per planned paired trial, including failures and pending slots |
| `trials.jsonl` | Full trial records, inference metadata and paired evaluation reports |
| `summary.json` | Every numeric metric's trial values, mean, median, sample SD, min/max, Q1/Q3, IQR |
| `summary.md` | Quick table showing each trial's paired estimated-cost ratio |
| `distributions.png`, `distributions.pdf` | Individual trial dots over box plots for estimated cost, GPU latency, search time and expansions |
| `prepared/<network>/` | Frozen actual graph, construction source, full rule catalog, exact prompt and baseline preflight |
| `trials/<network>__<model-hash>/trial_XX/` | Raw request/response, validated strategy, model usage, logs and full rule-level search traces |

The same frozen prompt is used for all five requests and all LLMs for a network.
Saved model IDs are paired with the model hash in every trial record. Returned
provider/model IDs are also retained. Five requests may produce identical
strategies; the runner does not force artificial diversity.

After downloading the batch, regenerate its tables/figures without TASO or a GPU:

```bash
python3 -m pip install -r taso/benchmarks/requirements.txt
python3 taso/benchmarks/run_suite.py --summarize \
  --batch-dir taso/results/one_shot/budget_models_100_v2
```

Quartiles use linear interpolation at `(n-1)*p`. Standard deviation uses `n-1`
and is undefined with fewer than two observations. Ratios are computed **within
each paired trial**, before aggregation. Below 1 means the LLM result is lower.
An unused budget is recorded: queue exhaustion may stop either policy before
100 expansions. Counts distinguish matching, generated graphs, admissions and
expansions; these are different quantities.

Each trial evaluates a fresh LLM strategy against a fresh TASO control in one
isolated process. The network and random weight seed (521) stay fixed. Both
policies receive a discarded warm-up search before their timed searches, sharing
TASO's operator-cost cache. Trial order alternates baseline-first and LLM-first
(three versus two across five trials). Warm-up cost is saved separately; this
reduces cache-order effects but does not prove all cache effects are eliminated.
GPU latency is one upstream `run_time()` observation per final graph, internally
averaged over repeated executions, rather than five measurements of one graph.
The displayed distribution is across five strategy/evaluation trials and includes
GPU cost-model noise. These five observations do not establish significance.
Inference time and API cost are reported separately; inference-plus-search time
excludes setup, graph construction and experimental warm-up. Numerical equivalence
is not checked by this runner.

## Architecture and paper provenance

These are fixed TASO workload variants, not downloaded pretrained models:

- SqueezeNet uses the SqueezeNet 1.1 Fire layout, without biases; output remains
  `[1,1000,1,1]`, and inference dropout is omitted. The layout follows
  [torchvision's implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py).
- ResNeXt-50 reuses `TASO/examples/resnext50.py` construction, including its
  projection ReLU and absence of a classification head.
- Inception-v3 reuses `TASO/examples/inceptionv3.py` construction at batch 1,
  including its 1x1 final pooling and absence of a classification head.

Only graph-building definitions/statements are extracted from upstream scripts;
their top-level optimizer/export calls never execute. The exact extracted source
and its hash are saved and supplied to the LLM. Updated graph extraction exposes
fused Conv activation, since many upstream convolutions already contain ReLU.

The [TASO paper, Section 7.1](https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_taso.pdf)
sets alpha to 1.05 and reports searches taking under ten minutes. It does **not**
specify per-network graph-expansion budgets. Our shared 100-expansion limit is
a project choice, not a claimed replication of the paper's budget. The public
examples use 100 for ResNeXt-50/NASNet-A and 30 for Inception-v3; this suite uses
100 for all three as requested. The scheduled API enforces an exact upper bound,
avoiding the legacy API's additional expansion. SqueezeNet and Inception-v3
were not among the five architectures evaluated in TASO's original paper.

## Folder names

`baseline` in a trial's JSON/trace means its all-rules TASO control. The suite
itself is an LLM experiment and uses the neutral `taso/run_experiment.py` launcher.
With no explicit `--batch-dir`, new batches are grouped under
`taso/results/one_shot/<run>/` or `taso/results/checkpoint/<run>/`. Historic results
under `taso/baseline/results/` remain readable at their original paths. Do not
reuse an old frozen batch for new evaluations after this source change.

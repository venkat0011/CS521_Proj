# One-shot LLM strategy through OpenRouter

`prompt_template.txt` follows this structure:

1. Actual model or graph-construction source.
2. Live TASO graph: operators, tensor shapes, edges, exposed attributes, outputs.
3. Registered rewrite catalog, with actual IDs and pattern/factory definitions.
4. TASO search source.
5. Fixed expansion budget, alpha, phase limit and execution semantics.
6. Exact phase-schedule JSON output contract.

The model produces the whole plan before search starts. It may choose one or
several phases; phases do not involve model calls or feedback. This is a first
one-shot experiment, not the later offline strategy-evolution loop across
training graphs. Online feedback is not implemented.

## Inspect the prompt without an API call

In the project directory of your Colab GPU terminal:

```sh
git pull --ff-only
python3 taso/baseline/setup_and_run.py --experiment llm --dry-run
```

The runner builds TASO if necessary, constructs the small CNN, extracts the
input and saves a prompt. The console prints the new run directory under
`taso/baseline/results/llm/<run>/`. Open `request.txt` to read the exact input.
No OpenRouter key is required for this mode.

## Make one DeepSeek V4 Pro request and evaluate

Enter your key without displaying it or saving it in shell history:

```sh
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
export OPENROUTER_API_KEY
printf '\n'
python3 taso/baseline/setup_and_run.py --experiment llm
```

Default model ID: `deepseek/deepseek-v4-pro`.
Change it via `--model MODEL_ID` or `OPENROUTER_MODEL`.
The adapter makes one completion request, using JSON-object output mode and
requiring provider support for requested parameters. It never automatically
repairs a response or retries a paid call. This mode consumes API credits.
Model listing: https://openrouter.ai/deepseek/deepseek-v4-pro
API documentation: https://openrouter.ai/docs/api/reference/overview
JSON outputs: https://openrouter.ai/docs/guides/features/structured-outputs

## What gets evaluated

The experiment uses B=24 expansions, alpha=1.5 and at most four phases. The
model chooses phase count, rule selections and expansion allocations. The
local validator requires budgets to sum to exactly B, checks exact fields,
rejects unknown/duplicate IDs and duplicate JSON keys, then uses the same
strategy compiler as `taso.optimize_with_strategy`.

Once validated, the generated strategy is compared with a single all-rules
phase on the same small CNN. No baseline search results or GPU measurements
are given to the model. Final GPU timings are evaluation only. All variants
reuse the operator-cost cache; these are smoke observations, not controlled
speedup measurements. Numerical equivalence is not verified by this script.

## Saved outputs

Each run has its own directory, preserving earlier attempts:

- request.txt / request.json: exact prompt and messages.
- request_metadata.json: input hash/byte size and experiment settings.
- api_request.json: model/parameters/messages, without authorization headers.
- api_response.json: provider response including usage when returned.
- inference_metadata.json: requested/returned model, provider, timing and usage.
- strategy.json: validated schedule, only after successful validation.
- baseline_trace.json / llm_trace.json: native search counters.
- summary.json: preparation, evaluation results, or failure.

To rerun a saved strategy without spending on another model request:

```sh
python3 taso/baseline/setup_and_run.py --experiment llm \
  --strategy-file /content/CS521_Proj/taso/baseline/results/llm/RUN_ID/strategy.json
```

Incomplete/token-limited responses stop the run. Inspect the response; you can
explicitly request a larger output limit using `--max-tokens 16384` for another
paid attempt. Reasoning models may spend output tokens before returning JSON.

## Code layout

- prepare_context.py: extract static rules/source once; source hashes trigger
  regeneration when the checkout changes.
- graph_context.py: inspect the current graph.
- prompt_template.txt: human-readable template.
- llm_strategy.py: assemble messages and validate a returned phase schedule.
- openrouter_client.py: standard-library HTTP adapter, key from environment.
- ../baseline/run_llm_strategy.py: one-shot generation and evaluation entry point.

The default model-source section is the real TASO CNN builder. We do not invent
PyTorch source for a model that was built in TASO. An actual model source file
can be supplied to the experiment using --model-source when running it directly
with the TASO environment and native library paths configured.

Existing public graph getters omit some attributes, including fused activation.
These gaps are recorded in graph warnings. Weights are not sent. The source
and full catalog make the prompt large; request_metadata.json reports bytes,
not exact tokens. Input tokens are recorded from the real provider response.
The input is sent on each stateless request; provider caching is not assumed.

No live paid API request has been made by the coding agent. The provider
adapter and validation have local fixture tests; real model behavior requires
your first Colab request.

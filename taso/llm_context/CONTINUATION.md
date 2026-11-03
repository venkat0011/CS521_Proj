# Bounded checkpoint continuation

The `checkpoint_continuation.json` preset implements a separate, recorded
completion protocol. The existing suite config, one-shot prompt, reasoning and
temperature profiles, compact checkpoint prompt, native TASO search and bindings
are unchanged. Do not rerun completed one-shot experiments for this update.

## Run

Inspect the full plan without building TASO or making any network request:

```bash
python3 taso/benchmarks/run_suite.py \
  --config taso/benchmarks/checkpoint_continuation.json --plan-only
```

Run the full study in the Colab GPU terminal with the API key already available:

```bash
cd /content/CS521_Proj
python3 taso/run_experiment.py --experiment suite \
  --suite-config taso/benchmarks/checkpoint_continuation.json \
  --batch-dir "/content/CS521_Proj/taso/results/checkpoint/checkpoint_25/continuation_$(date -u +%Y%m%dT%H%M%SZ)"
```

The preset selects 3 networks, 4 LLMs, 5 trials per pair, B=100, interval=25,
`max_tokens: "max"`, `completion_policy: "continue_on_length"` and `max_retries: 3`.
It permits at most 4 decisions and **16 API attempts per trial**, or **960 API
attempts across all 60 trials**. Most decisions can finish on their first attempt.
The original request plus three follow-ups share a single limit per checkpoint;
provider retries do not get a separate allowance.

For a small GPU/API validation, use `--suite-config
taso/benchmarks/continuation_smoke.json` and another fresh batch folder. This is
one SqueezeNet/Flash trial: at most 16 API attempts. Add `--dry-run` for a native
all-rule preflight with no model calls; it cannot establish provider replay
compatibility. Add `--model deepseek/deepseek-v4-pro` to test DeepSeek instead.

`--checkpoint-interval 10` or `5` overrides the preset. `--max-retries 0`, `1`,
`2` or `3` overrides the retry allowance. With other checkpoint configs,
`--max-retries 3 --max-tokens max` enables the same policy. `provider`/null caps
cannot be used with continuation because the output allowance must be explicit
for the context guard. Choose a new batch after source/config changes.

The existing Drive symlink is used as-is. The setup launcher creates/reuses the
venv and builds TASO as usual. Completed trial slots can be skipped when reusing
an unchanged batch, but an exited worker's native queue cannot be restored.

## Output allowances and automatic routing

`checkpoint_continuation_v2` lets OpenRouter choose the provider for every
request. No provider name, allowlist, order, ignored-provider list, quantization
selector or fallback restriction is sent. The existing `require_parameters`
check remains so OpenRouter selects endpoints supporting the requested settings.
The response's actual provider is recorded; it may change between attempts.

The model-level output/context ceilings in `checkpoint_completion.py` are dated
2026-11-03. DeepSeek's ceiling is corrected to 393216 from an actual backend
validation error; the earlier advertised 943718 value was rejected by GMICloud.
The code no longer treats that advertised endpoint number as reliable.

| Model | Configured output ceiling | Context guard | Input cap if separate |
|---|---:|---:|---:|
| DeepSeek V4 Pro | 393216 | 1048576 | — |
| Kimi K3.1 Pro | 943718 | 1048576 | — |
| GPT-5.3-Codex | 128000 | 400000 | 272000 |
| Gemini 3.8 Flash | 65536 | 1048576 | — |

`max` means this configured model ceiling subject to guarded context space;
it does not dynamically discover every provider's maximum. OpenRouter handles
routing with the requested output allowance. Backend capabilities can change;
validation errors remain explicit failures, rather than repeated identical
parameter errors. Numeric `--max-tokens` caps remain available. All limits are
recorded with the source snapshot for the batch.

The request cap is the minimum of the configured model ceiling, guarded remaining
context, and any explicit numeric user cap. The initial input estimate uses
serialized UTF-8 bytes for these text requests. Later estimates conservatively
include previous estimated/reported input, reported generated tokens and new-turn
overhead. A 4096-token margin is reserved, and an attempt is stopped locally if
less than 1024 output tokens remain or GPT's separate input cap is exceeded.
This is an estimate, **not a tokenizer or proof of exact provider tokenization**.
No rule, source or reasoning history is silently truncated to fit. A provider
context-length rejection remains a terminal, recorded failure.

The preset sets the HTTP/socket timeout to 1800 seconds and the worker wall-time
limit to 30000 seconds, including warm-up/GPU work and up to 16 attempts. Socket
timeouts apply to blocking network operations, not a guaranteed total request
duration. These are explicit preset values; old configs retain their 240-second
socket default and existing worker timeout. If a transport
timeout occurs, the checkpoint stops; it does not blindly duplicate an in-flight
request whose provider usage may be unknown.

## Completion and failure handling

At each reached boundary TASO stays paused inside the synchronous Python callback.
The native queue, visited set, incumbent graph and expansion counter remain alive.

1. A normal `stop` must contain a strategy satisfying the existing schema,
   available rule IDs and exact interval budget. Empty output, missing choices,
   invalid JSON or invalid strategy data can use the bounded follow-up allowance.
   A strategy is accepted only after validation succeeds.
2. A `length` response is recorded as confirmed output-token exhaustion. Empty
   or invalid strategies with other finish reasons are recorded separately;
   absence alone is not evidence of token exhaustion.
3. Returned partial answer text and readable reasoning/text summaries are reused
   as context, asking for one **complete replacement strategy JSON**. Partial
   JSON fragments are never blindly concatenated or executed. All original
   response fields, including opaque or signed reasoning blocks, remain in the
   saved API response. Provider-specific opaque/signature fields are not replayed
   to an arbitrary next provider. Only readable work is reused across routing.
4. With no readable work, the next attempt requests the strategy again. This is
   recorded as `no_readable_work_retry`; it does not pretend to resume hidden
   reasoning. Missing usage stays unknown, and context is guarded using the
   available text size instead. These retries still count against the same limit.
5. HTTP/provider errors 429, 500, 502, 503 and 504 can retry the same current
   history with bounded backoff. `Retry-After` is respected up to 60 seconds;
   a longer requested wait stops the checkpoint. Authentication, configuration,
   context, content-filter and unsupported-format errors remain terminal.
6. The original plus **three follow-ups total** is the shared limit for all retry
   causes at one checkpoint. On exhaustion the trial fails. An interruption also
   stops the checkpoint; the runner does not launch replacement calls on restart.

Continuation history exists only inside the current decision. The next TASO
checkpoint starts with its new compact graph/feedback and does not replay the
previous checkpoint's long reasoning.

## Artifacts and interpretation

Each `checkpoints/expansion_XXXX/` contains:

- Existing graph, feedback, compact prompt, progress and strategy artifacts.
- `attempts/attempt_01/` through at most `attempt_04/`, each with exact API request,
  returned API response when available, and inference metadata. Keys/headers are
  not saved. An intent record is persisted before contacting the provider.
- `attempts.json`, `generation.json` and trial `checkpoints.json` record every
  attempt, action (`initial`, `continue_after_length`, `retry_strategy`, `retry_provider`), failure kind, retry reason, replay
  mode, guarded input estimate, sent output cap, provider and status.
- Checkpoint and trial inference metadata sum **all returned usage**, including
  truncated/provider-failed requests. `usage_missing_attempts` flags unknown
  usage; reported totals do not pretend unknown attempts cost zero.

`model_calls.csv` contains one row per API attempt, including checkpoint expansion,
attempt number, status, finish reason, sent cap, prompt/output/reasoning tokens,
cost and latency. `trials.csv`/`summary.json` include total attempts, continuation
strategy-retry and provider-retry counts, first-attempt successes, recovered checkpoint counts
and failed checkpoint decisions. Historical records without new fields remain
readable. Count recovered graph-quality results under this new retry protocol,
and retain first-attempt reliability and all extra token/time costs separately.

`optimization_wall_seconds` includes callbacks, continuation work and retry waits.
`inference_seconds` sums request latency. `retry_wait_seconds` is recorded
separately; do not add any of these again to optimizer wall time. The legacy
inference-plus-search metric omits other callback overhead; an added
`llm_inference_search_and_retry_wait_seconds` includes backoff but still omits
other callback work. None includes setup/warm-up/model construction/final GPU
measurement. Failed trials are not silently included as successful graph-quality
observations, and B still limits graph expansions, not LLM tokens or API calls.

## Sources

- [OpenRouter reasoning preservation](https://github.com/OpenRouterTeam/docs/blob/main/guides/best-practices/reasoning-tokens.mdx#preserving-reasoning)
- [Provider routing and exact endpoint slugs](https://openrouter.ai/docs/guides/routing/provider-selection)
- [DeepSeek thinking and non-tool history](https://api-docs.deepseek.com/guides/thinking_mode/)
- [DeepSeek provider limits](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-pro-20260423/endpoints)
- [Kimi provider limits](https://openrouter.ai/api/v1/models/moonshotai/kimi-k3.1-pro/endpoints)
- [Codex provider limits](https://openrouter.ai/api/v1/models/openai/gpt-5.3-codex/endpoints)
- [Flash provider limits](https://openrouter.ai/api/v1/models/google/gemini-3.8-flash/endpoints)

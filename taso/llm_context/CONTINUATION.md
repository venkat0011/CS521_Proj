# Bounded checkpoint continuation

The new `checkpoint_continuation.json` preset implements a separate, recorded
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

## Initial output allowances and routing

The checked provider-specific snapshot is in `checkpoint_completion.py`, dated
2026-11-03. The new protocol pins these standard endpoints, with fallback
disabled, so a lower-limit endpoint does not silently replace one and opaque
reasoning is replayed through the same provider:

| Model | Provider slug | Advertised output cap | Context | Input cap if separate |
|---|---|---:|---:|---:|
| DeepSeek V4 Pro | gmicloud/fp8 | 943718 | 1048576 | — |
| Kimi K3.1 Pro | moonshotai/mxfp4 | 943718 | 1048576 | — |
| GPT-5.3-Codex | openai | 128000 | 400000 | 272000 |
| Gemini 3.8 Flash | google-ai-studio | 65536 | 1048576 | — |

These differ from default routing in historical runs. Provider availability,
prices and advertised limits can change. The plan and per-attempt artifacts
record these choices, and source fingerprints freeze the snapshot for a batch.
Unsupported/unavailable endpoint or parameter errors are reported; the code does
not switch provider/model, change reasoning or silently lower an advertised cap.

The request cap is the minimum of the checked endpoint maximum, guarded remaining
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

1. A normal `stop` returns one complete JSON strategy to the existing validator.
   A malformed strategy fails validation without a paid repair call.
2. A `length` response is saved, including usage and all returned response data.
   If follow-ups remain, reusable work is appended to the same decision's history
   and a new user message asks for one **complete replacement strategy JSON**.
   Partial JSON fragments are never blindly concatenated or executed.
3. `reasoning_details` are replayed unchanged, including order, signatures and
   opaque/encrypted data, through the pinned provider. Raw `reasoning` strings
   are replayed where applicable. For DeepSeek's readable non-tool reasoning,
   the text is passed explicitly as unfinished-work context because the native
   API can ignore old reasoning fields on ordinary turns.
4. This reuses returned work; it does **not** promise exact resumption of hidden
   reasoning. Models/providers may reason again or reject unsupported replay.
   Unreturned reasoning cannot be recovered. No reusable work or missing token
   usage causes an early stop rather than a blind repeated request.
5. HTTP/provider errors 429, 500, 502, 503 and 504 can retry the same current
   history with bounded backoff. `Retry-After` is respected up to 60 seconds;
   a longer requested wait stops the checkpoint. Authentication, configuration,
   context, content-filter and unsupported-format errors are terminal.
6. After three follow-ups, failure stays a failure. An interruption also stops
   the checkpoint; the runner will not launch replacement calls after restart.

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
  attempt, action (`initial`, `continue_after_length`, `retry_provider`), replay
  mode, guarded input estimate, sent output cap, provider and status.
- Checkpoint and trial inference metadata sum **all returned usage**, including
  truncated/provider-failed requests. `usage_missing_attempts` flags unknown
  usage; reported totals do not pretend unknown attempts cost zero.

`model_calls.csv` contains one row per API attempt, including checkpoint expansion,
attempt number, status, finish reason, sent cap, prompt/output/reasoning tokens,
cost and latency. `trials.csv`/`summary.json` include total attempts, continuation
and provider-retry counts, first-attempt successes, recovered checkpoint counts
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

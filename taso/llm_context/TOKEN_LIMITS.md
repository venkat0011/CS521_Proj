# Output limits and per-call usage

The separate [bounded checkpoint continuation protocol](CONTINUATION.md) adds
`--max-tokens max` with configured model-level ceilings and up to three follow-ups.
OpenRouter selects providers automatically; missing/invalid strategies can also
use that shared follow-up allowance. The provider-default mode below retains its
original no-retry behavior.

Use `--max-tokens provider` with `taso/run_experiment.py`, the suite runner or the
standalone one-shot runner to omit `max_tokens` entirely from OpenRouter requests.
This removes the application's output cap. It does not mean unlimited generation:
OpenRouter/provider defaults, model output/context limits and existing timeouts
still apply. A default can be lower than an explicit cap. Omission therefore does
not guarantee a larger budget or successful completion.

The default remains 32768 for the suite and 8192 for standalone one-shot runs.
An explicit positive integer such as `--max-tokens 65536` still sets that cap.
A suite configuration can use `"max_tokens": null` for provider-default mode;
the string `provider` is a CLI value, not a JSON configuration value.

Reasoning and temperature profiles, compact prompts, TASO search budgets, native
bindings and retry behaviour are unchanged. Output limits include reasoning and
final strategy tokens on the relevant reasoning endpoints. A `length` finish
remains a failed request even if our cap was omitted. No automatic paid retries,
strategy fallbacks or continuation calls are added.

## Full checkpoint run

After applying the update and pulling it in Colab, run from the project root:

```bash
python3 taso/run_experiment.py --experiment suite \
  --suite-config taso/benchmarks/config.json \
  --checkpoint-interval 25 \
  --max-tokens provider \
  --batch-dir "/content/CS521_Proj/taso/results/checkpoint/checkpoint_25/provider_default_$(date -u +%Y%m%dT%H%M%SZ)"
```

This uses 3 networks, 4 LLMs, 5 trials, B=100, and at most 4 calls per trial.
A fresh child directory preserves the existing Drive symlink and old results.
Changing the source/configuration requires a new batch under the existing
manifest contract. Already attempted checkpoint searches cannot resume their
native queue after worker exit.

## Saved usage

- Each request's `api_request.json` proves whether `max_tokens` was sent.
- `api_response.json` retains the returned usage and finish reason.
- `inference_metadata.json` records `max_tokens` (null when omitted),
  `output_token_limit_mode`, `finish_reason`, `native_finish_reason`, latency,
  usage and provider. `visible_completion_tokens` is completion minus reasoning
  tokens when both are reported; it is null when that calculation is unknown.
- Checkpoint aggregate metadata sums returned usage, including failed calls,
  and includes `finish_reason_counts` and total known non-reasoning output.
- `model_calls.csv` contains one row per observed model-call record: checkpoint
  expansion, status, finish reason, input/output/reasoning tokens, cost and latency.
  Missing provider usage is blank rather than invented. Historical records may
  lack newly added fields. The file also supports one-shot suite requests.
- `trials.csv` retains trial-level totals plus output-limit mode, requested cap,
  known non-reasoning output and length-finish count. `summary.json` retains
  complete trial distributions; `summary.md` adds a per-network/model usage table.

Do not sum the aggregate row and its checkpoint rows: they describe the same
requests at different levels. Failed requests can consume billable tokens; they
remain failures when comparing graph quality. Different output-limit settings
should be identified as different experimental configurations.

References: [OpenRouter parameters](https://openrouter.ai/docs/api_reference/parameters)
and [reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

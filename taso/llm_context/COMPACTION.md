# Compact checkpoint prompts

Output caps are independent of input compaction. See
[output limits and per-call usage](TOKEN_LIMITS.md) for `--max-tokens provider`.

`compact_checkpoint_prompt.py` generates `checkpoint_compact_v1` using only the
Python standard library. It does not contact an LLM, tokenize with a proxy model,
select a subset of rules or execute GPU search.

The checkpoint strategist caches this representation once per trial process and
uses `checkpoint_compact_prompt_template.txt` for every reached policy callback.
The initial call and later feedback calls all receive the complete compact rule
catalog. The old one-shot builder/template and the old full checkpoint template
remain available; the one-shot path is unchanged.

## What is retained

- All 18 built-in and 133 protobuf rule IDs. No applicability-based filtering.
- Every protobuf source/destination operator, input reference, parameter value
  and output-port mapping. Redundant numeric enum codes are saved separately.
- Exact C++ definitions of the five built-in factories and all `OpX* create_*`
  constraint helpers, including overloads. These remain source-backed code,
  rather than a model-generated semantic summary.
- The actual model-construction source.
- Every live graph node/edge, exposed attribute, tensor shape, output and warning.
  Shared input/output shapes are stored once; an exceptional input shape has an
  explicit override. Internal Enlarge/MergeGConv/constant nodes are retained.
- Current search state, global cumulative counters and all active-rule statistics
  from the last interval. Native rule indices become catalog ID strings.

Static catalog and graph records are expanded back and checked against the
original data before use. New/unhandled rule facts or graph fields fail instead
of silently disappearing. The compact search contract is tied to checked source
hashes for the search loop, matching and admission code and the native header.
If those implementations change, review the contract and update its pins.

The full native search/matching source dump is replaced by a short execution
contract. The five factory definitions and constructor helpers remain because
those encode rule constraints. All complete source and progress artifacts stay
available on disk. Native applicability checks are still authoritative.

## Bounded feedback

The model gets the latest four policy decisions, the current best graph,
improvement since the last callback, remaining budget, global cumulative counters
and the last interval's rule statistics. Earlier policies remain in the full
`feedback.json` and `checkpoints.json`, but are not repeated in the prompt.
All-zero statistic columns are omitted explicitly; no active-rule rows are cut.
This bounds policy-history growth at intervals such as 10 or 5.

The current client makes independent requests. Each call includes the compact
base again; sending it once does not create persistent model memory. Provider
prefix caching is not assumed.

## Inspect without a GPU/API call

From the project root:

```bash
python3 taso/llm_context/compact_checkpoint_prompt.py \
  --repo taso/TASO --out taso/results/compaction_preview
```

Or compact an existing frozen context and saved graph/feedback:

```bash
python3 taso/llm_context/compact_checkpoint_prompt.py \
  --context /path/to/batch/prepared/squeezenet/static_context.json \
  --graph /path/to/trial/checkpoints/expansion_0025/graph.json \
  --feedback /path/to/trial/checkpoints/expansion_0025/feedback.json \
  --out taso/results/compaction_preview
```

The static script saves `compact_static_context.json` and
`compaction_report.json`. Optional inputs also produce `graph_compact.json` and
`feedback_compact.json`. Static byte reduction is approximately 76% for the
current checkout. This is **UTF-8 bytes, not an exact model-token estimate**.

## Integrated run

Use a fresh batch subfolder under the existing Drive-backed link:

```bash
python3 taso/run_experiment.py --experiment suite \
  --suite-config taso/benchmarks/checkpoint_smoke.json \
  --checkpoint-interval 25 \
  --batch-dir /content/CS521_Proj/taso/results/checkpoint/checkpoint_25/compact_smoke_v1 \
  --dry-run
```

This tests SqueezeNet/Flash using all-rule preflight callbacks, with zero model
calls. Inspect the exact compact `request.json` under the prepared network's
`checkpoint_preflight/checkpoints/expansion_XXXX/`. To test DeepSeek instead, add
`--model deepseek/deepseek-v4-pro` to both commands. Remove `--dry-run` only for a
live trial; the smoke preset then allows at most four model calls.

To run the normal three-network/five-trial study, omit the smoke `--suite-config`
and use a separate batch subfolder. Model parameters, total expansion budget,
checkpoint interval behavior and output limit are unchanged by compaction.

## Saved evidence

- Trial root: `compact_static_context.json` includes all rule records, enum code
  metadata, full-source fingerprints and prompt version.
- Per checkpoint: original `graph.json`/`feedback.json`, compact
  `graph_compact.json`/`feedback_compact.json`, actual `request.json`, and
  `prompt_compaction.json` with component byte sizes and reduction fraction.
- `request_metadata.json` also contains these size measurements and hashes.
- Aggregate inference metadata/CSV include `prompt_version` and
  `prompt_bytes_total` for attempted requests. Provider usage gives actual tokens.

Artifacts are prettified for inspection; the prompt contains compact JSON. Size
comparisons exclude old/new template framing and explicitly say so. The actual
message-byte measurement is recorded separately.

Compaction reduces input volume; it does not reserve output tokens for JSON or
guarantee that a high-reasoning model will finish within 32,768 output tokens.
The existing `--max-tokens` option remains available. There are no hidden retries
or extra calls to summarize/repair a response.

Completed one-shot results remain valid historical observations and are not
rerun. Record this prompt version when comparing them with checkpoint results:
prompt representation is an additional experimental difference alongside feedback.

# TASO LLM input pipeline

The runtime entry point is `llm_strategy.request_strategy`. The extraction
machinery is isolated in `prepare_context.py` and does not run during experiments.

| File | Purpose | When used |
|---|---|---|
| prepare_context.py | Extract the registered rewrites and search source | Once per checkout change |
| prepared/static_context.json | Saved rules and source | Read during each request |
| graph_context.py | Read operators, shapes, edges and exposed attributes | Per workload |
| llm_strategy.py | Build messages, call a supplied model adapter, validate rule IDs | Per strategy request |

## Prepare once

Run from this directory:

```sh
python prepare_context.py
```

Paths default to the sibling `../TASO` checkout and local `prepared` folder.
The current checkout has 151 registrations: 18 C++ and 133 protobuf entries.
Those are rule instances, not necessarily unique identities or applicable
transformations. Extraction targets the inspected registration layout; review
its built-in expansion if changing registration. Regenerate the context after
changing source or rewrite files.

## Use in an experiment

With this directory on Python's import path and TASO installed:

```python
from llm_strategy import request_strategy

# graph is the live TASO graph you constructed, before calling optimize.
# call_llm is your provider-specific function: messages -> JSON response string.
strategy = request_strategy(graph, call_llm, save_to="runs/first_request")
print(strategy)
```

The model/provider adapter is intentionally supplied by the experiment.
No provider has been chosen or configured. There is no actual provider SDK
integration yet. The callable receives normal system/user message dictionaries
and must return the model's JSON text. Unknown/duplicate rule IDs and malformed
responses are rejected. Validated proposals do not execute in TASO yet.

To inspect the request before making an API call:

```python
from llm_strategy import build_messages
messages, catalog = build_messages(graph)
print(messages[0]["content"])
print(len(messages[1]["content"]))
```

## Current limits

- Live graph extraction requires a working TASO installation. Local checks use
  a fake graph fixture and do not establish compatibility with a live runtime.
- Existing getters omit fused activation and some other attributes. The graph
  observation records this warning; it is not a lossless serialization. The
  placeholder axes getter is omitted. Weight values are not sent.
- The complete catalog and source context are large. Check the chosen model's
  token budget before sending; bytes are not token counts.
- Static data is loaded from disk for each request, but not re-extracted. It
  must still be sent to a stateless provider unless a provider caching/session
  mechanism is added.
- C++ search control is a later step. This pipeline proposes existing rule IDs;
  it does not apply them to TASO.

The old combined build_context.py and placeholder prompt/context files were
replaced by these modules and the prepared static context.

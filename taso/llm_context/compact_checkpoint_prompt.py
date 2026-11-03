"""Deterministic checkpoint prompt compaction. Standard library only; no LLM calls.

Run on a Mac or Colab to prepare/inspect the static representation:
  python3 taso/llm_context/compact_checkpoint_prompt.py --repo taso/TASO --out /tmp/compact
Optional --context uses an existing static_context.json; --graph/--feedback compact
saved checkpoint inputs. Sizes are UTF-8 bytes, not model-token estimates.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
from string import Template

from prepare_context import function, prepare_context

HERE = Path(__file__).resolve().parent
PROMPT_VERSION = 'checkpoint_compact_v1'
RECENT_POLICIES = 4
# The short contract below was checked against these actual implementation bodies.
# Source changes require reviewing the contract, not silently summarizing new code.
CONTRACT_HASHES = {
    'Graph* Graph::optimize_search(': '14b093339dd7d9637f8b86c8a1a1b7b469269bb456049515ddcd80c0510f4269',
    'void GraphXfer::run(': '172b9be4121ed1249d223c512b90102e8feb051804a32cc6e1a161155a65de9f',
    'bool GraphXfer::can_match(': '26ec7351fa841160b3dbcf67f6a579e23afa4dc5e8deb5f313abbf3dd52e2322',
}
HEADER_HASH = '155b88b2631fd189a9f83c64f8364ef44cd455dd6d97f493b49129775cb2f3e7'
SEARCH_CONTRACT = (
    'TASO orders a persistent priority queue of whole graphs by estimated cost. '
    'One expansion pops one graph and tries every enabled rule at every valid match. '
    'Matching checks types, input wiring, parameter/input constraints and distinct matched operators. '
    'Replacement operators must be constructible; every externally consumed replaced output must be mapped. '
    'Candidates must be acyclic, pass native correctness checks, have estimated cost strictly below '
    'alpha times the best expanded-graph cost, fewer than twice the initial number of graph nodes, '
    'and a previously unseen graph hash. Queue, visited hashes, best graph and cost cache persist. '
    'The best found graph includes a better candidate generated on the last expansion. '
    'Skipped rules are not replayed on already expanded graphs. Queue exhaustion stops early; '
    'there is no restart, fallback or extra early policy call. Return one rule-selection phase '
    'for the next exact chunk; alpha, ranking, admission checks and rewrites cannot be changed. '
    'The supplied graph is the best found graph, possibly unexpanded, not a full frontier view. '
    'Fewer rules can lose beneficial rewrite paths. Native applicability checks remain authoritative. '
    'All cost feedback is estimated; final GPU latency is measured separately and is not supplied here.'
)
CATALOG_LEGEND = (
    'All 151 rules remain selectable. builtin rows=[rule_id,factory,arguments_after_model]; '
    'their exact C++ factory definitions and constraint-construction helpers follow. '
    'Factory new_tensor() denotes a shared external tensor; srcOps/dstOps define pattern order, '
    'outputs[i] denotes an output port and map_output connects source to replacement outputs. '
    'Helper isSrcOp defaults to true unless explicitly false. IN_i selects input i; DIM_j selects dimension j and DIM_ND means rank. '
    'protobuf rows=[rule_id,source_ops,destination_ops,output_maps]. '
    'Each op=[native_type_name,input_refs,parameters], each input_ref=[operator_id,output_port], '
    'each parameter=[native_parameter_name,value]. Negative operator_id is a shared external input; '
    'nonnegative IDs index operators within the respective source/destination pattern. '
    'Each output_map=[source_operator,source_port,destination_operator,destination_port]. '
    'PM_PAD: 0=SAME,1=VALID. PM_ACTI: 0=NONE,1=SIGMOID,2=RELU,3=TANH. '
    'Graph Conv corresponds to OP_CONV2D; graph Mul is OP_EW_MUL and ScalarMul is OP_MUL. '
    'Protobuf patterns are literal decoded data. Helpers include additional rank/shape constraints; '
    'do not infer that catalog membership guarantees a match. Only emit catalog rule_id strings.'
)
GRAPH_LEGEND = (
    'nodes=[id,type,attributes,output_shapes]; edges=[source,source_port,destination,destination_port]. '
    'An optional fifth edge item overrides its input shape; otherwise use the source output shape. '
    'external_sources marks edges whose source was outside the native operator list. '
    'outputs=[node,port]. All nodes, edges, exposed attributes, shapes and warnings are retained.'
)


def encoded(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def byte_size(value):
    return len(encoded(value).encode())


def _check_contract(static):
    excerpts = [s for item in static['source_context'] for s in item.get('excerpts', [])]
    for signature, expected in CONTRACT_HASHES.items():
        matches = [s for s in excerpts if s.startswith(signature)]
        if len(matches) != 1 or hashlib.sha256(matches[0].encode()).hexdigest() != expected:
            raise ValueError('Review compact search contract for changed/missing source: ' + signature)
    if static.get('source_inputs_sha256', {}).get('include/taso/ops.h') != HEADER_HASH:
        raise ValueError('Review compact enum legend against the changed/missing TASO header')


def expand_rules(compact):
    """Reconstruct all original catalog rule records, including native numeric codes."""
    result = []
    for rule_id, factory, arguments in compact['catalog']['builtin']:
        result.append({'rule_id': rule_id, 'taso_xfer_index': int(rule_id.split(':')[1]),
                       'origin': 'cpp', 'factory': factory, 'arguments_after_model': arguments,
                       'definition_reference': factory})
    types, params = compact['numeric_codes']['types'], compact['numeric_codes']['parameters']
    def operators(rows):
        return [{'type': types[kind], 'type_name': kind,
                 'inputs': [{'operator_id': ref[0], 'output_index': ref[1]} for ref in inputs],
                 'parameters': [{'key': params[name], 'key_name': name, 'value': value}
                                for name, value in attributes]} for kind, inputs, attributes in rows]
    for rule_id, source, destination, mappings in compact['catalog']['protobuf']:
        result.append({'rule_id': rule_id, 'taso_xfer_index': 18 + int(rule_id.split(':')[1]),
            'origin': 'graph_subst.pb', 'source': operators(source), 'destination': operators(destination),
            'output_mapping': [{'source_operator': a, 'source_output': b,
                                'destination_operator': c, 'destination_output': d} for a, b, c, d in mappings]})
    return result


def compact_static_context(static):
    _check_contract(static)
    catalog = static['catalog']
    if catalog.get('counts') != {'builtin': 18, 'protobuf': 133, 'total': 151}:
        raise ValueError('Expected the registered 18 built-in and 133 protobuf rules')
    full = [s['text'] for s in static['source_context']
            if s['path'] == 'src/core/substitution.cc' and 'text' in s]
    if len(full) != 1:
        raise ValueError('Full substitution source is required for exact factory/helper extraction')
    source = full[0]
    factories = copy.deepcopy(catalog['builtin_definitions'])
    for name, definition in factories.items():
        if definition != function(source, 'GraphXfer* GraphXfer::' + name + '('):
            raise ValueError('Built-in factory differs from supplied source: ' + name)
    helpers = defaultdict(list)
    for match in re.finditer(r'^OpX\* GraphXfer::(create_\w+)\(', source, re.M):
        helpers[match[1]].append(function(source[match.start():], match[0]))
    types, parameters, builtin, protobuf = {}, {}, [], []
    def remember(mapping, name, number):
        if name in mapping and mapping[name] != number:
            raise ValueError('Conflicting native enum: ' + name)
        mapping[name] = number
    def operators(ops):
        rows = []
        for op in ops:
            remember(types, op['type_name'], op['type'])
            attrs = []
            for item in op['parameters']:
                remember(parameters, item['key_name'], item['key'])
                attrs.append([item['key_name'], item['value']])
            rows.append([op['type_name'], [[i['operator_id'], i['output_index']] for i in op['inputs']], attrs])
        return rows
    for rule in catalog['rules']:
        if rule['origin'] == 'cpp':
            if rule['factory'] not in factories:
                raise ValueError('Missing factory definition: ' + rule['factory'])
            builtin.append([rule['rule_id'], rule['factory'], rule['arguments_after_model']])
        elif rule['origin'] == 'graph_subst.pb':
            protobuf.append([rule['rule_id'], operators(rule['source']), operators(rule['destination']),
                [[m['source_operator'], m['source_output'], m['destination_operator'], m['destination_output']]
                 for m in rule['output_mapping']]])
        else:
            raise ValueError('Unknown catalog origin')
    compact = {'prompt_version': PROMPT_VERSION, 'rule_count': len(catalog['rules']),
        'source_static_sha256': digest(static), 'source_inputs_sha256': static['source_inputs_sha256'],
        'numeric_codes': {'types': types, 'parameters': parameters},
        'catalog': {'builtin': builtin, 'protobuf': protobuf,
                    'factory_definitions': factories, 'constraint_helpers': dict(helpers)}}
    if expand_rules(compact) != catalog['rules']:
        raise ValueError('Catalog round-trip failed: a rule field, constraint or native index would be lost')
    ids = [row[0] for row in builtin + protobuf]
    if len(ids) != 151 or len(set(ids)) != len(ids):
        raise ValueError('Compact catalog lost or duplicated rule IDs')
    compact['rule_ids_sha256'] = digest(ids)
    return compact


def compact_graph(graph):
    nodes = {n['id']: n for n in graph['nodes']}
    if len(nodes) != len(graph['nodes']):
        raise ValueError('Duplicate graph node IDs')
    rows, external, seen_flags = [], set(), {}
    for node in graph['nodes']:
        if set(node) != {'id', 'type', 'attributes', 'output_shapes'}:
            raise ValueError('Graph node schema changed; review compaction')
    for edge in graph['edges']:
        if set(edge) != {'source', 'source_output', 'destination', 'destination_input', 'shape', 'external_source'}:
            raise ValueError('Graph edge schema changed; review compaction')
        src, port = edge['source'], edge['source_output']
        if src not in nodes or edge['destination'] not in nodes:
            raise ValueError('Graph edge references missing node')
        if src in seen_flags and seen_flags[src] != edge['external_source']:
            raise ValueError('Inconsistent external-source flags')
        seen_flags[src] = edge['external_source']
        if edge['external_source']:
            external.add(src)
        row = [src, port, edge['destination'], edge['destination_input']]
        if edge['shape'] != nodes[src]['output_shapes'][port]:
            row.append(edge['shape'])
        rows.append(row)
    for output in graph['outputs']:
        if set(output) != {'node', 'output_index'}:
            raise ValueError('Graph output schema changed; review compaction')
    return {'nodes': [[n['id'], n['type'], n['attributes'], n['output_shapes']] for n in graph['nodes']],
            'edges': rows, 'external_sources': sorted(external),
            'outputs': [[o['node'], o['output_index']] for o in graph['outputs']],
            'metadata': {k: v for k, v in graph.items() if k not in ('nodes', 'edges', 'outputs')}}


def expand_graph(compact):
    nodes = {row[0]: row for row in compact['nodes']}
    return {**compact['metadata'],
            'nodes': [{'id': a, 'type': b, 'attributes': c, 'output_shapes': d} for a, b, c, d in compact['nodes']],
            'edges': [{'source': r[0], 'source_output': r[1], 'destination': r[2], 'destination_input': r[3],
                       'shape': r[4] if len(r) == 5 else nodes[r[0]][3][r[1]],
                       'external_source': r[0] in compact['external_sources']} for r in compact['edges']],
            'outputs': [{'node': a, 'output_index': b} for a, b in compact['outputs']]}


def compact_feedback(feedback, catalog):
    ids = {r['taso_xfer_index']: r['rule_id'] for r in catalog['rules']}
    rows = feedback.get('last_interval_rules', [])
    metrics = sorted({k for row in rows for k, value in row.items() if k != 'rule_index' and value != 0})
    history = feedback.get('previous_decisions', [])
    recent = [{'at_expansion': r['at_expansion'], 'best_search_cost_ms': r.get('best_search_cost_ms'),
               'rule_ids': r['strategy']['phases'][0]['rule_ids'],
               'expansion_budget': r['strategy']['phases'][0]['expansion_budget']}
              for r in history[-RECENT_POLICIES:]]
    return {**{k: v for k, v in feedback.items() if k not in ('last_interval_rules', 'previous_decisions')},
            'last_rule_columns': ['rule_id'] + metrics,
            'last_rules': [[ids[r['rule_index']]] + [r.get(k, 0) for k in metrics] for r in rows],
            'previous_policy_count': len(history), 'recent_policies': recent,
            'history_note': f'Latest {RECENT_POLICIES} policies only; cumulative counters cover the entire search. Omitted rule-stat columns were zero for all active rules.'}


def build_checkpoint_messages(static, compact, observation, feedback, next_budget, alpha, model_source=None):
    from taso.strategy import compile_strategy
    if compact.get('prompt_version') != PROMPT_VERSION or compact['source_static_sha256'] != digest(static):
        raise ValueError('Compact catalog and source context differ; regenerate compaction')
    compile_strategy({'schema_version': 1, 'name': 'validation', 'phases': [
        {'name': 'next', 'rule_ids': 'all', 'expansion_budget': next_budget}]}, alpha)
    if next_budget != feedback['next_expansion_budget']:
        raise ValueError('Checkpoint feedback and next budget disagree')
    graph = compact_graph(observation)
    if expand_graph(graph) != observation:
        raise ValueError('Graph round-trip failed')
    compact_observation = compact_feedback(feedback, static['catalog'])
    objective = {'next_expansion_budget': next_budget, 'total_search_budget': feedback['total_budget'],
                 'expansions_consumed': feedback['expansions'], 'budget_remaining': feedback['budget_remaining'],
                 'alpha': alpha, 'quality_target': 'Minimize estimated graph cost; final GPU latency is evaluation only.'}
    use_gpu = 'gpu_feedback' in feedback
    contract = SEARCH_CONTRACT
    template = 'checkpoint_compact_prompt_template.txt'
    if use_gpu:
        template = 'checkpoint_gpu_prompt_template.txt'
        contract = SEARCH_CONTRACT.replace(
            'All cost feedback is estimated; final GPU latency is measured separately and is not supplied here.',
            'GPU feedback supplies measured snapshot-inference latency of the estimated-best graph at reached checkpoints. '
            'The queue, admission threshold and returned-graph selection still use estimated cost.')
        objective['quality_target'] = ('Seek low final measured GPU latency by selecting promising rules. '
            'Use both measured feedback and estimated costs; measurements cover only observed best graphs, '
            'and queue ranking still uses estimated costs.')
    model_source = model_source or 'Not supplied; use the actual live TASO graph.'
    content = Template((HERE / template).read_text()).substitute(
        contract=contract, catalog_legend=CATALOG_LEGEND, graph_legend=GRAPH_LEGEND,
        catalog=encoded(compact['catalog']), graph=encoded(graph), feedback=encoded(compact_observation),
        objective=encoded(objective), model_source=model_source)
    messages = [{'role': 'system', 'content': 'You select TASO rewrites for the NEXT checkpoint interval only. Return exact strategy JSON.'},
                {'role': 'user', 'content': content}]
    full_parts = {'catalog': byte_size(static['catalog']), 'search_source': byte_size(static['source_context']),
                  'graph': byte_size(observation), 'feedback': byte_size(feedback),
                  'model_source': len(model_source.encode())}
    compact_parts = {'catalog': byte_size(compact['catalog']),
                     'search_contract_and_legends': len((SEARCH_CONTRACT + CATALOG_LEGEND + GRAPH_LEGEND).encode()),
                     'graph': byte_size(graph), 'feedback': byte_size(compact_observation),
                     'model_source': len(model_source.encode())}
    full_size, compact_size = sum(full_parts.values()), sum(compact_parts.values())
    metrics = {'prompt_version': 'checkpoint_compact_gpu_v1' if use_gpu else PROMPT_VERSION, 'rule_count': compact['rule_count'],
        'rule_ids_sha256': compact['rule_ids_sha256'], 'messages_sha256': digest(messages),
        'messages_utf8_bytes': byte_size(messages), 'full_components_bytes': full_parts,
        'compact_components_bytes': compact_parts,
        'component_reduction_fraction': 1 - compact_size / full_size if full_size else 0,
        'measurement_note': 'UTF-8 bytes, not token estimates; component comparison excludes template framing. Provider usage reports actual tokens.'}
    return messages, {'graph': graph, 'feedback': compact_observation}, metrics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=HERE.parent / 'TASO')
    parser.add_argument('--context', type=Path, help='Existing full static_context.json')
    parser.add_argument('--graph', type=Path, help='Optional saved live graph.json')
    parser.add_argument('--feedback', type=Path, help='Optional saved feedback.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    context = args.context
    if context is None:
        prepare_context(args.repo, args.out / 'full')
        context = args.out / 'full/static_context.json'
    static = json.loads(context.read_text())
    compact = compact_static_context(static)
    write_json(args.out / 'compact_static_context.json', compact)
    full_size = byte_size(static['catalog']) + byte_size(static['source_context'])
    compact_size = byte_size(compact['catalog']) + len((SEARCH_CONTRACT + CATALOG_LEGEND).encode())
    report = {'prompt_version': PROMPT_VERSION, 'rule_count': compact['rule_count'],
              'catalog_round_trip': expand_rules(compact) == static['catalog']['rules'],
              'full_static_components_bytes': full_size, 'compact_static_components_bytes': compact_size,
              'static_reduction_fraction': 1 - compact_size / full_size,
              'measurement_unit': 'UTF-8 bytes, not tokens; no model requests made'}
    if args.graph:
        original = json.loads(args.graph.read_text())
        graph = compact_graph(original)
        write_json(args.out / 'graph_compact.json', graph)
        report.update(graph_round_trip=expand_graph(graph) == original,
                      full_graph_bytes=byte_size(original), compact_graph_bytes=byte_size(graph))
    if args.feedback:
        write_json(args.out / 'feedback_compact.json', compact_feedback(json.loads(args.feedback.read_text()), static['catalog']))
    write_json(args.out / 'compaction_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

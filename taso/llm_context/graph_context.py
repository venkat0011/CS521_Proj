"""Per-workload structural extraction from a live TASO PyGraph.

TASO is imported only when graph_context() is called.
"""

def graph_context(graph):
    """Read a live PyGraph without exporting weights or executing optimization."""
    import taso
    nodes, edges, warnings = [], [], []
    operators = graph.get_operator_list()
    ids = {int(op['guid']) for op in operators}
    external_nodes = {}
    for op in operators:
        guid = int(op['guid'])
        kind = graph.get_operator_type(op)
        attrs = {}
        for attr in taso.operator_attrs.get(kind, []):
            if attr == 'axes':
                warnings.append(f'{guid}: axes omitted; TASO getter contains a placeholder')
                continue
            attrs[attr] = graph.get_operator_attr(op, attr)
        if kind == 'Conv':
            try:
                mode = graph.get_operator_attr(op, 'activation')
                attrs['fused_activation_mode'] = mode
                attrs['fused_activation'] = {0: 'NONE', 1: 'SIGMOID', 2: 'RELU', 3: 'TANH'}.get(mode, 'UNKNOWN')
            except (AssertionError, AttributeError):
                warnings.append(f'{guid}: fused activation unavailable; rebuild the updated TASO bindings')
        nodes.append({'id': guid, 'type': kind, 'attributes': attrs,
                      'output_shapes': [graph.get_output_dims(op, i)
                                        for i in range(graph.get_num_outputs(op))]})
        for edge in graph.get_input_edges(op):
            src = int(edge['srcOp']['guid'])
            if src not in ids and src not in external_nodes:
                external_nodes[src] = {
                    'id': src, 'type': graph.get_operator_type(edge['srcOp']),
                    'attributes': {},
                    'output_shapes': [graph.get_output_dims(edge['srcOp'], i)
                                      for i in range(graph.get_num_outputs(edge['srcOp']))],
                }
            edges.append({'source': src, 'source_output': int(edge['srcIdx']),
                          'destination': guid, 'destination_input': int(edge['dstIdx']),
                          'shape': graph.get_input_dims(op, edge['dstIdx']),
                          'external_source': src not in ids})
    warnings.append('This uses public getters and is not a complete serialization; attributes without getters remain unavailable.')
    all_nodes = list(external_nodes.values()) + nodes
    consumed = {(edge['source'], edge['source_output']) for edge in edges}
    outputs = [{'node': node['id'], 'output_index': index}
               for node in all_nodes for index in range(len(node['output_shapes']))
               if (node['id'], index) not in consumed]
    return {'status': 'live_taso_graph', 'nodes': all_nodes, 'edges': edges,
            'outputs': outputs,
            'output_identification': 'Inferred from unconsumed tensor ports; explicit model-output annotations are unavailable.',
            'estimated_cost_ms': graph.cost(), 'warnings': warnings,
            'weight_values_included': False}

"""Restore supported archived inference graphs; require exact descriptor verification."""
import json
import math
from pathlib import Path
from candidate_identity import prepare, node_label, equivalent


def prepared_snapshot(snapshot, include_backend=False):
    nodes={n['guid']:dict(label=node_label(n),complete=n['complete']) for n in snapshot['nodes']}
    if include_backend:
        from candidate_identity import digest
        for n in snapshot['nodes']:nodes[n['guid']]['label']=digest([nodes[n['guid']]['label'],n.get('backend_algorithm')])
    return prepare(dict(nodes=list(nodes),edges=snapshot['edges']),nodes)


def restore(ts, path):
    import numpy as np
    from graph_archive import load_snapshot
    saved=load_snapshot(path);nodes={n['guid']:n for n in saved['nodes']}
    incoming={n:{} for n in nodes};consumers={n:[] for n in nodes}
    for s,sp,d,dp in saved['edges']:
        incoming[d][dp]=(s,sp);consumers[s].append(d)
    graph=ts.new_graph();outputs={};remaining=set(nodes)
    while remaining:
        ready=[n for n in sorted(remaining) if all(s in outputs for s,_ in incoming[n].values())]
        if not ready:raise ValueError('Archive is cyclic or contains unresolved dependencies')
        for gid in ready:
            node=nodes[gid];kind=node['type'];a=node['attributes'];edges=incoming[gid]
            def x(port=0):
                s,sp=edges[port];return outputs[s][sp]
            if kind==0:t=graph.audit_leaf(0,node['outputs'][0],[])
            elif kind==1:
                data=np.frombuffer(bytes.fromhex(node['weight_hex']),dtype=np.float32)
                t=graph.audit_leaf(1,node['outputs'][0],data)
            else:
                # Full archived state is authoritative. Independent native operators
                # preserve descriptors without cost-cache lookups or normalization.
                if sorted(edges)!=list(range(len(node['inputs']))):
                    raise ValueError('Missing/duplicate archived input ports')
                t=graph.audit_operator(kind,[x(port) for port in sorted(edges)],a,
                    node['inputs'],node['outputs'],node.get('backend_algorithm',-1))
            outputs[gid]=t if isinstance(t,list) else [t];remaining.remove(gid)
    # This archive is already folded. Preprocessing would rewrite the graph again.
    restored=json.loads(graph.audit_snapshot())
    if equivalent(prepared_snapshot(saved),prepared_snapshot(restored)) is not True:
        from collections import Counter
        from candidate_identity import node_label
        expected=Counter(node_label(n) for n in saved['nodes']);actual=Counter(node_label(n) for n in restored['nodes'])
        diagnostics=dict(archived_node_count=len(saved['nodes']),restored_node_count=len(restored['nodes']),
            archived_edge_count=len(saved['edges']),restored_edge_count=len(restored['edges']),
            missing_descriptor_count=sum((expected-actual).values()),extra_descriptor_count=sum((actual-expected).values()),
            missing_descriptors=[{k:v for k,v in n.items() if k not in ('weight_hex','weight_blob')} for n in saved['nodes'] if expected[node_label(n)]>actual[node_label(n)]][:12],
            extra_descriptors=[{k:v for k,v in n.items() if k not in ('weight_hex','weight_blob')} for n in restored['nodes'] if actual[node_label(n)]>expected[node_label(n)]][:12])
        Path(str(path)+'.restore_failure.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
        raise ValueError('Restored graph does not exactly match archived structure/settings/weights; see '+str(path)+'.restore_failure.json')
    # Algorithms are execution metadata, not computational graph identity.
    plan_match=equivalent(prepared_snapshot(saved,True),prepared_snapshot(restored,True))
    return graph,dict(exact_graph_restoration=True,exact_backend_plan_restoration=plan_match,import_mode="private_native_operators")


def terminal_shapes(snapshot):
    consumed={(s,sp) for s,sp,_,_ in snapshot['edges']}
    return [list(tensor[2:2+tensor[1]]) for node in snapshot['nodes']
            for port,tensor in enumerate(node['outputs']) if (node['guid'],port) not in consumed]

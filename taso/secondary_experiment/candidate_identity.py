"""GUID-independent candidate comparison; refinement is only an index, never proof."""
from collections import Counter, defaultdict, OrderedDict
import gzip
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def node_label(node):
    label = {k: node[k] for k in ('type', 'attributes', 'inputs', 'outputs', 'complete')}
    if node['type'] == 1:  # OP_WEIGHT; exact bytes, not a native address/hash.
        if 'weight_hex' in node:
            label['weight_sha256'] = hashlib.sha256(bytes.fromhex(node['weight_hex'])).hexdigest()
        elif 'weight_blob' in node:
            from graph_archive import weight_digest
            label['weight_sha256'] = weight_digest(node['weight_blob'])
        else:
            raise ValueError('Missing weight contents: cannot assert graph identity')
    return digest(label)


def prepare(record, nodes):
    ids = record['nodes']
    labels = {n: nodes[n]['label'] for n in ids}
    incoming, outgoing = defaultdict(list), defaultdict(list)
    edges = Counter(tuple(e) for e in record['edges'])
    for (src, sp, dst, dp), count in edges.items():
        if src not in labels or dst not in labels:
            raise ValueError('Candidate contains an edge to an unrecorded operator')
        incoming[dst].extend([(src, sp, dp)] * count)
        outgoing[src].extend([(dst, sp, dp)] * count)
    pairs=defaultdict(Counter)
    for (src,sp,dst,dp),count in edges.items(): pairs[src,dst][sp,dp] += count
    colors = labels
    for _ in range(3):
        colors = {n: digest([colors[n], sorted((colors[s], sp, dp) for s, sp, dp in incoming[n]),
                            sorted((colors[d], sp, dp) for d, sp, dp in outgoing[n])]) for n in ids}
    token = digest([sorted(colors.values()), sorted(
        (colors[s], sp, colors[d], dp, count) for (s, sp, d, dp), count in edges.items())])
    return dict(record=record, labels=labels, colors=colors, incoming=incoming,
                outgoing=outgoing, edges=edges, pairs=pairs, token=token,
                complete=all(nodes[n]['complete'] for n in ids))


def equivalent(a, b, max_steps=100000):
    """Exact labeled directed multigraph isomorphism; None means not established."""
    if not a['complete'] or not b['complete']:
        return None
    if a['token'] != b['token']:
        return False
    candidates = {n: [m for m in b['labels'] if a['colors'][n] == b['colors'][m]] for n in a['labels']}
    mapping, used = {}, set()
    steps = 0
    def between(graph, x, y):
        return graph['pairs'].get((x,y), Counter())
    def visit():
        nonlocal steps
        steps += 1
        if steps > max_steps:
            raise TimeoutError('Exact graph comparison reached its work limit')
        if len(mapping) == len(candidates):
            transformed = Counter((mapping[s], sp, mapping[d], dp) for (s, sp, d, dp), count
                                  in a['edges'].items() for _ in range(count))
            return transformed == b['edges']
        remaining = [n for n in candidates if n not in mapping]
        n = min(remaining, key=lambda x: (
            len([m for m in candidates[x] if m not in used]),
            -sum(y in mapping for y, _, _ in a['incoming'][x] + a['outgoing'][x])))
        for m in candidates[n]:
            if m in used or a['labels'][n] != b['labels'][m]:
                continue
            if between(a, n, n) != between(b, m, m):
                continue
            if any(between(a, n, x) != between(b, m, y) or
                   between(a, x, n) != between(b, y, m) for x, y in mapping.items()):
                continue
            mapping[n] = m; used.add(m)
            if visit():
                return True
            del mapping[n]; used.remove(m)
        return False
    try:
        return visit()
    except TimeoutError:
        return None


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.nodes, self.graphs, self.events = {}, {}, []
        opener = gzip.open if self.path.suffix == '.gz' else open
        with opener(self.path, 'rt') as f:
            for line in f:
                row = json.loads(line)
                if row['event'] == 'node':
                    node = row['node']
                    self.nodes[node['guid']] = dict(label=node_label(node), complete=node['complete'])
                elif row['event'] == 'graph':
                    if row['id'] in self.graphs:
                        raise ValueError('Duplicate graph identifier')
                    self.graphs[row['id']] = row
                else:
                    if row['graph'] not in self.graphs:
                        raise ValueError('Event references an unknown graph')
                    self.events.append(row)
        self._cache = OrderedDict()
        self.index = defaultdict(list)
        for gid in self.graphs:
            graph = self.get(gid)
            self.index[graph['token']].append(gid)
        self._cache.clear()
    def get(self, gid):
        if gid not in self._cache:
            self._cache[gid] = prepare(self.graphs[gid], self.nodes)
            if len(self._cache)>128:self._cache.popitem(last=False)
        self._cache.move_to_end(gid)
        return self._cache[gid]
    def find(self, graph, allowed):
        unknown = False
        for gid in self.index.get(graph['token'], []):
            if gid not in allowed:
                continue
            result = equivalent(graph, self.get(gid))
            if result is True:
                return dict(status='matched', graph=gid)
            unknown |= result is None
        return dict(status='unknown' if unknown or not graph['complete'] else 'not_found')
    def ids(self, event, budget=None):
        return {e['graph'] for e in self.events if e['event'] == event and
                (budget is None or e['expansion'] < budget)}
    def last(self, event):
        return next(e for e in reversed(self.events) if e['event'] == event)
    def lineage(self, gid):
        parents = {e['graph']: e for e in self.events if e['event'] == 'generated'}
        path, seen = [], set()
        while gid in parents:
            if gid in seen:
                raise ValueError('Cycle in generation provenance')
            seen.add(gid); e = parents[gid]; path.append(e); gid = e['parent']
        return list(reversed(path))

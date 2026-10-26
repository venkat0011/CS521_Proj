"""One-time rule and source extraction; run again when the TASO checkout changes."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def uncomment(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)


def function(text, signature):
    start = text.index(signature)
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def enum_names(header, name):
    body = re.search(r'enum\s+' + name + r'\s*\{(.*?)\}', uncomment(header), re.S)[1]
    result, value = {}, 0
    for item in body.split(','):
        item = item.strip()
        if not item:
            continue
        parts = item.split('=')
        if len(parts) == 2:
            value = int(parts[1].strip(), 0)
        result[value] = parts[0].strip()
        value += 1
    return result


def varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError('Truncated protobuf varint')
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError('Invalid protobuf varint')


# Minimal decoder for the checked-in rules.proto: int32 and nested messages.
# Rejects other wire types rather than silently guessing a changed schema.
SCHEMA = {
    'collection': {1: ('rules', 'rule', True)},
    'rule': {1: ('source', 'operator', True), 2: ('destination', 'operator', True),
             3: ('output_mapping', 'mapping', True)},
    'operator': {1: ('type', None, False), 2: ('inputs', 'tensor', True),
                 3: ('parameters', 'parameter', True)},
    'tensor': {1: ('operator_id', None, False), 2: ('output_index', None, False)},
    'parameter': {1: ('key', None, False), 2: ('value', None, False)},
    'mapping': {1: ('source_operator', None, False), 2: ('destination_operator', None, False),
                3: ('source_output', None, False), 4: ('destination_output', None, False)},
}


def decode(data, kind):
    schema = SCHEMA[kind]
    result = {name: [] for name, nested, repeated in schema.values() if repeated}
    pos = 0
    while pos < len(data):
        tag, pos = varint(data, pos)
        number, wire = tag >> 3, tag & 7
        if number not in schema:
            raise ValueError(f'Unexpected field {number} in {kind}')
        name, nested, repeated = schema[number]
        if nested:
            if wire != 2:
                raise ValueError('Expected nested protobuf message')
            size, pos = varint(data, pos)
            end = pos + size
            if end > len(data):
                raise ValueError('Truncated protobuf message')
            value = decode(data[pos:end], nested)
            pos = end
        else:
            if wire != 0:
                raise ValueError('Expected int32 protobuf field')
            value, pos = varint(data, pos)
            value &= 0xffffffff
            if value >= 0x80000000:
                value -= 0x100000000
            if name in result:
                raise ValueError('Duplicate singular protobuf field')
        if repeated:
            result[name].append(value)
        else:
            result[name] = value
    missing = [name for name, _, repeated in schema.values() if not repeated and name not in result]
    if missing:
        raise ValueError(f'Missing required fields: {missing}')
    return result


def rule_catalog(repo):
    ops = (repo / 'src/core/ops.cc').read_text()
    subst = (repo / 'src/core/substitution.cc').read_text()
    header = (repo / 'include/taso/ops.h').read_text()
    registration = function(ops, 'Graph* Graph::optimize_scheduled(').split('std::priority_queue')[0]
    active = uncomment(registration)
    expected = ['create_conv_relu', 'create_conv_batch', 'create_conv_mul',
                'create_enlarge_merge_convs', 'create_enlarge_merge_convs',
                'create_merge_group_convs', 'create_merge_group_convs',
                'create_merge_group_convs', 'create_merge_group_convs']
    found = re.findall(r'xfers.push_back\(GraphXfer::(\w+)\(', active)
    if found != expected or 'i < 3' not in active or 'j < 2' not in active:
        raise ValueError('Rule registration changed; update the built-in expansion first')
    calls = []
    for stride in (1, 2):
        for padding in ('PD_MODE_SAME', 'PD_MODE_VALID'):
            for name in ('create_conv_relu', 'create_conv_batch', 'create_conv_mul'):
                calls.append((name, [stride, stride, padding]))
    for activation in ('AC_MODE_NONE', 'AC_MODE_RELU'):
        calls.append(('create_enlarge_merge_convs', [activation]))
    for stride in (1, 2):
        for activation in ('AC_MODE_NONE', 'AC_MODE_RELU'):
            calls.append(('create_merge_group_convs', [stride, stride, activation]))
    rules = []
    for index, (name, args) in enumerate(calls):
        rules.append({'rule_id': f'builtin:{index:03}', 'taso_xfer_index': index,
                      'origin': 'cpp', 'factory': name, 'arguments_after_model': args,
                      'definition_reference': name})
    factories = {name: function(subst, 'GraphXfer* GraphXfer::' + name + '(')
                 for name, _ in calls}
    op_names = enum_names(header, 'OpType')
    param_names = enum_names(header, 'PMParameter')
    data = (repo / 'graph_subst.pb').read_bytes()
    for index, rule in enumerate(decode(data, 'collection')['rules']):
        for side in ('source', 'destination'):
            for op in rule[side]:
                op['type_name'] = op_names[op['type']]
                for param in op['parameters']:
                    param['key_name'] = param_names[param['key']]
                for tensor in op['inputs']:
                    oid = tensor['operator_id']
                    if oid >= 0 and oid >= len(rule[side]):
                        raise ValueError('Invalid rule operator reference')
        rules.append(dict(rule, rule_id=f'protobuf:{index:04}', origin='graph_subst.pb',
                          taso_xfer_index=len(calls) + index))
    return {'rules': rules, 'builtin_definitions': factories,
            'counts': {'builtin': len(calls), 'protobuf': len(rules) - len(calls), 'total': len(rules)},
            'registration_source': registration,
            'tensor_reference_convention': 'Negative operator_id denotes a shared external input; nonnegative IDs index the corresponding source/destination operator list.',
            'semantics_note': 'Protobuf patterns are decoded literally. C++ factories impose additional constraints; consult supplied factory and matching code. Catalog membership does not imply applicability to this graph.',
            'graph_subst_sha256': sha(data)}


def prepare_context(repo, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    catalog = rule_catalog(repo)
    selected = {
        'src/core/ops.cc': ['Graph* Graph::optimize(', 'Graph* Graph::optimize_scheduled(', 'float Graph::total_cost('],
        'src/core/substitution.cc': ['void GraphXfer::load_graph_xfer_from_pb_file(',
                                   'void GraphXfer::run(', 'Graph* GraphXfer::create_new_graph(',
                                   'bool GraphXfer::can_match('],
        'include/taso/substitution.h': ['class GraphCompare'],
    }
    sources = []
    for path, signatures in selected.items():
        text = (repo / path).read_text()
        sources.append({'path': path, 'sha256': sha(text.encode()),
                        'excerpts': [function(text, sig) for sig in signatures]})
    # These carry construction constraints used by both built-in and loaded rules.
    path = 'src/core/substitution.cc'
    text = (repo / path).read_text()
    sources.append({'path': path, 'sha256': sha(text.encode()),
                    'purpose': 'Full rule factory, constructor and matching semantics', 'text': text})
    context = {
        'schema_version': 1,
        'catalog': catalog,
        'source_context': sources,
        'limitations': [
            'Rule IDs identify registrations in this checkout; unmodified TASO cannot execute a strategy.',
            'Source includes search/rule code, not the whole repository.',
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'static_context.json'
    target.write_text(json.dumps(context, indent=2) + '\n')
    return {'path': str(target), 'counts': catalog['counts']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent.parent / 'TASO')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'prepared')
    args = parser.parse_args()
    print(json.dumps(prepare_context(args.repo, args.out), indent=2))

"""Expose native rewrite operators while extracting an intermediate checkpoint graph."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'llm_context'))
from graph_context import graph_context
from prepare_context import enum_names
import checkpoint_strategy

HEADER_ENUM = enum_names((ROOT / 'TASO/include/taso/ops.h').read_text(), 'OpType')
CORE_SOURCE = (ROOT / 'TASO/python/taso/_cython/core.pyx').read_text()
# Run the production table and getter logic against native enum values. Only the
# Cython type annotations and native graph boundary are replaced for this CPU test.
table = CORE_SOURCE[CORE_SOURCE.index('op_table = dict()'):CORE_SOURCE.index('cdef class PyGraph:')]
start = CORE_SOURCE.index('    def get_operator_type(self, Op op):')
stop = CORE_SOURCE.index('    def get_operator_attr(', start)
getter = CORE_SOURCE[start:stop].replace('self, Op op', 'self, op').replace('cdef OpType type', 'type')
namespace = {name: value for value, name in HEADER_ENUM.items()}
exec(table + '\nclass Binding:\n' + getter, namespace)
Binding = namespace['Binding']


class IntermediateGraph:
    def __init__(self, internal=True):
        self.nodes = {
            100: (1, (8, 3, 1, 1), []),
            101: (1, (8, 3, 3, 3), []),
            102: (20, (8, 3, 3, 3), [100, 101]),   # Enlarge a weight kernel.
            103: (21, (8, 6, 3, 3), [102]),        # Merge grouped-convolution weights.
            104: (1, (), []),
            105: (19, (8, 6, 3, 3), [103, 104]),   # Scalar multiply, distinct from EW_MUL.
            106: (22, (8, 8), []),
            107: (23, (8, 8, 3, 3), []),
            108: (24, (8,), []),
            109: (25, (8, 8, 3, 3), []),
        }
        self.internal = internal
        self.binding = Binding()
        self.binding.p_graph = types.SimpleNamespace(get_operator_type=lambda guid: self.nodes[guid][0])
    def get_operator_list(self):
        return [{'guid': guid} for guid in (102, 103, 105, 106, 107, 108, 109)] if self.internal else []
    def get_operator_type(self, op):
        return self.binding.get_operator_type(types.SimpleNamespace(guid=op['guid']))
    def get_num_outputs(self, op): return 1
    def get_output_dims(self, op, index): return list(self.nodes[op['guid']][1])
    def get_input_edges(self, op):
        return [{'srcOp': {'guid': src}, 'srcIdx': 0, 'dstIdx': index}
                for index, src in enumerate(self.nodes[op['guid']][2])]
    def get_input_dims(self, op, index):
        return list(self.nodes[self.nodes[op['guid']][2][index]][1])
    def cost(self): return .1


class OperatorIntrospectionTests(unittest.TestCase):
    def setUp(self):
        fake = types.ModuleType('taso')
        fake.operator_attrs = {}
        self.modules = patch.dict(sys.modules, {'taso': fake})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_cython_enum_names_match_the_native_header(self):
        pxd = (ROOT / 'TASO/python/taso/_cython/CCore.pxd').read_text()
        block = pxd.split('cdef enum OpType:', 1)[1].split('cdef enum PMParameter:', 1)[0]
        names = re.findall(r'^\s*(OP_\w+)\b', block, re.M)
        self.assertEqual(names, list(HEADER_ENUM.values()))

    def test_getter_names_every_declared_native_type_including_enlarge(self):
        binding = Binding()
        for raw, symbol in HEADER_ENUM.items():
            binding.p_graph = types.SimpleNamespace(get_operator_type=lambda guid, raw=raw: raw)
            with self.subTest(type=symbol):
                name = binding.get_operator_type(types.SimpleNamespace(guid=1))
                self.assertIsInstance(name, str)
                self.assertTrue(name)
                if raw == 20:
                    self.assertEqual(name, 'Enlarge')
        self.assertEqual(namespace['op_table'][21], 'MergeGConv')
        self.assertNotEqual(namespace['op_table'][19], namespace['op_table'][17])
        binding.p_graph = types.SimpleNamespace(get_operator_type=lambda guid: 999)
        with self.assertRaisesRegex(AssertionError, 'Undefined type: 999'):
            binding.get_operator_type(types.SimpleNamespace(guid=1))

    def test_intermediate_graph_keeps_rewrite_nodes_edges_and_shapes(self):
        snapshot = graph_context(IntermediateGraph())
        nodes = {node['id']: node for node in snapshot['nodes']}
        self.assertEqual(nodes[102]['type'], 'Enlarge')
        self.assertEqual(nodes[103]['type'], 'MergeGConv')
        self.assertEqual(nodes[105]['type'], 'ScalarMul')
        self.assertEqual(nodes[103]['output_shapes'], [[8, 6, 3, 3]])
        self.assertEqual(nodes[100]['output_shapes'], [[8, 3, 1, 1]])
        self.assertEqual(len(snapshot['edges']), 5)
        self.assertEqual(len(nodes), 10)
        json.dumps(snapshot, allow_nan=False)

    def test_checkpoint_preflight_extracts_internal_graph_without_model_calls(self):
        spec = importlib.util.spec_from_file_location(
            'taso.strategy', ROOT / 'TASO/python/taso/strategy.py')
        compiler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiler)
        cfg = json.loads((ROOT / 'benchmarks/checkpoint_smoke.json').read_text())
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'taso.strategy': compiler}):
            directory = Path(tmp)
            context = directory / 'context.json'
            from prepare_context import prepare_context
            prepare_context(ROOT / 'TASO', directory)
            context = directory / 'static_context.json'
            controller = checkpoint_strategy.CheckpointStrategist('google/gemini-3.8-flash', cfg,
                context, 'Actual model source', directory,
                client_factory=lambda *a, **kw: self.fail('Preflight made an API client'), dry_run=True)
            for consumed in (0, 25):
                progress = {'expansions': consumed, 'total_budget': 100, 'budget_remaining': 100-consumed,
                    'next_expansion_budget': 25, 'queue_remaining': 10, 'initial_cost_ms': .1,
                    'best_search_cost_ms': .1, 'phases': []}
                policy = controller(IntermediateGraph(internal=consumed > 0), progress)
                self.assertEqual(policy['phases'][0]['expansion_budget'], 25)
            self.assertEqual(controller.model_calls, 0)
            saved = json.loads((directory / 'checkpoints/expansion_0025/graph.json').read_text())
            self.assertIn('Enlarge', [node['type'] for node in saved['nodes']])


if __name__ == '__main__':
    unittest.main()

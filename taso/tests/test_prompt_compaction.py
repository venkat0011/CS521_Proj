"""Compact prompts retain every rule and graph fact while bounding history growth."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'llm_context'))
import compact_checkpoint_prompt as compact
import llm_strategy
from prepare_context import prepare_context
from graph_context import graph_context
from test_operator_introspection import IntermediateGraph

spec = importlib.util.spec_from_file_location('taso.strategy', ROOT / 'TASO/python/taso/strategy.py')
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


class CompactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            prepare_context(ROOT / 'TASO', tmp)
            cls.static = json.loads((Path(tmp) / 'static_context.json').read_text())
        cls.compacted = compact.compact_static_context(cls.static)

    def setUp(self):
        fake = types.ModuleType('taso')
        fake.operator_attrs = {}
        self.modules = patch.dict(sys.modules, {'taso': fake, 'taso.strategy': compiler})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def feedback(self, count=0):
        policy = {'schema_version': 1, 'name': 'fusion', 'phases': [
            {'name': 'fusion', 'rule_ids': ['builtin:000'], 'expansion_budget': 5}]}
        return {'expansions': count * 5, 'total_budget': 100, 'budget_remaining': 100-count*5,
            'next_expansion_budget': 5, 'queue_remaining': 20, 'initial_cost_ms': .1,
            'best_search_cost_ms': .08, 'improvement_since_previous_checkpoint_ms': .001,
            'cumulative_counters': {'attempts': count * 5, 'admitted': count * 10},
            'last_interval_rules': [{'rule_index': 0, 'attempts': 5, 'admitted': 10, 'duplicates': 0, 'seconds': .001}],
            'previous_decisions': [{'at_expansion': n*5, 'best_search_cost_ms': .1-n*.001,
                                    'strategy': copy.deepcopy(policy)} for n in range(count)],
            'feedback_type': 'estimated_cost_only; no GPU runtime observations'}

    def test_every_rule_round_trips_with_parameters_ports_and_native_codes(self):
        reconstructed = compact.expand_rules(self.compacted)
        self.assertEqual(reconstructed, self.static['catalog']['rules'])
        self.assertEqual(len(reconstructed), 151)
        self.assertEqual([r['rule_id'] for r in reconstructed], [r['rule_id'] for r in self.static['catalog']['rules']])
        self.assertEqual(self.compacted['catalog']['factory_definitions'], self.static['catalog']['builtin_definitions'])
        # Both overloads are needed: the two-input concat helper delegates to N-input concat.
        self.assertEqual(len(self.compacted['catalog']['constraint_helpers']['create_concat']), 2)
        self.assertTrue(any(m['destination_output'] > 0 for r in reconstructed if 'output_mapping' in r
                            for m in r['output_mapping']))

    def test_unknown_rule_fields_and_contract_changes_fail_before_a_request(self):
        unknown = copy.deepcopy(self.static)
        unknown['catalog']['rules'][20]['new_constraint'] = True
        with self.assertRaisesRegex(ValueError, 'round-trip'):
            compact.compact_static_context(unknown)
        changed = copy.deepcopy(self.static)
        changed['source_context'][0]['excerpts'] = [
            s+'\n// changed' if s.startswith('Graph* Graph::optimize_search(') else s
            for s in changed['source_context'][0]['excerpts']]
        with self.assertRaisesRegex(ValueError, 'Review compact search contract'):
            compact.compact_static_context(changed)
        missing = copy.deepcopy(self.static)
        missing['catalog']['rules'].pop()
        with self.assertRaises(ValueError):
            compact.compact_static_context(missing)

    def test_graph_round_trip_keeps_internal_nodes_and_shape_overrides(self):
        full = graph_context(IntermediateGraph())
        full['edges'][0]['shape'] = [99]  # Preserve an input-view shape differing from the source output.
        view = compact.compact_graph(full)
        self.assertEqual(compact.expand_graph(view), full)
        self.assertEqual(view['edges'][0][4], [99])
        self.assertEqual(len(view['nodes']), 10)
        self.assertEqual(len(view['edges']), 5)
        self.assertLess(compact.byte_size(view), compact.byte_size(full))
        broken = copy.deepcopy(full)
        broken['nodes'][0]['unhandled_fact'] = 'must not silently disappear'
        with self.assertRaises(ValueError):
            compact.compact_graph(broken)

    def test_feedback_history_is_bounded_without_changing_cumulative_counts(self):
        full = self.feedback(19)
        view = compact.compact_feedback(full, self.static['catalog'])
        self.assertEqual(view['previous_policy_count'], 19)
        self.assertEqual(len(view['recent_policies']), 4)
        self.assertEqual(view['recent_policies'][0]['at_expansion'], 75)
        self.assertEqual(view['cumulative_counters'], full['cumulative_counters'])
        self.assertEqual(view['last_rules'][0][0], 'builtin:000')
        self.assertNotIn('duplicates', view['last_rule_columns'])
        self.assertEqual(full['last_interval_rules'][0]['duplicates'], 0)
        self.assertLess(compact.byte_size(view), compact.byte_size(full))

    def test_checkpoint_prompt_is_smaller_complete_and_uses_next_interval_budget(self):
        observation = graph_context(IntermediateGraph())
        feedback = self.feedback(5)
        messages, artifacts, metrics = compact.build_checkpoint_messages(
            self.static, self.compacted, observation, feedback, 5, 1.05, 'actual_model_source')
        text = messages[1]['content']
        for rule in self.static['catalog']['rules']:
            self.assertIn(rule['rule_id'], text)
        self.assertIn('actual_model_source', text)
        self.assertIn('"next_expansion_budget":5', text)
        self.assertIn('source_port', text)
        self.assertNotIn('Full rule factory, constructor and matching semantics', text)
        self.assertNotIn('Graph* Graph::optimize_search(', text)
        self.assertEqual(compact.expand_graph(artifacts['graph']), observation)
        self.assertEqual(metrics['messages_utf8_bytes'], compact.byte_size(messages))
        self.assertGreater(metrics['component_reduction_fraction'], .70)
        self.assertEqual(metrics['rule_count'], 151)
        stale = dict(self.compacted, source_static_sha256='wrong')
        with self.assertRaisesRegex(ValueError, 'regenerate compaction'):
            compact.build_checkpoint_messages(self.static, stale, observation, feedback, 5, 1.05)

    def test_twenty_real_client_requests_keep_history_bounded_and_high_settings(self):
        import checkpoint_strategy
        import openrouter_client
        from test_checkpoints import Graph, optimize_with_checkpoints
        cfg = json.loads((ROOT / 'benchmarks/config.json').read_text())
        cfg.update(budget=100, checkpoint_interval=5)
        strategy = {'schema_version': 1, 'name': 'all', 'phases': [
            {'name': 'all', 'rule_ids': 'all', 'expansion_budget': 5}]}
        response = {'model': 'deepseek/deepseek-v4-pro', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': json.dumps(strategy)}}]}
        bodies = []
        def http(request, **kwargs):
            bodies.append(json.loads(request.data))
            return io.StringIO(json.dumps(response))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fake-key'}), contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            context = folder / 'static_context.json'
            context.write_text(json.dumps(self.static))
            with patch.object(checkpoint_strategy, 'compact_static_context', wraps=compact.compact_static_context) as prepare, patch.object(
                    openrouter_client.urllib.request, 'urlopen', side_effect=http):
                controller = checkpoint_strategy.CheckpointStrategist('deepseek/deepseek-v4-pro', cfg,
                    context, 'actual_model_source', folder)
                optimize_with_checkpoints(Graph(), controller, budget=100, checkpoint_interval=5)
                self.assertEqual(prepare.call_count, 1)
            self.assertEqual(controller.model_calls, 20)
            for body in bodies:
                self.assertEqual(body['temperature'], .2)
                self.assertEqual(body['reasoning'], {'effort': 'high'})
                self.assertEqual(body['max_tokens'], 32768)
            final = json.loads((folder / 'checkpoints/expansion_0095/feedback_compact.json').read_text())
            self.assertEqual(final['previous_policy_count'], 19)
            self.assertEqual(len(final['recent_policies']), 4)
            sizes = [r['prompt_compaction']['messages_utf8_bytes'] for r in controller.records]
            self.assertLess(max(sizes) - min(sizes), 4000)
            self.assertEqual(controller.aggregate()['prompt_bytes_total'], sum(sizes))

    def test_one_shot_builder_still_uses_original_template_and_full_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'static_context.json'
            path.write_text(json.dumps(self.static))
            with patch.object(compact, 'compact_static_context', side_effect=AssertionError('One-shot used checkpoint compaction')):
                messages, catalog = llm_strategy.build_messages(IntermediateGraph(), path, total_budget=100,
                                                                alpha=1.05, model_source='actual_model_source')
            self.assertIsNotNone(catalog)
            self.assertIn('Generate one complete, graph-specific search strategy', messages[1]['content'])
            self.assertIn('Full rule factory, constructor and matching semantics', messages[1]['content'])
            self.assertNotIn('SEARCH CONTRACT\n', messages[1]['content'])

    def test_standalone_compaction_script_needs_no_gpu_or_api(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as printed, patch(
                'urllib.request.urlopen', side_effect=AssertionError('Compaction contacted a service')):
            self.assertEqual(compact.main(['--repo', str(ROOT / 'TASO'), '--out', tmp]), 0)
            report = json.loads(printed.getvalue())
            self.assertTrue(report['catalog_round_trip'])
            self.assertEqual(report['rule_count'], 151)
            self.assertGreater(report['static_reduction_fraction'], .70)
            self.assertTrue((Path(tmp) / 'compact_static_context.json').exists())
            self.assertTrue((Path(tmp) / 'full/static_context.json').exists())


if __name__ == '__main__':
    unittest.main()

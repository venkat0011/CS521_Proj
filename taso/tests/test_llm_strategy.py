"""Prompt, budget and provider checks without GPU execution or paid API calls."""
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
CONTEXT = ROOT / 'llm_context'
sys.path.insert(0, str(CONTEXT))
import llm_strategy
import openrouter_client
from model_settings import resolve_settings
spec = importlib.util.spec_from_file_location('taso.strategy', ROOT / 'TASO/python/taso/strategy.py')
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)


class GraphFixture:
    def get_operator_list(self): return [{'guid': 2}]
    def get_operator_type(self, op): return {1: 'Input', 2: 'Relu'}[op['guid']]
    def get_num_outputs(self, op): return 1
    def get_output_dims(self, op, i): return [1, 8, 16, 16]
    def get_input_dims(self, op, i): return [1, 8, 16, 16]
    def get_input_edges(self, op): return [{'srcOp': {'guid': 1}, 'srcIdx': 0, 'dstIdx': 0}]
    def cost(self): return 0.1


class LlmTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {'rules': [{'rule_id': 'builtin:000'}, {'rule_id': 'protobuf:0000'}]}
        self.strategy = {'schema_version': 1, 'name': 'test', 'phases': [
            {'name': 'one', 'rule_ids': ['builtin:000'], 'expansion_budget': 4},
            {'name': 'two', 'rule_ids': 'all', 'expansion_budget': 20}]}
        fake = types.ModuleType('taso')
        fake.operator_attrs = {'Relu': []}
        self.modules = patch.dict(sys.modules, {'taso': fake, 'taso.strategy': strategy_module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_strategy_matches_executor_and_fixed_budget(self):
        self.assertEqual(llm_strategy.validate_strategy(json.dumps(self.strategy), self.catalog), self.strategy)
        for mutate in ('budget', 'unknown', 'phases'):
            bad = copy.deepcopy(self.strategy)
            if mutate == 'budget': bad['phases'][0]['expansion_budget'] = 5
            if mutate == 'unknown': bad['phases'][0]['rule_ids'] = ['protobuf:9999']
            if mutate == 'phases': bad['phases'] = [dict(bad['phases'][0], expansion_budget=1)] * 24
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                llm_strategy.validate_strategy(json.dumps(bad), self.catalog)
        with self.assertRaises(ValueError):
            llm_strategy.validate_strategy('{"schema_version":1,"schema_version":2}', self.catalog)

    def test_dry_run_saves_real_request_without_calling_model(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            context = directory / 'context.json'
            context.write_text(json.dumps({'catalog': self.catalog, 'source_context': []}))
            def forbidden(messages): raise AssertionError('Dry run contacted the model')
            result = llm_strategy.request_strategy(GraphFixture(), forbidden, context, directory,
                                                    model_source='Actual TASO construction', dry_run=True)
            self.assertIsNone(result)
            text = (directory / 'request.txt').read_text()
            self.assertIn('Actual TASO construction', text)
            self.assertIn('total_graph_expansions', text)
            self.assertIn('positive integer', text)
            self.assertNotIn('${graph}', text)
            self.assertFalse((directory / 'strategy.json').exists())

    def test_one_call_validates_and_saves_strategy(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            context = directory / 'context.json'
            context.write_text(json.dumps({'catalog': self.catalog, 'source_context': []}))
            calls = []
            def caller(messages):
                calls.append(messages)
                return json.dumps(self.strategy)
            result = llm_strategy.request_strategy(GraphFixture(), caller, context, directory)
            self.assertEqual(result, self.strategy)
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads((directory / 'strategy.json').read_text()), self.strategy)

    def test_provider_saves_usage_but_not_credentials(self):
        response = {'id': 'fixture', 'model': 'deepseek/deepseek-v4-pro', 'provider': 'fixture',
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 20},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(self.strategy)}}]}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key-never-save'}):
            client = openrouter_client.OpenRouterClient(save_to=temp)
            with patch.object(openrouter_client.urllib.request, 'urlopen', return_value=io.StringIO(json.dumps(response))) as http:
                content = client([{'role': 'user', 'content': 'fixture'}])
            self.assertEqual(json.loads(content), self.strategy)
            self.assertEqual(client.request_attempts, 1)
            self.assertEqual(client.metadata['usage']['completion_tokens'], 20)
            self.assertIn('deepseek-v4-pro', json.loads(http.call_args.args[0].data)['model'])
            for path in Path(temp).iterdir():
                self.assertNotIn('test-key-never-save', path.read_text())

    def test_per_model_request_parameters_and_saved_metadata(self):
        cases = [
            ('google/gemini-3.8-flash', .2, 'medium'),
            ('deepseek/deepseek-v4-pro', .2, 'high'),
            ('google/gemini-4-argon', .2, 'medium'),
            ('anthropic/claude-opus-5.5', None, 'medium'),
            ('openai/gpt-5.3-codex', None, 'medium'),
            ('moonshotai/kimi-k3.1-pro', None, 'high'),
        ]
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
        for model, temperature, effort in cases:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temp, patch.dict(
                    os.environ, {'OPENROUTER_API_KEY': 'test-key-never-save'}):
                client = openrouter_client.OpenRouterClient(
                    model, save_to=temp, max_tokens=32768, temperature=.2,
                    reasoning_effort='medium',
                    reasoning_effort_overrides={'moonshotai/kimi-k3.1-pro': 'high',
                                                'deepseek/deepseek-v4-pro': 'high'})
                with patch.object(openrouter_client.urllib.request, 'urlopen',
                                  return_value=io.StringIO(json.dumps(response))) as http:
                    client([{'role': 'user', 'content': 'fixture'}])
                body = json.loads(http.call_args.args[0].data)
                self.assertEqual(body['reasoning'], {'effort': effort})
                self.assertEqual(body.get('temperature'), temperature)
                self.assertEqual('temperature' in body, temperature is not None)
                self.assertEqual(body['provider'], {'require_parameters': True})
                self.assertEqual(body['max_tokens'], 32768)
                self.assertEqual(json.loads((Path(temp) / 'api_request.json').read_text()), body)
                metadata = json.loads((Path(temp) / 'inference_metadata.json').read_text())
                self.assertEqual(metadata['requested_temperature'], .2)
                self.assertEqual(metadata['temperature'], temperature)
                self.assertEqual(metadata['requested_reasoning_effort'], 'medium')
                self.assertEqual(metadata['reasoning_effort'], effort)
                self.assertEqual('temperature' in metadata['omitted_parameters'], temperature is None)
                self.assertEqual(metadata['reasoning_effort_override_applied'], effort == 'high')
                for path in Path(temp).iterdir():
                    self.assertNotIn('test-key-never-save', path.read_text())

    def test_unsupported_effort_and_unknown_model_fail_before_api_call(self):
        with patch.object(openrouter_client.urllib.request, 'urlopen') as http:
            with self.assertRaisesRegex(ValueError, 'does not support'):
                openrouter_client.OpenRouterClient('moonshotai/kimi-k3.1-pro', reasoning_effort='medium')
            with self.assertRaisesRegex(ValueError, 'No verified parameter profile'):
                openrouter_client.OpenRouterClient('unverified/model')
            http.assert_not_called()
        # Existing standalone DeepSeek calls retain provider-default reasoning.
        settings = resolve_settings('deepseek/deepseek-v4-pro')
        self.assertEqual(settings['parameters'], {'temperature': .2})
        with self.assertRaisesRegex(ValueError, 'does not support'):
            resolve_settings('deepseek/deepseek-v4-pro', reasoning_effort='medium')

    def test_incomplete_response_stops_without_retry(self):
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': '{'}}]}
        client = openrouter_client.OpenRouterClient()
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}), patch.object(
                openrouter_client.urllib.request, 'urlopen', return_value=io.StringIO(json.dumps(response))) as http:
            with self.assertRaises(RuntimeError):
                client([{'role': 'user', 'content': 'fixture'}])
            self.assertEqual(http.call_count, 1)


if __name__ == '__main__':
    unittest.main()

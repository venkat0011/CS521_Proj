"""Output-cap omission, failed-call accounting and CLI integration; no paid calls."""
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'llm_context'))
sys.path.insert(0, str(ROOT / 'benchmarks'))
import checkpoint_strategy
import openrouter_client
import results
import run_suite
from token_limits import normalize_max_tokens, parse_max_tokens


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TokenLimitTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / 'benchmarks/config.json').read_text())

    def test_cli_dispatch_distinguishes_provider_from_no_override(self):
        launcher = load('launcher_token_limits', ROOT / 'run_experiment.py')
        one_shot = load('oneshot_token_limits', ROOT / 'experiments/run_llm_strategy.py')
        for value in ('provider', '65536'):
            for mode in ('suite', 'llm'):
                args = launcher.parse_args(['--experiment', mode, '--max-tokens', value])
                command = launcher.experiment_command(ROOT.parent, args, Path('/tmp/python'), Path('/tmp/out'))
                parser = run_suite.parse_args if mode == 'suite' else one_shot.parse_args
                forwarded = parser([str(part) for part in command[2:]])
                self.assertEqual(forwarded.max_tokens, 'provider' if value == 'provider' else 65536)
        self.assertIsNone(run_suite.parse_args([]).max_tokens)
        self.assertEqual(one_shot.parse_args([]).max_tokens, 8192)
        self.assertEqual(self.config()['max_tokens'], 32768)

    def test_plan_and_config_accept_null_without_changing_other_settings(self):
        cfg = self.config()
        cfg['max_tokens'] = None
        self.assertIs(run_suite.validate_config(cfg), cfg)
        command = [sys.executable, str(ROOT / 'benchmarks/run_suite.py'), '--plan-only',
                   '--checkpoint-interval', '25', '--max-tokens', 'provider']
        plan = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
        self.assertIsNone(plan['max_tokens'])
        self.assertEqual(plan['output_token_limit_mode'], 'provider_default')
        self.assertEqual(plan['max_model_calls'], 240)
        self.assertEqual(plan['model_settings']['deepseek/deepseek-v4-pro']['reasoning_effort'], 'high')
        self.assertEqual(plan['model_settings']['deepseek/deepseek-v4-pro']['temperature'], .2)
        # JSON null is the only config representation for omitting our cap.
        for bad in (0, -1, True, 1.5, 'provider', '65536', 2**31):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_suite.validate_config(dict(cfg, max_tokens=bad))
        for bad in (0, -1, True, 1.5, 'anything'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_max_tokens(bad)
        self.assertEqual(parse_max_tokens('65536'), 65536)

    def test_provider_mode_omits_field_and_saves_returned_usage(self):
        response = {'id': 'fixture', 'usage': {'prompt_tokens': 38000, 'completion_tokens': 16060,
                    'completion_tokens_details': {'reasoning_tokens': 16012}},
                    'choices': [{'finish_reason': 'stop', 'native_finish_reason': 'stop',
                                 'message': {'content': '{}'}}]}
        for cap in ('provider', None, 65536):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as tmp, patch.dict(
                    os.environ, {'OPENROUTER_API_KEY': 'secret-fixture-never-save'}):
                client = openrouter_client.OpenRouterClient(save_to=tmp, max_tokens=cap,
                    reasoning_effort='high', temperature=.2)
                with patch.object(openrouter_client.urllib.request, 'urlopen',
                                  return_value=io.StringIO(json.dumps(response))) as http:
                    self.assertEqual(client([{'role': 'user', 'content': 'Return JSON'}]), '{}')
                body = json.loads(http.call_args.args[0].data)
                self.assertEqual('max_tokens' in body, cap == 65536)
                self.assertNotIn('max_completion_tokens', body)
                self.assertEqual(body['reasoning'], {'effort': 'high'})
                self.assertEqual(body['temperature'], .2)
                self.assertEqual(client.metadata['finish_reason'], 'stop')
                self.assertEqual(client.metadata['visible_completion_tokens'], 48)
                self.assertEqual(client.metadata['output_token_limit_mode'],
                                 'explicit' if cap == 65536 else 'provider_default')
                for path in Path(tmp).iterdir():
                    self.assertNotIn('secret-fixture-never-save', path.read_text())

    def test_provider_limit_failure_retains_usage_and_does_not_retry(self):
        response = {'usage': {'completion_tokens': 65536,
                    'completion_tokens_details': {'reasoning_tokens': 65536}},
                    'choices': [{'finish_reason': 'length', 'message': {'content': ''}}]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture'}):
            client = openrouter_client.OpenRouterClient(save_to=tmp, max_tokens=None)
            with patch.object(openrouter_client.urllib.request, 'urlopen',
                              return_value=io.StringIO(json.dumps(response))) as http:
                with self.assertRaisesRegex(RuntimeError, 'provider_default'):
                    client([{'role': 'user', 'content': 'Return JSON'}])
                self.assertEqual(http.call_count, 1)
            metadata = json.loads((Path(tmp) / 'inference_metadata.json').read_text())
            self.assertEqual(metadata['finish_reason'], 'length')
            self.assertEqual(metadata['visible_completion_tokens'], 0)
            self.assertEqual(metadata['usage']['completion_tokens'], 65536)

    def test_missing_reasoning_usage_is_unknown_instead_of_zero(self):
        response = {'usage': {'completion_tokens': 20},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
        client = openrouter_client.OpenRouterClient(max_tokens=None)
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture'}), patch.object(
                openrouter_client.urllib.request, 'urlopen', return_value=io.StringIO(json.dumps(response))):
            client([{'role': 'user', 'content': 'JSON'}])
        self.assertIsNone(client.metadata['visible_completion_tokens'])

    def test_fourth_checkpoint_failure_is_in_aggregate_and_per_call_report(self):
        strategy_module = load('taso.strategy', ROOT / 'TASO/python/taso/strategy.py')
        fake = types.ModuleType('taso')
        fake.operator_attrs = {}
        class Graph:
            def get_operator_list(self): return []
            def cost(self): return 10
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture'}), patch.dict(
                sys.modules, {'taso': fake, 'taso.strategy': strategy_module}):
            directory = Path(tmp)
            from prepare_context import prepare_context
            prepare_context(ROOT / 'TASO', directory)
            cfg = dict(self.config(), max_tokens=None, search_mode='checkpoint', checkpoint_interval=25)
            controller = checkpoint_strategy.CheckpointStrategist('deepseek/deepseek-v4-pro', cfg,
                directory / 'static_context.json', 'Actual graph builder', directory)
            responses, bodies = [], []
            for index in range(4):
                failed = index == 3
                policy = {'schema_version': 1, 'name': 'fixture', 'phases': [
                    {'name': 'next', 'rule_ids': 'all', 'expansion_budget': 25}]}
                responses.append({'id': f'fixture-{index}', 'usage': {
                    'prompt_tokens': 38000, 'completion_tokens': 65536 if failed else 16060,
                    'total_tokens': 103536 if failed else 54060, 'cost': .2 if failed else .1,
                    'completion_tokens_details': {'reasoning_tokens': 65536 if failed else 16012}},
                    'choices': [{'finish_reason': 'length' if failed else 'stop',
                                 'message': {'content': '' if failed else json.dumps(policy)}}]})
            def http(request, **kwargs):
                bodies.append(json.loads(request.data))
                return io.StringIO(json.dumps(responses[len(bodies)-1]))
            with patch.object(openrouter_client.urllib.request, 'urlopen', side_effect=http):
                for consumed in (0, 25, 50, 75):
                    progress = {'expansions': consumed, 'total_budget': 100, 'budget_remaining': 100-consumed,
                        'next_expansion_budget': 25, 'queue_remaining': 10, 'initial_cost_ms': 10,
                        'best_search_cost_ms': 10-consumed*.01, 'phases': []}
                    if consumed == 75:
                        with self.assertRaisesRegex(RuntimeError, 'Incomplete completion'):
                            controller(Graph(), progress)
                    else:
                        controller(Graph(), progress)
            self.assertEqual(len(bodies), 4)
            self.assertTrue(all('max_tokens' not in b for b in bodies))
            metadata = controller.aggregate()
            self.assertEqual(metadata['finish_reason_counts'], {'stop': 3, 'length': 1})
            self.assertEqual(metadata['usage']['completion_tokens'], 113716)
            self.assertEqual(metadata['visible_completion_tokens'], 144)
            self.assertAlmostEqual(metadata['usage']['cost'], .5)
            report = {'network': 'squeezenet', 'model': controller.model, 'trial': 1,
                'search_mode': 'checkpoint', 'status': 'failed_evaluation', 'model_calls': 4,
                'inference': metadata, 'evaluation': {'checkpoint_decisions': controller.records}}
            before = copy.deepcopy(report)
            stats = results.write_exports(directory / 'exports', [report], plots=False)
            self.assertEqual(report, before)
            with (directory / 'exports/model_calls.csv').open() as handle:
                calls = list(csv.DictReader(handle))
            self.assertEqual(len(calls), 4)
            self.assertEqual([r['expansion'] for r in calls], ['0', '25', '50', '75'])
            self.assertEqual(calls[-1]['completion_tokens'], '65536')
            self.assertEqual(calls[-1]['visible_completion_tokens'], '0')
            self.assertEqual(calls[-1]['finish_reason'], 'length')
            self.assertEqual(calls[-1]['max_tokens'], '')
            self.assertEqual(results.flatten(report)['api_length_completions'], 1)
            self.assertEqual(stats['groups'][0]['metrics']['api_completion_tokens']['samples'][0]['value'], 113716)
            self.assertIn('LLM token usage', (directory / 'exports/summary.md').read_text())


if __name__ == '__main__':
    unittest.main()

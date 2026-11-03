"""Bounded continuation, response replay and paused-checkpoint integration, without paid calls."""
import copy
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
import urllib.error
from email.message import Message

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'llm_context'))
sys.path.insert(0, str(ROOT / 'benchmarks'))
import checkpoint_completion as completion
import checkpoint_strategy
import openrouter_client
import results
import run_suite


def policy(budget=25):
    return {'schema_version': 1, 'name': 'fixture', 'phases': [
        {'name': 'next', 'rule_ids': 'all', 'expansion_budget': budget}]}


def response(reason='stop', *, content=None, reasoning=None, details=None, prompt=100, output=20,
             thought=10, error=None):
    message = {'role': 'assistant', 'content': json.dumps(policy()) if content is None else content}
    if reasoning is not None:
        message['reasoning'] = reasoning
    if details is not None:
        message['reasoning_details'] = details
    choice = {'finish_reason': reason, 'message': message}
    if error:
        choice['error'] = error
    return {'id': 'fixture', 'provider': 'fixture', 'usage': {
        'prompt_tokens': prompt, 'completion_tokens': output, 'total_tokens': prompt+output,
        'cost': .01, 'completion_tokens_details': {'reasoning_tokens': thought}}, 'choices': [choice]}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'benchmarks/checkpoint_continuation.json').read_text())
        self.key = patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fixture-secret-never-save'})
        self.key.start()
        self.addCleanup(self.key.stop)
        self.messages = [{'role': 'system', 'content': 'Return JSON'}, {'role': 'user', 'content': 'Same decision'}]

    def run_completion(self, directory, responses, *, model='deepseek/deepseek-v4-pro', cfg=None, sleeper=None):
        bodies = []
        responses = iter(responses)
        def http(request, **kwargs):
            bodies.append(json.loads(request.data))
            value = next(responses)
            if isinstance(value, BaseException):
                raise value
            return io.StringIO(json.dumps(value))
        runner = completion.CheckpointCompletion(model, cfg or self.cfg, directory,
                                                 sleep=sleeper or (lambda _: None))
        return runner, bodies, patch.object(openrouter_client.urllib.request, 'urlopen', side_effect=http)

    def test_maximum_and_model_settings_for_all_four_models(self):
        caps = [393216, 943718, 128000, 65536]
        for model, cap in zip(self.cfg['models'], caps):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                runner, bodies, http = self.run_completion(tmp, [response()], model=model)
                original = copy.deepcopy(self.messages)
                with http:
                    self.assertEqual(json.loads(runner(self.messages)), policy())
                self.assertEqual(self.messages, original)
                self.assertEqual(bodies[0]['max_tokens'], cap)
                expected = 'high' if model.startswith(('deepseek/', 'moonshotai/')) else 'medium'
                self.assertEqual(bodies[0]['reasoning'], {'effort': expected})
                self.assertEqual(bodies[0]['provider'], {'require_parameters': True})
                self.assertEqual(runner.request_attempts, 1)
                self.assertEqual(runner.metadata['continuation_requests'], 0)
                self.assertTrue((Path(tmp) / 'attempts/attempt_01/api_response.json').exists())
                for path in Path(tmp).rglob('*.json'):
                    self.assertNotIn('fixture-secret-never-save', path.read_text())

    def test_deepseek_all_reasoning_is_reused_as_explicit_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp, [
                response('length', content='', reasoning='Reuse this unfinished observation', output=200, thought=200),
                response()])
            with http:
                self.assertEqual(json.loads(runner(self.messages)), policy())
            self.assertEqual(len(bodies), 2)
            self.assertEqual(bodies[1]['messages'][:2], self.messages)
            self.assertIn('Reuse this unfinished observation', bodies[1]['messages'][-1]['content'])
            self.assertIn('COMPLETE strategy JSON', bodies[1]['messages'][-1]['content'])
            self.assertEqual(runner.attempts[1]['replay_mode'], 'returned_plaintext_as_context')
            self.assertEqual(runner.metadata['finish_reason_counts'], {'length': 1, 'stop': 1})
            self.assertEqual(runner.metadata['usage']['completion_tokens'], 220)
            self.assertAlmostEqual(runner.metadata['usage']['cost'], .02)

    def test_raw_blocks_saved_and_portable_work_reused_without_concatenating_json(self):
        details = [{'type': 'reasoning.summary', 'summary': 'A partial decision', 'id': 'summary'},
                   {'type': 'reasoning.encrypted', 'data': 'opaque-fixture', 'id': 'signed', 'index': 1}]
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp, [
                response('length', content='{"schema_version":', details=details), response()],
                model='google/gemini-3.8-flash')
            with http:
                text = runner(self.messages)
            self.assertEqual(json.loads(text), policy())
            replay = bodies[1]['messages'][-2]
            self.assertEqual(replay['content'], '{"schema_version":')
            self.assertIn('A partial decision', bodies[1]['messages'][-1]['content'])
            self.assertNotIn('opaque-fixture', json.dumps(bodies[1]['messages']))
            raw = json.loads((Path(tmp)/'attempts/attempt_01/api_response.json').read_text())
            self.assertEqual(raw['choices'][0]['message']['reasoning_details'], details)
            self.assertEqual(runner.attempts[1]['replay_mode'], 'returned_plaintext_as_context')

    def test_three_retries_means_exactly_four_attempts_and_all_usage_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp, [
                response('length', content='{', reasoning='unfinished') for _ in range(5)])
            with http:
                with self.assertRaisesRegex(RuntimeError, 'after 4 API attempts'):
                    runner(self.messages)
            self.assertEqual(len(bodies), 4)
            self.assertEqual(len(runner.attempts), 4)
            self.assertEqual(runner.metadata['continuation_requests'], 3)
            self.assertEqual(runner.metadata['usage']['completion_tokens'], 80)
            self.assertAlmostEqual(runner.metadata['usage']['cost'], .04)
            self.assertEqual(len(bodies[-1]['messages']), len(self.messages)+6)
            saved = json.loads((Path(tmp) / 'attempts.json').read_text())
            self.assertTrue(all(a['status'] == 'failed' for a in saved))

    def test_provider_failures_share_the_same_followup_allowance(self):
        headers = Message()
        headers['Retry-After'] = '3'
        bad = urllib.error.HTTPError('fixture', 503, 'busy', headers, io.BytesIO(b'provider unavailable'))
        delays = []
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp, [bad,
                response('length', content='{', reasoning='unfinished'),
                response('error', content='', error={'code': 502, 'message': 'provider unavailable'}), response()],
                sleeper=delays.append)
            with http:
                self.assertEqual(json.loads(runner(self.messages)), policy())
            self.assertEqual(len(bodies), 4)
            self.assertEqual([a['action'] for a in runner.attempts],
                             ['initial', 'retry_provider', 'continue_after_length', 'retry_provider'])
            self.assertEqual(delays, [3., 8])
            self.assertEqual(runner.metadata['provider_retries'], 2)
            self.assertEqual(runner.metadata['usage_missing_attempts'], 1)
            self.assertAlmostEqual(runner.metadata['usage']['cost'], .03)

    def test_authentication_and_invalid_requests_are_not_retried(self):
        for status in (400, 401, 402, 403, 404):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                bad = urllib.error.HTTPError('fixture', status, 'terminal', {}, io.BytesIO(b'invalid'))
                runner, bodies, http = self.run_completion(tmp, [bad, response()])
                with http, self.assertRaises(openrouter_client.ProviderFailure):
                    runner(self.messages)
                self.assertEqual(len(bodies), 1)

    def test_no_work_retries_but_content_filter_is_terminal(self):
        for result, expected_calls in ((response('length', content='', thought=20), 2),
                       (response('content_filter', content='partial'), 1)):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as tmp:
                runner, bodies, http = self.run_completion(tmp, [result, response()])
                with http:
                    if expected_calls == 1:
                        with self.assertRaises(RuntimeError):
                            runner(self.messages)
                    else:
                        self.assertEqual(json.loads(runner(self.messages)), policy())
                        self.assertEqual(runner.attempts[1]['replay_mode'], 'no_readable_work_retry')
                self.assertEqual(len(bodies), expected_calls)

    def test_missing_usage_uses_conservative_text_guard_and_remains_unknown(self):
        truncated = response('length', content='{')
        truncated.pop('usage')
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp, [truncated, response()])
            with http:
                runner(self.messages)
            self.assertEqual(len(bodies), 2)
            self.assertEqual(runner.metadata['usage_missing_attempts'], 1)

    def test_context_guard_clamps_output_and_stops_before_an_impossible_request(self):
        limits = completion.MODEL_LIMITS['google/gemini-3.8-flash']
        self.assertEqual(completion.output_cap(limits, 'max', 1040000), 4480)
        self.assertEqual(completion.output_cap(limits, 2000, 100), 2000)
        with self.assertRaisesRegex(RuntimeError, 'Insufficient'):
            completion.output_cap(limits, 'max', 1048576)
        with self.assertRaisesRegex(RuntimeError, 'input-token'):
            completion.output_cap(completion.MODEL_LIMITS['openai/gpt-5.3-codex'], 'max', 270000)
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp, [response()])
            with http, self.assertRaisesRegex(RuntimeError, 'Insufficient'):
                runner([{'role': 'user', 'content': 'x'*1100000}])
            self.assertEqual(len(bodies), 0)

    def test_large_generated_history_reduces_next_allowance(self):
        with tempfile.TemporaryDirectory() as tmp:
            truncated = response('length', content='{', reasoning='work', prompt=38000, output=943718, thought=943700)
            runner, bodies, http = self.run_completion(tmp, [truncated, response()], model='moonshotai/kimi-k3.1-pro')
            with http:
                runner(self.messages)
            self.assertEqual(bodies[0]['max_tokens'], 943718)
            self.assertLess(bodies[1]['max_tokens'], 65536)
            estimate = runner.attempts[1]['estimated_input_tokens']
            self.assertLessEqual(estimate + bodies[1]['max_tokens'] + completion.MARGIN, 1048576)

    def test_presets_plan_and_cli_dispatch(self):
        command = [sys.executable, str(ROOT/'benchmarks/run_suite.py'), '--config',
                   str(ROOT/'benchmarks/checkpoint_continuation.json'), '--plan-only']
        plan = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
        self.assertEqual(plan['max_model_calls'], 960)
        self.assertEqual(plan['max_model_calls_per_trial'], 16)
        self.assertEqual(plan['max_retries_per_checkpoint'], 3)
        spec = importlib.util.spec_from_file_location('launcher_continuation', ROOT/'run_experiment.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        args = launcher.parse_args(['--experiment','suite','--checkpoint-interval','25',
                                    '--max-retries','3','--max-tokens','max'])
        forwarded = launcher.experiment_command(ROOT.parent, args, Path('/tmp/python'), Path('/tmp/out'))
        parsed = run_suite.parse_args([str(x) for x in forwarded[2:]])
        self.assertEqual(parsed.max_retries, 3)
        self.assertEqual(parsed.max_tokens, 'max')
        for retries in (-1, 4, True):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                run_suite.validate_config(dict(self.cfg, max_retries=retries))
        for change in ({'search_mode':'one_shot'}, {'max_tokens':None}):
            with self.assertRaises(ValueError):
                run_suite.validate_config(dict(self.cfg, **change))

    def test_invalid_completed_strategy_retries_without_claiming_token_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp, [response(content='bad json'), response()])
            with http:
                self.assertEqual(json.loads(runner(self.messages)), policy())
            self.assertEqual(len(bodies), 2)
            self.assertEqual(runner.attempts[0]['failure_kind'], 'invalid_strategy')
            self.assertEqual(runner.attempts[1]['action'], 'retry_strategy')
            self.assertEqual(runner.metadata['finish_reason_counts'], {'stop': 2})
            self.assertNotIn('reached its output-token limit', bodies[1]['messages'][-1]['content'])

    def test_missing_strategy_with_stop_or_no_choices_is_retried_and_not_called_length(self):
        empty = response(content='', thought=20)
        no_choices = response(content='')
        no_choices['choices'] = []
        for initial in (empty, no_choices):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as tmp:
                runner, bodies, http = self.run_completion(tmp, [initial, response()])
                with http:
                    self.assertEqual(json.loads(runner(self.messages)), policy())
                self.assertEqual(len(bodies), 2)
                self.assertEqual(runner.attempts[0]['failure_kind'], 'missing_strategy')
                self.assertEqual(runner.attempts[1]['retry_reason'], 'missing_strategy')
                self.assertEqual(runner.metadata['strategy_retries'], 1)
                self.assertNotIn('length', runner.metadata['finish_reason_counts'])
                self.assertNotIn('reached its output-token limit', bodies[1]['messages'][-1]['content'])

    def test_invalid_strategy_exhaustion_does_not_exceed_four_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp, [response(content='bad json') for _ in range(5)])
            with http, self.assertRaisesRegex(RuntimeError, 'after 4 API attempts'):
                runner(self.messages)
            self.assertEqual(len(bodies), 4)
            self.assertEqual(runner.metadata['strategy_retries'], 3)
            self.assertEqual(runner.metadata['finish_reason_counts'], {'stop': 4})
            self.assertTrue(all(a['failure_kind'] == 'invalid_strategy' for a in runner.attempts))

    def test_provider_can_change_and_only_readable_work_is_sent_back(self):
        first = response('length', content='', details=[
            {'type': 'reasoning.summary', 'summary': 'The branches share a convolution'},
            {'type': 'reasoning.encrypted', 'data': 'provider-specific-opaque-block'}])
        first['provider'] = 'Crusoe'
        final = response()
        final['provider'] = 'Relace'
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp, [first, final])
            with http:
                runner(self.messages)
            self.assertEqual(runner.metadata['provider'], 'Crusoe,Relace')
            self.assertEqual(runner.metadata['provider_routing'], 'openrouter_auto')
            self.assertTrue(all(b['provider'] == {'require_parameters': True} for b in bodies))
            self.assertIn('The branches share a convolution', json.dumps(bodies[1]['messages']))
            self.assertNotIn('provider-specific-opaque-block', json.dumps(bodies[1]))
            self.assertTrue(all('provider_tag' not in a['limits'] for a in runner.attempts))

    def test_checkpoint_uses_real_schema_validation_before_accepting_attempt(self):
        import test_checkpoints as fixture
        spec = importlib.util.spec_from_file_location('taso.strategy', ROOT/'TASO/python/taso/strategy.py')
        strategy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy)
        fake = types.ModuleType('taso')
        fake.operator_attrs = {}
        bad = policy()
        bad['phases'][0]['rule_ids'] = ['unknown:999']
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'taso':fake,'taso.strategy':strategy}):
            directory = Path(tmp)
            from prepare_context import prepare_context
            prepare_context(ROOT/'TASO',directory)
            controller = checkpoint_strategy.CheckpointStrategist('deepseek/deepseek-v4-pro', self.cfg,
                directory/'static_context.json', 'builder', directory)
            pending = iter([response(content=json.dumps(bad)), response(), response(), response(), response()])
            bodies = []
            def http(request, **kwargs):
                bodies.append(json.loads(request.data))
                return io.StringIO(json.dumps(next(pending)))
            graph = fixture.Graph()
            with patch.object(openrouter_client.urllib.request, 'urlopen', side_effect=http):
                fixture.optimize_with_checkpoints(graph,controller,budget=100,checkpoint_interval=25)
            self.assertEqual(graph.positions,[0,25,50,75])
            self.assertEqual(controller.model_calls,5)
            self.assertEqual(controller.records[0]['attempts'][0]['failure_kind'],'invalid_strategy')
            self.assertEqual(controller.records[0]['strategy'],policy())
            self.assertEqual(controller.aggregate()['strategy_retries'],1)
            self.assertTrue(all(b['provider'] == {'require_parameters':True} for b in bodies))

    def test_paused_native_wrapper_and_exports_include_recovered_attempt(self):
        import test_checkpoints as fixture
        spec = importlib.util.spec_from_file_location('taso.strategy', ROOT/'TASO/python/taso/strategy.py')
        strategy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy)
        fake = types.ModuleType('taso')
        fake.operator_attrs = {}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'taso':fake,'taso.strategy':strategy}):
            directory = Path(tmp)
            from prepare_context import prepare_context
            prepare_context(ROOT/'TASO', directory)
            strategist = checkpoint_strategy.CheckpointStrategist('deepseek/deepseek-v4-pro',self.cfg,
                directory/'static_context.json','actual graph builder',directory)
            pending = iter([response(), response(), response(),
                            response('length', content='', reasoning='unfinished fusion decision'), response()])
            bodies = []
            def http(request, **kwargs):
                bodies.append(json.loads(request.data))
                return io.StringIO(json.dumps(next(pending)))
            graph = fixture.Graph()
            with patch.object(openrouter_client.urllib.request,'urlopen',side_effect=http):
                final = fixture.optimize_with_checkpoints(graph,strategist,budget=100,checkpoint_interval=25)
            self.assertEqual(graph.positions,[0,25,50,75])
            self.assertAlmostEqual(final.cost(),9)
            self.assertEqual(strategist.model_calls,5)
            self.assertEqual(len(strategist.records),4)
            self.assertEqual(len(strategist.records[-1]['attempts']),2)
            self.assertEqual(strategist.aggregate()['finish_reason_counts'],{'stop':4,'length':1})
            self.assertEqual(strategist.aggregate()['checkpoints_reached'],4)
            report = {'network':'squeezenet','model':'deepseek/deepseek-v4-pro','trial':1,
                'status':'completed','search_mode':'checkpoint','model_calls':5,
                'inference':strategist.aggregate(),'evaluation':{'checkpoint_decisions':strategist.records,
                'variants': {'baseline': {'estimated_ms':1.,'measured_ms':1.,'search_seconds':.1,'expansions':100},
                             'llm': {'estimated_ms':.9,'measured_ms':.9,'search_seconds':.2,'expansions':100}}}}
            rows = results.model_call_rows([report])
            self.assertEqual(len(rows),5)
            self.assertEqual([r['expansion'] for r in rows],[0,25,50,75,75])
            self.assertEqual(rows[-1]['action'],'continue_after_length')
            self.assertEqual(results.flatten(report)['recovered_checkpoints'],1)
            stats = results.write_exports(directory/'exports',[report],plots=False)
            self.assertEqual(stats['groups'][0]['completed'],1)
            self.assertAlmostEqual(stats['groups'][0]['metrics']['llm_over_baseline_measured_ms']['mean'],.9)
            self.assertAlmostEqual(stats['groups'][0]['metrics']['api_cost']['mean'],.05)

    def test_interruption_keeps_attempt_intent_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, bodies, http = self.run_completion(tmp,[KeyboardInterrupt(),response()])
            with http, self.assertRaises(KeyboardInterrupt):
                runner(self.messages)
            saved = json.loads((Path(tmp)/'attempts.json').read_text())
            self.assertEqual(len(saved),1)
            self.assertEqual(saved[0]['status'],'started')
            self.assertEqual(saved[0]['model_calls'],1)
            self.assertEqual(len(bodies),1)

    def test_checkpoint_intent_survives_interruption_and_restart_cannot_repay(self):
        import test_checkpoints as fixture
        spec = importlib.util.spec_from_file_location('taso.strategy', ROOT/'TASO/python/taso/strategy.py')
        strategy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy)
        fake = types.ModuleType('taso')
        fake.operator_attrs = {}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'taso':fake,'taso.strategy':strategy}):
            directory = Path(tmp)
            from prepare_context import prepare_context
            prepare_context(ROOT/'TASO',directory)
            context = directory/'static_context.json'
            controller = checkpoint_strategy.CheckpointStrategist('google/gemini-3.8-flash',self.cfg,
                context,'actual builder',directory)
            calls = iter([response('length', content='{'),KeyboardInterrupt()])
            def http(*args,**kwargs):
                value = next(calls)
                if isinstance(value,BaseException):
                    raise value
                return io.StringIO(json.dumps(value))
            with patch.object(openrouter_client.urllib.request,'urlopen',side_effect=http) as mocked:
                with self.assertRaises(KeyboardInterrupt):
                    fixture.optimize_with_checkpoints(fixture.Graph(),controller,budget=100,checkpoint_interval=25)
                saved = json.loads((directory/'checkpoints/expansion_0000/generation.json').read_text())
                self.assertEqual(saved['model_calls'],2)
                self.assertEqual(len(saved['attempts']),2)
                again = checkpoint_strategy.CheckpointStrategist('google/gemini-3.8-flash',self.cfg,
                    context,'actual builder',directory)
                with self.assertRaisesRegex(RuntimeError,'already attempted'):
                    fixture.optimize_with_checkpoints(fixture.Graph(),again,budget=100,checkpoint_interval=25)
                self.assertEqual(mocked.call_count,2)


if __name__ == '__main__':
    unittest.main()

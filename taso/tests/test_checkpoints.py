"""Online checkpoint contract, API accounting and integration without a GPU/key."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
sys.path.insert(0, str(ROOT / 'llm_context'))
import checkpoint_strategy
import checkpoint_worker
import openrouter_client
import results
import run_suite
import worker

spec = importlib.util.spec_from_file_location('taso.strategy', ROOT / 'TASO/python/taso/strategy.py')
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)
# Exercise the public wrapper's actual source without importing ONNX/native CUDA.
tree = ast.parse((ROOT / 'TASO/python/taso/__init__.py').read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'optimize_with_checkpoints')
namespace = {'__name__': 'taso', '__package__': 'taso'}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'public_checkpoint_wrapper', 'exec'), namespace)
optimize_with_checkpoints = namespace['optimize_with_checkpoints']


def strategy(chunk):
    return {'schema_version': 1, 'name': 'all', 'phases': [
        {'name': 'all', 'rule_ids': 'all', 'expansion_budget': chunk}]}


def progress(consumed, budget, interval):
    phases = []
    for offset in range(0, budget, interval):
        used = min(interval, max(0, consumed - offset))
        phases.append({'index': len(phases), 'budget': min(interval, budget - offset),
                       'expansions': used, 'rules': [{'rule_index': 0, 'attempts': used,
                           'admitted': used * 2, 'duplicates': used, 'seconds': .001}]})
    return {'rule_count': 151, 'initial_cost_ms': 10, 'best_search_cost_ms': 10 - consumed * .01,
            'expansions': consumed, 'queue_remaining': consumed + 1,
            'stop_reason': 'checkpoint', 'phases': phases}


class Graph:
    def __init__(self, cost=10):
        self.value = cost
        self.positions = []
    def cost(self): return self.value
    def get_operator_list(self): return []
    def run_time(self): return .9
    def optimize_checkpointed(self, alpha, budget, interval, choose, print_subst, trace_file):
        callbacks = 0
        for consumed in range(0, budget, interval):
            self.positions.append(consumed)
            started = time.perf_counter()
            choose(Graph(10 - consumed * .01), json.dumps(progress(consumed, budget, interval)))
            callbacks += time.perf_counter() - started
        final = progress(budget, budget, interval)
        final.update(stop_reason='budget', checkpoint_callback_seconds=callbacks)
        if trace_file:
            worker.save(Path(trace_file), final)
        return Graph(10 - budget * .01)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'benchmarks/config.json').read_text())
        self.cfg.update(search_mode='checkpoint', checkpoint_interval=25)
        fake = types.ModuleType('taso')
        fake.operator_attrs = {}
        self.modules = patch.dict(sys.modules, {'taso': fake, 'taso.strategy': strategy_module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def context(self, directory):
        worker.save(directory / 'static_context.json', {'catalog': {
            'rules': [{'rule_id': 'builtin:000'}]}, 'source_context': []})
        (directory / 'model_source.py').write_text('Actual graph construction source')
        return directory / 'static_context.json'

    def response(self, chunk):
        return io.StringIO(json.dumps({'id': 'request-fixture', 'model': 'google/gemini-3.8-flash',
            'provider': 'fixture', 'usage': {'prompt_tokens': 100, 'completion_tokens': 20,
                'total_tokens': 120, 'cost': .01,
                'completion_tokens_details': {'reasoning_tokens': 10}},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(strategy(chunk))}}]}))

    def test_actual_context_extraction_keeps_catalog_ids_and_includes_shared_loop(self):
        from prepare_context import prepare_context, rule_catalog
        catalog = rule_catalog(ROOT / 'TASO')
        self.assertEqual(catalog['counts'], {'builtin': 18, 'protobuf': 133, 'total': 151})
        self.assertEqual([r['taso_xfer_index'] for r in catalog['rules']], list(range(151)))
        self.assertEqual(catalog['rules'][0]['rule_id'], 'builtin:000')
        self.assertEqual(catalog['rules'][-1]['rule_id'], 'protobuf:0132')
        with tempfile.TemporaryDirectory() as tmp:
            prepare_context(ROOT / 'TASO', tmp)
            context = json.loads((Path(tmp) / 'static_context.json').read_text())
            source = '\n'.join(context['source_context'][0]['excerpts'])
            self.assertIn('Graph::optimize_search(', source)
            self.assertIn('Graph::optimize_checkpointed(', source)

    def test_shared_wrapper_preserves_exact_budget_and_configurable_intervals(self):
        for interval in (25, 10, 5, 200):
            with self.subTest(interval=interval):
                graph, chunks = Graph(), []
                def select(live, state):
                    chunks.append(state['next_expansion_budget'])
                    return strategy(chunks[-1])
                result = optimize_with_checkpoints(graph, select, budget=63, checkpoint_interval=interval)
                self.assertEqual(graph.positions, list(range(0, 63, interval)))
                self.assertEqual(sum(chunks), 63)
                self.assertAlmostEqual(result.cost(), 9.37)
        with self.assertRaisesRegex(ValueError, 'exact next-chunk budget'):
            optimize_with_checkpoints(Graph(), lambda *args: strategy(100), checkpoint_interval=25)
        for interval in (0, -1, True, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                optimize_with_checkpoints(Graph(), lambda *args: strategy(1), checkpoint_interval=interval)

    def test_real_controller_sends_actual_feedback_and_sums_all_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fake-never-save'}):
            directory = Path(tmp)
            self.cfg['budget'] = 63
            context = self.context(directory)
            controller = checkpoint_strategy.CheckpointStrategist(
                'google/gemini-3.8-flash', self.cfg, context, 'Actual model source', directory)
            responses = iter([25, 25, 13])
            bodies = []
            def http(request, **kwargs):
                bodies.append(json.loads(request.data))
                return self.response(next(responses))
            with patch.object(openrouter_client.urllib.request, 'urlopen', side_effect=http):
                optimize_with_checkpoints(Graph(), controller, budget=63, checkpoint_interval=25)
            self.assertEqual(controller.model_calls, 3)
            self.assertEqual([r['expansions'] for r in controller.records], [0, 25, 50])
            feedback = json.loads((directory / 'checkpoints/expansion_0025/feedback.json').read_text())
            self.assertEqual(feedback['budget_remaining'], 38)
            self.assertAlmostEqual(feedback['best_search_cost_ms'], 9.75)
            self.assertEqual(feedback['cumulative_counters']['admitted'], 50)
            self.assertEqual(len(feedback['previous_decisions']), 1)
            self.assertEqual(len({json.dumps(b['messages']) for b in bodies}), 3)
            self.assertIn('NEXT checkpoint interval only', bodies[0]['messages'][1]['content'])
            self.assertNotIn('There are no LLM calls', bodies[0]['messages'][1]['content'])
            metadata = controller.aggregate()
            self.assertEqual(metadata['usage']['total_tokens'], 360)
            self.assertEqual(metadata['usage']['completion_tokens_details']['reasoning_tokens'], 30)
            self.assertAlmostEqual(metadata['usage']['cost'], .03)
            self.assertTrue(all(b['temperature'] == .2 and b['reasoning']['effort'] == 'medium' for b in bodies))
            for path in directory.rglob('*.json'):
                self.assertNotIn('fake-never-save', path.read_text())

    def test_dry_run_exercises_checkpoint_callbacks_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            context = self.context(directory)
            with patch.object(openrouter_client.urllib.request, 'urlopen') as http:
                controller = checkpoint_strategy.CheckpointStrategist(
                    'google/gemini-3.8-flash', self.cfg, context, 'source', directory,
                    client_factory=lambda *a, **kw: self.fail('Dry run constructed an API client'), dry_run=True)
                optimize_with_checkpoints(Graph(), controller, budget=100, checkpoint_interval=25)
                http.assert_not_called()
            self.assertEqual(controller.model_calls, 0)
            self.assertEqual([r['expansions'] for r in controller.records], [0, 25, 50, 75])
            self.assertTrue(all(r['status'] == 'dry_run' for r in controller.records))
            self.assertTrue((directory / 'checkpoints/expansion_0025/graph.json').exists())

    def test_failed_checkpoint_stops_and_is_not_paid_again(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fake-key'}):
            directory = Path(tmp)
            context = self.context(directory)
            controller = checkpoint_strategy.CheckpointStrategist(
                'google/gemini-3.8-flash', self.cfg, context, 'source', directory)
            invalid = io.StringIO(json.dumps({'choices': [
                {'finish_reason': 'length', 'message': {'content': '{'}}]}))
            with patch.object(openrouter_client.urllib.request, 'urlopen',
                              side_effect=[self.response(25), invalid]) as http:
                with self.assertRaisesRegex(RuntimeError, 'Incomplete completion'):
                    optimize_with_checkpoints(Graph(), controller, budget=100, checkpoint_interval=25)
                self.assertEqual(http.call_count, 2)
                again = checkpoint_strategy.CheckpointStrategist(
                    'google/gemini-3.8-flash', self.cfg, context, 'source', directory)
                with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                    optimize_with_checkpoints(Graph(), again, budget=100, checkpoint_interval=25)
                self.assertEqual(http.call_count, 2)
            self.assertEqual(controller.model_calls, 2)
            self.assertEqual(controller.records[-1]['status'], 'failed')

    def test_paired_worker_uses_no_paid_warmup_and_reports_wall_time(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fake-key'}):
            directory = Path(tmp)
            self.context(directory)
            baseline_calls = []
            def optimize_static(graph, policy, *, alpha, trace_file, print_subst):
                baseline_calls.append(policy)
                trace = progress(100, 100, 100)
                trace.update(stop_reason='budget', checkpoint_callback_seconds=0)
                worker.save(Path(trace_file), trace)
                return Graph(9)
            fake_ts = types.SimpleNamespace(optimize_with_strategy=optimize_static,
                                             optimize_with_checkpoints=optimize_with_checkpoints)
            args = types.SimpleNamespace(prepared=directory, model='google/gemini-3.8-flash', out=directory, trial=2)
            report = {}
            with patch.object(openrouter_client.urllib.request, 'urlopen',
                              side_effect=lambda *args, **kwargs: self.response(25)) as http:
                checkpoint_worker.evaluate_checkpoint_trial(fake_ts, lambda: (Graph(), 'source', {}),
                                                             self.cfg, args, report)
                self.assertEqual(http.call_count, 4)
            self.assertEqual(len(baseline_calls), 3)  # Two free warm-ups, then baseline.
            self.assertEqual(report['evaluation_order'], ['llm', 'baseline'])
            self.assertEqual(report['model_calls'], 4)
            llm = report['variants']['llm']
            self.assertEqual(llm['expansions'], 100)
            self.assertEqual(llm['measured_ms'], .9)
            self.assertGreaterEqual(llm['optimization_wall_seconds'], llm['search_seconds'])
            self.assertGreater(llm['checkpoint_callback_seconds'], 0)
            row = results.flatten({'evaluation': report, 'inference': report['inference'],
                                   'status': 'completed', 'network': 'squeezenet', 'model': args.model})
            self.assertIn('llm_optimization_wall_seconds', row)
            self.assertAlmostEqual(row['api_cost'], .04)

    def test_plan_and_setup_support_25_10_5_without_api_calls(self):
        for interval, calls in ((25, 4), (10, 10), (5, 20)):
            with self.subTest(interval=interval), contextlib.redirect_stdout(io.StringIO()) as output, patch.object(
                    openrouter_client.urllib.request, 'urlopen') as http:
                self.assertEqual(run_suite.main(['--plan-only', '--model', 'google/gemini-3.8-flash',
                                                 '--checkpoint-interval', str(interval)]), 0)
                plan = json.loads(output.getvalue())
                self.assertEqual(plan['max_model_calls_per_trial'], calls)
                self.assertEqual(plan['max_model_calls'], calls * 15)
                self.assertEqual(plan['checkpoint_interval'], interval)
                http.assert_not_called()
        spec = importlib.util.spec_from_file_location('setup_checkpoint_fixture', ROOT / 'baseline/setup_and_run.py')
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        args = setup.parse_args(['--experiment', 'suite', '--checkpoint-interval'])
        self.assertEqual(args.checkpoint_interval, 25)
        command = setup.experiment_command(ROOT.parent, args, Path('/tmp/python'), Path('/tmp/out'))
        self.assertEqual(run_suite.parse_args([str(a) for a in command[2:]]).checkpoint_interval, 25)
        cfg = dict(self.cfg, checkpoint_interval=0)
        with self.assertRaisesRegex(ValueError, 'checkpoint_interval'):
            run_suite.validate_config(cfg)

    def test_suite_dry_run_paid_trial_resume_and_frozen_interval(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fake-key'}), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            context = self.context(directory)
            cfg = dict(self.cfg, networks=['squeezenet'], models=['google/gemini-3.8-flash'], trials=1)
            config = directory / 'input.json'
            worker.save(config, cfg)
            args = ['--config', str(config), '--batch-dir', str(directory / 'batch')]
            fake_ts = types.ModuleType('taso')
            fake_ts.operator_attrs = {}
            fake_ts.optimize_with_checkpoints = optimize_with_checkpoints
            def optimize_static(graph, policy, *, alpha, trace_file, print_subst):
                trace = progress(100, 100, 100)
                trace.update(stop_reason='budget', checkpoint_callback_seconds=0)
                worker.save(Path(trace_file), trace)
                return Graph(9)
            fake_ts.optimize_with_strategy = optimize_static
            def launch(stage, network, out, batch, cfg, repo, trial=1, model=None):
                command = ['--stage', stage, '--network', network, '--out', str(out),
                           '--repo', str(repo), '--config', str(batch / 'config.json'),
                           '--prepared', str(batch / 'prepared' / network), '--trial', str(trial)]
                if model:
                    command += ['--model', model]
                code = worker.main(command)
                report = json.loads((out / 'worker_result.json').read_text())
                self.assertEqual(code, 0, report.get('error'))
                return report
            from experiments import run_llm_strategy
            with patch.dict(sys.modules, {'taso': fake_ts}), patch.object(
                    worker, 'build_network', side_effect=lambda *a: (Graph(), 'actual source', {})), patch.object(
                    worker, 'environment', return_value={}), patch.object(
                    run_llm_strategy, 'ensure_context', return_value=context), patch.object(
                    run_suite, 'source_hashes', return_value={'fixture': 'frozen'}), patch.object(
                    run_suite, 'run_worker', side_effect=launch), patch.object(
                    results, 'plot_distributions'), patch.object(
                    openrouter_client.urllib.request, 'urlopen', side_effect=lambda *a, **kw: self.response(25)) as http:
                self.assertEqual(run_suite.main(args + ['--dry-run']), 0)
                http.assert_not_called()
                self.assertEqual(run_suite.main(args), 0)
                self.assertEqual(http.call_count, 4)
                self.assertEqual(run_suite.main(args), 0)
                self.assertEqual(http.call_count, 4)
                with self.assertRaisesRegex(ValueError, 'configuration or sources changed'):
                    run_suite.main(args + ['--checkpoint-interval', '10'])
                self.assertEqual(http.call_count, 4)
            reports = run_suite.trial_reports(directory / 'batch', cfg)
            self.assertEqual(reports[0]['status'], 'completed')
            self.assertEqual(reports[0]['model_calls'], 4)
            self.assertIn('llm_optimization_wall_seconds', (directory / 'batch/trials.csv').read_text())

    def test_interrupted_trial_keeps_attempt_counts_and_never_relaunches(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            batch = Path(tmp)
            model = 'google/gemini-3.8-flash'
            out = batch / 'trials' / run_suite.model_directory('squeezenet', model) / 'trial_01'
            out.mkdir(parents=True)
            worker.save(out / 'trial.json', {'status': 'evaluating', 'model_calls': 0})
            worker.save(out / 'checkpoint_execution.json', {'status': 'started'})
            worker.save(out / 'checkpoints.json', [{'model_calls': 1, 'status': 'started'}])
            worker.save(out / 'worker_result.json', {'status': 'started', 'model_calls': 0})
            with patch.object(run_suite, 'run_worker', side_effect=AssertionError('Must not restart')):
                report = run_suite.run_trial('squeezenet', model, 1, batch, self.cfg, ROOT / 'TASO')
                self.assertEqual(report['status'], 'failed_evaluation')
                self.assertEqual(report['model_calls'], 1)
                self.assertEqual(report['interrupted_attempts'], 1)
                self.assertEqual(run_suite.run_trial('squeezenet', model, 1, batch, self.cfg, ROOT / 'TASO'), report)


if __name__ == '__main__':
    unittest.main()

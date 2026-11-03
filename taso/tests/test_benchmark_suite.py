"""Graph shapes, paired exports, resume semantics and CLI without a GPU/API key."""
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import networks
import results
import run_suite
import worker
import openrouter_client


class Tensor:
    def __init__(self, dims): self.dims = tuple(dims)
    def dim(self, i): return self.dims[i]


class ShapeGraph:
    def __init__(self):
        self.ops = []
        self.parameters = 0
        self.last = None
    def new_input(self, *, dims): return Tensor(dims)
    def new_weight(self, *, dims):
        self.parameters += math.prod(dims)
        return Tensor(dims)
    def output(self, name, dims):
        self.ops.append(name)
        self.last = Tensor(dims)
        assert all(v > 0 for v in dims)
        return self.last
    def conv2d(self, *, input, weight, strides, padding, activation='NONE'):
        n, c, h, w = input.dims
        out, per_group, kh, kw = weight.dims
        assert c % per_group == 0 and out % (c // per_group) == 0
        dims = (n, out, *[math.ceil(size / stride) if padding == 'SAME' else
                         (size-kernel)//stride + 1
                         for size, kernel, stride in zip((h, w), (kh, kw), strides)])
        return self.output('Conv', dims)
    def maxpool2d(self, *, input, kernels, strides, padding):
        n, c, h, w = input.dims
        return self.output('Pool', (n, c, *[
            math.ceil(size / stride) if padding == 'SAME' else (size-kernel)//stride+1
            for size, kernel, stride in zip((h, w), kernels, strides)]))
    avgpool2d = maxpool2d
    def relu(self, x): return self.output('Relu', x.dims)
    def add(self, a, b):
        assert a.dims == b.dims, (a.dims, b.dims)
        return self.output('Add', a.dims)
    def concat(self, axis, tensors):
        dims = list(tensors[0].dims)
        assert all(all(x.dim(i) == dims[i] for i in range(len(dims)) if i != axis) for x in tensors)
        dims[axis] = sum(x.dim(axis) for x in tensors)
        return self.output('Concat', dims)


class BenchmarkTests(unittest.TestCase):
    def fake_modules(self):
        spec = importlib.util.spec_from_file_location('taso.strategy', ROOT / 'TASO/python/taso/strategy.py')
        strategy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy)
        return patch.dict(sys.modules, {'taso': types.ModuleType('taso'), 'taso.strategy': strategy,
                                       'numpy': types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda _: None))})

    def test_native_builders_shapes_and_no_example_search(self):
        with self.fake_modules():
            for name in networks.NETWORKS:
                with self.subTest(name=name):
                    ts = types.SimpleNamespace(new_graph=ShapeGraph)
                    graph, source, info = networks.build_network(name, ts, ROOT / 'TASO')
                    self.assertGreater(len(graph.ops), 40)
                    self.assertGreater(graph.parameters, 1_000_000)
                    self.assertEqual(info['weight_seed'], 521)
                    self.assertNotIn('.optimize(', source)
                    if name == 'squeezenet':
                        self.assertEqual(graph.last.dims, (1, 1000, 1, 1))
                        self.assertEqual(graph.parameters, 1_231_552)
                    if name == 'resnext50':
                        self.assertEqual(graph.last.dims, (1, 2048, 7, 7))

    def test_distribution_keeps_values_and_pairing_including_failures(self):
        reports = []
        for trial, value in enumerate([.8, .9, 1, 1.1], 1):
            reports.append({'network': 'squeezenet', 'model': 'fake', 'trial': trial,
                            'status': 'completed', 'budget': 100, 'alpha': 1.05,
                            'evaluation': {'variants': {
                                'baseline': {'estimated_ms': trial * 10, 'measured_ms': 2},
                                'llm': {'estimated_ms': value * trial * 10, 'measured_ms': value * 2}}}})
        reports.append({'network': 'squeezenet', 'model': 'fake', 'trial': 5,
                        'status': 'failed_generation', 'error': 'invalid JSON'})
        with tempfile.TemporaryDirectory() as temp:
            stats = results.write_exports(temp, reports, plots=False)
            group = stats['groups'][0]
            self.assertEqual((group['completed'], group['failed'], group['planned_trials']), (4, 1, 5))
            metric = group['metrics']['llm_over_baseline_estimated_ms']
            self.assertEqual(metric['n'], 4)
            self.assertAlmostEqual(metric['median'], .95)
            self.assertAlmostEqual(metric['q1'], .875)
            self.assertAlmostEqual(metric['q3'], 1.025)
            self.assertEqual([s['trial'] for s in metric['samples']], [1, 2, 3, 4])
            self.assertEqual((group['estimated_cost_wins'], group['estimated_cost_ties'], group['estimated_cost_losses']), (2, 1, 1))
            self.assertEqual(len((Path(temp) / 'trials.jsonl').read_text().splitlines()), 5)
            self.assertIn('failed_generation', (Path(temp) / 'trials.csv').read_text())
            self.assertIsNone(results.distribution([(1, 4)])['sample_stddev'])

    def fake_worker(self, stage, network, directory, batch, cfg, repo, trial=1):
        directory.mkdir(parents=True, exist_ok=True)
        if stage == 'prepare':
            worker.save(directory / 'request.json', [{'role': 'user', 'content': 'same frozen graph'}])
            worker.save(directory / 'request_metadata.json', {'request_sha256': 'same-for-five'})
            worker.save(directory / 'catalog.json', {'rules': [{'rule_id': 'builtin:000'}]})
            report = {'status': 'completed'}
        else:
            report = {'status': 'completed', 'variants': {
                'baseline': {'estimated_ms': 10, 'measured_ms': 11, 'search_seconds': 2, 'expansions': 100},
                'llm': {'estimated_ms': 9, 'measured_ms': 10, 'search_seconds': 1, 'expansions': 80}}}
        worker.save(directory / 'worker_result.json', report)
        return report

    def test_five_independent_calls_frozen_prompt_and_resume_without_new_calls(self):
        strategy = {'schema_version': 1, 'name': 'all', 'phases': [
            {'name': 'all', 'rule_ids': 'all', 'expansion_budget': 100}]}
        response = {'id': 'mock', 'model': 'fake', 'usage': {'cost': .01},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(strategy)}}]}
        with tempfile.TemporaryDirectory() as temp, self.fake_modules(), contextlib.redirect_stdout(io.StringIO()):
            cfg = json.loads((ROOT / 'benchmarks/config.json').read_text())
            cfg.update(networks=['squeezenet'])
            config = Path(temp) / 'input.json'
            worker.save(config, cfg)
            args = ['--batch-dir', str(Path(temp) / 'batch'), '--config', str(config)]
            calls = []
            def http(request, **kwargs):
                calls.append(json.loads(request.data))
                return io.StringIO(json.dumps(dict(response, model=calls[-1]['model'])))
            with patch.object(run_suite, 'run_worker', side_effect=self.fake_worker), patch.object(
                    run_suite, 'source_hashes', return_value={'fixture': 'fixed'}), patch.dict(
                    os.environ, {'OPENROUTER_API_KEY': 'fake-key'}), patch.object(
                    openrouter_client.urllib.request, 'urlopen', side_effect=http), patch.object(results, 'plot_distributions'):
                self.assertEqual(run_suite.main(args), 0)
                self.assertEqual(len(calls), 20)
                self.assertEqual(len({json.dumps(c['messages']) for c in calls}), 1)
                for model in cfg['models']:
                    requests = [c for c in calls if c['model'] == model]
                    self.assertEqual(len(requests), 5)
                    for body in requests:
                        self.assertEqual(body['reasoning']['effort'],
                                         'high' if model.startswith(('deepseek/', 'moonshotai/')) else 'medium')
                        self.assertEqual('temperature' in body, model.startswith(('deepseek/', 'google/')))
                self.assertEqual(run_suite.main(args), 0)
                self.assertEqual(len(calls), 20)
            reports = run_suite.trial_reports(Path(temp) / 'batch', cfg)
            self.assertEqual([r['trial'] for r in reports], [1, 2, 3, 4, 5] * 4)
            self.assertTrue(all(r['model_calls'] == 1 for r in reports))

    def test_plan_shows_every_model_setting_without_api_or_gpu(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(
                openrouter_client.urllib.request, 'urlopen') as http, patch.object(
                run_suite, 'run_worker') as gpu:
            self.assertEqual(run_suite.main(['--plan-only']), 0)
            http.assert_not_called()
            gpu.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['max_model_calls'], 60)
        self.assertEqual(plan['trials_per_pair'], 5)
        self.assertEqual(plan['budget'], 100)
        self.assertEqual(set(plan['networks']), {'squeezenet', 'resnext50', 'inceptionv3'})
        self.assertEqual(set(plan['models']), {
            'deepseek/deepseek-v4-pro', 'moonshotai/kimi-k3.1-pro',
            'openai/gpt-5.3-codex', 'google/gemini-3.8-flash'})
        for model, settings in plan['model_settings'].items():
            high = model.startswith(('deepseek/', 'moonshotai/'))
            self.assertEqual(settings['reasoning_effort'], 'high' if high else 'medium')
            self.assertEqual(settings['requested_temperature'], .2)
            self.assertEqual(settings['temperature'], .2 if model.startswith(('deepseek/', 'google/')) else None)

    def test_export_distinguishes_requested_and_sent_settings(self):
        inference = openrouter_client.resolve_settings(
            'moonshotai/kimi-k3.1-pro', .2, 'medium',
            {'moonshotai/kimi-k3.1-pro': 'high'})
        inference['usage'] = {'completion_tokens_details': {'reasoning_tokens': 42}}
        row = results.flatten({'inference': inference})
        self.assertEqual(row['requested_temperature'], .2)
        self.assertIsNone(row['temperature'])
        self.assertEqual(row['requested_reasoning_effort'], 'medium')
        self.assertEqual(row['reasoning_effort'], 'high')
        self.assertEqual(row['api_reasoning_tokens'], 42)
        self.assertIn('temperature', json.loads(row['omitted_parameters']))

    def test_failed_calls_are_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temp, self.fake_modules(), contextlib.redirect_stdout(io.StringIO()):
            batch = Path(temp)
            cfg = json.loads((ROOT / 'benchmarks/config.json').read_text())
            self.fake_worker('prepare', 'squeezenet', batch / 'prepared/squeezenet', batch, cfg, ROOT / 'TASO')
            client = types.SimpleNamespace(request_attempts=1, metadata={'usage': {'cost': .01}})
            class BadClient:
                request_attempts = client.request_attempts
                metadata = client.metadata
                def __init__(self, *args, **kwargs): pass
                def __call__(self, messages): return 'invalid json'
            with patch.object(openrouter_client, 'OpenRouterClient', BadClient), patch.dict(
                    os.environ, {'OPENROUTER_API_KEY': 'fake-key'}):
                first = run_suite.run_trial('squeezenet', 'fake', 1, batch, cfg, ROOT / 'TASO')
            self.assertEqual(first['status'], 'failed_generation')
            with patch.object(openrouter_client, 'OpenRouterClient', side_effect=AssertionError('Unexpected API retry')):
                self.assertEqual(run_suite.run_trial('squeezenet', 'fake', 1, batch, cfg, ROOT / 'TASO'), first)
                self.assertEqual(run_suite.run_trial('squeezenet', 'fake', 1, batch, cfg, ROOT / 'TASO', True), first)

    def test_setup_suite_dispatch_arguments_parse(self):
        spec = importlib.util.spec_from_file_location('setup_runner_suite', ROOT / 'baseline/setup_and_run.py')
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        args = setup.parse_args(['--experiment', 'suite', '--batch-dir', '/tmp/batch', '--dry-run',
                                 '--model', 'fake', '--suite-config', '/tmp/config.json'])
        command = setup.experiment_command(ROOT.parent, args, Path('/tmp/python'), Path('/tmp/results'))
        suite_args = run_suite.parse_args([str(a) for a in command[2:]])
        self.assertIsNone(suite_args.max_tokens)  # Suite config owns its 32768 default.
        self.assertEqual(suite_args.batch_dir, Path('/tmp/batch').resolve())
        self.assertEqual(suite_args.config, Path('/tmp/config.json').resolve())
        self.assertTrue(suite_args.dry_run)

    def test_gpu_worker_warms_both_policies_and_alternates_timed_order(self):
        class Graph:
            def __init__(self): self.consumed = False
            def cost(self):
                assert not self.consumed
                return 1.0
            def get_operator_list(self):
                assert not self.consumed
                return []
            def run_time(self): return .9
        calls = []
        def optimizer(graph, policy, *, alpha, trace_file, print_subst):
            graph.consumed = True
            calls.append(policy['name'])
            worker.save(trace_file, {'rule_count': 151, 'expansions': 100, 'queue_remaining': 1,
                        'stop_reason': 'budget', 'phases': [{'rules': [{'rule_index': 0, 'admitted': 101}]}]})
            return Graph()
        fake_ts = types.SimpleNamespace(optimize_with_strategy=optimizer)
        with tempfile.TemporaryDirectory() as temp, self.fake_modules(), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temp)
            cfg = json.loads((ROOT / 'benchmarks/config.json').read_text())
            worker.save(directory / 'config.json', cfg)
            worker.save(directory / 'catalog.json', {'rules': [{'rule_id': 'builtin:000'}]})
            worker.save(directory / 'strategy.json', {'schema_version': 1, 'name': 'llm', 'phases': [
                {'name': 'all', 'rule_ids': 'all', 'expansion_budget': 100}]})
            with patch.dict(sys.modules, {'taso': fake_ts}), patch.object(worker, 'environment', return_value={}), patch.object(
                    worker, 'inspect_binding', return_value={'status': 'passed'}), patch.object(
                    worker, 'build_network', side_effect=lambda *a: (Graph(), 'source', {'network': 'squeezenet'})):
                code = worker.main(['--stage', 'evaluate', '--network', 'squeezenet', '--repo', str(ROOT / 'TASO'),
                                    '--out', temp, '--config', str(directory / 'config.json'),
                                    '--prepared', temp, '--trial', '2'])
            self.assertEqual(code, 0)
            self.assertEqual(calls, ['llm', 'all_rules', 'llm', 'all_rules'])
            report = json.loads((directory / 'worker_result.json').read_text())
            self.assertEqual(report['evaluation_order'], ['llm', 'baseline'])
            self.assertEqual(report['variants']['baseline']['expansions'], 100)
            self.assertEqual(report['variants']['llm']['measured_ms'], .9)

    def test_setup_resolves_relative_paths_before_child_changes_directory(self):
        spec = importlib.util.spec_from_file_location('setup_relative_paths', ROOT / 'baseline/setup_and_run.py')
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        with tempfile.TemporaryDirectory() as temp, contextlib.chdir(temp):
            args = setup.parse_args(['--experiment', 'suite', '--batch-dir', 'taso/baseline/results/batch',
                                     '--suite-config', 'custom.json'])
            command = setup.experiment_command(ROOT.parent, args, Path('/tmp/python'), Path('/tmp/results'))
            parsed = run_suite.parse_args([str(a) for a in command[2:]])
            self.assertEqual(parsed.batch_dir, Path(temp).resolve() / 'taso/baseline/results/batch')
            self.assertEqual(parsed.config, Path(temp).resolve() / 'custom.json')
            self.assertTrue(parsed.batch_dir.is_absolute())
            args = setup.parse_args(['--experiment', 'llm', '--strategy-file', 'saved.json'])
            command = [str(a) for a in setup.experiment_command(ROOT.parent, args, Path('/tmp/python'), Path('/tmp/results'))]
            self.assertEqual(command[command.index('--strategy-file') + 1], str(Path(temp).resolve() / 'saved.json'))

    def test_interrupted_generation_does_not_repeat_the_request(self):
        with tempfile.TemporaryDirectory() as temp, self.fake_modules(), contextlib.redirect_stdout(io.StringIO()):
            batch = Path(temp)
            cfg = json.loads((ROOT / 'benchmarks/config.json').read_text())
            self.fake_worker('prepare', 'squeezenet', batch / 'prepared/squeezenet', batch, cfg, ROOT / 'TASO')
            directory = batch / 'trials' / run_suite.model_directory('squeezenet', 'fake') / 'trial_01'
            directory.mkdir(parents=True)
            worker.save(directory / 'trial.json', {'status': 'generating', 'network': 'squeezenet',
                         'model': 'fake', 'trial': 1, 'model_calls': 1})
            worker.save(directory / 'generation.json', {'status': 'started'})
            with patch.object(openrouter_client, 'OpenRouterClient', side_effect=AssertionError('Unexpected retry')):
                report = run_suite.run_trial('squeezenet', 'fake', 1, batch, cfg, ROOT / 'TASO')
            self.assertEqual(report['status'], 'failed_generation')
            self.assertEqual(report['model_calls'], 1)


if __name__ == '__main__':
    unittest.main()

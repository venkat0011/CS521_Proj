"""One validated LLM policy per reached expansion checkpoint; no paid retries."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from graph_context import graph_context
from llm_strategy import build_messages, validate_strategy
from model_settings import resolve_settings
from openrouter_client import OpenRouterClient


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class CheckpointStrategist:
    def __init__(self, model, cfg, context_path, source, out, client_factory=OpenRouterClient, dry_run=False):
        self.model, self.cfg = model, cfg
        self.context_path, self.source, self.out = Path(context_path), source, Path(out)
        self.client_factory, self.dry_run = client_factory, dry_run
        self.records = []
        self.out.mkdir(parents=True, exist_ok=True)
        self.settings = resolve_settings(model, cfg['temperature'], cfg.get('reasoning_effort'),
                                         cfg.get('reasoning_effort_overrides'))

    def aggregate(self):
        metadata = [record.get('inference') or {} for record in self.records]
        usage, reasoning = Counter(), 0
        for item in metadata:
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
                value = (item.get('usage') or {}).get(key)
                if isinstance(value, (int, float)):
                    usage[key] += value
            reasoning += ((item.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0
        if reasoning:
            usage['completion_tokens_details'] = {'reasoning_tokens': reasoning}
        def distinct(key):
            values = sorted({item[key] for item in metadata if item.get(key)})
            return ','.join(values) if values else None
        return {'requested_model': self.model, 'returned_model': distinct('returned_model'),
                'provider': distinct('provider'),
                'usage': dict(usage),
                'request_ids': [item['request_id'] for item in metadata if item.get('request_id')],
                'inference_seconds': sum(item.get('inference_seconds', 0) or 0 for item in metadata),
                'max_tokens': self.cfg['max_tokens'],
                **{k: v for k, v in self.settings.items() if k != 'parameters'}}

    @property
    def model_calls(self):
        return sum(record.get('model_calls', 0) for record in self.records)

    def __call__(self, graph, progress):
        consumed = progress['expansions']
        interval, budget = self.cfg['checkpoint_interval'], self.cfg['budget']
        if consumed != len(self.records) * interval or consumed >= budget:
            raise ValueError('Unexpected checkpoint position; refusing an extra model call')
        chunk = min(interval, budget - consumed)
        directory = self.out / 'checkpoints' / f'expansion_{consumed:04d}'
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / 'generation.json'
        if marker.exists():
            raise RuntimeError('Checkpoint already attempted; no automatic replacement API call. '
                               'Search state is in memory only. Use a new trial/batch after interruption.')
        save(directory / 'progress.json', progress)
        save(directory / 'graph.json', graph_context(graph))
        completed = [p for p in progress['phases'] if p['expansions']]
        counters = Counter()
        for phase in completed:
            for rule in phase['rules']:
                counters.update({k: v for k, v in rule.items() if k not in ('rule_index', 'seconds')})
        previous_cost = self.records[-1]['best_search_cost_ms'] if self.records else progress['initial_cost_ms']
        feedback = {k: progress[k] for k in ('expansions', 'total_budget', 'budget_remaining',
                    'next_expansion_budget', 'queue_remaining', 'initial_cost_ms', 'best_search_cost_ms')}
        feedback.update(
            improvement_since_previous_checkpoint_ms=previous_cost - progress['best_search_cost_ms'],
            cumulative_counters=dict(counters),
            last_interval_rules=[r for r in completed[-1]['rules'] if r['attempts']] if completed else [],
            previous_decisions=[{'at_expansion': r['expansions'], 'strategy': r['strategy']}
                                for r in self.records if 'strategy' in r],
            feedback_type='estimated_cost_only; no GPU runtime observations')
        messages, catalog = build_messages(graph, self.context_path, chunk, self.cfg['alpha'],
                                           1, self.source, checkpoint_feedback=feedback)
        save(directory / 'feedback.json', feedback)
        save(directory / 'request.json', messages)
        save(directory / 'request_metadata.json', {
            'request_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
            'expansions': consumed, 'next_expansion_budget': chunk})
        if self.dry_run:
            # Exercise every reached boundary using a deterministic all-rule policy.
            # These are preflight observations, not predictions of later LLM feedback.
            strategy = {'schema_version': 1, 'name': 'all_rules_preflight', 'phases': [
                {'name': 'all', 'rule_ids': 'all', 'expansion_budget': chunk}]}
            save(directory / 'strategy.json', strategy)
            self.records.append({'expansions': consumed, 'next_expansion_budget': chunk,
                'best_search_cost_ms': progress['best_search_cost_ms'],
                'queue_remaining': progress['queue_remaining'], 'status': 'dry_run',
                'strategy': strategy, 'model_calls': 0})
            save(self.out / 'checkpoints.json', self.records)
            return strategy
        print(f'[checkpoint {consumed}/{budget}] requesting policy for next {chunk} expansions', flush=True)
        client = self.client_factory(self.model, save_to=directory,
            max_tokens=self.cfg['max_tokens'], temperature=self.cfg['temperature'],
            reasoning_effort=self.cfg.get('reasoning_effort'),
            reasoning_effort_overrides=self.cfg.get('reasoning_effort_overrides'))
        record = {'expansions': consumed, 'next_expansion_budget': chunk,
                  'best_search_cost_ms': progress['best_search_cost_ms'],
                  'queue_remaining': progress['queue_remaining'], 'status': 'started', 'model_calls': 1}
        self.records.append(record)
        # Persist intent BEFORE contacting the provider, including for interrupted requests.
        save(marker, record)
        save(self.out / 'checkpoints.json', self.records)
        started = time.perf_counter()
        try:
            raw = client(messages)
            save(directory / 'response.json', {'raw_response': raw})
            strategy = validate_strategy(raw, catalog, chunk, 1, self.cfg['alpha'])
            save(directory / 'strategy.json', strategy)
            record.update(status='completed', strategy=strategy)
            return strategy
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record.update(model_calls=client.request_attempts, inference=client.metadata)
            # HTTP failures have no response usage; retain measured request latency.
            if client.request_attempts and client.metadata is None:
                record['inference'] = {'inference_seconds': time.perf_counter() - started}
            save(marker, record)
            save(self.out / 'checkpoints.json', self.records)
            save(self.out / 'inference_metadata.json', self.aggregate())

"""Bounded continuation for one paused TASO checkpoint; no cross-checkpoint history."""
from collections import Counter
import copy
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import time

from openrouter_client import IncompleteCompletion, OpenRouterClient, ProviderFailure, StrategyUnavailable

POLICY_VERSION = 'checkpoint_continuation_v2'
CHECKED_ON = '2026-11-03'
# Model-level limits, with no provider selection. DeepSeek's output cap is
# corrected from an advertised 943718 to the backend-validated 393216 range
# reported in the user's HTTP 400. OpenRouter chooses the serving provider.
MODEL_LIMITS = {
    'deepseek/deepseek-v4-pro': {'max_output': 393216, 'context': 1048576,
        'max_input': None},
    'moonshotai/kimi-k3.1-pro': {'max_output': 943718, 'context': 1048576,
        'max_input': None},
    'openai/gpt-5.3-codex': {'max_output': 128000, 'context': 400000,
        'max_input': 272000},
    'google/gemini-3.8-flash': {'max_output': 65536, 'context': 1048576,
        'max_input': None},
}
MARGIN = 4096
MIN_OUTPUT = 1024
CONTINUE = (
    'The previous response reached its output-token limit. Continue the same TASO '
    'checkpoint decision using the work supplied above. Do not repeat completed analysis. '
    'Return exactly one COMPLETE strategy JSON object satisfying the original schema '
    'and next-interval expansion budget. Replace any unfinished JSON with a complete '
    'object; do not return only a suffix, a code fence, or explanatory prose. '
    'Do not change the optimization objective or invent rule IDs.'
)
RETRY_STRATEGY = CONTINUE.replace(
    'The previous response reached its output-token limit.',
    'The previous response did not supply a complete valid strategy.')


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def message_bytes(messages):
    return len(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode())


def output_cap(limits, requested, estimated_input):
    """Conservative guard, not a tokenizer or a guarantee against provider rejection."""
    if limits['max_input'] is not None and estimated_input + MARGIN > limits['max_input']:
        raise RuntimeError('Continuation exceeds the guarded input-token allowance; no request sent')
    remaining = limits['context'] - estimated_input - MARGIN
    cap = min(limits['max_output'], remaining)
    if type(requested) is int:
        cap = min(cap, requested)
    elif requested != 'max':
        raise ValueError('Continuation requires max_tokens="max" or an explicit positive integer')
    if cap < MIN_OUTPUT:
        raise RuntimeError('Insufficient guarded context space for another completion; no request sent')
    return cap


def extend_history(messages, result, model, reason='length'):
    """Reuse readable work across automatic routing; save opaque data in artifacts."""
    choices = result.get('choices') or []
    original = choices[0].get('message', {}) if choices else {}
    if original.get('tool_calls'):
        raise RuntimeError('Unexpected tool calls; this strategy protocol does not execute tools')
    content = original.get('content') or ''
    if not isinstance(content, str):
        raise RuntimeError('Unsupported partial content format; refusing a lossy continuation')
    details = original.get('reasoning_details')
    reasoning = original.get('reasoning') or original.get('reasoning_content') or ''
    if reasoning and not isinstance(reasoning, str):
        raise RuntimeError('Unsupported reasoning format')
    if details is not None and (not isinstance(details, list) or any(not isinstance(d, dict) for d in details)):
        raise RuntimeError('Unsupported reasoning_details format')
    details = details or []
    readable = reasoning or '\n'.join(d.get('text') or d.get('summary') or '' for d in details)
    history = copy.deepcopy(messages)
    instruction = CONTINUE if reason == 'length' else RETRY_STRATEGY
    if content:
        history.append({'role': 'assistant', 'content': content})
    # Do not replay provider-specific encrypted/signature blocks to an arbitrary
    # next provider. All original blocks remain unchanged in api_response.json.
    if readable:
        history.append({'role': 'user', 'content':
            'Unfinished work returned by your previous request (context for the same task):\n'
            + readable + '\n\n' + instruction})
        mode = 'returned_plaintext_as_context'
    else:
        history.append({'role': 'user', 'content': instruction})
        mode = 'partial_answer_context' if content.strip() else 'no_readable_work_retry'
    return history, mode


def aggregate_attempts(attempts, model, requested, settings=None):
    metadata = [a.get('inference') or {} for a in attempts if a.get('model_calls')]
    usage = Counter()
    reasoning = []
    for item in metadata:
        current = item.get('usage') or {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
            value = current.get(key)
            if type(value) in (int, float):
                usage[key] += value
        value = (current.get('completion_tokens_details') or {}).get('reasoning_tokens')
        if type(value) is int:
            reasoning.append(value)
    if reasoning:
        usage['completion_tokens_details'] = {'reasoning_tokens': sum(reasoning)}
    def distinct(key):
        values = sorted({item[key] for item in metadata if item.get(key)})
        return ','.join(values) if values else None
    return {'requested_model': model, 'returned_model': distinct('returned_model'),
        'provider': distinct('provider'), 'usage': dict(usage),
        'usage_observations': sum(bool(item.get('usage')) for item in metadata),
        'usage_missing_attempts': sum(not bool(item.get('usage')) for item in metadata),
        'request_ids': [item['request_id'] for item in metadata if item.get('request_id')],
        'inference_seconds': sum(item.get('inference_seconds', 0) or 0 for item in metadata),
        'max_tokens': requested, 'output_token_limit_mode': 'model_max' if requested == 'max' else 'explicit',
        'finish_reason_counts': dict(Counter(item['finish_reason'] for item in metadata if item.get('finish_reason'))),
        'visible_completion_tokens': (sum(item['visible_completion_tokens'] for item in metadata)
            if metadata and all(type(item.get('visible_completion_tokens')) is int for item in metadata) else None),
        'completion_policy': POLICY_VERSION,
        'continuation_requests': sum(a.get('action') == 'continue_after_length' for a in attempts if a.get('model_calls')),
        'provider_retries': sum(a.get('action') == 'retry_provider' for a in attempts if a.get('model_calls')),
        'strategy_retries': sum(a.get('action') == 'retry_strategy' for a in attempts if a.get('model_calls')),
        'provider_routing': 'openrouter_auto',
        'retry_wait_seconds': sum(a.get('retry_wait_seconds', 0) for a in attempts),
        'prompt_bytes_total': sum(a.get('prompt_bytes', 0) for a in attempts if a.get('model_calls')),
        **{k: v for k, v in (settings or {}).items() if k != 'parameters'}}


class CheckpointCompletion:
    def __init__(self, model, cfg, directory, client_factory=OpenRouterClient, on_update=None, sleep=time.sleep, validator=None):
        self.model, self.cfg, self.directory = model, cfg, Path(directory)
        self.client_factory, self.on_update, self.sleep = client_factory, on_update, sleep
        self.attempts, self.settings = [], None
        self.validator = validator or validate_json_object
        self.strategy = None
        self.limits = MODEL_LIMITS[model]
        retries = cfg.get('max_retries', 3)
        if type(retries) is not int or not 0 <= retries <= 3:
            raise ValueError('max_retries must be between 0 and 3')
        self.max_retries = retries

    @property
    def request_attempts(self):
        return sum(a.get('model_calls', 0) for a in self.attempts)

    @property
    def metadata(self):
        return aggregate_attempts(self.attempts, self.model, self.cfg['max_tokens'], self.settings)

    def persist(self):
        save(self.directory / 'attempts.json', self.attempts)
        save(self.directory / 'inference_metadata.json', self.metadata)
        if self.on_update:
            self.on_update(self)

    def __call__(self, messages):
        history = copy.deepcopy(messages)
        # UTF-8 serialized bytes plus margin are a deliberately conservative
        # initial estimate for these text requests, not an exact token count.
        estimated = message_bytes(history)
        action, replay, retry_reason = 'initial', None, None
        last_error = None
        for number in range(1, self.max_retries + 2):
            cap = output_cap(self.limits, self.cfg['max_tokens'], estimated)
            directory = self.directory / 'attempts' / f'attempt_{number:02d}'
            directory.mkdir(parents=True, exist_ok=True)
            client = self.client_factory(self.model, save_to=directory, max_tokens=cap,
                temperature=self.cfg['temperature'], reasoning_effort=self.cfg.get('reasoning_effort'),
                reasoning_effort_overrides=self.cfg.get('reasoning_effort_overrides'),
                timeout=self.cfg.get('request_timeout_seconds', 240))
            self.settings = getattr(client, 'settings', None)
            record = {'attempt': number, 'action': action, 'status': 'started', 'model_calls': 1,
                'prompt_bytes': message_bytes(history), 'estimated_input_tokens': estimated,
                'context_guard': 'conservative_estimate_not_tokenizer', 'max_tokens': cap,
                'provider_routing': 'openrouter_auto', 'limits_checked_on': CHECKED_ON,
                'limits': self.limits, 'replay_mode': replay, 'retry_reason': retry_reason}
            self.attempts.append(record)
            self.persist()  # intent is durable before contacting the provider
            print(f'  [attempt {number}/{self.max_retries+1}] {action}; output cap={cap}; '
                  'routing=OpenRouter automatic', flush=True)
            started = time.perf_counter()
            result, error = None, None
            try:
                text = client(history)
                try:
                    self.strategy = self.validator(text)
                except ValueError as exc:
                    raise StrategyUnavailable(f'Invalid strategy: {exc}',
                        getattr(client, 'last_response', None) or {}, 'invalid_strategy') from exc
                record['status'] = 'completed'
                return text
            except (IncompleteCompletion, ProviderFailure, StrategyUnavailable) as exc:
                error = last_error = exc
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                if isinstance(exc, (IncompleteCompletion, StrategyUnavailable)):
                    result = exc.response
                record['failure_kind'] = (exc.finish_reason if isinstance(exc, IncompleteCompletion)
                    else exc.reason if isinstance(exc, StrategyUnavailable) else 'provider_error')
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                record.update(model_calls=client.request_attempts, inference=client.metadata or {
                    'inference_seconds': time.perf_counter() - started})
                self.persist()
            if number == self.max_retries + 1:
                break
            if ((isinstance(error, IncompleteCompletion) and error.finish_reason == 'length')
                    or isinstance(error, StrategyUnavailable)):
                retry_reason = error.finish_reason if isinstance(error, IncompleteCompletion) else error.reason
                next_history, replay = extend_history(history, result, self.model, retry_reason)
                usage = (client.metadata or {}).get('usage') or {}
                prompt, output = usage.get('prompt_tokens'), usage.get('completion_tokens')
                # Count reported generated tokens conservatively, even if some
                # hidden tokens were not returned. Extra metadata/turn overhead
                # is guarded separately; no source/rule/history is dropped.
                extra = len(CONTINUE.encode()) + 512
                if replay == 'returned_plaintext_as_context':
                    extra += 256
                if replay == 'no_readable_work_retry':
                    estimated += message_bytes(next_history[len(history):])
                elif type(prompt) is int and type(output) is int and min(prompt, output) >= 0:
                    estimated = max(estimated, prompt) + output + extra
                else:
                    estimated = max(estimated, message_bytes(next_history))
                history = next_history
                action = 'continue_after_length' if retry_reason == 'length' and replay != 'no_readable_work_retry' else 'retry_strategy'
                continue
            if isinstance(error, ProviderFailure) and error.status in (429, 500, 502, 503, 504):
                delay = min(2 ** number, 30)
                if error.retry_after:
                    try:
                        retry_delay = float(error.retry_after)
                    except ValueError:
                        retry_delay = parsedate_to_datetime(error.retry_after).timestamp() - time.time()
                    if retry_delay > 60:
                        raise RuntimeError('Provider requested a retry delay over 60 seconds; stopping this checkpoint') from error
                    delay = max(delay, retry_delay)
                record['retry_delay_seconds'] = delay
                self.persist()
                waiting = time.perf_counter()
                try:
                    self.sleep(delay)
                finally:
                    record['retry_wait_seconds'] = time.perf_counter() - waiting
                    self.persist()
                action = 'retry_provider'
                retry_reason = 'provider_error'
                continue
            raise error
        raise RuntimeError(f'Checkpoint failed after {self.request_attempts} API attempts '
                           f'(at most {self.max_retries} follow-ups): {last_error}') from last_error


def validate_json_object(text):
    value = json.loads(text)
    if not isinstance(value, dict) or not value:
        raise ValueError('Expected a nonempty strategy JSON object')
    return value

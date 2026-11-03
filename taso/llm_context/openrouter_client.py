"""Small standard-library OpenRouter adapter; credentials come from the environment."""
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from model_settings import resolve_settings
from token_limits import limit_mode, normalize_max_tokens

DEFAULT_MODEL = 'deepseek/deepseek-v4-pro'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'


class IncompleteCompletion(RuntimeError):
    """A returned response, distinct from transport failure; usage is still billable."""
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response
        choices = response.get('choices') or []
        self.finish_reason = choices[0].get('finish_reason') if choices else None


class ProviderFailure(RuntimeError):
    def __init__(self, message, status=None, retry_after=None):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


class OpenRouterClient:
    def __init__(self, model=None, save_to=None, max_tokens=8192, temperature=0.2, timeout=240,
                 reasoning_effort=None, reasoning_effort_overrides=None, provider_tag=None):
        self.model = model or os.environ.get('OPENROUTER_MODEL', DEFAULT_MODEL)
        self.save_to = Path(save_to) if save_to is not None else None
        self.max_tokens = normalize_max_tokens(max_tokens)
        if self.max_tokens == 'max':
            raise ValueError('max must be resolved by the checkpoint continuation policy before an API call')
        self.temperature = temperature
        self.settings = resolve_settings(self.model, temperature, reasoning_effort,
                                         reasoning_effort_overrides)
        self.timeout = timeout
        self.metadata = None
        self.request_attempts = 0
        self.provider_tag = provider_tag

    def __call__(self, messages):
        key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        if not key:
            raise ValueError('Set OPENROUTER_API_KEY in your terminal; do not put it in source files.')
        body = {'model': self.model, 'messages': messages,
                'stream': False,
                'response_format': {'type': 'json_object'},
                'provider': {'require_parameters': True}}
        body.update(self.settings['parameters'])
        if self.provider_tag:
            body['provider'].update(only=[self.provider_tag], allow_fallbacks=False)
        if self.max_tokens is not None:
            body['max_tokens'] = self.max_tokens
        # Never persist authentication headers or the key.
        if self.save_to is not None:
            self.save_to.mkdir(parents=True, exist_ok=True)
            (self.save_to / 'api_request.json').write_text(json.dumps(body, indent=2) + '\n')
        request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                         headers={'Authorization': 'Bearer ' + key,
                                                  'Content-Type': 'application/json'})
        started = time.perf_counter()
        self.request_attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode('utf-8', errors='replace').replace(key, '[redacted]')
            finally:
                exc.close()
            self.metadata = {'requested_model': self.model, 'usage': None,
                'inference_seconds': time.perf_counter() - started, 'http_status': exc.code,
                'max_tokens': self.max_tokens, 'output_token_limit_mode': limit_mode(self.max_tokens),
                **{k: v for k, v in self.settings.items() if k != 'parameters'}}
            if self.save_to is not None:
                (self.save_to / 'inference_metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
            raise ProviderFailure(f'OpenRouter HTTP {exc.code}: {detail[:2000]}', exc.code,
                                  exc.headers.get('Retry-After') if exc.headers else None) from None
        elapsed = time.perf_counter() - started
        choices = result.get('choices') or []
        choice = choices[0] if choices else {}
        usage = result.get('usage') or {}
        completion = usage.get('completion_tokens')
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        visible = (completion - reasoning if type(completion) is int and type(reasoning) is int
                   and 0 <= reasoning <= completion else None)
        self.metadata = {'requested_model': self.model, 'returned_model': result.get('model'),
                         'provider': result.get('provider'), 'usage': result.get('usage'),
                         'request_id': result.get('id'), 'inference_seconds': elapsed,
                         'max_tokens': self.max_tokens,
                         'output_token_limit_mode': limit_mode(self.max_tokens),
                         'finish_reason': choice.get('finish_reason'),
                         'native_finish_reason': choice.get('native_finish_reason'),
                         'provider_tag': self.provider_tag,
                         'visible_completion_tokens': visible,
                         **{k: v for k, v in self.settings.items() if k != 'parameters'}}
        if self.save_to is not None:
            (self.save_to / 'api_response.json').write_text(json.dumps(result, indent=2) + '\n')
            (self.save_to / 'inference_metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        returned_error = result.get('error') or choice.get('error')
        if returned_error:
            status = returned_error.get('code') if isinstance(returned_error, dict) else None
            if isinstance(status, str) and status.isdigit():
                status = int(status)
            raise ProviderFailure(f'OpenRouter returned an error: {returned_error}', status)
        if not choices:
            raise RuntimeError('OpenRouter returned no completion choices')
        if choice.get('finish_reason') != 'stop':
            raise IncompleteCompletion(f'Incomplete completion ({choice.get("finish_reason")}); inspect api_response.json. '
                               f'Output limit mode: {limit_mode(self.max_tokens)}; '
                               f'requested max_tokens: {self.max_tokens}. '
                               'Reasoning shares the output budget. Provider/model limits still apply '
                               'when the application omits max_tokens; no automatic retry in this adapter.', result)
        content = choice.get('message', {}).get('content')
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError('No strategy text returned; inspect api_response.json and token usage.')
        return content

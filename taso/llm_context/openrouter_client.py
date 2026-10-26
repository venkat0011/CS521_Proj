"""Small standard-library OpenRouter adapter; credentials come from the environment."""
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = 'deepseek/deepseek-v4-pro'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'


class OpenRouterClient:
    def __init__(self, model=None, save_to=None, max_tokens=8192, temperature=0.2, timeout=240):
        self.model = model or os.environ.get('OPENROUTER_MODEL', DEFAULT_MODEL)
        self.save_to = Path(save_to) if save_to is not None else None
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.metadata = None
        self.request_attempts = 0

    def __call__(self, messages):
        key = os.environ.get('OPENROUTER_API_KEY', '').strip()
        if not key:
            raise ValueError('Set OPENROUTER_API_KEY in your terminal; do not put it in source files.')
        body = {'model': self.model, 'messages': messages, 'max_tokens': self.max_tokens,
                'temperature': self.temperature, 'stream': False,
                'response_format': {'type': 'json_object'},
                'provider': {'require_parameters': True}}
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
            detail = exc.read().decode('utf-8', errors='replace').replace(key, '[redacted]')
            raise RuntimeError(f'OpenRouter HTTP {exc.code}: {detail[:2000]}') from None
        elapsed = time.perf_counter() - started
        self.metadata = {'requested_model': self.model, 'returned_model': result.get('model'),
                         'provider': result.get('provider'), 'usage': result.get('usage'),
                         'request_id': result.get('id'), 'inference_seconds': elapsed,
                         'max_tokens': self.max_tokens, 'temperature': self.temperature}
        if self.save_to is not None:
            (self.save_to / 'api_response.json').write_text(json.dumps(result, indent=2) + '\n')
            (self.save_to / 'inference_metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        if result.get('error'):
            raise RuntimeError(f'OpenRouter returned an error: {result["error"]}')
        choices = result.get('choices') or []
        if not choices:
            raise RuntimeError('OpenRouter returned no completion choices')
        choice = choices[0]
        if choice.get('finish_reason') != 'stop':
            raise RuntimeError(f'Incomplete completion ({choice.get("finish_reason")}); inspect api_response.json. '
                               'If token-limited, explicitly increase --max-tokens before another request.')
        content = choice.get('message', {}).get('content')
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError('No strategy text returned; inspect api_response.json and token usage.')
        return content

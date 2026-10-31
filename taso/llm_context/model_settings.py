"""Verified OpenRouter parameter profiles, kept fixed for reproducible batches.

Source: https://openrouter.ai/api/v1/models and each model's /endpoints.
These are a dated snapshot, not a guarantee of future provider availability.
Unknown models fail locally; add a verified profile before running a new model.
"""
import math

CHECKED_ON = '2026-10-31'
PROFILES = {
    'google/gemini-4-argon': {
        'temperature': True, 'efforts': ('low', 'medium', 'high')},
    'anthropic/claude-opus-5.5': {
        'temperature': False, 'efforts': ('low', 'medium', 'high', 'xhigh', 'max')},
    'openai/gpt-5.3-codex': {
        'temperature': False, 'efforts': ('low', 'medium', 'high', 'xhigh')},
    'moonshotai/kimi-k3.1-pro': {
        'temperature': False, 'efforts': ('none', 'low', 'high', 'max')},
    'deepseek/deepseek-v4-pro': {
        'temperature': True, 'efforts': ('none', 'high', 'xhigh')},
}


def resolve_settings(model, temperature=0.2, reasoning_effort=None,
                     reasoning_effort_overrides=None):
    """Return actual request fields plus an explicit record of omissions/overrides.

    None means leave reasoning to the provider; it does not disable thinking.
    Only an explicit per-model override may change the requested effort.
    """
    if model not in PROFILES:
        raise ValueError(f'No verified parameter profile for {model!r}; update model_settings.py first')
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError('temperature must be finite and between 0 and 2')
    overrides = reasoning_effort_overrides or {}
    effort = overrides.get(model, reasoning_effort)
    profile = PROFILES[model]
    if effort is not None and (not isinstance(effort, str) or effort not in profile['efforts']):
        raise ValueError(f'{model} does not support reasoning effort {effort!r}; '
                         f"supported: {', '.join(profile['efforts'])}. Set an explicit model override.")
    parameters, omitted = {}, {}
    if profile['temperature']:
        parameters['temperature'] = temperature
    else:
        omitted['temperature'] = 'Not advertised by this model in the verified OpenRouter capability snapshot'
    if effort is not None:
        parameters['reasoning'] = {'effort': effort}
    return {
        'requested_temperature': temperature,
        'temperature': parameters.get('temperature'),
        'requested_reasoning_effort': reasoning_effort,
        'reasoning_effort': effort,
        'reasoning_effort_override_applied': model in overrides,
        'omitted_parameters': omitted,
        'parameters': parameters,
        'capabilities_checked_on': CHECKED_ON,
    }

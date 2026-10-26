"""Prepare and validate one-shot phase schedules for the existing TASO executor."""
import hashlib
import json
from pathlib import Path
from string import Template

from graph_context import graph_context

HERE = Path(__file__).resolve().parent
DEFAULT_CONTEXT = HERE / 'prepared' / 'static_context.json'
SYSTEM_PROMPT = ('You generate TASO phase schedules using only supplied rewrite rules. '
                 'Follow the requested execution contract and return only the exact JSON format.')


def build_messages(graph, context_path=DEFAULT_CONTEXT, total_budget=24, alpha=1.5,
                   max_phases=4, model_source=None):
    from taso.strategy import compile_strategy
    if type(total_budget) is not int or not 1 <= total_budget <= 2**31 - 1:
        raise ValueError('total_budget must be a positive integer <= INT_MAX')
    if type(max_phases) is not int or max_phases < 1:
        raise ValueError('max_phases must be positive')
    # Reuse the executor's validation for alpha before any model call.
    compile_strategy({'schema_version': 1, 'name': 'validation', 'phases': [
        {'name': 'all', 'rule_ids': 'all', 'expansion_budget': total_budget}]}, alpha)
    static = json.loads(Path(context_path).read_text())
    observation = graph_context(graph)
    objective = {
        'total_graph_expansions': total_budget, 'alpha': alpha,
        'max_phases': max_phases, 'feedback_during_search': 'none',
        'quality_target': 'Minimize the final graph estimated cost; final GPU measurements are evaluation only.',
        'candidate_ordering': 'Existing TASO estimated-cost ordering',
    }
    content = Template((HERE / 'prompt_template.txt').read_text()).substitute(
        model_source=model_source or 'Not supplied. Use the extracted TASO graph; do not invent PyTorch source.',
        graph=json.dumps(observation, separators=(',', ':')),
        catalog=json.dumps(static['catalog'], separators=(',', ':')),
        source_context=json.dumps(static['source_context'], separators=(',', ':')),
        objective=json.dumps(objective, indent=2))
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': content}], static['catalog']


def validate_strategy(raw, catalog, total_budget=24, max_phases=4, alpha=1.5):
    from taso.strategy import compile_strategy
    if not isinstance(raw, str):
        raise ValueError('Model caller must return a JSON string')
    # Refuse duplicate JSON keys rather than silently replacing a rule/budget.
    def unique_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    strategy = json.loads(raw, object_pairs_hook=unique_keys)
    budgets, _ = compile_strategy(strategy, alpha)
    if sum(budgets) != total_budget:
        raise ValueError(f'Phase budgets must sum to exactly {total_budget}')
    if len(budgets) > max_phases:
        raise ValueError(f'At most {max_phases} phases are allowed')
    known = {rule['rule_id'] for rule in catalog['rules']}
    for phase in strategy['phases']:
        ids = phase['rule_ids']
        if ids != 'all':
            unknown = set(ids) - known
            if unknown:
                raise ValueError(f'Unknown rule IDs: {sorted(unknown)}')
    return strategy


def request_strategy(graph, call_llm, context_path=DEFAULT_CONTEXT, save_to=None,
                     total_budget=24, alpha=1.5, max_phases=4, model_source=None,
                     dry_run=False):
    """One model call; no paid retries or automatic strategy repair.

    call_llm(messages) -> JSON string. A dry run saves the input only.
    Does not execute TASO; the experiment explicitly applies a valid strategy.
    """
    messages, catalog = build_messages(graph, context_path, total_budget, alpha,
                                       max_phases, model_source)
    out = Path(save_to) if save_to is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for stale in ('strategy.json', 'response.json'):
            (out / stale).unlink(missing_ok=True)
        encoded = json.dumps(messages, ensure_ascii=False).encode()
        (out / 'request.json').write_text(json.dumps(messages, indent=2) + '\n')
        (out / 'request.txt').write_text(messages[0]['content'] + '\n\n' + messages[1]['content'])
        (out / 'request_metadata.json').write_text(json.dumps({
            'request_sha256': hashlib.sha256(encoded).hexdigest(),
            'request_bytes': len(encoded), 'total_expansion_budget': total_budget,
            'max_phases': max_phases, 'alpha': alpha,
        }, indent=2) + '\n')
    if dry_run:
        return None
    raw = call_llm(messages)
    if out is not None:
        (out / 'response.json').write_text(json.dumps({'raw_response': raw}, indent=2) + '\n')
    strategy = validate_strategy(raw, catalog, total_budget, max_phases, alpha)
    if out is not None:
        (out / 'strategy.json').write_text(json.dumps(strategy, indent=2) + '\n')
    return strategy

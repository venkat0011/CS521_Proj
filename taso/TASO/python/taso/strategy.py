"""Small, fixed strategy language for phased rule selection.

Skipped rules are pruned on that expansion; subsequent phases do not replay
previously expanded graphs. An empty native rule vector means all rules.
"""
import math
import re

INT_MAX = 2**31 - 1
BUILTIN_COUNT = 18


def compile_strategy(strategy, alpha=1.05):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 1:
        raise ValueError('alpha must be finite and at least 1')
    if not isinstance(strategy, dict) or set(strategy) != {'schema_version', 'name', 'phases'}:
        raise ValueError('Strategy needs exactly schema_version, name and phases')
    if type(strategy['schema_version']) is not int or strategy['schema_version'] != 1:
        raise ValueError('Unsupported strategy schema')
    if not isinstance(strategy['name'], str) or not strategy['name'].strip():
        raise ValueError('Strategy name is required')
    phases = strategy['phases']
    if not isinstance(phases, list) or not phases:
        raise ValueError('At least one phase is required')
    budgets, rule_lists = [], []
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) != {'name', 'rule_ids', 'expansion_budget'}:
            raise ValueError('Each phase needs name, rule_ids and expansion_budget')
        if not isinstance(phase['name'], str) or not phase['name'].strip():
            raise ValueError('Phase name is required')
        budget = phase['expansion_budget']
        if type(budget) is not int or budget <= 0 or sum(budgets) + budget > INT_MAX:
            raise ValueError('Positive integer budgets with total <= INT_MAX are required')
        ids = phase['rule_ids']
        indexes = []
        if ids != 'all':
            if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
                raise ValueError('rule_ids must be all or a nonempty list of catalog IDs')
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate rule IDs')
            for rule in ids:
                match = re.fullmatch(r'(builtin):(\d{3})|(protobuf):(\d{4})', rule)
                if match is None:
                    raise ValueError(f'Invalid rule ID: {rule}')
                if match[1]:
                    index = int(match[2])
                    if index >= BUILTIN_COUNT:
                        raise ValueError(f'Unknown built-in rule: {rule}')
                else:
                    index = BUILTIN_COUNT + int(match[4])
                indexes.append(index)
        budgets.append(budget)
        rule_lists.append(indexes)
    # The native executor checks indices against the actual loaded rule count.
    return budgets, rule_lists

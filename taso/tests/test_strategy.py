"""Validation tests run without importing TASO's GPU extension."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('strategy', ROOT / 'TASO/python/taso/strategy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.strategy = json.loads((ROOT / 'strategies/manual_conv.json').read_text())

    def test_rule_ids_map_to_registration_order(self):
        budgets, rules = module.compile_strategy(self.strategy)
        self.assertEqual(budgets, [2, 22])
        self.assertEqual(rules, [[0, 3, 6, 9], []])
        self.strategy['phases'][0]['rule_ids'] = ['protobuf:0000', 'protobuf:0132']
        self.assertEqual(module.compile_strategy(self.strategy)[1][0], [18, 150])

    def test_rejects_ambiguous_or_invalid_budgets(self):
        for value in (True, 0, -1, 2.5, '2', 2**31):
            candidate = copy.deepcopy(self.strategy)
            candidate['phases'][0]['expansion_budget'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.compile_strategy(candidate)

    def test_rejects_invalid_rule_selections(self):
        for value in ([], ['builtin:000', 'builtin:000'], ['builtin:018'], ['missing:000'], ['protobuf:1'], [None]):
            candidate = copy.deepcopy(self.strategy)
            candidate['phases'][0]['rule_ids'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.compile_strategy(candidate)

    def test_rejects_unimplemented_strategy_actions(self):
        candidate = copy.deepcopy(self.strategy)
        candidate['phases'][0]['restart_search'] = True
        with self.assertRaises(ValueError):
            module.compile_strategy(candidate)
        candidate = copy.deepcopy(self.strategy)
        candidate['schema_version'] = True
        with self.assertRaises(ValueError):
            module.compile_strategy(candidate)

    def test_invalid_alpha(self):
        for value in (0.9, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.compile_strategy(self.strategy, alpha=value)


if __name__ == '__main__':
    unittest.main()

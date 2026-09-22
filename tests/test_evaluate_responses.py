"""Smoke-test the evaluator's built-in deterministic examples."""

import unittest

from src.evaluate_responses import self_test


class TestEvaluatorSelfTest(unittest.TestCase):
    def test_builtin_examples(self):
        self_test()


if __name__ == "__main__":
    unittest.main(verbosity=2)

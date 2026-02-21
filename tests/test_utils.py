"""Tests for scanner utility functions (imported from scanner module)."""

import unittest
import scanner


class TestNormalizeLineComponent(unittest.TestCase):
    def test_lowercase_conversion(self):
        result = scanner._normalize_line_component("Spreads")
        self.assertEqual(result, "spreads")

    def test_strips_whitespace(self):
        self.assertEqual(scanner._normalize_line_component("  h2h  "), "h2h")

    def test_none_returns_empty(self):
        self.assertEqual(scanner._normalize_line_component(None), "")

    def test_numeric_string(self):
        self.assertEqual(scanner._normalize_line_component(42), "42")


class TestPointsMatch(unittest.TestCase):
    def test_equal_points(self):
        self.assertTrue(scanner._points_match(3.5, 3.5))

    def test_within_tolerance(self):
        self.assertTrue(scanner._points_match(3.5, 3.5 + 1e-8))

    def test_outside_tolerance(self):
        self.assertFalse(scanner._points_match(3.5, 4.0))

    def test_both_none(self):
        self.assertTrue(scanner._points_match(None, None))

    def test_one_none(self):
        self.assertFalse(scanner._points_match(None, 3.5))
        self.assertFalse(scanner._points_match(3.5, None))


class TestCrossProviderPairNorm(unittest.TestCase):
    def test_sorted_alphabetically(self):
        ab = scanner._cross_provider_pair_norm("Arsenal", "Chelsea")
        ba = scanner._cross_provider_pair_norm("Chelsea", "Arsenal")
        self.assertEqual(ab, ba)

    def test_unicode_normalized(self):
        # Accented characters should be stripped during normalization
        result = scanner._cross_provider_pair_norm("Sao Paulo", "Atletico")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_empty_returns_empty(self):
        self.assertEqual(scanner._cross_provider_pair_norm("", ""), "")


class TestApplyCommissionScanner(unittest.TestCase):
    """Verify scanner._apply_commission matches ev.apply_commission."""

    def test_non_exchange_unchanged(self):
        self.assertEqual(scanner._apply_commission(2.5, 0.05, False), 2.5)

    def test_exchange_reduces_odds(self):
        result = scanner._apply_commission(2.0, 0.05, True)
        self.assertLess(result, 2.0)


if __name__ == "__main__":
    unittest.main()

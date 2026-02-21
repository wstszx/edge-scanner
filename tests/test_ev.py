"""Tests for ev.py — commission and sharp-reference helpers."""

import unittest
from ev import apply_commission, _apply_commission
from arbitrage import remove_vig, calculate_edge_percent, calculate_ev


class TestApplyCommission(unittest.TestCase):
    def test_non_exchange_unchanged(self):
        """Bookmaker odds must never be modified."""
        price = 2.5
        result = apply_commission(price, 0.05, is_exchange=False)
        self.assertEqual(result, price)

    def test_exchange_commission_reduces_price(self):
        """Exchange odds minus 5% commission should be lower than raw odds."""
        raw = 2.0
        result = apply_commission(raw, 0.05, is_exchange=True)
        self.assertLess(result, raw)
        self.assertGreater(result, 1.0)

    def test_exchange_zero_commission_unchanged(self):
        raw = 2.0
        result = apply_commission(raw, 0.0, is_exchange=True)
        self.assertAlmostEqual(result, raw, places=6)

    def test_odds_le_one_unchanged(self):
        """An odd of exactly 1.0 (no profit) should pass through."""
        result = apply_commission(1.0, 0.05, is_exchange=True)
        self.assertEqual(result, 1.0)

    def test_private_alias_identical(self):
        self.assertEqual(
            _apply_commission(2.5, 0.05, True),
            apply_commission(2.5, 0.05, True),
        )

    def test_full_commission(self):
        """100% commission collapses all profit — result should be 1.0."""
        result = apply_commission(3.0, 1.0, is_exchange=True)
        self.assertAlmostEqual(result, 1.0, places=6)


class TestRemoveVigIntegration(unittest.TestCase):
    """Integration: verify that vig removal produces valid probabilities."""

    def test_fair_odds_sum_to_one(self):
        fair_a, fair_b, _ = remove_vig(1.9, 1.9)
        prob_a = 1 / fair_a
        prob_b = 1 / fair_b
        self.assertAlmostEqual(prob_a + prob_b, 1.0, places=5)

    def test_asymmetric_market_odds(self):
        fair_a, fair_b, _ = remove_vig(1.5, 2.8)
        self.assertGreater(fair_a, 1.5)
        self.assertGreater(fair_b, 2.8)


class TestEdgeAndEv(unittest.TestCase):
    def test_edge_percent_positive(self):
        edge = calculate_edge_percent(2.2, 2.0)
        self.assertAlmostEqual(edge, 10.0, places=4)

    def test_ev_scales_with_stake(self):
        ev_100 = calculate_ev(0.55, 2.0, 100.0)
        ev_200 = calculate_ev(0.55, 2.0, 200.0)
        self.assertAlmostEqual(ev_200, ev_100 * 2, places=1)


if __name__ == "__main__":
    unittest.main()

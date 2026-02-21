"""Tests for arbitrage.py — core math helpers."""

import unittest
from arbitrage import (
    remove_vig,
    kelly_stake,
    calculate_edge_percent,
    calculate_ev,
    arbitrage_roi,
    fair_odds_from_prices,
    # private aliases must also work
    _remove_vig,
    _kelly_stake,
    _calculate_edge_percent,
    _calculate_ev,
)


class TestRemoveVig(unittest.TestCase):
    def test_symmetric_equal_odds_zero_vig(self):
        """50/50 market with equal decimal odds should have ~0% vig."""
        fair_a, fair_b, vig = remove_vig(2.0, 2.0)
        self.assertAlmostEqual(vig, 0.0, places=5)
        self.assertAlmostEqual(fair_a, 2.0, places=5)
        self.assertAlmostEqual(fair_b, 2.0, places=5)

    def test_vig_present_raises_fair_odds(self):
        """With vig, fair odds should be higher than raw odds."""
        fair_a, fair_b, vig = remove_vig(1.9, 1.9)
        self.assertGreater(vig, 0.0)
        self.assertGreater(fair_a, 1.9)
        self.assertGreater(fair_b, 1.9)

    def test_typical_pinnacle_vig(self):
        """Pinnacle typical ~2% vig on a close line."""
        # Roughly -110 / -110 American = 1.909 decimal each
        _, _, vig = remove_vig(1.909, 1.909)
        self.assertAlmostEqual(vig, 4.77, delta=0.1)

    def test_sub_one_odds_returned_unchanged(self):
        """Odds below 1.0 cannot be valid; function returns unchanged."""
        fair_a, fair_b, vig = remove_vig(0.5, 2.0)
        self.assertEqual(vig, 0.0)
        self.assertEqual(fair_a, 0.5)

    def test_private_alias_identical(self):
        self.assertEqual(_remove_vig(2.0, 2.0), remove_vig(2.0, 2.0))


class TestKellyStake(unittest.TestCase):
    def test_guard_zero_bankroll(self):
        self.assertEqual(kelly_stake(0.6, 2.0, 0.0, 0.25), (0.0, 0.0, 0.0))

    def test_guard_odds_at_one(self):
        self.assertEqual(kelly_stake(0.6, 1.0, 1000.0, 0.25), (0.0, 0.0, 0.0))

    def test_guard_negative_edge(self):
        # True prob 0.1 but odds only 2.0 → negative edge
        self.assertEqual(kelly_stake(0.1, 2.0, 1000.0, 0.25), (0.0, 0.0, 0.0))

    def test_positive_path_returns_nonzero(self):
        full_pct, fraction_pct, stake = kelly_stake(0.6, 2.2, 1000.0, 0.5)
        self.assertGreater(full_pct, 0.0)
        self.assertGreater(fraction_pct, 0.0)
        self.assertGreater(stake, 0.0)

    def test_fraction_halves_stake(self):
        _, _, stake_full = kelly_stake(0.55, 2.0, 1000.0, 1.0)
        _, _, stake_half = kelly_stake(0.55, 2.0, 1000.0, 0.5)
        self.assertAlmostEqual(stake_full / 2, stake_half, places=1)

    def test_private_alias_identical(self):
        self.assertEqual(_kelly_stake(0.6, 2.2, 1000.0, 0.5), kelly_stake(0.6, 2.2, 1000.0, 0.5))


class TestCalculateEdgePercent(unittest.TestCase):
    def test_no_edge_when_equal(self):
        self.assertAlmostEqual(calculate_edge_percent(2.0, 2.0), 0.0, places=5)

    def test_positive_edge(self):
        edge = calculate_edge_percent(2.2, 2.0)
        self.assertAlmostEqual(edge, 10.0, places=5)

    def test_zero_fair_odds(self):
        self.assertEqual(calculate_edge_percent(2.0, 0.0), 0.0)

    def test_private_alias(self):
        self.assertEqual(_calculate_edge_percent(2.2, 2.0), calculate_edge_percent(2.2, 2.0))


class TestCalculateEv(unittest.TestCase):
    def test_positive_ev(self):
        # 60% win prob, 2.0 decimal odds, $100 stake → EV = 0.6*100 - 0.4*100 = $20
        ev = calculate_ev(0.6, 2.0, 100.0)
        self.assertAlmostEqual(ev, 20.0, places=1)

    def test_negative_ev(self):
        ev = calculate_ev(0.4, 2.0, 100.0)
        self.assertLess(ev, 0.0)

    def test_private_alias(self):
        self.assertEqual(_calculate_ev(0.6, 2.0, 100.0), calculate_ev(0.6, 2.0, 100.0))


class TestArbitrageRoi(unittest.TestCase):
    def test_genuine_arbitrage(self):
        # 1/2.1 + 1/2.1 < 1 → positive ROI
        roi = arbitrage_roi([2.1, 2.1])
        self.assertGreater(roi, 0.0)

    def test_no_arbitrage(self):
        roi = arbitrage_roi([1.9, 1.9])
        self.assertLess(roi, 0.0)

    def test_three_way_arbitrage(self):
        # Synthetic 3-way with large enough odds
        roi = arbitrage_roi([3.5, 3.5, 3.5])
        self.assertGreater(roi, 0.0)


if __name__ == "__main__":
    unittest.main()

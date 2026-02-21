"""Tests for middles.py — gap detection and EV calculation."""

import unittest
from middles import (
    spread_gap_info,
    total_gap_info,
    estimate_middle_probability,
    calculate_middle_stakes,
    calculate_middle_outcomes,
    calculate_middle_ev,
    format_middle_zone,
)


class TestSpreadGapInfo(unittest.TestCase):
    def test_valid_gap(self):
        result = spread_gap_info(-2.5, 3.5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["gap_points"], 1.0, places=3)
        self.assertIn(3, result["middle_integers"])

    def test_no_gap_same_line(self):
        self.assertIsNone(spread_gap_info(-3.5, 3.5))

    def test_no_gap_favorite_bigger(self):
        self.assertIsNone(spread_gap_info(-4.0, 3.5))

    def test_requires_negative_favorite(self):
        self.assertIsNone(spread_gap_info(3.0, 4.0))

    def test_none_inputs(self):
        self.assertIsNone(spread_gap_info(None, 3.5))
        self.assertIsNone(spread_gap_info(-3.0, None))

    def test_large_gap(self):
        result = spread_gap_info(-3.0, 7.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["middle_integers"], [4, 5, 6])


class TestTotalGapInfo(unittest.TestCase):
    def test_valid_gap(self):
        result = total_gap_info(44.5, 46.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["gap_points"], 1.5, places=3)
        self.assertIn(45, result["middle_integers"])

    def test_no_gap_equal_lines(self):
        self.assertIsNone(total_gap_info(45.0, 45.0))

    def test_inverted_lines(self):
        self.assertIsNone(total_gap_info(46.0, 44.5))

    def test_half_point_gap_no_integer(self):
        # 44.5 to 45 has no integer strictly between them
        self.assertIsNone(total_gap_info(44.5, 45.0))


class TestEstimateMiddleProbability(unittest.TestCase):
    def test_empty_integers(self):
        self.assertEqual(estimate_middle_probability([], "americanfootball_nfl", "spreads"), 0.0)

    def test_nfl_key_number_3_boosted(self):
        p_nfl = estimate_middle_probability([3], "americanfootball_nfl", "spreads")
        p_generic = estimate_middle_probability([8], "americanfootball_nfl", "spreads")
        self.assertGreater(p_nfl, p_generic)

    def test_within_max_probability(self):
        # Very large window should not exceed MAX_MIDDLE_PROBABILITY
        prob = estimate_middle_probability(list(range(1, 50)), "basketball_nba", "totals")
        self.assertLessEqual(prob, 0.35)

    def test_sport_specific_rate(self):
        # MLB totals use 0.045 base rate
        p_mlb = estimate_middle_probability([1], "baseball_mlb", "totals")
        p_default = estimate_middle_probability([1], "unknown_sport", "totals")
        self.assertGreater(p_mlb, p_default)


class TestCalculateMiddleStakes(unittest.TestCase):
    def test_equal_odds_splits_evenly(self):
        a, b = calculate_middle_stakes(2.0, 2.0, 100.0)
        self.assertAlmostEqual(a, 50.0, places=1)
        self.assertAlmostEqual(b, 50.0, places=1)

    def test_total_preserved(self):
        for odds in [(2.0, 3.0), (1.8, 2.5), (3.5, 1.6)]:
            a, b = calculate_middle_stakes(odds[0], odds[1], 100.0)
            self.assertAlmostEqual(a + b, 100.0, places=1)

    def test_zero_stake_returns_zero(self):
        self.assertEqual(calculate_middle_stakes(2.0, 2.0, 0.0), (0.0, 0.0))

    def test_invalid_odds_returns_zero(self):
        self.assertEqual(calculate_middle_stakes(1.0, 2.0, 100.0), (0.0, 0.0))


class TestCalculateMiddleOutcomes(unittest.TestCase):
    def test_win_both_positive_for_middle(self):
        outcomes = calculate_middle_outcomes(50.0, 50.0, 2.1, 2.1)
        self.assertGreater(outcomes["win_both_profit"], 0.0)

    def test_side_losses_negative(self):
        outcomes = calculate_middle_outcomes(50.0, 50.0, 1.9, 1.9)
        self.assertLess(outcomes["side_a_wins_profit"], 0.0)
        self.assertLess(outcomes["side_b_wins_profit"], 0.0)


class TestCalculateMiddleEv(unittest.TestCase):
    def test_high_probability_positive_ev(self):
        ev = calculate_middle_ev(
            win_both_profit=10.0, side_a_profit=-5.0, side_b_profit=-5.0, probability=0.9
        )
        self.assertGreater(ev, 0.0)

    def test_zero_probability_only_miss_ev(self):
        ev = calculate_middle_ev(
            win_both_profit=100.0, side_a_profit=-5.0, side_b_profit=-5.0, probability=0.0
        )
        self.assertAlmostEqual(ev, -5.0, places=1)


class TestFormatMiddleZone(unittest.TestCase):
    def test_single_integer_spread(self):
        result = format_middle_zone("Kansas City Chiefs", [3], is_total=False)
        self.assertIn("3", result)

    def test_multi_integer_total(self):
        result = format_middle_zone("Total", [44, 45, 46], is_total=True)
        self.assertIn("44", result)
        self.assertIn("46", result)

    def test_total_prefix(self):
        result = format_middle_zone("x", [45], is_total=True)
        self.assertTrue(result.startswith("Total"))


if __name__ == "__main__":
    unittest.main()

"""Tests for config.markets_for_sport fallback behavior."""

import unittest

from config import markets_for_sport


class ConfigMarketsTests(unittest.TestCase):
    def test_default_unknown_sport_uses_h2h_only(self) -> None:
        self.assertEqual(markets_for_sport("tennis_atp"), ["h2h"])

    def test_unknown_americanfootball_league_uses_three_core_markets(self) -> None:
        self.assertEqual(
            markets_for_sport("americanfootball_ncaaf"),
            ["h2h", "spreads", "totals"],
        )

    def test_unknown_soccer_league_uses_soccer_defaults(self) -> None:
        self.assertEqual(markets_for_sport("soccer_uefa_champs_league"), ["spreads", "totals"])


if __name__ == "__main__":
    unittest.main()

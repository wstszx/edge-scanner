"""Tests for config.markets_for_sport fallback behavior."""

import unittest

from config import ALL_BOOKMAKER_KEYS, BOOKMAKER_LABELS, BOOKMAKER_URLS, EXCHANGE_BOOKMAKERS, EXCHANGE_CONFIG_WARNINGS, markets_for_sport


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

    def test_azuro_football_league_uses_soccer_defaults(self) -> None:
        self.assertEqual(
            markets_for_sport("azuro__football__premier-league__england"),
            ["spreads", "totals"],
        )

    def test_azuro_basketball_league_uses_core_markets(self) -> None:
        self.assertEqual(
            markets_for_sport("azuro__basketball__euroleague__international-tournaments"),
            ["h2h", "spreads", "totals"],
        )

    def test_exchange_config_warnings_are_empty(self) -> None:
        self.assertEqual(EXCHANGE_CONFIG_WARNINGS, ())

    def test_exchange_bookmakers_have_required_metadata(self) -> None:
        for key in EXCHANGE_BOOKMAKERS:
            self.assertIn(key, ALL_BOOKMAKER_KEYS)
            self.assertTrue(str(BOOKMAKER_LABELS.get(key) or "").strip())
            self.assertTrue(str(BOOKMAKER_URLS.get(key) or "").strip())


if __name__ == "__main__":
    unittest.main()

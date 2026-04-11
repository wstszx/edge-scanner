"""Tests for config.markets_for_sport fallback behavior."""

import importlib
import os
import unittest
from unittest.mock import patch

import config as config_module


class ConfigMarketsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_patch = patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        os.environ.pop("DEFAULT_SPORT_KEYS", None)
        importlib.reload(config_module)

    def tearDown(self) -> None:
        self._env_patch.stop()
        importlib.reload(config_module)

    def test_generic_tennis_and_rugby_keys_are_enabled_by_default(self) -> None:
        self.assertIn("tennis_atp", config_module.DEFAULT_SPORT_KEYS)
        self.assertIn("tennis_wta", config_module.DEFAULT_SPORT_KEYS)
        self.assertIn("rugby_union", config_module.DEFAULT_SPORT_KEYS)

    def test_legacy_tennis_and_rugby_event_keys_are_flagged_in_sport_options(self) -> None:
        by_key = {row["key"]: row for row in config_module.SPORT_OPTIONS}
        self.assertTrue(by_key["tennis_atp_indian_wells"].get("legacy"))
        self.assertTrue(by_key["tennis_wta_indian_wells"].get("legacy"))
        self.assertTrue(by_key["rugby_union_six_nations"].get("legacy"))
        self.assertFalse(by_key["tennis_atp"].get("legacy"))
        self.assertFalse(by_key["tennis_wta"].get("legacy"))
        self.assertFalse(by_key["rugby_union"].get("legacy"))

    def test_default_unknown_sport_uses_h2h_only(self) -> None:
        self.assertEqual(config_module.markets_for_sport("tennis_atp"), ["h2h"])

    def test_unknown_americanfootball_league_uses_three_core_markets(self) -> None:
        self.assertEqual(
            config_module.markets_for_sport("americanfootball_ncaaf"),
            ["h2h", "spreads", "totals"],
        )

    def test_unknown_soccer_league_uses_soccer_defaults(self) -> None:
        self.assertEqual(
            config_module.markets_for_sport("soccer_uefa_champs_league"),
            ["h2h", "h2h_3_way", "spreads", "totals"],
        )

    def test_azuro_football_league_uses_soccer_defaults(self) -> None:
        self.assertEqual(
            config_module.markets_for_sport("azuro__football__premier-league__england"),
            ["h2h", "h2h_3_way", "spreads", "totals"],
        )

    def test_azuro_basketball_league_uses_core_markets(self) -> None:
        self.assertEqual(
            config_module.markets_for_sport("azuro__basketball__euroleague__international-tournaments"),
            ["h2h", "spreads", "totals"],
        )

    def test_exchange_config_warnings_are_empty(self) -> None:
        self.assertEqual(config_module.EXCHANGE_CONFIG_WARNINGS, ())

    def test_exchange_bookmakers_have_required_metadata(self) -> None:
        for key in config_module.EXCHANGE_BOOKMAKERS:
            self.assertIn(key, config_module.ALL_BOOKMAKER_KEYS)
            self.assertTrue(str(config_module.BOOKMAKER_LABELS.get(key) or "").strip())
            self.assertTrue(str(config_module.BOOKMAKER_URLS.get(key) or "").strip())


if __name__ == "__main__":
    unittest.main()

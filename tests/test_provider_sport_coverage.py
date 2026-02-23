"""Regression tests for provider-side sport key coverage."""

import unittest

from providers import betdex, bookmaker_xyz, overtimemarkets_xyz, polymarket, purebet, sx_bet


class ProviderSportCoverageTests(unittest.TestCase):
    def test_ncaaf_present_in_provider_maps(self) -> None:
        self.assertIn("americanfootball_ncaaf", sx_bet.SX_SPORT_ID_MAP)
        self.assertIn("americanfootball_ncaaf", bookmaker_xyz.SPORT_SLUG_HINTS)
        self.assertIn("americanfootball_ncaaf", betdex.SPORT_SUBCATEGORY_DEFAULTS)
        self.assertIn("americanfootball_ncaaf", overtimemarkets_xyz.SPORT_LABEL_HINTS)
        self.assertIn("americanfootball_ncaaf", polymarket.SPORT_ALIASES)

    def test_purebet_active_league_inference_supports_ncaaf(self) -> None:
        league = {
            "sportName": "American Football",
            "sport": "8",
            "name": "NCAAF",
            "abbr": "NCAAF",
            "country": "US",
        }
        self.assertEqual(
            purebet._infer_sport_key_from_active_league(league),
            "americanfootball_ncaaf",
        )


if __name__ == "__main__":
    unittest.main()

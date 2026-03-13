"""Regression tests for provider-side sport key coverage."""

import unittest

from providers import betdex, bookmaker_xyz, polymarket, purebet, sx_bet


class ProviderSportCoverageTests(unittest.TestCase):
    def test_ncaaf_present_in_provider_maps(self) -> None:
        self.assertIn("americanfootball_ncaaf", sx_bet.SX_SPORT_ID_MAP)
        self.assertIn("americanfootball_ncaaf", bookmaker_xyz.SPORT_SLUG_HINTS)
        self.assertIn("americanfootball_ncaaf", betdex.SPORT_SUBCATEGORY_DEFAULTS)
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

    def test_bookmaker_xyz_soccer_epl_requires_english_context(self) -> None:
        singapore_game = {
            "title": "Young Lions - Hougang United FC",
            "sport": {"slug": "football", "name": "Football"},
            "league": {"slug": "premier-league", "name": "Premier League"},
            "country": {"slug": "singapore", "name": "Singapore"},
        }
        england_game = {
            "title": "Arsenal - Chelsea",
            "sport": {"slug": "football", "name": "Football"},
            "league": {"slug": "premier-league", "name": "Premier League"},
            "country": {"slug": "england", "name": "England"},
        }

        self.assertFalse(bookmaker_xyz._sport_matches_requested("soccer_epl", singapore_game))
        self.assertTrue(bookmaker_xyz._sport_matches_requested("soccer_epl", england_game))

    def test_bookmaker_xyz_dynamic_azuro_sport_key_matches_exact_filter(self) -> None:
        euroleague_game = {
            "title": "Real Madrid - Fenerbahce",
            "sport": {"slug": "basketball", "name": "Basketball"},
            "league": {"slug": "euroleague", "name": "EuroLeague"},
            "country": {
                "slug": "international-tournaments",
                "name": "International Tournaments",
            },
        }
        wrong_country_game = {
            **euroleague_game,
            "country": {"slug": "spain", "name": "Spain"},
        }

        self.assertTrue(
            bookmaker_xyz._sport_matches_requested(
                "azuro__basketball__euroleague__international-tournaments",
                euroleague_game,
            )
        )
        self.assertFalse(
            bookmaker_xyz._sport_matches_requested(
                "azuro__basketball__euroleague__international-tournaments",
                wrong_country_game,
            )
        )


if __name__ == "__main__":
    unittest.main()

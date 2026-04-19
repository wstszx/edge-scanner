"""Regression tests for provider-side sport key coverage."""

import unittest

from providers import PROVIDER_CAPABILITIES, artline, betdex, bookmaker_xyz, polymarket, sx_bet


class ProviderSportCoverageTests(unittest.TestCase):
    def test_ncaaf_presence_stays_tied_to_provider_source_maps(self) -> None:
        self.assertIn("americanfootball_ncaaf", sx_bet.SX_SPORT_ID_MAP)
        self.assertIn("americanfootball_ncaaf", bookmaker_xyz.SPORT_SLUG_HINTS)
        self.assertIn("americanfootball_ncaaf", betdex.SPORT_SUBCATEGORY_DEFAULTS)
        self.assertIn("americanfootball_ncaaf", polymarket.SPORT_ALIASES)

        self.assertIn("americanfootball_ncaaf", PROVIDER_CAPABILITIES["sx_bet"].supported_sport_keys)
        self.assertIn(
            "americanfootball_ncaaf",
            PROVIDER_CAPABILITIES["bookmaker_xyz"].supported_sport_keys,
        )
        self.assertIn("americanfootball_ncaaf", PROVIDER_CAPABILITIES["betdex"].supported_sport_keys)
        self.assertIn(
            "americanfootball_ncaaf",
            PROVIDER_CAPABILITIES["polymarket"].supported_sport_keys,
        )

    def test_generic_tennis_and_rugby_keys_stay_tied_to_provider_source_maps(self) -> None:
        self.assertIn("tennis_atp", bookmaker_xyz.SPORT_SLUG_HINTS)
        self.assertIn("tennis_wta", bookmaker_xyz.SPORT_SLUG_HINTS)
        self.assertIn("rugby_union", bookmaker_xyz.SPORT_SLUG_HINTS)
        self.assertIn("tennis_atp", betdex.SPORT_SUBCATEGORY_DEFAULTS)
        self.assertIn("tennis_wta", betdex.SPORT_SUBCATEGORY_DEFAULTS)
        self.assertIn("rugby_union", betdex.SPORT_SUBCATEGORY_DEFAULTS)
        self.assertIn("tennis_atp", betdex.SPORT_LEAGUE_HINTS)
        self.assertIn("tennis_wta", betdex.SPORT_LEAGUE_HINTS)
        self.assertIn("rugby_union", betdex.SPORT_LEAGUE_HINTS)

        self.assertIn("tennis_atp", PROVIDER_CAPABILITIES["bookmaker_xyz"].supported_sport_keys)
        self.assertIn("tennis_wta", PROVIDER_CAPABILITIES["bookmaker_xyz"].supported_sport_keys)
        self.assertIn("rugby_union", PROVIDER_CAPABILITIES["bookmaker_xyz"].supported_sport_keys)
        self.assertIn("tennis_atp", PROVIDER_CAPABILITIES["betdex"].supported_sport_keys)
        self.assertIn("tennis_wta", PROVIDER_CAPABILITIES["betdex"].supported_sport_keys)
        self.assertIn("rugby_union", PROVIDER_CAPABILITIES["betdex"].supported_sport_keys)

    def test_betdex_capability_registry_tracks_verified_expansion_source_mappings(self) -> None:
        for sport_key in (
            "basketball_euroleague",
            "basketball_germany_bbl",
            "soccer_england_championship",
            "soccer_england_league_one",
            "soccer_england_league_two",
            "soccer_brazil_serie_a",
            "soccer_netherlands_eredivisie",
            "soccer_argentina_liga_profesional",
            "mma_ufc",
        ):
            self.assertIn(sport_key, betdex.SPORT_SUBCATEGORY_DEFAULTS)
            self.assertIn(sport_key, betdex.SPORT_LEAGUE_HINTS)
            self.assertIn(sport_key, PROVIDER_CAPABILITIES["betdex"].supported_sport_keys)

    def test_artline_capability_registry_tracks_verified_source_mappings(self) -> None:
        for sport_key in (
            "basketball_nba",
            "icehockey_nhl",
            "soccer_italy_serie_a",
            "soccer_usa_mls",
            "soccer_england_championship",
            "soccer_netherlands_eredivisie",
            "soccer_mexico_liga_mx",
            "basketball_france_pro_a",
            "icehockey_ahl",
            "soccer_portugal_primeira_liga",
            "soccer_argentina_liga_profesional",
            "soccer_england_league_one",
            "soccer_england_league_two",
            "basketball_germany_bbl",
            "baseball_mlb",
        ):
            self.assertIn(sport_key, artline.ARTLINE_SPORT_FILTERS)
            self.assertIn(sport_key, PROVIDER_CAPABILITIES["artline"].supported_sport_keys)

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

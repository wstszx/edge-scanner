from __future__ import annotations

import unittest

from providers import betdex, bookmaker_xyz, overtimemarkets_xyz, sx_bet


class TestBetdexSegmentation(unittest.TestCase):
    def test_half_time_totals_alias_prefers_segmented_key(self) -> None:
        aliases = betdex._market_aliases_for_type("FOOTBALL_OVER_UNDER_HALF_TIME_TOTAL_GOALS")
        self.assertIn("totals_h1", aliases)
        self.assertNotIn("totals", aliases)

    def test_full_time_totals_alias_keeps_base_key(self) -> None:
        aliases = betdex._market_aliases_for_type("FOOTBALL_OVER_UNDER_TOTAL_GOALS")
        self.assertIn("totals", aliases)

    def test_scoped_market_key_for_half_time(self) -> None:
        key = betdex._scoped_market_key("totals", "FOOTBALL_OVER_UNDER_HALF_TIME_TOTAL_GOALS")
        self.assertEqual(key, "totals_h1")


class TestSxBetSegmentation(unittest.TestCase):
    def test_half_time_totals_alias_is_segmented(self) -> None:
        aliases = sx_bet._market_type_aliases(
            {"marketType": "OVER_UNDER", "marketName": "First Half Total Goals"}
        )
        self.assertIn("totals_h1", aliases)
        self.assertNotIn("totals", aliases)

    def test_half_time_totals_not_mapped_to_base_totals(self) -> None:
        market = {
            "marketType": "OVER_UNDER",
            "marketName": "First Half Total Goals",
            "bestOddsOutcomeOne": 2.02,
            "bestOddsOutcomeTwo": 1.88,
            "outcomeOneName": "Over 1.5",
            "outcomeTwoName": "Under 1.5",
            "marketValue": "1.5",
            "marketHash": "abc",
        }
        skipped = sx_bet._normalize_fixture_market(
            market,
            requested_markets={"totals"},
            home_team="Team A",
            away_team="Team B",
        )
        self.assertIsNone(skipped)

        segmented = sx_bet._normalize_fixture_market(
            market,
            requested_markets={"totals_h1"},
            home_team="Team A",
            away_team="Team B",
        )
        self.assertIsNotNone(segmented)
        self.assertEqual(segmented["market_key"], "totals_h1")


class TestOvertimeSegmentation(unittest.TestCase):
    def test_half_time_totals_not_mapped_to_base_totals(self) -> None:
        row = {
            "type": "OVER_UNDER",
            "marketName": "First Half Total Goals",
            "odds": [2.1, 1.8],
            "line": 1.5,
        }
        skipped = overtimemarkets_xyz._build_market(
            row,
            requested_markets={"totals"},
            home_team="Team A",
            away_team="Team B",
        )
        self.assertIsNone(skipped)

        segmented = overtimemarkets_xyz._build_market(
            row,
            requested_markets={"totals_h1"},
            home_team="Team A",
            away_team="Team B",
        )
        self.assertIsNotNone(segmented)
        self.assertEqual(segmented["key"], "totals_h1")


class TestBookmakerFallbackSegmentation(unittest.TestCase):
    def test_segmented_fallback_h2h_is_rejected(self) -> None:
        condition = {
            "name": "1st Half Winner",
            "outcomes": [
                {"currentOdds": 1.9, "sortOrder": 1},
                {"currentOdds": 1.9, "sortOrder": 2},
            ],
        }
        market = bookmaker_xyz._fallback_h2h_market(condition, "Home", "Away")
        self.assertIsNone(market)

    def test_full_time_fallback_h2h_allowed(self) -> None:
        condition = {
            "name": "Match Winner",
            "outcomes": [
                {"currentOdds": 1.9, "sortOrder": 1},
                {"currentOdds": 1.9, "sortOrder": 2},
            ],
        }
        market = bookmaker_xyz._fallback_h2h_market(condition, "Home", "Away")
        self.assertIsNotNone(market)
        self.assertEqual(market["key"], "h2h")


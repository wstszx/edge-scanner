from __future__ import annotations

import datetime as dt
import json
import unittest
from unittest.mock import patch

from providers import betdex, bookmaker_xyz, polymarket, sx_bet


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
    def test_summary_scaled_implied_odds_are_converted(self) -> None:
        odds = sx_bet._moneyline_decimal_from_summary("53125000000000000000")
        self.assertIsNotNone(odds)
        self.assertAlmostEqual(odds, 1.8823529411764706, places=9)

    def test_summary_legacy_decimal_odds_are_preserved(self) -> None:
        odds = sx_bet._moneyline_decimal_from_summary(2.05)
        self.assertEqual(odds, 2.05)

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

    def test_orders_stake_map_aggregates_best_remaining_size(self) -> None:
        sx_bet.ORDERS_CACHE["expires_at"] = 0.0
        sx_bet.ORDERS_CACHE["entries"] = {}
        payload = {
            "status": "success",
            "data": [
                {
                    "marketHash": "m1",
                    "isMakerBettingOutcomeOne": True,
                    "percentageOdds": "50000000000000000000",
                    "totalBetSize": "2000000",
                    "fillAmount": "500000",
                },
                {
                    "marketHash": "m1",
                    "isMakerBettingOutcomeOne": True,
                    "percentageOdds": "50000000000000000000",
                    "totalBetSize": "1000000",
                    "fillAmount": "0",
                },
                {
                    "marketHash": "m1",
                    "isMakerBettingOutcomeOne": False,
                    "percentageOdds": "25000000000000000000",
                    "totalBetSize": "3000000",
                    "fillAmount": "1000000",
                },
            ],
        }
        with patch("providers.sx_bet._request_json", return_value=(payload, 0)):
            stake_map, retries_used, meta = sx_bet._load_best_stake_map(
                market_hashes=["m1", "m2"],
                base_token="usdc",
                retries=0,
                backoff_seconds=0.0,
                base_token_decimals=6,
            )
        self.assertEqual(retries_used, 0)
        self.assertEqual(meta.get("orders_rows"), 3)
        self.assertEqual(stake_map.get("m1"), (2.5, 2.0))
        self.assertEqual(stake_map.get("m2"), (None, None))

    def test_numeric_type_h2h_alias_infers_market_key(self) -> None:
        market = {
            "type": 226,
            "teamOneName": "Home Team",
            "teamTwoName": "Away Team",
            "outcomeOneName": "Home Team",
            "outcomeTwoName": "Away Team",
            "bestOddsOutcomeOne": 2.02,
            "bestOddsOutcomeTwo": 1.95,
            "marketHash": "mh-1",
        }
        normalized = sx_bet._normalize_fixture_market(
            market=market,
            requested_markets={"h2h"},
            home_team="Home Team",
            away_team="Away Team",
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized.get("market_key"), "h2h")

    def test_markets_active_mainline_filter_preserves_scope_without_mainline(self) -> None:
        rows = [
            {
                "sportId": 1,
                "sportXeventId": "evt-1",
                "teamOneName": "Home Team",
                "teamTwoName": "Away Team",
                "gameTime": 1770000000,
                "type": 342,
                "line": -3.5,
                "mainLine": False,
                "outcomeOneName": "Home Team -3.5",
                "outcomeTwoName": "Away Team +3.5",
                "marketHash": "spread-1",
            }
        ]
        fixtures, meta = sx_bet._build_fixtures_from_markets_active(
            rows=rows,
            sport_id=1,
            only_main_line=True,
        )
        self.assertEqual(len(fixtures), 1)
        fixture_markets = fixtures[0].get("markets") or []
        self.assertEqual(len(fixture_markets), 1)
        self.assertEqual(meta.get("markets_rows_main_line_filtered"), 0)

    def test_auto_fixture_loader_falls_back_to_summary_when_markets_active_fails(self) -> None:
        with (
            patch("providers.sx_bet._load_upcoming_fixtures_markets_active", side_effect=sx_bet.ProviderError("boom")),
            patch(
                "providers.sx_bet._load_upcoming_fixtures_summary",
                return_value=([{"eventId": "evt-1", "teamOne": "A", "teamTwo": "B", "gameTime": 1770000000}], {"fixture_source": "summary"}),
            ),
            patch("providers.sx_bet._fixture_source_mode", return_value="auto"),
        ):
            fixtures, meta = sx_bet._load_upcoming_fixtures(
                sport_id=1,
                base_token="usdc",
                retries=0,
                backoff_seconds=0.0,
            )
        self.assertEqual(len(fixtures), 1)
        self.assertTrue(meta.get("fallback_used"))
        self.assertIn("boom", str(meta.get("fallback_reason")))

    def test_active_league_ids_are_cached_between_calls(self) -> None:
        sx_bet.LEAGUES_CACHE["expires_at"] = 0.0
        sx_bet.LEAGUES_CACHE["entries"] = {}
        payload = {
            "status": "success",
            "data": [
                {"leagueId": 1, "sportId": 1},
                {"leagueId": 2, "sportId": 1},
                {"leagueId": 999, "sportId": 2},
            ],
        }
        with patch("providers.sx_bet._request_json", return_value=(payload, 0)) as mocked_request:
            league_ids_first, retries_first = sx_bet._load_active_league_ids(
                sport_id=1,
                retries=0,
                backoff_seconds=0.0,
            )
            league_ids_second, retries_second = sx_bet._load_active_league_ids(
                sport_id=1,
                retries=0,
                backoff_seconds=0.0,
            )
        self.assertEqual(league_ids_first, [1, 2])
        self.assertEqual(league_ids_second, [1, 2])
        self.assertEqual(retries_first, 0)
        self.assertEqual(retries_second, 0)
        self.assertEqual(mocked_request.call_count, 1)


class TestPolymarketDepth(unittest.TestCase):
    def test_book_depth_uses_asks_notional(self) -> None:
        depth = polymarket._book_ask_depth_notional(
            {
                "asks": [
                    {"price": "0.62", "size": "120"},
                    {"price": "0.63", "size": "30"},
                ]
            }
        )
        self.assertEqual(depth, 93.3)

    def test_direct_h2h_market_carries_clob_stake(self) -> None:
        event = {
            "markets": [
                {
                    "question": "Home vs Away",
                    "outcomes": json.dumps(["Home", "Away"]),
                    "outcomePrices": json.dumps([0.4, 0.6]),
                    "clobTokenIds": json.dumps(["tok_home", "tok_away"]),
                    "volumeNum": 100,
                }
            ]
        }
        markets = polymarket._pick_match_markets(
            event=event,
            home_team="Home",
            away_team="Away",
            requested_markets={"h2h"},
            now_utc=dt.datetime.now(dt.timezone.utc),
            clob_depth_map={"tok_home": 120.5, "tok_away": 95.25},
        )
        self.assertEqual(len(markets), 1)
        outcomes = markets[0].get("outcomes") or []
        self.assertEqual(outcomes[0].get("stake"), 120.5)
        self.assertEqual(outcomes[1].get("stake"), 95.25)


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

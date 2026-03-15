from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import scanner
from providers import betdex, bookmaker_xyz, polymarket, sx_bet


SPORT_KEY = "basketball_nba"
COMMENCE_TIME = "2026-03-20T00:00:00Z"
HOME_TEAM = "Home Team"
AWAY_TEAM = "Away Team"


def _reset_provider_caches() -> None:
    betdex.ACCESS_TOKEN_CACHE["token"] = ""
    betdex.ACCESS_TOKEN_CACHE["expires_at"] = 0.0

    polymarket.disable_scan_cache()

    sx_bet.UPCOMING_CACHE["expires_at"] = 0.0
    sx_bet.UPCOMING_CACHE["entries"] = {}
    sx_bet.ORDERS_CACHE["expires_at"] = 0.0
    sx_bet.ORDERS_CACHE["entries"] = {}
    sx_bet.LEAGUES_CACHE["expires_at"] = 0.0
    sx_bet.LEAGUES_CACHE["entries"] = {}

    bookmaker_xyz.disable_scan_cache()
    bookmaker_xyz.CONDITIONS_CACHE["expires_at"] = 0.0
    bookmaker_xyz.CONDITIONS_CACHE["conditions"] = []
    bookmaker_xyz.CONDITIONS_CACHE["meta"] = {}
    bookmaker_xyz.CONDITIONS_CACHE["key"] = ""
    bookmaker_xyz.DICTIONARY_CACHE["expires_at"] = 0.0
    bookmaker_xyz.DICTIONARY_CACHE["data"] = None
    bookmaker_xyz.DICTIONARY_CACHE["source"] = ""


def _standardized_outcomes(events: list[dict], provider_key: str) -> dict[str, dict]:
    assert len(events) == 1
    bookmakers = events[0].get("bookmakers") or []
    assert len(bookmakers) == 1
    book = bookmakers[0]
    assert book.get("key") == provider_key
    markets = book.get("markets") or []
    h2h_market = next(item for item in markets if item.get("key") == "h2h")
    return {
        str(outcome.get("name")): outcome
        for outcome in (h2h_market.get("outcomes") or [])
        if isinstance(outcome, dict)
    }


def _make_counterparty_fetcher(
    provider_key: str,
    provider_title: str,
    *,
    home_price: float = 1.72,
    away_price: float = 2.22,
):
    async def _fetcher(
        sport_key: str,
        markets: list[str],
        regions: list[str],
        bookmakers=None,
    ):
        lowered = {
            str(book).strip().lower()
            for book in (bookmakers or [])
            if str(book).strip()
        }
        if lowered and provider_key not in lowered and provider_title.lower() not in lowered:
            return []
        _fetcher.last_stats = {
            "provider": provider_key,
            "sport_key": sport_key,
            "markets": list(markets),
            "regions": list(regions),
            "events_returned_count": 1,
        }
        return [
            {
                "id": f"{provider_key}-event",
                "sport_key": sport_key,
                "home_team": HOME_TEAM,
                "away_team": AWAY_TEAM,
                "commence_time": COMMENCE_TIME,
                "bookmakers": [
                    {
                        "key": provider_key,
                        "title": provider_title,
                        "event_id": f"{provider_key}-event",
                        "event_url": f"https://example.com/{provider_key}",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": HOME_TEAM, "price": home_price},
                                    {"name": AWAY_TEAM, "price": away_price},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

    _fetcher.last_stats = {}
    return _fetcher


async def _fake_shared_client(*args, **kwargs):
    return object()


async def _fake_empty_mapping(*args, **kwargs):
    return {}


class ProviderArbitragePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_provider_caches()

    def tearDown(self) -> None:
        _reset_provider_caches()
        scanner._set_current_request_logger(None)
        with scanner._REQUEST_TRACE_LOCK:
            scanner._REQUEST_TRACE_ACTIVE.clear()

    def _run_provider_only_scan(
        self,
        provider_key: str,
        counterparty_key: str,
        counterparty_fetcher,
    ) -> dict:
        with (
            patch.dict(
                scanner.PROVIDER_FETCHERS,
                {counterparty_key: counterparty_fetcher},
                clear=False,
            ),
            patch.object(scanner, "_persist_provider_snapshots", return_value={}),
            patch.object(scanner, "_persist_cross_provider_match_report", return_value=""),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
            patch.object(scanner, "_provider_fetch_max_workers", return_value=1),
        ):
            return scanner.run_scan(
                api_key="",
                sports=[SPORT_KEY],
                regions=["us"],
                bookmakers=[provider_key, counterparty_key],
                include_providers=[provider_key, counterparty_key],
                stake_amount=100.0,
            )

    def _assert_arbitrage_result(
        self,
        result: dict,
        provider_key: str,
        provider_title: str,
        counterparty_title: str,
        *,
        stats_assertion=None,
    ) -> None:
        self.assertTrue(result.get("success"))
        arbitrage = result.get("arbitrage") or {}
        self.assertGreater(arbitrage.get("opportunities_count", 0), 0)

        first_opportunity = arbitrage["opportunities"][0]
        self.assertEqual(first_opportunity.get("market"), "h2h")
        self.assertGreater(first_opportunity.get("roi_percent", 0), 0)

        books_used = {
            str(item.get("bookmaker"))
            for item in (first_opportunity.get("best_odds") or [])
            if isinstance(item, dict)
        }
        self.assertIn(provider_title, books_used)
        self.assertIn(counterparty_title, books_used)

        provider_summary = (result.get("custom_providers") or {}).get(provider_key) or {}
        self.assertEqual(provider_summary.get("events_merged"), 1)
        provider_sports = provider_summary.get("sports") or []
        self.assertEqual(len(provider_sports), 1)
        self.assertEqual(provider_sports[0].get("sport_key"), SPORT_KEY)
        self.assertEqual(provider_sports[0].get("events_returned"), 1)

        stats = provider_sports[0].get("stats") or {}
        self.assertEqual(stats.get("events_returned_count"), 1)
        if callable(stats_assertion):
            stats_assertion(stats)

    def test_polymarket_pipeline_produces_standardized_h2h_and_arbitrage(self) -> None:
        polymarket_event = {
            "id": "poly-1",
            "slug": "home-vs-away",
            "title": f"{HOME_TEAM} vs {AWAY_TEAM}",
            "active": True,
            "closed": False,
            "archived": False,
            "startTime": COMMENCE_TIME,
            "tags": [
                {"id": "1", "slug": "sports"},
                {"id": "999", "slug": "nba"},
            ],
            "markets": [
                {
                    "question": f"{HOME_TEAM} vs {AWAY_TEAM}",
                    "outcomes": f'["{HOME_TEAM}", "{AWAY_TEAM}"]',
                    "outcomePrices": "[0.444444, 0.588235]",
                    "clobTokenIds": '["tok-home", "tok-away"]',
                    "volumeNum": 100,
                    "active": True,
                    "closed": False,
                    "archived": False,
                }
            ],
        }
        counterparty_fetcher = _make_counterparty_fetcher("sx_bet", sx_bet.PROVIDER_TITLE)

        async def _fake_load_active_game_events_async(*args, **kwargs):
            return [polymarket_event], {"cache": "miss", "pages_fetched": 1, "retries_used": 0}

        with (
            patch.object(polymarket, "_websocket_realtime_enabled", return_value=False),
            patch.object(polymarket, "get_shared_client", new=_fake_shared_client),
            patch.object(polymarket, "_load_sport_tag_mapping_async", new=_fake_empty_mapping),
            patch.object(
                polymarket,
                "_load_active_game_events_async",
                new=_fake_load_active_game_events_async,
            ),
        ):
            events = asyncio.run(
                polymarket.fetch_events_async(
                    SPORT_KEY,
                    ["h2h"],
                    ["us"],
                    bookmakers=[polymarket.PROVIDER_KEY],
                )
            )

            outcomes = _standardized_outcomes(events, polymarket.PROVIDER_KEY)
            self.assertAlmostEqual(outcomes[HOME_TEAM]["price"], 2.250002, places=6)
            self.assertAlmostEqual(outcomes[AWAY_TEAM]["price"], 1.700001, places=6)
            self.assertNotIn("stake", outcomes[HOME_TEAM])
            direct_stats = dict(polymarket.fetch_events_async.last_stats)
            self.assertFalse(direct_stats.get("clob_http_fallback_enabled"))
            self.assertEqual(direct_stats.get("clob_http_fallback_skipped"), 2)

            result = self._run_provider_only_scan(
                provider_key=polymarket.PROVIDER_KEY,
                counterparty_key="sx_bet",
                counterparty_fetcher=counterparty_fetcher,
            )

        self._assert_arbitrage_result(
            result,
            provider_key=polymarket.PROVIDER_KEY,
            provider_title=polymarket.PROVIDER_TITLE,
            counterparty_title=sx_bet.PROVIDER_TITLE,
            stats_assertion=lambda stats: (
                self.assertFalse(stats.get("clob_http_fallback_enabled")),
                self.assertEqual(stats.get("events_with_market_count"), 1),
            ),
        )

    def test_betdex_pipeline_produces_standardized_h2h_and_arbitrage(self) -> None:
        async def _fake_betdex_request(
            client,
            url,
            params=None,
            access_token=None,
            retries=0,
            backoff_seconds=0.0,
            timeout=20,
        ):
            if url.endswith("/api/session"):
                return {"sessions": [{"accessToken": "token-1"}]}, 0
            if url.endswith("/events"):
                return (
                    {
                        "events": [
                            {
                                "id": "bet-event-1",
                                "name": f"{HOME_TEAM} vs {AWAY_TEAM}",
                                "active": True,
                                "expectedStartTime": COMMENCE_TIME,
                                "eventGroup": {"_ids": ["group-1"]},
                                "participants": {"_ids": ["p-home", "p-away"]},
                            }
                        ],
                        "eventGroups": [
                            {
                                "id": "group-1",
                                "name": "NBA",
                                "subcategory": {"_ids": ["BBALL"]},
                            }
                        ],
                        "participants": [
                            {"id": "p-home", "name": HOME_TEAM},
                            {"id": "p-away", "name": AWAY_TEAM},
                        ],
                        "_meta": {"_page": {"_totalPages": 1}},
                    },
                    0,
                )
            if url.endswith("/markets"):
                return (
                    {
                        "markets": [
                            {
                                "id": "market-1",
                                "published": True,
                                "suspended": False,
                                "event": {"_ids": ["bet-event-1"]},
                                "marketType": {"_ids": ["MONEYLINE"]},
                                "name": "Match Winner",
                                "marketOutcomes": {"_ids": ["out-home", "out-away"]},
                            }
                        ],
                        "marketOutcomes": [
                            {"id": "out-home", "title": HOME_TEAM},
                            {"id": "out-away", "title": AWAY_TEAM},
                        ],
                        "_meta": {"_page": {"_totalPages": 1}},
                    },
                    0,
                )
            if url.endswith("/market-prices"):
                return (
                    {
                        "prices": [
                            {
                                "marketId": "market-1",
                                "prices": [
                                    {
                                        "outcomeId": "out-home",
                                        "side": "against",
                                        "price": 2.25,
                                        "amount": 140,
                                    },
                                    {
                                        "outcomeId": "out-away",
                                        "side": "against",
                                        "price": 1.70,
                                        "amount": 125,
                                    },
                                ],
                            }
                        ]
                    },
                    0,
                )
            raise AssertionError(url)

        counterparty_fetcher = _make_counterparty_fetcher("sx_bet", sx_bet.PROVIDER_TITLE)

        with (
            patch.object(betdex, "get_shared_client", new=_fake_shared_client),
            patch.object(betdex, "_request_json_async", side_effect=_fake_betdex_request),
        ):
            events = asyncio.run(
                betdex.fetch_events_async(
                    SPORT_KEY,
                    ["h2h"],
                    ["us"],
                    bookmakers=[betdex.PROVIDER_KEY],
                )
            )

            outcomes = _standardized_outcomes(events, betdex.PROVIDER_KEY)
            self.assertEqual(outcomes[HOME_TEAM]["price"], 2.25)
            self.assertEqual(outcomes[AWAY_TEAM]["price"], 1.7)
            self.assertEqual(outcomes[HOME_TEAM]["stake"], 140.0)
            self.assertEqual(outcomes[AWAY_TEAM]["stake"], 125.0)
            direct_stats = dict(betdex.fetch_events_async.last_stats)
            self.assertEqual(direct_stats.get("session_cache"), "miss")
            self.assertEqual(direct_stats.get("prices_payload_count"), 1)

            result = self._run_provider_only_scan(
                provider_key=betdex.PROVIDER_KEY,
                counterparty_key="sx_bet",
                counterparty_fetcher=counterparty_fetcher,
            )

        self._assert_arbitrage_result(
            result,
            provider_key=betdex.PROVIDER_KEY,
            provider_title=betdex.PROVIDER_TITLE,
            counterparty_title=sx_bet.PROVIDER_TITLE,
            stats_assertion=lambda stats: (
                self.assertEqual(stats.get("session_cache"), "hit"),
                self.assertEqual(stats.get("price_rows_against"), 2),
            ),
        )

    def test_sx_bet_pipeline_produces_standardized_h2h_and_arbitrage(self) -> None:
        fixtures = [
            {
                "eventId": "sx-event-1",
                "id": "sx-event-1",
                "teamOne": HOME_TEAM,
                "teamTwo": AWAY_TEAM,
                "leagueLabel": "NBA",
                "gameTime": COMMENCE_TIME,
                "markets": [
                    {
                        "type": 226,
                        "teamOneName": HOME_TEAM,
                        "teamTwoName": AWAY_TEAM,
                        "outcomeOneName": HOME_TEAM,
                        "outcomeTwoName": AWAY_TEAM,
                        "bestOddsOutcomeOne": 2.25,
                        "bestOddsOutcomeTwo": 1.70,
                        "marketHash": "sx-h2h",
                    }
                ],
            }
        ]

        async def _fake_load_upcoming_fixtures_async(*args, **kwargs):
            return (
                fixtures,
                {
                    "cache": "miss",
                    "pages_fetched": 1,
                    "retries_used": 0,
                    "fixture_source": "summary",
                },
            )

        async def _fake_load_best_stake_map_async(*args, **kwargs):
            return (
                {"sx-h2h": (150.0, 120.0)},
                0,
                {
                    "orders_rows": 1,
                    "orders_missing_market_hash": 0,
                },
            )

        async def _fake_load_best_odds_map_async(*args, **kwargs):
            return (
                {
                    "sx-h2h": {
                        "odds_one": 2.35,
                        "odds_two": 1.66,
                        "updated_at_one": "2026-03-14T08:00:00Z",
                        "updated_at_two": "2026-03-14T08:00:01Z",
                        "source_one": "rest_snapshot",
                        "source_two": "rest_snapshot",
                    }
                },
                0,
                {
                    "best_odds_items": 1,
                    "best_odds_null_count": 0,
                    "best_odds_with_any_odds": 1,
                    "best_odds_with_both_odds": 1,
                },
            )

        counterparty_fetcher = _make_counterparty_fetcher("betdex", betdex.PROVIDER_TITLE)

        with (
            patch.object(sx_bet, "get_shared_client", new=_fake_shared_client),
            patch.object(sx_bet, "_load_upcoming_fixtures_async", new=_fake_load_upcoming_fixtures_async),
            patch.object(sx_bet, "_load_best_odds_map_async", new=_fake_load_best_odds_map_async),
            patch.object(sx_bet, "_load_best_stake_map_async", new=_fake_load_best_stake_map_async),
        ):
            events = asyncio.run(
                sx_bet.fetch_events_async(
                    SPORT_KEY,
                    ["h2h"],
                    ["us"],
                    bookmakers=[sx_bet.PROVIDER_KEY],
                )
            )

            outcomes = _standardized_outcomes(events, sx_bet.PROVIDER_KEY)
            self.assertEqual(outcomes[HOME_TEAM]["price"], 2.35)
            self.assertEqual(outcomes[AWAY_TEAM]["price"], 1.66)
            self.assertEqual(outcomes[HOME_TEAM]["stake"], 150.0)
            self.assertEqual(outcomes[AWAY_TEAM]["stake"], 120.0)
            self.assertEqual(outcomes[HOME_TEAM]["last_updated"], "2026-03-14T08:00:00Z")
            self.assertEqual(outcomes[AWAY_TEAM]["last_updated"], "2026-03-14T08:00:01Z")
            self.assertEqual(outcomes[HOME_TEAM]["quote_source"], "rest_snapshot")
            self.assertEqual(outcomes[AWAY_TEAM]["quote_source"], "rest_snapshot")
            direct_stats = dict(sx_bet.fetch_events_async.last_stats)
            self.assertEqual(direct_stats.get("fixture_source_used"), "summary")
            self.assertEqual(direct_stats.get("odds_lookup_requested"), 1)
            self.assertEqual(direct_stats.get("orders_lookup_with_any_stake"), 1)

            result = self._run_provider_only_scan(
                provider_key=sx_bet.PROVIDER_KEY,
                counterparty_key="betdex",
                counterparty_fetcher=counterparty_fetcher,
            )

        self._assert_arbitrage_result(
            result,
            provider_key=sx_bet.PROVIDER_KEY,
            provider_title=sx_bet.PROVIDER_TITLE,
            counterparty_title=betdex.PROVIDER_TITLE,
            stats_assertion=lambda stats: (
                self.assertEqual(stats.get("fixture_source_used"), "summary"),
                self.assertEqual(stats.get("odds_lookup_requested"), 1),
                self.assertEqual(stats.get("orders_lookup_with_any_stake"), 1),
            ),
        )

    def test_bookmaker_xyz_pipeline_produces_standardized_h2h_and_arbitrage(self) -> None:
        dictionaries = {
            "marketNames": {"19-76-76": "Match Winner"},
            "outcomes": {
                "6983": {
                    "selectionId": 10009,
                    "marketId": 19,
                    "gamePeriodId": 76,
                    "gameTypeId": 76,
                    "pointsId": None,
                    "teamPlayerId": None,
                },
                "6984": {
                    "selectionId": 10010,
                    "marketId": 19,
                    "gamePeriodId": 76,
                    "gameTypeId": 76,
                    "pointsId": None,
                    "teamPlayerId": None,
                },
            },
            "selections": {"10009": "1", "10010": "2"},
            "teamPlayers": {},
            "points": {},
        }
        conditions = [
            {
                "__chain_id": "137",
                "state": "Active",
                "name": "Match Winner",
                "game": {
                    "gameId": "game-1",
                    "slug": "home-vs-away",
                    "title": f"{HOME_TEAM} vs {AWAY_TEAM}",
                    "startsAt": COMMENCE_TIME,
                    "participants": [
                        {"name": HOME_TEAM},
                        {"name": AWAY_TEAM},
                    ],
                    "sport": {"slug": "basketball", "name": "Basketball"},
                    "league": {"slug": "nba", "name": "NBA"},
                    "country": {"slug": "united-states", "name": "United States"},
                },
                "outcomes": [
                    {"outcomeId": "6983", "currentOdds": "2.25", "sortOrder": 1},
                    {"outcomeId": "6984", "currentOdds": "1.70", "sortOrder": 2},
                ],
            }
        ]

        async def _fake_load_dictionaries_async(*args, **kwargs):
            return (
                dictionaries,
                {
                    "cache": "miss",
                    "source": "test-dictionaries",
                    "retries_used": 0,
                },
            )

        async def _fake_load_market_manager_snapshot_async(*args, **kwargs):
            return (
                conditions,
                {
                    "cache": "miss",
                    "pages_fetched": 1,
                    "requests_made": 1,
                    "retries_used": 0,
                    "source_strategy": "official_market_manager",
                    "environments": ["PolygonUSDT"],
                },
            )

        counterparty_fetcher = _make_counterparty_fetcher("sx_bet", sx_bet.PROVIDER_TITLE)

        with (
            patch.object(bookmaker_xyz, "get_shared_client", new=_fake_shared_client),
            patch.object(bookmaker_xyz, "_load_dictionaries_async", new=_fake_load_dictionaries_async),
            patch.object(
                bookmaker_xyz,
                "_load_market_manager_snapshot_async",
                new=_fake_load_market_manager_snapshot_async,
            ),
        ):
            events = asyncio.run(
                bookmaker_xyz.fetch_events_async(
                    SPORT_KEY,
                    ["h2h"],
                    ["us"],
                    bookmakers=[bookmaker_xyz.PROVIDER_KEY],
                )
            )

            outcomes = _standardized_outcomes(events, bookmaker_xyz.PROVIDER_KEY)
            self.assertEqual(outcomes[HOME_TEAM]["price"], 2.25)
            self.assertEqual(outcomes[AWAY_TEAM]["price"], 1.7)
            direct_stats = dict(bookmaker_xyz.fetch_events_async.last_stats)
            self.assertTrue(direct_stats.get("dictionary_loaded"))
            self.assertEqual(direct_stats.get("dictionary_market_count"), 1)

            result = self._run_provider_only_scan(
                provider_key=bookmaker_xyz.PROVIDER_KEY,
                counterparty_key="sx_bet",
                counterparty_fetcher=counterparty_fetcher,
            )

        self._assert_arbitrage_result(
            result,
            provider_key=bookmaker_xyz.PROVIDER_KEY,
            provider_title=bookmaker_xyz.PROVIDER_TITLE,
            counterparty_title=sx_bet.PROVIDER_TITLE,
            stats_assertion=lambda stats: (
                self.assertTrue(stats.get("dictionary_loaded")),
                self.assertEqual(stats.get("dictionary_market_count"), 1),
            ),
        )


if __name__ == "__main__":
    unittest.main()

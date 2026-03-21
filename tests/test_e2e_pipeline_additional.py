from __future__ import annotations

import datetime as dt
import unittest
from typing import Callable
from unittest.mock import patch

import scanner


SPORT_KEY = "basketball_nba"
HOME_TEAM = "Edge Home"
AWAY_TEAM = "Edge Away"
SCAN_EPOCH = int(dt.datetime(2026, 3, 21, 12, 0, tzinfo=dt.timezone.utc).timestamp())
COMMENCE_TIME = "2026-03-21T12:05:00Z"
SOCCER_LIVE_SPORT_KEY = "soccer_usa_mls"
SOCCER_HOME_TEAM = "Toronto FC"
SOCCER_AWAY_TEAM = "Columbus Crew"
SOCCER_COMMENCE_TIME = "2026-03-21T17:00:00Z"


def _make_live_provider_fetcher(
    provider_key: str,
    provider_title: str,
    *,
    home_price: float,
    away_price: float,
) -> Callable:
    async def _fetcher(
        sport_key: str,
        markets: list[str],
        regions: list[str],
        bookmakers=None,
        context=None,
    ):
        lowered = {
            str(book).strip().lower()
            for book in (bookmakers or [])
            if str(book).strip()
        }
        if lowered and provider_key not in lowered and provider_title.lower() not in lowered:
            return []
        _fetcher.last_stats = {
            "sport_key": sport_key,
            "markets": list(markets),
            "regions": list(regions),
            "bookmakers": list(bookmakers or []),
            "context": dict(context or {}),
        }
        return [
            {
                "id": "live-e2e-event",
                "sport_key": sport_key,
                "home_team": HOME_TEAM,
                "away_team": AWAY_TEAM,
                "commence_time": COMMENCE_TIME,
                "live_state": {
                    "status": "live",
                    "period": "Q1",
                    "score": "7-3",
                    "clock": "10:12",
                    "updated_at": SCAN_EPOCH - 1,
                },
                "bookmakers": [
                    {
                        "key": provider_key,
                        "title": provider_title,
                        "event_id": f"{provider_key}-event-id",
                        "event_url": f"https://example.com/{provider_key}/event/live-e2e-event",
                        "live_state": {
                            "status": "live",
                            "period": "Q1",
                            "score": "7-3",
                            "clock": "10:12",
                            "updated_at": SCAN_EPOCH - 1,
                        },
                        "markets": [
                            {
                                "key": "h2h",
                                "updated_at": SCAN_EPOCH - 1,
                                "outcomes": [
                                    {
                                        "name": HOME_TEAM,
                                        "price": home_price,
                                        "updated_at": SCAN_EPOCH - 1,
                                        "quote_source": "mock_live_feed",
                                    },
                                    {
                                        "name": AWAY_TEAM,
                                        "price": away_price,
                                        "updated_at": SCAN_EPOCH - 1,
                                        "quote_source": "mock_live_feed",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

    _fetcher.last_stats = {}
    return _fetcher


class E2EPipelineAdditionalTests(unittest.TestCase):
    def tearDown(self) -> None:
        scanner._set_current_request_logger(None)
        with scanner._REQUEST_TRACE_LOCK:
            scanner._REQUEST_TRACE_ACTIVE.clear()

    def test_prematch_e2e_pipeline_from_api_fetch_to_arbitrage_output(self) -> None:
        sports_payload = [
            {
                "key": SPORT_KEY,
                "title": "NBA",
                "active": True,
                "has_outrights": False,
            }
        ]
        odds_events = [
            {
                "id": "prematch-e2e-event",
                "sport_key": SPORT_KEY,
                "home_team": HOME_TEAM,
                "away_team": AWAY_TEAM,
                "commence_time": COMMENCE_TIME,
                "bookmakers": [
                    {
                        "key": "book_a",
                        "title": "Book A",
                        "event_id": "book-a-event",
                        "event_url": "https://example.com/book-a/prematch-e2e-event",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": HOME_TEAM, "price": 2.22},
                                    {"name": AWAY_TEAM, "price": 1.85},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "book_b",
                        "title": "Book B",
                        "event_id": "book-b-event",
                        "event_url": "https://example.com/book-b/prematch-e2e-event",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": HOME_TEAM, "price": 1.82},
                                    {"name": AWAY_TEAM, "price": 2.25},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]

        with (
            patch.object(scanner, "fetch_sports", return_value=sports_payload) as mocked_fetch_sports,
            patch.object(
                scanner,
                "fetch_odds_for_sport_multi_market",
                return_value=(odds_events, []),
            ) as mocked_fetch_odds,
            patch.object(scanner, "_persist_provider_snapshots", return_value={}),
            patch.object(scanner, "_persist_cross_provider_match_report", return_value=""),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
            patch.object(scanner, "time") as mocked_time,
        ):
            mocked_time.time.return_value = SCAN_EPOCH
            mocked_time.perf_counter.side_effect = (1000.0 + i * 0.01 for i in range(10000))
            result = scanner.run_scan(
                api_key="dummy-key",
                sports=[SPORT_KEY],
                scan_mode="prematch",
                regions=["us"],
                include_providers=[],
                stake_amount=100.0,
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("scan_mode"), "prematch")
        self.assertIn("us", result.get("regions") or [])
        self.assertEqual(result.get("sport_errors"), [])
        mocked_fetch_sports.assert_called_once()
        mocked_fetch_odds.assert_called_once()
        fetch_args = mocked_fetch_odds.call_args.args
        self.assertEqual(fetch_args[1], SPORT_KEY)
        self.assertIn("h2h", fetch_args[2])
        self.assertIn("us", fetch_args[3])

        arbitrage = result.get("arbitrage") or {}
        self.assertGreater(arbitrage.get("opportunities_count", 0), 0)
        first = arbitrage["opportunities"][0]
        self.assertEqual(first.get("market"), "h2h")
        self.assertGreater(first.get("roi_percent", 0), 0)
        self.assertGreater(first.get("stakes", {}).get("guaranteed_profit", 0), 0)

        books = {
            item.get("bookmaker")
            for item in (first.get("best_odds") or [])
            if isinstance(item, dict)
        }
        self.assertEqual(books, {"Book A", "Book B"})
        summary = arbitrage.get("summary") or {}
        self.assertEqual(summary.get("events_scanned"), 1)
        self.assertEqual(summary.get("sports_scanned"), 1)

    def test_prematch_soccer_pipeline_requests_and_analyzes_h2h_markets(self) -> None:
        sports_payload = [
            {
                "key": SOCCER_LIVE_SPORT_KEY,
                "title": "MLS",
                "active": True,
                "has_outrights": False,
            }
        ]
        odds_events = [
            {
                "id": "prematch-soccer-event",
                "sport_key": SOCCER_LIVE_SPORT_KEY,
                "home_team": SOCCER_HOME_TEAM,
                "away_team": SOCCER_AWAY_TEAM,
                "commence_time": SOCCER_COMMENCE_TIME,
                "bookmakers": [
                    {
                        "key": "book_a",
                        "title": "Book A",
                        "event_id": "book-a-soccer-event",
                        "event_url": "https://example.com/book-a/prematch-soccer-event",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": SOCCER_HOME_TEAM, "price": 2.24},
                                    {"name": SOCCER_AWAY_TEAM, "price": 1.8},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "book_b",
                        "title": "Book B",
                        "event_id": "book-b-soccer-event",
                        "event_url": "https://example.com/book-b/prematch-soccer-event",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": SOCCER_HOME_TEAM, "price": 1.8},
                                    {"name": SOCCER_AWAY_TEAM, "price": 2.24},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]

        with (
            patch.object(scanner, "fetch_sports", return_value=sports_payload) as mocked_fetch_sports,
            patch.object(
                scanner,
                "fetch_odds_for_sport_multi_market",
                return_value=(odds_events, []),
            ) as mocked_fetch_odds,
            patch.object(scanner, "_persist_provider_snapshots", return_value={}),
            patch.object(scanner, "_persist_cross_provider_match_report", return_value=""),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
            patch.object(scanner, "time") as mocked_time,
        ):
            mocked_time.time.return_value = SCAN_EPOCH
            mocked_time.perf_counter.side_effect = (1500.0 + i * 0.01 for i in range(10000))
            result = scanner.run_scan(
                api_key="dummy-key",
                sports=[SOCCER_LIVE_SPORT_KEY],
                scan_mode="prematch",
                regions=["us"],
                include_providers=[],
                stake_amount=100.0,
            )

        mocked_fetch_sports.assert_called_once()
        mocked_fetch_odds.assert_called_once()
        fetch_args = mocked_fetch_odds.call_args.args
        self.assertEqual(fetch_args[1], SOCCER_LIVE_SPORT_KEY)
        self.assertIn("h2h", fetch_args[2])
        self.assertIn("h2h_3_way", fetch_args[2])

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("scan_mode"), "prematch")
        arbitrage = result.get("arbitrage") or {}
        self.assertGreater(arbitrage.get("opportunities_count", 0), 0)
        first = arbitrage["opportunities"][0]
        self.assertEqual(first.get("market"), "h2h")
        self.assertEqual(first.get("home_team"), SOCCER_HOME_TEAM)
        self.assertEqual(first.get("away_team"), SOCCER_AWAY_TEAM)

    def test_live_e2e_pipeline_from_provider_fetch_to_arbitrage_output(self) -> None:
        sx_fetcher = _make_live_provider_fetcher(
            "sx_bet",
            "SX Bet",
            home_price=2.32,
            away_price=1.80,
        )
        betdex_fetcher = _make_live_provider_fetcher(
            "betdex",
            "BetDEX",
            home_price=1.78,
            away_price=2.34,
        )

        with (
            patch.dict(
                scanner.PROVIDER_FETCHERS,
                {"sx_bet": sx_fetcher, "betdex": betdex_fetcher},
                clear=False,
            ),
            patch.object(scanner, "fetch_sports") as mocked_fetch_sports,
            patch.object(scanner, "_persist_provider_snapshots", return_value={}),
            patch.object(scanner, "_persist_cross_provider_match_report", return_value=""),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
            patch.object(scanner, "_provider_fetch_max_workers", return_value=1),
            patch.object(scanner, "time") as mocked_time,
        ):
            mocked_time.time.return_value = SCAN_EPOCH
            mocked_time.perf_counter.side_effect = (2000.0 + i * 0.01 for i in range(10000))
            result = scanner.run_scan(
                api_key="",
                sports=[SPORT_KEY],
                scan_mode="live",
                regions=["us"],
                include_providers=["sx_bet", "betdex"],
                bookmakers=["sx_bet", "betdex"],
                stake_amount=100.0,
            )

        mocked_fetch_sports.assert_not_called()
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("scan_mode"), "live")
        self.assertIn("us", result.get("regions") or [])

        arbitrage = result.get("arbitrage") or {}
        self.assertGreater(arbitrage.get("opportunities_count", 0), 0)
        first = arbitrage["opportunities"][0]
        self.assertEqual(first.get("market"), "h2h")
        self.assertGreater(first.get("roi_percent", 0), 0)
        self.assertEqual((first.get("live_state") or {}).get("status"), "live")
        self.assertEqual(
            first.get("event"),
            f"{AWAY_TEAM} vs {HOME_TEAM}",
        )

        best_odds = first.get("best_odds") or []
        self.assertEqual(len(best_odds), 2)
        self.assertTrue(all(item.get("quote_updated_at") for item in best_odds))
        self.assertTrue(all(item.get("book_event_url") for item in best_odds))
        self.assertTrue(all(item.get("book_event_id") for item in best_odds))

        custom_providers = result.get("custom_providers") or {}
        sx_summary = custom_providers.get("sx_bet") or {}
        betdex_summary = custom_providers.get("betdex") or {}
        self.assertEqual(sx_summary.get("events_merged"), 1)
        self.assertEqual(betdex_summary.get("events_merged"), 1)
        sx_stats = ((sx_summary.get("sports") or [{}])[0]).get("stats") or {}
        betdex_stats = ((betdex_summary.get("sports") or [{}])[0]).get("stats") or {}
        self.assertEqual(sx_stats.get("context"), {"scan_mode": "live", "live": True})
        self.assertEqual(betdex_stats.get("context"), {"scan_mode": "live", "live": True})

    def test_live_e2e_pipeline_does_not_call_odds_api_even_when_api_key_exists(self) -> None:
        sx_fetcher = _make_live_provider_fetcher(
            "sx_bet",
            "SX Bet",
            home_price=2.31,
            away_price=1.80,
        )
        betdex_fetcher = _make_live_provider_fetcher(
            "betdex",
            "BetDEX",
            home_price=1.79,
            away_price=2.30,
        )

        with (
            patch.dict(
                scanner.PROVIDER_FETCHERS,
                {"sx_bet": sx_fetcher, "betdex": betdex_fetcher},
                clear=False,
            ),
            patch.object(
                scanner,
                "fetch_odds_for_sport_multi_market",
                side_effect=AssertionError("live mode should not fetch odds API"),
            ) as mocked_fetch_odds,
            patch.object(scanner, "fetch_sports") as mocked_fetch_sports,
            patch.object(scanner, "_persist_provider_snapshots", return_value={}),
            patch.object(scanner, "_persist_cross_provider_match_report", return_value=""),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
            patch.object(scanner, "_provider_fetch_max_workers", return_value=1),
            patch.object(scanner, "time") as mocked_time,
        ):
            mocked_time.time.return_value = SCAN_EPOCH
            mocked_time.perf_counter.side_effect = (3000.0 + i * 0.01 for i in range(10000))
            result = scanner.run_scan(
                api_key="live-has-api-key",
                sports=[SPORT_KEY],
                scan_mode="live",
                regions=["us"],
                include_providers=["sx_bet", "betdex"],
                bookmakers=["sx_bet", "betdex"],
                stake_amount=100.0,
            )

        mocked_fetch_sports.assert_not_called()
        mocked_fetch_odds.assert_not_called()
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("scan_mode"), "live")
        self.assertGreater((result.get("arbitrage") or {}).get("opportunities_count", 0), 0)

    def test_live_soccer_pipeline_analyzes_provider_h2h_markets_even_when_base_defaults_exclude_them(self) -> None:
        def _soccer_fetcher(
            provider_key: str,
            provider_title: str,
            home_price: float,
            away_price: float,
        ):
            async def _fetcher(
                sport_key: str,
                markets: list[str],
                regions: list[str],
                bookmakers=None,
                context=None,
            ) -> list[dict]:
                lowered = {
                    str(book).strip().lower()
                    for book in (bookmakers or [])
                    if str(book).strip()
                }
                if lowered and provider_key not in lowered and provider_title.lower() not in lowered:
                    return []
                _fetcher.last_stats = {
                    "sport_key": sport_key,
                    "markets": list(markets),
                    "regions": list(regions),
                    "bookmakers": list(bookmakers or []),
                    "context": dict(context or {}),
                }
                return [
                    {
                        "id": "live-soccer-event",
                        "sport_key": sport_key,
                        "home_team": SOCCER_HOME_TEAM,
                        "away_team": SOCCER_AWAY_TEAM,
                        "commence_time": SOCCER_COMMENCE_TIME,
                        "live_state": {
                            "status": "live",
                            "period": "2H",
                            "score": "1-1",
                            "clock": "79:00",
                            "updated_at": SCAN_EPOCH - 1,
                        },
                        "bookmakers": [
                            {
                                "key": provider_key,
                                "title": provider_title,
                                "event_id": f"{provider_key}-soccer-live-event",
                                "event_url": f"https://example.com/{provider_key}/event/live-soccer-event",
                                "live_state": {
                                    "status": "live",
                                    "period": "2H",
                                    "score": "1-1",
                                    "clock": "79:00",
                                    "updated_at": SCAN_EPOCH - 1,
                                },
                                "markets": [
                                    {
                                        "key": "h2h",
                                        "updated_at": SCAN_EPOCH - 1,
                                        "outcomes": [
                                            {
                                                "name": SOCCER_HOME_TEAM,
                                                "price": home_price,
                                                "updated_at": SCAN_EPOCH - 1,
                                            },
                                            {
                                                "name": SOCCER_AWAY_TEAM,
                                                "price": away_price,
                                                "updated_at": SCAN_EPOCH - 1,
                                            },
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]

            _fetcher.last_stats = {}
            return _fetcher

        sx_fetcher = _soccer_fetcher("sx_bet", "SX Bet", 2.32, 1.8)
        poly_fetcher = _soccer_fetcher("polymarket", "Polymarket", 1.78, 2.34)

        with (
            patch.dict(
                scanner.PROVIDER_FETCHERS,
                {"sx_bet": sx_fetcher, "polymarket": poly_fetcher},
                clear=False,
            ),
            patch.object(scanner, "fetch_sports") as mocked_fetch_sports,
            patch.object(scanner, "_persist_provider_snapshots", return_value={}),
            patch.object(scanner, "_persist_cross_provider_match_report", return_value=""),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
            patch.object(scanner, "_provider_fetch_max_workers", return_value=1),
            patch.object(scanner, "time") as mocked_time,
        ):
            mocked_time.time.return_value = SCAN_EPOCH
            mocked_time.perf_counter.side_effect = (3500.0 + i * 0.01 for i in range(10000))
            result = scanner.run_scan(
                api_key="",
                sports=[SOCCER_LIVE_SPORT_KEY],
                scan_mode="live",
                regions=["us"],
                include_providers=["sx_bet", "polymarket"],
                bookmakers=["sx_bet", "polymarket"],
                stake_amount=100.0,
            )

        mocked_fetch_sports.assert_not_called()
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("scan_mode"), "live")
        arbitrage = result.get("arbitrage") or {}
        self.assertGreater(arbitrage.get("opportunities_count", 0), 0)
        first = arbitrage["opportunities"][0]
        self.assertEqual(first.get("market"), "h2h")
        self.assertEqual(first.get("home_team"), SOCCER_HOME_TEAM)
        self.assertEqual(first.get("away_team"), SOCCER_AWAY_TEAM)
        self.assertEqual((first.get("live_state") or {}).get("status"), "live")

    def test_prematch_e2e_pipeline_merges_api_and_provider_events_before_arbitrage(self) -> None:
        sports_payload = [
            {
                "key": SPORT_KEY,
                "title": "NBA",
                "active": True,
                "has_outrights": False,
            }
        ]
        odds_events = [
            {
                "id": "prematch-merge-event",
                "sport_key": SPORT_KEY,
                "home_team": HOME_TEAM,
                "away_team": AWAY_TEAM,
                "commence_time": COMMENCE_TIME,
                "bookmakers": [
                    {
                        "key": "book_a",
                        "title": "Book A",
                        "event_id": "api-book-a",
                        "event_url": "https://example.com/book-a/prematch-merge-event",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": HOME_TEAM, "price": 2.15},
                                    {"name": AWAY_TEAM, "price": 1.82},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        async def _provider_fetcher(
            sport_key: str,
            markets: list[str],
            regions: list[str],
            bookmakers=None,
            context=None,
        ) -> list[dict]:
            _provider_fetcher.last_stats = {
                "sport_key": sport_key,
                "markets": list(markets),
                "regions": list(regions),
                "bookmakers": list(bookmakers or []),
                "context": dict(context or {}),
            }
            return [
                {
                    "id": "prematch-merge-event",
                    "sport_key": sport_key,
                    "home_team": HOME_TEAM,
                    "away_team": AWAY_TEAM,
                    "commence_time": COMMENCE_TIME,
                    "bookmakers": [
                        {
                            "key": "sx_bet",
                            "title": "SX Bet",
                            "event_id": "sx-book",
                            "event_url": "https://example.com/sx/prematch-merge-event",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": HOME_TEAM, "price": 1.80},
                                        {"name": AWAY_TEAM, "price": 2.25},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        _provider_fetcher.last_stats = {}

        with (
            patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _provider_fetcher}, clear=False),
            patch.object(scanner, "fetch_sports", return_value=sports_payload),
            patch.object(
                scanner,
                "fetch_odds_for_sport_multi_market",
                return_value=(odds_events, []),
            ) as mocked_fetch_odds,
            patch.object(scanner, "_persist_provider_snapshots", return_value={}),
            patch.object(scanner, "_persist_cross_provider_match_report", return_value=""),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
            patch.object(scanner, "_provider_fetch_max_workers", return_value=1),
            patch.object(scanner, "time") as mocked_time,
        ):
            mocked_time.time.return_value = SCAN_EPOCH
            mocked_time.perf_counter.side_effect = (4000.0 + i * 0.01 for i in range(10000))
            result = scanner.run_scan(
                api_key="dummy-key",
                sports=[SPORT_KEY],
                scan_mode="prematch",
                regions=["us"],
                include_providers=["sx_bet"],
                stake_amount=100.0,
            )

        mocked_fetch_odds.assert_called_once()
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("scan_mode"), "prematch")
        arbitrage = result.get("arbitrage") or {}
        self.assertGreater(arbitrage.get("opportunities_count", 0), 0)
        first = arbitrage["opportunities"][0]
        books = {
            item.get("bookmaker")
            for item in (first.get("best_odds") or [])
            if isinstance(item, dict)
        }
        self.assertEqual(books, {"Book A", "SX Bet"})
        self.assertEqual(first.get("market"), "h2h")
        self.assertGreater(first.get("roi_percent", 0), 0.0)

        sx_summary = ((result.get("custom_providers") or {}).get("sx_bet") or {})
        self.assertEqual(sx_summary.get("events_merged"), 1)
        sx_stats = ((sx_summary.get("sports") or [{}])[0]).get("stats") or {}
        self.assertEqual(sx_stats.get("context"), {"scan_mode": "prematch", "live": False})

    def test_prematch_provider_only_selection_reports_api_skip_warning(self) -> None:
        async def _provider_fetcher(
            sport_key: str,
            markets: list[str],
            regions: list[str],
            bookmakers=None,
            context=None,
        ) -> list[dict]:
            return [
                {
                    "id": "provider-only-event",
                    "sport_key": sport_key,
                    "home_team": HOME_TEAM,
                    "away_team": AWAY_TEAM,
                    "commence_time": COMMENCE_TIME,
                    "bookmakers": [
                        {
                            "key": "sx_bet",
                            "title": "SX Bet",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": HOME_TEAM, "price": 1.82},
                                        {"name": AWAY_TEAM, "price": 2.24},
                                    ],
                                }
                            ],
                        },
                        {
                            "key": "betdex",
                            "title": "BetDEX",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": HOME_TEAM, "price": 2.24},
                                        {"name": AWAY_TEAM, "price": 1.82},
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ]

        with (
            patch.dict(
                scanner.PROVIDER_FETCHERS,
                {"sx_bet": _provider_fetcher, "betdex": _provider_fetcher},
                clear=False,
            ),
            patch.object(
                scanner,
                "fetch_odds_for_sport_multi_market",
                side_effect=AssertionError("provider-only selection should skip odds API"),
            ) as mocked_fetch_odds,
            patch.object(scanner, "fetch_sports") as mocked_fetch_sports,
            patch.object(scanner, "_persist_provider_snapshots", return_value={}),
            patch.object(scanner, "_persist_cross_provider_match_report", return_value=""),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
            patch.object(scanner, "_provider_fetch_max_workers", return_value=1),
            patch.object(scanner, "time") as mocked_time,
        ):
            mocked_time.time.return_value = SCAN_EPOCH
            mocked_time.perf_counter.side_effect = (5000.0 + i * 0.01 for i in range(10000))
            result = scanner.run_scan(
                api_key="dummy-key",
                sports=[SPORT_KEY],
                scan_mode="prematch",
                regions=["us"],
                bookmakers=["sx_bet", "betdex"],
                include_providers=["sx_bet", "betdex"],
                stake_amount=100.0,
            )

        mocked_fetch_sports.assert_not_called()
        mocked_fetch_odds.assert_not_called()
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("api_disabled_reason"), "provider_only_bookmakers_selected")
        self.assertIn(
            "Odds API fetch skipped because only custom provider bookmakers were selected.",
            result.get("warnings") or [],
        )

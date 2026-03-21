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

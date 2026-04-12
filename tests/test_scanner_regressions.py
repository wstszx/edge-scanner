import asyncio
import concurrent.futures
import datetime as dt
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scanner


class ScannerRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        scanner._set_current_request_logger(None)
        with scanner._REQUEST_TRACE_LOCK:
            scanner._REQUEST_TRACE_ACTIVE.clear()

    def _make_provider_event(self, event_id: str, commence_ts: int, live_state=None) -> dict:
        event = {
            "id": event_id,
            "sport_key": "basketball_nba",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "commence_time": dt.datetime.fromtimestamp(commence_ts, dt.timezone.utc).isoformat(),
            "bookmakers": [
                {
                    "key": "sx_bet",
                    "title": "SX Bet",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.0},
                                {"name": "Away Team", "price": 1.9},
                            ],
                        }
                    ],
                }
            ],
        }
        if live_state is not None:
            event["live_state"] = live_state
        return event

    def test_normalize_bookmakers_filters_non_supported_soft_books(self) -> None:
        self.assertEqual(
            scanner._normalize_bookmakers(["DraftKings", "pinnacle", "SX Bet", "betdex"]),
            ["pinnacle", "sx_bet", "betdex"],
        )

    def test_run_scan_derives_regions_from_selected_bookmakers_when_missing(self) -> None:
        captured_regions = []
        sports_payload = [
            {
                "key": "americanfootball_nfl",
                "title": "NFL",
                "active": True,
                "has_outrights": False,
            }
        ]

        def _fake_scan_single_sport(**kwargs):
            captured_regions.append(list(kwargs.get("normalized_regions") or []))
            sport = kwargs.get("sport") or {}
            sport_key = sport.get("key") or ""
            sport_title = sport.get("title") or sport_key
            return {
                "skipped": False,
                "sport_key": sport_key,
                "sport_timing": {
                    "sport_key": sport_key,
                    "sport": sport_title,
                    "api_fetch_ms": 0.0,
                    "provider_fetch_ms": 0.0,
                    "analysis_ms": 0.0,
                    "events_scanned": 0,
                    "providers": [],
                    "total_ms": 0.0,
                },
                "timing_steps": [],
                "api_market_skips": [],
                "sport_errors": [],
                "provider_updates": {},
                "provider_snapshot_updates": {},
                "events_scanned": 0,
                "total_profit": 0.0,
                "arb_opportunities": [],
                "middle_opportunities": [],
                "plus_ev_opportunities": [],
                "stale_event_filters": [],
                "successful": 1,
            }

        with (
            patch.object(scanner, "fetch_sports", return_value=sports_payload),
            patch.object(scanner, "_scan_single_sport", side_effect=_fake_scan_single_sport),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
        ):
            result = scanner.run_scan(
                api_key="dummy",
                sports=["americanfootball_nfl"],
                bookmakers=["betfair_ex_uk", "sx_bet"],
                sharp_book="pinnacle",
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(captured_regions, [["uk", "eu"]])

    def test_kelly_stake_guard_paths_return_triplet(self) -> None:
        guard_cases = [
            (0.5, 2.0, 0.0, 0.25),
            (0.5, 1.0, 1000.0, 0.25),
            (0.1, 2.0, 1000.0, 0.25),
        ]
        for args in guard_cases:
            with self.subTest(args=args):
                self.assertEqual(scanner._kelly_stake(*args), (0.0, 0.0, 0.0))

    def test_kelly_stake_positive_path_returns_triplet(self) -> None:
        full_pct, fraction_pct, stake = scanner._kelly_stake(0.6, 2.2, 1000.0, 0.5)
        self.assertGreater(full_pct, 0.0)
        self.assertGreater(fraction_pct, 0.0)
        self.assertGreater(stake, 0.0)

    def test_select_request_logger_uses_current_thread_context(self) -> None:
        logger_a = scanner._ScanRequestLogger("scan-a")
        logger_b = scanner._ScanRequestLogger("scan-b")
        scanner._set_current_request_logger(logger_a)
        selected = scanner._select_request_logger([logger_a, logger_b])
        self.assertIs(selected, logger_a)

    def test_select_request_logger_does_not_cross_scan_when_context_mismatch(self) -> None:
        logger_a = scanner._ScanRequestLogger("scan-a")
        logger_b = scanner._ScanRequestLogger("scan-b")
        scanner._set_current_request_logger(logger_a)
        self.assertIsNone(scanner._select_request_logger([logger_b]))

        scanner._set_current_request_logger(None)
        self.assertIs(scanner._select_request_logger([logger_b]), logger_b)
        self.assertIsNone(scanner._select_request_logger([logger_a, logger_b]))

    def test_submit_with_request_logger_propagates_context_to_worker(self) -> None:
        logger = scanner._ScanRequestLogger("scan-main")
        scanner._set_current_request_logger(logger)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = scanner._submit_with_request_logger(
                executor,
                scanner._current_request_logger,
            )
            self.assertIs(future.result(), logger)

    def test_fetch_provider_events_for_sport_accepts_async_fetcher(self) -> None:
        async def _fake_fetcher(
            sport_key: str,
            markets: list[str],
            regions: list[str],
            bookmakers=None,
        ):
            _fake_fetcher.last_stats = {
                "sport_key": sport_key,
                "markets": list(markets),
                "regions": list(regions),
                "bookmakers": list(bookmakers or []),
            }
            return [{"id": "event-1"}]

        _fake_fetcher.last_stats = {}

        with patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _fake_fetcher}):
            result = asyncio.run(
                scanner._fetch_provider_events_for_sport(
                    provider_key="sx_bet",
                    sport_key="americanfootball_nfl",
                    provider_markets=["h2h"],
                    regions=["us"],
                    bookmakers=["sx_bet"],
                )
            )

        self.assertEqual(result.get("events"), [{"id": "event-1"}])
        self.assertEqual(
            result.get("stats"),
            {
                "sport_key": "americanfootball_nfl",
                "markets": ["h2h"],
                "regions": ["us"],
                "bookmakers": ["sx_bet"],
            },
        )

    def test_fetch_provider_events_for_sport_passes_context_when_supported(self) -> None:
        async def _fake_fetcher(
            sport_key: str,
            markets: list[str],
            regions: list[str],
            bookmakers=None,
            context=None,
        ):
            _fake_fetcher.last_stats = {
                "sport_key": sport_key,
                "context": dict(context or {}),
            }
            return [{"id": "event-1"}]

        _fake_fetcher.last_stats = {}

        with patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _fake_fetcher}):
            result = asyncio.run(
                scanner._fetch_provider_events_for_sport(
                    provider_key="sx_bet",
                    sport_key="basketball_nba",
                    provider_markets=["h2h"],
                    regions=["us"],
                    bookmakers=["sx_bet"],
                    provider_context={"scan_mode": "live", "live": True},
                )
            )

        self.assertEqual(result.get("events"), [{"id": "event-1"}])
        stats = result.get("stats") or {}
        self.assertEqual(stats.get("context"), {"scan_mode": "live", "live": True})

    def test_cleanup_old_request_logs_keeps_newest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            created = []
            for index in range(4):
                path = target_dir / f"requests_20260317_12000{index}.jsonl"
                path.write_text(f"log-{index}\n", encoding="utf-8")
                os.utime(path, (100 + index, 100 + index))
                created.append(path)

            with patch.object(scanner, "SCAN_REQUEST_LOG_RETENTION_FILES", 2):
                scanner._cleanup_old_request_logs(target_dir)

            remaining = sorted(path.name for path in target_dir.glob("requests_*.jsonl"))
            self.assertEqual(
                remaining,
                [
                    created[2].name,
                    created[3].name,
                ],
            )

    def test_run_scan_async_live_mode_skips_odds_api_and_passes_scan_mode(self) -> None:
        def _fake_scan_single_sport(**kwargs):
            return {
                "skipped": False,
                "sport_key": kwargs["sport"]["key"],
                "sport_timing": {"sport_key": kwargs["sport"]["key"], "total_ms": 0.0},
                "timing_steps": [],
                "api_market_skips": [],
                "sport_errors": [],
                "provider_updates": {},
                "provider_snapshot_updates": {},
                "events_scanned": 0,
                "total_profit": 0.0,
                "arb_opportunities": [],
                "middle_opportunities": [],
                "plus_ev_opportunities": [],
                "stale_event_filters": [],
                "successful": 1,
            }

        with (
            patch.object(scanner, "fetch_sports", side_effect=AssertionError("Odds API should not be called")),
            patch.object(scanner, "_scan_single_sport", side_effect=_fake_scan_single_sport) as mocked_scan,
        ):
            result = asyncio.run(
                scanner.run_scan_async(
                    api_key="",
                    sports=["basketball_nba"],
                    scan_mode="live",
                    regions=["us"],
                )
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("scan_mode"), "live")
        self.assertEqual(mocked_scan.call_args.kwargs.get("scan_mode"), "live")
        self.assertEqual(
            mocked_scan.call_args.kwargs.get("enabled_provider_keys"),
            scanner._default_live_provider_keys(),
        )

    def test_default_live_provider_keys_prefers_curated_supported_list(self) -> None:
        defaults = scanner._default_live_provider_keys()
        self.assertEqual(defaults, ['sx_bet', 'betdex', 'polymarket', 'bookmaker_xyz', 'artline'])

    def test_registered_provider_fetchers_use_async_entrypoints_for_migrated_providers(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(scanner.PROVIDER_FETCHERS["betdex"]))
        self.assertTrue(inspect.iscoroutinefunction(scanner.PROVIDER_FETCHERS["bookmaker_xyz"]))
        self.assertTrue(inspect.iscoroutinefunction(scanner.PROVIDER_FETCHERS["sx_bet"]))
        self.assertTrue(inspect.iscoroutinefunction(scanner.PROVIDER_FETCHERS["polymarket"]))
        self.assertNotIn("dexsport_io", scanner.PROVIDER_FETCHERS)
        self.assertNotIn("sportbet_one", scanner.PROVIDER_FETCHERS)

    def test_dedupe_proxy_provider_keys_returns_unique_registered_keys(self) -> None:
        deduped = scanner._dedupe_proxy_provider_keys(
            ["bookmaker_xyz", "sx_bet", "bookmaker_xyz", "betdex"],
            explicit_provider_keys=["sx_bet"],
        )
        self.assertIn("bookmaker_xyz", deduped)
        self.assertIn("sx_bet", deduped)
        self.assertIn("betdex", deduped)
        self.assertEqual(deduped.count("bookmaker_xyz"), 1)

    def test_dedupe_proxy_provider_keys_ignores_unregistered_provider_keys(self) -> None:
        deduped = scanner._dedupe_proxy_provider_keys(
            ["bookmaker_xyz", "dexsport_io", "sportbet_one", "sx_bet"],
            explicit_provider_keys=["dexsport_io", "sx_bet"],
        )
        self.assertIn("bookmaker_xyz", deduped)
        self.assertIn("sx_bet", deduped)
        self.assertNotIn("dexsport_io", deduped)
        self.assertNotIn("sportbet_one", deduped)

    def test_scan_single_sport_retries_async_provider_network_errors(self) -> None:
        calls = {"count": 0}

        async def _flaky_fetcher(
            sport_key: str,
            markets: list[str],
            regions: list[str],
            bookmakers=None,
        ):
            calls["count"] += 1
            _flaky_fetcher.last_stats = {
                "sport_key": sport_key,
                "attempt": calls["count"],
            }
            if calls["count"] == 1:
                raise RuntimeError("request timed out")
            return []

        _flaky_fetcher.last_stats = {}

        with (
            patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _flaky_fetcher}),
            patch.dict(scanner.PROVIDER_TITLES, {"sx_bet": "SX Bet"}),
            patch.object(scanner, "_provider_network_retry_delay_seconds", return_value=0.0),
        ):
            result = asyncio.run(
                scanner._scan_single_sport(
                    sport={"key": "americanfootball_nfl", "title": "NFL"},
                    scan_mode="prematch",
                    all_markets=False,
                    should_fetch_api=False,
                    api_pool=scanner.ApiKeyPool([]),
                    normalized_regions=["us"],
                    api_bookmakers=[],
                    provider_target_sport_keys=["americanfootball_nfl"],
                    enabled_provider_keys=["sx_bet"],
                    normalized_bookmakers=["sx_bet"],
                    stake_amount=100.0,
                    commission_rate=0.0,
                    sharp_priority=scanner._sharp_priority("pinnacle"),
                    min_edge_percent=0.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                )
            )

        self.assertEqual(calls["count"], 2)
        self.assertEqual(result.get("sport_errors"), [])
        provider_updates = result.get("provider_updates") or {}
        self.assertIn("sx_bet", provider_updates)
        provider_sports = provider_updates["sx_bet"].get("sports") or []
        self.assertTrue(provider_sports)
        stats = provider_sports[0].get("stats") or {}
        self.assertTrue(stats.get("network_retry_recovered"))

    def test_scan_single_sport_snapshot_updates_keep_provider_payloads_isolated(self) -> None:
        async def _bookmaker_fetcher(
            sport_key: str,
            markets: list[str],
            regions: list[str],
            bookmakers=None,
        ):
            _bookmaker_fetcher.last_stats = {"provider": "bookmaker_xyz", "sport_key": sport_key}
            return [
                {
                    "id": "bookmaker-event",
                    "sport_key": sport_key,
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "commence_time": "2026-03-13T00:00:00Z",
                    "bookmakers": [
                        {
                            "key": "bookmaker_xyz",
                            "title": "bookmaker.xyz",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Home Team", "price": 2.1},
                                        {"name": "Away Team", "price": 1.8},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        async def _sx_fetcher(
            sport_key: str,
            markets: list[str],
            regions: list[str],
            bookmakers=None,
        ):
            _sx_fetcher.last_stats = {"provider": "sx_bet", "sport_key": sport_key}
            return [
                {
                    "id": "sx-event",
                    "sport_key": sport_key,
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "commence_time": "2026-03-13T00:00:00Z",
                    "bookmakers": [
                        {
                            "key": "sx_bet",
                            "title": "SX Bet",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Home Team", "price": 2.2},
                                        {"name": "Away Team", "price": 1.75},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        _bookmaker_fetcher.last_stats = {}
        _sx_fetcher.last_stats = {}

        with (
            patch.dict(
                scanner.PROVIDER_FETCHERS,
                {
                    "bookmaker_xyz": _bookmaker_fetcher,
                    "sx_bet": _sx_fetcher,
                },
            ),
            patch.dict(
                scanner.PROVIDER_TITLES,
                {
                    "bookmaker_xyz": "bookmaker.xyz",
                    "sx_bet": "SX Bet",
                },
            ),
        ):
            result = asyncio.run(
                scanner._scan_single_sport(
                    sport={"key": "basketball_nba", "title": "NBA"},
                    scan_mode="prematch",
                    all_markets=False,
                    should_fetch_api=False,
                    api_pool=scanner.ApiKeyPool([]),
                    normalized_regions=["us"],
                    api_bookmakers=[],
                    provider_target_sport_keys=["basketball_nba"],
                    enabled_provider_keys=["bookmaker_xyz", "sx_bet"],
                    normalized_bookmakers=["bookmaker_xyz", "sx_bet"],
                    stake_amount=100.0,
                    commission_rate=0.0,
                    sharp_priority=scanner._sharp_priority("pinnacle"),
                    min_edge_percent=0.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                )
            )

        snapshot_updates = result.get("provider_snapshot_updates") or {}
        bookmaker_events = snapshot_updates["bookmaker_xyz"]["events"]
        sx_events = snapshot_updates["sx_bet"]["events"]

        self.assertEqual(len(bookmaker_events), 1)
        self.assertEqual(len(sx_events), 1)
        self.assertEqual(len(bookmaker_events[0]["bookmakers"]), 1)
        self.assertEqual(len(sx_events[0]["bookmakers"]), 1)
        self.assertEqual(bookmaker_events[0]["bookmakers"][0]["key"], "bookmaker_xyz")
        self.assertEqual(sx_events[0]["bookmakers"][0]["key"], "sx_bet")

    def test_scan_single_sport_live_filter_stats_exposed(self) -> None:
        base_epoch = 1700000000

        async def _fetcher(sport_key, markets, regions, bookmakers=None):
            return [
                self._make_provider_event("live-event", base_epoch + 120, {"status": "live"}),
                self._make_provider_event("scheduled-event", base_epoch + 60, {"status": "scheduled"}),
                self._make_provider_event("past-event", base_epoch - 3600),
            ]

        _fetcher.last_stats = {}

        with (
            patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _fetcher}),
            patch.dict(scanner.PROVIDER_TITLES, {"sx_bet": "SX Bet"}),
            patch.object(scanner.time, "time", return_value=base_epoch),
            patch.object(scanner, "EVENT_MAX_PAST_MINUTES_RAW", "30"),
            patch.object(scanner, "LIVE_EVENT_MAX_FUTURE_SECONDS_RAW", "600"),
        ):
            result = asyncio.run(
                scanner._scan_single_sport(
                    sport={"key": "basketball_nba", "title": "NBA"},
                    scan_mode="live",
                    all_markets=False,
                    should_fetch_api=False,
                    api_pool=scanner.ApiKeyPool([]),
                    normalized_regions=["us"],
                    api_bookmakers=[],
                    provider_target_sport_keys=["basketball_nba"],
                    enabled_provider_keys=["sx_bet"],
                    normalized_bookmakers=["sx_bet"],
                    stake_amount=100.0,
                    commission_rate=0.0,
                    sharp_priority=scanner._sharp_priority("pinnacle"),
                    min_edge_percent=0.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                )
            )

        provider_updates = result.get("provider_updates") or {}
        self.assertIn("sx_bet", provider_updates)
        provider_sports = provider_updates["sx_bet"].get("sports") or []
        self.assertTrue(provider_sports)
        sport_row = provider_sports[0]
        self.assertEqual(sport_row.get("events_fetched_raw"), 3)
        self.assertEqual(sport_row.get("events_returned"), sport_row.get("events_fetched_raw"))
        self.assertEqual(sport_row.get("events_after_live_filter"), 1)
        snapshot_updates = result.get("provider_snapshot_updates") or {}
        self.assertEqual(
            sport_row.get("live_filter_stats"),
            {
                "dropped_not_live_state": 1,
                "dropped_terminal_state": 0,
                "dropped_past": 1,
                "dropped_future": 0,
                "dropped_missing_time": 0,
                "suspicious_explicit_live_future": 0,
            },
        )
        self.assertEqual(
            len(snapshot_updates.get("sx_bet", {}).get("events", [])),
            sport_row.get("events_fetched_raw"),
        )

    def test_scan_single_sport_live_filter_stats_retry_recaptured(self) -> None:
        base_epoch = 1700000000
        calls = {"count": 0}
        retry_events = [
            self._make_provider_event("live-event", base_epoch + 120, {"status": "live"}),
            self._make_provider_event("scheduled-event", base_epoch + 60, {"status": "scheduled"}),
        ]

        async def _fetcher(sport_key, markets, regions, bookmakers=None):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("timed out")
            return retry_events

        _fetcher.last_stats = {}

        with (
            patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _fetcher}),
            patch.dict(scanner.PROVIDER_TITLES, {"sx_bet": "SX Bet"}),
            patch.object(scanner.time, "time", return_value=base_epoch),
            patch.object(scanner, "EVENT_MAX_PAST_MINUTES_RAW", "30"),
            patch.object(scanner, "LIVE_EVENT_MAX_FUTURE_SECONDS_RAW", "600"),
            patch.object(scanner, "_provider_network_retry_delay_seconds", return_value=0.0),
        ):
            result = asyncio.run(
                scanner._scan_single_sport(
                    sport={"key": "basketball_nba", "title": "NBA"},
                    scan_mode="live",
                    all_markets=False,
                    should_fetch_api=False,
                    api_pool=scanner.ApiKeyPool([]),
                    normalized_regions=["us"],
                    api_bookmakers=[],
                    provider_target_sport_keys=["basketball_nba"],
                    enabled_provider_keys=["sx_bet"],
                    normalized_bookmakers=["sx_bet"],
                    stake_amount=100.0,
                    commission_rate=0.0,
                    sharp_priority=scanner._sharp_priority("pinnacle"),
                    min_edge_percent=0.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                )
            )

        provider_updates = result.get("provider_updates") or {}
        self.assertIn("sx_bet", provider_updates)
        provider_sports = provider_updates["sx_bet"].get("sports") or []
        self.assertTrue(provider_sports)
        sport_row = provider_sports[0]
        self.assertEqual(sport_row.get("events_fetched_raw"), 2)
        self.assertEqual(sport_row.get("events_after_live_filter"), 1)
        self.assertEqual(provider_updates["sx_bet"].get("events_merged"), 1)
        self.assertEqual(
            sport_row.get("live_filter_stats"),
            {
                "dropped_not_live_state": 1,
                "dropped_terminal_state": 0,
                "dropped_past": 0,
                "dropped_future": 0,
                "dropped_missing_time": 0,
                "suspicious_explicit_live_future": 0,
            },
        )
        snapshot_updates = result.get("provider_snapshot_updates") or {}
        self.assertEqual(
            len(snapshot_updates.get("sx_bet", {}).get("events", [])),
            len(retry_events),
        )
        self.assertEqual(sport_row.get("events_returned"), sport_row.get("events_fetched_raw"))
    def test_scan_single_sport_prematch_rows_skip_live_funnel_fields(self) -> None:
        base_epoch = 1700000000

        async def _fetcher(sport_key, markets, regions, bookmakers=None):
            return [self._make_provider_event("live-event", base_epoch + 120, {"status": "live"})]

        _fetcher.last_stats = {}

        with (
            patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _fetcher}),
            patch.dict(scanner.PROVIDER_TITLES, {"sx_bet": "SX Bet"}),
            patch.object(scanner.time, "time", return_value=base_epoch),
            patch.object(scanner, "EVENT_MAX_PAST_MINUTES_RAW", "30"),
            patch.object(scanner, "LIVE_EVENT_MAX_FUTURE_SECONDS_RAW", "600"),
        ):
            result = asyncio.run(
                scanner._scan_single_sport(
                    sport={"key": "basketball_nba", "title": "NBA"},
                    scan_mode="prematch",
                    all_markets=False,
                    should_fetch_api=False,
                    api_pool=scanner.ApiKeyPool([]),
                    normalized_regions=["us"],
                    api_bookmakers=[],
                    provider_target_sport_keys=["basketball_nba"],
                    enabled_provider_keys=["sx_bet"],
                    normalized_bookmakers=["sx_bet"],
                    stake_amount=100.0,
                    commission_rate=0.0,
                    sharp_priority=scanner._sharp_priority("pinnacle"),
                    min_edge_percent=0.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                )
            )

        provider_updates = result.get("provider_updates") or {}
        self.assertIn("sx_bet", provider_updates)
        sport_row = provider_updates["sx_bet"].get("sports", [])[0]
        self.assertNotIn("events_fetched_raw", sport_row)
        self.assertNotIn("events_after_live_filter", sport_row)
        self.assertNotIn("live_filter_stats", sport_row)

    def test_collect_market_entries_accepts_three_way_h2h(self) -> None:
        game = {
            "sport_key": "soccer_epl",
            "sport_display": "Premier League",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "bookmakers": [
                {
                    "key": "book_home",
                    "title": "Book Home",
                    "markets": [
                        {
                            "key": "h2h_3_way",
                            "outcomes": [
                                {"name": "Home FC", "price": 4.0},
                                {"name": "Draw", "price": 2.0},
                                {"name": "Away FC", "price": 2.0},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_draw",
                    "title": "Book Draw",
                    "markets": [
                        {
                            "key": "h2h_3_way",
                            "outcomes": [
                                {"name": "Home FC", "price": 2.0},
                                {"name": "Draw", "price": 4.0},
                                {"name": "Away FC", "price": 2.0},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_away",
                    "title": "Book Away",
                    "markets": [
                        {
                            "key": "h2h_3_way",
                            "outcomes": [
                                {"name": "Home FC", "price": 2.0},
                                {"name": "Draw", "price": 2.0},
                                {"name": "Away FC", "price": 4.0},
                            ],
                        }
                    ],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h_3_way",
            stake_total=100.0,
            commission_rate=0.0,
        )
        self.assertTrue(entries)
        self.assertGreater(entries[0].get("roi_percent", 0.0), 0.0)

    def test_collect_market_entries_rejects_single_book_arbitrage(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "sx_bet",
                    "title": "SX Bet",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.06},
                                {"name": "Away Team", "price": 2.04},
                            ],
                        }
                    ],
                }
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
        )
        self.assertEqual(entries, [])

    def test_collect_market_entries_keeps_cross_book_fallback_combo(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.3},
                                {"name": "Away Team", "price": 2.3},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.18},
                                {"name": "Away Team", "price": 1.8},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_c",
                    "title": "Book C",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 1.8},
                                {"name": "Away Team", "price": 2.18},
                            ],
                        }
                    ],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
        )
        self.assertTrue(entries)
        best_odds = entries[0].get("best_odds") or []
        selected_books = {
            str(item.get("bookmaker_key") or "").strip().lower()
            for item in best_odds
            if isinstance(item, dict)
        }
        self.assertGreaterEqual(len(selected_books), 2)
        self.assertGreater(entries[0].get("roi_percent", 0.0), 0.0)

    def test_collect_market_entries_accepts_string_price_values(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": "2.1"},
                                {"name": "Away Team", "price": "2.0"},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": "2.0"},
                                {"name": "Away Team", "price": "2.1"},
                            ],
                        }
                    ],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
        )
        self.assertTrue(entries)
        self.assertGreater(entries[0].get("roi_percent", 0.0), 0.0)

    def test_collect_market_entries_normalizes_outcome_name_case(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.1},
                                {"name": "Away Team", "price": 2.0},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "home team", "price": 2.0},
                                {"name": "away team", "price": 2.1},
                            ],
                        }
                    ],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
        )
        self.assertTrue(entries)
        self.assertGreater(entries[0].get("roi_percent", 0.0), 0.0)

    def test_collect_market_entries_matches_team_alias_outcomes_for_nba_moneyline(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Los Angeles Clippers",
            "away_team": "Golden State Warriors",
            "bookmakers": [
                {
                    "key": "sx_bet",
                    "title": "SX Bet",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Los Angeles Clippers", "price": 1.24},
                                {"name": "Golden State Warriors", "price": 4.5},
                            ],
                        }
                    ],
                },
                {
                    "key": "polymarket",
                    "title": "Polymarket",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Clippers", "price": 1.35},
                                {"name": "Warriors", "price": 3.33},
                            ],
                        }
                    ],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
        )

        self.assertTrue(entries)
        self.assertGreater(entries[0].get("roi_percent", 0.0), 0.0)
        best_odds = entries[0].get("best_odds") or []
        outcome_names = {str(item.get("outcome") or "").strip() for item in best_odds if isinstance(item, dict)}
        self.assertIn("Clippers", outcome_names)
        self.assertIn("Golden State Warriors", outcome_names)

    def test_collect_market_entries_preserves_quote_source(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "sx_bet",
                    "title": "SX Bet",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {
                                    "name": "Home Team",
                                    "price": 2.2,
                                    "quote_source": "ws",
                                    "last_updated": "2026-03-16T01:02:03Z",
                                },
                                {
                                    "name": "Away Team",
                                    "price": 1.8,
                                    "quote_source": "ws",
                                    "last_updated": "2026-03-16T01:02:03Z",
                                },
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {
                                    "name": "Home Team",
                                    "price": 1.8,
                                    "quote_source": "rest_snapshot",
                                    "last_updated": "2026-03-16T01:02:01Z",
                                },
                                {
                                    "name": "Away Team",
                                    "price": 2.2,
                                    "quote_source": "rest_snapshot",
                                    "last_updated": "2026-03-16T01:02:01Z",
                                },
                            ],
                        }
                    ],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
        )

        self.assertTrue(entries)
        best_odds = entries[0].get("best_odds") or []
        source_by_book = {
            str(item.get("bookmaker_key") or item.get("bookmaker") or "").strip().lower(): item.get("quote_source")
            for item in best_odds
            if isinstance(item, dict)
        }
        self.assertEqual(source_by_book.get("sx_bet"), "ws")
        self.assertEqual(source_by_book.get("book_b"), "rest_snapshot")

    def test_collect_market_entries_preserves_raw_percentage_odds(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "sx_bet",
                    "title": "SX Bet",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {
                                    "name": "Home Team",
                                    "price": 2.2,
                                    "raw_percentage_odds": "50875000000000000000",
                                },
                                {
                                    "name": "Away Team",
                                    "price": 1.8,
                                    "raw_percentage_odds": "47750000000000000000",
                                },
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {
                                    "name": "Home Team",
                                    "price": 1.8,
                                },
                                {
                                    "name": "Away Team",
                                    "price": 2.2,
                                },
                            ],
                        }
                    ],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
        )

        self.assertTrue(entries)
        best_odds = entries[0].get("best_odds") or []
        raw_by_book = {
            str(item.get("bookmaker_key") or item.get("bookmaker") or "").strip().lower(): item.get("raw_percentage_odds")
            for item in best_odds
            if isinstance(item, dict)
        }
        self.assertEqual(raw_by_book.get("sx_bet"), "50875000000000000000")
        self.assertIsNone(raw_by_book.get("book_b"))

    def test_filter_live_events_prefers_explicit_live_state_over_time_window(self) -> None:
        now_epoch = 1_700_000_000
        future_live_event = {
            "id": "future-live",
            "commence_time": "2030-01-01T00:00:00Z",
            "live_state": {"status": "live"},
        }
        current_scheduled_event = {
            "id": "current-scheduled",
            "commence_time": "2023-11-14T22:13:20Z",
            "live_state": {"status": "scheduled"},
        }

        with patch("scanner.time.time", return_value=now_epoch):
            filtered, stats = scanner._filter_live_events_for_scan(
                [future_live_event, current_scheduled_event]
            )

        filtered_ids = {event.get("id") for event in filtered}
        self.assertIn("future-live", filtered_ids)
        self.assertNotIn("current-scheduled", filtered_ids)
        self.assertEqual(stats.get("dropped_not_live_state"), 1)

    def test_collect_market_entries_filters_stale_live_quotes(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "live_state": {"status": "live", "updated_at": 99},
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.2, "last_updated": 90},
                                {"name": "Away Team", "price": 1.8, "last_updated": 90},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 1.8, "last_updated": 99},
                                {"name": "Away Team", "price": 2.2, "last_updated": 99},
                            ],
                        }
                    ],
                },
            ],
        }

        with patch("scanner.time.time", return_value=100.0), patch.object(
            scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"
        ):
            live_entries = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="live",
            )
            prematch_entries = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="prematch",
            )

        self.assertEqual(live_entries, [])
        self.assertTrue(prematch_entries)

    def test_collect_market_entries_filters_stale_prematch_quotes_only_when_timestamp_exists(self) -> None:
        game = {
            "sport_key": "icehockey_nhl",
            "sport_display": "NHL",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "live_state": {"status": "scheduled", "updated_at": 50},
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 2.1, "point": 6.5},
                                {"name": "Under", "price": 1.8, "point": 6.5},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.8, "point": 6.5, "last_updated": 10},
                                {"name": "Under", "price": 2.2, "point": 6.5, "last_updated": 10},
                            ],
                        }
                    ],
                },
            ],
        }

        with patch("scanner.time.time", return_value=100.0), patch.object(
            scanner, "PREMATCH_QUOTE_MAX_AGE_SECONDS_RAW", "30"
        ):
            entries = scanner._collect_market_entries(
                game,
                market_key="totals",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="prematch",
            )

        self.assertEqual(entries, [])

    def test_collect_market_entries_filters_stale_prematch_quotes_by_default_when_timestamp_exists(self) -> None:
        game = {
            "sport_key": "icehockey_nhl",
            "sport_display": "NHL",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "live_state": {"status": "scheduled"},
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.91, "point": 6.5, "last_updated": 10000},
                                {"name": "Under", "price": 1.91, "point": 6.5, "last_updated": 10000},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.8, "point": 6.5, "last_updated": 10},
                                {"name": "Under", "price": 2.2, "point": 6.5, "last_updated": 10},
                            ],
                        }
                    ],
                },
            ],
        }

        with patch("scanner.time.time", return_value=10000.0):
            entries = scanner._collect_market_entries(
                game,
                market_key="totals",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="prematch",
            )

        self.assertEqual(entries, [])

    def test_collect_market_entries_prematch_freshness_ignores_observed_at_refresh_of_stale_snapshot(self) -> None:
        game = {
            "sport_key": "icehockey_nhl",
            "sport_display": "NHL",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "live_state": {"status": "scheduled"},
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.9, "point": 6.5},
                                {"name": "Under", "price": 1.9, "point": 6.5},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.8, "point": 6.5, "last_updated": 10, "observed_at": 99},
                                {"name": "Under", "price": 2.2, "point": 6.5, "last_updated": 10, "observed_at": 99},
                            ],
                        }
                    ],
                },
            ],
        }

        with patch("scanner.time.time", return_value=100.0), patch.object(
            scanner, "PREMATCH_QUOTE_MAX_AGE_SECONDS_RAW", "30"
        ):
            entries = scanner._collect_market_entries(
                game,
                market_key="totals",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="prematch",
            )

        self.assertEqual(entries, [])

    def test_collect_plus_ev_filters_stale_live_quotes(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "live_state": {"status": "live", "updated_at": 99},
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 1.9, "last_updated": 99},
                                {"name": "Away Team", "price": 1.9, "last_updated": 99},
                            ],
                        }
                    ],
                },
                {
                    "key": "DraftKings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.2, "last_updated": 90},
                                {"name": "Away Team", "price": 1.7, "last_updated": 90},
                            ],
                        }
                    ],
                },
            ],
        }

        with patch("scanner.time.time", return_value=100.0), patch.object(
            scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"
        ):
            live_entries = scanner._collect_plus_ev_opportunities(
                game,
                markets=["h2h"],
                sharp_priority=scanner._sharp_priority("pinnacle"),
                commission_rate=0.0,
                min_edge_percent=0.0,
                bankroll=1000.0,
                kelly_fraction=0.25,
                scan_mode="live",
            )
            prematch_entries = scanner._collect_plus_ev_opportunities(
                game,
                markets=["h2h"],
                sharp_priority=scanner._sharp_priority("pinnacle"),
                commission_rate=0.0,
                min_edge_percent=0.0,
                bankroll=1000.0,
                kelly_fraction=0.25,
                scan_mode="prematch",
            )

        self.assertEqual(live_entries, [])
        self.assertTrue(prematch_entries)

    def test_collect_market_entries_filters_mismatched_live_state_context(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "live_state": {"status": "live"},
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "live_state": {"status": "live", "period": "Q2", "score": "55-50", "clock": "05:40"},
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.2},
                                {"name": "Away Team", "price": 1.8},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "live_state": {"status": "live", "period": "Q3", "score": "55-50", "clock": "11:58"},
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 1.8},
                                {"name": "Away Team", "price": 2.2},
                            ],
                        }
                    ],
                },
            ],
        }

        live_entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="live",
        )
        prematch_entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="prematch",
        )

        self.assertEqual(live_entries, [])
        self.assertTrue(prematch_entries)

    def test_collect_middle_filters_mismatched_live_state_context(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "live_state": {"status": "live"},
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "live_state": {"status": "live", "period": "Q2", "score": "55-50"},
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 210.5, "price": 2.0},
                                {"name": "Under", "point": 211.5, "price": 2.0},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "live_state": {"status": "live", "period": "Q3", "score": "55-50"},
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 210.5, "price": 2.0},
                                {"name": "Under", "point": 211.5, "price": 2.0},
                            ],
                        }
                    ],
                },
            ],
        }

        live_entries = scanner._collect_middle_opportunities(
            game,
            market_key="totals",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="live",
        )
        prematch_entries = scanner._collect_middle_opportunities(
            game,
            market_key="totals",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="prematch",
        )

        self.assertEqual(live_entries, [])
        self.assertTrue(prematch_entries)

    def test_collect_plus_ev_filters_mismatched_live_state_context(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "live_state": {"status": "live"},
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "live_state": {"status": "live", "period": "Q2", "score": "55-50", "clock": "05:40"},
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 1.9},
                                {"name": "Away Team", "price": 1.9},
                            ],
                        }
                    ],
                },
                {
                    "key": "DraftKings",
                    "title": "DraftKings",
                    "live_state": {"status": "live", "period": "Q3", "score": "55-50", "clock": "11:58"},
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.2},
                                {"name": "Away Team", "price": 1.7},
                            ],
                        }
                    ],
                },
            ],
        }

        live_entries = scanner._collect_plus_ev_opportunities(
            game,
            markets=["h2h"],
            sharp_priority=scanner._sharp_priority("pinnacle"),
            commission_rate=0.0,
            min_edge_percent=0.0,
            bankroll=1000.0,
            kelly_fraction=0.25,
            scan_mode="live",
        )
        prematch_entries = scanner._collect_plus_ev_opportunities(
            game,
            markets=["h2h"],
            sharp_priority=scanner._sharp_priority("pinnacle"),
            commission_rate=0.0,
            min_edge_percent=0.0,
            bankroll=1000.0,
            kelly_fraction=0.25,
            scan_mode="prematch",
        )

        self.assertEqual(live_entries, [])
        self.assertTrue(prematch_entries)

    def test_collect_middle_opportunities_applies_commission_for_mixed_case_exchange_key(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "BetDEX",
                    "title": "BetDEX",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 210.5, "price": 2.0},
                                {"name": "Under", "point": 211.5, "price": 2.0},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 210.5, "price": 2.0},
                                {"name": "Under", "point": 211.5, "price": 2.0},
                            ],
                        }
                    ],
                },
            ],
        }

        entries = scanner._collect_middle_opportunities(
            game,
            market_key="totals",
            stake_total=100.0,
            commission_rate=0.05,
        )
        self.assertTrue(entries)
        exchange_legs = []
        for entry in entries:
            for side_key in ("side_a", "side_b"):
                side = entry.get(side_key) or {}
                if (side.get("bookmaker") or "").strip().lower() == "betdex":
                    exchange_legs.append(side)
        self.assertTrue(exchange_legs)
        self.assertTrue(
            any(
                float(side.get("effective_price") or 0.0)
                < float(side.get("price") or 0.0)
                for side in exchange_legs
            )
        )

    def test_collect_plus_ev_accepts_mixed_case_soft_book_key(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 1.9},
                                {"name": "Away Team", "price": 1.9},
                            ],
                        }
                    ],
                },
                {
                    "key": "DraftKings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.2},
                                {"name": "Away Team", "price": 1.7},
                            ],
                        }
                    ],
                },
            ],
        }

        entries = scanner._collect_plus_ev_opportunities(
            game,
            markets=["h2h"],
            sharp_priority=scanner._sharp_priority("pinnacle"),
            commission_rate=0.0,
            min_edge_percent=0.0,
            bankroll=1000.0,
            kelly_fraction=0.25,
        )
        self.assertTrue(entries)
        self.assertGreater(entries[0].get("edge_percent", 0.0), 0.0)

    def test_merge_events_handles_home_away_flipped_provider_feeds(self) -> None:
        base_events = [
            {
                "id": "base-1",
                "sport_key": "baseball_mlb",
                "home_team": "Los Angeles Dodgers",
                "away_team": "Cleveland Guardians",
                "commence_time": "2026-03-03T20:05:00Z",
                "bookmakers": [
                    {
                        "key": "sx_bet",
                        "title": "SX Bet",
                        "event_id": "L17406189",
                        "markets": [{"key": "h2h", "outcomes": []}],
                    }
                ],
            }
        ]
        extra_events = [
            {
                "id": "extra-1",
                "sport_key": "baseball_mlb",
                "home_team": "Cleveland Guardians",
                "away_team": "Los Angeles Dodgers",
                "commence_time": "2026-03-03T20:05:00Z",
                "bookmakers": [
                    {
                        "key": "polymarket",
                        "title": "Polymarket",
                        "event_id": "242604",
                        "markets": [{"key": "h2h", "outcomes": []}],
                    }
                ],
            }
        ]

        merged = scanner._merge_events(base_events, extra_events)
        self.assertEqual(len(merged), 1)
        books = merged[0].get("bookmakers") or []
        keys = {str(book.get("key") or "").strip().lower() for book in books if isinstance(book, dict)}
        self.assertIn("sx_bet", keys)
        self.assertIn("polymarket", keys)

    def test_merge_events_with_stats_reports_reverse_team_match(self) -> None:
        base_events = [
            {
                "id": "base-1",
                "sport_key": "baseball_mlb",
                "home_team": "Los Angeles Dodgers",
                "away_team": "Cleveland Guardians",
                "commence_time": "2026-03-03T20:05:00Z",
                "bookmakers": [
                    {
                        "key": "sx_bet",
                        "title": "SX Bet",
                        "markets": [{"key": "h2h", "outcomes": []}],
                    }
                ],
            }
        ]
        extra_events = [
            {
                "id": "extra-1",
                "sport_key": "baseball_mlb",
                "home_team": "Cleveland Guardians",
                "away_team": "Los Angeles Dodgers",
                "commence_time": "2026-03-03T20:05:00Z",
                "bookmakers": [
                    {
                        "key": "polymarket",
                        "title": "Polymarket",
                        "markets": [{"key": "h2h", "outcomes": []}],
                    }
                ],
            }
        ]

        stats = {}
        merged = scanner._merge_events_with_stats(base_events, extra_events, stats=stats)
        self.assertEqual(len(merged), 1)
        self.assertEqual(stats.get("incoming_events"), 1)
        self.assertEqual(stats.get("matched_existing"), 1)
        self.assertEqual(stats.get("matched_reverse_team"), 1)
        self.assertEqual(stats.get("appended_new"), 0)

    def test_merge_events_with_stats_reorients_reversed_spread_outcomes_to_base_event(self) -> None:
        base_events = [
            {
                "id": "base-1",
                "sport_key": "soccer_usa_mls",
                "home_team": "Nashville SC",
                "away_team": "Charlotte FC",
                "commence_time": "2026-04-12T07:30:00Z",
                "bookmakers": [
                    {
                        "key": "sx_bet",
                        "title": "SX Bet",
                        "markets": [
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Nashville SC", "point": 2.5, "price": 2.68},
                                    {"name": "Charlotte FC", "point": -2.5, "price": 1.5},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        extra_events = [
            {
                "id": "poly-1",
                "sport_key": "soccer_usa_mls",
                "home_team": "Charlotte FC",
                "away_team": "Nashville SC",
                "commence_time": "2026-04-12T07:30:00Z",
                "bookmakers": [
                    {
                        "key": "polymarket",
                        "title": "Polymarket",
                        "markets": [
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Charlotte FC", "point": -2.5, "price": 21.978022},
                                    {"name": "Nashville SC", "point": 2.5, "price": 1.047669},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        merged = scanner._merge_events_with_stats(base_events, extra_events, stats={})
        self.assertEqual(len(merged), 1)

        entries = scanner._collect_market_entries(
            merged[0],
            market_key="spreads",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="prematch",
        )

        self.assertEqual(len(entries), 1)
        best_odds = entries[0].get("best_odds") or []
        self.assertEqual(len(best_odds), 2)
        self.assertEqual(best_odds[0].get("outcome"), "Charlotte FC")
        self.assertEqual(best_odds[0].get("point"), -2.5)
        self.assertEqual(best_odds[1].get("outcome"), "Nashville SC")
        self.assertEqual(best_odds[1].get("point"), 2.5)

    def test_merge_events_with_stats_merges_nba_team_aliases_with_reversed_orientation(self) -> None:
        base_events = [
            {
                "id": "sx-1",
                "sport_key": "basketball_nba",
                "home_team": "Los Angeles Clippers",
                "away_team": "Golden State Warriors",
                "commence_time": "2026-04-13T00:30:00Z",
                "bookmakers": [
                    {
                        "key": "sx_bet",
                        "title": "SX Bet",
                        "markets": [{"key": "h2h", "outcomes": []}],
                    }
                ],
            }
        ]
        extra_events = [
            {
                "id": "poly-1",
                "sport_key": "basketball_nba",
                "home_team": "Warriors",
                "away_team": "Clippers",
                "commence_time": "2026-04-13T00:30:00Z",
                "bookmakers": [
                    {
                        "key": "polymarket",
                        "title": "Polymarket",
                        "markets": [{"key": "h2h", "outcomes": []}],
                    }
                ],
            }
        ]

        stats = {}
        merged = scanner._merge_events_with_stats(base_events, extra_events, stats=stats)

        self.assertEqual(len(merged), 1)
        self.assertEqual(stats.get("incoming_events"), 1)
        self.assertEqual(stats.get("matched_existing"), 1)
        self.assertEqual(stats.get("appended_new"), 0)
        books = merged[0].get("bookmakers") or []
        keys = {str(book.get("key") or "").strip().lower() for book in books if isinstance(book, dict)}
        self.assertIn("sx_bet", keys)
        self.assertIn("polymarket", keys)

    def test_cross_provider_match_report_flags_same_pair_time_mismatch(self) -> None:
        snapshots = {
            "sx_bet": {
                "events": [
                    {
                        "id": "sx-1",
                        "sport_key": "soccer_usa_mls",
                        "home_team": "Columbus Crew",
                        "away_team": "Orlando City",
                        "commence_time": "2026-03-10T23:30:00Z",
                        "bookmakers": [{"key": "sx_bet", "markets": [{"key": "h2h"}]}],
                    }
                ]
            },
            "polymarket": {
                "events": [
                    {
                        "id": "poly-1",
                        "sport_key": "soccer_usa_mls",
                        "home_team": "Columbus Crew",
                        "away_team": "Orlando City",
                        "commence_time": "2026-03-11T03:45:00Z",
                        "bookmakers": [{"key": "polymarket", "markets": [{"key": "h2h"}]}],
                    }
                ]
            },
        }

        report = scanner._build_cross_provider_match_report("2026-03-10T20:00:00Z", snapshots)

        self.assertIsInstance(report, dict)
        singles = report.get("single_provider_samples") or []
        sx_sample = next(
            (
                item
                for item in singles
                if isinstance(item, dict)
                and item.get("provider") == "sx_bet"
                and item.get("event_id") == "sx-1"
            ),
            None,
        )
        self.assertIsNotNone(sx_sample)
        self.assertEqual(sx_sample.get("reason_code"), "same_pair_time_mismatch")
        closest = sx_sample.get("closest_candidate") or {}
        self.assertEqual(closest.get("provider"), "polymarket")
        self.assertEqual(closest.get("event_id"), "poly-1")
        self.assertGreater(closest.get("time_delta_minutes", 0), 180)
        summary = report.get("summary") or {}
        self.assertIn("event_match_tolerance_minutes", summary)
        self.assertIn("event_match_fuzzy_threshold", summary)

    def test_build_scan_diagnostics_requires_positive_arb_for_arbitrage_found(self) -> None:
        diagnostics = scanner._build_scan_diagnostics(
            provider_summaries={
                "sx_bet": {
                    "name": "SX Bet",
                    "enabled": True,
                    "events_merged": 2,
                    "sports": [
                        {
                            "events_returned": 2,
                            "merge_stats": {
                                "matched_existing": 1,
                                "matched_identity": 1,
                                "matched_team": 0,
                                "matched_reverse_team": 0,
                                "matched_fuzzy": 0,
                                "appended_new": 1,
                            },
                        }
                    ],
                },
                "betdex": {
                    "name": "BetDEX",
                    "enabled": True,
                    "events_merged": 2,
                    "sports": [
                        {
                            "events_returned": 2,
                            "merge_stats": {
                                "matched_existing": 1,
                                "matched_identity": 1,
                                "matched_team": 0,
                                "matched_reverse_team": 0,
                                "matched_fuzzy": 0,
                                "appended_new": 1,
                            },
                        }
                    ],
                },
            },
            cross_provider_report={
                "summary": {
                    "total_raw_records": 4,
                    "total_match_clusters": 2,
                    "overlap_clusters": 1,
                }
            },
            events_scanned=2,
            arbitrage_count=3,
            positive_arbitrage_count=0,
            middle_count=0,
            positive_middle_count=0,
            plus_ev_count=0,
            sport_errors=[],
            stale_event_filters=[],
        )

        self.assertEqual(diagnostics.get("arbitrage_count"), 3)
        self.assertEqual(diagnostics.get("positive_arbitrage_count"), 0)
        self.assertEqual(diagnostics.get("reason_code"), "matched_but_no_arbitrage")

    def test_build_scan_diagnostics_surfaces_positive_middle_without_positive_arb(self) -> None:
        diagnostics = scanner._build_scan_diagnostics(
            provider_summaries={
                "sx_bet": {
                    "name": "SX Bet",
                    "enabled": True,
                    "events_merged": 2,
                    "sports": [
                        {
                            "events_returned": 2,
                            "merge_stats": {
                                "matched_existing": 1,
                                "matched_identity": 1,
                                "matched_team": 0,
                                "matched_reverse_team": 0,
                                "matched_fuzzy": 0,
                                "appended_new": 1,
                            },
                        }
                    ],
                },
                "betdex": {
                    "name": "BetDEX",
                    "enabled": True,
                    "events_merged": 2,
                    "sports": [
                        {
                            "events_returned": 2,
                            "merge_stats": {
                                "matched_existing": 1,
                                "matched_identity": 1,
                                "matched_team": 0,
                                "matched_reverse_team": 0,
                                "matched_fuzzy": 0,
                                "appended_new": 1,
                            },
                        }
                    ],
                },
            },
            cross_provider_report={
                "summary": {
                    "total_raw_records": 4,
                    "total_match_clusters": 2,
                    "overlap_clusters": 1,
                }
            },
            events_scanned=2,
            arbitrage_count=4,
            positive_arbitrage_count=0,
            middle_count=5,
            positive_middle_count=2,
            plus_ev_count=0,
            sport_errors=[],
            stale_event_filters=[],
        )

        self.assertEqual(diagnostics.get("positive_arbitrage_count"), 0)
        self.assertEqual(diagnostics.get("positive_middle_count"), 2)
        self.assertEqual(diagnostics.get("reason_code"), "positive_middle_found")

    def test_merge_events_merges_markets_for_same_bookmaker(self) -> None:
        base_events = [
            {
                "id": "base-1",
                "sport_key": "basketball_nba",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "commence_time": "2026-03-10T00:00:00Z",
                "bookmakers": [
                    {
                        "key": "sx_bet",
                        "title": "SX Bet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Los Angeles Lakers", "price": 2.05},
                                    {"name": "Boston Celtics", "price": 1.88},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        extra_events = [
            {
                "id": "extra-1",
                "sport_key": "basketball_nba",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "commence_time": "2026-03-10T00:00:00Z",
                "bookmakers": [
                    {
                        "key": "sx_bet",
                        "title": "SX Bet",
                        "markets": [
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Los Angeles Lakers", "point": -1.5, "price": 1.95},
                                    {"name": "Boston Celtics", "point": 1.5, "price": 1.95},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        merged = scanner._merge_events(base_events, extra_events)
        self.assertEqual(len(merged), 1)
        books = merged[0].get("bookmakers") or []
        self.assertEqual(len(books), 1)
        market_keys = {
            str(market.get("key") or "").strip().lower()
            for market in (books[0].get("markets") or [])
            if isinstance(market, dict)
        }
        self.assertIn("h2h", market_keys)
        self.assertIn("spreads", market_keys)

    def test_merge_events_merges_soccer_more_markets_event_into_main_event(self) -> None:
        merged = scanner._merge_events(
            [],
            [
                {
                    "id": "poly-main",
                    "sport_key": "soccer_usa_mls",
                    "home_team": "Columbus Crew",
                    "away_team": "Orlando City",
                    "commence_time": "2026-03-10T23:30:00Z",
                    "bookmakers": [
                        {
                            "key": "polymarket",
                            "title": "Polymarket",
                            "event_id": "poly-main",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Columbus Crew", "price": 2.1},
                                        {"name": "Orlando City", "price": 1.82},
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "id": "poly-more-markets",
                    "sport_key": "soccer_usa_mls",
                    "home_team": "Columbus Crew SC",
                    "away_team": "Orlando City SC",
                    "commence_time": "2026-03-10T23:30:00Z",
                    "bookmakers": [
                        {
                            "key": "polymarket",
                            "title": "Polymarket",
                            "event_id": "poly-more-markets",
                            "markets": [
                                {
                                    "key": "both_teams_to_score",
                                    "outcomes": [
                                        {"name": "Yes", "price": 1.95},
                                        {"name": "No", "price": 1.9},
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ],
        )

        self.assertEqual(len(merged), 1)
        books = merged[0].get("bookmakers") or []
        self.assertEqual(len(books), 1)
        self.assertEqual(books[0].get("key"), "polymarket")
        market_keys = {
            str(market.get("key") or "").strip().lower()
            for market in (books[0].get("markets") or [])
            if isinstance(market, dict)
        }
        self.assertEqual(market_keys, {"h2h", "both_teams_to_score"})

    def test_run_scan_all_sports_expands_provider_target_sports(self) -> None:
        sports_payload = [
            {
                "key": "americanfootball_nfl",
                "title": "NFL",
                "active": True,
                "has_outrights": False,
            },
            {
                "key": "tennis_atp",
                "title": "ATP",
                "active": True,
                "has_outrights": False,
            },
        ]
        captured_targets = []

        def _fake_scan_single_sport(**kwargs):
            captured_targets.append(set(kwargs.get("provider_target_sport_keys") or []))
            sport = kwargs.get("sport") or {}
            sport_key = sport.get("key") or ""
            sport_title = sport.get("title") or sport_key
            return {
                "skipped": False,
                "sport_key": sport_key,
                "sport_timing": {
                    "sport_key": sport_key,
                    "sport": sport_title,
                    "api_fetch_ms": 0.0,
                    "provider_fetch_ms": 0.0,
                    "analysis_ms": 0.0,
                    "events_scanned": 0,
                    "providers": [],
                    "total_ms": 0.0,
                },
                "timing_steps": [],
                "api_market_skips": [],
                "sport_errors": [],
                "provider_updates": {},
                "provider_snapshot_updates": {},
                "events_scanned": 0,
                "total_profit": 0.0,
                "arb_opportunities": [],
                "middle_opportunities": [],
                "plus_ev_opportunities": [],
                "stale_event_filters": [],
                "successful": 1,
            }

        with (
            patch.object(scanner, "fetch_sports", return_value=sports_payload),
            patch.object(scanner, "_scan_single_sport", side_effect=_fake_scan_single_sport),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
        ):
            result = scanner.run_scan(
                api_key="dummy",
                sports=["americanfootball_nfl"],
                all_sports=True,
                include_providers=["sx_bet"],
            )

        self.assertTrue(result.get("success"))
        self.assertTrue(captured_targets)
        self.assertIn("tennis_atp", captured_targets[0])

    def test_run_scan_include_providers_empty_keeps_api_fetch_enabled(self) -> None:
        sports_payload = [
            {
                "key": "americanfootball_nfl",
                "title": "NFL",
                "active": True,
                "has_outrights": False,
            }
        ]
        captured_should_fetch_api = []

        def _fake_scan_single_sport(**kwargs):
            captured_should_fetch_api.append(bool(kwargs.get("should_fetch_api")))
            sport = kwargs.get("sport") or {}
            sport_key = sport.get("key") or ""
            sport_title = sport.get("title") or sport_key
            return {
                "skipped": False,
                "sport_key": sport_key,
                "sport_timing": {
                    "sport_key": sport_key,
                    "sport": sport_title,
                    "api_fetch_ms": 0.0,
                    "provider_fetch_ms": 0.0,
                    "analysis_ms": 0.0,
                    "events_scanned": 0,
                    "providers": [],
                    "total_ms": 0.0,
                },
                "timing_steps": [],
                "api_market_skips": [],
                "sport_errors": [],
                "provider_updates": {},
                "provider_snapshot_updates": {},
                "events_scanned": 0,
                "total_profit": 0.0,
                "arb_opportunities": [],
                "middle_opportunities": [],
                "plus_ev_opportunities": [],
                "stale_event_filters": [],
                "successful": 1,
            }

        with (
            patch.object(scanner, "fetch_sports", return_value=sports_payload),
            patch.object(scanner, "_scan_single_sport", side_effect=_fake_scan_single_sport),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
        ):
            result = scanner.run_scan(
                api_key="dummy",
                sports=["americanfootball_nfl"],
                include_providers=[],
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(captured_should_fetch_api, [True])

    def test_run_scan_include_providers_non_empty_keeps_api_fetch_enabled(self) -> None:
        sports_payload = [
            {
                "key": "americanfootball_nfl",
                "title": "NFL",
                "active": True,
                "has_outrights": False,
            }
        ]
        captured_should_fetch_api = []

        def _fake_scan_single_sport(**kwargs):
            captured_should_fetch_api.append(bool(kwargs.get("should_fetch_api")))
            sport = kwargs.get("sport") or {}
            sport_key = sport.get("key") or ""
            sport_title = sport.get("title") or sport_key
            return {
                "skipped": False,
                "sport_key": sport_key,
                "sport_timing": {
                    "sport_key": sport_key,
                    "sport": sport_title,
                    "api_fetch_ms": 0.0,
                    "provider_fetch_ms": 0.0,
                    "analysis_ms": 0.0,
                    "events_scanned": 0,
                    "providers": [],
                    "total_ms": 0.0,
                },
                "timing_steps": [],
                "api_market_skips": [],
                "sport_errors": [],
                "provider_updates": {},
                "provider_snapshot_updates": {},
                "events_scanned": 0,
                "total_profit": 0.0,
                "arb_opportunities": [],
                "middle_opportunities": [],
                "plus_ev_opportunities": [],
                "stale_event_filters": [],
                "successful": 1,
            }

        with (
            patch.object(scanner, "fetch_sports", return_value=sports_payload),
            patch.object(scanner, "_scan_single_sport", side_effect=_fake_scan_single_sport),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
        ):
            result = scanner.run_scan(
                api_key="dummy",
                sports=["americanfootball_nfl"],
                include_providers=["sx_bet"],
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(captured_should_fetch_api, [True])

    def test_run_scan_include_providers_without_api_keys_runs_provider_only(self) -> None:
        captured_should_fetch_api = []

        def _fake_scan_single_sport(**kwargs):
            captured_should_fetch_api.append(bool(kwargs.get("should_fetch_api")))
            sport = kwargs.get("sport") or {}
            sport_key = sport.get("key") or ""
            sport_title = sport.get("title") or sport_key
            return {
                "skipped": False,
                "sport_key": sport_key,
                "sport_timing": {
                    "sport_key": sport_key,
                    "sport": sport_title,
                    "api_fetch_ms": 0.0,
                    "provider_fetch_ms": 0.0,
                    "analysis_ms": 0.0,
                    "events_scanned": 0,
                    "providers": [],
                    "total_ms": 0.0,
                },
                "timing_steps": [],
                "api_market_skips": [],
                "sport_errors": [],
                "provider_updates": {},
                "provider_snapshot_updates": {},
                "events_scanned": 0,
                "total_profit": 0.0,
                "arb_opportunities": [],
                "middle_opportunities": [],
                "plus_ev_opportunities": [],
                "stale_event_filters": [],
                "successful": 1,
            }

        with (
            patch.object(scanner, "_scan_single_sport", side_effect=_fake_scan_single_sport),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
        ):
            result = scanner.run_scan(
                api_key="",
                sports=["americanfootball_nfl"],
                include_providers=["sx_bet"],
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(captured_should_fetch_api, [False])

    def test_run_scan_uses_persistent_async_runtime_for_sync_calls(self) -> None:
        observed = {}

        class _FakeRuntime:
            def submit(self, coroutine):
                observed["is_coroutine"] = inspect.iscoroutine(coroutine)
                if inspect.iscoroutine(coroutine):
                    coroutine.close()
                return {"success": True, "source": "runtime"}

        with patch.object(scanner, "_get_scan_async_runtime", return_value=_FakeRuntime()):
            result = scanner.run_scan(api_key="dummy", sports=["americanfootball_nfl"])

        self.assertEqual(result, {"success": True, "source": "runtime"})
        self.assertTrue(observed.get("is_coroutine"))

    def test_run_scan_empty_results_include_plus_ev_shape(self) -> None:
        with patch.object(scanner, "fetch_sports", return_value=[]):
            result = scanner.run_scan(
                api_key="dummy",
                sports=["americanfootball_nfl"],
            )

        self.assertTrue(result.get("success"))
        plus_ev = result.get("plus_ev")
        self.assertIsInstance(plus_ev, dict)
        self.assertEqual(plus_ev.get("opportunities_count"), 0)
        self.assertEqual(plus_ev.get("opportunities"), [])

    def test_deduplicate_middles_keeps_distinct_commence_time(self) -> None:
        opportunities = [
            {
                "sport": "basketball_nba",
                "event": "Away vs Home",
                "commence_time": "2026-03-04T01:00:00Z",
                "market": "spreads",
                "gap": {"middle_integers": [3]},
                "ev_percent": 3.0,
            },
            {
                "sport": "basketball_nba",
                "event": "Away vs Home",
                "commence_time": "2026-03-05T01:00:00Z",
                "market": "spreads",
                "gap": {"middle_integers": [3]},
                "ev_percent": 2.0,
            },
        ]
        deduped = scanner._deduplicate_middles(opportunities)
        self.assertEqual(len(deduped), 2)

    def test_deduplicate_plus_ev_keeps_distinct_commence_time(self) -> None:
        opportunities = [
            {
                "sport": "basketball_nba",
                "event": "Away vs Home",
                "commence_time": "2026-03-04T01:00:00Z",
                "market": "h2h",
                "edge_percent": 4.2,
                "bet": {"outcome": "Home", "point": None},
            },
            {
                "sport": "basketball_nba",
                "event": "Away vs Home",
                "commence_time": "2026-03-05T01:00:00Z",
                "market": "h2h",
                "edge_percent": 3.1,
                "bet": {"outcome": "Home", "point": None},
            },
        ]
        deduped = scanner._deduplicate_plus_ev(opportunities)
        self.assertEqual(len(deduped), 2)

    def test_run_scan_async_emits_scan_started_and_scan_completed_progress_events(self) -> None:
        progress_events = []

        with patch.object(scanner, "fetch_sports", return_value=[]):
            result = asyncio.run(
                scanner.run_scan_async(
                    api_key="dummy",
                    sports=["basketball_nba"],
                    progress_callback=progress_events.append,
                )
            )

        self.assertTrue(result.get("success"))
        self.assertTrue(progress_events)
        self.assertEqual(progress_events[0].get("type"), "scan_started")
        self.assertEqual(progress_events[-1].get("type"), "scan_completed")

    def test_scan_single_sport_emits_provider_completed_progress_event(self) -> None:
        progress_events = []

        async def _provider_fetcher(sport_key: str, markets: list[str], regions: list[str], bookmakers=None):
            return []

        with (
            patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _provider_fetcher}),
            patch.dict(scanner.PROVIDER_TITLES, {"sx_bet": "SX Bet"}),
        ):
            result = asyncio.run(
                scanner._scan_single_sport(
                    sport={"key": "basketball_nba", "title": "NBA"},
                    scan_mode="prematch",
                    all_markets=False,
                    should_fetch_api=False,
                    api_pool=scanner.ApiKeyPool([]),
                    normalized_regions=["us"],
                    api_bookmakers=[],
                    provider_target_sport_keys=["basketball_nba"],
                    enabled_provider_keys=["sx_bet"],
                    normalized_bookmakers=["sx_bet"],
                    stake_amount=100.0,
                    commission_rate=0.0,
                    sharp_priority=scanner._sharp_priority("pinnacle"),
                    min_edge_percent=0.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                    progress_callback=progress_events.append,
                )
            )

        self.assertFalse(result.get("skipped"))
        provider_events = [item for item in progress_events if item.get("type") == "provider_completed"]
        self.assertEqual(len(provider_events), 1)
        self.assertEqual(provider_events[0].get("provider_key"), "sx_bet")
        self.assertEqual(provider_events[0].get("sport_key"), "basketball_nba")
        self.assertIsInstance(provider_events[0].get("result"), dict)

    def test_scan_single_sport_emits_provider_completed_once_per_provider(self) -> None:
        progress_events = []

        async def _provider_fetcher(sport_key: str, markets: list[str], regions: list[str], bookmakers=None):
            return [
                {
                    "id": "provider-event-1",
                    "sport_key": sport_key,
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "commence_time": "2099-03-13T00:00:00Z",
                    "bookmakers": [
                        {
                            "key": "sx_bet",
                            "title": "SX Bet",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Home Team", "price": 2.1},
                                        {"name": "Away Team", "price": 1.8},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        with (
            patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _provider_fetcher}),
            patch.dict(scanner.PROVIDER_TITLES, {"sx_bet": "SX Bet"}),
        ):
            asyncio.run(
                scanner._scan_single_sport(
                    sport={"key": "basketball_nba", "title": "NBA"},
                    scan_mode="prematch",
                    all_markets=False,
                    should_fetch_api=False,
                    api_pool=scanner.ApiKeyPool([]),
                    normalized_regions=["us"],
                    api_bookmakers=[],
                    provider_target_sport_keys=["basketball_nba"],
                    enabled_provider_keys=["sx_bet"],
                    normalized_bookmakers=["sx_bet"],
                    stake_amount=100.0,
                    commission_rate=0.0,
                    sharp_priority=scanner._sharp_priority("pinnacle"),
                    min_edge_percent=0.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                    progress_callback=progress_events.append,
                )
            )

        provider_events = [item for item in progress_events if item.get("type") == "provider_completed"]
        self.assertEqual(len(provider_events), 1)
        self.assertEqual(provider_events[0].get("provider_key"), "sx_bet")

    def test_scan_single_sport_provider_progress_surfaces_cross_book_arb_before_sport_completion(self) -> None:
        progress_events = []

        async def _provider_a(sport_key: str, markets: list[str], regions: list[str], bookmakers=None):
            return [
                {
                    "id": "shared-event",
                    "sport_key": sport_key,
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "commence_time": "2099-03-13T00:00:00Z",
                    "bookmakers": [
                        {
                            "key": "book_a",
                            "title": "Book A",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Home Team", "price": 2.3},
                                        {"name": "Away Team", "price": 1.8},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        async def _provider_b(sport_key: str, markets: list[str], regions: list[str], bookmakers=None):
            return [
                {
                    "id": "shared-event",
                    "sport_key": sport_key,
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "commence_time": "2099-03-13T00:00:00Z",
                    "bookmakers": [
                        {
                            "key": "book_b",
                            "title": "Book B",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Home Team", "price": 1.8},
                                        {"name": "Away Team", "price": 2.3},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        with (
            patch.dict(
                scanner.PROVIDER_FETCHERS,
                {
                    "book_a": _provider_a,
                    "book_b": _provider_b,
                },
            ),
            patch.dict(
                scanner.PROVIDER_TITLES,
                {
                    "book_a": "Book A",
                    "book_b": "Book B",
                },
            ),
        ):
            asyncio.run(
                scanner._scan_single_sport(
                    sport={"key": "basketball_nba", "title": "NBA"},
                    scan_mode="prematch",
                    all_markets=False,
                    should_fetch_api=False,
                    api_pool=scanner.ApiKeyPool([]),
                    normalized_regions=["us"],
                    api_bookmakers=[],
                    provider_target_sport_keys=["basketball_nba"],
                    enabled_provider_keys=["book_a", "book_b"],
                    normalized_bookmakers=["book_a", "book_b"],
                    stake_amount=100.0,
                    commission_rate=0.0,
                    sharp_priority=scanner._sharp_priority("pinnacle"),
                    min_edge_percent=0.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                    progress_callback=progress_events.append,
                )
            )

        provider_events = [item for item in progress_events if item.get("type") == "provider_completed"]
        self.assertEqual(len(provider_events), 2)
        self.assertEqual(provider_events[0].get("provider_key"), "book_a")
        self.assertEqual(provider_events[1].get("provider_key"), "book_b")
        second_result = provider_events[1].get("result") or {}
        second_arb = second_result.get("arb_opportunities") or []
        self.assertTrue(second_arb)
        self.assertGreater(second_arb[0].get("roi_percent", 0.0), 0.0)

    def test_scan_single_sport_provider_progress_is_emitted_as_each_provider_finishes(self) -> None:
        progress_events = []

        async def _slow_provider(sport_key: str, markets: list[str], regions: list[str], bookmakers=None):
            await asyncio.sleep(0.2)
            return [
                {
                    "id": "slow-event",
                    "sport_key": sport_key,
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "commence_time": "2099-03-13T00:00:00Z",
                    "bookmakers": [
                        {
                            "key": "slow_book",
                            "title": "Slow Book",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Home Team", "price": 2.02},
                                        {"name": "Away Team", "price": 1.9},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        async def _fast_provider(sport_key: str, markets: list[str], regions: list[str], bookmakers=None):
            return [
                {
                    "id": "fast-event",
                    "sport_key": sport_key,
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "commence_time": "2099-03-13T00:00:00Z",
                    "bookmakers": [
                        {
                            "key": "fast_book",
                            "title": "Fast Book",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Home Team", "price": 1.9},
                                        {"name": "Away Team", "price": 2.02},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        with (
            patch.dict(
                scanner.PROVIDER_FETCHERS,
                {
                    "slow_book": _slow_provider,
                    "fast_book": _fast_provider,
                },
            ),
            patch.dict(
                scanner.PROVIDER_TITLES,
                {
                    "slow_book": "Slow Book",
                    "fast_book": "Fast Book",
                },
            ),
            patch.object(scanner, "_provider_fetch_max_workers", return_value=2),
        ):
            asyncio.run(
                scanner._scan_single_sport(
                    sport={"key": "basketball_nba", "title": "NBA"},
                    scan_mode="prematch",
                    all_markets=False,
                    should_fetch_api=False,
                    api_pool=scanner.ApiKeyPool([]),
                    normalized_regions=["us"],
                    api_bookmakers=[],
                    provider_target_sport_keys=["basketball_nba"],
                    enabled_provider_keys=["slow_book", "fast_book"],
                    normalized_bookmakers=["slow_book", "fast_book"],
                    stake_amount=100.0,
                    commission_rate=0.0,
                    sharp_priority=scanner._sharp_priority("pinnacle"),
                    min_edge_percent=0.0,
                    bankroll=1000.0,
                    kelly_fraction=0.25,
                    progress_callback=progress_events.append,
                )
            )

        provider_events = [item for item in progress_events if item.get("type") == "provider_completed"]
        self.assertEqual(len(provider_events), 2)
        self.assertEqual(provider_events[0].get("provider_key"), "fast_book")
        self.assertEqual(provider_events[1].get("provider_key"), "slow_book")

    def test_run_scan_async_emits_sport_completed_progress_event(self) -> None:
        progress_events = []
        sports_payload = [{"key": "basketball_nba", "title": "NBA", "active": True, "has_outrights": False}]

        def _fake_scan_single_sport(**kwargs):
            return {
                "skipped": False,
                "sport_key": "basketball_nba",
                "sport_timing": {"sport_key": "basketball_nba", "sport": "NBA", "total_ms": 0.0},
                "timing_steps": [],
                "api_market_skips": [],
                "sport_errors": [],
                "provider_updates": {},
                "provider_snapshot_updates": {},
                "events_scanned": 1,
                "total_profit": 0.0,
                "arb_opportunities": [{"event": "A vs B", "roi_percent": 1.5, "stakes": {"guaranteed_profit": 1.0}}],
                "middle_opportunities": [],
                "plus_ev_opportunities": [],
                "stale_event_filters": [],
                "successful": 1,
            }

        with (
            patch.object(scanner, "fetch_sports", return_value=sports_payload),
            patch.object(scanner, "_scan_single_sport", side_effect=_fake_scan_single_sport),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
        ):
            result = asyncio.run(
                scanner.run_scan_async(
                    api_key="dummy",
                    sports=["basketball_nba"],
                    progress_callback=progress_events.append,
                )
            )

        self.assertTrue(result.get("success"))
        sport_events = [item for item in progress_events if item.get("type") == "sport_completed"]
        self.assertEqual(len(sport_events), 1)
        self.assertEqual(sport_events[0].get("sport_key"), "basketball_nba")
        self.assertEqual(((sport_events[0].get("result") or {}).get("arb_opportunities") or [])[0].get("event"), "A vs B")


if __name__ == "__main__":
    unittest.main()

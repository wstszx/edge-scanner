import concurrent.futures
import unittest
from unittest.mock import patch

import scanner


class ScannerRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        scanner._set_current_request_logger(None)
        with scanner._REQUEST_TRACE_LOCK:
            scanner._REQUEST_TRACE_ACTIVE.clear()

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
                "purebet_update": {
                    "events_merged": 0,
                    "sports": [],
                    "details": {"requested": 0, "success": 0, "failed": 0, "empty": 0, "retries": 0},
                    "league_sync": {
                        "live_updates": 0,
                        "cache_hits": 0,
                        "stale_cache_uses": 0,
                        "dynamic_added": 0,
                        "unresolved": 0,
                    },
                },
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
                include_purebet=True,
            )

        self.assertTrue(result.get("success"))
        self.assertTrue(captured_targets)
        self.assertIn("tennis_atp", captured_targets[0])


if __name__ == "__main__":
    unittest.main()

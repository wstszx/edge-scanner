import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from io import StringIO
from contextlib import redirect_stdout

import provider_verification as pv


class ProviderVerificationTests(unittest.TestCase):
    def test_main_includes_live_availability_paths_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_live_paths = {
                "json": str(Path(temp_dir) / "live_availability.json"),
                "markdown": str(Path(temp_dir) / "live_availability.md"),
            }
            stdout = StringIO()
            with (
                patch.object(pv, "run_provider_tests", return_value={"ran": False}),
                patch.object(pv, "run_live_scan", return_value={"success": True, "scan_diagnostics": {}}),
                patch.object(pv, "build_report", return_value={"providers": [], "scan": {}, "tests": {"ran": False}}),
                patch.object(pv, "write_report", return_value={"json": str(Path(temp_dir) / "provider.json"), "markdown": str(Path(temp_dir) / "provider.md")}),
                patch.object(pv, "write_live_scan_report", return_value=fake_live_paths) as mocked_live_writer,
                redirect_stdout(stdout),
            ):
                exit_code = pv.main([
                    "--skip-tests",
                    "--json-stdout",
                    "--include-live-availability",
                    "--out-dir",
                    temp_dir,
                ])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(mocked_live_writer.called)
        self.assertEqual(payload["live_availability"], fake_live_paths)

    def test_summarize_provider_statuses_marks_errors(self) -> None:
        result = {
            "custom_providers": {
                "bookmaker_xyz": {
                    "enabled": True,
                    "events_merged": 5,
                    "sports": [{"sport_key": "basketball_nba", "events_returned": 5}],
                },
                "sx_bet": {
                    "enabled": True,
                    "events_merged": 0,
                    "sports": [{"sport_key": "basketball_nba", "error": "upstream 521"}],
                },
            },
        }

        statuses = pv.summarize_provider_statuses(
            result,
            provider_keys=["bookmaker_xyz", "sx_bet"],
            stake_amount=100.0,
        )

        self.assertEqual(statuses[0].status, "ok")
        self.assertEqual(statuses[1].status, "error")
        self.assertIn("upstream 521", statuses[1].errors)

    def test_summarize_provider_statuses_marks_empty_azuro_feed_note(self) -> None:
        result = {
            "custom_providers": {
                "bookmaker_xyz": {
                    "enabled": True,
                    "events_merged": 0,
                    "events_payload_count": 4800,
                    "conditions_sport_filtered_count": 0,
                    "dictionary_loaded": True,
                    "dictionary_error": "",
                    "sports": [],
                }
            }
        }

        statuses = pv.summarize_provider_statuses(
            result,
            provider_keys=["bookmaker_xyz"],
            stake_amount=100.0,
        )

        self.assertEqual(statuses[0].status, "warning")
        self.assertTrue(
            any("Official Azuro feed currently has no leagues matching this sport key." in note for note in statuses[0].notes)
        )

    def test_summarize_provider_statuses_marks_live_feed_empty_as_note(self) -> None:
        result = {
            "custom_providers": {
                "artline": {
                    "enabled": True,
                    "events_merged": 0,
                    "sports": [
                        {
                            "sport_key": "icehockey_nhl",
                            "events_returned": 0,
                            "stats": {
                                "games_type": "live",
                                "live_feed_empty": True,
                                "payload_games_count": 0,
                            },
                        }
                    ],
                }
            }
        }

        statuses = pv.summarize_provider_statuses(
            result,
            provider_keys=["artline"],
            stake_amount=100.0,
        )

        self.assertEqual(statuses[0].status, "warning")
        self.assertTrue(
            any("Live endpoint was reachable but returned zero events at check time." in note for note in statuses[0].notes)
        )

    def test_summarize_scan_flags_liquidity_limited_arbitrage(self) -> None:
        result = {
            "success": True,
            "partial": False,
            "arbitrage": {
                "opportunities_count": 1,
                "opportunities": [
                    {
                        "event": "A vs B",
                        "market": "h2h",
                        "roi_percent": 12.5,
                        "best_odds": [
                            {"bookmaker": "Book A", "price": 3.1, "max_stake": 8.0},
                            {"bookmaker": "Book B", "price": 1.7, "max_stake": None},
                        ],
                        "stakes": {
                            "limited_by_max_stake": True,
                            "requested_total": 100.0,
                            "total": 18.0,
                        },
                    }
                ],
            },
            "middles": {"opportunities_count": 0, "opportunities": []},
            "plus_ev": {"opportunities_count": 0, "opportunities": []},
        }

        summary = pv.summarize_scan(result, top_n=1)

        self.assertEqual(summary["arbitrage_count"], 1)
        self.assertIn("liquidity-limited", summary["top_arbitrage"][0]["note"])

    def test_summarize_scan_exposes_positive_only_counts(self) -> None:
        result = {
            "success": True,
            "partial": False,
            "arbitrage": {
                "opportunities_count": 3,
                "opportunities": [
                    {"event": "A vs B", "market": "h2h", "roi_percent": 1.5},
                    {"event": "C vs D", "market": "h2h", "roi_percent": 0.0},
                    {"event": "E vs F", "market": "totals", "roi_percent": -0.4},
                ],
            },
            "middles": {
                "opportunities_count": 3,
                "opportunities": [
                    {"event": "A vs B", "market": "totals", "ev_percent": 2.0},
                    {"event": "C vs D", "market": "spreads", "ev_percent": 0.0},
                    {"event": "E vs F", "market": "totals", "ev_percent": -1.1},
                ],
            },
            "plus_ev": {"opportunities_count": 0, "opportunities": []},
        }

        summary = pv.summarize_scan(result, top_n=2)

        self.assertEqual(summary["arbitrage_count"], 3)
        self.assertEqual(summary["positive_arbitrage_count"], 1)
        self.assertEqual(summary["middle_count"], 3)
        self.assertEqual(summary["positive_middle_count"], 1)

    def test_write_report_creates_json_and_markdown(self) -> None:
        report = {
            "generated_at": "2026-03-13T00:00:00Z",
            "sport_key": "basketball_nba",
            "regions": ["us", "eu"],
            "stake_amount": 100.0,
            "providers": [
                {
                    "key": "polymarket",
                    "name": "Polymarket",
                    "status": "ok",
                    "enabled": True,
                    "events_merged": 10,
                    "errors": [],
                    "notes": [],
                    "docs": ["https://docs.polymarket.com/"],
                }
            ],
            "tests": {"ran": False, "ok": None, "returncode": None},
            "scan": {
                "success": True,
                "partial": False,
                "arbitrage_count": 0,
                "middle_count": 0,
                "plus_ev_count": 0,
                "top_arbitrage": [],
                "top_middles": [],
                "top_plus_ev": [],
                "sport_errors": [],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            written = pv.write_report(report, Path(temp_dir))
            self.assertTrue(Path(written["json"]).is_file())
            self.assertTrue(Path(written["markdown"]).is_file())
            self.assertTrue(Path(written["latest_json"]).is_file())
            self.assertTrue(Path(written["latest_markdown"]).is_file())

    def test_report_to_markdown_includes_scan_diagnostics(self) -> None:
        report = {
            'generated_at': '2026-03-22T00:00:00Z',
            'sport_key': 'basketball_nba',
            'regions': ['us', 'eu'],
            'stake_amount': 100.0,
            'providers': [],
            'tests': {'ran': True, 'ok': True, 'returncode': 0},
            'scan': {
                'success': True,
                'partial': False,
                'arbitrage_count': 6,
                'positive_arbitrage_count': 0,
                'middle_count': 34,
                'positive_middle_count': 8,
                'plus_ev_count': 0,
                'scan_diagnostics': {
                    'reason_code': 'positive_middle_found',
                    'raw_provider_events': 64,
                    'merged_events_scanned': 0,
                },
                'top_arbitrage': [],
                'top_middles': [],
                'top_plus_ev': [],
                'sport_errors': [],
            },
        }

        markdown = pv.report_to_markdown(report)

        self.assertIn('## Scan Summary', markdown)
        self.assertIn('- positive_arbitrage_count: 0', markdown)
        self.assertIn('- positive_middle_count: 8', markdown)
        self.assertIn('- reason_code: positive_middle_found', markdown)
        self.assertIn('## Scan Diagnostics', markdown)
        self.assertIn('- reason_code: positive_middle_found', markdown)
        self.assertIn('- raw_provider_events: 64', markdown)
        self.assertIn('- merged_events_scanned: 0', markdown)

    def test_build_console_summary_surfaces_provider_and_result_alerts(self) -> None:
        report = {
            "sport_key": "basketball_nba",
            "regions": ["us", "eu"],
            "providers": [
                {
                    "key": "sx_bet",
                    "name": "SX Bet",
                    "status": "error",
                    "enabled": True,
                    "events_merged": 0,
                    "errors": ["upstream 521"],
                    "notes": ["check docs"],
                    "docs": ["https://api.docs.sx.bet/"],
                }
            ],
            "tests": {"ran": True, "ok": True, "returncode": 0},
            "scan": {
                "success": True,
                "partial": True,
                "arbitrage_count": 6,
                "positive_arbitrage_count": 0,
                "middle_count": 34,
                "positive_middle_count": 8,
                "plus_ev_count": 0,
                "scan_diagnostics": {
                    "reason_code": "positive_middle_found",
                },
                "sport_errors": [{"sport_key": "basketball_nba", "error": "SX Bet: upstream 521"}],
                "top_arbitrage": [
                    {
                        "event": "A vs B",
                        "market": "h2h",
                        "roi_percent": 25.0,
                        "books": [{"bookmaker": "Book A", "price": 3.0, "max_stake": 10.0}],
                        "note": "",
                    }
                ],
                "top_middles": [
                    {
                        "event": "C vs D",
                        "market": "totals",
                        "ev_percent": -0.5,
                    }
                ],
                "top_plus_ev": [],
            },
        }

        summary = pv.build_console_summary(
            report,
            written={"latest_json": "data/x.json", "latest_markdown": "data/x.md"},
        )

        self.assertIn("provider alerts:", summary)
        self.assertIn("SX Bet [error]", summary)
        self.assertIn("reason_code=positive_middle_found", summary)
        self.assertIn("arbitrage=6 (positive=0)", summary)
        self.assertIn("middles=34 (positive=8)", summary)
        self.assertIn("arbitrage suspect", summary)
        self.assertIn("middle negative EV", summary)
        self.assertIn("latest_json=data/x.json", summary)

    def test_report_has_alerts_false_for_clean_report(self) -> None:
        report = {
            "providers": [
                {
                    "key": "polymarket",
                    "name": "Polymarket",
                    "status": "ok",
                    "enabled": True,
                    "events_merged": 10,
                    "errors": [],
                    "notes": [],
                    "docs": [],
                }
            ],
            "tests": {"ran": True, "ok": True, "returncode": 0},
            "scan": {
                "success": True,
                "partial": False,
                "sport_errors": [],
                "top_arbitrage": [],
                "top_middles": [],
                "top_plus_ev": [],
            },
        }

        self.assertFalse(pv.report_has_alerts(report))

    def test_summarize_scan_preserves_scan_diagnostics(self) -> None:
        result = {
            'success': True,
            'partial': False,
            'scan_diagnostics': {
                'reason_code': 'no_cross_provider_overlap',
                'raw_provider_events': 64,
                'merged_events_scanned': 0,
            },
            'arbitrage': {'opportunities_count': 0, 'opportunities': []},
            'middles': {'opportunities_count': 0, 'opportunities': []},
            'plus_ev': {'opportunities_count': 0, 'opportunities': []},
        }

        summary = pv.summarize_scan(result, top_n=1)

        self.assertEqual(
            summary['scan_diagnostics'],
            {
                'reason_code': 'no_cross_provider_overlap',
                'raw_provider_events': 64,
                'merged_events_scanned': 0,
            },
        )

    def test_run_live_scan_passes_live_scan_mode(self) -> None:
        with patch.object(pv, "run_scan", return_value={"success": True}) as mocked_run_scan:
            result = pv.run_live_scan(
                sport_key="basketball_nba",
                provider_keys=["sx_bet", "polymarket"],
                regions=["us"],
                stake_amount=100.0,
                all_markets=False,
            )

        self.assertEqual(result, {"success": True})
        self.assertEqual(mocked_run_scan.call_args.kwargs.get("scan_mode"), "live")
        self.assertEqual(
            mocked_run_scan.call_args.kwargs.get("include_providers"),
            ["sx_bet", "polymarket"],
        )


if __name__ == "__main__":
    unittest.main()

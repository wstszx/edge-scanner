import tempfile
import unittest
from pathlib import Path

import provider_verification as pv


class ProviderVerificationTests(unittest.TestCase):
    def test_summarize_provider_statuses_marks_proxy_notes_and_errors(self) -> None:
        result = {
            "custom_providers": {
                "dexsport_io": {
                    "enabled": True,
                    "events_merged": 5,
                    "sports": [{"sport_key": "basketball_nba", "events_returned": 5}],
                },
                "purebet": {
                    "enabled": True,
                    "events_merged": 0,
                    "sports": [{"sport_key": "basketball_nba", "error": "upstream 521"}],
                },
            },
            "purebet": {
                "enabled": True,
                "events_merged": 0,
                "sports": [{"sport_key": "basketball_nba", "error": "upstream 521"}],
            },
        }

        statuses = pv.summarize_provider_statuses(
            result,
            provider_keys=["dexsport_io", "purebet"],
            stake_amount=100.0,
        )

        self.assertEqual(statuses[0].status, "ok")
        self.assertTrue(any("bookmaker.xyz" in note for note in statuses[0].notes))
        self.assertEqual(statuses[1].status, "error")
        self.assertIn("upstream 521", statuses[1].errors)

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

    def test_build_console_summary_surfaces_provider_and_result_alerts(self) -> None:
        report = {
            "sport_key": "basketball_nba",
            "regions": ["us", "eu"],
            "providers": [
                {
                    "key": "purebet",
                    "name": "Purebet",
                    "status": "error",
                    "enabled": True,
                    "events_merged": 0,
                    "errors": ["upstream 521"],
                    "notes": ["check docs"],
                    "docs": ["https://docs.purebet.io/"],
                }
            ],
            "tests": {"ran": True, "ok": True, "returncode": 0},
            "scan": {
                "success": True,
                "partial": True,
                "arbitrage_count": 1,
                "middle_count": 1,
                "plus_ev_count": 0,
                "sport_errors": [{"sport_key": "basketball_nba", "error": "Purebet: upstream 521"}],
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
        self.assertIn("Purebet [error]", summary)
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


if __name__ == "__main__":
    unittest.main()

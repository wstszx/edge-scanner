import json
import tempfile
import unittest
from pathlib import Path

import live_availability as la


class LiveAvailabilityTests(unittest.TestCase):
    def test_build_report_data_summarizes_provider_state_and_overlap(self) -> None:
        samples = {
            "sx_bet": [
                {
                    "home_team": "Home A",
                    "away_team": "Away A",
                    "live_state": {"is_live": True, "status": "live"},
                    "bookmakers": [{"markets": [{"key": "h2h"}, {"key": "spreads"}]}],
                },
                {
                    "home_team": "Home B",
                    "away_team": "Away B",
                    "live_state": {"is_live": False, "status": "scheduled"},
                    "bookmakers": [{"markets": [{"key": "h2h"}]}],
                },
            ],
            "betdex": [
                {
                    "home_team": "Away A",
                    "away_team": "Home A",
                    "live_state": {"is_live": True, "status": "live"},
                    "bookmakers": [{"markets": [{"key": "h2h"}, {"key": "totals"}]}],
                }
            ],
            "polymarket": [],
        }

        report = la.build_report_data(samples, scan_mode="live")

        self.assertEqual(report["scan_mode"], "live")
        self.assertEqual(report["providers"]["sx_bet"]["raw_events"], 2)
        self.assertEqual(report["providers"]["sx_bet"]["explicit_live_events"], 1)
        self.assertEqual(report["providers"]["sx_bet"]["scheduled_or_prematch_events"], 1)
        self.assertEqual(report["providers"]["betdex"]["raw_events"], 1)
        pair = next(item for item in report["overlap_pairs"] if item["providers"] == ["betdex", "sx_bet"])
        self.assertEqual(pair["overlap_matchups"], 1)
        self.assertEqual(pair["common_market_keys"], ["h2h"])

    def test_write_report_creates_json_and_markdown_files(self) -> None:
        report = {
            "generated_at": "2026-03-28T00:00:00Z",
            "scan_mode": "live",
            "providers": {
                "sx_bet": {
                    "raw_events": 2,
                    "explicit_live_events": 1,
                    "scheduled_or_prematch_events": 1,
                    "unique_matchups": 2,
                    "sample_events": [],
                }
            },
            "overlap_pairs": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = la.write_report(report, Path(temp_dir))
            self.assertTrue(Path(paths["json"]).is_file())
            self.assertTrue(Path(paths["markdown"]).is_file())
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["scan_mode"], "live")
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("# Live Availability Report", markdown)
            self.assertIn("sx_bet", markdown)

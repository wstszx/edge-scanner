"""Tests for history.py — HistoryManager and helpers."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from history import HistoryManager, _flatten_record, _utc_now


class TestHistoryManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.hm = HistoryManager(history_dir=self.tmpdir, max_records=100, enabled=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_scan_result(self):
        return {
            "success": True,
            "opportunities": [
                {
                    "event": "Team A vs Team B",
                    "sport": "basketball_nba",
                    "sport_display": "NBA",
                    "market": "h2h",
                    "commence_time": "2024-01-01T20:00:00Z",
                    "roi_percent": 1.5,
                    "best_odds": [
                        {"outcome": "Team A", "bookmaker": "DraftKings", "price": 2.1},
                        {"outcome": "Team B", "bookmaker": "FanDuel", "price": 2.1},
                    ],
                }
            ],
            "middles": [],
            "plus_ev": [
                {
                    "event": "Team C vs Team D",
                    "sport": "americanfootball_nfl",
                    "sport_display": "NFL",
                    "market": "spreads",
                    "commence_time": "2024-01-02T18:00:00Z",
                    "edge_percent": 5.2,
                    "ev_per_100": 4.1,
                    "bet": {"soft_book": "BetMGM", "soft_odds": 2.05},
                    "sharp": {"book": "Pinnacle", "fair_odds": 1.95},
                }
            ],
        }

    def test_save_and_load(self):
        scan = self._make_scan_result()
        written = self.hm.save_opportunities(scan, scan_time="2024-01-01T12:00:00Z")
        self.assertEqual(written, 2)  # 1 arb + 1 ev

        records = self.hm.load_recent(limit=10)
        self.assertEqual(len(records), 2)

    def test_load_by_mode(self):
        scan = self._make_scan_result()
        self.hm.save_opportunities(scan, scan_time="2024-01-01T12:00:00Z")

        arb_records = self.hm.load_recent(limit=10, mode="arbitrage")
        self.assertEqual(len(arb_records), 1)
        self.assertEqual(arb_records[0]["mode"], "arbitrage")

        ev_records = self.hm.load_recent(limit=10, mode="ev")
        self.assertEqual(len(ev_records), 1)
        self.assertEqual(ev_records[0]["mode"], "ev")

    def test_load_ordered_by_scan_time_descending(self):
        for i in range(3):
            self.hm.save_opportunities(
                self._make_scan_result(),
                scan_time=f"2024-01-0{i+1}T12:00:00Z",
            )
        records = self.hm.load_recent(limit=10, mode="arbitrage")
        times = [r["scan_time"] for r in records]
        self.assertEqual(times, sorted(times, reverse=True))

    def test_get_stats(self):
        scan = self._make_scan_result()
        self.hm.save_opportunities(scan, scan_time="2024-01-01T12:00:00Z")
        stats = self.hm.get_stats()
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["modes"]["arbitrage"]["count"], 1)
        self.assertEqual(stats["modes"]["ev"]["count"], 1)
        self.assertEqual(stats["modes"]["middles"]["count"], 0)

    def test_trim_respects_max_records(self):
        hm = HistoryManager(history_dir=self.tmpdir, max_records=5, enabled=True)
        scan = self._make_scan_result()
        for _ in range(10):
            hm.save_opportunities(scan, scan_time="2024-01-01T12:00:00Z")
        records = hm.load_recent(limit=100, mode="arbitrage")
        self.assertLessEqual(len(records), 5)

    def test_disabled_saves_nothing(self):
        hm = HistoryManager(history_dir=self.tmpdir, enabled=False)
        written = hm.save_opportunities(self._make_scan_result(), "2024-01-01T12:00:00Z")
        self.assertEqual(written, 0)

    def test_invalid_result_saves_nothing(self):
        written = self.hm.save_opportunities("not a dict", "2024-01-01T12:00:00Z")
        self.assertEqual(written, 0)

    def test_clear(self):
        self.hm.save_opportunities(self._make_scan_result(), "2024-01-01T12:00:00Z")
        self.hm.clear()
        records = self.hm.load_recent(limit=100)
        self.assertEqual(len(records), 0)

    def test_save_nested_run_scan_result_shape(self):
        scan = {
            "success": True,
            "arbitrage": {
                "opportunities": [
                    {
                        "event": "Team A vs Team B",
                        "sport": "basketball_nba",
                        "sport_display": "NBA",
                        "market": "h2h",
                        "commence_time": "2024-01-01T20:00:00Z",
                        "roi_percent": 1.5,
                        "best_odds": [
                            {"outcome": "Team A", "bookmaker": "DraftKings", "price": 2.1},
                            {"outcome": "Team B", "bookmaker": "FanDuel", "price": 2.1},
                        ],
                    }
                ]
            },
            "middles": {
                "opportunities": [
                    {
                        "event": "Team C vs Team D",
                        "sport": "americanfootball_nfl",
                        "sport_display": "NFL",
                        "market": "spreads",
                        "commence_time": "2024-01-02T18:00:00Z",
                        "ev_percent": 2.4,
                        "probability_percent": 12.5,
                        "gap": {"points": 3.0},
                        "side_a": {"bookmaker": "Book A", "line": -3.5},
                        "side_b": {"bookmaker": "Book B", "line": 1.5},
                    }
                ]
            },
            "plus_ev": {
                "opportunities": [
                    {
                        "event": "Team E vs Team F",
                        "sport": "baseball_mlb",
                        "sport_display": "MLB",
                        "market": "h2h",
                        "commence_time": "2024-01-03T18:00:00Z",
                        "edge_percent": 3.2,
                        "ev_per_100": 2.3,
                        "bet": {"soft_book": "BetMGM", "soft_odds": 2.1},
                        "sharp": {"book": "Pinnacle", "fair_odds": 1.95},
                    }
                ]
            },
        }

        written = self.hm.save_opportunities(scan, scan_time="2024-01-01T12:00:00Z")
        self.assertEqual(written, 3)
        stats = self.hm.get_stats()
        self.assertEqual(stats["modes"]["arbitrage"]["count"], 1)
        self.assertEqual(stats["modes"]["middles"]["count"], 1)
        self.assertEqual(stats["modes"]["ev"]["count"], 1)


class TestFlattenRecord(unittest.TestCase):
    def test_arbitrage_record(self):
        item = {
            "event": "A vs B",
            "sport": "nba",
            "sport_display": "NBA",
            "market": "h2h",
            "commence_time": "2024-01-01T20:00:00Z",
            "roi_percent": 2.1,
            "best_odds": [{"outcome": "A", "bookmaker": "DK", "price": 2.2}],
        }
        rec = _flatten_record(item, "2024-01-01T12:00:00Z", "arbitrage")
        self.assertEqual(rec["mode"], "arbitrage")
        self.assertEqual(rec["roi_percent"], 2.1)
        self.assertIsInstance(rec["books"], list)

    def test_ev_record(self):
        item = {
            "event": "C vs D",
            "market": "spreads",
            "edge_percent": 5.0,
            "ev_per_100": 3.5,
            "bet": {"soft_book": "FanDuel", "soft_odds": 2.1},
            "sharp": {"book": "Pinnacle", "fair_odds": 2.0},
        }
        rec = _flatten_record(item, "2024-01-01T12:00:00Z", "ev")
        self.assertEqual(rec["edge_percent"], 5.0)
        self.assertEqual(rec["soft_book"], "FanDuel")

    def test_middles_record_new_shape(self):
        item = {
            "event": "E vs F",
            "market": "totals",
            "ev_percent": 1.8,
            "probability_percent": 10.0,
            "gap": {"points": 2.0},
            "side_a": {"bookmaker": "Book A", "line": 219.5},
            "side_b": {"bookmaker": "Book B", "line": 221.5},
        }
        rec = _flatten_record(item, "2024-01-01T12:00:00Z", "middles")
        self.assertEqual(rec["ev"], 1.8)
        self.assertEqual(rec["gap_points"], 2.0)
        self.assertAlmostEqual(rec["probability"], 0.1, places=5)
        self.assertEqual(rec["books"][0]["bookmaker"], "Book A")
        self.assertEqual(rec["books"][1]["line"], 221.5)


if __name__ == "__main__":
    unittest.main()

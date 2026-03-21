from __future__ import annotations

import datetime as dt
import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import scanner


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "provider_snapshot_golden.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _scan_now_epoch() -> int:
    return int(
        dt.datetime(2026, 3, 20, 1, 0, tzinfo=dt.timezone.utc).timestamp()
    )


class ProviderSnapshotGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()

    def test_polymarket_real_snapshot_h2h_structure_is_consumable(self) -> None:
        event = deepcopy(self.fixture["polymarket_h2h"])

        with patch("scanner.time.time", return_value=_scan_now_epoch()):
            filtered_events, dropped = scanner._filter_events_for_scan([event])
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(dropped, {"dropped_past": 0, "dropped_missing_time": 0})

        bookmakers = event.get("bookmakers") or []
        self.assertEqual(len(bookmakers), 1)
        self.assertEqual(bookmakers[0].get("key"), "polymarket")
        self.assertEqual(bookmakers[0].get("event_url"), "https://polymarket.com/event/nba-nop-det-2026-03-26")

        markets = bookmakers[0].get("markets") or []
        self.assertEqual([market.get("key") for market in markets], ["h2h"])
        outcomes = markets[0].get("outcomes") or []
        self.assertEqual(
            outcomes,
            [
                {
                    "name": "Pelicans",
                    "price": 2.0,
                    "stake": 2.5,
                    "raw_percentage_odds": "0.26",
                    "quote_source": "clob_book_best_ask",
                },
                {
                    "name": "Pistons",
                    "price": 1.030928,
                    "stake": 32.3301,
                    "raw_percentage_odds": "0.74",
                    "quote_source": "clob_book_best_ask",
                },
            ],
        )

    def test_real_snapshot_bookmaker_xyz_and_sx_bet_totals_regression(self) -> None:
        payload = deepcopy(self.fixture["pairs"]["san_jose_totals_bookmaker_vs_sx"])
        bookmaker_event = payload["events"]["bookmaker_xyz"]
        sx_event = payload["events"]["sx_bet"]

        merged = scanner._merge_events([bookmaker_event], [sx_event])
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            {book.get("key") for book in (merged[0].get("bookmakers") or [])},
            {"bookmaker_xyz", "sx_bet"},
        )

        entries = scanner._collect_market_entries(merged[0], payload["market"], 100.0, 0.0)
        self.assertTrue(entries)
        best_entry = max(entries, key=lambda item: item.get("roi_percent", 0))

        self.assertEqual(best_entry.get("market"), "totals")
        self.assertAlmostEqual(best_entry.get("roi_percent", 0), payload["expected"]["roi_percent"], places=2)
        selected = [
            {
                "outcome": item.get("outcome"),
                "bookmaker_key": item.get("bookmaker_key"),
                "price": item.get("price"),
                "point": item.get("point"),
            }
            for item in (best_entry.get("best_odds") or [])
        ]
        self.assertEqual(selected, payload["expected"]["best_odds"])

    def test_real_snapshot_betdex_and_sx_bet_spreads_regression(self) -> None:
        payload = deepcopy(self.fixture["pairs"]["san_jose_spreads_betdex_vs_sx"])
        betdex_event = payload["events"]["betdex"]
        sx_event = payload["events"]["sx_bet"]

        merged = scanner._merge_events([betdex_event], [sx_event])
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            {book.get("key") for book in (merged[0].get("bookmakers") or [])},
            {"betdex", "sx_bet"},
        )

        entries = scanner._collect_market_entries(merged[0], payload["market"], 100.0, 0.0)
        self.assertTrue(entries)
        best_entry = max(entries, key=lambda item: item.get("roi_percent", 0))

        self.assertEqual(best_entry.get("market"), "spreads")
        self.assertAlmostEqual(best_entry.get("roi_percent", 0), payload["expected"]["roi_percent"], places=2)
        selected = [
            {
                "outcome": item.get("outcome"),
                "bookmaker_key": item.get("bookmaker_key"),
                "price": item.get("price"),
                "point": item.get("point"),
            }
            for item in (best_entry.get("best_odds") or [])
        ]
        self.assertEqual(selected, payload["expected"]["best_odds"])

    def test_real_snapshot_betdex_and_sx_bet_alias_merge_regression(self) -> None:
        payload = deepcopy(self.fixture["pairs"]["atlanta_totals_betdex_vs_sx"])
        betdex_event = payload["events"]["betdex"]
        sx_event = payload["events"]["sx_bet"]

        merged = scanner._merge_events([betdex_event], [sx_event])
        self.assertEqual(len(merged), 1)
        merged_event = merged[0]
        self.assertEqual(merged_event.get("home_team"), "Atlanta United")
        self.assertEqual(merged_event.get("away_team"), "Philadelphia Union")
        self.assertEqual(
            {book.get("key") for book in (merged_event.get("bookmakers") or [])},
            {"betdex", "sx_bet"},
        )

        entries = scanner._collect_market_entries(merged_event, payload["market"], 100.0, 0.0)
        self.assertTrue(entries)
        best_entry = max(entries, key=lambda item: item.get("roi_percent", 0))

        self.assertEqual(best_entry.get("market"), "totals")
        self.assertAlmostEqual(best_entry.get("roi_percent", 0), payload["expected"]["roi_percent"], places=2)
        selected = [
            {
                "outcome": item.get("outcome"),
                "bookmaker_key": item.get("bookmaker_key"),
                "price": item.get("price"),
                "point": item.get("point"),
            }
            for item in (best_entry.get("best_odds") or [])
        ]
        self.assertEqual(selected, payload["expected"]["best_odds"])


if __name__ == "__main__":
    unittest.main()

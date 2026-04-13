from __future__ import annotations

import copy
import datetime as dt
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import scanner


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "artline_snapshot_golden.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _scan_now_epoch() -> int:
    return int(
        dt.datetime(2026, 3, 21, 12, 0, tzinfo=dt.timezone.utc).timestamp()
    )


class ArtlineSnapshotGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()

    def test_artline_and_sx_bet_totals_snapshot_regression(self) -> None:
        payload = copy.deepcopy(self.fixture["pair"])
        artline_event = payload["events"]["artline"]
        sx_event = payload["events"]["sx_bet"]

        merged = scanner._merge_events([artline_event], [sx_event])
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            {book.get("key") for book in (merged[0].get("bookmakers") or [])},
            {"artline", "sx_bet"},
        )

        with patch("scanner.time.time", return_value=_scan_now_epoch()):
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


if __name__ == "__main__":
    unittest.main()

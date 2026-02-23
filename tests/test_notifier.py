"""Tests for notifier.py compatibility with current scan payload fields."""

import unittest
from unittest.mock import patch

from notifier import Notifier


class NotifierTests(unittest.TestCase):
    def test_middle_filter_accepts_ev_percent(self) -> None:
        notifier = Notifier(
            webhook_url="https://example.com/webhook",
            min_ev=1.0,
        )
        middle_list = [
            {
                "event": "Team A vs Team B",
                "market": "spreads",
                "ev_percent": 2.4,
                "gap": {"points": 3.0},
                "probability_percent": 12.5,
            }
        ]
        with patch.object(notifier, "_send_webhook", return_value={"ok": True}) as mocked_send:
            result = notifier.notify_opportunities(
                arb_list=[],
                middle_list=middle_list,
                ev_list=[],
                scan_time="2026-02-22T12:00:00Z",
            )

        self.assertTrue(result.get("sent"))
        mocked_send.assert_called_once()
        summary = mocked_send.call_args.args[0]
        self.assertEqual(summary.get("middles_count"), 1)
        top_middle = (summary.get("top_middles") or [{}])[0]
        self.assertEqual(top_middle.get("gap_points"), 3.0)
        self.assertEqual(top_middle.get("ev"), 2.4)
        self.assertAlmostEqual(top_middle.get("probability"), 0.125, places=5)


if __name__ == "__main__":
    unittest.main()

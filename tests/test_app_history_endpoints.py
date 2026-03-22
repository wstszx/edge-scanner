import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import app as app_module


class HistoryEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def test_history_returns_records_and_count(self) -> None:
        records = [
            {"mode": "arbitrage", "event": "A vs B"},
            {"mode": "middles", "event": "C vs D"},
        ]
        history_manager = MagicMock()
        history_manager.load_recent.return_value = records

        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertTrue(payload.get("success"))
        self.assertEqual(payload.get("records"), records)
        self.assertEqual(payload.get("count"), len(records))
        history_manager.load_recent.assert_called_once_with(limit=200, mode=None)

    def test_history_normalizes_mode_and_clamps_limit(self) -> None:
        history_manager = MagicMock()
        history_manager.load_recent.return_value = []

        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history?mode=MIDDLES&limit=5000")

        self.assertEqual(response.status_code, 200)
        history_manager.load_recent.assert_called_once_with(limit=1000, mode="middles")

    def test_history_falls_back_to_default_limit_for_invalid_value(self) -> None:
        history_manager = MagicMock()
        history_manager.load_recent.return_value = []

        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history?limit=not-a-number")

        self.assertEqual(response.status_code, 200)
        history_manager.load_recent.assert_called_once_with(limit=200, mode=None)

    def test_history_returns_500_when_manager_raises(self) -> None:
        history_manager = MagicMock()
        history_manager.load_recent.side_effect = RuntimeError("history unavailable")

        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history")

        self.assertEqual(response.status_code, 500)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertIn("history unavailable", payload.get("error", ""))

    def test_history_stats_returns_manager_stats(self) -> None:
        stats = {
            "enabled": True,
            "dir": "tmp/history",
            "modes": {
                "arbitrage": {"count": 1, "size_bytes": 10},
                "middles": {"count": 2, "size_bytes": 20},
                "ev": {"count": 3, "size_bytes": 30},
            },
        }
        history_manager = MagicMock()
        history_manager.get_stats.return_value = stats

        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history/stats")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertTrue(payload.get("success"))
        self.assertEqual(payload.get("enabled"), True)
        self.assertEqual(payload.get("dir"), stats["dir"])
        self.assertEqual(payload.get("modes"), stats["modes"])
        history_manager.get_stats.assert_called_once_with()

    def test_history_stats_returns_500_when_manager_raises(self) -> None:
        history_manager = MagicMock()
        history_manager.get_stats.side_effect = RuntimeError("stats unavailable")

        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history/stats")

        self.assertEqual(response.status_code, 500)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertIn("stats unavailable", payload.get("error", ""))


class ProviderSnapshotEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def test_provider_snapshot_rejects_invalid_provider_key(self) -> None:
        with patch.object(app_module, "_provider_snapshot_path", return_value=None):
            response = self.client.get("/provider-snapshots/not-valid")

        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertEqual(payload.get("error"), "Invalid provider key")

    def test_provider_snapshot_returns_404_when_snapshot_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "bookmaker_xyz.json"
            with patch.object(app_module, "_provider_snapshot_path", return_value=missing_path):
                response = self.client.get("/provider-snapshots/bookmaker_xyz")

        self.assertEqual(response.status_code, 404)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertIn("No snapshot found", payload.get("error", ""))

    def test_provider_snapshot_returns_snapshot_payload(self) -> None:
        snapshot = {"provider": "bookmaker_xyz", "events": [{"id": "evt-1"}]}

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "bookmaker_xyz.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            with patch.object(app_module, "_provider_snapshot_path", return_value=snapshot_path):
                response = self.client.get("/provider-snapshots/bookmaker_xyz")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertTrue(payload.get("success"))
        self.assertEqual(payload.get("provider_key"), "bookmaker_xyz")
        self.assertEqual(payload.get("snapshot"), snapshot)

    def test_provider_snapshot_returns_500_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "bookmaker_xyz.json"
            snapshot_path.write_text("{invalid json", encoding="utf-8")
            with patch.object(app_module, "_provider_snapshot_path", return_value=snapshot_path):
                response = self.client.get("/provider-snapshots/bookmaker_xyz")

        self.assertEqual(response.status_code, 500)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertTrue(payload.get("error"))


class CrossProviderReportEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def test_cross_provider_report_rejects_invalid_report_path(self) -> None:
        with patch.object(app_module, "_cross_provider_report_path", return_value=None):
            response = self.client.get("/cross-provider-report")

        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertEqual(payload.get("error"), "Invalid report path")

    def test_cross_provider_report_returns_404_when_report_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "cross_provider_match_report.json"
            with patch.object(app_module, "_cross_provider_report_path", return_value=missing_path):
                response = self.client.get("/cross-provider-report")

        self.assertEqual(response.status_code, 404)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertEqual(payload.get("error"), "Cross-provider report file not found")

    def test_cross_provider_report_returns_report_payload(self) -> None:
        report = {"matched_events": [{"event_id": "evt-1"}], "summary": {"count": 1}}

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "cross_provider_match_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(app_module, "_cross_provider_report_path", return_value=report_path):
                response = self.client.get("/cross-provider-report")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertTrue(payload.get("success"))
        self.assertEqual(payload.get("report"), report)

    def test_cross_provider_report_returns_500_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "cross_provider_match_report.json"
            report_path.write_text("{invalid json", encoding="utf-8")
            with patch.object(app_module, "_cross_provider_report_path", return_value=report_path):
                response = self.client.get("/cross-provider-report")

        self.assertEqual(response.status_code, 500)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertTrue(payload.get("error"))


if __name__ == "__main__":
    unittest.main()

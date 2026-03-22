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

    def test_history_scans_returns_records_and_count(self) -> None:
        records = [
            {"scan_mode": "prematch", "scan_time": "2026-03-22T12:00:00Z"},
            {"scan_mode": "live", "scan_time": "2026-03-22T11:00:00Z"},
        ]
        history_manager = MagicMock()
        history_manager.load_recent_scan_summaries.return_value = records

        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history/scans")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertTrue(payload.get("success"))
        self.assertEqual(
            payload.get("records"),
            [
                {"scan_mode": "prematch", "scan_time": "2026-03-22T20:00:00+08:00"},
                {"scan_mode": "live", "scan_time": "2026-03-22T19:00:00+08:00"},
            ],
        )
        self.assertEqual(payload.get("count"), len(records))
        history_manager.load_recent_scan_summaries.assert_called_once_with(limit=100)

    def test_history_scans_clamps_limit_and_handles_invalid_values(self) -> None:
        history_manager = MagicMock()
        history_manager.load_recent_scan_summaries.return_value = []

        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history/scans?limit=5000")

        self.assertEqual(response.status_code, 200)
        history_manager.load_recent_scan_summaries.assert_called_once_with(limit=1000)

        history_manager = MagicMock()
        history_manager.load_recent_scan_summaries.return_value = []
        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history/scans?limit=not-a-number")

        self.assertEqual(response.status_code, 200)
        history_manager.load_recent_scan_summaries.assert_called_once_with(limit=100)

    def test_history_scans_returns_500_when_manager_raises(self) -> None:
        history_manager = MagicMock()
        history_manager.load_recent_scan_summaries.side_effect = RuntimeError("scan summary unavailable")

        with patch.object(app_module, "get_history_manager", return_value=history_manager):
            response = self.client.get("/history/scans")

        self.assertEqual(response.status_code, 500)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertIn("scan summary unavailable", payload.get("error", ""))

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

    def test_provider_snapshot_converts_nested_timestamp_fields(self) -> None:
        snapshot = {
            "provider": "bookmaker_xyz",
            "saved_at": "2026-03-14T09:15:00Z",
            "events": [
                {
                    "id": "evt-1",
                    "commence_time": "2026-03-15T10:00:00Z",
                    "updated_at": "2026-03-15T09:14:58Z",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "bookmaker_xyz.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            with patch.object(app_module, "_provider_snapshot_path", return_value=snapshot_path):
                response = self.client.get("/provider-snapshots/bookmaker_xyz")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        event = ((payload.get("snapshot") or {}).get("events") or [{}])[0]
        self.assertEqual((payload.get("snapshot") or {}).get("saved_at"), "2026-03-14T17:15:00+08:00")
        self.assertEqual(event.get("commence_time"), "2026-03-15T18:00:00+08:00")
        self.assertEqual(event.get("updated_at"), "2026-03-15T17:14:58+08:00")

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

    def test_cross_provider_report_converts_nested_timestamp_fields(self) -> None:
        report = {
            "saved_at": "2026-03-14T09:15:00Z",
            "matched_events": [
                {
                    "event_id": "evt-1",
                    "representative_time_utc": "2026-03-15T10:00:00Z",
                }
            ],
            "summary": {"count": 1},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "cross_provider_match_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(app_module, "_cross_provider_report_path", return_value=report_path):
                response = self.client.get("/cross-provider-report")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        matched_event = ((payload.get("report") or {}).get("matched_events") or [{}])[0]
        self.assertEqual((payload.get("report") or {}).get("saved_at"), "2026-03-14T17:15:00+08:00")
        self.assertEqual(matched_event.get("representative_time_utc"), "2026-03-15T18:00:00+08:00")

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

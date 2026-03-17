from __future__ import annotations

import copy
import unittest
from unittest.mock import MagicMock, patch

import app as app_module


class _FakeStopEvent:
    def __init__(self, wait_results: list[bool] | None = None) -> None:
        self._flag = False
        self.wait_calls: list[float] = []
        self.wait_results = list(wait_results or [])

    def is_set(self) -> bool:
        return self._flag

    def set(self) -> None:
        self._flag = True

    def clear(self) -> None:
        self._flag = False

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_calls.append(float(timeout or 0.0))
        if self.wait_results:
            result = self.wait_results.pop(0)
            if result:
                self._flag = True
            return result
        return self._flag


class ServerAutoScanAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._server_auto_scan_config = copy.deepcopy(app_module._SERVER_AUTO_SCAN_CONFIG)
        self._server_auto_scan_version = app_module._SERVER_AUTO_SCAN_CONFIG_VERSION
        self._env_save_scan = app_module.ENV_SAVE_SCAN
        self._background_started = app_module._BACKGROUND_SERVICES_STARTED

    def tearDown(self) -> None:
        app_module._SERVER_AUTO_SCAN_CONFIG = copy.deepcopy(self._server_auto_scan_config)
        app_module._SERVER_AUTO_SCAN_CONFIG_VERSION = self._server_auto_scan_version
        app_module.ENV_SAVE_SCAN = self._env_save_scan
        app_module._BACKGROUND_SERVICES_STARTED = self._background_started
        while app_module._SCAN_EXECUTION_LOCK.locked():
            app_module._SCAN_EXECUTION_LOCK.release()

    def test_server_auto_scan_config_round_trips_normalized_payload(self) -> None:
        payload = {
            "enabled": "true",
            "intervalMinutes": "15",
            "payload": {
                "scanMode": "live",
                "sports": ["icehockey_nhl"],
                "allSports": "false",
                "allMarkets": "true",
                "stake": "75",
                "regions": ["us", "eu"],
                "bookmakers": ["sx_bet", "betdex", ""],
                "includeProviders": ["sx_bet", "betdex", ""],
                "commission": "3.5",
                "sharpBook": "Pinnacle",
                "minEdgePercent": "1.2",
                "bankroll": "2500",
                "kellyFraction": "0.4",
            },
        }
        expected_config = {
            "enabled": True,
            "interval_minutes": 15,
            "payload": {
                "scanMode": "live",
                "sports": ["icehockey_nhl"],
                "allSports": False,
                "allMarkets": True,
                "stake": 75.0,
                "regions": ["us", "eu"],
                "bookmakers": ["sx_bet", "betdex"],
                "includeProviders": ["sx_bet", "betdex"],
                "commission": 3.5,
                "sharpBook": "pinnacle",
                "minEdgePercent": 1.2,
                "bankroll": 2500.0,
                "kellyFraction": 0.4,
            },
        }

        with patch.object(app_module, "_persist_server_auto_scan_config") as mocked_persist:
            response = self.client.post("/server-auto-scan-config", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.get_json() or {}
        self.assertTrue(body.get("success"))
        self.assertEqual(body.get("config"), expected_config)
        mocked_persist.assert_called_once_with(expected_config)

        response = self.client.get("/server-auto-scan-config")
        self.assertEqual(response.status_code, 200)
        body = response.get_json() or {}
        self.assertTrue(body.get("success"))
        self.assertEqual(body.get("config"), expected_config)

    def test_server_auto_scan_config_rejects_non_object_payload(self) -> None:
        response = self.client.post(
            "/server-auto-scan-config",
            data='["not-an-object"]',
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        body = response.get_json() or {}
        self.assertFalse(body.get("success"))
        self.assertEqual(body.get("error"), "Auto scan config payload must be a JSON object")

    def test_server_auto_scan_config_rejects_unsupported_bookmakers_only(self) -> None:
        payload = {
            "enabled": True,
            "intervalMinutes": 10,
            "payload": {
                "scanMode": "prematch",
                "sports": ["icehockey_nhl"],
                "bookmakers": ["DraftKings", "FanDuel"],
                "includeProviders": [],
            },
        }

        with patch.object(app_module, "_persist_server_auto_scan_config") as mocked_persist:
            response = self.client.post("/server-auto-scan-config", json=payload)

        self.assertEqual(response.status_code, 400)
        body = response.get_json() or {}
        self.assertFalse(body.get("success"))
        self.assertEqual(
            body.get("error"),
            "No supported arbitrage platforms were selected",
        )
        mocked_persist.assert_not_called()

    def test_background_server_auto_scan_uses_synced_frontend_platform_selection(self) -> None:
        synced_payload = {
            "enabled": True,
            "intervalMinutes": 10,
            "payload": {
                "scanMode": "prematch",
                "sports": ["icehockey_nhl"],
                "allSports": False,
                "allMarkets": False,
                "stake": 100,
                "regions": ["us"],
                "bookmakers": ["SX Bet"],
                "includeProviders": [],
                "commission": 0,
                "sharpBook": "pinnacle",
            },
        }
        fake_event = _FakeStopEvent()
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False

        with patch.object(app_module, "_persist_server_auto_scan_config"):
            response = self.client.post("/server-auto-scan-config", json=synced_payload)
        self.assertEqual(response.status_code, 200)

        def _fake_run_scan(**kwargs):
            fake_event.set()
            return {
                "success": True,
                "scan_time": "2026-03-14T08:00:00Z",
                "arbitrage": {"opportunities": []},
                "middles": {"opportunities": []},
                "plus_ev": {"opportunities": []},
            }

        with (
            patch.object(app_module, "_SERVER_AUTO_SCAN_STOP_EVENT", fake_event),
            patch.object(app_module, "_start_background_provider_services") as mocked_start,
            patch.object(app_module, "get_history_manager", return_value=history_manager),
            patch.object(app_module, "get_notifier", return_value=notifier),
            patch.object(app_module, "run_scan", side_effect=_fake_run_scan) as mocked_run_scan,
            patch.object(app_module, "time") as mocked_time,
            patch.object(app_module, "ENV_SERVER_AUTO_SCAN_RUN_ON_START", True),
        ):
            mocked_time.time.side_effect = [1000.0, 1000.0, 1000.0]
            app_module._server_auto_scan_loop()

        mocked_start.assert_called_once_with(wait_timeout=0.0)
        history_manager.save_opportunities.assert_called_once()
        kwargs = mocked_run_scan.call_args.kwargs
        self.assertEqual(kwargs.get("api_key"), [])
        self.assertEqual(kwargs.get("scan_mode"), "prematch")
        self.assertEqual(kwargs.get("sports"), ["icehockey_nhl"])
        self.assertEqual(kwargs.get("regions"), ["us"])
        self.assertEqual(kwargs.get("bookmakers"), ["sx_bet"])
        self.assertEqual(kwargs.get("include_providers"), ["sx_bet"])
        self.assertEqual(kwargs.get("sharp_book"), "pinnacle")

    def test_server_auto_scan_loop_idles_when_config_disabled(self) -> None:
        fake_event = _FakeStopEvent(wait_results=[True])
        disabled_config = {
            "enabled": False,
            "interval_minutes": 10,
            "payload": {"sports": ["icehockey_nhl"]},
        }

        with (
            patch.object(app_module, "_SERVER_AUTO_SCAN_STOP_EVENT", fake_event),
            patch.object(app_module, "_get_server_auto_scan_config", return_value=(disabled_config, 1)),
            patch.object(app_module, "_execute_scan_payload") as mocked_execute,
        ):
            app_module._server_auto_scan_loop()

        mocked_execute.assert_not_called()
        self.assertEqual(fake_event.wait_calls, [1.0])

    def test_execute_scan_payload_background_rejects_parallel_scan(self) -> None:
        app_module._SCAN_EXECUTION_LOCK.acquire()
        try:
            with (
                patch.object(app_module, "_start_background_provider_services") as mocked_start,
                patch.object(app_module, "run_scan") as mocked_run_scan,
            ):
                result = app_module._execute_scan_payload({"sports": ["icehockey_nhl"]}, background=True)
        finally:
            if app_module._SCAN_EXECUTION_LOCK.locked():
                app_module._SCAN_EXECUTION_LOCK.release()

        self.assertEqual(
            result,
            {
                "success": False,
                "error": "Scan already in progress",
                "error_code": 409,
            },
        )
        mocked_start.assert_not_called()
        mocked_run_scan.assert_not_called()

    def test_start_server_auto_scan_skips_thread_when_lease_is_held_elsewhere(self) -> None:
        with (
            patch.object(app_module, "ENV_SERVER_AUTO_SCAN_ENABLED", True),
            patch.object(app_module, "_SERVER_AUTO_SCAN_THREAD", None),
            patch.object(app_module, "_try_acquire_server_auto_scan_lease", return_value=False),
            patch.object(app_module.threading, "Thread") as mocked_thread,
        ):
            app_module._start_server_auto_scan()

        mocked_thread.assert_not_called()

    def test_refresh_server_auto_scan_config_from_disk_updates_memory_when_file_changes(self) -> None:
        current_config = {
            "enabled": False,
            "interval_minutes": 5,
            "payload": {"sports": ["icehockey_nhl"]},
        }
        updated_config = {
            "enabled": True,
            "interval_minutes": 15,
            "payload": {"sports": ["basketball_nba"]},
        }
        with (
            patch.object(app_module, "_server_auto_scan_config_mtime", return_value=200.0),
            patch.object(app_module, "_load_server_auto_scan_config", return_value=updated_config),
            patch.object(app_module, "_get_server_auto_scan_config", return_value=(current_config, 1)),
            patch.object(app_module, "_set_server_auto_scan_config") as mocked_set,
        ):
            refreshed_mtime = app_module._refresh_server_auto_scan_config_from_disk(100.0)

        self.assertEqual(refreshed_mtime, 200.0)
        mocked_set.assert_called_once_with(updated_config, persist=False)


if __name__ == "__main__":
    unittest.main()

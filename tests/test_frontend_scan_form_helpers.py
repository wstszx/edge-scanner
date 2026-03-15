from __future__ import annotations

import copy
import json
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import app as app_module


ROOT_DIR = Path(__file__).resolve().parents[1]
JS_TEST_PATH = Path("tests") / "frontend_scan_form_helpers.test.js"


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


class FrontendScanFormHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._server_auto_scan_config = copy.deepcopy(app_module._SERVER_AUTO_SCAN_CONFIG)
        self._server_auto_scan_version = app_module._SERVER_AUTO_SCAN_CONFIG_VERSION
        self._env_provider_only_mode = app_module.ENV_PROVIDER_ONLY_MODE

    def tearDown(self) -> None:
        app_module._SERVER_AUTO_SCAN_CONFIG = copy.deepcopy(self._server_auto_scan_config)
        app_module._SERVER_AUTO_SCAN_CONFIG_VERSION = self._server_auto_scan_version
        app_module.ENV_PROVIDER_ONLY_MODE = self._env_provider_only_mode

    def _run_helper(self, function_name: str, options: dict) -> dict:
        node_binary = shutil.which("node")
        self.assertIsNotNone(node_binary, "node is required to run frontend helper tests")
        script = (
            "const helpers = require('./static/scan_form_helpers.js');"
            "const fn = process.argv[1];"
            "const options = JSON.parse(process.argv[2]);"
            "process.stdout.write(JSON.stringify(helpers[fn](options)));"
        )
        result = subprocess.run(
            [node_binary, "-e", script, function_name, json.dumps(options)],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                f"frontend helper call failed for {function_name}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            self.fail(f"frontend helper returned invalid JSON: {exc}\nstdout:\n{result.stdout}")

    def test_node_frontend_helper_suite_passes(self) -> None:
        node_binary = shutil.which("node")
        self.assertIsNotNone(node_binary, "node is required to run frontend helper tests")

        result = subprocess.run(
            [node_binary, "--test", str(JS_TEST_PATH)],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                "frontend helper tests failed\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    def test_index_loads_scan_form_helper_script(self) -> None:
        client = app_module.app.test_client()
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("/static/scan_form_helpers.js", html)

    def test_helper_generated_auto_scan_payload_drives_background_scan(self) -> None:
        helper_payload = self._run_helper(
            "buildServerAutoScanConfigPayload",
            {
                "sports": ["icehockey_nhl"],
                "regions": ["us"],
                "useAllBookmakers": False,
                "allBookmakers": ["draftkings", "sx_bet", "betdex"],
                "checkedBookmakers": ["draftkings", "sx_bet"],
                "providerOnlyMode": True,
                "customProviderKeys": ["polymarket", "betdex", "sx_bet", "bookmaker_xyz"],
                "allSports": False,
                "allMarkets": False,
                "stake": 125,
                "commission": 2.5,
                "sharpBook": "Pinnacle",
                "minEdgePercent": 1.3,
                "bankroll": 3500,
                "kellyFraction": 0.35,
                "intervalMinutes": 7,
                "defaults": {
                    "allMarkets": False,
                    "sharpBook": "pinnacle",
                    "minEdgePercent": 1,
                    "bankroll": 1000,
                    "kellyFraction": 0.25,
                    "commission": 0,
                },
            },
        )
        fake_event = _FakeStopEvent()
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False

        with patch.object(app_module, "_persist_server_auto_scan_config"):
            response = self.client.post("/server-auto-scan-config", json=helper_payload)
        self.assertEqual(response.status_code, 200)

        def _fake_run_scan(**kwargs):
            fake_event.set()
            return {
                "success": True,
                "scan_time": "2026-03-14T08:15:00Z",
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
        self.assertEqual(kwargs.get("stake_amount"), 125.0)
        self.assertEqual(kwargs.get("commission_rate"), 0.025)
        self.assertEqual(kwargs.get("sharp_book"), "pinnacle")
        self.assertEqual(kwargs.get("min_edge_percent"), 1.3)
        self.assertEqual(kwargs.get("bankroll"), 3500.0)
        self.assertEqual(kwargs.get("kelly_fraction"), 0.35)

    def test_helper_generated_scan_payload_round_trips_through_scan_route(self) -> None:
        helper_payload = self._run_helper(
            "buildRunScanPayload",
            {
                "apiKey": "should-be-cleared",
                "sports": ["icehockey_nhl"],
                "regions": ["us"],
                "useAllBookmakers": False,
                "allBookmakers": ["draftkings", "sx_bet", "betdex"],
                "checkedBookmakers": ["draftkings", "sx_bet"],
                "providerOnlyMode": True,
                "customProviderKeys": ["polymarket", "betdex", "sx_bet", "bookmaker_xyz"],
                "commission": 3.5,
                "allSports": False,
                "allMarkets": True,
                "stake": 90,
                "sharpBook": "Pinnacle",
                "minEdgePercent": 1.1,
                "bankroll": 1800,
                "kellyFraction": 0.4,
                "defaults": {
                    "allMarkets": False,
                    "sharpBook": "pinnacle",
                    "minEdgePercent": 1,
                    "bankroll": 1000,
                    "kellyFraction": 0.25,
                    "commission": 0,
                },
            },
        )
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False

        with (
            patch.object(app_module, "_start_background_provider_services") as mocked_start,
            patch.object(app_module, "get_history_manager", return_value=history_manager),
            patch.object(app_module, "get_notifier", return_value=notifier),
            patch.object(app_module, "run_scan", return_value={"success": True}) as mocked_run_scan,
            patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True),
        ):
            response = self.client.post("/scan", json=helper_payload)

        self.assertEqual(response.status_code, 200)
        mocked_start.assert_called_once_with(wait_timeout=0.0)
        kwargs = mocked_run_scan.call_args.kwargs
        self.assertEqual(kwargs.get("api_key"), [])
        self.assertEqual(kwargs.get("scan_mode"), "prematch")
        self.assertEqual(kwargs.get("sports"), ["icehockey_nhl"])
        self.assertEqual(kwargs.get("regions"), ["us"])
        self.assertEqual(kwargs.get("bookmakers"), ["sx_bet"])
        self.assertEqual(kwargs.get("include_providers"), ["sx_bet"])
        self.assertEqual(kwargs.get("commission_rate"), 0.035)
        self.assertEqual(kwargs.get("stake_amount"), 90.0)
        self.assertEqual(kwargs.get("sharp_book"), "pinnacle")
        self.assertEqual(kwargs.get("min_edge_percent"), 1.1)
        self.assertEqual(kwargs.get("bankroll"), 1800.0)
        self.assertEqual(kwargs.get("kelly_fraction"), 0.4)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

import app as app_module


class ScanInputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._env_save_scan = app_module.ENV_SAVE_SCAN
        self._background_started = app_module._BACKGROUND_SERVICES_STARTED

    def tearDown(self) -> None:
        app_module.ENV_SAVE_SCAN = self._env_save_scan
        app_module._BACKGROUND_SERVICES_STARTED = self._background_started

    def test_scan_rejects_invalid_json_payload(self) -> None:
        response = self.client.post(
            "/scan",
            data='{"bad":',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertEqual(payload.get("error"), "Invalid JSON payload")
        self.assertEqual(payload.get("error_code"), 400)
        self.assertFalse(payload.get("success"))

    def test_scan_rejects_non_object_payload(self) -> None:
        response = self.client.post(
            "/scan",
            data='["not-an-object"]',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertEqual(payload.get("error"), "Scan payload must be a JSON object")
        self.assertEqual(payload.get("error_code"), 400)
        self.assertFalse(payload.get("success"))

    def test_scan_uses_default_sharp_book_for_non_string(self) -> None:
        with patch.object(app_module, "run_scan", return_value={"success": True}) as mocked_run_scan:
            response = self.client.post("/scan", json={"sharpBook": 123})
        self.assertEqual(response.status_code, 200)
        kwargs = mocked_run_scan.call_args.kwargs
        self.assertEqual(kwargs.get("sharp_book"), app_module.DEFAULT_SHARP_BOOK)
        self.assertEqual(kwargs.get("scan_mode"), "prematch")

    def test_scan_parses_boolean_strings_from_payload(self) -> None:
        app_module.ENV_SAVE_SCAN = True
        with patch.object(app_module, "run_scan", return_value={"success": True}) as mocked_run_scan:
            response = self.client.post(
                "/scan",
                json={
                    "saveScan": "false",
                    "allSports": "false",
                    "allMarkets": "false",
                    "includePurebet": "false",
                },
            )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertNotIn("scan_saved_path", payload)
        kwargs = mocked_run_scan.call_args.kwargs
        self.assertFalse(kwargs.get("all_sports"))
        self.assertFalse(kwargs.get("all_markets"))
        self.assertFalse(kwargs.get("include_purebet"))

    def test_scan_derives_include_providers_when_empty_list_is_sent(self) -> None:
        with patch.object(app_module, "run_scan", return_value={"success": True}) as mocked_run_scan:
            response = self.client.post(
                "/scan",
                json={
                    "bookmakers": ["SX Bet"],
                    "includeProviders": [],
                },
            )
        self.assertEqual(response.status_code, 200)
        kwargs = mocked_run_scan.call_args.kwargs
        self.assertEqual(kwargs.get("include_providers"), ["sx_bet"])

    def test_scan_accepts_live_mode_without_api_key(self) -> None:
        with patch.object(app_module, "run_scan", return_value={"success": True}) as mocked_run_scan:
            response = self.client.post(
                "/scan",
                json={
                    "scanMode": "live",
                    "sports": ["basketball_nba"],
                    "regions": ["us"],
                },
            )
        self.assertEqual(response.status_code, 200)
        kwargs = mocked_run_scan.call_args.kwargs
        self.assertEqual(kwargs.get("scan_mode"), "live")

    def test_scan_saves_history_from_nested_result_shape(self) -> None:
        result_payload = {
            "success": True,
            "scan_time": "2026-02-22T12:00:00Z",
            "arbitrage": {"opportunities": [{"event": "A vs B"}]},
            "middles": {"opportunities": [{"event": "C vs D"}]},
            "plus_ev": {"opportunities": [{"event": "E vs F"}]},
        }
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False
        with (
            patch.object(app_module, "run_scan", return_value=result_payload),
            patch.object(app_module, "get_history_manager", return_value=history_manager),
            patch.object(app_module, "get_notifier", return_value=notifier),
        ):
            response = self.client.post("/scan", json={})

        self.assertEqual(response.status_code, 200)
        history_manager.save_opportunities.assert_called_once()
        history_payload = history_manager.save_opportunities.call_args.args[0]
        self.assertEqual(len(history_payload.get("opportunities") or []), 1)
        self.assertEqual(len(history_payload.get("middles") or []), 1)
        self.assertEqual(len(history_payload.get("plus_ev") or []), 1)

    def test_index_prewarms_background_services(self) -> None:
        with patch.object(app_module, "_start_background_provider_services") as mocked_start:
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        mocked_start.assert_called_once_with(wait_timeout=0.0)

    def test_index_hides_soft_bookmakers_from_supported_platforms_list(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('value="sx_bet"', html)
        self.assertIn('value="pinnacle"', html)
        self.assertNotIn('value="draftkings"', html)
        self.assertNotIn('value="fanduel"', html)

    def test_scan_rejects_unsupported_bookmakers_only(self) -> None:
        with patch.object(app_module, "run_scan") as mocked_run_scan:
            response = self.client.post(
                "/scan",
                json={
                    "bookmakers": ["DraftKings", "FanDuel"],
                    "includeProviders": [],
                },
            )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))
        self.assertEqual(
            payload.get("error"),
            "No supported arbitrage platforms were selected",
        )
        mocked_run_scan.assert_not_called()

    def test_provider_runtime_returns_status_payload(self) -> None:
        runtime_payload = {
            "enabled": True,
            "started": True,
            "ready": True,
            "status": {"market_connected": True},
        }
        with (
            patch.object(app_module, "_start_background_provider_services") as mocked_start,
            patch.object(app_module, "_provider_runtime_status", return_value=runtime_payload),
        ):
            response = self.client.get("/provider-runtime/polymarket")
        self.assertEqual(response.status_code, 200)
        mocked_start.assert_called_once_with(wait_timeout=0.0)
        payload = response.get_json() or {}
        self.assertTrue(payload.get("success"))
        self.assertEqual(payload.get("provider_key"), "polymarket")
        self.assertTrue(payload.get("ready"))

    def test_provider_runtime_rejects_unknown_provider(self) -> None:
        with patch.object(app_module, "_start_background_provider_services") as mocked_start:
            response = self.client.get("/provider-runtime/unknown")
        self.assertEqual(response.status_code, 404)
        mocked_start.assert_called_once_with(wait_timeout=0.0)
        payload = response.get_json() or {}
        self.assertFalse(payload.get("success"))


if __name__ == "__main__":
    unittest.main()

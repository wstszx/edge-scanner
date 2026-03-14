from __future__ import annotations

import copy
import itertools
import unittest
from unittest.mock import MagicMock, patch

import app as app_module
import scanner


class _FakeStopEvent:
    def __init__(self) -> None:
        self._flag = False
        self.wait_calls: list[float] = []

    def is_set(self) -> bool:
        return self._flag

    def set(self) -> None:
        self._flag = True

    def clear(self) -> None:
        self._flag = False

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_calls.append(float(timeout or 0.0))
        return self._flag


class _FakeCacheModule:
    def __init__(self) -> None:
        self.active = False
        self.enable_count = 0
        self.disable_count = 0

    def enable_scan_cache(self) -> None:
        self.active = True
        self.enable_count += 1

    def disable_scan_cache(self) -> None:
        self.active = False
        self.disable_count += 1


class ScanStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._server_auto_scan_config = copy.deepcopy(app_module._SERVER_AUTO_SCAN_CONFIG)
        self._server_auto_scan_version = app_module._SERVER_AUTO_SCAN_CONFIG_VERSION
        self._env_provider_only_mode = app_module.ENV_PROVIDER_ONLY_MODE
        self._background_started = app_module._BACKGROUND_SERVICES_STARTED

    def tearDown(self) -> None:
        app_module._SERVER_AUTO_SCAN_CONFIG = copy.deepcopy(self._server_auto_scan_config)
        app_module._SERVER_AUTO_SCAN_CONFIG_VERSION = self._server_auto_scan_version
        app_module.ENV_PROVIDER_ONLY_MODE = self._env_provider_only_mode
        app_module._BACKGROUND_SERVICES_STARTED = self._background_started
        if app_module._SCAN_EXECUTION_LOCK.locked():
            app_module._SCAN_EXECUTION_LOCK.release()
        scanner._set_current_request_logger(None)
        with scanner._REQUEST_TRACE_LOCK:
            scanner._REQUEST_TRACE_ACTIVE.clear()

    def test_server_auto_scan_loop_survives_many_cycles_with_failures_and_exceptions(self) -> None:
        stop_event = _FakeStopEvent()
        execute_results = [
            {"success": True, "scan_time": "2026-03-14T08:00:00Z"},
            {"success": False, "error": "provider timeout"},
            RuntimeError("transport closed"),
        ]
        execute_count = {"value": 0}
        target_runs = 18
        config = {
            "enabled": True,
            "interval_minutes": 1,
            "payload": {
                "sports": ["icehockey_nhl"],
                "regions": ["us"],
                "bookmakers": ["sx_bet"],
                "includeProviders": ["sx_bet"],
            },
        }

        def _fake_execute(payload, *, save_scan_override=None, background=False):
            index = execute_count["value"]
            execute_count["value"] += 1
            if execute_count["value"] >= target_runs:
                stop_event.set()
            result = execute_results[index % len(execute_results)]
            if isinstance(result, Exception):
                raise result
            return dict(result)

        time_values = iter(1000.0 + 61.0 * index for index in range(200))

        with (
            patch.object(app_module, "_SERVER_AUTO_SCAN_STOP_EVENT", stop_event),
            patch.object(app_module, "_get_server_auto_scan_config", return_value=(config, 1)),
            patch.object(app_module, "_execute_scan_payload", side_effect=_fake_execute) as mocked_execute,
            patch.object(app_module, "ENV_SERVER_AUTO_SCAN_RUN_ON_START", True),
            patch.object(app_module, "time") as mocked_time,
        ):
            mocked_time.time.side_effect = lambda: next(time_values)
            app_module._server_auto_scan_loop()

        self.assertEqual(mocked_execute.call_count, target_runs)
        self.assertTrue(stop_event.is_set())
        self.assertFalse(app_module._SCAN_EXECUTION_LOCK.locked())

    def test_execute_scan_payload_releases_lock_and_saves_history_only_for_successes_across_many_runs(self) -> None:
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False
        payload = {
            "sports": ["icehockey_nhl"],
            "regions": ["us"],
            "bookmakers": ["sx_bet"],
            "includeProviders": ["sx_bet"],
            "stake": 100,
            "commission": 0,
        }
        success_count = 0
        raise_count = 0
        outcome_cycle = itertools.cycle(
            [
                {"success": True, "scan_time": "2026-03-14T08:00:00Z", "arbitrage": {"opportunities": []}, "middles": {"opportunities": []}, "plus_ev": {"opportunities": []}},
                {"success": False, "error": "scan failed"},
                RuntimeError("provider crashed"),
            ]
        )

        def _fake_run_scan(**kwargs):
            outcome = next(outcome_cycle)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with (
            patch.object(app_module, "_start_background_provider_services"),
            patch.object(app_module, "get_history_manager", return_value=history_manager),
            patch.object(app_module, "get_notifier", return_value=notifier),
            patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True),
            patch.object(app_module, "run_scan", side_effect=_fake_run_scan),
        ):
            for _ in range(24):
                try:
                    result = app_module._execute_scan_payload(dict(payload), background=False, save_scan_override=False)
                except RuntimeError:
                    raise_count += 1
                    result = None
                if isinstance(result, dict) and result.get("success"):
                    success_count += 1
                self.assertFalse(app_module._SCAN_EXECUTION_LOCK.locked())

        self.assertEqual(history_manager.save_opportunities.call_count, success_count)
        self.assertGreater(raise_count, 0)

    def test_run_scan_repeatedly_enables_and_disables_provider_scan_cache_without_leaking_state(self) -> None:
        fake_cache_module = _FakeCacheModule()
        fetch_calls = {"count": 0}

        async def _provider_fetcher(sport_key, markets, regions, bookmakers=None):
            fetch_calls["count"] += 1
            if fetch_calls["count"] % 5 == 0:
                raise RuntimeError("temporary upstream failure")
            _provider_fetcher.last_stats = {
                "provider": "sx_bet",
                "events_returned_count": 1,
            }
            return [
                {
                    "id": f"sx-event-{fetch_calls['count']}",
                    "sport_key": sport_key,
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "commence_time": "2026-03-15T10:00:00Z",
                    "bookmakers": [
                        {
                            "key": "sx_bet",
                            "title": "SX Bet",
                            "event_id": f"sx-event-{fetch_calls['count']}",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Home Team", "price": 2.05},
                                        {"name": "Away Team", "price": 1.96},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        _provider_fetcher.last_stats = {}

        def _fake_import_module(name: str):
            if name == "providers.sx_bet":
                return fake_cache_module
            raise ImportError(name)

        with (
            patch.dict(scanner.PROVIDER_FETCHERS, {"sx_bet": _provider_fetcher}, clear=False),
            patch.dict(scanner.PROVIDER_TITLES, {"sx_bet": "SX Bet"}, clear=False),
            patch.object(scanner.importlib, "import_module", side_effect=_fake_import_module),
            patch.object(scanner, "_persist_provider_snapshots", return_value={}),
            patch.object(scanner, "_persist_cross_provider_match_report", return_value=""),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
            patch.object(scanner, "_provider_fetch_max_workers", return_value=1),
        ):
            for _ in range(20):
                result = scanner.run_scan(
                    api_key="",
                    sports=["icehockey_nhl"],
                    regions=["us"],
                    bookmakers=["sx_bet"],
                    include_providers=["sx_bet"],
                    stake_amount=100.0,
                )
                self.assertTrue(result.get("success"))
                self.assertFalse(fake_cache_module.active)

        self.assertEqual(fake_cache_module.enable_count, 20)
        self.assertEqual(fake_cache_module.disable_count, 20)


if __name__ == "__main__":
    unittest.main()

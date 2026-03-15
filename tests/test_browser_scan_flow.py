from __future__ import annotations

import copy
import socket
import threading
import time
import unittest
from contextlib import ExitStack
from typing import Callable
from unittest.mock import MagicMock, patch

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

import app as app_module


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _LiveServer:
    def __init__(self, flask_app, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = _find_free_port()
        self._server = make_server(self.host, self.port, flask_app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=2.0)


def _sample_arbitrage_opportunity() -> dict:
    return {
        "event": "Seattle Kraken vs Vancouver Canucks",
        "event_id": "evt-1",
        "sport": "icehockey_nhl",
        "sport_display": "NHL",
        "market": "totals",
        "market_point": 6.5,
        "commence_time": "2026-03-15T10:00:00Z",
        "has_exchange": True,
        "best_odds": [
            {
                "outcome": "Over",
                "bookmaker": "SX Bet",
                "bookmaker_key": "sx_bet",
                "price": 2.37,
                "effective_price": 2.37,
                "point": 6.5,
                "max_stake": 50.0,
                "book_event_id": "sx-evt-1",
                "book_event_url": "https://sx.bet/markets/evt-1",
                "quote_source": "ws",
                "quote_updated_at": "2026-03-15T09:14:58Z",
            },
            {
                "outcome": "Under",
                "bookmaker": "BetDEX",
                "bookmaker_key": "betdex",
                "price": 1.86,
                "effective_price": 1.83,
                "point": 6.5,
                "max_stake": 12.37,
                "is_exchange": True,
                "book_event_id": "betdex-evt-1",
                "book_event_url": "https://betdex.com/markets/evt-1",
                "quote_source": "rest_snapshot",
                "quote_updated_at": "2026-03-15T09:14:54Z",
            },
        ],
    }


def _sample_scan_result(scan_mode: str = "prematch", opportunities: list[dict] | None = None) -> dict:
    arbitrage_items = copy.deepcopy(opportunities if opportunities is not None else [_sample_arbitrage_opportunity()])
    return {
        "success": True,
        "scan_mode": scan_mode,
        "scan_time": "2026-03-14T09:15:00Z",
        "arbitrage": {
            "opportunities": arbitrage_items,
            "opportunities_count": len(arbitrage_items),
        },
        "middles": {"opportunities": []},
        "plus_ev": {"opportunities": []},
        "custom_providers": {
            "sx_bet": {
                "key": "sx_bet",
                "name": "SX Bet",
                "enabled": True,
                "events_merged": 1,
                "sports": [
                    {
                        "sport_key": "icehockey_nhl",
                        "events_returned": 1,
                        "requested_markets": ["totals"],
                    }
                ],
            }
        },
        "provider_snapshot_paths": {
            "sx_bet": "data/provider_snapshots/sx_bet.json",
        },
    }


class BrowserScanFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._server_auto_scan_config = copy.deepcopy(app_module._SERVER_AUTO_SCAN_CONFIG)
        self._server_auto_scan_version = app_module._SERVER_AUTO_SCAN_CONFIG_VERSION
        self._env_provider_only_mode = app_module.ENV_PROVIDER_ONLY_MODE

    def tearDown(self) -> None:
        app_module._SERVER_AUTO_SCAN_CONFIG = copy.deepcopy(self._server_auto_scan_config)
        app_module._SERVER_AUTO_SCAN_CONFIG_VERSION = self._server_auto_scan_version
        app_module.ENV_PROVIDER_ONLY_MODE = self._env_provider_only_mode

    def _configure_selected_provider(self, page) -> None:
        page.click("#open-advanced")
        page.evaluate(
            """() => {
              const dispatchChange = (element) => {
                element.dispatchEvent(new Event('change', { bubbles: true }));
              };
              const selectSingle = (selector, value) => {
                document.querySelectorAll(selector).forEach((input) => {
                  input.checked = input.value === value;
                  dispatchChange(input);
                });
              };
              const allSports = document.getElementById('all-sports');
              const allBookmakers = document.getElementById('all-bookmakers');
              allSports.checked = false;
              dispatchChange(allSports);
              allBookmakers.checked = false;
              dispatchChange(allBookmakers);
              selectSingle('input[name="sports"]', 'icehockey_nhl');
              selectSingle('input[name="regions"]', 'us');
              selectSingle('input[name="bookmakers"]', 'sx_bet');
            }"""
        )
        page.wait_for_timeout(400)
        page.click("#close-advanced")

    def _wait_for(self, predicate: Callable[[], bool], timeout_seconds: float = 5.0) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail("timed out waiting for browser condition")

    def test_browser_manual_scan_renders_results_for_selected_provider(self) -> None:
        persisted_configs: list[dict] = []
        run_scan_calls: list[dict] = []
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False

        def _fake_run_scan(**kwargs):
            run_scan_calls.append(copy.deepcopy(kwargs))
            return _sample_scan_result()

        with ExitStack() as stack:
            stack.enter_context(patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True))
            stack.enter_context(
                patch.object(
                    app_module,
                    "_persist_server_auto_scan_config",
                    side_effect=lambda config: persisted_configs.append(copy.deepcopy(config)),
                )
            )
            stack.enter_context(patch.object(app_module, "_start_background_provider_services"))
            stack.enter_context(patch.object(app_module, "get_history_manager", return_value=history_manager))
            stack.enter_context(patch.object(app_module, "get_notifier", return_value=notifier))
            stack.enter_context(patch.object(app_module, "run_scan", side_effect=_fake_run_scan))

            server = _LiveServer(app_module.app)
            server.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    context = browser.new_context()
                    page = context.new_page()
                    page.route("https://fonts.googleapis.com/*", lambda route: route.abort())
                    page.route("https://fonts.gstatic.com/*", lambda route: route.abort())
                    page.goto(server.base_url, wait_until="domcontentloaded")

                    self._configure_selected_provider(page)
                    self._wait_for(lambda: any(config.get("payload", {}).get("bookmakers") == ["sx_bet"] for config in persisted_configs))

                    with page.expect_response(lambda response: response.url.endswith("/scan") and response.request.method == "POST"):
                        page.evaluate("document.getElementById('scan-form').requestSubmit()")

                    page.wait_for_selector("#arb-desktop-list .arb-opportunity-card")
                    self.assertEqual(page.locator("#arb-desktop-list .arb-opportunity-card").count(), 1)
                    self.assertEqual(page.locator("#arb-desktop-list .arb-opportunity-leg").count(), 2)
                    self.assertIn("Seattle Kraken vs Vancouver Canucks", page.locator("#arb-desktop-list").inner_text())
                    self.assertIn("SX Bet", page.locator("#arb-desktop-list").inner_text())
                    self.assertIn("BetDEX", page.locator("#arb-desktop-list").inner_text())
                    desktop_text = page.locator("#arb-desktop-list").inner_text()
                    self.assertIn("WS", desktop_text)
                    self.assertTrue("Snapshot" in desktop_text or "快照" in desktop_text)
                    self.assertIn("1", page.locator("#table-count").inner_text())
                    self.assertEqual(page.locator("#provider-data-list .provider-card").count(), 1)
                    self.assertIn("SX Bet", page.locator("#provider-data-list").inner_text())

                    browser.close()
            finally:
                server.stop()

        self.assertEqual(len(run_scan_calls), 1)
        kwargs = run_scan_calls[0]
        self.assertEqual(kwargs.get("api_key"), [])
        self.assertEqual(kwargs.get("sports"), ["icehockey_nhl"])
        self.assertEqual(kwargs.get("regions"), ["us"])
        self.assertEqual(kwargs.get("bookmakers"), ["sx_bet"])
        self.assertEqual(kwargs.get("include_providers"), ["sx_bet"])

    def test_browser_scan_mode_tabs_switch_arbitrage_results_by_mode(self) -> None:
        run_scan_calls: list[dict] = []
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False

        def _fake_run_scan(**kwargs):
            run_scan_calls.append(copy.deepcopy(kwargs))
            scan_mode = kwargs.get("scan_mode") or "prematch"
            if scan_mode == "live":
                return _sample_scan_result(scan_mode="live", opportunities=[])
            return _sample_scan_result(scan_mode="prematch")

        with ExitStack() as stack:
            stack.enter_context(patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True))
            stack.enter_context(patch.object(app_module, "_start_background_provider_services"))
            stack.enter_context(patch.object(app_module, "get_history_manager", return_value=history_manager))
            stack.enter_context(patch.object(app_module, "get_notifier", return_value=notifier))
            stack.enter_context(patch.object(app_module, "run_scan", side_effect=_fake_run_scan))

            server = _LiveServer(app_module.app)
            server.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    context = browser.new_context()
                    page = context.new_page()
                    page.route("https://fonts.googleapis.com/*", lambda route: route.abort())
                    page.route("https://fonts.gstatic.com/*", lambda route: route.abort())
                    page.goto(server.base_url, wait_until="domcontentloaded")

                    self._configure_selected_provider(page)

                    with page.expect_response(lambda response: response.url.endswith("/scan") and response.request.method == "POST"):
                        page.evaluate("document.getElementById('scan-form').requestSubmit()")

                    page.wait_for_selector("#arb-desktop-list .arb-opportunity-card")
                    self.assertEqual(page.locator("#arb-desktop-list .arb-opportunity-card").count(), 1)
                    self.assertIn("Seattle Kraken vs Vancouver Canucks", page.locator("#arb-desktop-list").inner_text())

                    page.click('.tab-btn[data-tab="live"]')
                    page.wait_for_timeout(150)
                    self.assertEqual(page.locator("#arb-desktop-list .arb-opportunity-card").count(), 0)
                    pre_scan_empty_title = page.locator("#arb-empty-title").inner_text().strip()
                    self.assertTrue(pre_scan_empty_title)

                    with page.expect_response(lambda response: response.url.endswith("/scan") and response.request.method == "POST"):
                        page.evaluate("document.getElementById('scan-form').requestSubmit()")

                    page.wait_for_timeout(250)
                    self.assertEqual(page.locator("#arb-desktop-list .arb-opportunity-card").count(), 0)
                    scan_empty_title = page.locator("#arb-empty-title").inner_text().strip()
                    self.assertTrue(scan_empty_title)
                    self.assertNotEqual(scan_empty_title, pre_scan_empty_title)

                    page.click('.tab-btn[data-tab="prematch"]')
                    page.wait_for_timeout(150)
                    self.assertEqual(page.locator("#arb-desktop-list .arb-opportunity-card").count(), 1)
                    self.assertIn("Seattle Kraken vs Vancouver Canucks", page.locator("#arb-desktop-list").inner_text())

                    page.click('.tab-btn[data-tab="live"]')
                    page.wait_for_timeout(150)
                    self.assertEqual(page.locator("#arb-desktop-list .arb-opportunity-card").count(), 0)
                    self.assertEqual(page.locator("#arb-empty-title").inner_text().strip(), scan_empty_title)

                    browser.close()
            finally:
                server.stop()

        self.assertEqual([call.get("scan_mode") for call in run_scan_calls], ["prematch", "live"])

    def test_browser_auto_scan_triggers_without_manual_submit(self) -> None:
        persisted_configs: list[dict] = []
        run_scan_calls: list[dict] = []
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False

        def _fake_run_scan(**kwargs):
            run_scan_calls.append(copy.deepcopy(kwargs))
            return _sample_scan_result()

        with ExitStack() as stack:
            stack.enter_context(patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True))
            stack.enter_context(
                patch.object(
                    app_module,
                    "_persist_server_auto_scan_config",
                    side_effect=lambda config: persisted_configs.append(copy.deepcopy(config)),
                )
            )
            stack.enter_context(patch.object(app_module, "_start_background_provider_services"))
            stack.enter_context(patch.object(app_module, "get_history_manager", return_value=history_manager))
            stack.enter_context(patch.object(app_module, "get_notifier", return_value=notifier))
            stack.enter_context(patch.object(app_module, "run_scan", side_effect=_fake_run_scan))

            server = _LiveServer(app_module.app)
            server.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    context = browser.new_context()
                    context.add_init_script(
                        """
                        (() => {
                          const nativeSetInterval = window.setInterval.bind(window);
                          window.setInterval = (callback, ms, ...args) => nativeSetInterval(callback, Math.min(ms, 50), ...args);
                        })();
                        """
                    )
                    page = context.new_page()
                    page.route("https://fonts.googleapis.com/*", lambda route: route.abort())
                    page.route("https://fonts.gstatic.com/*", lambda route: route.abort())
                    page.goto(server.base_url, wait_until="domcontentloaded")

                    self._configure_selected_provider(page)
                    page.click("#open-advanced")
                    page.check("#auto-scan-toggle")
                    page.fill("#auto-scan-interval", "1")
                    page.dispatch_event("#auto-scan-interval", "change")
                    page.click("#close-advanced")

                    self._wait_for(lambda: len(run_scan_calls) >= 1)
                    page.wait_for_selector("#arb-desktop-list .arb-opportunity-card")

                    self.assertGreaterEqual(len(run_scan_calls), 1)
                    self.assertEqual(page.locator("#arb-desktop-list .arb-opportunity-card").count(), 1)
                    self.assertIn("Seattle Kraken vs Vancouver Canucks", page.locator("#arb-desktop-list").inner_text())
                    self.assertIn("1", page.locator("#table-count").inner_text())
                    self.assertTrue(any(config.get("enabled") is True for config in persisted_configs))

                    browser.close()
            finally:
                server.stop()

        kwargs = run_scan_calls[0]
        self.assertEqual(kwargs.get("bookmakers"), ["sx_bet"])
        self.assertEqual(kwargs.get("include_providers"), ["sx_bet"])


if __name__ == "__main__":
    unittest.main()

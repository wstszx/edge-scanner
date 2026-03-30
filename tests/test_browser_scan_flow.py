from __future__ import annotations

import copy
import json
import socket
import threading
import time
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


def _sample_arbitrage_opportunity(over_price: float = 2.37) -> dict:
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
                "price": over_price,
                "effective_price": over_price,
                "point": 6.5,
                "max_stake": 50.0,
                "book_event_id": "L18053574",
                "book_event_url": "https://sx.bet/hockey/nhl/game-lines/L18053574",
                "quote_source": "ws",
                "quote_updated_at": "2026-03-15T09:14:58Z",
                "raw_percentage_odds": "50875000000000000000",
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


def _sample_scan_result(
    scan_mode: str = "prematch",
    opportunities: list[dict] | None = None,
    *,
    custom_providers: dict | None = None,
    cross_provider_match_report_path: str = "",
) -> dict:
    arbitrage_items = copy.deepcopy(opportunities if opportunities is not None else [_sample_arbitrage_opportunity()])
    by_sport: dict[str, int] = {}
    for item in arbitrage_items:
        sport_key = item.get("sport_display") or item.get("sport") or "unknown"
        by_sport[sport_key] = by_sport.get(sport_key, 0) + 1
    unique_events = {
        (item.get("event_id") or item.get("event") or "", item.get("commence_time") or "")
        for item in arbitrage_items
    }
    providers = copy.deepcopy(custom_providers) if custom_providers is not None else {
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
    }
    result = {
        "success": True,
        "scan_mode": scan_mode,
        "scan_time": "2026-03-14T09:15:00Z",
        "arbitrage": {
            "opportunities": arbitrage_items,
            "opportunities_count": len(arbitrage_items),
            "summary": {
                "positive_count": sum(1 for item in arbitrage_items if float(item.get("roi_percent") or 0) > 0),
                "by_sport": by_sport,
                "by_roi_band": {},
                "sports_scanned": len(by_sport),
                "events_scanned": len(unique_events),
                "api_calls_used": 0,
                "total_guaranteed_profit": 0.0,
            },
        },
        "middles": {"opportunities": [], "summary": {"count": 0, "positive_count": 0}},
        "plus_ev": {"opportunities": [], "summary": {"count": 0}},
        "custom_providers": providers,
        "provider_snapshot_paths": {
            str(key): f"data/provider_snapshots/{key}.json"
            for key in providers.keys()
        },
    }
    if cross_provider_match_report_path:
        result["cross_provider_match_report_path"] = cross_provider_match_report_path
    return result


def _sample_cross_provider_report() -> dict:
    return {
        "saved_at": "2026-03-14T09:15:00Z",
        "summary": {
            "providers_considered": ["sx_bet", "polymarket", "betdex"],
            "provider_event_counts": {
                "sx_bet": 3,
                "polymarket": 2,
                "betdex": 2,
            },
            "total_raw_records": 7,
            "total_match_clusters": 4,
            "overlap_clusters": 2,
            "clusters_by_provider_count": {"2": 2, "1": 2},
            "provider_cluster_presence": {
                "sx_bet": 3,
                "polymarket": 2,
                "betdex": 2,
            },
            "pair_overlap_clusters": {
                "polymarket__sx_bet": 1,
                "betdex__sx_bet": 1,
            },
            "single_provider_cluster_counts": {
                "sx_bet": 1,
                "betdex": 1,
            },
            "single_provider_reason_counts": {
                "same_pair_time_mismatch": 1,
                "possible_name_mismatch": 1,
            },
            "tolerance_minutes": 180,
            "event_match_tolerance_minutes": 15,
            "event_match_fuzzy_threshold": 0.72,
        },
        "clusters": [
            {
                "cluster_id": 1,
                "sport_key": "icehockey_nhl",
                "pair_norm": "seattle kraken vs vancouver canucks",
                "representative_time_utc": "2026-03-15T10:00:00Z",
                "providers": ["sx_bet", "polymarket"],
                "provider_count": 2,
                "events": [
                    {"provider": "sx_bet", "event_id": "sx-1", "markets_count": 2},
                    {"provider": "polymarket", "event_id": "poly-1", "markets_count": 1},
                ],
            },
            {
                "cluster_id": 2,
                "sport_key": "icehockey_nhl",
                "pair_norm": "edmonton oilers vs calgary flames",
                "representative_time_utc": "2026-03-15T12:30:00Z",
                "providers": ["betdex", "sx_bet"],
                "provider_count": 2,
                "events": [
                    {"provider": "sx_bet", "event_id": "sx-2", "markets_count": 1},
                    {"provider": "betdex", "event_id": "betdex-2", "markets_count": 1},
                ],
            },
        ],
        "single_provider_samples": [
            {
                "cluster_id": 3,
                "sport_key": "icehockey_nhl",
                "pair_norm": "new jersey devils vs new york rangers",
                "provider": "sx_bet",
                "event_id": "sx-near-1",
                "commence_time": "2026-03-16T01:00:00Z",
                "markets_count": 1,
                "reason_code": "same_pair_time_mismatch",
                "closest_candidate": {
                    "provider": "polymarket",
                    "event_id": "poly-near-1",
                    "pair_norm": "new jersey devils vs new york rangers",
                    "time_delta_minutes": 215,
                    "pair_similarity_percent": 100.0,
                },
            },
            {
                "cluster_id": 4,
                "sport_key": "icehockey_nhl",
                "pair_norm": "montreal canadiens vs ottawa senators",
                "provider": "betdex",
                "event_id": "betdex-near-1",
                "commence_time": "2026-03-16T03:00:00Z",
                "markets_count": 1,
                "reason_code": "possible_name_mismatch",
                "closest_candidate": {
                    "provider": "sx_bet",
                    "event_id": "sx-near-2",
                    "pair_norm": "montreal canadiens vs ottawa sens",
                    "time_delta_minutes": 12,
                    "pair_similarity_percent": 74.0,
                },
            },
        ],
    }


def _sample_provider_snapshot(provider_key: str, provider_name: str, events: list[dict]) -> dict:
    return {
        "saved_at": "2026-03-14T09:15:00Z",
        "provider_key": provider_key,
        "provider_name": provider_name,
        "sports": [{"sport_key": "icehockey_nhl", "events_returned": len(events)}],
        "events": copy.deepcopy(events),
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
                    self.assertTrue("Maker 50.88%" in desktop_text or "挂单方 50.88%" in desktop_text)
                    self.assertIn("1", page.locator("#table-count").inner_text())
                    summary_info_text = page.locator("#arb-summary-info").inner_text()
                    self.assertTrue(
                        "with 1 positive ROI" in summary_info_text
                        or "其中正 ROI 1" in summary_info_text
                    )
                    self.assertEqual(page.locator("#provider-data-list .provider-card").count(), 1)
                    self.assertIn("SX Bet", page.locator("#provider-data-list").inner_text())
                    self.assertEqual(
                        page.locator("#arb-desktop-list .arb-opportunity-leg a.arb-leg-open-link").first.get_attribute("href"),
                        "https://sx.bet/hockey/nhl/game-lines/L18053574/OVER_UNDER",
                    )

                    deep_links = page.evaluate(
                        """() => {
                          const sxBook = {
                            key: 'sx_bet',
                            name: 'SX Bet',
                            eventId: 'L18053574',
                            eventUrl: 'https://sx.bet/hockey/nhl/game-lines/L18053574',
                          };
                          return {
                            moneyline: resolveBookmakerOpenUrl(sxBook, { event_id: 'L18053574', market: 'h2h' }),
                            spread: resolveBookmakerOpenUrl(sxBook, { event_id: 'L18053574', market: 'spreads' }),
                            total: resolveBookmakerOpenUrl(sxBook, { event_id: 'L18053574', market: 'totals' }),
                            untouched: resolveBookmakerOpenUrl(
                              {
                                key: 'betdex',
                                name: 'BetDEX',
                                eventId: '20578',
                                eventUrl: 'https://www.betdex.com/events/icehky/nhl/20578?market=359445',
                              },
                              { event_id: '20578', market: 'totals' }
                            ),
                          };
                        }"""
                    )
                    self.assertEqual(
                        deep_links,
                        {
                            "moneyline": "https://sx.bet/hockey/nhl/game-lines/L18053574/MONEY_LINE",
                            "spread": "https://sx.bet/hockey/nhl/game-lines/L18053574/SPREAD",
                            "total": "https://sx.bet/hockey/nhl/game-lines/L18053574/OVER_UNDER",
                            "untouched": "https://www.betdex.com/events/icehky/nhl/20578?market=359445",
                        },
                    )

                    browser.close()
            finally:
                server.stop()

        self.assertEqual(len(run_scan_calls), 1)
        kwargs = run_scan_calls[0]
        self.assertEqual(kwargs.get("api_key"), [])
        self.assertEqual(kwargs.get("sports"), ["icehockey_nhl"])
        self.assertEqual(kwargs.get("regions"), ["eu"])
        self.assertEqual(kwargs.get("bookmakers"), ["sx_bet"])
        self.assertEqual(kwargs.get("include_providers"), ["sx_bet"])

    def test_provider_data_auto_loads_unmatched_provider_snapshot(self) -> None:
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False
        custom_providers = {
            "sx_bet": {
                "key": "sx_bet",
                "name": "SX Bet",
                "enabled": True,
                "events_merged": 2,
                "sports": [
                    {
                        "sport_key": "icehockey_nhl",
                        "events_returned": 2,
                        "requested_markets": ["h2h", "totals"],
                        "merge_stats": {
                            "matched_existing": 0,
                            "matched_fuzzy": 0,
                            "appended_new": 2,
                        },
                    }
                ],
            },
            "betdex": {
                "key": "betdex",
                "name": "BetDEX",
                "enabled": True,
                "events_merged": 1,
                "sports": [
                    {
                        "sport_key": "icehockey_nhl",
                        "events_returned": 1,
                        "requested_markets": ["h2h"],
                        "merge_stats": {
                            "matched_existing": 1,
                            "matched_fuzzy": 0,
                            "appended_new": 0,
                        },
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_paths = {
                "sx_bet": Path(temp_dir) / "sx_bet.json",
                "betdex": Path(temp_dir) / "betdex.json",
            }
            snapshot_paths["sx_bet"].write_text(
                json.dumps(
                    _sample_provider_snapshot(
                        "sx_bet",
                        "SX Bet",
                        [
                            {
                                "id": "sx-auto-1",
                                "sport_key": "icehockey_nhl",
                                "home_team": "Seattle Kraken",
                                "away_team": "Vancouver Canucks",
                                "commence_time": "2026-03-15T10:00:00Z",
                                "bookmakers": [
                                    {
                                        "key": "sx_bet",
                                        "title": "SX Bet",
                                        "event_id": "sx-auto-1",
                                        "markets": [{"key": "h2h"}, {"key": "totals"}],
                                    }
                                ],
                            }
                        ],
                    )
                ),
                encoding="utf-8",
            )
            snapshot_paths["betdex"].write_text(
                json.dumps(
                    _sample_provider_snapshot(
                        "betdex",
                        "BetDEX",
                        [
                            {
                                "id": "betdex-auto-1",
                                "sport_key": "icehockey_nhl",
                                "home_team": "Edmonton Oilers",
                                "away_team": "Calgary Flames",
                                "commence_time": "2026-03-15T12:30:00Z",
                                "bookmakers": [
                                    {
                                        "key": "betdex",
                                        "title": "BetDEX",
                                        "event_id": "betdex-auto-1",
                                        "markets": [{"key": "h2h"}],
                                    }
                                ],
                            }
                        ],
                    )
                ),
                encoding="utf-8",
            )

            def _fake_run_scan(**kwargs):
                return _sample_scan_result(custom_providers=custom_providers)

            with ExitStack() as stack:
                stack.enter_context(patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True))
                stack.enter_context(patch.object(app_module, "_start_background_provider_services"))
                stack.enter_context(patch.object(app_module, "get_history_manager", return_value=history_manager))
                stack.enter_context(patch.object(app_module, "get_notifier", return_value=notifier))
                stack.enter_context(
                    patch.object(
                        app_module,
                        "_provider_snapshot_path",
                        side_effect=lambda provider_key: snapshot_paths.get(str(provider_key)),
                    )
                )
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

                        with page.expect_response(
                            lambda response: response.url.endswith("/scan")
                            and response.request.method == "POST"
                        ):
                            page.evaluate("document.getElementById('scan-form').requestSubmit()")

                        page.wait_for_selector("#arb-desktop-list .arb-opportunity-card")
                        page.click('button[data-tab="providers"]')
                        page.wait_for_selector("#provider-data-list .provider-card")

                        review_summary_text = page.locator("#provider-data-review-summary").inner_text()
                        self.assertIn("SX Bet", review_summary_text)
                        self.assertTrue(
                            "Top review" in review_summary_text
                            or "首要排查" in review_summary_text
                        )
                        self.assertTrue(
                            "Needs Match Review" in review_summary_text
                            or "待查匹配" in review_summary_text
                        )
                        sx_card_class = (
                            page.locator('details[data-provider-key="sx_bet"]').get_attribute("class") or ""
                        )
                        self.assertIn("is-primary-review", sx_card_class)
                        betdex_card_class = (
                            page.locator('details[data-provider-key="betdex"]').get_attribute("class") or ""
                        )
                        self.assertNotIn("is-primary-review", betdex_card_class)

                        sx_snapshot_items = page.locator(
                            'details[data-provider-key="sx_bet"] [data-provider-json="sx_bet"] .provider-json-item'
                        )
                        self._wait_for(lambda: sx_snapshot_items.count() > 0)
                        self.assertTrue(
                            page.locator('details[data-provider-key="sx_bet"]').evaluate("el => el.open")
                        )
                        sx_card_text = page.locator('details[data-provider-key="sx_bet"]').inner_text()
                        self.assertTrue(
                            "Needs Match Review" in sx_card_text
                            or "待查匹配" in sx_card_text
                        )
                        self.assertTrue(
                            "did not merge into shared matches" in sx_card_text
                            or "没有并入任何共享匹配" in sx_card_text
                        )

                        self.assertTrue(
                            "Suggested checks" in sx_card_text
                            or "建议先查" in sx_card_text
                        )
                        self.assertIn("Cross-Provider Match Report", sx_card_text)

                        page.eval_on_selector(
                            'details[data-provider-key="sx_bet"]',
                            "el => { el.open = true; }",
                        )
                        self.assertIn(
                            'provider_key: "sx_bet"',
                            sx_card_text,
                        )

                        sx_button_text = page.locator(
                            'details[data-provider-key="sx_bet"] button[data-provider-key="sx_bet"]'
                        ).inner_text()
                        self.assertTrue(
                            "Reload Snapshot" in sx_button_text
                            or "重新加载" in sx_button_text
                        )

                        self.assertFalse(
                            page.locator('details[data-provider-key="betdex"]').evaluate("el => el.open")
                        )
                        page.eval_on_selector(
                            'details[data-provider-key="betdex"]',
                            "el => { el.open = true; }",
                        )
                        betdex_button_text = page.locator(
                            'details[data-provider-key="betdex"] button[data-provider-key="betdex"]'
                        ).inner_text()
                        self.assertTrue(
                            "Load Snapshot" in betdex_button_text
                            or "加载原始数据" in betdex_button_text
                        )
                        self.assertEqual(
                            page.locator(
                                'details[data-provider-key="betdex"] [data-provider-json="betdex"] .provider-json-item'
                            ).count(),
                            0,
                        )

                        browser.close()
                finally:
                    server.stop()

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
                          const nativeSetTimeout = window.setTimeout.bind(window);
                          const nativeSetInterval = window.setInterval.bind(window);
                          window.setTimeout = (callback, ms, ...args) => nativeSetTimeout(callback, Math.min(ms, 50), ...args);
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
                    page.fill("#auto-scan-interval", "0")
                    page.dispatch_event("#auto-scan-interval", "change")
                    page.click("#close-advanced")

                    self._wait_for(lambda: len(run_scan_calls) >= 2)
                    page.wait_for_selector("#arb-desktop-list .arb-opportunity-card")

                    self.assertGreaterEqual(len(run_scan_calls), 2)
                    self.assertEqual(page.locator("#arb-desktop-list .arb-opportunity-card").count(), 1)
                    self.assertIn("Seattle Kraken vs Vancouver Canucks", page.locator("#arb-desktop-list").inner_text())
                    self.assertIn("1", page.locator("#table-count").inner_text())
                    self.assertTrue(any(config.get("enabled") is True for config in persisted_configs))
                    self.assertTrue(any(config.get("interval_minutes") == 0 for config in persisted_configs))

                    browser.close()
            finally:
                server.stop()

    def test_browser_provider_report_filters_by_pair_and_reason(self) -> None:
        persisted_configs: list[dict] = []
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False
        provider_report = _sample_cross_provider_report()
        custom_providers = {
            "sx_bet": {
                "key": "sx_bet",
                "name": "SX Bet",
                "enabled": True,
                "events_merged": 3,
                "sports": [{"sport_key": "icehockey_nhl", "events_returned": 3}],
            },
            "polymarket": {
                "key": "polymarket",
                "name": "Polymarket",
                "enabled": True,
                "events_merged": 2,
                "sports": [{"sport_key": "icehockey_nhl", "events_returned": 2}],
            },
            "betdex": {
                "key": "betdex",
                "name": "BetDEX",
                "enabled": True,
                "events_merged": 2,
                "sports": [{"sport_key": "icehockey_nhl", "events_returned": 2}],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "cross_provider_match_report.json"
            report_path.write_text(json.dumps(provider_report), encoding="utf-8")
            snapshot_paths = {
                "sx_bet": Path(temp_dir) / "sx_bet.json",
                "polymarket": Path(temp_dir) / "polymarket.json",
                "betdex": Path(temp_dir) / "betdex.json",
            }
            snapshot_paths["sx_bet"].write_text(
                json.dumps(
                    _sample_provider_snapshot(
                        "sx_bet",
                        "SX Bet",
                        [
                            {
                                "id": "sx-near-1",
                                "sport_key": "icehockey_nhl",
                                "home_team": "New Jersey Devils",
                                "away_team": "New York Rangers",
                                "commence_time": "2026-03-16T01:00:00Z",
                                "bookmakers": [
                                    {
                                        "key": "sx_bet",
                                        "title": "SX Bet",
                                        "event_id": "sx-near-1",
                                        "markets": [{"key": "h2h"}, {"key": "totals"}],
                                    }
                                ],
                            },
                            {
                                "id": "sx-near-2",
                                "sport_key": "icehockey_nhl",
                                "home_team": "Montreal Canadiens",
                                "away_team": "Ottawa Sens",
                                "commence_time": "2026-03-16T03:12:00Z",
                                "bookmakers": [
                                    {
                                        "key": "sx_bet",
                                        "title": "SX Bet",
                                        "event_id": "sx-near-2",
                                        "markets": [{"key": "h2h"}],
                                    }
                                ],
                            },
                        ],
                    )
                ),
                encoding="utf-8",
            )
            snapshot_paths["polymarket"].write_text(
                json.dumps(
                    _sample_provider_snapshot(
                        "polymarket",
                        "Polymarket",
                        [
                            {
                                "id": "poly-near-1",
                                "sport_key": "icehockey_nhl",
                                "home_team": "New Jersey Devils",
                                "away_team": "New York Rangers",
                                "commence_time": "2026-03-16T04:35:00Z",
                                "bookmakers": [
                                    {
                                        "key": "polymarket",
                                        "title": "Polymarket",
                                        "event_id": "poly-near-1",
                                        "markets": [{"key": "h2h"}],
                                    }
                                ],
                            }
                        ],
                    )
                ),
                encoding="utf-8",
            )
            snapshot_paths["betdex"].write_text(
                json.dumps(
                    _sample_provider_snapshot(
                        "betdex",
                        "BetDEX",
                        [
                            {
                                "id": "betdex-near-1",
                                "sport_key": "icehockey_nhl",
                                "home_team": "Montreal Canadiens",
                                "away_team": "Ottawa Senators",
                                "commence_time": "2026-03-16T03:00:00Z",
                                "bookmakers": [
                                    {
                                        "key": "betdex",
                                        "title": "BetDEX",
                                        "event_id": "betdex-near-1",
                                        "markets": [{"key": "h2h"}],
                                    }
                                ],
                            }
                        ],
                    )
                ),
                encoding="utf-8",
            )

            def _fake_run_scan(**kwargs):
                return _sample_scan_result(
                    custom_providers=custom_providers,
                    cross_provider_match_report_path=str(report_path),
                )

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
                stack.enter_context(patch.object(app_module, "_cross_provider_report_path", return_value=report_path))
                stack.enter_context(
                    patch.object(
                        app_module,
                        "_provider_snapshot_path",
                        side_effect=lambda provider_key: snapshot_paths.get(str(provider_key)),
                    )
                )
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
                        self._wait_for(
                            lambda: any(
                                config.get("payload", {}).get("bookmakers") == ["sx_bet"]
                                for config in persisted_configs
                            )
                        )

                        with page.expect_response(
                            lambda response: response.url.endswith("/scan")
                            and response.request.method == "POST"
                        ):
                            page.evaluate("document.getElementById('scan-form').requestSubmit()")

                        page.wait_for_selector("#arb-desktop-list .arb-opportunity-card")
                        page.click('button[data-tab="providers"]')
                        page.wait_for_selector("#provider-match-content")

                        self.assertEqual(page.locator("#provider-match-table tbody tr").count(), 2)
                        self.assertEqual(page.locator("#provider-single-table tbody tr").count(), 2)
                        self.assertIn("SX Bet", page.locator("#provider-match-table").inner_text())
                        self.assertIn("Polymarket", page.locator("#provider-match-table").inner_text())

                        page.select_option("#provider-match-provider-filter", "betdex")
                        self._wait_for(
                            lambda: "1/2" in page.locator("#provider-match-count").inner_text()
                            and "1/2" in page.locator("#provider-single-count").inner_text()
                        )
                        self.assertEqual(page.locator("#provider-match-table tbody tr").count(), 1)
                        self.assertEqual(page.locator("#provider-single-table tbody tr").count(), 1)
                        provider_filter_summary = page.locator("#provider-match-summary").inner_text()
                        self.assertIn("BetDEX", provider_filter_summary)
                        page.wait_for_selector("#provider-compare-content")
                        focused_compare_text = page.locator("#provider-compare-content").inner_text()
                        self.assertIn("BetDEX", focused_compare_text)
                        self.assertIn("SX Bet", focused_compare_text)
                        self.assertIn("betdex-near-1", focused_compare_text)
                        self.assertIn("sx-near-2", focused_compare_text)
                        self.assertTrue(
                            "Likely Team-Name Normalization Issue" in focused_compare_text
                            or "更像队名归一化问题" in focused_compare_text
                        )

                        page.select_option("#provider-match-provider-filter", "all")
                        self._wait_for(
                            lambda: page.locator("#provider-match-table tbody tr").count() == 2
                            and page.locator("#provider-single-table tbody tr").count() == 2
                        )

                        page.select_option("#provider-match-pair-filter", "polymarket__sx_bet")
                        self._wait_for(
                            lambda: "1/2" in page.locator("#provider-match-count").inner_text()
                            and "1/2" in page.locator("#provider-single-count").inner_text()
                        )
                        self.assertEqual(page.locator("#provider-match-table tbody tr").count(), 1)
                        self.assertEqual(page.locator("#provider-single-table tbody tr").count(), 1)
                        single_text = page.locator("#provider-single-table").inner_text()
                        self.assertTrue(
                            "Same teams, but kickoff window differs" in single_text
                            or "队名一致，但开赛时间窗口不一致" in single_text
                        )

                        page.locator("#provider-single-table tbody tr").first.click()
                        page.wait_for_selector("#provider-compare-content")
                        compare_text = page.locator("#provider-compare-content").inner_text()
                        self.assertIn("SX Bet", compare_text)
                        self.assertIn("Polymarket", compare_text)
                        self.assertIn("sx-near-1", compare_text)
                        self.assertIn("poly-near-1", compare_text)
                        self.assertIn("EVENT_TIME_TOLERANCE_MINUTES", compare_text)
                        self.assertIn("CROSS_PROVIDER_MATCH_TOLERANCE_MINUTES", compare_text)
                        self.assertIn("0.72", compare_text)
                        self.assertTrue("15m" in compare_text or "180m" in compare_text)
                        self.assertTrue(
                            "Likely Kickoff-Window Mismatch" in compare_text
                            or "更像开赛时间窗口不匹配" in compare_text
                        )
                        self.assertGreater(page.locator(".provider-compare-diff-row.is-different").count(), 0)
                        self.assertGreater(page.locator(".provider-compare-status-badge").count(), 0)

                        page.select_option("#provider-match-reason-filter", "possible_name_mismatch")
                        self._wait_for(
                            lambda: "No near misses match the current filters."
                            in page.locator("#provider-single-table").inner_text()
                            or "当前筛选下没有近似未匹配样本。"
                            in page.locator("#provider-single-table").inner_text()
                        )
                        self.assertEqual(page.locator("#provider-single-table tbody tr").count(), 1)
                        filtered_text = page.locator("#provider-single-table").inner_text()
                        self.assertTrue(
                            "No near misses match the current filters." in filtered_text
                            or "当前筛选下没有近似未匹配样本。" in filtered_text
                        )

                        browser.close()
                finally:
                    server.stop()

    def test_scan_diagnostics_surfaces_likely_matching_issue(self) -> None:
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False
        provider_report = {
            "saved_at": "2026-03-14T09:15:00Z",
            "summary": {
                "providers_considered": ["sx_bet", "polymarket"],
                "provider_event_counts": {
                    "sx_bet": 2,
                    "polymarket": 2,
                },
                "total_raw_records": 4,
                "total_match_clusters": 2,
                "overlap_clusters": 0,
                "clusters_by_provider_count": {"1": 2},
                "provider_cluster_presence": {
                    "sx_bet": 2,
                    "polymarket": 2,
                },
                "pair_overlap_clusters": {},
                "single_provider_cluster_counts": {
                    "sx_bet": 1,
                    "polymarket": 1,
                },
                "single_provider_reason_counts": {
                    "same_pair_time_mismatch": 2,
                },
                "tolerance_minutes": 180,
                "event_match_tolerance_minutes": 15,
                "event_match_fuzzy_threshold": 0.72,
            },
            "clusters": [],
            "single_provider_samples": [
                {
                    "cluster_id": 1,
                    "sport_key": "icehockey_nhl",
                    "pair_norm": "new jersey devils vs new york rangers",
                    "provider": "sx_bet",
                    "event_id": "sx-near-1",
                    "commence_time": "2026-03-16T01:00:00Z",
                    "markets_count": 1,
                    "reason_code": "same_pair_time_mismatch",
                    "closest_candidate": {
                        "provider": "polymarket",
                        "event_id": "poly-near-1",
                        "pair_norm": "new jersey devils vs new york rangers",
                        "time_delta_minutes": 215,
                        "pair_similarity_percent": 100.0,
                    },
                }
            ],
        }
        custom_providers = {
            "sx_bet": {
                "key": "sx_bet",
                "name": "SX Bet",
                "enabled": True,
                "events_merged": 2,
                "sports": [{"sport_key": "icehockey_nhl", "events_returned": 2}],
            },
            "polymarket": {
                "key": "polymarket",
                "name": "Polymarket",
                "enabled": True,
                "events_merged": 2,
                "sports": [{"sport_key": "icehockey_nhl", "events_returned": 2}],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "cross_provider_match_report.json"
            report_path.write_text(json.dumps(provider_report), encoding="utf-8")

            def _fake_run_scan(**kwargs):
                result = _sample_scan_result(
                    opportunities=[],
                    custom_providers=custom_providers,
                    cross_provider_match_report_path=str(report_path),
                )
                result["scan_diagnostics"] = {
                    **(result.get("scan_diagnostics") or {}),
                    "reason_code": "low_merge_overlap",
                    "arbitrage_count": 2,
                    "positive_arbitrage_count": 0,
                    "middle_count": 1,
                    "positive_middle_count": 0,
                }
                return result

            with ExitStack() as stack:
                stack.enter_context(patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True))
                stack.enter_context(patch.object(app_module, "_start_background_provider_services"))
                stack.enter_context(patch.object(app_module, "get_history_manager", return_value=history_manager))
                stack.enter_context(patch.object(app_module, "get_notifier", return_value=notifier))
                stack.enter_context(patch.object(app_module, "_cross_provider_report_path", return_value=report_path))
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

                        with page.expect_response(
                            lambda response: response.url.endswith("/scan")
                            and response.request.method == "POST"
                        ):
                            page.evaluate("document.getElementById('scan-form').requestSubmit()")

                        page.click('button[data-tab="providers"]')
                        page.wait_for_selector("#provider-match-content")
                        self._wait_for(
                            lambda: "Likely Matching Issue" in page.locator("#scan-diagnostics-likely").inner_text()
                            or "更像赛事匹配问题" in page.locator("#scan-diagnostics-likely").inner_text()
                        )

                        likely_text = page.locator("#scan-diagnostics-likely").inner_text()
                        diagnostics_summary_text = page.locator("#scan-diagnostics-summary").inner_text()
                        self.assertTrue(
                            "Likely Matching Issue" in likely_text
                            or "更像赛事匹配问题" in likely_text
                        )
                        self.assertTrue(
                            "Same teams, but kickoff window differs" in likely_text
                            or "队名一致，但开赛时间窗口不一致" in likely_text
                        )
                        self.assertTrue(
                            "Arbitrage: 2" in diagnostics_summary_text
                            or "套利：2" in diagnostics_summary_text
                        )
                        self.assertTrue(
                            "Middles: 1" in diagnostics_summary_text
                            or "中间盘：1" in diagnostics_summary_text
                        )

                        browser.close()
                finally:
                    server.stop()

    def test_scan_diagnostics_surfaces_positive_middle_result(self) -> None:
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False
        provider_report = {
            "saved_at": "2026-03-14T09:15:00Z",
            "summary": {
                "providers_considered": ["sx_bet", "betdex"],
                "provider_event_counts": {
                    "sx_bet": 2,
                    "betdex": 2,
                },
                "total_raw_records": 4,
                "total_match_clusters": 2,
                "overlap_clusters": 1,
                "clusters_by_provider_count": {"2": 1, "1": 1},
                "provider_cluster_presence": {
                    "sx_bet": 2,
                    "betdex": 2,
                },
                "pair_overlap_clusters": {"betdex|sx_bet": 1},
                "single_provider_cluster_counts": {
                    "sx_bet": 1,
                    "betdex": 0,
                },
                "single_provider_reason_counts": {},
                "tolerance_minutes": 180,
                "event_match_tolerance_minutes": 15,
                "event_match_fuzzy_threshold": 0.72,
            },
            "clusters": [],
            "single_provider_samples": [],
        }
        custom_providers = {
            "sx_bet": {
                "key": "sx_bet",
                "name": "SX Bet",
                "enabled": True,
                "events_merged": 2,
                "sports": [{"sport_key": "icehockey_nhl", "events_returned": 2}],
            },
            "betdex": {
                "key": "betdex",
                "name": "BetDEX",
                "enabled": True,
                "events_merged": 2,
                "sports": [{"sport_key": "icehockey_nhl", "events_returned": 2}],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "cross_provider_match_report.json"
            report_path.write_text(json.dumps(provider_report), encoding="utf-8")

            def _fake_run_scan(**kwargs):
                result = _sample_scan_result(
                    opportunities=[],
                    custom_providers=custom_providers,
                    cross_provider_match_report_path=str(report_path),
                )
                result["middles"] = {
                    "opportunities": [
                        {
                            "event": "New Jersey Devils vs New York Rangers",
                            "sport": "Ice Hockey",
                            "market": "totals",
                            "total_line_a": 5.5,
                            "total_line_b": 6.5,
                            "expected_value": 0.8,
                        }
                    ],
                    "opportunities_count": 1,
                    "summary": {"count": 1, "positive_count": 1},
                }
                result["scan_diagnostics"] = {
                    "reason_code": "positive_middle_found",
                    "enabled_provider_count": 2,
                    "providers_with_events": 2,
                    "providers_with_errors": 0,
                    "providers_with_match_hits": 2,
                    "raw_provider_events": 4,
                    "merged_events_scanned": 2,
                    "total_match_clusters": 2,
                    "overlap_clusters": 1,
                    "single_provider_clusters": 1,
                    "total_merge_hits": 2,
                    "total_fuzzy_matches": 0,
                    "total_new_events": 2,
                    "arbitrage_count": 0,
                    "positive_arbitrage_count": 0,
                    "middle_count": 1,
                    "positive_middle_count": 1,
                    "plus_ev_count": 0,
                    "sport_error_count": 0,
                    "stale_filter_drop_total": 0,
                    "provider_breakdown": [
                        {
                            "provider_key": "sx_bet",
                            "provider_name": "SX Bet",
                            "enabled": True,
                            "raw_events": 2,
                            "matched_existing": 1,
                            "matched_fuzzy": 0,
                            "appended_new": 1,
                            "error_count": 0,
                        },
                        {
                            "provider_key": "betdex",
                            "provider_name": "BetDEX",
                            "enabled": True,
                            "raw_events": 2,
                            "matched_existing": 1,
                            "matched_fuzzy": 0,
                            "appended_new": 1,
                            "error_count": 0,
                        },
                    ],
                }
                return result

            with ExitStack() as stack:
                stack.enter_context(patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True))
                stack.enter_context(patch.object(app_module, "_start_background_provider_services"))
                stack.enter_context(patch.object(app_module, "get_history_manager", return_value=history_manager))
                stack.enter_context(patch.object(app_module, "get_notifier", return_value=notifier))
                stack.enter_context(patch.object(app_module, "_cross_provider_report_path", return_value=report_path))
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

                        with page.expect_response(
                            lambda response: response.url.endswith("/scan")
                            and response.request.method == "POST"
                        ):
                            page.evaluate("document.getElementById('scan-form').requestSubmit()")

                        page.click('button[data-tab="providers"]')
                        page.wait_for_selector("#provider-match-content")
                        self._wait_for(
                            lambda: "Positive Middle Confirmed" in page.locator("#scan-diagnostics-likely").inner_text()
                            or "已确认存在正 EV 中间盘" in page.locator("#scan-diagnostics-likely").inner_text()
                        )

                        likely_text = page.locator("#scan-diagnostics-likely").inner_text()
                        message_text = page.locator("#scan-diagnostics-message").inner_text()
                        diagnostics_summary_text = page.locator("#scan-diagnostics-summary").inner_text()
                        self.assertTrue(
                            "Positive Middle Confirmed" in likely_text
                            or "已确认存在正 EV 中间盘" in likely_text
                        )
                        self.assertTrue(
                            "positive-EV middle was produced" in likely_text
                            or "至少一个正 EV 中间盘机会" in likely_text
                        )
                        self.assertTrue(
                            "positive-EV middle opportunities were found" in message_text
                            or "已经找到了正 EV 中间盘机会" in message_text
                        )
                        self.assertTrue(
                            "Positive middle EV: 1" in diagnostics_summary_text
                            or "正 EV 中间盘：1" in diagnostics_summary_text
                        )

                        browser.close()
                finally:
                    server.stop()

    def test_history_tab_shows_scan_summaries_and_opportunity_log(self) -> None:
        history_manager = MagicMock()
        now_utc = datetime.now(timezone.utc).replace(microsecond=0)
        matching_scan_dt = now_utc - timedelta(hours=1)
        live_scan_dt = now_utc - timedelta(hours=2)
        older_scan_dt = now_utc - timedelta(days=9)
        local_tz = datetime.now().astimezone().tzinfo or timezone.utc

        def _iso_z(value: datetime) -> str:
            return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        def _display_time(value: datetime) -> str:
            return value.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S")

        history_manager.load_recent_scan_summaries.return_value = [
            {
                "scan_time": _iso_z(matching_scan_dt),
                "scan_mode": "prematch",
                "partial": False,
                "arbitrage_count": 0,
                "positive_arbitrage_count": 0,
                "middle_count": 0,
                "positive_middle_count": 0,
                "plus_ev_count": 1,
                "scan_diagnostics": {
                    "reason_code": "low_merge_overlap",
                    "enabled_provider_count": 2,
                    "providers_with_events": 2,
                    "raw_provider_events": 4,
                    "merged_events_scanned": 0,
                    "overlap_clusters": 0,
                    "single_provider_clusters": 2,
                    "total_merge_hits": 0,
                    "sport_error_count": 0,
                    "stale_filter_drop_total": 0,
                    "provider_breakdown": [
                        {
                            "provider_key": "betmgm",
                            "provider_name": "BetMGM",
                            "enabled": True,
                            "raw_events": 2,
                            "matched_existing": 0,
                            "matched_fuzzy": 0,
                            "error_count": 0,
                        },
                        {
                            "provider_key": "fanduel",
                            "provider_name": "FanDuel",
                            "enabled": True,
                            "raw_events": 2,
                            "matched_existing": 0,
                            "matched_fuzzy": 0,
                            "error_count": 0,
                        },
                    ],
                },
                "cross_provider_match_report_summary": {
                    "total_raw_records": 4,
                    "total_match_clusters": 2,
                    "overlap_clusters": 0,
                    "single_provider_reason_counts": {
                        "same_pair_time_mismatch": 2,
                    },
                },
            },
            {
                "scan_time": _iso_z(live_scan_dt),
                "scan_mode": "live",
                "partial": False,
                "arbitrage_count": 2,
                "positive_arbitrage_count": 1,
                "middle_count": 1,
                "positive_middle_count": 1,
                "plus_ev_count": 0,
                "scan_diagnostics": {
                    "reason_code": "arbitrage_found",
                    "enabled_provider_count": 2,
                    "providers_with_events": 2,
                    "raw_provider_events": 6,
                    "merged_events_scanned": 3,
                    "overlap_clusters": 2,
                    "single_provider_clusters": 1,
                    "total_merge_hits": 3,
                    "sport_error_count": 0,
                    "stale_filter_drop_total": 0,
                    "provider_breakdown": [
                        {
                            "provider_key": "polymarket",
                            "provider_name": "Polymarket",
                            "enabled": True,
                            "raw_events": 3,
                            "matched_existing": 2,
                            "matched_fuzzy": 1,
                            "error_count": 0,
                        },
                        {
                            "provider_key": "sxbet",
                            "provider_name": "SX Bet",
                            "enabled": True,
                            "raw_events": 3,
                            "matched_existing": 1,
                            "matched_fuzzy": 0,
                            "error_count": 0,
                        },
                    ],
                },
                "cross_provider_match_report_summary": {
                    "total_raw_records": 6,
                    "total_match_clusters": 3,
                    "overlap_clusters": 2,
                    "single_provider_reason_counts": {},
                },
            },
            {
                "scan_time": _iso_z(older_scan_dt),
                "scan_mode": "prematch",
                "partial": True,
                "arbitrage_count": 0,
                "positive_arbitrage_count": 0,
                "middle_count": 0,
                "positive_middle_count": 0,
                "plus_ev_count": 0,
                "scan_diagnostics": {
                    "reason_code": "partial_errors",
                    "enabled_provider_count": 2,
                    "providers_with_events": 1,
                    "raw_provider_events": 2,
                    "merged_events_scanned": 0,
                    "overlap_clusters": 0,
                    "single_provider_clusters": 1,
                    "total_merge_hits": 0,
                    "sport_error_count": 1,
                    "stale_filter_drop_total": 0,
                    "provider_breakdown": [
                        {
                            "provider_key": "betmgm",
                            "provider_name": "BetMGM",
                            "enabled": True,
                            "raw_events": 2,
                            "matched_existing": 0,
                            "matched_fuzzy": 0,
                            "error_count": 1,
                        },
                        {
                            "provider_key": "draftkings",
                            "provider_name": "DraftKings",
                            "enabled": True,
                            "raw_events": 0,
                            "matched_existing": 0,
                            "matched_fuzzy": 0,
                            "error_count": 0,
                        },
                    ],
                },
                "cross_provider_match_report_summary": {
                    "total_raw_records": 2,
                    "total_match_clusters": 1,
                    "overlap_clusters": 0,
                    "single_provider_reason_counts": {
                        "no_close_candidate": 1,
                    },
                },
            },
        ]
        history_manager.load_recent.return_value = [
            {
                "scan_time": "2026-03-22T12:00:00Z",
                "mode": "ev",
                "sport_display": "NBA",
                "event": "Boston Celtics vs Miami Heat",
                "market": "h2h",
                "edge_percent": 3.2,
                "soft_book": "BetMGM",
                "soft_odds": 2.1,
                "sharp_fair": 1.95,
            }
        ]
        matching_scan_time = _display_time(matching_scan_dt)
        live_scan_time = _display_time(live_scan_dt)
        older_scan_time = _display_time(older_scan_dt)

        with ExitStack() as stack:
            stack.enter_context(patch.object(app_module, "ENV_PROVIDER_ONLY_MODE", True))
            stack.enter_context(patch.object(app_module, "_start_background_provider_services"))
            stack.enter_context(patch.object(app_module, "get_history_manager", return_value=history_manager))
            stack.enter_context(patch.object(app_module, "get_notifier", return_value=MagicMock(is_configured=False)))

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

                    page.click('button[data-tab="history"]')
                    page.wait_for_selector("#history-summary-table tbody tr")
                    page.wait_for_selector("#history-provider-table tbody tr")
                    page.wait_for_selector("#history-provider-watchlist-summary")

                    summary_text = page.locator("#history-summary-table").inner_text()
                    provider_history_text = page.locator("#history-provider-table").inner_text()
                    provider_watchlist_text = page.locator("#history-provider-watchlist").inner_text()
                    self.assertTrue(
                        "Likely Matching Issue" in summary_text
                        or "更像赛事匹配问题" in summary_text
                    )
                    self.assertTrue(
                        "Same teams, but kickoff window differs" in summary_text
                        or "队名一致，但开赛时间窗口不一致" in summary_text
                    )
                    self.assertIn(matching_scan_time, summary_text)
                    self.assertIn(live_scan_time, summary_text)
                    self.assertIn(older_scan_time, summary_text)
                    self.assertTrue(
                        "Arb 2 (1 positive)" in summary_text
                        or "套利 2 · 其中正 ROI 1" in summary_text
                    )
                    self.assertIn("BetMGM", provider_history_text)
                    self.assertTrue(
                        "Likely Fetch Risk" in provider_history_text
                        or "更像抓取风险" in provider_history_text
                    )
                    self.assertIn("BetMGM", provider_watchlist_text)
                    self.assertIn("FanDuel", provider_watchlist_text)
                    self.assertEqual(page.locator("#history-provider-watchlist .history-provider-watch-card").count(), 2)
                    self.assertEqual(page.locator("#history-provider-table tbody tr.is-watch").count(), 2)
                    self.assertIn("5", page.locator("#history-provider-count").inner_text())
                    self.assertEqual(page.locator("#history-provider-table tbody tr").count(), 5)

                    page.select_option("#history-summary-likely-filter", "matching")
                    self._wait_for(
                        lambda: "1/3" in page.locator("#history-summary-count").inner_text()
                        and page.locator("#history-summary-table tbody tr").count() == 1
                    )
                    filtered_matching_text = page.locator("#history-summary-table").inner_text()
                    self.assertEqual(page.locator("#history-summary-table tbody tr").count(), 1)
                    self.assertIn(matching_scan_time, filtered_matching_text)
                    self.assertNotIn(live_scan_time, filtered_matching_text)
                    self.assertNotIn(older_scan_time, filtered_matching_text)

                    page.select_option("#history-summary-likely-filter", "all")
                    page.select_option("#history-summary-mode-filter", "live")
                    self._wait_for(
                        lambda: "1/3" in page.locator("#history-summary-count").inner_text()
                        and page.locator("#history-summary-table tbody tr").count() == 1
                    )
                    filtered_live_text = page.locator("#history-summary-table").inner_text()
                    self.assertEqual(page.locator("#history-summary-table tbody tr").count(), 1)
                    self.assertIn(live_scan_time, filtered_live_text)
                    self.assertNotIn(matching_scan_time, filtered_live_text)
                    self.assertNotIn(older_scan_time, filtered_live_text)

                    page.select_option("#history-summary-mode-filter", "all")
                    page.select_option("#history-summary-provider-filter", "betmgm")
                    self._wait_for(
                        lambda: "2/3" in page.locator("#history-summary-count").inner_text()
                        and page.locator("#history-summary-table tbody tr").count() == 2
                        and page.locator("#history-provider-table tbody tr").count() == 1
                        and page.locator("#history-provider-watchlist .history-provider-watch-card").count() == 1
                    )
                    filtered_provider_text = page.locator("#history-summary-table").inner_text()
                    filtered_provider_history_text = page.locator("#history-provider-table").inner_text()
                    filtered_provider_watchlist_text = page.locator("#history-provider-watchlist").inner_text()
                    self.assertEqual(page.locator("#history-summary-table tbody tr").count(), 2)
                    self.assertEqual(page.locator("#history-provider-table tbody tr").count(), 1)
                    self.assertEqual(page.locator("#history-provider-watchlist .history-provider-watch-card").count(), 1)
                    self.assertEqual(page.locator("#history-provider-table tbody tr.is-watch").count(), 1)
                    self.assertEqual(
                        page.locator("#history-provider-table tbody tr .history-provider-trend-chip").count(),
                        2,
                    )
                    self.assertEqual(
                        page.locator("#history-provider-table tbody tr .history-provider-rate-segment").count(),
                        2,
                    )
                    self.assertIn(matching_scan_time, filtered_provider_text)
                    self.assertIn(older_scan_time, filtered_provider_text)
                    self.assertNotIn(live_scan_time, filtered_provider_text)
                    self.assertIn("BetMGM", filtered_provider_history_text)
                    self.assertNotIn("Polymarket", filtered_provider_history_text)
                    self.assertIn("BetMGM", filtered_provider_watchlist_text)
                    self.assertNotIn("FanDuel", filtered_provider_watchlist_text)
                    self.assertTrue(
                        "Fetch" in filtered_provider_history_text
                        or "抓取" in filtered_provider_history_text
                    )
                    self.assertTrue(
                        "Review" in filtered_provider_history_text
                        or "匹配" in filtered_provider_history_text
                    )
                    self.assertIn("50%", filtered_provider_history_text)

                    page.select_option("#history-summary-provider-filter", "all")
                    page.select_option("#history-summary-time-filter", "24h")
                    self._wait_for(
                        lambda: "2/3" in page.locator("#history-summary-count").inner_text()
                        and page.locator("#history-summary-table tbody tr").count() == 2
                        and page.locator("#history-provider-table tbody tr").count() == 4
                        and page.locator("#history-provider-watchlist .history-provider-watch-card").count() == 2
                    )
                    filtered_time_text = page.locator("#history-summary-table").inner_text()
                    filtered_time_provider_history_text = page.locator("#history-provider-table").inner_text()
                    filtered_time_watchlist_text = page.locator("#history-provider-watchlist").inner_text()
                    self.assertEqual(page.locator("#history-summary-table tbody tr").count(), 2)
                    self.assertEqual(page.locator("#history-provider-table tbody tr").count(), 4)
                    self.assertEqual(page.locator("#history-provider-watchlist .history-provider-watch-card").count(), 2)
                    self.assertIn(matching_scan_time, filtered_time_text)
                    self.assertIn(live_scan_time, filtered_time_text)
                    self.assertNotIn(older_scan_time, filtered_time_text)
                    self.assertIn("Polymarket", filtered_time_provider_history_text)
                    self.assertNotIn("DraftKings", filtered_time_provider_history_text)
                    self.assertIn("BetMGM", filtered_time_watchlist_text)
                    self.assertIn("FanDuel", filtered_time_watchlist_text)
                    self.assertNotIn("DraftKings", filtered_time_watchlist_text)
                    self.assertIn("100%", filtered_time_provider_history_text)

                    opportunity_text = page.locator("#history-table").inner_text()
                    self.assertIn("Boston Celtics vs Miami Heat", opportunity_text)
                    self.assertTrue("3.20% Edge" in opportunity_text or "3.20% Edge" in opportunity_text)

                    browser.close()
            finally:
                server.stop()

    def test_browser_pause_auto_scan_stops_future_runs_and_restores_editing(self) -> None:
        persisted_configs: list[dict] = []
        run_scan_calls: list[dict] = []
        first_scan_started = threading.Event()
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False

        def _fake_run_scan(**kwargs):
            run_scan_calls.append(copy.deepcopy(kwargs))
            first_scan_started.set()
            time.sleep(0.2)
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
                          const nativeSetTimeout = window.setTimeout.bind(window);
                          const nativeSetInterval = window.setInterval.bind(window);
                          window.setTimeout = (callback, ms, ...args) => nativeSetTimeout(callback, Math.min(ms, 50), ...args);
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
                    page.fill("#auto-scan-interval", "0")
                    page.dispatch_event("#auto-scan-interval", "change")
                    page.click("#close-advanced")

                    self._wait_for(lambda: first_scan_started.is_set())
                    page.wait_for_function("document.getElementById('scan-btn').disabled === true")

                    page.click("#open-advanced")
                    self.assertTrue(page.locator("#commission-input").is_disabled())
                    self.assertTrue(page.locator("#pause-auto-scan").evaluate("el => !el.disabled"))
                    running_state_text = (page.locator("#auto-scan-state-text").text_content() or "").strip()
                    running_button_text = (page.locator("#pause-auto-scan").text_content() or "").strip()
                    self.assertEqual(
                        page.locator("#auto-scan-state-chip").evaluate("el => el.dataset.state"),
                        "continuous",
                    )
                    self.assertTrue(running_state_text)
                    self.assertEqual(
                        page.locator("#pause-auto-scan").evaluate("el => el.dataset.state"),
                        "active-scanning",
                    )
                    self.assertTrue(running_button_text)

                    page.click("#pause-auto-scan")
                    self.assertFalse(page.locator("#auto-scan-toggle").is_checked())
                    self.assertTrue(page.locator("#pause-auto-scan").evaluate("el => el.disabled"))
                    page.wait_for_function(
                        "['pause-pending', 'paused'].includes(document.getElementById('auto-scan-state-chip').dataset.state)"
                    )
                    pending_state_text = (page.locator("#auto-scan-state-text").text_content() or "").strip()
                    pending_button_text = (page.locator("#pause-auto-scan").text_content() or "").strip()
                    paused_transition_state = page.locator("#auto-scan-state-chip").evaluate("el => el.dataset.state")
                    self.assertIn(paused_transition_state, {"pause-pending", "paused"})
                    self.assertTrue(pending_state_text)
                    self.assertNotEqual(pending_state_text, running_state_text)
                    paused_button_state = page.locator("#pause-auto-scan").evaluate("el => el.dataset.state")
                    self.assertIn(paused_button_state, {"pause-pending", "paused"})
                    self.assertTrue(pending_button_text)
                    self.assertNotEqual(pending_button_text, running_button_text)

                    page.wait_for_function("document.getElementById('scan-btn').disabled === false")
                    page.wait_for_function("document.getElementById('commission-input').disabled === false")
                    self.assertFalse(page.locator("#commission-input").is_disabled())
                    paused_state_text = (page.locator("#auto-scan-state-text").text_content() or "").strip()
                    paused_button_text = (page.locator("#pause-auto-scan").text_content() or "").strip()
                    self.assertEqual(
                        page.locator("#auto-scan-state-chip").evaluate("el => el.dataset.state"),
                        "paused",
                    )
                    self.assertTrue(paused_state_text)
                    self.assertEqual(
                        page.locator("#pause-auto-scan").evaluate("el => el.dataset.state"),
                        "paused",
                    )
                    self.assertTrue(paused_button_text)

                    self._wait_for(lambda: any(config.get("enabled") is False for config in persisted_configs))
                    calls_after_pause = len(run_scan_calls)
                    page.wait_for_timeout(250)
                    self.assertEqual(len(run_scan_calls), calls_after_pause)
                    self.assertIn(calls_after_pause, {1, 2})

                    browser.close()
            finally:
                server.stop()

    def test_browser_calculator_toggle_controls_latest_odds_sync(self) -> None:
        run_scan_calls: list[dict] = []
        history_manager = MagicMock()
        notifier = MagicMock()
        notifier.is_configured = False
        prices = [2.37, 2.51, 2.63]

        def _fake_run_scan(**kwargs):
            run_scan_calls.append(copy.deepcopy(kwargs))
            price = prices[min(len(run_scan_calls) - 1, len(prices) - 1)]
            return _sample_scan_result(
                opportunities=[_sample_arbitrage_opportunity(over_price=price)]
            )

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
                    page.click("#arb-desktop-list .arb-opportunity-card")
                    page.wait_for_selector("#arb-calc-body:not(.hidden)")

                    initial_odds = float(page.input_value("#arb-calc-odds-a"))
                    initial_stake = float(page.input_value("#arb-calc-stake-a"))
                    self.assertAlmostEqual(initial_odds, 2.37, places=2)

                    with page.expect_response(lambda response: response.url.endswith("/scan") and response.request.method == "POST"):
                        page.evaluate("document.getElementById('scan-form').requestSubmit()")

                    self._wait_for(lambda: len(run_scan_calls) >= 2)
                    self.assertAlmostEqual(float(page.input_value("#arb-calc-odds-a")), 2.37, places=2)
                    self.assertAlmostEqual(float(page.input_value("#arb-calc-stake-a")), initial_stake, places=2)

                    page.check("#arb-calc-live-odds-toggle")
                    self.assertTrue(page.is_checked("#arb-calc-live-odds-toggle"))

                    with page.expect_response(lambda response: response.url.endswith("/scan") and response.request.method == "POST"):
                        page.evaluate("document.getElementById('scan-form').requestSubmit()")

                    self._wait_for(lambda: len(run_scan_calls) >= 3)
                    self._wait_for(lambda: abs(float(page.input_value("#arb-calc-odds-a")) - 2.63) < 0.01)
                    self.assertAlmostEqual(float(page.input_value("#arb-calc-stake-a")), initial_stake, places=2)

                    browser.close()
            finally:
                server.stop()

        self.assertEqual(len(run_scan_calls), 3)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from providers import artline


def _deepcopy(value):
    return copy.deepcopy(value)


async def _fake_shared_client(*args, **kwargs):
    return object()


class ArtlineProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_event_url_uses_public_match_route(self) -> None:
        self.assertEqual(
            artline._event_url(
                {
                    "event_id": "370030005200042",
                    "sport_key": "soccer_italy_serie_a",
                    "live_state": {"is_live": False},
                }
            ),
            "https://artline.bet/bookmaker/match/prematch/football/370030005200042",
        )

    def test_normalize_game_markets_maps_two_way_moneyline(self) -> None:
        markets = artline._normalize_game_markets(
            [
                {"event_name_value": "0_ml_1", "value": 1.91, "status": 1},
                {"event_name_value": "0_ml_2", "value": 1.95, "status": 1},
            ],
            home_team="Lakers",
            away_team="Celtics",
            sport_key="basketball_nba",
            requested_markets={"h2h"},
        )

        self.assertEqual(
            markets,
            [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Lakers", "price": 1.91},
                        {"name": "Celtics", "price": 1.95},
                    ],
                }
            ],
        )

    def test_normalize_game_markets_preserves_betslip_selection_ids(self) -> None:
        markets = artline._normalize_game_markets(
            [
                {
                    "id": "3857631858183570063",
                    "event_name_value": "0_ml_1",
                    "value": 1.06,
                    "status": 1,
                },
                {
                    "id": "3857631858183570064",
                    "event_name_value": "0_ml_2",
                    "value": 14.5,
                    "status": 1,
                },
            ],
            home_team="Marko Miladinovic",
            away_team="Caheer Warik",
            sport_key="tennis_wta",
            requested_markets={"h2h"},
        )

        self.assertEqual(
            markets[0]["outcomes"],
            [
                {
                    "name": "Marko Miladinovic",
                    "price": 1.06,
                    "selection_id": "3857631858183570063",
                    "provider_event_name": "0_ml_1",
                },
                {
                    "name": "Caheer Warik",
                    "price": 14.5,
                    "selection_id": "3857631858183570064",
                    "provider_event_name": "0_ml_2",
                },
            ],
        )

    def test_build_web_max_bet_payload_uses_betslip_selection_id(self) -> None:
        payload = artline.build_web_max_bet_payload(
            sport="tennis",
            game_id="385763185818357",
            selection_id="3857631858183570064",
            is_live=False,
        )

        self.assertEqual(payload["type"], "solo")
        self.assertEqual(payload["sum"], 0)
        self.assertEqual(payload["currency_type"], "balance")
        self.assertIn('"sport":"tennis"', payload["events"])
        self.assertIn('"game_id":"385763185818357"', payload["events"])
        self.assertIn('"event_id":"3857631858183570064"', payload["events"])
        self.assertIn('"is_live":false', payload["events"])

    def test_preflight_web_max_bet_tries_csrf_bootstrap_without_browser_probe(self) -> None:
        with (
            patch.dict("os.environ", {"ARTLINE_COOKIE": ""}, clear=False),
            patch.object(artline, "resolve_artline_browser_cookie_header") as browser_cookie,
            patch.object(
                artline,
                "_bootstrap_artline_csrf",
                return_value={"cookie_header": "", "source": "csrf_bootstrap", "http_status": 204},
            ) as bootstrap,
        ):
            result = artline.preflight_web_max_bet(
                sport="tennis",
                game_id="385763185818357",
                selection_id="3857631858183570064",
                is_live=False,
                stake=5.35,
            )

        self.assertEqual(result["status"], "auth_required")
        self.assertEqual(result["reason"], "missing_artline_cookie")
        self.assertEqual(result["cookie_source"], "csrf_bootstrap")
        bootstrap.assert_called_once()
        browser_cookie.assert_not_called()

    def test_csrf_token_can_be_resolved_from_cookie_header(self) -> None:
        token = artline._csrf_token_from_cookie_header(
            "laravel_session=session-id; XSRF-TOKEN=abc%2B123%3D; theme=dark"
        )

        self.assertEqual(token, "abc+123=")

    def test_cookie_names_are_safe_for_reports(self) -> None:
        names = artline._cookie_names_from_header(
            "XSRF-TOKEN=token; apiato=session; 8HJsTVhdvQYdl1ou56PX06IgzA832cxkBGvkBvSj=value; theme=dark"
        )

        self.assertEqual(names, ["XSRF-TOKEN", "apiato", "session_cookie", "other_cookie"])

    def test_preflight_web_max_bet_uses_browser_cookie_when_enabled(self) -> None:
        class _Response:
            status_code = 200

            def json(self):
                return {"data": {"max_bet": 125.0}}

        class _Client:
            posted_headers = {}

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, *, json, headers):
                self.__class__.posted_headers = headers
                return _Response()

        with (
            patch.dict(
                "os.environ",
                {
                    "ARTLINE_COOKIE": "",
                    "ARTLINE_CSRF_TOKEN": "",
                    "ARTLINE_XSRF_TOKEN": "",
                    "ARTLINE_AUTO_BROWSER_COOKIES": "1",
                },
                clear=False,
            ),
            patch.object(
                artline,
                "resolve_artline_browser_cookie_header",
                return_value={
                    "cookie_header": "laravel_session=session-id; XSRF-TOKEN=abc%2B123%3D",
                    "source": "chrome:Default",
                    "cookie_names": ["laravel_session", "XSRF-TOKEN"],
                    "errors": [],
                },
            ) as browser_cookie,
            patch("providers.artline.httpx.Client", new=_Client),
        ):
            result = artline.preflight_web_max_bet(
                sport="tennis",
                game_id="385763185818357",
                selection_id="3857631858183570064",
                is_live=False,
                stake=50,
            )

        browser_cookie.assert_called_once()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["cookie_source"], "chrome:Default")
        self.assertEqual(result["cookie_names"], ["laravel_session", "XSRF-TOKEN"])
        self.assertEqual(_Client.posted_headers["X-XSRF-TOKEN"], "abc+123=")
        self.assertNotIn("X-CSRF-TOKEN", _Client.posted_headers)
        self.assertNotIn("abc%2B123%3D", str(result))

    def test_preflight_web_max_bet_bootstraps_csrf_cookie(self) -> None:
        class _Response:
            def __init__(self, status_code, payload=None):
                self.status_code = status_code
                self._payload = payload

            def json(self):
                if self._payload is None:
                    raise ValueError("no json")
                return self._payload

        class _Client:
            posted_headers = {}

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, *, headers):
                return _Response(204)

            def post(self, url, *, json, headers):
                self.__class__.posted_headers = headers
                return _Response(200, {"data": {"max_bet": 80.0}})

            @property
            def cookies(self):
                return {"XSRF-TOKEN": "boot%2Btoken", "laravel_session": "session-id"}

        with (
            patch.dict(
                "os.environ",
                {"ARTLINE_COOKIE": "", "ARTLINE_AUTO_BROWSER_COOKIES": ""},
                clear=False,
            ),
            patch("providers.artline.httpx.Client", new=_Client),
        ):
            result = artline.preflight_web_max_bet(
                sport="tennis",
                game_id="385763185818357",
                selection_id="3857631858183570064",
                is_live=False,
                stake=25,
            )

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["cookie_source"], "csrf_bootstrap")
        self.assertEqual(_Client.posted_headers["X-XSRF-TOKEN"], "boot+token")
        self.assertNotIn("X-CSRF-TOKEN", _Client.posted_headers)

    def test_bootstrap_artline_csrf_uses_api_host_without_api_prefix(self) -> None:
        class _Response:
            status_code = 204

        class _Client:
            requested_url = ""

            def get(self, url, *, headers):
                self.__class__.requested_url = url
                return _Response()

            @property
            def cookies(self):
                return {"XSRF-TOKEN": "boot%2Btoken", "laravel_session": "session-id"}

        with patch.object(artline, "ARTLINE_API_BASE", "https://api.artline.bet/api"):
            result = artline._bootstrap_artline_csrf(_Client())

        self.assertEqual(_Client.requested_url, "https://api.artline.bet/sanctum/csrf-cookie")
        self.assertEqual(result["cookie_names"], ["XSRF-TOKEN", "laravel_session"])

    def test_preflight_web_max_bet_reports_bootstrap_when_csrf_cookie_missing(self) -> None:
        class _Response:
            status_code = 204

            def json(self):
                raise ValueError("no json")

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, *, headers):
                return _Response()

            @property
            def cookies(self):
                return {}

        with (
            patch.dict("os.environ", {"ARTLINE_COOKIE": "", "ARTLINE_AUTO_BROWSER_COOKIES": ""}, clear=False),
            patch("providers.artline.httpx.Client", new=_Client),
        ):
            result = artline.preflight_web_max_bet(
                sport="tennis",
                game_id="385763185818357",
                selection_id="3857631858183570064",
                is_live=False,
                stake=5,
            )

        self.assertEqual(result["status"], "auth_required")
        self.assertEqual(result["reason"], "missing_artline_cookie")
        self.assertEqual(result["cookie_source"], "csrf_bootstrap")
        self.assertEqual(result["bootstrap_http_status"], 204)

    def test_preflight_web_max_bet_reports_browser_cookie_probe_errors_safely(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"ARTLINE_COOKIE": "", "ARTLINE_AUTO_BROWSER_COOKIES": "1"},
                clear=False,
            ),
            patch.object(
                artline,
                "resolve_artline_browser_cookie_header",
                return_value={
                    "cookie_header": "",
                    "source": None,
                    "cookie_names": [],
                    "errors": ["chrome:Default:PermissionError"],
                },
            ),
            patch("providers.artline.httpx.Client") as client_cls,
        ):
            result = artline.preflight_web_max_bet(
                sport="tennis",
                game_id="385763185818357",
                selection_id="3857631858183570064",
                is_live=False,
                stake=5.35,
            )

        self.assertEqual(result["status"], "auth_required")
        self.assertEqual(result["reason"], "missing_artline_cookie")
        self.assertEqual(result["cookie_source"], "csrf_bootstrap")
        self.assertEqual(result["browser_cookie_errors"], ["chrome:Default:PermissionError"])
        client_cls.assert_called_once()

    def test_resolve_browser_cookie_prefers_cdp_debug_cookie_source(self) -> None:
        with (
            patch.object(
                artline,
                "_resolve_artline_cdp_cookie_header",
                return_value={
                    "cookie_header": "XSRF-TOKEN=abc; laravel_session=session-id",
                    "source": "chrome-cdp:9222",
                    "cookie_names": ["XSRF-TOKEN", "laravel_session"],
                    "errors": [],
                },
            ) as cdp_probe,
            patch.object(artline, "_candidate_chromium_cookie_dbs") as db_probe,
        ):
            result = artline.resolve_artline_browser_cookie_header()

        cdp_probe.assert_called_once()
        db_probe.assert_not_called()
        self.assertEqual(result["source"], "chrome-cdp:9222")
        self.assertEqual(result["cookie_names"], ["XSRF-TOKEN", "laravel_session"])

    def test_resolve_browser_cookie_sanitizes_cookie_names_from_db(self) -> None:
        with (
            patch.object(
                artline,
                "_resolve_artline_cdp_cookie_header",
                return_value={"cookie_header": "", "source": None, "cookie_names": [], "errors": []},
            ),
            patch.object(
                artline,
                "_candidate_chromium_cookie_dbs",
                return_value=[("chrome:Default", object(), object())],
            ),
            patch.object(
                artline,
                "_read_artline_cookies_from_db",
                return_value=[
                    [("XSRF-TOKEN", "token"), ("8HJsTVhdvQYdl1ou56PX06IgzA832cxkBGvkBvSj", "session")],
                    [],
                ],
            ),
        ):
            result = artline.resolve_artline_browser_cookie_header()

        self.assertEqual(result["cookie_names"], ["XSRF-TOKEN", "session_cookie"])

    async def test_fetch_events_async_surfaces_artline_max_bet_diagnostics_without_max_stake(self) -> None:
        payload = {
            "data": {
                "basketball": {
                    "games": [
                        {
                            "id": 385680025900261,
                            "is_live": False,
                            "max_bet": 0.01,
                            "start_at_timestamp": 1779841800,
                            "team_1": {"value": "Oklahoma City Thunder"},
                            "team_2": {"value": "San Antonio Spurs"},
                            "events": [
                                {"event_name_value": "0_ml_1", "value": 1.54, "status": 1},
                                {"event_name_value": "0_ml_2", "value": 2.6, "status": 1},
                            ],
                        }
                    ]
                }
            }
        }

        with (
            patch.object(artline, "get_shared_client", new=_fake_shared_client),
            patch.object(artline, "_request_json_async", return_value=(_deepcopy(payload), 0)),
        ):
            events = await artline.fetch_events_async(
                "basketball_nba",
                ["h2h"],
                ["us"],
                bookmakers=["artline"],
            )

        bookmaker = events[0]["bookmakers"][0]
        self.assertNotIn("max_stake", bookmaker["markets"][0]["outcomes"][0])
        self.assertEqual(
            bookmaker["execution_diagnostics"],
            {
                "artline_max_bet": 0.01,
                "artline_min_bet": 5.0,
                "executable": False,
                "reason": "max_bet_below_min_bet",
            },
        )
        self.assertEqual(artline.fetch_events_async.last_stats.get("max_bet_below_min_bet_count"), 1)

    def test_normalize_game_markets_maps_hockey_detail_moneyline_prefix(self) -> None:
        markets = artline._normalize_game_markets(
            [
                {"event_name_value": "1_ml_1", "value": 1.97, "status": 1},
                {"event_name_value": "1_ml_2", "value": 1.77, "status": 1},
            ],
            home_team="Florida Panthers",
            away_team="New York Rangers",
            sport_key="icehockey_nhl",
            requested_markets={"h2h"},
        )

        self.assertEqual(
            markets,
            [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Florida Panthers", "price": 1.97},
                        {"name": "New York Rangers", "price": 1.77},
                    ],
                }
            ],
        )

    def test_normalize_game_markets_maps_three_way_result_market(self) -> None:
        markets = artline._normalize_game_markets(
            [
                {"event_name_value": "0_win_0", "value": 3.7, "status": 1},
                {"event_name_value": "0_win_1", "value": 1.61, "status": 1},
                {"event_name_value": "0_win_2", "value": 5.5, "status": 1},
            ],
            home_team="Newcastle",
            away_team="Sunderland",
            sport_key="soccer_italy_serie_a",
            requested_markets={"h2h_3_way"},
        )

        self.assertEqual(
            markets,
            [
                {
                    "key": "h2h_3_way",
                    "outcomes": [
                        {"name": "Newcastle", "price": 1.61},
                        {"name": "Draw", "price": 3.7},
                        {"name": "Sunderland", "price": 5.5},
                    ],
                }
            ],
        )

    def test_normalize_game_markets_maps_spreads_and_totals(self) -> None:
        markets = artline._normalize_game_markets(
            [
                {"event_name_value": "0_to-main_0_239.5", "value": 1.91, "status": 1},
                {"event_name_value": "0_tu-main_0_239.5", "value": 1.95, "status": 1},
                {"event_name_value": "0_f-main_1_-6.5", "value": 2.05, "status": 1},
                {"event_name_value": "0_f-main_2_6.5", "value": 1.82, "status": 1},
                {"event_name_value": "0_to-main_1_120.5", "value": 1.8, "status": 1},
                {"event_name_value": "0_f-main_1_6.5", "value": 1.6, "status": 1},
                {"event_name_value": "0_f-main_2_-6.5", "value": 2.2, "status": 1},
            ],
            home_team="Lakers",
            away_team="Celtics",
            sport_key="basketball_nba",
            requested_markets={"spreads", "totals"},
        )

        self.assertEqual(
            markets,
            [
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": 1.91, "point": 239.5},
                        {"name": "Under", "price": 1.95, "point": 239.5},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Lakers", "price": 2.05, "point": -6.5},
                        {"name": "Celtics", "price": 1.82, "point": 6.5},
                    ],
                },
            ],
        )

    def test_normalize_game_markets_maps_team_totals_with_description(self) -> None:
        markets = artline._normalize_game_markets(
            [
                {"event_name_value": "0_to-sec_1_125.5", "value": 1.95, "status": 1},
                {"event_name_value": "0_tu-sec_1_125.5", "value": 1.9, "status": 1},
                {"event_name_value": "0_to-sec_2_116.5", "value": 1.95, "status": 1},
                {"event_name_value": "0_tu-sec_2_116.5", "value": 1.9, "status": 1},
            ],
            home_team="Denver Nuggets",
            away_team="Portland Trail Blazers",
            sport_key="basketball_nba",
            requested_markets={"team_totals"},
        )

        self.assertEqual(
            markets,
            [
                {
                    "key": "team_totals",
                    "outcomes": [
                        {
                            "name": "Over",
                            "price": 1.95,
                            "point": 125.5,
                            "description": "Denver Nuggets",
                        },
                        {
                            "name": "Under",
                            "price": 1.9,
                            "point": 125.5,
                            "description": "Denver Nuggets",
                        },
                    ],
                },
                {
                    "key": "team_totals",
                    "outcomes": [
                        {
                            "name": "Over",
                            "price": 1.95,
                            "point": 116.5,
                            "description": "Portland Trail Blazers",
                        },
                        {
                            "name": "Under",
                            "price": 1.9,
                            "point": 116.5,
                            "description": "Portland Trail Blazers",
                        },
                    ],
                },
            ],
        )

    async def test_fetch_events_async_replays_lines_payload(self) -> None:
        payload = {
            "data": {
                "football": {
                    "games": [
                        {
                            "id": 370030005200042,
                            "is_live": False,
                            "start_at_timestamp": 1774208700,
                            "team_1": {"value": "Fiorentina"},
                            "team_2": {"value": "Inter"},
                            "events": [
                                {"event_name_value": "0_win_0", "value": 4.1, "status": 1},
                                {"event_name_value": "0_win_1", "value": 5.0, "status": 1},
                                {"event_name_value": "0_win_2", "value": 1.68, "status": 1},
                                {"event_name_value": "0_to-main_0_2.5", "value": 1.8, "status": 1},
                                {"event_name_value": "0_tu-main_0_2.5", "value": 2.08, "status": 1},
                                {"event_name_value": "0_f-main_1_-0.5", "value": 2.66, "status": 1},
                                {"event_name_value": "0_f-main_2_0.5", "value": 1.42, "status": 1},
                            ],
                        }
                    ]
                }
            }
        }

        with (
            patch.object(artline, "get_shared_client", new=_fake_shared_client),
            patch.object(artline, "_request_json_async", return_value=(_deepcopy(payload), 0)),
        ):
            events = await artline.fetch_events_async(
                "soccer_italy_serie_a",
                ["h2h", "h2h_3_way", "spreads", "totals"],
                ["eu"],
            )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["home_team"], "Fiorentina")
        self.assertEqual(event["away_team"], "Inter")
        self.assertEqual(event["commence_time"], "2026-03-22T19:45:00Z")
        bookmaker = event["bookmakers"][0]
        self.assertEqual(bookmaker["key"], "artline")
        self.assertEqual(
            bookmaker["event_url"],
            "https://artline.bet/bookmaker/match/prematch/football/370030005200042",
        )
        self.assertEqual(
            [market["key"] for market in bookmaker["markets"]],
            ["h2h_3_way", "totals", "spreads"],
        )
        self.assertEqual(artline.fetch_events_async.last_stats.get("events_returned_count"), 1)
        self.assertEqual(artline.fetch_events_async.last_stats.get("payload_games_count"), 1)

    async def test_fetch_events_async_enriches_hockey_h2h_from_detail_feed(self) -> None:
        payload = {
            "data": {
                "hockey": {
                    "games": [
                        {
                            "id": 375350180001809,
                            "is_live": False,
                            "start_at_timestamp": 1776121200,
                            "team_1": {"value": "Florida Panthers"},
                            "team_2": {"value": "New York Rangers"},
                            "events": [
                                {"event_name_value": "0_to-main_0_6", "value": 2.04, "status": 1},
                                {"event_name_value": "0_tu-main_0_6", "value": 1.86, "status": 1},
                            ],
                        }
                    ]
                }
            }
        }
        detail_events = [
            {"event_name_value": "1_ml_1", "value": 1.97, "status": 1},
            {"event_name_value": "1_ml_2", "value": 1.77, "status": 1},
        ]

        with (
            patch.object(artline, "get_shared_client", new=_fake_shared_client),
            patch.object(artline, "_request_json_async", return_value=(_deepcopy(payload), 0)),
            patch.object(artline, "_load_game_detail_events_async", return_value=_deepcopy(detail_events)),
        ):
            events = await artline.fetch_events_async(
                "icehockey_nhl",
                ["h2h"],
                ["eu"],
                bookmakers=["artline"],
            )

        self.assertEqual(len(events), 1)
        bookmaker = events[0]["bookmakers"][0]
        market = bookmaker["markets"][0]
        self.assertEqual(market["key"], "h2h")
        self.assertEqual(
            [(row["name"], row["price"]) for row in market["outcomes"]],
            [("Florida Panthers", 1.97), ("New York Rangers", 1.77)],
        )
        self.assertTrue(all(row.get("quote_source") == "rest_snapshot" for row in market["outcomes"]))
        self.assertTrue(all(row.get("observed_at") for row in market["outcomes"]))
        self.assertEqual(artline.fetch_events_async.last_stats.get("detail_enrichment_requested"), 1)
        self.assertEqual(artline.fetch_events_async.last_stats.get("detail_enrichment_succeeded"), 1)

    async def test_fetch_events_async_enriches_tennis_h2h_selection_ids_from_detail_feed(self) -> None:
        payload = {
            "data": {
                "tennis": {
                    "games": [
                        {
                            "id": 385763185818357,
                            "is_live": False,
                            "max_bet": 0.01,
                            "start_at_timestamp": 1779891300,
                            "team_1": {"value": "Marko Miladinovic"},
                            "team_2": {"value": "Caheer Warik"},
                            "events": [
                                {"event_name_value": "0_ml_1", "value": 1.06, "status": 1},
                                {"event_name_value": "0_ml_2", "value": 14.5, "status": 1},
                            ],
                        }
                    ]
                }
            }
        }
        detail_events = [
            {
                "id": "3857631858183570063",
                "event_name_value": "0_ml_1",
                "value": 1.06,
                "status": 1,
            },
            {
                "id": "3857631858183570064",
                "event_name_value": "0_ml_2",
                "value": 14.5,
                "status": 1,
            },
        ]

        with (
            patch.object(artline, "get_shared_client", new=_fake_shared_client),
            patch.object(artline, "_request_json_async", return_value=(_deepcopy(payload), 0)),
            patch.object(artline, "_load_game_detail_events_async", return_value=_deepcopy(detail_events)),
        ):
            events = await artline.fetch_events_async(
                "tennis_wta",
                ["h2h"],
                ["eu"],
                bookmakers=["artline"],
            )

        outcomes = events[0]["bookmakers"][0]["markets"][0]["outcomes"]
        self.assertEqual(outcomes[0]["selection_id"], "3857631858183570063")
        self.assertEqual(outcomes[1]["selection_id"], "3857631858183570064")
        self.assertEqual(artline.fetch_events_async.last_stats.get("detail_enrichment_requested"), 1)
        self.assertEqual(artline.fetch_events_async.last_stats.get("detail_enrichment_succeeded"), 1)

    async def test_fetch_events_async_respects_bookmaker_filter(self) -> None:
        events = await artline.fetch_events_async(
            "soccer_italy_serie_a",
            ["h2h"],
            ["eu"],
            bookmakers=["sx_bet"],
        )
        self.assertEqual(events, [])

    async def test_fetch_events_async_marks_unsupported_sport(self) -> None:
        events = await artline.fetch_events_async(
            "basketball_ncaab",
            ["h2h"],
            ["us"],
        )
        self.assertEqual(events, [])
        self.assertTrue(artline.fetch_events_async.last_stats.get("skipped_unsupported_sport"))

    async def test_fetch_events_async_marks_live_feed_empty_when_live_payload_has_no_games(self) -> None:
        payload = {"data": []}

        with (
            patch.object(artline, "get_shared_client", new=_fake_shared_client),
            patch.object(artline, "_request_json_async", return_value=(_deepcopy(payload), 0)),
        ):
            events = await artline.fetch_events_async(
                "icehockey_nhl",
                ["h2h", "spreads", "totals"],
                ["us"],
                bookmakers=["artline"],
                context={"live": True, "scan_mode": "live"},
            )

        self.assertEqual(events, [])
        self.assertEqual(artline.fetch_events_async.last_stats.get("games_type"), "live")
        self.assertEqual(artline.fetch_events_async.last_stats.get("payload_shape"), "list")
        self.assertTrue(artline.fetch_events_async.last_stats.get("live_feed_empty"))

    async def test_fetch_events_async_records_live_all_sports_when_live_sport_is_empty(self) -> None:
        requested_payload = {'data': []}
        live_all_payload = {
            'data': {
                'basketball': {'games': [{'id': '1'}, {'id': '2'}]},
                'football': {'games': [{'id': '3'}]},
                'tennis': {'games': [{'id': '4'}]},
                'hockey': {'games': []},
            }
        }
        responses = [(_deepcopy(requested_payload), 0), (_deepcopy(live_all_payload), 0)]

        async def _fake_request_json_async(*args, **kwargs):
            return responses.pop(0)

        with (
            patch.object(artline, 'get_shared_client', new=_fake_shared_client),
            patch.object(artline, '_request_json_async', side_effect=_fake_request_json_async),
        ):
            events = await artline.fetch_events_async(
                'icehockey_nhl',
                ['h2h', 'spreads', 'totals'],
                ['us'],
                bookmakers=['artline'],
                context={'live': True, 'scan_mode': 'live'},
            )

        self.assertEqual(events, [])
        self.assertTrue(artline.fetch_events_async.last_stats.get('live_feed_empty'))
        self.assertEqual(
            artline.fetch_events_async.last_stats.get('live_all_sports_available'),
            {'basketball': 2, 'football': 1, 'tennis': 1},
        )
        self.assertEqual(artline.fetch_events_async.last_stats.get('live_all_total_games'), 4)

    async def test_fetch_events_async_enriches_team_totals_from_game_detail(self) -> None:
        lines_payload = {
            "data": {
                "basketball": {
                    "games": [
                        {
                            "id": "370050024800258",
                            "is_live": False,
                            "period": 0,
                            "start_at_timestamp": 1774213200,
                            "team_1": {"value": "Denver Nuggets"},
                            "team_2": {"value": "Portland Trail Blazers"},
                            "events": [
                                {"event_name_value": "0_ml_1", "value": 1.26, "status": 1},
                                {"event_name_value": "0_ml_2", "value": 3.85, "status": 1},
                            ],
                        }
                    ]
                }
            }
        }
        detail_payload = {
            "data": {
                "events": [
                    {"event_name_value": "0_ml_1", "value": 1.26, "status": 1},
                    {"event_name_value": "0_ml_2", "value": 3.85, "status": 1},
                    {"event_name_value": "0_to-sec_1_125.5", "value": 1.95, "status": 1},
                    {"event_name_value": "0_tu-sec_1_125.5", "value": 1.9, "status": 1},
                    {"event_name_value": "0_to-sec_2_116.5", "value": 1.95, "status": 1},
                    {"event_name_value": "0_tu-sec_2_116.5", "value": 1.9, "status": 1},
                ]
            }
        }

        async def _fake_request_json_async(
            client,
            method,
            path,
            *,
            params=None,
            json_payload=None,
            retries=None,
            backoff_seconds=None,
        ):
            if method == "POST" and path == "lines":
                return _deepcopy(lines_payload), 0
            if method == "GET" and path == "lines/game/prematch/basketball/370050024800258":
                return _deepcopy(detail_payload), 0
            raise AssertionError((method, path))

        with (
            patch.object(artline, "get_shared_client", new=_fake_shared_client),
            patch.object(artline, "_request_json_async", side_effect=_fake_request_json_async),
        ):
            events = await artline.fetch_events_async(
                "basketball_nba",
                ["h2h", "team_totals"],
                ["us"],
                bookmakers=["artline"],
            )

        self.assertEqual(len(events), 1)
        bookmaker = events[0]["bookmakers"][0]
        team_total_markets = [market for market in (bookmaker.get("markets") or []) if market.get("key") == "team_totals"]
        self.assertEqual(len(team_total_markets), 2)
        self.assertEqual(artline.fetch_events_async.last_stats.get("detail_enrichment_requested"), 1)
        self.assertEqual(artline.fetch_events_async.last_stats.get("detail_enrichment_succeeded"), 1)
        self.assertEqual(artline.fetch_events_async.last_stats.get("detail_enrichment_failed"), 0)


if __name__ == "__main__":
    unittest.main()

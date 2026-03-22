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

    def test_normalize_game_markets_maps_three_way_result_market(self) -> None:
        markets = artline._normalize_game_markets(
            [
                {"event_name_value": "0_win_0", "value": 3.7, "status": 1},
                {"event_name_value": "0_win_1", "value": 1.61, "status": 1},
                {"event_name_value": "0_win_2", "value": 5.5, "status": 1},
            ],
            home_team="Newcastle",
            away_team="Sunderland",
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

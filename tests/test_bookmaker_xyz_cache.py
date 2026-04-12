import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from providers import bookmaker_xyz


class BookmakerXyzCacheTests(unittest.TestCase):
    def test_load_market_manager_snapshot_live_context_requests_only_live_games(self) -> None:
        captured_states = []

        async def _fake_request(*args, **kwargs):
            params = kwargs.get("params") or {}
            captured_states.append(params.get("gameState"))
            return ({"games": []}, 0)

        async def _run():
            with patch.object(bookmaker_xyz, "_request_market_manager_json_async", side_effect=_fake_request):
                await bookmaker_xyz._load_market_manager_snapshot_async(
                    client=None,
                    sport_key="soccer_epl",
                    retries=0,
                    backoff_seconds=0.0,
                    timeout=1,
                    context={"live": True},
                )

        asyncio.run(_run())

        self.assertTrue(captured_states)
        self.assertEqual(sorted(set(captured_states)), ["Live"])

    def test_normalize_condition_market_accepts_market_manager_full_game_ids(self) -> None:
        dictionaries = {
            "marketNames": {
                "19-76-76": "Match Winner",
                "3-76-76": "Handicap incl. OT",
                "4-76-76": "Total Points incl. OT",
            },
            "outcomes": {
                "6983": {"selectionId": 10009, "marketId": 19, "gamePeriodId": 76, "gameTypeId": 76, "pointsId": None, "teamPlayerId": None},
                "6984": {"selectionId": 10010, "marketId": 19, "gamePeriodId": 76, "gameTypeId": 76, "pointsId": None, "teamPlayerId": None},
                "2441": {"selectionId": 7, "marketId": 3, "gamePeriodId": 76, "gameTypeId": 76, "pointsId": 180, "teamPlayerId": 1},
                "2442": {"selectionId": 8, "marketId": 3, "gamePeriodId": 76, "gameTypeId": 76, "pointsId": 180, "teamPlayerId": 2},
                "8287": {"selectionId": 9, "marketId": 4, "gamePeriodId": 76, "gameTypeId": 76, "pointsId": 627, "teamPlayerId": None},
                "8288": {"selectionId": 10, "marketId": 4, "gamePeriodId": 76, "gameTypeId": 76, "pointsId": 627, "teamPlayerId": None},
            },
            "selections": {
                "7": "H1",
                "8": "H2",
                "9": "Over",
                "10": "Under",
                "10009": "1",
                "10010": "2",
            },
            "teamPlayers": {"1": "H1", "2": "H2"},
            "points": {"180": "15.5", "627": "231"},
        }

        h2h_market = bookmaker_xyz._normalize_condition_market(
            {"outcomes": [{"outcomeId": "6983", "odds": "2.10"}, {"outcomeId": "6984", "odds": "1.80"}]},
            home_team="Home",
            away_team="Away",
            requested_markets={"h2h"},
            dictionaries=dictionaries,
        )
        spreads_market = bookmaker_xyz._normalize_condition_market(
            {"outcomes": [{"outcomeId": "2441", "odds": "1.94"}, {"outcomeId": "2442", "odds": "1.79"}]},
            home_team="Home",
            away_team="Away",
            requested_markets={"spreads"},
            dictionaries=dictionaries,
        )
        totals_market = bookmaker_xyz._normalize_condition_market(
            {"outcomes": [{"outcomeId": "8287", "odds": "1.74"}, {"outcomeId": "8288", "odds": "2.00"}]},
            home_team="Home",
            away_team="Away",
            requested_markets={"totals"},
            dictionaries=dictionaries,
        )

        self.assertEqual(h2h_market["key"], "h2h")
        self.assertEqual(spreads_market["outcomes"][0]["name"], "Home")
        self.assertEqual(spreads_market["outcomes"][0]["point"], 15.5)
        self.assertEqual(totals_market["outcomes"][0]["name"], "Over")
        self.assertEqual(totals_market["outcomes"][0]["point"], 231.0)

    def test_normalize_condition_market_rejects_team_totals_from_base_totals(self) -> None:
        dictionaries = {
            'marketNames': {
                '4-76-76': 'Team Total Points',
            },
            'outcomes': {
                '9001': {'selectionId': 9, 'marketId': 4, 'gamePeriodId': 76, 'gameTypeId': 76, 'pointsId': 120, 'teamPlayerId': None},
                '9002': {'selectionId': 10, 'marketId': 4, 'gamePeriodId': 76, 'gameTypeId': 76, 'pointsId': 120, 'teamPlayerId': None},
            },
            'selections': {
                '9': 'Over',
                '10': 'Under',
            },
            'teamPlayers': {},
            'points': {'120': '123.5'},
        }

        totals_market = bookmaker_xyz._normalize_condition_market(
            {'outcomes': [{'outcomeId': '9001', 'odds': '1.91'}, {'outcomeId': '9002', 'odds': '1.91'}]},
            home_team='Home',
            away_team='Away',
            requested_markets={'totals'},
            dictionaries=dictionaries,
        )

        self.assertIsNone(totals_market)

    def test_normalize_condition_market_rejects_segmented_team_total_points_from_totals(self) -> None:
        dictionaries = {
            'marketNames': {
                '4-76-76': 'Total Points',
            },
            'outcomes': {
                '9101': {'selectionId': 9, 'marketId': 4, 'gamePeriodId': 76, 'gameTypeId': 76, 'pointsId': 120, 'teamPlayerId': None},
                '9102': {'selectionId': 10, 'marketId': 4, 'gamePeriodId': 76, 'gameTypeId': 76, 'pointsId': 120, 'teamPlayerId': None},
            },
            'selections': {
                '9': 'Over',
                '10': 'Under',
            },
            'teamPlayers': {},
            'points': {'120': '123.5'},
        }

        totals_market = bookmaker_xyz._normalize_condition_market(
            {
                'conditionName': 'Chicago Bulls Team Total O/U 123.5',
                'outcomes': [
                    {'outcomeId': '9101', 'odds': '2.24'},
                    {'outcomeId': '9102', 'odds': '1.58'},
                ],
            },
            home_team='Chicago Bulls',
            away_team='Memphis Grizzlies',
            requested_markets={'totals'},
            dictionaries=dictionaries,
        )

        self.assertIsNone(totals_market)

    def test_normalize_condition_market_rejects_individual_total_points_from_totals(self) -> None:
        dictionaries = {
            'marketNames': {
                '7-76-76-2': 'Team 2 - Total Points incl. OT',
            },
            'outcomes': {
                '9201': {'selectionId': 15, 'marketId': 7, 'gamePeriodId': 76, 'gameTypeId': 76, 'pointsId': 132, 'teamPlayerId': 2},
                '9202': {'selectionId': 16, 'marketId': 7, 'gamePeriodId': 76, 'gameTypeId': 76, 'pointsId': 132, 'teamPlayerId': 2},
            },
            'selections': {
                '15': 'Over',
                '16': 'Under',
            },
            'teamPlayers': {'2': 'Team 2'},
            'points': {'132': '123.5'},
        }

        totals_market = bookmaker_xyz._normalize_condition_market(
            {
                'outcomes': [
                    {'outcomeId': '9201', 'odds': '2.24'},
                    {'outcomeId': '9202', 'odds': '1.58'},
                ],
            },
            home_team='Chicago Bulls',
            away_team='Memphis Grizzlies',
            requested_markets={'totals'},
            dictionaries=dictionaries,
        )

        self.assertIsNone(totals_market)

    def test_normalize_condition_market_rejects_first_half_total_from_full_game_totals(self) -> None:
        dictionaries = {
            'marketNames': {
                '4-50-76': '1st Half: Total',
            },
            'outcomes': {
                '9301': {'selectionId': 9, 'marketId': 4, 'gamePeriodId': 50, 'gameTypeId': 76, 'pointsId': 122, 'teamPlayerId': None},
                '9302': {'selectionId': 10, 'marketId': 4, 'gamePeriodId': 50, 'gameTypeId': 76, 'pointsId': 122, 'teamPlayerId': None},
            },
            'selections': {
                '9': 'Over',
                '10': 'Under',
            },
            'teamPlayers': {},
            'points': {'122': '119.5'},
        }

        totals_market = bookmaker_xyz._normalize_condition_market(
            {
                'outcomes': [
                    {'outcomeId': '9301', 'odds': '1.83'},
                    {'outcomeId': '9302', 'odds': '1.83'},
                ],
            },
            home_team='Chicago Bulls',
            away_team='Memphis Grizzlies',
            requested_markets={'totals'},
            dictionaries=dictionaries,
        )

        self.assertIsNone(totals_market)

    def test_parse_official_dictionaries_materializes_outcome_proxy(self) -> None:
        module_source = """
const dictionaries = {
  marketNames: {"19-76-76": "Match Winner"},
  marketDescriptions: {},
  selections: {"10009": "1"},
  teamPlayers: {},
  points: {},
  outcomes: new Proxy(
    {"6983": [10009, 19, 76, 76, 1, null, null]},
    {
      get(target, prop) {
        if (!target[prop]) return undefined;
        const [selectionId, marketId, gamePeriodId, gameTypeId, gameVarietyId, pointsId, teamPlayerId] = target[prop];
        return {
          selectionId,
          marketId,
          gamePeriodId,
          gameTypeId,
          gameVarietyId,
          pointsId: pointsId || null,
          teamPlayerId: teamPlayerId || null,
        };
      },
    }
  ),
};
module.exports = { dictionaries };
"""

        payload = bookmaker_xyz._parse_official_dictionaries_via_node(module_source, timeout=5)

        self.assertIn("6983", payload["outcomes"])
        self.assertEqual(payload["outcomes"]["6983"]["marketId"], 19)
        self.assertEqual(payload["outcomes"]["6983"]["gamePeriodId"], 76)

    def test_load_dictionaries_prefers_official_package_source(self) -> None:
        dictionaries = {"marketNames": {"1-1-1": "Full Time Result"}}

        original_cache = dict(bookmaker_xyz.DICTIONARY_CACHE)
        try:
            bookmaker_xyz.DICTIONARY_CACHE["expires_at"] = 0.0
            bookmaker_xyz.DICTIONARY_CACHE["data"] = None
            bookmaker_xyz.DICTIONARY_CACHE["source"] = ""
            with (
                patch.object(
                    bookmaker_xyz,
                    "_load_dictionaries_from_official_package",
                    return_value=(
                        dictionaries,
                        {
                            "cache": "miss",
                            "source": "https://registry.npmjs.org/@azuro-org/dictionaries/-/dictionaries-3.0.28.tgz",
                            "source_strategy": "official_package",
                        },
                    ),
                ),
                patch.object(
                    bookmaker_xyz,
                    "_retrying_get",
                    side_effect=AssertionError("website fallback should not run"),
                ),
            ):
                loaded, meta = bookmaker_xyz._load_dictionaries(
                    retries=0,
                    backoff_seconds=0.0,
                    timeout=1,
                )
        finally:
            bookmaker_xyz.DICTIONARY_CACHE.clear()
            bookmaker_xyz.DICTIONARY_CACHE.update(original_cache)

        self.assertEqual(loaded, dictionaries)
        self.assertEqual(meta.get("source_strategy"), "official_package")

    def test_load_dictionaries_uses_disk_cache_before_parsing_const_bundle(self) -> None:
        const_url = "https://bookmaker.xyz/assets/const-test.js"
        dictionaries = {
            "marketNames": {"1-1-1": "Winner"},
            "outcomes": {"29": {"marketId": 1}},
        }
        home_html = '<script src="/assets/const-test.js"></script>'

        original_cache = dict(bookmaker_xyz.DICTIONARY_CACHE)
        try:
            bookmaker_xyz.DICTIONARY_CACHE["expires_at"] = 0.0
            bookmaker_xyz.DICTIONARY_CACHE["data"] = None
            bookmaker_xyz.DICTIONARY_CACHE["source"] = ""
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch.object(bookmaker_xyz, "BOOKMAKER_XYZ_DICT_DISK_CACHE_DIR", temp_dir):
                    bookmaker_xyz._persist_dictionaries_to_disk_cache(const_url, dictionaries)
                    with (
                        patch.object(
                            bookmaker_xyz,
                            "_load_dictionaries_from_official_package",
                            return_value=(None, {"cache": "miss", "source": "", "error": "official unavailable"}),
                        ),
                        patch.object(bookmaker_xyz, "_retrying_get", return_value=(home_html, 0)),
                        patch.object(bookmaker_xyz, "_parse_dictionaries_via_node", side_effect=AssertionError("node should not run")),
                    ):
                        loaded, meta = bookmaker_xyz._load_dictionaries(
                            retries=0,
                            backoff_seconds=0.0,
                            timeout=1,
                        )
        finally:
            bookmaker_xyz.DICTIONARY_CACHE.clear()
            bookmaker_xyz.DICTIONARY_CACHE.update(original_cache)

        self.assertEqual(loaded, dictionaries)
        self.assertEqual(meta.get("cache"), "disk_hit")
        self.assertEqual(meta.get("source"), const_url)

    def test_load_dictionaries_ignores_expired_disk_cache(self) -> None:
        const_url = "https://bookmaker.xyz/assets/const-test.js"
        dictionaries = {"marketNames": {"1-1-1": "Winner"}}
        home_html = '<script src="/assets/const-test.js"></script>'
        const_js = "const x={ marketNames: { '1-1-1': 'Winner' } };"

        original_cache = dict(bookmaker_xyz.DICTIONARY_CACHE)
        try:
            bookmaker_xyz.DICTIONARY_CACHE["expires_at"] = 0.0
            bookmaker_xyz.DICTIONARY_CACHE["data"] = None
            bookmaker_xyz.DICTIONARY_CACHE["source"] = ""
            with tempfile.TemporaryDirectory() as temp_dir:
                with (
                    patch.object(bookmaker_xyz, "BOOKMAKER_XYZ_DICT_DISK_CACHE_DIR", temp_dir),
                    patch.object(bookmaker_xyz, "BOOKMAKER_XYZ_DICT_CACHE_TTL_RAW", "1"),
                ):
                    bookmaker_xyz._persist_dictionaries_to_disk_cache(const_url, dictionaries)
                    cache_path = bookmaker_xyz._dictionary_disk_cache_path(const_url)
                    stale_time = time.time() - 10
                    os.utime(cache_path, (stale_time, stale_time))
                    with (
                        patch.object(
                            bookmaker_xyz,
                            "_load_dictionaries_from_official_package",
                            return_value=(None, {"cache": "miss", "source": "", "error": "official unavailable"}),
                        ),
                        patch.object(bookmaker_xyz, "_retrying_get", side_effect=[(home_html, 0), (const_js, 0)]),
                        patch.object(bookmaker_xyz, "_extract_x_object_literal", return_value="{}"),
                        patch.object(bookmaker_xyz, "_parse_dictionaries_via_node", return_value=dictionaries) as mocked_parse,
                    ):
                        loaded, meta = bookmaker_xyz._load_dictionaries(
                            retries=0,
                            backoff_seconds=0.0,
                            timeout=1,
                        )
        finally:
            bookmaker_xyz.DICTIONARY_CACHE.clear()
            bookmaker_xyz.DICTIONARY_CACHE.update(original_cache)

        self.assertEqual(loaded, dictionaries)
        self.assertEqual(meta.get("cache"), "miss")
        self.assertEqual(meta.get("source"), const_url)
        self.assertTrue(mocked_parse.called)

    def test_load_dictionaries_ignores_empty_outcomes_disk_cache(self) -> None:
        const_url = "https://bookmaker.xyz/assets/const-test.js"
        dictionaries = {"marketNames": {"1-1-1": "Winner"}, "outcomes": {"29": {"marketId": 1}}}
        invalid_payload = {"marketNames": {"1-1-1": "Winner"}, "outcomes": {}}

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(bookmaker_xyz, "BOOKMAKER_XYZ_DICT_DISK_CACHE_DIR", temp_dir):
                cache_path = bookmaker_xyz._dictionary_disk_cache_path(const_url)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as handle:
                    json_payload = {
                        "saved_at": "2026-03-13T00:00:00Z",
                        "source_url": const_url,
                        "dictionaries": invalid_payload,
                    }
                    json.dump(json_payload, handle)

                with (
                    patch.object(bookmaker_xyz, "_retrying_get", return_value=('<script src="/assets/const-test.js"></script>', 0)),
                    patch.object(bookmaker_xyz, "_extract_x_object_literal", return_value="{}"),
                    patch.object(bookmaker_xyz, "_parse_dictionaries_via_node", return_value=dictionaries),
                    patch.object(
                        bookmaker_xyz,
                        "_load_dictionaries_from_official_package",
                        return_value=(None, {"cache": "miss", "source": "", "error": "official unavailable"}),
                    ),
                ):
                    loaded, meta = bookmaker_xyz._load_dictionaries(
                        retries=0,
                        backoff_seconds=0.0,
                        timeout=1,
                    )

        self.assertEqual(loaded, dictionaries)
        self.assertEqual(meta.get("cache"), "miss")


    def test_extract_x_object_literal_supports_current_const_bundle_marker(self) -> None:
        js_source = 'prefix B={marketNames:{"4-1-2":"Team Total Goals","3-1-2":"Handicap"},outcomes:{"17252":{selectionId:8}}};suffix'

        payload = bookmaker_xyz._extract_x_object_literal(js_source)

        self.assertEqual(
            payload,
            '{marketNames:{"4-1-2":"Team Total Goals","3-1-2":"Handicap"},outcomes:{"17252":{selectionId:8}}}',
        )

    def test_load_dictionaries_auto_merges_site_market_names_when_official_package_is_missing_keys(self) -> None:
        official_dictionaries = {
            "marketNames": {"4-1-1": "Total Goals"},
            "outcomes": {"23": {"marketId": 4}},
            "selections": {},
            "teamPlayers": {},
            "points": {},
        }
        site_dictionaries = {
            "marketNames": {"4-1-1": "Total Goals", "4-1-2": "Team Total Goals", "3-1-2": "Handicap"},
            "outcomes": {"23": {"marketId": 4}, "17252": {"marketId": 3}},
            "selections": {},
            "teamPlayers": {},
            "points": {},
        }
        home_html = '<script src="/assets/const-test.js"></script>'
        const_js = 'prefix B={marketNames:{"4-1-1":"Total Goals","4-1-2":"Team Total Goals","3-1-2":"Handicap"},outcomes:{"23":{marketId:4},"17252":{marketId:3}},selections:{},teamPlayers:{},points:{}};suffix'

        original_cache = dict(bookmaker_xyz.DICTIONARY_CACHE)
        try:
            bookmaker_xyz.DICTIONARY_CACHE["expires_at"] = 0.0
            bookmaker_xyz.DICTIONARY_CACHE["data"] = None
            bookmaker_xyz.DICTIONARY_CACHE["source"] = ""
            with (
                patch.object(
                    bookmaker_xyz,
                    "_load_dictionaries_from_official_package",
                    return_value=(
                        official_dictionaries,
                        {
                            "cache": "miss",
                            "source": "https://registry.npmjs.org/@azuro-org/dictionaries/-/dictionaries-3.0.28.tgz",
                            "source_strategy": "official_package",
                        },
                    ),
                ),
                patch.object(bookmaker_xyz, "_retrying_get", side_effect=[(home_html, 0), (const_js, 0)]),
                patch.object(bookmaker_xyz, "_parse_dictionaries_via_node", return_value=site_dictionaries),
            ):
                loaded, meta = bookmaker_xyz._load_dictionaries(
                    retries=0,
                    backoff_seconds=0.0,
                    timeout=1,
                )
        finally:
            bookmaker_xyz.DICTIONARY_CACHE.clear()
            bookmaker_xyz.DICTIONARY_CACHE.update(original_cache)

        self.assertEqual(loaded["marketNames"].get("4-1-2"), "Team Total Goals")
        self.assertEqual(loaded["marketNames"].get("3-1-2"), "Handicap")
        self.assertEqual(loaded["outcomes"].get("17252"), {"marketId": 3})
        self.assertEqual(meta.get("source_strategy"), "official_package")


    def test_normalize_snapshot_to_events_does_not_fallback_h2h_for_untyped_two_way_condition(self) -> None:
        conditions = [
            {
                "__chain_id": "137",
                "state": "Active",
                "game": {
                    "gameId": "game-nyr-dal",
                    "slug": "rangers-vs-stars",
                    "title": "New York Rangers vs Dallas Stars",
                    "startsAt": "2026-04-11T20:59:00Z",
                    "participants": [
                        {"name": "New York Rangers"},
                        {"name": "Dallas Stars"},
                    ],
                    "sport": {"slug": "ice-hockey", "name": "Ice Hockey"},
                    "league": {"slug": "nhl", "name": "NHL"},
                    "country": {"slug": "united-states", "name": "United States"},
                },
                "outcomes": [
                    {"outcomeId": "177", "odds": "1.18", "sortOrder": 1},
                    {"outcomeId": "178", "odds": "4.35", "sortOrder": 2},
                ],
            }
        ]
        dictionaries = {
            "marketNames": {"4-1-1": "Total Goals"},
            "outcomes": {
                "177": {"selectionId": 9, "marketId": 4, "gamePeriodId": 1, "gameTypeId": 1, "pointsId": 299, "teamPlayerId": None},
                "178": {"selectionId": 10, "marketId": 4, "gamePeriodId": 1, "gameTypeId": 1, "pointsId": 299, "teamPlayerId": None},
            },
            "selections": {"9": "Over", "10": "Under"},
            "teamPlayers": {},
            "points": {"299": "4"},
        }

        events, stats = bookmaker_xyz._normalize_snapshot_to_events(
            conditions=conditions,
            sport_key="icehockey_nhl",
            requested_markets={"h2h"},
            dictionaries=dictionaries,
        )

        self.assertEqual(events, [])
        self.assertEqual(stats.get("fallback_h2h_used_count"), 0)

    def test_game_live_state_payload_does_not_use_starts_at_as_quote_update_time(self) -> None:
        payload = bookmaker_xyz._game_live_state_payload(
            {
                "state": "Active",
                "startsAt": "2026-04-19T23:00:00Z",
            }
        )

        self.assertEqual(payload.get("status"), "active")
        self.assertNotIn("updated_at", payload)


if __name__ == "__main__":
    unittest.main()

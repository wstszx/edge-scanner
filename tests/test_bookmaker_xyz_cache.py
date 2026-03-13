import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from providers import bookmaker_xyz


class BookmakerXyzCacheTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

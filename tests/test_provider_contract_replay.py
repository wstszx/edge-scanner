from __future__ import annotations

import copy
import datetime as real_dt
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from providers import betdex, bookmaker_xyz, polymarket, sx_bet


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "provider_contract_replay.json"


def _load_fixture() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deepcopy(value):
    return copy.deepcopy(value)


def _unique_keys(markets):
    seen = set()
    keys = []
    for market in markets or []:
        key = str((market or {}).get("key") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


async def _fake_shared_client(*args, **kwargs):
    return object()


class _FrozenPolymarketDateTime(real_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        frozen = real_dt.datetime(2026, 3, 14, 8, 0, 0, tzinfo=real_dt.timezone.utc)
        if tz is None:
            return frozen.replace(tzinfo=None)
        return frozen.astimezone(tz)


class ProviderContractReplayTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()["providers"]

    def setUp(self) -> None:
        betdex.ACCESS_TOKEN_CACHE["token"] = ""
        betdex.ACCESS_TOKEN_CACHE["expires_at"] = 0.0

        polymarket.disable_scan_cache()
        polymarket.EVENTS_CACHE["expires_at"] = 0.0
        polymarket.EVENTS_CACHE["events"] = []
        polymarket.SPORT_TAG_CACHE["expires_at"] = 0.0
        polymarket.SPORT_TAG_CACHE["mapping"] = {}
        polymarket.CLOB_BOOK_CACHE["expires_at"] = 0.0
        polymarket.CLOB_BOOK_CACHE["entries"] = {}

        sx_bet.UPCOMING_CACHE["expires_at"] = 0.0
        sx_bet.UPCOMING_CACHE["entries"] = {}
        sx_bet.ODDS_CACHE["expires_at"] = 0.0
        sx_bet.ODDS_CACHE["entries"] = {}
        sx_bet.ORDERS_CACHE["expires_at"] = 0.0
        sx_bet.ORDERS_CACHE["entries"] = {}

        bookmaker_xyz.disable_scan_cache()
        bookmaker_xyz.CONDITIONS_CACHE["expires_at"] = 0.0
        bookmaker_xyz.CONDITIONS_CACHE["conditions"] = []
        bookmaker_xyz.CONDITIONS_CACHE["meta"] = {}
        bookmaker_xyz.CONDITIONS_CACHE["key"] = ""
        bookmaker_xyz.DICTIONARY_CACHE["expires_at"] = 0.0
        bookmaker_xyz.DICTIONARY_CACHE["data"] = None
        bookmaker_xyz.DICTIONARY_CACHE["source"] = ""

    async def test_polymarket_replays_recorded_sports_and_events_payload(self) -> None:
        provider_fixture = self.fixture["polymarket"]

        async def _fake_request_json_async(client, path, params, retries, backoff_seconds):
            if path == "sports":
                return _deepcopy(provider_fixture["requests"]["sports"]), 0
            if path == "events":
                return _deepcopy(provider_fixture["requests"]["events"]), 0
            raise AssertionError(f"Unexpected Polymarket request path: {path}")

        with (
            patch.object(polymarket, "get_shared_client", new=_fake_shared_client),
            patch.object(polymarket, "_request_json_async", side_effect=_fake_request_json_async),
            patch.object(polymarket, "_websocket_realtime_enabled", return_value=False),
            patch.object(polymarket, "_http_clob_depth_enabled", return_value=False),
            patch.object(polymarket.dt, "datetime", _FrozenPolymarketDateTime),
        ):
            events = await polymarket.fetch_events_async(
                provider_fixture["sport_key"],
                ["h2h"],
                ["us"],
            )

        self.assertEqual(len(events), provider_fixture["expected"]["event_count"])
        event = events[0]
        self.assertEqual(event["home_team"], provider_fixture["expected"]["home_team"])
        self.assertEqual(event["away_team"], provider_fixture["expected"]["away_team"])
        market_keys = _unique_keys(event["bookmakers"][0]["markets"])
        self.assertEqual(sorted(market_keys), sorted(provider_fixture["expected"]["market_keys"]))
        self.assertEqual(polymarket.fetch_events_async.last_stats.get("events_returned_count"), 1)
        self.assertEqual(polymarket.fetch_events_async.last_stats.get("pages_fetched"), 1)

    async def test_betdex_replays_recorded_session_events_markets_and_prices(self) -> None:
        provider_fixture = self.fixture["betdex"]

        async def _fake_request_json_async(
            client,
            url,
            params,
            access_token,
            retries,
            backoff_seconds,
            timeout,
        ):
            if url == betdex._session_url():
                return _deepcopy(provider_fixture["requests"]["session"]), 0
            if url.endswith("/events"):
                return _deepcopy(provider_fixture["requests"]["events"]), 0
            if url.endswith("/markets"):
                return _deepcopy(provider_fixture["requests"]["markets"]), 0
            if url.endswith("/market-prices"):
                return _deepcopy(provider_fixture["requests"]["prices"]), 0
            raise AssertionError(f"Unexpected BetDEX request URL: {url}")

        with (
            patch.object(betdex, "get_shared_client", new=_fake_shared_client),
            patch.object(betdex, "_request_json_async", side_effect=_fake_request_json_async),
        ):
            events = await betdex.fetch_events_async(
                provider_fixture["sport_key"],
                ["h2h", "spreads", "totals"],
                ["us"],
            )

        self.assertEqual(len(events), provider_fixture["expected"]["event_count"])
        event = events[0]
        self.assertEqual(event["home_team"], provider_fixture["expected"]["home_team"])
        self.assertEqual(event["away_team"], provider_fixture["expected"]["away_team"])
        market_keys = _unique_keys(event["bookmakers"][0]["markets"])
        self.assertEqual(sorted(market_keys), sorted(provider_fixture["expected"]["market_keys"]))
        live_state = event.get("live_state") or {}
        self.assertEqual(live_state.get("status"), "scheduled")
        self.assertFalse(live_state.get("is_live"))
        self.assertEqual(live_state.get("in_play_status"), "preplay")
        self.assertTrue(event["bookmakers"][0].get("live_state"))
        first_market = (event["bookmakers"][0].get("markets") or [])[0]
        first_outcome = (first_market.get("outcomes") or [])[0]
        self.assertTrue(first_outcome.get("last_updated"))
        self.assertEqual(betdex.fetch_events_async.last_stats.get("events_returned_count"), 1)
        self.assertEqual(betdex.fetch_events_async.last_stats.get("session_cache"), "miss")

    async def test_sx_bet_replays_recorded_summary_best_odds_and_orders_payload(self) -> None:
        provider_fixture = self.fixture["sx_bet"]

        async def _fake_request_json_async(client, path, params=None, retries=None, backoff_seconds=None):
            if path.startswith("summary/upcoming/"):
                return _deepcopy(provider_fixture["requests"]["summary"]), 0
            if path == "orders/odds/best":
                return _deepcopy(provider_fixture["requests"]["best_odds"]), 0
            if path == "orders":
                return _deepcopy(provider_fixture["requests"]["orders"]), 0
            raise AssertionError(f"Unexpected SX Bet request path: {path}")

        with (
            patch.object(sx_bet, "get_shared_client", new=_fake_shared_client),
            patch.object(sx_bet, "_request_json_async", side_effect=_fake_request_json_async),
            patch.object(sx_bet, "_fixture_source_mode", return_value="summary"),
        ):
            events = await sx_bet.fetch_events_async(
                provider_fixture["sport_key"],
                ["h2h", "spreads", "totals"],
                ["us"],
            )

        self.assertEqual(len(events), provider_fixture["expected"]["event_count"])
        event = events[0]
        self.assertEqual(event["home_team"], provider_fixture["expected"]["home_team"])
        self.assertEqual(event["away_team"], provider_fixture["expected"]["away_team"])
        market_keys = _unique_keys(event["bookmakers"][0]["markets"])
        self.assertEqual(sorted(market_keys), sorted(provider_fixture["expected"]["market_keys"]))
        normalized_markets = event["bookmakers"][0].get("markets") or []
        self.assertTrue(
            any(
                any(outcome.get("last_updated") for outcome in (market.get("outcomes") or []))
                for market in normalized_markets
            )
        )
        self.assertEqual(sx_bet.fetch_events_async.last_stats.get("events_returned_count"), 1)
        self.assertEqual(sx_bet.fetch_events_async.last_stats.get("fixture_source_used"), "summary")

    async def test_bookmaker_xyz_replays_recorded_market_manager_and_dictionary_payload(self) -> None:
        provider_fixture = self.fixture["bookmaker_xyz"]

        async def _fake_load_dictionaries_async(client, retries, backoff_seconds, timeout):
            return _deepcopy(provider_fixture["raw"]["dictionaries"]), _deepcopy(provider_fixture["raw"]["dictionary_meta"])

        async def _fake_load_market_manager_snapshot_async(client, sport_key, retries, backoff_seconds, timeout):
            return _deepcopy(provider_fixture["raw"]["conditions"]), _deepcopy(provider_fixture["raw"]["payload_meta"])

        with (
            patch.object(bookmaker_xyz, "get_shared_client", new=_fake_shared_client),
            patch.object(bookmaker_xyz, "_load_dictionaries_async", side_effect=_fake_load_dictionaries_async),
            patch.object(
                bookmaker_xyz,
                "_load_market_manager_snapshot_async",
                side_effect=_fake_load_market_manager_snapshot_async,
            ),
        ):
            events = await bookmaker_xyz.fetch_events_async(
                provider_fixture["sport_key"],
                ["h2h", "spreads", "totals"],
                ["us"],
            )

        self.assertEqual(len(events), provider_fixture["expected"]["event_count"])
        event = events[0]
        self.assertEqual(event["home_team"], provider_fixture["expected"]["home_team"])
        self.assertEqual(event["away_team"], provider_fixture["expected"]["away_team"])
        market_keys = _unique_keys(event["bookmakers"][0]["markets"])
        self.assertEqual(sorted(market_keys), sorted(provider_fixture["expected"]["market_keys"]))
        self.assertEqual(bookmaker_xyz.fetch_events_async.last_stats.get("events_returned_count"), 1)
        self.assertTrue(bookmaker_xyz.fetch_events_async.last_stats.get("dictionary_loaded"))


if __name__ == "__main__":
    unittest.main()

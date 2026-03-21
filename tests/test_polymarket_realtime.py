import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from providers import polymarket


def _sample_event(slug: str = "game-1") -> dict:
    return {
        "id": "evt-1",
        "gameId": 90091303,
        "slug": slug,
        "title": "Team A vs Team B",
        "active": True,
        "closed": False,
        "archived": False,
        "markets": [
            {
                "question": "Team A vs Team B",
                "outcomes": '["Team A", "Team B"]',
                "outcomePrices": '["0.4", "0.6"]',
                "clobTokenIds": '["token-a", "token-b"]',
                "volumeNum": "100",
                "active": True,
                "closed": False,
                "archived": False,
            }
        ],
        "tags": [
            {"id": "1", "slug": "sports"},
            {"id": "999", "slug": "nba"},
        ],
        "startTime": "2026-03-10T00:00:00Z",
    }


class _FakeRealtimeManager:
    def __init__(self, depth_map=None, quote_map=None, sports_map=None):
        self.depth_map = dict(depth_map or {})
        self.quote_map = dict(quote_map or {})
        self.sports_map = dict(sports_map or {})
        self.subscriptions = []
        self.started = False

    def ensure_started(self) -> bool:
        self.started = True
        return True

    def subscribe_assets(self, asset_ids):
        normalized = [str(asset_id) for asset_id in asset_ids]
        self.subscriptions.append(normalized)
        return len(normalized)

    def wait_for_assets(self, asset_ids, timeout_seconds: float) -> bool:
        return True

    def wait_for_quotes(self, asset_ids, timeout_seconds: float) -> bool:
        return bool(self.get_quote_map(asset_ids, max_age_seconds=max(timeout_seconds, 0.0)))

    def get_depth_map(self, asset_ids, max_age_seconds: float):
        return {
            str(asset_id): self.depth_map[str(asset_id)]
            for asset_id in asset_ids
            if str(asset_id) in self.depth_map
        }

    def get_quote_map(self, asset_ids, max_age_seconds: float):
        return {
            str(asset_id): dict(self.quote_map[str(asset_id)])
            for asset_id in asset_ids
            if str(asset_id) in self.quote_map
        }

    def get_sport_result(self, slug):
        return self.sports_map.get(str(slug))

    def get_sport_results(self, max_age_seconds: float = 0.0):
        return {
            str(key): dict(value)
            for key, value in self.sports_map.items()
            if isinstance(value, dict)
        }

    def snapshot(self):
        return {
            "started": self.started,
            "market_connected": True,
            "sports_connected": True,
            "subscribed_assets": sum(len(batch) for batch in self.subscriptions),
            "market_messages_received": 4,
            "sports_messages_received": 2,
        }


class PolymarketRealtimeTests(unittest.TestCase):
    def test_realtime_manager_updates_ask_depth_from_book_and_price_change(self) -> None:
        manager = polymarket.PolymarketRealtimeManager()

        asyncio.run(
            manager._handle_market_message(
                {
                    "event_type": "book",
                    "asset_id": "token-a",
                    "bids": [{"price": "0.45", "size": "8"}],
                    "asks": [
                        {"price": "0.55", "size": "10"},
                        {"price": "0.60", "size": "5"},
                    ],
                }
            )
        )
        asyncio.run(
            manager._handle_market_message(
                {
                    "event_type": "price_change",
                    "asset_id": "token-a",
                    "price_changes": [
                        {"side": "SELL", "price": "0.60", "size": "0"},
                        {"side": "SELL", "price": "0.58", "size": "8"},
                    ],
                }
            )
        )

        depth_map = manager.get_depth_map(["token-a"], max_age_seconds=30.0)
        self.assertAlmostEqual(depth_map["token-a"], 10.14, places=6)

    def test_realtime_manager_returns_best_ask_quote_from_book_levels(self) -> None:
        manager = polymarket.PolymarketRealtimeManager()

        asyncio.run(
            manager._handle_market_message(
                {
                    "event_type": "book",
                    "asset_id": "token-a",
                    "asks": [
                        {"price": "0.62", "size": "4"},
                        {"price": "0.58", "size": "10"},
                        {"price": "0.60", "size": "5"},
                    ],
                }
            )
        )

        quote_map = manager.get_quote_map(["token-a"], max_age_seconds=30.0)
        self.assertAlmostEqual(quote_map["token-a"]["decimal_odds"], 1.724138, places=6)
        self.assertAlmostEqual(quote_map["token-a"]["stake"], 5.8, places=6)

    def test_sports_result_is_tradeable_rejects_terminal_status(self) -> None:
        self.assertFalse(polymarket._sports_result_is_tradeable({"status": "final"}))
        self.assertFalse(polymarket._sports_result_is_tradeable({"ended": True}))
        self.assertTrue(polymarket._sports_result_is_tradeable({"status": "live"}))

    def test_event_and_market_remain_tradeable_when_accepting_orders_after_end(self) -> None:
        now_utc = polymarket.dt.datetime.now(polymarket.dt.timezone.utc)
        event = _sample_event()
        event["endDate"] = "2026-03-10T00:00:00Z"
        event["markets"][0]["endDate"] = "2026-03-10T00:00:00Z"
        event["markets"][0]["acceptingOrders"] = True

        self.assertTrue(polymarket._market_is_tradeable(event["markets"][0], now_utc))
        self.assertTrue(polymarket._event_is_tradeable(event, now_utc))

    def test_realtime_manager_reads_shared_snapshot_when_not_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner_path = os.path.join(temp_dir, "owner.json")
            snapshot_path = os.path.join(temp_dir, "snapshot.json")
            with open(owner_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "owner_id": "other-owner",
                        "pid": 9999,
                        "heartbeat_at": time.time(),
                        "saved_at": "2026-03-07T00:00:00Z",
                    },
                    handle,
                )
            with open(snapshot_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "status": {
                            "started": True,
                            "owner_active": True,
                            "market_connected": True,
                            "sports_connected": False,
                            "market_books_cached": 1,
                            "subscribed_assets": 1,
                        },
                        "market_books": {
                            "token-a": {
                                "asks": {"0.6000000000": 10.0},
                                "updated_at": time.time(),
                            }
                        },
                        "sports_results": {
                            "game-1": {"slug": "game-1", "status": "live", "updated_at": time.time()}
                        },
                    },
                    handle,
                )

            with patch.object(polymarket, "POLYMARKET_REALTIME_SHARED_DIR", temp_dir):
                manager = polymarket.PolymarketRealtimeManager()
                started = manager.ensure_started()
                depth_map = manager.get_depth_map(["token-a"], max_age_seconds=30.0)
                sport_state = manager.get_sport_result("game-1")
                snapshot = manager.snapshot()

            self.assertFalse(started)
            self.assertEqual(depth_map["token-a"], 6.0)
            self.assertEqual(sport_state["status"], "live")
            self.assertTrue(snapshot.get("shared_snapshot_loaded"))
            self.assertTrue(snapshot.get("market_connected"))

    def test_realtime_status_marks_stale_shared_snapshot_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = os.path.join(temp_dir, "snapshot.json")
            with open(snapshot_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "saved_at": "2026-03-06T00:00:00Z",
                        "status": {
                            "started": True,
                            "owner_active": True,
                            "market_connected": True,
                            "sports_connected": False,
                            "market_last_message_age_seconds": 5,
                        },
                        "market_books": {},
                        "sports_results": {},
                    },
                    handle,
                )

            with patch.object(polymarket, "POLYMARKET_REALTIME_SHARED_DIR", temp_dir):
                manager = polymarket.PolymarketRealtimeManager()
                with patch.object(polymarket, "_get_realtime_manager", return_value=manager):
                    status = polymarket.realtime_status()

        self.assertTrue(status["status"].get("shared_snapshot_loaded"))
        self.assertTrue(status["status"].get("shared_snapshot_stale"))
        self.assertFalse(status["status"].get("shared_owner_active"))
        self.assertFalse(status["status"].get("market_connected"))
        self.assertFalse(status.get("ready"))

    def test_prune_realtime_state_unsubscribes_expired_assets(self) -> None:
        manager = polymarket.PolymarketRealtimeManager()
        now = time.time()
        manager._owner_active = True
        manager._subscribed_asset_ids = {"token-a": now - 5000, "token-b": now}
        manager._market_books = {
            "token-a": {"asks": {"0.5": 1.0}, "updated_at": now - 5000},
            "token-b": {"asks": {"0.5": 2.0}, "updated_at": now},
        }
        unsubscribed = []

        async def _fake_unsubscribe(asset_ids):
            unsubscribed.extend(asset_ids)

        manager._send_market_unsubscribe = _fake_unsubscribe  # type: ignore[method-assign]

        asyncio.run(manager._prune_realtime_state())

        self.assertNotIn("token-a", manager._subscribed_asset_ids)
        self.assertNotIn("token-a", manager._market_books)
        self.assertIn("token-a", unsubscribed)

    def test_market_subscription_uses_incremental_subscribe_operation(self) -> None:
        sent_payloads = []

        class _FakeWebSocket:
            async def send(self, payload):
                sent_payloads.append(json.loads(payload))

        manager = polymarket.PolymarketRealtimeManager()
        manager._market_ws = _FakeWebSocket()
        manager._market_connected = True
        manager._market_subscription_initialized = True

        asyncio.run(manager._send_market_subscription(["token-a", "token-b"]))
        asyncio.run(manager._send_market_unsubscribe(["token-a"]))

        self.assertEqual(
            sent_payloads,
            [
                {
                    "assets_ids": ["token-a", "token-b"],
                    "operation": "subscribe",
                    "custom_feature_enabled": True,
                },
                {
                    "assets_ids": ["token-a"],
                    "operation": "unsubscribe",
                },
            ],
        )

    def test_handle_market_message_prefers_payload_timestamp(self) -> None:
        manager = polymarket.PolymarketRealtimeManager()

        asyncio.run(
            manager._handle_market_message(
                {
                    "event_type": "book",
                    "asset_id": "token-a",
                    "bids": [],
                    "asks": [{"price": "0.45", "size": "10"}],
                    "timestamp": 1773593001000,
                }
            )
        )

        self.assertAlmostEqual(manager._market_books["token-a"]["updated_at"], 1773593001.0, places=6)

    def test_event_live_state_payload_promotes_live_game_state(self) -> None:
        payload = polymarket._event_live_state_payload(
            event={},
            realtime_state={
                "gameId": 90091303,
                "live": True,
                "score": "2-1",
                "period": "2H",
                "eventState": {
                    "updatedAt": "2026-03-21T14:06:38.623182769Z",
                    "elapsed": "61",
                    "live": True,
                },
            },
        )

        self.assertEqual(payload["status"], "live")
        self.assertTrue(payload["is_live"])
        self.assertEqual(payload["score"], "2-1")
        self.assertEqual(payload["period"], "2H")
        self.assertEqual(payload["elapsed"], "61")


class PolymarketFetchRealtimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_events_async_prefers_realtime_quotes_before_rest(self) -> None:
        manager = _FakeRealtimeManager(
            quote_map={
                "token-a": {
                    "decimal_odds": 1.818182,
                    "stake": 12.0,
                    "quote_source": "ws_book_best_ask",
                    "observed_at": 1773593000.0,
                    "updated_at": 1773593000.0,
                }
            }
        )
        rest_calls = []

        async def _fake_load_clob_quote_map_async(**kwargs):
            rest_calls.append(list(kwargs.get("token_ids") or []))
            return (
                {
                    "token-b": {
                        "decimal_odds": 1.538462,
                        "stake": 15.0,
                        "quote_source": "clob_book_best_ask",
                        "observed_at": 1773593001.0,
                        "updated_at": 1773593001.0,
                    }
                },
                {
                    "token_count_requested": 1,
                    "token_count_considered": 1,
                    "token_count_truncated": 0,
                    "books_fetched": 1,
                    "books_with_quotes": 1,
                    "book_errors": 0,
                    "retries_used": 0,
                },
            )

        with (
            patch.object(polymarket, "_websocket_realtime_enabled", return_value=True),
            patch.object(polymarket, "_market_websocket_enabled", return_value=True),
            patch.object(polymarket, "_sports_websocket_enabled", return_value=False),
            patch.object(polymarket, "_ws_warmup_seconds", return_value=0.0),
            patch.object(polymarket, "_get_realtime_manager", return_value=manager),
            patch.object(polymarket, "_load_sport_tag_mapping_async", new=AsyncMock(return_value={})),
            patch.object(
                polymarket,
                "_load_active_game_events_async",
                new=AsyncMock(
                    return_value=(
                        [_sample_event()],
                        {"cache": "miss", "pages_fetched": 1, "retries_used": 0},
                    )
                ),
            ),
            patch.object(polymarket, "_load_clob_quote_map_async", side_effect=_fake_load_clob_quote_map_async),
        ):
            events = await polymarket.fetch_events_async(
                "basketball_nba",
                ["h2h"],
                ["us"],
                bookmakers=["polymarket"],
            )

        self.assertEqual(rest_calls, [["token-b"]])
        self.assertEqual(len(events), 1)
        outcomes = events[0]["bookmakers"][0]["markets"][0]["outcomes"]
        self.assertAlmostEqual(outcomes[0]["price"], 1.818182, places=6)
        self.assertAlmostEqual(outcomes[1]["price"], 1.538462, places=6)
        self.assertEqual(outcomes[0]["stake"], 12.0)
        self.assertEqual(outcomes[1]["stake"], 15.0)
        self.assertEqual(outcomes[0]["observed_at"], 1773593000.0)
        self.assertEqual(outcomes[1]["observed_at"], 1773593001.0)
        self.assertEqual(outcomes[0]["last_updated"], 1773593000.0)
        self.assertEqual(outcomes[1]["last_updated"], 1773593001.0)
        stats = polymarket.fetch_events_async.last_stats
        self.assertEqual(stats.get("realtime_market_books_hit"), 1)
        self.assertEqual(stats.get("realtime_market_books_missed"), 1)
        self.assertEqual(stats.get("clob_books_fetched"), 1)
        self.assertEqual(stats.get("clob_books_with_quotes"), 2)

    async def test_fetch_events_async_uses_game_id_for_sports_ws_state(self) -> None:
        manager = _FakeRealtimeManager(
            sports_map={
                "90091303": {
                    "gameId": 90091303,
                    "live": True,
                    "score": "2-1",
                    "period": "2H",
                    "eventState": {
                        "updatedAt": "2026-03-21T14:06:38.623182769Z",
                        "elapsed": "61",
                        "live": True,
                    },
                }
            }
        )

        with (
            patch.object(polymarket, "_websocket_realtime_enabled", return_value=True),
            patch.object(polymarket, "_market_websocket_enabled", return_value=False),
            patch.object(polymarket, "_sports_websocket_enabled", return_value=True),
            patch.object(polymarket, "_get_realtime_manager", return_value=manager),
            patch.object(polymarket, "_load_sport_tag_mapping_async", new=AsyncMock(return_value={})),
            patch.object(
                polymarket,
                "_load_active_game_events_async",
                new=AsyncMock(
                    return_value=(
                        [_sample_event()],
                        {"cache": "miss", "pages_fetched": 1, "retries_used": 0},
                    )
                ),
            ),
            patch.object(
                polymarket,
                "_load_clob_quote_map_async",
                new=AsyncMock(
                    return_value=(
                        {
                            "token-a": {
                                "decimal_odds": 1.818182,
                                "stake": 12.0,
                                "quote_source": "clob_book_best_ask",
                            },
                            "token-b": {
                                "decimal_odds": 1.538462,
                                "stake": 15.0,
                                "quote_source": "clob_book_best_ask",
                            },
                        },
                        {
                            "token_count_requested": 2,
                            "token_count_considered": 2,
                            "token_count_truncated": 0,
                            "books_fetched": 2,
                            "books_with_quotes": 2,
                            "book_errors": 0,
                            "retries_used": 0,
                        },
                    )
                ),
            ),
        ):
            events = await polymarket.fetch_events_async(
                "basketball_nba",
                ["h2h"],
                ["us"],
                bookmakers=["polymarket"],
            )

        self.assertEqual(len(events), 1)
        live_state = events[0]["live_state"]
        self.assertEqual(live_state["status"], "live")
        self.assertTrue(live_state["is_live"])
        self.assertEqual(live_state["score"], "2-1")
        self.assertEqual(live_state["elapsed"], "61")
        stats = polymarket.fetch_events_async.last_stats
        self.assertEqual(stats.get("realtime_sports_state_hits"), 1)

    async def test_fetch_events_async_supplements_missing_live_event_from_sports_ws_game_id(self) -> None:
        base_event = _sample_event(slug="other-game")
        base_event["id"] = "evt-base"
        base_event["gameId"] = 11111111

        live_event = _sample_event(slug="live-game")
        live_event["id"] = "evt-live"
        live_event["gameId"] = 90091303
        live_event["title"] = "Team C vs Team D"
        live_event["endDate"] = "2026-03-10T00:00:00Z"
        live_event["markets"] = [
            {
                "question": "Team C vs Team D",
                "outcomes": '["Team C", "Team D"]',
                "outcomePrices": '["0.45", "0.55"]',
                "clobTokenIds": '["token-c", "token-d"]',
                "volumeNum": "100",
                "endDate": "2026-03-10T00:00:00Z",
                "active": True,
                "closed": False,
                "archived": False,
                "acceptingOrders": True,
            }
        ]

        manager = _FakeRealtimeManager(
            sports_map={
                "90091303": {
                    "gameId": 90091303,
                    "live": True,
                    "score": "2-1",
                    "period": "2H",
                    "eventState": {
                        "updatedAt": "2026-03-21T14:06:38.623182769Z",
                        "elapsed": "61",
                        "live": True,
                    },
                    "leagueAbbreviation": "nba",
                },
                "10077242": {
                    "gameId": 10077242,
                    "status": "InProgress",
                    "homeTeam": "NYM",
                    "awayTeam": "HOU",
                    "leagueAbbreviation": "mlb",
                }
            }
        )

        supplemental_loader = AsyncMock(
            return_value=(
                [live_event],
                {
                    "game_ids_requested": 1,
                    "lookups": 1,
                    "events_added": 1,
                    "retries_used": 0,
                    "lookup_errors": 0,
                },
            )
        )

        with (
            patch.object(polymarket, "_websocket_realtime_enabled", return_value=True),
            patch.object(polymarket, "_market_websocket_enabled", return_value=False),
            patch.object(polymarket, "_sports_websocket_enabled", return_value=True),
            patch.object(polymarket, "_get_realtime_manager", return_value=manager),
            patch.object(polymarket, "_load_sport_tag_mapping_async", new=AsyncMock(return_value={})),
            patch.object(
                polymarket,
                "_load_active_game_events_async",
                new=AsyncMock(
                    return_value=(
                        [base_event],
                        {"cache": "miss", "pages_fetched": 1, "retries_used": 0},
                    )
                ),
            ),
            patch.object(polymarket, "_load_game_events_by_ids_async", new=supplemental_loader),
            patch.object(
                polymarket,
                "_load_clob_quote_map_async",
                new=AsyncMock(
                    return_value=(
                        {
                            "token-c": {
                                "decimal_odds": 2.222222,
                                "stake": 10.0,
                                "quote_source": "clob_book_best_ask",
                            },
                            "token-d": {
                                "decimal_odds": 1.818182,
                                "stake": 12.0,
                                "quote_source": "clob_book_best_ask",
                            },
                        },
                        {
                            "token_count_requested": 2,
                            "token_count_considered": 2,
                            "token_count_truncated": 0,
                            "books_fetched": 2,
                            "books_with_quotes": 2,
                            "book_errors": 0,
                            "retries_used": 0,
                        },
                    )
                ),
            ),
        ):
            events = await polymarket.fetch_events_async(
                "basketball_nba",
                ["h2h"],
                ["us"],
                bookmakers=["polymarket"],
            )

        live_ids = [event["id"] for event in events]
        self.assertIn("evt-live", live_ids)
        live_event_out = next(event for event in events if event["id"] == "evt-live")
        self.assertEqual(live_event_out["live_state"]["status"], "live")
        self.assertEqual(supplemental_loader.await_args.kwargs.get("game_ids"), ["90091303"])
        stats = polymarket.fetch_events_async.last_stats
        self.assertEqual(stats.get("realtime_sports_game_ids_observed"), 1)
        self.assertEqual(stats.get("realtime_sports_event_lookups"), 1)
        self.assertEqual(stats.get("realtime_sports_events_supplemented"), 1)
        self.assertEqual(stats.get("realtime_sports_state_hits"), 1)

    async def test_fetch_events_async_only_prefetches_match_relevant_clob_tokens(self) -> None:
        event = _sample_event()
        event["markets"] = [
            {
                "question": "Will Team C win?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.2", "0.8"]',
                "clobTokenIds": '["junk-1", "junk-2"]',
                "active": True,
                "closed": False,
                "archived": False,
            },
            {
                "question": "Will Team D win?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.3", "0.7"]',
                "clobTokenIds": '["junk-3", "junk-4"]',
                "active": True,
                "closed": False,
                "archived": False,
            },
            *event["markets"],
        ]

        requested_batches = []

        async def _fake_load_clob_quote_map_async(**kwargs):
            requested_batches.append(list(kwargs.get("token_ids") or []))
            return (
                {
                    "token-a": {
                        "decimal_odds": 1.818182,
                        "stake": 12.0,
                        "quote_source": "clob_book_best_ask",
                    },
                    "token-b": {
                        "decimal_odds": 1.538462,
                        "stake": 15.0,
                        "quote_source": "clob_book_best_ask",
                    },
                },
                {
                    "token_count_requested": 2,
                    "token_count_considered": 2,
                    "token_count_truncated": 0,
                    "books_fetched": 2,
                    "books_with_quotes": 2,
                    "book_errors": 0,
                    "retries_used": 0,
                },
            )

        with (
            patch.object(polymarket, "POLYMARKET_CLOB_MAX_BOOKS_RAW", "2"),
            patch.object(polymarket, "_websocket_realtime_enabled", return_value=False),
            patch.object(polymarket, "_load_sport_tag_mapping_async", new=AsyncMock(return_value={})),
            patch.object(
                polymarket,
                "_load_active_game_events_async",
                new=AsyncMock(
                    return_value=(
                        [event],
                        {"cache": "miss", "pages_fetched": 1, "retries_used": 0},
                    )
                ),
            ),
            patch.object(polymarket, "_load_clob_quote_map_async", side_effect=_fake_load_clob_quote_map_async),
        ):
            events = await polymarket.fetch_events_async(
                "basketball_nba",
                ["h2h"],
                ["us"],
                bookmakers=["polymarket"],
            )

        self.assertEqual(requested_batches, [["token-a", "token-b"]])
        outcomes = events[0]["bookmakers"][0]["markets"][0]["outcomes"]
        self.assertEqual(
            [outcome.get("quote_source") for outcome in outcomes],
            ["clob_book_best_ask", "clob_book_best_ask"],
        )
        stats = polymarket.fetch_events_async.last_stats
        self.assertEqual(stats.get("clob_tokens_requested"), 2)
        self.assertEqual(stats.get("clob_tokens_truncated"), 0)

    async def test_fetch_events_async_filters_terminal_sports_ws_state(self) -> None:
        manager = _FakeRealtimeManager(sports_map={"game-1": {"slug": "game-1", "status": "final"}})

        with (
            patch.object(polymarket, "_websocket_realtime_enabled", return_value=True),
            patch.object(polymarket, "_market_websocket_enabled", return_value=False),
            patch.object(polymarket, "_sports_websocket_enabled", return_value=True),
            patch.object(polymarket, "_get_realtime_manager", return_value=manager),
            patch.object(polymarket, "_load_sport_tag_mapping_async", new=AsyncMock(return_value={})),
            patch.object(
                polymarket,
                "_load_active_game_events_async",
                new=AsyncMock(
                    return_value=(
                        [_sample_event()],
                        {"cache": "miss", "pages_fetched": 1, "retries_used": 0},
                    )
                ),
            ),
            patch.object(polymarket, "_load_clob_quote_map_async", new=AsyncMock(return_value=({}, {}))),
        ):
            events = await polymarket.fetch_events_async(
                "basketball_nba",
                ["h2h"],
                ["us"],
                bookmakers=["polymarket"],
            )

        self.assertEqual(events, [])
        stats = polymarket.fetch_events_async.last_stats
        self.assertEqual(stats.get("realtime_sports_state_hits"), 1)
        self.assertEqual(stats.get("realtime_sports_state_filtered_count"), 1)


class PolymarketSportsResultMatchingTests(unittest.TestCase):
    def test_sports_result_matches_sport_filters_cross_sport_live_results(self) -> None:
        self.assertTrue(
            polymarket._sports_result_matches_sport(
                {"leagueAbbreviation": "cwbb"},
                "basketball_ncaab",
            )
        )
        self.assertFalse(
            polymarket._sports_result_matches_sport(
                {"leagueAbbreviation": "mlb"},
                "basketball_ncaab",
            )
        )

    def test_sports_result_matches_event_with_college_basketball_abbreviations(self) -> None:
        payload = {
            "leagueAbbreviation": "cbb",
            "homeTeam": "MICH",
            "awayTeam": "STLOU",
        }
        event = {
            "title": "Saint Louis Billikens vs. Michigan Wolverines",
            "tags": [{"slug": "cbb"}],
        }

        self.assertTrue(polymarket._sports_result_matches_event(payload, event, "basketball_ncaab"))

    def test_sports_result_matches_event_with_ohio_state_abbreviation(self) -> None:
        payload = {
            "leagueAbbreviation": "cwbb",
            "homeTeam": "OSU",
            "awayTeam": "HOWARD",
        }
        event = {
            "title": "Howard Bison vs. Ohio State Buckeyes (W)",
            "tags": [{"slug": "basketball"}],
        }

        self.assertTrue(polymarket._sports_result_matches_event(payload, event, "basketball_ncaab"))


if __name__ == "__main__":
    unittest.main()

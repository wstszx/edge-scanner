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
    def __init__(self, depth_map=None, sports_map=None):
        self.depth_map = dict(depth_map or {})
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

    def get_depth_map(self, asset_ids, max_age_seconds: float):
        return {
            str(asset_id): self.depth_map[str(asset_id)]
            for asset_id in asset_ids
            if str(asset_id) in self.depth_map
        }

    def get_sport_result(self, slug):
        return self.sports_map.get(str(slug))

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

    def test_sports_result_is_tradeable_rejects_terminal_status(self) -> None:
        self.assertFalse(polymarket._sports_result_is_tradeable({"status": "final"}))
        self.assertFalse(polymarket._sports_result_is_tradeable({"ended": True}))
        self.assertTrue(polymarket._sports_result_is_tradeable({"status": "live"}))

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


class PolymarketFetchRealtimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_events_async_prefers_realtime_depth_before_rest(self) -> None:
        manager = _FakeRealtimeManager(depth_map={"token-a": 12.0})
        rest_calls = []

        async def _fake_load_clob_depth_map_async(**kwargs):
            rest_calls.append(list(kwargs.get("token_ids") or []))
            return (
                {"token-b": 15.0},
                {
                    "token_count_requested": 1,
                    "token_count_considered": 1,
                    "token_count_truncated": 0,
                    "books_fetched": 1,
                    "books_with_depth": 1,
                    "book_errors": 0,
                    "retries_used": 0,
                },
            )

        with (
            patch.object(polymarket, "_websocket_realtime_enabled", return_value=True),
            patch.object(polymarket, "_market_websocket_enabled", return_value=True),
            patch.object(polymarket, "_sports_websocket_enabled", return_value=False),
            patch.object(polymarket, "_http_clob_depth_enabled", return_value=True),
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
            patch.object(polymarket, "_load_clob_depth_map_async", side_effect=_fake_load_clob_depth_map_async),
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
        self.assertEqual(outcomes[0]["stake"], 12.0)
        self.assertEqual(outcomes[1]["stake"], 15.0)
        stats = polymarket.fetch_events_async.last_stats
        self.assertEqual(stats.get("realtime_market_books_hit"), 1)
        self.assertEqual(stats.get("realtime_market_books_missed"), 1)
        self.assertEqual(stats.get("clob_books_fetched"), 1)

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
            patch.object(polymarket, "_load_clob_depth_map_async", new=AsyncMock(return_value=({}, {}))),
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


if __name__ == "__main__":
    unittest.main()

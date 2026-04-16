from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from providers import sx_bet


class _FakeRealtimeMessage:
    def __init__(self, data) -> None:
        self.data = data


class _FakeRealtimeManager:
    def __init__(self, odds_map=None):
        self.odds_map = dict(odds_map or {})
        self.started = False
        self.wait_calls: list[float] = []
        self.merge_calls: list[tuple[str, dict]] = []

    def ensure_started(self) -> bool:
        self.started = True
        return True

    def wait_until_ready(self, timeout_seconds: float) -> bool:
        self.wait_calls.append(float(timeout_seconds))
        return True

    def snapshot(self) -> dict:
        return {
            "started": self.started,
            "connected": True,
            "messages_received": 3,
            "best_odds_cached": len(self.odds_map),
            "last_message_age_seconds": 0.2,
        }

    def get_best_odds_map(self, market_hashes, max_age_seconds: float) -> dict:
        _ = max_age_seconds
        return {
            str(market_hash): dict(self.odds_map[str(market_hash)])
            for market_hash in market_hashes
            if str(market_hash) in self.odds_map
        }

    def merge_best_odds_map(self, odds_map, source: str = "snapshot") -> int:
        merged = 0
        normalized = {}
        for market_hash, payload in (odds_map or {}).items():
            if not isinstance(payload, dict):
                continue
            normalized[str(market_hash)] = dict(payload)
            self.odds_map[str(market_hash)] = dict(payload)
            merged += 1
        self.merge_calls.append((source, normalized))
        return merged


class _FakeAsyncConnectionState:
    def __init__(self, value: str = "connected") -> None:
        self.value = value


class _FakeAsyncConnection:
    def __init__(self) -> None:
        self.state = _FakeAsyncConnectionState()
        self.error_reason = None


class _FakeAsyncChannel:
    def __init__(self, manager: sx_bet.SXBetRealtimeManager, rows) -> None:
        self._manager = manager
        self._rows = rows
        self.subscribed = False
        self.unsubscribed = False

    async def subscribe(self, listener) -> None:
        self.subscribed = True
        listener(_FakeRealtimeMessage(self._rows))
        asyncio.get_running_loop().call_later(0.05, self._manager._stop_requested.set)

    def unsubscribe(self, *args, **kwargs) -> None:
        _ = args, kwargs
        self.unsubscribed = True


class _FakeAsyncRealtimeClient:
    def __init__(self, manager: sx_bet.SXBetRealtimeManager, rows) -> None:
        self.connection = _FakeAsyncConnection()
        self._channel = _FakeAsyncChannel(manager, rows)
        self.channels = self
        self.closed = False
        self.channel_names: list[str] = []

    def get(self, channel_name: str):
        self.channel_names.append(channel_name)
        return self._channel

    async def close(self) -> None:
        self.closed = True
        self.connection.state.value = "closed"


class SXBetRealtimeTests(unittest.TestCase):
    def test_realtime_manager_maps_maker_side_to_taker_outcomes(self) -> None:
        manager = sx_bet.SXBetRealtimeManager()

        manager._handle_best_odds_message(
            _FakeRealtimeMessage(
                [
                    {
                        "marketHash": "m1",
                        "isMakerBettingOutcomeOne": False,
                        "percentageOdds": "46250000000000000000",
                        "updatedAt": 1773592013439,
                    },
                    {
                        "marketHash": "m1",
                        "isMakerBettingOutcomeOne": True,
                        "percentageOdds": "51375000000000000000",
                        "updatedAt": 1773592882711,
                    },
                ]
            )
        )

        odds_map = manager.get_best_odds_map(["m1"], max_age_seconds=0.0)
        self.assertAlmostEqual(odds_map["m1"]["odds_one"], 1.860465, places=6)
        self.assertAlmostEqual(odds_map["m1"]["odds_two"], 2.056555, places=6)
        self.assertEqual(odds_map["m1"]["updated_at_one"], 1773592013439)
        self.assertEqual(odds_map["m1"]["updated_at_two"], 1773592882711)

    def test_fetch_events_async_prefers_realtime_best_odds_in_live_mode(self) -> None:
        fixtures = [
            {
                "eventId": "sx-live-1",
                "id": "sx-live-1",
                "teamOne": "Home Team",
                "teamTwo": "Away Team",
                "leagueLabel": "NBA",
                "gameTime": "2026-03-16T19:00:00Z",
                "markets": [
                    {
                        "type": 226,
                        "teamOneName": "Home Team",
                        "teamTwoName": "Away Team",
                        "outcomeOneName": "Home Team",
                        "outcomeTwoName": "Away Team",
                        "bestOddsOutcomeOne": 2.25,
                        "bestOddsOutcomeTwo": 1.70,
                        "marketHash": "sx-live-h2h",
                    }
                ],
            }
        ]
        realtime_manager = _FakeRealtimeManager(
            odds_map={
                "sx-live-h2h": {
                    "odds_one": 2.4,
                    "odds_two": 1.63,
                    "updated_at_one": 1773593000000,
                    "updated_at_two": 1773593001000,
                    "observed_at_one": 1773593002.0,
                    "observed_at_two": 1773593002.5,
                    "source_one": "ws",
                    "source_two": "ws",
                }
            }
        )

        async def _fake_load_upcoming_fixtures_async(*args, **kwargs):
            return (
                fixtures,
                {
                    "cache": "miss",
                    "pages_fetched": 1,
                    "retries_used": 0,
                    "fixture_source": "summary",
                },
            )

        async def _fake_load_best_odds_map_async(*args, **kwargs):
            return (
                {
                    "sx-live-h2h": {
                        "odds_one": 2.35,
                        "odds_two": 1.66,
                        "updated_at_one": "older-one",
                        "updated_at_two": "older-two",
                    }
                },
                0,
                {
                    "best_odds_items": 1,
                    "best_odds_null_count": 0,
                    "best_odds_with_any_odds": 1,
                    "best_odds_with_both_odds": 1,
                },
            )

        async def _fake_load_best_stake_map_async(*args, **kwargs):
            return (
                {"sx-live-h2h": (100.0, 120.0)},
                0,
                {
                    "orders_rows": 1,
                    "orders_missing_market_hash": 0,
                },
            )

        async def _fake_shared_client(*args, **kwargs):
            return object()

        with (
            patch.object(sx_bet, "_sx_best_odds_ws_enabled", return_value=True),
            patch.object(sx_bet, "_get_realtime_manager", return_value=realtime_manager),
            patch.object(sx_bet, "get_shared_client", new=_fake_shared_client),
            patch.object(sx_bet, "_load_upcoming_fixtures_async", new=_fake_load_upcoming_fixtures_async),
            patch.object(sx_bet, "_load_best_odds_map_async", new=_fake_load_best_odds_map_async),
            patch.object(sx_bet, "_load_best_stake_map_async", new=_fake_load_best_stake_map_async),
        ):
            events = asyncio.run(
                sx_bet.fetch_events_async(
                    "basketball_nba",
                    ["h2h"],
                    ["us"],
                    context={"live": True},
                )
            )

        outcomes = (events[0]["bookmakers"][0]["markets"][0].get("outcomes") or [])
        self.assertEqual(outcomes[0]["price"], 2.4)
        self.assertEqual(outcomes[1]["price"], 1.63)
        self.assertEqual(outcomes[0]["last_updated"], 1773593000000)
        self.assertEqual(outcomes[1]["last_updated"], 1773593001000)
        self.assertEqual(outcomes[0]["observed_at"], 1773593002.0)
        self.assertEqual(outcomes[1]["observed_at"], 1773593002.5)
        self.assertEqual(outcomes[0]["quote_source"], "ws")
        self.assertEqual(outcomes[1]["quote_source"], "ws")
        stats = sx_bet.fetch_events_async.last_stats
        self.assertEqual(stats.get("realtime_odds_hits"), 1)
        self.assertEqual(stats.get("odds_lookup_requested"), 0)

    def test_fetch_events_async_live_mode_skips_markets_active_fixture_without_live_evidence(self) -> None:
        fixtures = [
            {
                "eventId": "sx-future-1",
                "id": "sx-future-1",
                "teamOne": "Home Team",
                "teamTwo": "Away Team",
                "leagueLabel": "NBA",
                "gameTime": "2099-01-01T00:00:00Z",
                "live_state": {"is_live": False, "status": "scheduled"},
                "markets": [
                    {
                        "type": 226,
                        "teamOneName": "Home Team",
                        "teamTwoName": "Away Team",
                        "outcomeOneName": "Home Team",
                        "outcomeTwoName": "Away Team",
                        "bestOddsOutcomeOne": 2.1,
                        "bestOddsOutcomeTwo": 1.8,
                        "marketHash": "sx-future-h2h",
                    }
                ],
            }
        ]

        async def _fake_load_upcoming_fixtures_async(*args, **kwargs):
            return (
                fixtures,
                {
                    "cache": "miss",
                    "pages_fetched": 1,
                    "retries_used": 0,
                    "fixture_source": "markets_active",
                },
            )

        async def _fake_load_fixture_status_map_async(*args, **kwargs):
            return {}

        async def _fake_load_live_scores_map_async(*args, **kwargs):
            return {}

        async def _fake_load_best_odds_map_async(*args, **kwargs):
            return (
                {
                    "sx-future-h2h": {
                        "odds_one": 2.1,
                        "odds_two": 1.8,
                        "updated_at_one": 1773593000000,
                        "updated_at_two": 1773593001000,
                    }
                },
                0,
                {
                    "best_odds_items": 1,
                    "best_odds_null_count": 0,
                    "best_odds_with_any_odds": 1,
                    "best_odds_with_both_odds": 1,
                },
            )

        async def _fake_load_best_stake_map_async(*args, **kwargs):
            return (
                {"sx-future-h2h": (100.0, 100.0)},
                0,
                {
                    "orders_rows": 1,
                    "orders_missing_market_hash": 0,
                },
            )

        async def _fake_shared_client(*args, **kwargs):
            return object()

        with (
            patch.object(sx_bet, "get_shared_client", new=_fake_shared_client),
            patch.object(sx_bet, "_load_upcoming_fixtures_async", new=_fake_load_upcoming_fixtures_async),
            patch.object(sx_bet, "_load_fixture_status_map_async", new=_fake_load_fixture_status_map_async),
            patch.object(sx_bet, "_load_live_scores_map_async", new=_fake_load_live_scores_map_async),
            patch.object(sx_bet, "_load_best_odds_map_async", new=_fake_load_best_odds_map_async),
            patch.object(sx_bet, "_load_best_stake_map_async", new=_fake_load_best_stake_map_async),
        ):
            events = asyncio.run(
                sx_bet.fetch_events_async(
                    "basketball_nba",
                    ["h2h"],
                    ["us"],
                    context={"live": True},
                )
            )

        self.assertEqual(events, [])

    def test_realtime_manager_snapshot_merge_preserves_newer_ws_odds(self) -> None:
        manager = sx_bet.SXBetRealtimeManager()

        manager._handle_best_odds_message(
            _FakeRealtimeMessage(
                [
                    {
                        "marketHash": "m2",
                        "isMakerBettingOutcomeOne": False,
                        "percentageOdds": "46250000000000000000",
                        "updatedAt": 1773592013439,
                    }
                ]
            )
        )
        merged = manager.merge_best_odds_map(
            {
                "m2": {
                    "odds_one": 1.72,
                    "updated_at_one": 1773591000000,
                },
                "m3": {
                    "odds_one": 2.15,
                    "odds_two": 1.8,
                    "updated_at_one": 1773593000000,
                    "updated_at_two": 1773593001000,
                },
            },
            source="rest_snapshot",
        )

        odds_map = manager.get_best_odds_map(["m2", "m3"], max_age_seconds=0.0)
        self.assertEqual(merged, 1)
        self.assertAlmostEqual(odds_map["m2"]["odds_one"], 1.860465, places=6)
        self.assertEqual(odds_map["m2"]["source_one"], "ws")
        self.assertAlmostEqual(odds_map["m3"]["odds_one"], 2.15, places=6)
        self.assertAlmostEqual(odds_map["m3"]["odds_two"], 1.8, places=6)
        self.assertEqual(odds_map["m3"]["source_one"], "rest_snapshot")
        self.assertEqual(odds_map["m3"]["source_two"], "rest_snapshot")
        self.assertIn("m3", manager.get_best_odds_map(["m3"], max_age_seconds=5.0))

    def test_fetch_events_async_seeds_realtime_cache_from_rest_snapshot(self) -> None:
        fixtures = [
            {
                "eventId": "sx-live-2",
                "id": "sx-live-2",
                "teamOne": "Home Team",
                "teamTwo": "Away Team",
                "leagueLabel": "NBA",
                "gameTime": "2026-03-16T19:00:00Z",
                "markets": [
                    {
                        "type": 226,
                        "teamOneName": "Home Team",
                        "teamTwoName": "Away Team",
                        "outcomeOneName": "Home Team",
                        "outcomeTwoName": "Away Team",
                        "bestOddsOutcomeOne": 2.05,
                        "bestOddsOutcomeTwo": 1.8,
                        "marketHash": "sx-live-seed",
                    }
                ],
            }
        ]
        realtime_manager = _FakeRealtimeManager()

        async def _fake_load_upcoming_fixtures_async(*args, **kwargs):
            return (
                fixtures,
                {
                    "cache": "miss",
                    "pages_fetched": 1,
                    "retries_used": 0,
                    "fixture_source": "summary",
                },
            )

        async def _fake_load_best_odds_map_async(*args, **kwargs):
            return (
                {
                    "sx-live-seed": {
                        "odds_one": 2.21,
                        "odds_two": 1.74,
                        "updated_at_one": 1773594000000,
                        "updated_at_two": 1773594000500,
                        "observed_at_one": 1773594002.0,
                        "observed_at_two": 1773594002.5,
                        "source_one": "rest_snapshot",
                        "source_two": "rest_snapshot",
                    }
                },
                0,
                {
                    "best_odds_items": 1,
                    "best_odds_null_count": 0,
                    "best_odds_with_any_odds": 1,
                    "best_odds_with_both_odds": 1,
                },
            )

        async def _fake_load_best_stake_map_async(*args, **kwargs):
            return (
                {"sx-live-seed": (90.0, 95.0)},
                0,
                {
                    "orders_rows": 1,
                    "orders_missing_market_hash": 0,
                },
            )

        async def _fake_shared_client(*args, **kwargs):
            return object()

        with (
            patch.object(sx_bet, "_sx_best_odds_ws_enabled", return_value=True),
            patch.object(sx_bet, "_get_realtime_manager", return_value=realtime_manager),
            patch.object(sx_bet, "get_shared_client", new=_fake_shared_client),
            patch.object(sx_bet, "_load_upcoming_fixtures_async", new=_fake_load_upcoming_fixtures_async),
            patch.object(sx_bet, "_load_best_odds_map_async", new=_fake_load_best_odds_map_async),
            patch.object(sx_bet, "_load_best_stake_map_async", new=_fake_load_best_stake_map_async),
        ):
            events = asyncio.run(
                sx_bet.fetch_events_async(
                    "basketball_nba",
                    ["h2h"],
                    ["us"],
                    context={"live": True},
                )
            )

        outcomes = (events[0]["bookmakers"][0]["markets"][0].get("outcomes") or [])
        self.assertEqual(outcomes[0]["price"], 2.21)
        self.assertEqual(outcomes[1]["price"], 1.74)
        self.assertEqual(outcomes[0]["observed_at"], 1773594002.0)
        self.assertEqual(outcomes[1]["observed_at"], 1773594002.5)
        self.assertEqual(outcomes[0]["quote_source"], "rest_snapshot")
        self.assertEqual(outcomes[1]["quote_source"], "rest_snapshot")
        self.assertEqual(realtime_manager.merge_calls[0][0], "rest_snapshot")
        self.assertIn("sx-live-seed", realtime_manager.odds_map)
        stats = sx_bet.fetch_events_async.last_stats
        self.assertEqual(stats.get("realtime_stream_hits"), 0)
        self.assertEqual(stats.get("realtime_snapshot_seeded"), 1)
        self.assertEqual(stats.get("realtime_odds_hits"), 1)
        self.assertEqual(stats.get("realtime_odds_missed"), 0)

    def test_ensure_started_runs_async_client_in_thread(self) -> None:
        manager = sx_bet.SXBetRealtimeManager()
        fake_client = _FakeAsyncRealtimeClient(
            manager,
            rows=[
                {
                    "marketHash": "thread-live-1",
                    "isMakerBettingOutcomeOne": False,
                    "percentageOdds": "46250000000000000000",
                    "updatedAt": 1773592013439,
                }
            ],
        )

        with (
            patch.object(sx_bet, "_sx_best_odds_ws_enabled", return_value=True),
            patch.object(manager, "_create_ably_client", return_value=fake_client),
        ):
            self.assertTrue(manager.ensure_started())
            self.assertTrue(manager.wait_until_ready(1.0))
            manager.stop(timeout_seconds=2.0)

        odds_map = manager.get_best_odds_map(["thread-live-1"], max_age_seconds=0.0)
        self.assertAlmostEqual(odds_map["thread-live-1"]["odds_one"], 1.860465, places=6)
        self.assertTrue(fake_client._channel.subscribed)
        self.assertTrue(fake_client._channel.unsubscribed)
        self.assertTrue(fake_client.closed)
        self.assertIn(f"best_odds:{sx_bet.SX_BET_BASE_TOKEN}", fake_client.channel_names)


if __name__ == "__main__":
    unittest.main()

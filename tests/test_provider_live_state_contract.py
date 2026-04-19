from __future__ import annotations

import unittest
from unittest.mock import patch

import scanner
from providers import sx_bet


class ProviderLiveStateContractTests(unittest.TestCase):
    def _provider_event(self, event_id: str, commence_time: str, live_state) -> dict:
        event = {
            "id": event_id,
            "sport_key": "basketball_nba",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "commence_time": commence_time,
            "bookmakers": [
                {
                    "key": "sx_bet",
                    "title": "SX Bet",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.0},
                                {"name": "Away Team", "price": 1.9},
                            ],
                        }
                    ],
                }
            ],
        }
        if live_state is not None:
            event["live_state"] = live_state
            event["bookmakers"][0]["live_state"] = dict(live_state)
        return event

    def test_sx_bet_explicit_scheduled_state_overrides_active_market_status(self) -> None:
        fixtures, _ = sx_bet._build_fixtures_from_markets_active(
            rows=[
                {
                    "sportId": 1,
                    "sportXeventId": "evt-scheduled",
                    "teamOneName": "Home Team",
                    "teamTwoName": "Away Team",
                    "gameTime": "2026-03-16T10:00:00Z",
                    "type": 226,
                    "marketHash": "mh-scheduled",
                    "outcomeOneName": "Home Team",
                    "outcomeTwoName": "Away Team",
                    "status": "ACTIVE",
                    "liveEnabled": True,
                    "isLive": False,
                }
            ],
            sport_id=1,
            only_main_line=False,
        )

        live_state = fixtures[0].get("live_state")
        event = self._provider_event("evt-scheduled", "2026-03-16T10:00:00Z", live_state)

        self.assertFalse(sx_bet._fixture_has_live_evidence(fixtures[0]))
        self.assertFalse(scanner._event_is_explicitly_live(event))

    def test_sx_bet_explicit_live_state_is_recognized_by_scanner(self) -> None:
        fixtures, _ = sx_bet._build_fixtures_from_markets_active(
            rows=[
                {
                    "sportId": 1,
                    "sportXeventId": "evt-live",
                    "teamOneName": "Home Team",
                    "teamTwoName": "Away Team",
                    "gameTime": "2026-03-16T10:00:00Z",
                    "type": 226,
                    "marketHash": "mh-live",
                    "outcomeOneName": "Home Team",
                    "outcomeTwoName": "Away Team",
                    "status": "ACTIVE",
                    "liveEnabled": True,
                    "isLive": True,
                }
            ],
            sport_id=1,
            only_main_line=False,
        )

        live_state = fixtures[0].get("live_state")
        event = self._provider_event("evt-live", "2026-03-16T10:00:00Z", live_state)

        self.assertTrue(sx_bet._fixture_has_live_evidence(fixtures[0]))
        self.assertTrue(scanner._event_is_explicitly_live(event))

    def test_non_explicit_provider_event_defers_to_scanner_live_time_window(self) -> None:
        fixtures, _ = sx_bet._build_fixtures_from_markets_active(
            rows=[
                {
                    "sportId": 1,
                    "sportXeventId": "evt-window",
                    "teamOneName": "Home Team",
                    "teamTwoName": "Away Team",
                    "gameTime": "2026-03-16T10:00:00Z",
                    "type": 226,
                    "marketHash": "mh-window",
                    "outcomeOneName": "Home Team",
                    "outcomeTwoName": "Away Team",
                    "status": "ACTIVE",
                    "liveEnabled": True,
                }
            ],
            sport_id=1,
            only_main_line=False,
        )

        live_state = fixtures[0].get("live_state")
        recent_event = self._provider_event("evt-recent", "2026-03-16T10:00:30Z", live_state)
        stale_event = self._provider_event("evt-stale", "2026-03-16T09:55:00Z", live_state)

        with (
            patch("scanner.time.time", return_value=1773655230),
            patch("scanner._event_max_past_seconds", return_value=60),
            patch("scanner._live_event_max_future_seconds", return_value=0),
        ):
            filtered, stats = scanner._filter_live_events_for_scan([recent_event, stale_event])

        self.assertIsNone(scanner._event_is_explicitly_live(recent_event))
        self.assertEqual([event.get("id") for event in filtered], ["evt-recent"])
        self.assertEqual(stats.get("dropped_past"), 1)


if __name__ == "__main__":
    unittest.main()

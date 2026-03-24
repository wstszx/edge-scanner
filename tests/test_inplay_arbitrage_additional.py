import unittest
from unittest.mock import patch

import scanner


class InplayArbitrageAdditionalTests(unittest.TestCase):
    @staticmethod
    def _build_live_h2h_game() -> dict:
        return {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "commence_time": "2026-03-20T00:00:00Z",
            "live_state": {"status": "live", "period": "Q2", "score": "55-50", "clock": "05:20"},
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "live_state": {"status": "live", "period": "Q2", "score": "55-50", "clock": "05:20"},
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 2.2, "last_updated": 99},
                                {"name": "Away Team", "price": 1.8, "last_updated": 99},
                            ],
                        }
                    ],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "live_state": {"status": "live", "period": "Q2", "score": "55-50", "clock": "05:05"},
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home Team", "price": 1.8, "last_updated": 99},
                                {"name": "Away Team", "price": 2.2, "last_updated": 99},
                            ],
                        }
                    ],
                },
            ],
        }

    def test_live_quote_age_uses_outcome_timestamp_first(self) -> None:
        game = {"live_state": {"status": "live", "updated_at": 10}, "updated_at": 5}
        bookmaker = {"updated_at": 15}
        market = {"updated_at": 30}
        outcome = {"last_updated": 99}

        age = scanner._live_quote_age_seconds(game, bookmaker, market, outcome, now_epoch=100.0)
        self.assertEqual(age, 1.0)

        with patch.object(scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"):
            self.assertTrue(
                scanner._is_live_quote_fresh(
                    game,
                    bookmaker,
                    market,
                    outcome,
                    scan_mode="live",
                    now_epoch=100.0,
                )
            )

    def test_live_quote_age_prefers_observed_at_over_older_provider_update_time(self) -> None:
        game = {"live_state": {"status": "live", "updated_at": 10}, "updated_at": 5}
        bookmaker = {"updated_at": 15}
        market = {"updated_at": 30, "observed_at": 99}
        outcome = {"last_updated": 60}

        age = scanner._live_quote_age_seconds(game, bookmaker, market, outcome, now_epoch=100.0)
        self.assertEqual(age, 1.0)

        with patch.object(scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"):
            self.assertTrue(
                scanner._is_live_quote_fresh(
                    game,
                    bookmaker,
                    market,
                    outcome,
                    scan_mode="live",
                    now_epoch=100.0,
                )
            )

    def test_live_quote_freshness_boundary_allows_equal_max_age(self) -> None:
        game = {"live_state": {"status": "live"}}
        bookmaker = {}
        market = {}
        outcome = {"last_updated": 95}

        with patch.object(scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"):
            self.assertTrue(
                scanner._is_live_quote_fresh(
                    game,
                    bookmaker,
                    market,
                    outcome,
                    scan_mode="live",
                    now_epoch=100.0,
                )
            )
            self.assertFalse(
                scanner._is_live_quote_fresh(
                    game,
                    bookmaker,
                    market,
                    {"last_updated": 94},
                    scan_mode="live",
                    now_epoch=100.0,
                )
            )

    def test_live_quote_without_any_timestamp_is_treated_as_fresh(self) -> None:
        game = {"live_state": {"status": "live"}}
        bookmaker = {}
        market = {}
        outcome = {"name": "Home Team", "price": 2.1}

        with patch.object(scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"):
            self.assertIsNone(
                scanner._live_quote_age_seconds(
                    game,
                    bookmaker,
                    market,
                    outcome,
                    now_epoch=100.0,
                )
            )
            self.assertTrue(
                scanner._is_live_quote_fresh(
                    game,
                    bookmaker,
                    market,
                    outcome,
                    scan_mode="live",
                    now_epoch=100.0,
                )
            )

    def test_live_quote_age_falls_back_across_market_bookmaker_event_state(self) -> None:
        game = {"live_state": {"status": "live", "updated_at": 96}, "updated_at": 95}
        bookmaker = {"updated_at": 97}
        market = {"updated_at": 98}

        self.assertEqual(
            scanner._live_quote_age_seconds(game, bookmaker, market, outcome={}, now_epoch=100.0),
            2.0,
        )
        self.assertEqual(
            scanner._live_quote_age_seconds(game, bookmaker, market={}, outcome={}, now_epoch=100.0),
            3.0,
        )
        self.assertEqual(
            scanner._live_quote_age_seconds(game, bookmaker={}, market={}, outcome={}, now_epoch=100.0),
            4.0,
        )

    def test_filter_live_events_tracks_terminal_and_non_live_states(self) -> None:
        events = [
            {
                "id": "cancelled-event",
                "commence_time": "2030-01-01T00:00:00Z",
                "live_state": {"status": "cancelled"},
            },
            {
                "id": "scheduled-event",
                "commence_time": "2030-01-01T00:00:00Z",
                "live_state": {"status": "scheduled"},
            },
            {
                "id": "explicit-live-event",
                "commence_time": "2030-01-01T00:00:00Z",
                "live_state": {"status": "live"},
            },
        ]

        with patch("scanner.time.time", return_value=1_700_000_000):
            filtered, stats = scanner._filter_live_events_for_scan(events)

        filtered_ids = {event.get("id") for event in filtered}
        self.assertEqual(filtered_ids, {"explicit-live-event"})
        self.assertEqual(stats.get("dropped_terminal_state"), 1)
        self.assertEqual(stats.get("dropped_not_live_state"), 1)

    def test_filter_live_events_explicit_live_still_respects_future_window(self) -> None:
        events = [
            {
                "id": "explicit-live-within-window",
                "commence_time": "2023-11-14T22:13:30Z",
                "live_state": {"status": "live", "is_live": True},
            },
            {
                "id": "explicit-live-outside-window",
                "commence_time": "2023-11-14T22:50:00Z",
                "live_state": {"status": "live", "is_live": True},
            },
        ]

        with (
            patch("scanner.time.time", return_value=1_700_000_000),
            patch.object(scanner, "LIVE_EVENT_MAX_FUTURE_SECONDS_RAW", "10"),
        ):
            filtered, stats = scanner._filter_live_events_for_scan(events)

        filtered_ids = {event.get("id") for event in filtered}
        self.assertEqual(
            filtered_ids,
            {"explicit-live-within-window", "explicit-live-outside-window"},
        )
        self.assertEqual(stats.get("dropped_future"), 0)
        self.assertEqual(stats.get("suspicious_explicit_live_future"), 1)

    def test_filter_live_events_keeps_merged_event_when_any_bookmaker_is_explicitly_live(self) -> None:
        events = [
            {
                "id": "merged-live-event",
                "commence_time": "2023-11-14T22:13:20Z",
                "live_state": {"status": "scheduled", "is_live": False},
                "bookmakers": [
                    {
                        "key": "book_a",
                        "live_state": {"status": "scheduled", "is_live": False},
                    },
                    {
                        "key": "book_b",
                        "live_state": {"status": "live", "is_live": True},
                    },
                ],
            }
        ]

        with patch("scanner.time.time", return_value=1_700_000_000):
            filtered, stats = scanner._filter_live_events_for_scan(events)

        filtered_ids = {event.get("id") for event in filtered}
        self.assertEqual(filtered_ids, {"merged-live-event"})
        self.assertEqual(stats.get("dropped_not_live_state"), 0)
        self.assertEqual(stats.get("dropped_terminal_state"), 0)

    def test_filter_live_events_drops_future_events_without_explicit_live_state(self) -> None:
        events = [
            {
                "id": "future-no-live-state",
                "commence_time": "2023-11-14T22:20:00Z",
                "bookmakers": [
                    {"key": "book_a"},
                    {"key": "book_b"},
                ],
            }
        ]

        with (
            patch("scanner.time.time", return_value=1_700_000_000),
            patch.object(scanner, "LIVE_EVENT_MAX_FUTURE_SECONDS_RAW", "0"),
        ):
            filtered, stats = scanner._filter_live_events_for_scan(events)

        self.assertEqual(filtered, [])
        self.assertEqual(stats.get("dropped_future"), 1)
        self.assertEqual(stats.get("suspicious_explicit_live_future"), 0)

    def test_live_state_compatibility_enforces_clock_tolerance(self) -> None:
        state_a = {"status": "live", "period": "Q2", "score": "55-50", "clock": "05:00"}
        state_b = {"status": "live", "period": "Q2", "score": "50-55", "clock": "08:10"}

        with patch.object(scanner, "LIVE_STATE_CLOCK_TOLERANCE_SECONDS_RAW", "180"):
            self.assertFalse(scanner._live_states_are_compatible([state_a, state_b], "live"))

        with patch.object(scanner, "LIVE_STATE_CLOCK_TOLERANCE_SECONDS_RAW", "200"):
            self.assertTrue(scanner._live_states_are_compatible([state_a, state_b], "live"))

    def test_collect_market_entries_transitions_when_score_race_resolves(self) -> None:
        game = self._build_live_h2h_game()
        game["bookmakers"][1]["live_state"] = {
            "status": "live",
            "period": "Q2",
            "score": "56-50",
            "clock": "05:12",
        }

        with patch("scanner.time.time", return_value=100.0), patch.object(
            scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"
        ):
            mismatched = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="live",
            )

            game["bookmakers"][1]["live_state"] = {
                "status": "live",
                "period": "Q2",
                "score": "55-50",
                "clock": "05:12",
            }
            resolved = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="live",
            )

        self.assertEqual(mismatched, [])
        self.assertTrue(resolved)

    def test_collect_market_entries_transitions_when_live_state_race_resolves(self) -> None:
        game = self._build_live_h2h_game()
        game["bookmakers"][1]["live_state"] = {
            "status": "live",
            "period": "Q3",
            "score": "55-50",
            "clock": "11:45",
        }

        with patch("scanner.time.time", return_value=100.0), patch.object(
            scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"
        ):
            mismatched = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="live",
            )

            game["bookmakers"][1]["live_state"] = {
                "status": "live",
                "period": "Q2",
                "score": "55-50",
                "clock": "05:12",
            }
            resolved = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="live",
            )

        self.assertEqual(mismatched, [])
        self.assertTrue(resolved)

    def test_collect_market_entries_transitions_from_stale_to_fresh_under_fast_updates(self) -> None:
        game = self._build_live_h2h_game()
        for book in game["bookmakers"]:
            for outcome in book["markets"][0]["outcomes"]:
                outcome["last_updated"] = 90

        with patch("scanner.time.time", return_value=100.0), patch.object(
            scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"
        ):
            stale_entries = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="live",
            )

            game["bookmakers"][0]["markets"][0]["outcomes"][0]["last_updated"] = 99
            game["bookmakers"][0]["markets"][0]["outcomes"][1]["last_updated"] = 99
            game["bookmakers"][1]["markets"][0]["outcomes"][0]["last_updated"] = 98
            game["bookmakers"][1]["markets"][0]["outcomes"][1]["last_updated"] = 98
            fresh_entries = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="live",
            )

        self.assertEqual(stale_entries, [])
        self.assertTrue(fresh_entries)

    def test_filter_live_events_handles_paused_interrupted_and_cancelled(self) -> None:
        events = [
            {
                "id": "paused",
                "commence_time": "2023-11-14T22:13:20Z",
                "live_state": {"status": "paused", "is_live": False},
            },
            {
                "id": "interrupted",
                "commence_time": "2023-11-14T22:13:20Z",
                "live_state": {"status": "interrupted", "is_live": False},
            },
            {
                "id": "cancelled",
                "commence_time": "2023-11-14T22:13:20Z",
                "live_state": {"status": "cancelled"},
            },
            {
                "id": "live-ok",
                "commence_time": "2023-11-14T22:13:20Z",
                "live_state": {"status": "live"},
            },
        ]

        with patch("scanner.time.time", return_value=1_700_000_000):
            filtered, stats = scanner._filter_live_events_for_scan(events)

        filtered_ids = {event.get("id") for event in filtered}
        self.assertEqual(filtered_ids, {"live-ok"})
        self.assertEqual(stats.get("dropped_not_live_state"), 2)
        self.assertEqual(stats.get("dropped_terminal_state"), 1)

    def test_filter_live_events_treats_paused_and_interrupted_without_boolean_flags_as_not_live(self) -> None:
        events = [
            {
                "id": "paused-no-flag",
                "commence_time": "2023-11-14T22:13:20Z",
                "live_state": {"status": "paused"},
            },
            {
                "id": "interrupted-no-flag",
                "commence_time": "2023-11-14T22:13:20Z",
                "live_state": {"status": "interrupted"},
            },
            {
                "id": "live-ok",
                "commence_time": "2023-11-14T22:13:20Z",
                "live_state": {"status": "live"},
            },
        ]

        with patch("scanner.time.time", return_value=1_700_000_000):
            filtered, stats = scanner._filter_live_events_for_scan(events)

        filtered_ids = {event.get("id") for event in filtered}
        self.assertEqual(filtered_ids, {"live-ok"})
        self.assertEqual(stats.get("dropped_not_live_state"), 2)
        self.assertEqual(stats.get("dropped_terminal_state"), 0)

    def test_collect_market_entries_rejects_paused_live_state_without_boolean_flag(self) -> None:
        game = self._build_live_h2h_game()
        game["live_state"] = {
            "status": "paused",
            "period": "Q2",
            "score": "55-50",
            "clock": "05:20",
        }
        for book in game["bookmakers"]:
            book["live_state"] = {
                "status": "paused",
                "period": "Q2",
                "score": "55-50",
                "clock": "05:20",
            }
            for outcome in book["markets"][0]["outcomes"]:
                outcome["last_updated"] = 99

        with patch("scanner.time.time", return_value=100.0), patch.object(
            scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"
        ):
            entries = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="live",
            )

        self.assertEqual(entries, [])

    def test_live_state_explicit_check_prioritizes_non_live_secondary_tokens(self) -> None:
        self.assertFalse(
            scanner._live_state_is_explicitly_live(
                {"status": "live", "is_live": True, "market_status": "suspended"}
            )
        )
        self.assertFalse(
            scanner._live_state_is_explicitly_live(
                {"status": "live", "is_live": True, "provider_status": "postponed"}
            )
        )
        self.assertFalse(
            scanner._live_state_is_explicitly_live(
                {"status": "live", "is_live": True, "provider_status": "abandoned"}
            )
        )

    def test_collect_market_entries_rejects_suspended_market_status_even_if_status_is_live(self) -> None:
        game = self._build_live_h2h_game()
        game["live_state"] = {
            "status": "live",
            "is_live": True,
            "market_status": "suspended",
            "period": "Q2",
            "score": "55-50",
            "clock": "05:20",
        }
        for book in game["bookmakers"]:
            book["live_state"] = {
                "status": "live",
                "is_live": True,
                "market_status": "suspended",
                "period": "Q2",
                "score": "55-50",
                "clock": "05:20",
            }
            for outcome in book["markets"][0]["outcomes"]:
                outcome["last_updated"] = 99

        with patch("scanner.time.time", return_value=100.0), patch.object(
            scanner, "LIVE_QUOTE_MAX_AGE_SECONDS_RAW", "5"
        ):
            entries = scanner._collect_market_entries(
                game,
                market_key="h2h",
                stake_total=100.0,
                commission_rate=0.0,
                scan_mode="live",
            )

        self.assertEqual(entries, [])

    def test_filter_live_events_uses_future_window_when_state_not_explicit(self) -> None:
        events = [
            {
                "id": "within-future-window",
                "commence_time": "2023-11-14T22:13:30Z",
                "live_state": {},
            },
            {
                "id": "outside-future-window",
                "commence_time": "2023-11-14T22:13:32Z",
                "live_state": {},
            },
        ]

        with (
            patch("scanner.time.time", return_value=1_700_000_000),
            patch.object(scanner, "LIVE_EVENT_MAX_FUTURE_SECONDS_RAW", "10"),
        ):
            filtered, stats = scanner._filter_live_events_for_scan(events)

        filtered_ids = {event.get("id") for event in filtered}
        self.assertEqual(filtered_ids, {"within-future-window"})
        self.assertEqual(stats.get("dropped_future"), 1)


if __name__ == "__main__":
    unittest.main()

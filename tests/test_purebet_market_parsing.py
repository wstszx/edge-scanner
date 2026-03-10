from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from providers import purebet


class PurebetMarketParsingTests(unittest.TestCase):
    def _build_one_x_two_market(self, mkt: int, side0_odds: float, side1_odds: float) -> dict:
        return {
            "marketType": "1X2",
            "mkt": mkt,
            "period": 1,
            "side0odds": [{"odds": side0_odds, "stake": 200}],
            "side1odds": [{"odds": side1_odds, "stake": 200}],
        }

    def test_binary_one_x_two_rows_are_rebuilt_as_three_way_market(self) -> None:
        payload = [
            self._build_one_x_two_market(1, 1.38, 3.40),
            self._build_one_x_two_market(2, 5.00, 1.22),
            self._build_one_x_two_market(3, 11.11, 1.08),
        ]
        markets = purebet._normalize_purebet_markets_payload(
            payload,
            "Real Madrid",
            "Getafe",
            ["h2h", "h2h_3_way"],
        )
        three_way = [market for market in markets if market.get("key") == "h2h_3_way"]
        self.assertEqual(len(three_way), 1)
        outcomes = three_way[0].get("outcomes") or []
        self.assertEqual([item.get("name") for item in outcomes], ["Real Madrid", "Draw", "Getafe"])
        self.assertEqual([item.get("price") for item in outcomes], [1.38, 5.0, 11.11])
        # Ensure old invalid two-way interpretation is not produced for the same source rows.
        self.assertFalse(any(market.get("key") == "h2h" for market in markets))

    def test_three_way_rows_are_not_downgraded_to_two_way_market(self) -> None:
        payload = [
            self._build_one_x_two_market(1, 1.38, 3.40),
            self._build_one_x_two_market(2, 5.00, 1.22),
            self._build_one_x_two_market(3, 11.11, 1.08),
        ]
        markets = purebet._normalize_purebet_markets_payload(
            payload,
            "Real Madrid",
            "Getafe",
            ["h2h"],
        )
        self.assertEqual(markets, [])

    def test_v3_event_inline_odds_are_skipped_in_details_mode(self) -> None:
        payload = [
            {
                "event": 123,
                "league": 2196,
                "homeTeam": "Home FC",
                "awayTeam": "Away FC",
                "startTime": 1772481600,
                "odds": [
                    {"odds": 1.9, "market": {"id": "m1", "type": "moneyline", "side": 0, "point": 0}},
                    {"odds": 2.0, "market": {"id": "m1", "type": "moneyline", "side": 1, "point": 0}},
                ],
            }
        ]
        league_map = {"2196": "soccer_spain_la_liga"}

        without_details = purebet._normalize_purebet_v3_events(
            payload,
            "soccer_spain_la_liga",
            ["h2h"],
            league_map=league_map,
            allow_empty_markets=False,
        )
        self.assertEqual(len(without_details), 1)
        self.assertTrue((without_details[0]["bookmakers"][0].get("markets") or []))

        with_details = purebet._normalize_purebet_v3_events(
            payload,
            "soccer_spain_la_liga",
            ["h2h"],
            league_map=league_map,
            allow_empty_markets=True,
        )
        self.assertEqual(len(with_details), 1)
        self.assertEqual(with_details[0]["bookmakers"][0].get("markets"), [])

    def test_fetch_events_bookmaker_filter_is_case_insensitive(self) -> None:
        sample_events = [
            {
                "id": "evt-1",
                "sport_key": "soccer_epl",
                "home_team": "Home FC",
                "away_team": "Away FC",
                "commence_time": "2026-03-10T12:00:00Z",
                "bookmakers": [
                    {
                        "key": "purebet",
                        "title": "Purebet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Home FC", "price": 2.0},
                                    {"name": "Away FC", "price": 2.0},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        with (
            patch.object(purebet, "PUREBET_SOURCE", "file"),
            patch.object(purebet, "_load_event_list", return_value=sample_events),
        ):
            events = purebet.fetch_events(
                sport_key="soccer_epl",
                markets=["h2h"],
                regions=["eu"],
                bookmakers=["PureBet"],
            )

        self.assertEqual(len(events), 1)
        books = events[0].get("bookmakers") or []
        self.assertEqual(len(books), 1)
        self.assertEqual((books[0].get("key") or "").strip().lower(), "purebet")

    def test_fetch_events_async_supports_file_source(self) -> None:
        sample_events = [
            {
                "id": "evt-1",
                "sport_key": "soccer_epl",
                "home_team": "Home FC",
                "away_team": "Away FC",
                "commence_time": "2026-03-10T12:00:00Z",
                "bookmakers": [
                    {
                        "key": "purebet",
                        "title": "Purebet",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Home FC", "price": 2.0},
                                    {"name": "Away FC", "price": 2.0},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        with (
            patch.object(purebet, "PUREBET_SOURCE", "file"),
            patch.object(purebet, "_load_event_list", return_value=sample_events),
        ):
            events = asyncio.run(
                purebet.fetch_events_async(
                    sport_key="soccer_epl",
                    markets=["h2h"],
                    regions=["eu"],
                    bookmakers=["PureBet"],
                )
            )

        self.assertEqual(len(events), 1)
        stats = purebet.fetch_events_async.last_stats
        self.assertEqual(stats.get("events_returned_count"), 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()

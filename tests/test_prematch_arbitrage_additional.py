import math
import unittest
from unittest.mock import patch

import scanner
from arbitrage import arbitrage_roi


class PrematchArbitrageAdditionalTests(unittest.TestCase):
    def test_arbitrage_formula_threshold_matches_roi_sign(self) -> None:
        positive_prices = [2.2, 2.2]
        positive_inverse_sum = sum(1.0 / p for p in positive_prices)
        self.assertLess(positive_inverse_sum, 1.0)
        self.assertGreater(arbitrage_roi(positive_prices), 0.0)

        boundary_prices = [2.0, 2.0]
        boundary_inverse_sum = sum(1.0 / p for p in boundary_prices)
        self.assertAlmostEqual(boundary_inverse_sum, 1.0, places=8)
        self.assertAlmostEqual(arbitrage_roi(boundary_prices), 0.0, places=8)

        negative_prices = [1.95, 1.95]
        negative_inverse_sum = sum(1.0 / p for p in negative_prices)
        self.assertGreater(negative_inverse_sum, 1.0)
        self.assertLess(arbitrage_roi(negative_prices), 0.0)

    def test_collect_market_entries_detects_known_arb_and_keeps_boundary_roi(self) -> None:
        arbitrage_game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "book_home",
                    "title": "Book Home",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 2.2}, {"name": "Away Team", "price": 1.7}]}],
                },
                {
                    "key": "book_away",
                    "title": "Book Away",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 1.7}, {"name": "Away Team", "price": 2.2}]}],
                },
            ],
        }
        non_arb_game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "book_home",
                    "title": "Book Home",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 2.0}, {"name": "Away Team", "price": 1.95}]}],
                },
                {
                    "key": "book_away",
                    "title": "Book Away",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 1.95}, {"name": "Away Team", "price": 2.0}]}],
                },
            ],
        }

        arbitrage_entries = scanner._collect_market_entries(
            arbitrage_game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="prematch",
        )
        non_arb_entries = scanner._collect_market_entries(
            non_arb_game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="prematch",
        )

        self.assertTrue(arbitrage_entries)
        self.assertGreater(arbitrage_entries[0]["roi_percent"], 0.0)
        self.assertTrue(non_arb_entries)
        self.assertAlmostEqual(non_arb_entries[0]["roi_percent"], 0.0, places=2)
        self.assertAlmostEqual(non_arb_entries[0]["stakes"]["guaranteed_profit"], 0.0, places=2)

    def test_calculate_stakes_three_way_arbitrage_distribution_and_profit(self) -> None:
        outcomes = [
            {
                "name": "Home",
                "bookmaker": "Book Home",
                "display_price": 4.0,
                "effective_price": 4.0,
            },
            {
                "name": "Draw",
                "bookmaker": "Book Draw",
                "display_price": 4.0,
                "effective_price": 4.0,
            },
            {
                "name": "Away",
                "bookmaker": "Book Away",
                "display_price": 4.0,
                "effective_price": 4.0,
            },
        ]

        stakes = scanner._calculate_stakes(outcomes, 120.0, price_field="effective_price")
        payouts = [item["payout"] for item in stakes["breakdown"]]
        stake_values = [item["stake"] for item in stakes["breakdown"]]

        self.assertEqual(stakes["total"], 120.0)
        self.assertEqual(len(stakes["breakdown"]), 3)
        self.assertTrue(all(math.isclose(stake, 40.0, rel_tol=0, abs_tol=1e-9) for stake in stake_values))
        self.assertTrue(all(math.isclose(payout, 160.0, rel_tol=0, abs_tol=1e-9) for payout in payouts))
        self.assertAlmostEqual(stakes["guaranteed_profit"], 40.0, places=2)
        self.assertAlmostEqual(stakes["roi_percent"], 33.3333, places=4)

    def test_calculate_stakes_respects_max_stake_limit(self) -> None:
        outcomes = [
            {
                "name": "Home",
                "bookmaker": "Book Home",
                "display_price": 2.2,
                "effective_price": 2.2,
                "max_stake": 20.0,
            },
            {
                "name": "Away",
                "bookmaker": "Book Away",
                "display_price": 2.2,
                "effective_price": 2.2,
                "max_stake": 500.0,
            },
        ]

        stakes = scanner._calculate_stakes(outcomes, 100.0, price_field="effective_price")

        self.assertEqual(stakes["requested_total"], 100.0)
        self.assertEqual(stakes["total"], 40.0)
        self.assertEqual(stakes["max_executable_total"], 40.0)
        self.assertTrue(stakes["limited_by_max_stake"])
        self.assertEqual([item["stake"] for item in stakes["breakdown"]], [20.0, 20.0])
        self.assertAlmostEqual(stakes["guaranteed_profit"], 4.0, places=2)
        self.assertAlmostEqual(stakes["roi_percent"], 10.0, places=4)

    def test_collect_market_entries_selects_highest_profit_cross_book_combo(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 2.5}, {"name": "Away Team", "price": 1.6}]}],
                },
                {
                    "key": "book_b",
                    "title": "Book B",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 1.6}, {"name": "Away Team", "price": 2.5}]}],
                },
                {
                    "key": "book_c",
                    "title": "Book C",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 2.2}, {"name": "Away Team", "price": 2.2}]}],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="prematch",
        )

        self.assertTrue(entries)
        best = entries[0]
        best_by_outcome = {item["outcome"]: item for item in best["best_odds"]}
        self.assertEqual(best_by_outcome["Home Team"]["bookmaker_key"], "book_a")
        self.assertEqual(best_by_outcome["Away Team"]["bookmaker_key"], "book_b")
        self.assertAlmostEqual(best["roi_percent"], 25.0, places=2)
        self.assertAlmostEqual(best["stakes"]["guaranteed_profit"], 25.0, places=2)

    def test_collect_market_entries_rejects_single_bookmaker_only_combo(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "book_a",
                    "title": "Book A",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 2.4}, {"name": "Away Team", "price": 2.4}]}],
                }
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="prematch",
        )

        self.assertEqual(entries, [])

    def test_collect_market_entries_exchange_commission_can_remove_nominal_arbitrage(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "book_home",
                    "title": "Book Home",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 2.0}, {"name": "Away Team", "price": 1.5}]}],
                },
                {
                    "key": "betdex",
                    "title": "BetDEX",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 1.6}, {"name": "Away Team", "price": 2.05}]}],
                },
            ],
        }

        no_commission = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.0,
            scan_mode="prematch",
        )
        with_commission = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.05,
            scan_mode="prematch",
        )

        self.assertTrue(no_commission)
        self.assertGreater(no_commission[0]["roi_percent"], 0.0)
        self.assertTrue(with_commission)
        self.assertLess(with_commission[0]["roi_percent"], 0.0)
        self.assertLess(with_commission[0]["stakes"]["guaranteed_profit"], 0.0)

    def test_collect_market_entries_keeps_negative_roi_cross_book_combos(self) -> None:
        game = {
            "sport_key": "basketball_nba",
            "sport_display": "NBA",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "book_home",
                    "title": "Book Home",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 2.0}, {"name": "Away Team", "price": 1.5}]}],
                },
                {
                    "key": "betdex",
                    "title": "BetDEX",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home Team", "price": 1.6}, {"name": "Away Team", "price": 2.05}]}],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=100.0,
            commission_rate=0.05,
            scan_mode="prematch",
        )

        self.assertEqual(len(entries), 1)
        entry = entries[0]
        best_by_outcome = {item["outcome"]: item for item in entry["best_odds"]}
        self.assertEqual(best_by_outcome["Home Team"]["bookmaker_key"], "book_home")
        self.assertEqual(best_by_outcome["Away Team"]["bookmaker_key"], "betdex")
        self.assertLess(entry["roi_percent"], 0.0)
        self.assertLess(entry["stakes"]["guaranteed_profit"], 0.0)

    def test_collect_market_entries_three_way_known_arbitrage_profit(self) -> None:
        game = {
            "sport_key": "soccer_epl",
            "sport_display": "EPL",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "bookmakers": [
                {
                    "key": "book_home",
                    "title": "Book Home",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home FC", "price": 3.8}, {"name": "Draw", "price": 3.0}, {"name": "Away FC", "price": 2.4}]}],
                },
                {
                    "key": "book_draw",
                    "title": "Book Draw",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home FC", "price": 3.2}, {"name": "Draw", "price": 3.9}, {"name": "Away FC", "price": 2.1}]}],
                },
                {
                    "key": "book_away",
                    "title": "Book Away",
                    "markets": [{"key": "h2h", "outcomes": [{"name": "Home FC", "price": 3.1}, {"name": "Draw", "price": 3.1}, {"name": "Away FC", "price": 3.7}]}],
                },
            ],
        }

        entries = scanner._collect_market_entries(
            game,
            market_key="h2h",
            stake_total=120.0,
            commission_rate=0.0,
            scan_mode="prematch",
        )

        self.assertTrue(entries)
        best = entries[0]
        by_outcome = {item["outcome"]: item for item in best["best_odds"]}
        self.assertEqual(by_outcome["Home FC"]["bookmaker_key"], "book_home")
        self.assertEqual(by_outcome["Draw"]["bookmaker_key"], "book_draw")
        self.assertEqual(by_outcome["Away FC"]["bookmaker_key"], "book_away")
        self.assertGreater(best["stakes"]["guaranteed_profit"], 0.0)
        self.assertGreater(best["roi_percent"], 0.0)

    def test_calculate_stakes_extreme_odds_are_finite_and_positive(self) -> None:
        outcomes = [
            {"name": "A", "bookmaker": "Book A", "display_price": 1000.0, "effective_price": 1000.0},
            {"name": "B", "bookmaker": "Book B", "display_price": 1000.0, "effective_price": 1000.0},
        ]
        stakes = scanner._calculate_stakes(outcomes, 100.0, price_field="effective_price")
        self.assertTrue(math.isfinite(stakes["guaranteed_profit"]))
        self.assertTrue(math.isfinite(stakes["roi_percent"]))
        self.assertGreater(stakes["guaranteed_profit"], 0.0)
        self.assertGreater(stakes["roi_percent"], 0.0)

    def test_filter_events_for_scan_mode_prematch_uses_time_not_live_state(self) -> None:
        with patch("scanner.time.time", return_value=1_700_000_000):
            filtered, stats = scanner._filter_events_for_scan_mode(
                [
                    {
                        "id": "cancelled-but-future",
                        "commence_time": "2023-11-14T22:13:30Z",
                        "live_state": {"status": "cancelled"},
                    },
                    {
                        "id": "missing-time",
                        "live_state": {"status": "scheduled"},
                    },
                ],
                scan_mode="prematch",
            )

        filtered_ids = {event["id"] for event in filtered}
        self.assertIn("cancelled-but-future", filtered_ids)
        self.assertEqual(stats["dropped_missing_time"], 1)
        self.assertEqual(stats["dropped_past"], 0)

    def test_run_scan_passes_prematch_mode_to_single_sport_worker(self) -> None:
        observed_modes = []
        sports_payload = [{"key": "basketball_nba", "title": "NBA", "active": True, "has_outrights": False}]

        def _fake_scan_single_sport(**kwargs):
            observed_modes.append(kwargs.get("scan_mode"))
            return {
                "skipped": False,
                "sport_key": "basketball_nba",
                "sport_timing": {
                    "sport_key": "basketball_nba",
                    "sport": "NBA",
                    "api_fetch_ms": 0.0,
                    "provider_fetch_ms": 0.0,
                    "analysis_ms": 0.0,
                    "events_scanned": 0,
                    "providers": [],
                    "total_ms": 0.0,
                },
                "timing_steps": [],
                "api_market_skips": [],
                "sport_errors": [],
                "provider_updates": {},
                "provider_snapshot_updates": {},
                "events_scanned": 0,
                "total_profit": 0.0,
                "arb_opportunities": [],
                "middle_opportunities": [],
                "plus_ev_opportunities": [],
                "stale_event_filters": [],
                "successful": 1,
            }

        with (
            patch.object(scanner, "fetch_sports", return_value=sports_payload),
            patch.object(scanner, "_scan_single_sport", side_effect=_fake_scan_single_sport),
            patch.object(scanner, "_sport_scan_max_workers", return_value=1),
        ):
            result = scanner.run_scan(
                api_key="dummy",
                sports=["basketball_nba"],
                scan_mode="prematch",
                stake_amount=100.0,
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(observed_modes, ["prematch"])


if __name__ == "__main__":
    unittest.main()

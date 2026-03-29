from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import verification_matrix as vm


class VerificationMatrixTests(unittest.TestCase):
    def test_run_matrix_returns_one_row_per_sport_and_mode(self) -> None:
        captured_calls = []

        def _fake_run_scan(*, sport_key, provider_keys, scan_mode, regions, stake_amount, all_markets):
            captured_calls.append({
                "sport_key": sport_key,
                "provider_keys": list(provider_keys),
                "scan_mode": scan_mode,
                "regions": list(regions),
                "stake_amount": stake_amount,
                "all_markets": all_markets,
            })
            return {
                "success": True,
                "partial": False,
                "scan_diagnostics": {"reason_code": f"{scan_mode}:{sport_key}"},
                "arbitrage": {"opportunities_count": 1},
                "middles": {"opportunities_count": 2},
                "plus_ev": {"opportunities_count": 3},
                "custom_providers": {key: {"enabled": True, "events_merged": 1} for key in provider_keys},
            }

        with patch.object(vm, "run_provider_scan", side_effect=_fake_run_scan):
            rows = vm.run_matrix(
                sports=["basketball_nba", "icehockey_nhl"],
                provider_keys=["sx_bet", "polymarket"],
                scan_modes=["prematch", "live"],
                regions=["us", "eu"],
                stake_amount=100.0,
                all_markets=True,
            )

        self.assertEqual(len(rows), 4)
        self.assertEqual(len(captured_calls), 4)
        self.assertEqual(
            {(row["sport_key"], row["scan_mode"]) for row in rows},
            {
                ("basketball_nba", "prematch"),
                ("basketball_nba", "live"),
                ("icehockey_nhl", "prematch"),
                ("icehockey_nhl", "live"),
            },
        )
        first = rows[0]
        self.assertIn("reason_code", first)
        self.assertIn("providers", first)
        self.assertEqual(first["arbitrage_count"], 1)
        self.assertEqual(first["middle_count"], 2)
        self.assertEqual(first["plus_ev_count"], 3)

    def test_run_matrix_exposes_positive_only_counts_from_summary(self) -> None:
        fake_result = {
            "success": True,
            "partial": False,
            "scan_diagnostics": {"reason_code": "ok"},
            "arbitrage": {
                "opportunities_count": 3,
                "opportunities": [
                    {"event": "A", "market": "totals", "roi_percent": 1.2},
                    {"event": "B", "market": "h2h", "roi_percent": 0},
                    {"event": "C", "market": "spreads", "roi_percent": -0.4},
                ],
            },
            "middles": {
                "opportunities_count": 3,
                "opportunities": [
                    {"event": "A", "market": "totals", "ev_percent": 2.5},
                    {"event": "B", "market": "spreads", "ev_percent": 0},
                    {"event": "C", "market": "totals", "ev_percent": -1.0},
                ],
            },
            "plus_ev": {"opportunities_count": 1, "opportunities": []},
            "custom_providers": {"sx_bet": {"enabled": True, "events_merged": 1}},
        }

        with patch.object(vm, "run_provider_scan", return_value=fake_result):
            rows = vm.run_matrix(
                sports=["basketball_nba"],
                provider_keys=["sx_bet"],
                scan_modes=["prematch"],
                regions=["us"],
                stake_amount=100.0,
                all_markets=True,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["arbitrage_count"], 3)
        self.assertEqual(rows[0]["middle_count"], 3)
        self.assertEqual(rows[0]["positive_arbitrage_count"], 1)
        self.assertEqual(rows[0]["positive_middle_count"], 1)

    def test_main_prints_json_matrix_rows(self) -> None:
        fake_rows = [{"sport_key": "basketball_nba", "scan_mode": "live", "reason_code": "ok"}]

        with patch.object(vm, "run_matrix", return_value=fake_rows):
            output = vm.main([
                "--sports", "basketball_nba",
                "--scan-modes", "live",
                "--providers", "sx_bet", "polymarket",
            ])

        payload = json.loads(output)
        self.assertEqual(payload[0]["sport_key"], "basketball_nba")
        self.assertEqual(payload[0]["scan_mode"], "live")
        self.assertEqual(payload[0]["reason_code"], "ok")


if __name__ == "__main__":
    unittest.main()

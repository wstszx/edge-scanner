from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "artline_api_probe.py"
SPEC = importlib.util.spec_from_file_location("artline_api_probe", SCRIPT_PATH)
artline_api_probe = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(artline_api_probe)


class ArtlineApiProbeTests(unittest.TestCase):
    def test_build_url_adds_api_prefix_for_regular_endpoints(self) -> None:
        self.assertEqual(
            artline_api_probe._build_url("/settings"),
            "https://api.artline.bet/api/settings",
        )
        self.assertEqual(
            artline_api_probe._build_url("api/settings"),
            "https://api.artline.bet/api/settings",
        )

    def test_build_url_keeps_sanctum_endpoint_outside_api_prefix(self) -> None:
        self.assertEqual(
            artline_api_probe._build_url("/sanctum/csrf-cookie"),
            "https://api.artline.bet/sanctum/csrf-cookie",
        )

    def test_build_lines_payload_joins_repeatable_filters(self) -> None:
        args = argparse.Namespace(
            games_type="prematch",
            page=2,
            search=None,
            sport_type="common",
            sport=["football", "basketball"],
            tournament_id=["1", "2"],
            region_id=["10"],
            except_games_id=["100", "200"],
        )

        payload = artline_api_probe._build_lines_payload(args)

        self.assertEqual(
            payload,
            {
                "games_type": "prematch",
                "page": 2,
                "sport_type": "common",
                "sport": "football,basketball",
                "tournament_id": "1,2",
                "region_id": "10",
                "except_games_id": "100,200",
            },
        )

    def test_collect_lines_games_flattens_grouped_response(self) -> None:
        result = {
            "payload": {
                "data": {
                    "football": {
                        "games": [
                            {"id": 1, "team_1": {"value": "A"}, "team_2": {"value": "B"}},
                            {"id": 2, "team_1": {"value": "C"}, "team_2": {"value": "D"}},
                        ]
                    },
                    "basketball": {
                        "games": [
                            {"id": 3, "team_1": {"value": "E"}, "team_2": {"value": "F"}},
                        ]
                    },
                }
            }
        }

        games = artline_api_probe._collect_lines_games(result)

        self.assertEqual(
            games,
            [
                ("football", {"id": 1, "team_1": {"value": "A"}, "team_2": {"value": "B"}}),
                ("football", {"id": 2, "team_1": {"value": "C"}, "team_2": {"value": "D"}}),
                ("basketball", {"id": 3, "team_1": {"value": "E"}, "team_2": {"value": "F"}}),
            ],
        )


if __name__ == "__main__":
    unittest.main()

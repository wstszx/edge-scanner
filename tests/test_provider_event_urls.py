from __future__ import annotations

import json
import unittest
from pathlib import Path

from providers import betdex, bookmaker_xyz, polymarket, sx_bet


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "provider_contract_replay.json"


def _load_fixture() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)["providers"]


class ProviderEventUrlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()

    def test_polymarket_event_url_uses_event_slug(self) -> None:
        event = self.fixture["polymarket"]["requests"]["events"][0]

        self.assertEqual(
            polymarket._event_url(event),
            "https://polymarket.com/event/nhl-van-lak-2026-04-09",
        )

    def test_betdex_event_url_uses_subcategory_group_event_and_market(self) -> None:
        provider_fixture = self.fixture["betdex"]["requests"]
        event = provider_fixture["events"]["events"][0]
        event_groups = provider_fixture["events"]["eventGroups"]
        event_groups_by_id = {
            str((item or {}).get("id") or ""): item
            for item in event_groups
            if isinstance(item, dict)
        }
        first_market_id = str(provider_fixture["markets"]["markets"][0]["id"])

        self.assertEqual(
            betdex._event_url(event, event_groups_by_id, first_market_id),
            "https://www.betdex.com/events/icehky/nhl/20578?market=359445",
        )

    def test_sx_bet_event_url_uses_public_sport_and_league_route(self) -> None:
        self.assertEqual(
            sx_bet._event_url(
                {
                    "event_id": "L18016513",
                    "sport_key": "icehockey_nhl",
                    "league_label": "NHL",
                }
            ),
            "https://sx.bet/hockey/nhl/game-lines/L18016513",
        )

    def test_sx_bet_event_url_uses_known_league_override_for_mls(self) -> None:
        self.assertEqual(
            sx_bet._event_url(
                {
                    "event_id": "L18225577",
                    "sport_key": "soccer_usa_mls",
                    "league_label": "Major League Soccer",
                }
            ),
            "https://sx.bet/soccer/mls/game-lines/L18225577",
        )

    def test_sx_bet_event_url_falls_back_to_legacy_event_route_without_context(self) -> None:
        self.assertEqual(
            sx_bet._event_url("L18016513"),
            "https://sx.bet/event/L18016513",
        )

    def test_bookmaker_xyz_event_url_uses_chain_sport_country_league_and_game_id(self) -> None:
        game = self.fixture["bookmaker_xyz"]["raw"]["conditions"][0]["game"]

        self.assertEqual(
            bookmaker_xyz._event_url(game),
            "https://bookmaker.xyz/polygon/sports/ice-hockey/usa/nhl/1006000000000029214152",
        )


if __name__ == "__main__":
    unittest.main()

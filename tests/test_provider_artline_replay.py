from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from providers import artline


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "artline_contract_replay.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


async def _fake_shared_client(*args, **kwargs):
    return object()


class ArtlineProviderReplayTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _load_fixture()

    async def test_artline_replays_recorded_lines_payload(self) -> None:
        fixture = self.fixture

        async def _fake_request_json_async(
            client,
            method,
            path,
            *,
            params=None,
            json_payload=None,
            retries=None,
            backoff_seconds=None,
        ):
            self.assertEqual(method, "POST")
            self.assertEqual(path, "lines")
            return copy.deepcopy(fixture["requests"]["lines"]), 0

        with (
            patch.object(artline, "get_shared_client", new=_fake_shared_client),
            patch.object(artline, "_request_json_async", side_effect=_fake_request_json_async),
        ):
            events = await artline.fetch_events_async(
                fixture["sport_key"],
                ["h2h", "h2h_3_way", "spreads", "totals"],
                ["eu"],
            )

        self.assertEqual(len(events), fixture["expected"]["event_count"])
        event = events[0]
        self.assertEqual(event["home_team"], fixture["expected"]["home_team"])
        self.assertEqual(event["away_team"], fixture["expected"]["away_team"])
        bookmaker = event["bookmakers"][0]
        market_keys = sorted({market.get("key") for market in (bookmaker.get("markets") or [])})
        self.assertEqual(market_keys, sorted(fixture["expected"]["market_keys"]))
        self.assertTrue(any(market.get("key") == "spreads" for market in (bookmaker.get("markets") or [])))
        self.assertTrue(any(market.get("key") == "totals" for market in (bookmaker.get("markets") or [])))
        self.assertEqual(artline.fetch_events_async.last_stats.get("payload_games_count"), 1)
        self.assertEqual(artline.fetch_events_async.last_stats.get("events_returned_count"), 1)


if __name__ == "__main__":
    unittest.main()

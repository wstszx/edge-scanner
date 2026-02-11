from __future__ import annotations

from typing import List, Optional, Sequence

PROVIDER_KEY = "bookmaker_xyz"
PROVIDER_TITLE = "bookmaker.xyz"


def fetch_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    fetch_events.last_stats = {
        "provider": PROVIDER_KEY,
        "implemented": False,
        "events_returned_count": 0,
    }
    return []


fetch_events.last_stats = {
    "provider": PROVIDER_KEY,
    "implemented": False,
    "events_returned_count": 0,
}

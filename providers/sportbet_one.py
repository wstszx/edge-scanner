from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Sequence

from .bookmaker_xyz import fetch_events as fetch_bookmaker_xyz_events

PROVIDER_KEY = "sportbet_one"
PROVIDER_TITLE = "Sportbet.one"

SPORTBET_ONE_SOURCE = os.getenv("SPORTBET_ONE_SOURCE", "bookmaker_xyz").strip().lower()
SPORTBET_ONE_SAMPLE_PATH = os.getenv(
    "SPORTBET_ONE_SAMPLE_PATH",
    os.path.join("data", "sportbet_one_sample.json"),
).strip()
SPORTBET_ONE_PUBLIC_BASE = os.getenv("SPORTBET_ONE_PUBLIC_BASE", "https://sportbet.one/").strip()


class ProviderError(Exception):
    """Raised for provider-specific recoverable issues."""


def _public_base() -> str:
    base = (SPORTBET_ONE_PUBLIC_BASE or "").strip() or "https://sportbet.one/"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/") + "/"


def _requested_market_keys(markets: Sequence[str]) -> set[str]:
    requested = set()
    for value in markets or []:
        token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
        if token:
            requested.add(token)
    if "both_teams_to_score" in requested:
        requested.add("btts")
    if "btts" in requested:
        requested.add("both_teams_to_score")
    return requested


def _is_selected(bookmakers: Optional[Sequence[str]]) -> bool:
    if not bookmakers:
        return True
    lowered = {str(book).strip().lower() for book in bookmakers if isinstance(book, str)}
    aliases = {
        PROVIDER_KEY,
        PROVIDER_TITLE.lower(),
        "sportbet",
        "sportbet.one",
    }
    return any(alias in lowered for alias in aliases)


def _tag_provider(events: List[dict]) -> List[dict]:
    base_url = _public_base()
    for event in events:
        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        for book in bookmakers:
            if not isinstance(book, dict):
                continue
            book["key"] = PROVIDER_KEY
            book["title"] = PROVIDER_TITLE
            if not book.get("event_url"):
                book["event_url"] = base_url
    return events


def _load_file_events(path: str) -> List[dict]:
    if not path:
        raise ProviderError("SPORTBET_ONE_SAMPLE_PATH is empty")
    if not os.path.exists(path):
        raise ProviderError(f"Sportbet.one sample file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ProviderError(f"Failed to read Sportbet.one sample file: {exc}") from exc
    if not isinstance(payload, list):
        raise ProviderError("Sportbet.one sample payload must be a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def fetch_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    _ = regions  # Reserved for future region-specific routing.
    stats = {
        "provider": PROVIDER_KEY,
        "source": SPORTBET_ONE_SOURCE or "bookmaker_xyz",
        "proxy_provider": "bookmaker_xyz",
        "events_returned_count": 0,
        "selected": _is_selected(bookmakers),
    }
    fetch_events.last_stats = stats

    if not _requested_market_keys(markets):
        return []
    if not stats["selected"]:
        return []

    source = (SPORTBET_ONE_SOURCE or "bookmaker_xyz").lower()
    if source == "file":
        events = _load_file_events(SPORTBET_ONE_SAMPLE_PATH)
    elif source in {"bookmaker_xyz", "api"}:
        events = fetch_bookmaker_xyz_events(
            sport_key,
            markets,
            regions,
            bookmakers=["bookmaker_xyz"],
        )
        upstream_stats = dict(getattr(fetch_bookmaker_xyz_events, "last_stats", {}) or {})
        if upstream_stats:
            stats["upstream"] = upstream_stats
    else:
        raise ProviderError("Sportbet.one provider supports SPORTBET_ONE_SOURCE=bookmaker_xyz, api, or file")

    events = _tag_provider(events)
    stats["events_returned_count"] = len(events)
    fetch_events.last_stats = stats
    return events


fetch_events.last_stats = {
    "provider": PROVIDER_KEY,
    "source": SPORTBET_ONE_SOURCE or "bookmaker_xyz",
    "proxy_provider": "bookmaker_xyz",
    "events_returned_count": 0,
}


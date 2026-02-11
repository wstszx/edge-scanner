from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from .bookmaker_xyz import PROVIDER_TITLE as BOOKMAKER_XYZ_TITLE
from .bookmaker_xyz import fetch_events as fetch_bookmaker_xyz_events
from .overtimemarkets_xyz import PROVIDER_TITLE as OVERTIMEMARKETS_XYZ_TITLE
from .overtimemarkets_xyz import fetch_events as fetch_overtimemarkets_xyz_events
from .polymarket import PROVIDER_TITLE as POLYMARKET_TITLE
from .polymarket import fetch_events as fetch_polymarket_events
from .purebet import PROVIDER_TITLE as PUREBET_TITLE
from .purebet import fetch_events as fetch_purebet_events
from .sx_bet import PROVIDER_TITLE as SX_BET_TITLE
from .sx_bet import fetch_events as fetch_sx_bet_events

ProviderFetcher = Callable[
    [str, Sequence[str], Sequence[str], Optional[Sequence[str]]],
    List[dict],
]

PROVIDER_FETCHERS: Dict[str, ProviderFetcher] = {
    "purebet": fetch_purebet_events,
    "bookmaker_xyz": fetch_bookmaker_xyz_events,
    "sx_bet": fetch_sx_bet_events,
    "overtimemarkets_xyz": fetch_overtimemarkets_xyz_events,
    "polymarket": fetch_polymarket_events,
}

PROVIDER_TITLES: Dict[str, str] = {
    "purebet": PUREBET_TITLE,
    "bookmaker_xyz": BOOKMAKER_XYZ_TITLE,
    "sx_bet": SX_BET_TITLE,
    "overtimemarkets_xyz": OVERTIMEMARKETS_XYZ_TITLE,
    "polymarket": POLYMARKET_TITLE,
}

PROVIDER_ALIASES: Dict[str, str] = {
    "purebet": "purebet",
    "bookmaker.xyz": "bookmaker_xyz",
    "bookmaker_xyz": "bookmaker_xyz",
    "bookmakerxyz": "bookmaker_xyz",
    "sx bet": "sx_bet",
    "sx_bet": "sx_bet",
    "sxbet": "sx_bet",
    "overtimemarkets.xyz": "overtimemarkets_xyz",
    "overtimemarkets_xyz": "overtimemarkets_xyz",
    "overtimemarketsxyz": "overtimemarkets_xyz",
    "polymarket": "polymarket",
}


def resolve_provider_key(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if not raw:
        return None
    if raw in PROVIDER_FETCHERS:
        return raw
    if raw in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[raw]
    compact = raw.replace("-", "_").replace(" ", "_")
    if compact in PROVIDER_FETCHERS:
        return compact
    if compact in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[compact]
    dotted = compact.replace("_", ".")
    if dotted in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[dotted]
    return None

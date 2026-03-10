from __future__ import annotations

from typing import Awaitable, Callable, Dict, List, Optional, Sequence

from .betdex import PROVIDER_TITLE as BETDEX_TITLE
from .betdex import fetch_events as fetch_betdex_events
from .betdex import fetch_events_async as fetch_betdex_events_async
from .bookmaker_xyz import PROVIDER_TITLE as BOOKMAKER_XYZ_TITLE
from .bookmaker_xyz import fetch_events as fetch_bookmaker_xyz_events
from .bookmaker_xyz import fetch_events_async as fetch_bookmaker_xyz_events_async
from .polymarket import PROVIDER_TITLE as POLYMARKET_TITLE
from .polymarket import fetch_events as fetch_polymarket_events
from .polymarket import fetch_events_async as fetch_polymarket_events_async
from .purebet import PROVIDER_TITLE as PUREBET_TITLE
from .purebet import fetch_events as fetch_purebet_events
from .purebet import fetch_events_async as fetch_purebet_events_async
from .dexsport_io import PROVIDER_TITLE as DEXSPORT_IO_TITLE
from .dexsport_io import fetch_events as fetch_dexsport_io_events
from .dexsport_io import fetch_events_async as fetch_dexsport_io_events_async
from .sportbet_one import PROVIDER_TITLE as SPORTBET_ONE_TITLE
from .sportbet_one import fetch_events as fetch_sportbet_one_events
from .sportbet_one import fetch_events_async as fetch_sportbet_one_events_async
from .sx_bet import PROVIDER_TITLE as SX_BET_TITLE
from .sx_bet import fetch_events as fetch_sx_bet_events
from .sx_bet import fetch_events_async as fetch_sx_bet_events_async

ProviderFetcher = Callable[
    [str, Sequence[str], Sequence[str], Optional[Sequence[str]]],
    List[dict] | Awaitable[List[dict]],
]

PROVIDER_FETCHERS: Dict[str, ProviderFetcher] = {
    "purebet": fetch_purebet_events_async,
    "betdex": fetch_betdex_events_async,
    "bookmaker_xyz": fetch_bookmaker_xyz_events_async,
    "sx_bet": fetch_sx_bet_events_async,
    "polymarket": fetch_polymarket_events_async,
    "dexsport_io": fetch_dexsport_io_events_async,
    "sportbet_one": fetch_sportbet_one_events_async,
}

PROVIDER_TITLES: Dict[str, str] = {
    "purebet": PUREBET_TITLE,
    "betdex": BETDEX_TITLE,
    "bookmaker_xyz": BOOKMAKER_XYZ_TITLE,
    "sx_bet": SX_BET_TITLE,
    "polymarket": POLYMARKET_TITLE,
    "dexsport_io": DEXSPORT_IO_TITLE,
    "sportbet_one": SPORTBET_ONE_TITLE,
}

PROVIDER_ALIASES: Dict[str, str] = {
    "purebet": "purebet",
    "betdex": "betdex",
    "bet dex": "betdex",
    "bet-dex": "betdex",
    "bookmaker.xyz": "bookmaker_xyz",
    "bookmaker_xyz": "bookmaker_xyz",
    "bookmakerxyz": "bookmaker_xyz",
    "sx bet": "sx_bet",
    "sx_bet": "sx_bet",
    "sxbet": "sx_bet",
    "polymarket": "polymarket",
    "dexsport_io": "dexsport_io",
    "dexsport.io": "dexsport_io",
    "dexsport": "dexsport_io",
    "sportbet_one": "sportbet_one",
    "sportbet.one": "sportbet_one",
    "sportbetone": "sportbet_one",
    "sportbet": "sportbet_one",
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

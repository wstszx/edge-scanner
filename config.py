"""Configuration constants for the arbitrage scanner."""

from __future__ import annotations

import json
import os
import re


def _env_text(key: str) -> str:
    value = os.getenv(key)
    return value.strip() if isinstance(value, str) else ""


def _env_bool(key: str, default: bool) -> bool:
    raw = _env_text(key)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = _env_text(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = _env_text(key)
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_list(key: str) -> list[str]:
    raw = _env_text(key)
    if not raw:
        return []
    if raw.startswith("["):
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    return [item.strip() for item in re.split(r"[,\s]+", raw) if item.strip()]


def _normalize_choice(value: str, allowed: set[str], default: str) -> str:
    if value in allowed:
        return value
    return default


REGION_CONFIG = {
    "us": {"name": "United States", "default": True},
    "us2": {"name": "United States (Additional)", "default": False},
    "uk": {"name": "United Kingdom", "default": False},
    "eu": {"name": "Europe", "default": True},
    "au": {"name": "Australia", "default": False},
}

_DEFAULT_REGION_KEYS = [key for key, meta in REGION_CONFIG.items() if meta.get("default")]
_ENV_DEFAULT_REGIONS = _env_list("DEFAULT_REGION_KEYS")
if _ENV_DEFAULT_REGIONS:
    DEFAULT_REGION_KEYS = [key for key in _ENV_DEFAULT_REGIONS if key in REGION_CONFIG] or _DEFAULT_REGION_KEYS
else:
    DEFAULT_REGION_KEYS = _DEFAULT_REGION_KEYS

REGION_OPTIONS = [
    {"key": key, "label": meta["name"], "default": key in DEFAULT_REGION_KEYS}
    for key, meta in REGION_CONFIG.items()
]

EXCHANGE_BOOKMAKERS = {
    "betfair_ex_eu": {"name": "Betfair"},
    "betfair_ex_uk": {"name": "Betfair"},
    "betfair_ex_au": {"name": "Betfair"},
    "sportsbet_ex": {"name": "Sportsbet Exchange"},
    "matchbook": {"name": "Matchbook"},
    "betdex": {"name": "BetDEX"},
}

EXCHANGE_KEYS = set(EXCHANGE_BOOKMAKERS.keys())

DEFAULT_COMMISSION = _env_float("DEFAULT_COMMISSION", 0.05)  # 5%
DEFAULT_MIN_ROI = _env_float("DEFAULT_MIN_ROI", 0.0)
DEFAULT_EXCHANGE_ONLY = _env_bool("DEFAULT_EXCHANGE_ONLY", False)
DEFAULT_ARBITRAGE_SORT = _normalize_choice(
    _env_text("DEFAULT_ARBITRAGE_SORT"),
    {"roi", "profit", "time"},
    "roi",
)
DEFAULT_ALL_SPORTS = _env_bool("DEFAULT_ALL_SPORTS", False)
DEFAULT_ODDS_FORMAT = _normalize_choice(
    _env_text("DEFAULT_ODDS_FORMAT"),
    {"decimal", "american"},
    "decimal",
)
DEFAULT_DENSITY = _normalize_choice(
    _env_text("DEFAULT_DENSITY"),
    {"comfort", "compact"},
    "comfort",
)
DEFAULT_THEME = _normalize_choice(
    _env_text("DEFAULT_THEME"),
    {"light", "dark"},
    "light",
)
DEFAULT_LANGUAGE = _normalize_choice(
    _env_text("DEFAULT_LANGUAGE"),
    {"en", "zh"},
    "",
)
DEFAULT_AUTO_SCAN_ENABLED = _env_bool("DEFAULT_AUTO_SCAN_ENABLED", False)
DEFAULT_AUTO_SCAN_MINUTES = max(1, _env_int("DEFAULT_AUTO_SCAN_MINUTES", 10))
DEFAULT_NOTIFY_SOUND_ENABLED = _env_bool("DEFAULT_NOTIFY_SOUND_ENABLED", False)
DEFAULT_NOTIFY_POPUP_ENABLED = _env_bool("DEFAULT_NOTIFY_POPUP_ENABLED", False)

# -----------------------------------------------------------------------------
# Middles configuration
# -----------------------------------------------------------------------------

MIN_MIDDLE_GAP = _env_float("MIN_MIDDLE_GAP", 1.5)
DEFAULT_MIDDLE_SORT = _normalize_choice(
    _env_text("DEFAULT_MIDDLE_SORT"),
    {"ev", "probability", "gap", "time"},
    "ev",
)
SHOW_POSITIVE_EV_ONLY = _env_bool("SHOW_POSITIVE_EV_ONLY", True)

PROBABILITY_PER_INTEGER = {
    # Spreads
    "americanfootball_nfl_spreads": 0.025,
    "americanfootball_ncaaf_spreads": 0.025,
    "basketball_nba_spreads": 0.025,
    "basketball_ncaab_spreads": 0.025,
    "baseball_mlb_spreads": 0.030,
    "icehockey_nhl_spreads": 0.030,
    # Totals
    "americanfootball_nfl_totals": 0.030,
    "americanfootball_ncaaf_totals": 0.030,
    "basketball_nba_totals": 0.020,
    "basketball_ncaab_totals": 0.020,
    "baseball_mlb_totals": 0.045,
    "icehockey_nhl_totals": 0.055,
    "default": 0.030,
}

NFL_KEY_NUMBER_PROBABILITY = {
    3: 0.150,
    7: 0.090,
    10: 0.060,
    6: 0.050,
    14: 0.045,
    4: 0.040,
    1: 0.035,
    17: 0.035,
    13: 0.030,
    11: 0.025,
}

KEY_NUMBER_SPORTS = {
    "americanfootball_nfl",
    "americanfootball_ncaaf",
}

MAX_MIDDLE_PROBABILITY = 0.35

SPORT_DISPLAY_NAMES = {
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "NCAAF",
    "basketball_nba": "NBA",
    "basketball_ncaab": "NCAAB",
    "basketball_euroleague": "EuroLeague",
    "basketball_germany_bbl": "Basketball Bundesliga",
    "basketball_spain_liga_acb": "Liga ACB",
    "basketball_italy_serie_a": "Lega Basket Serie A",
    "basketball_france_pro_a": "LNB Pro A",
    "baseball_mlb": "MLB",
    "baseball_mlb_spring_training": "MLB Spring Training",
    "icehockey_nhl": "NHL",
    "icehockey_khl": "KHL",
    "icehockey_ahl": "AHL",
    "soccer_epl": "Premier League",
    "soccer_england_championship": "EFL Championship",
    "soccer_england_league_one": "EFL League One",
    "soccer_england_league_two": "EFL League Two",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
    "soccer_usa_mls": "MLS",
    "soccer_portugal_primeira_liga": "Primeira Liga",
    "soccer_netherlands_eredivisie": "Eredivisie",
    "soccer_brazil_serie_a": "Brasileirao Serie A",
    "soccer_argentina_liga_profesional": "Liga Profesional",
    "soccer_mexico_liga_mx": "Liga MX",
    "soccer_turkey_super_lig": "Super Lig",
    "mma_ufc": "UFC",
    "boxing_professional": "Professional Boxing",
    "rugby_union_six_nations": "Six Nations",
    "rugby_league_nrl": "National Rugby League",
    "tennis_atp_indian_wells": "ATP Indian Wells",
    "tennis_wta_indian_wells": "WTA Indian Wells",
}

AMERICAN_SPORTS = {
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "basketball_nba",
    "basketball_ncaab",
    "basketball_euroleague",
    "basketball_germany_bbl",
    "basketball_spain_liga_acb",
    "basketball_italy_serie_a",
    "basketball_france_pro_a",
    "baseball_mlb",
    "baseball_mlb_spring_training",
    "icehockey_nhl",
    "icehockey_khl",
    "icehockey_ahl",
}

SOCCER_SPORTS = {
    "soccer_epl",
    "soccer_england_championship",
    "soccer_england_league_one",
    "soccer_england_league_two",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
    "soccer_portugal_primeira_liga",
    "soccer_netherlands_eredivisie",
    "soccer_brazil_serie_a",
    "soccer_argentina_liga_profesional",
    "soccer_mexico_liga_mx",
    "soccer_turkey_super_lig",
}

_DEFAULT_SPORT_KEYS = [
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "basketball_nba",
    "basketball_ncaab",
    "baseball_mlb",
    "icehockey_nhl",
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
]
_ENV_DEFAULT_SPORTS = _env_list("DEFAULT_SPORT_KEYS")
if _ENV_DEFAULT_SPORTS:
    DEFAULT_SPORT_KEYS = _ENV_DEFAULT_SPORTS
else:
    DEFAULT_SPORT_KEYS = _DEFAULT_SPORT_KEYS

SPORT_OPTIONS = [
    {"key": key, "label": label, "default": key in DEFAULT_SPORT_KEYS}
    for key, label in SPORT_DISPLAY_NAMES.items()
]
DEFAULT_SPORT_OPTIONS = SPORT_OPTIONS

AMERICAN_MARKETS = ["h2h", "spreads", "totals"]
SOCCER_MARKETS = ["h2h", "h2h_3_way", "spreads", "totals"]
GENERIC_MARKETS = ["h2h"]

ROI_BANDS = [
    (float("-inf"), 0.0, "<0%"),
    (0.0, 1.0, "0-1%"),
    (1.0, 2.0, "1-2%"),
    (2.0, float("inf"), "2%+"),
]

# -----------------------------------------------------------------------------
# +EV configuration
# -----------------------------------------------------------------------------

SHARP_BOOKS = [
    {"key": "pinnacle", "name": "Pinnacle", "region": "eu", "type": "bookmaker"},
    {"key": "betfair_ex_eu", "name": "Betfair Exchange", "region": "eu", "type": "exchange"},
    {"key": "matchbook", "name": "Matchbook", "region": "eu", "type": "exchange"},
]

BOOKMAKER_REGION_KEYS = {
    "betfair_ex_uk": "uk",
    "betfair_ex_au": "au",
    "sportsbet_ex": "au",
}
for _book in SHARP_BOOKS:
    _key = _book.get("key")
    _region = _book.get("region")
    if _key and _region in REGION_CONFIG and _key not in BOOKMAKER_REGION_KEYS:
        BOOKMAKER_REGION_KEYS[_key] = _region

DEFAULT_SHARP_BOOK = _env_text("DEFAULT_SHARP_BOOK") or "pinnacle"
DEFAULT_BANKROLL = _env_float("DEFAULT_BANKROLL", 1000.0)
MIN_EDGE_PERCENT = _env_float("MIN_EDGE_PERCENT", 1.0)
DEFAULT_KELLY_FRACTION = _env_float("DEFAULT_KELLY_FRACTION", 0.25)
DEFAULT_STAKE_AMOUNT = _env_float("DEFAULT_STAKE_AMOUNT", 100.0)
DEFAULT_PLUS_EV_SORT = _normalize_choice(
    _env_text("DEFAULT_PLUS_EV_SORT"),
    {"edge", "ev", "kelly", "time"},
    "edge",
)
KELLY_OPTIONS = [
    {"label": "Full Kelly", "value": 1.0},
    {"label": "Half Kelly", "value": 0.5},
    {"label": "Quarter Kelly", "value": 0.25},
    {"label": "Tenth Kelly", "value": 0.1},
]

SOFT_BOOK_KEYS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "pointsbetus",
    "pointsbet",
    "betrivers",
    "unibet",
    "888sport",
    "bovada",
    "betonlineag",
    "lowvig",
    "williamhill_us",
    "betway",
    "sportsbettingag",
    "barstool",
]

SOFT_BOOK_LABELS = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "pointsbetus": "PointsBet US",
    "pointsbet": "PointsBet",
    "betrivers": "BetRivers",
    "unibet": "Unibet",
    "888sport": "888sport",
    "bovada": "Bovada",
    "betonlineag": "BetOnline",
    "lowvig": "LowVig",
    "williamhill_us": "William Hill US",
    "betway": "Betway",
    "sportsbettingag": "SportsBetting",
    "barstool": "Barstool",
}

BOOKMAKER_LABELS = {
    **{key: meta["name"] for key, meta in EXCHANGE_BOOKMAKERS.items()},
    **{book["key"]: book["name"] for book in SHARP_BOOKS if book.get("key")},
    **SOFT_BOOK_LABELS,
    "artline": "Artline",
    "bookmaker_xyz": "bookmaker.xyz",
    "sx_bet": "SX Bet",
    "polymarket": "Polymarket",
}

BOOKMAKER_URLS = {
    "draftkings": "https://sportsbook.draftkings.com/",
    "fanduel": "https://sportsbook.fanduel.com/",
    "betmgm": "https://sports.betmgm.com/",
    "caesars": "https://www.caesars.com/sportsbook-and-casino",
    "pointsbetus": "https://pointsbet.com/",
    "pointsbet": "https://pointsbet.com.au/",
    "betrivers": "https://www.betrivers.com/",
    "unibet": "https://www.unibet.com/",
    "888sport": "https://www.888sport.com/",
    "bovada": "https://www.bovada.lv/",
    "betonlineag": "https://www.betonline.ag/",
    "lowvig": "https://www.lowvig.ag/",
    "williamhill_us": "https://www.williamhill.com/us/",
    "betway": "https://www.betway.com/",
    "sportsbettingag": "https://www.sportsbetting.ag/",
    "barstool": "https://www.barstoolsportsbook.com/",
    "pinnacle": "https://www.pinnacle.com/",
    "matchbook": "https://www.matchbook.com/",
    "betfair_ex_eu": "https://www.betfair.com/exchange/",
    "betfair_ex_uk": "https://www.betfair.com/exchange/",
    "betfair_ex_au": "https://www.betfair.com.au/exchange/",
    "sportsbet_ex": "https://www.sportsbet.com.au/",
    "betdex": "https://www.betdex.com/",
    "artline": "https://artline.bet/",
    "bookmaker_xyz": "https://bookmaker.xyz/",
    "sx_bet": "https://sx.bet/",
    "polymarket": "https://polymarket.com/",
}

ALL_BOOKMAKER_KEYS: list[str] = []
for key in [
    "pinnacle",
    "polymarket",
    "betdex",
    "artline",
    "bookmaker_xyz",
    "sx_bet",
]:
    if key not in ALL_BOOKMAKER_KEYS:
        ALL_BOOKMAKER_KEYS.append(key)
for key in EXCHANGE_BOOKMAKERS:
    if key not in ALL_BOOKMAKER_KEYS:
        ALL_BOOKMAKER_KEYS.append(key)
for book in SHARP_BOOKS:
    key = book.get("key")
    if key and key not in ALL_BOOKMAKER_KEYS:
        ALL_BOOKMAKER_KEYS.append(key)
for key in SOFT_BOOK_KEYS:
    if key not in ALL_BOOKMAKER_KEYS:
        ALL_BOOKMAKER_KEYS.append(key)

_DEFAULT_ALLOWED_ARBITRAGE_BOOKMAKER_KEYS = [
    "pinnacle",
    "polymarket",
    "betdex",
    "artline",
    "bookmaker_xyz",
    "sx_bet",
    "betfair_ex_eu",
    "betfair_ex_uk",
    "betfair_ex_au",
    "sportsbet_ex",
    "matchbook",
]


def _validate_exchange_bookmaker_config() -> tuple[str, ...]:
    warnings: list[str] = []
    for key, meta in EXCHANGE_BOOKMAKERS.items():
        label = str(meta.get("name") or "").strip()
        if not label:
            warnings.append(f"{key} is missing an exchange display name")
        if key not in ALL_BOOKMAKER_KEYS:
            warnings.append(f"{key} is missing from ALL_BOOKMAKER_KEYS")
        if not str(BOOKMAKER_LABELS.get(key) or "").strip():
            warnings.append(f"{key} is missing a bookmaker label")
        if not str(BOOKMAKER_URLS.get(key) or "").strip():
            warnings.append(f"{key} is missing a bookmaker URL")
        if key not in _DEFAULT_ALLOWED_ARBITRAGE_BOOKMAKER_KEYS:
            warnings.append(f"{key} is missing from default arbitrage bookmaker keys")
    return tuple(dict.fromkeys(warnings))


EXCHANGE_CONFIG_WARNINGS = _validate_exchange_bookmaker_config()

_ENV_ALLOWED_ARBITRAGE_BOOKMAKERS = (
    _env_list("ARBITRAGE_ALLOWED_BOOKMAKER_KEYS")
    or _env_list("SUPPORTED_ARBITRAGE_BOOKMAKER_KEYS")
)

if _ENV_ALLOWED_ARBITRAGE_BOOKMAKERS:
    BOOKMAKER_KEYS = [
        key for key in _ENV_ALLOWED_ARBITRAGE_BOOKMAKERS if key in ALL_BOOKMAKER_KEYS
    ]
else:
    BOOKMAKER_KEYS = [
        key for key in _DEFAULT_ALLOWED_ARBITRAGE_BOOKMAKER_KEYS
        if key in ALL_BOOKMAKER_KEYS
    ]

if not BOOKMAKER_KEYS:
    BOOKMAKER_KEYS = [
        key for key in _DEFAULT_ALLOWED_ARBITRAGE_BOOKMAKER_KEYS
        if key in ALL_BOOKMAKER_KEYS
    ]

ALLOWED_ARBITRAGE_BOOKMAKER_KEYS = BOOKMAKER_KEYS.copy()

DEFAULT_BOOKMAKER_KEYS: list[str] = []
_ENV_DEFAULT_BOOKMAKERS = _env_list("DEFAULT_BOOKMAKER_KEYS")
if _ENV_DEFAULT_BOOKMAKERS:
    DEFAULT_BOOKMAKER_KEYS = [
        key for key in _ENV_DEFAULT_BOOKMAKERS if key in BOOKMAKER_KEYS
    ] or list(BOOKMAKER_KEYS)
else:
    DEFAULT_BOOKMAKER_KEYS = list(BOOKMAKER_KEYS)

BOOKMAKER_OPTIONS = [
    {"key": key, "label": BOOKMAKER_LABELS.get(key, key)}
    for key in BOOKMAKER_KEYS
]

_BOOKMAKER_KEY_LOOKUP = {key.lower(): key for key in BOOKMAKER_KEYS}
_BOOKMAKER_LABEL_LOOKUP_RAW: dict[str, str] = {}
_BOOKMAKER_LABEL_CONFLICTS: set[str] = set()
for key in BOOKMAKER_KEYS:
    label = str(BOOKMAKER_LABELS.get(key, key)).strip().lower()
    if not label:
        continue
    existing = _BOOKMAKER_LABEL_LOOKUP_RAW.get(label)
    if existing and existing != key:
        _BOOKMAKER_LABEL_CONFLICTS.add(label)
        continue
    _BOOKMAKER_LABEL_LOOKUP_RAW[label] = key
_BOOKMAKER_LABEL_LOOKUP = {
    label: key
    for label, key in _BOOKMAKER_LABEL_LOOKUP_RAW.items()
    if label not in _BOOKMAKER_LABEL_CONFLICTS
}


def canonical_bookmaker_key(value: object) -> str:
    if value is None:
        return ""
    token = str(value).strip().lower()
    if not token:
        return ""
    return _BOOKMAKER_KEY_LOOKUP.get(token) or _BOOKMAKER_LABEL_LOOKUP.get(token) or ""


def normalize_supported_bookmakers(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen = set()
    for value in values:
        key = canonical_bookmaker_key(value)
        if not key or key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized


def derive_required_regions(
    bookmakers: object,
    *,
    sharp_book: object = DEFAULT_SHARP_BOOK,
) -> list[str]:
    regions: list[str] = []
    seen = set()
    for value in bookmakers if isinstance(bookmakers, list) else []:
        key = canonical_bookmaker_key(value)
        region = BOOKMAKER_REGION_KEYS.get(key)
        if not region or region in seen or region not in REGION_CONFIG:
            continue
        regions.append(region)
        seen.add(region)

    sharp_key = canonical_bookmaker_key(sharp_book)
    if not sharp_key and isinstance(sharp_book, str):
        sharp_key = sharp_book.strip().lower()
    sharp_region = BOOKMAKER_REGION_KEYS.get(sharp_key)
    if sharp_region and sharp_region in REGION_CONFIG and sharp_region not in seen:
        regions.append(sharp_region)
        seen.add(sharp_region)

    return regions or list(DEFAULT_REGION_KEYS)

EDGE_BANDS = [
    (1.0, 3.0, "1-3%"),
    (3.0, 5.0, "3-5%"),
    (5.0, 10.0, "5-10%"),
    (10.0, float("inf"), "10%+"),
]


def markets_for_sport(sport_key: str) -> list[str]:
    key = (sport_key or "").strip().lower()
    if key.startswith("azuro__"):
        parts = [part for part in key.split("__") if part]
        family = parts[1] if len(parts) >= 4 else ""
        if family in {"football"}:
            return SOCCER_MARKETS.copy()
        if family in {"american-football", "basketball", "baseball", "ice-hockey"}:
            return AMERICAN_MARKETS.copy()
        return GENERIC_MARKETS.copy()
    if key in AMERICAN_SPORTS:
        return AMERICAN_MARKETS
    if key in SOCCER_SPORTS:
        return SOCCER_MARKETS
    # Support unknown leagues under known families (e.g., basketball_wnba, americanfootball_ncaaf).
    if key.startswith(("americanfootball_", "basketball_", "baseball_", "icehockey_")):
        return AMERICAN_MARKETS.copy()
    if key.startswith("soccer_"):
        return SOCCER_MARKETS.copy()
    # Generic fallback keeps scans broad without forcing potentially invalid spread/total markets.
    return GENERIC_MARKETS.copy()

# -----------------------------------------------------------------------------
# History configuration
# -----------------------------------------------------------------------------

HISTORY_ENABLED: bool = _env_bool("HISTORY_ENABLED", True)
HISTORY_DIR: str = _env_text("HISTORY_DIR") or "data/history"
HISTORY_MAX_RECORDS: int = max(100, _env_int("HISTORY_MAX_RECORDS", 10000))

# -----------------------------------------------------------------------------
# Notification configuration
# -----------------------------------------------------------------------------

NOTIFY_WEBHOOK_URL: str = _env_text("NOTIFY_WEBHOOK_URL")
NOTIFY_WEBHOOK_SECRET: str = _env_text("NOTIFY_WEBHOOK_SECRET")
NOTIFY_TELEGRAM_TOKEN: str = _env_text("NOTIFY_TELEGRAM_TOKEN")
NOTIFY_TELEGRAM_CHAT_ID: str = _env_text("NOTIFY_TELEGRAM_CHAT_ID")
NOTIFY_MIN_ROI: float = _env_float("NOTIFY_MIN_ROI", 0.0)
NOTIFY_MIN_EDGE: float = _env_float("NOTIFY_MIN_EDGE", 0.0)
NOTIFY_MIN_EV: float = _env_float("NOTIFY_MIN_EV", 0.0)
NOTIFY_TIMEOUT_SECONDS: int = max(1, _env_int("NOTIFY_TIMEOUT_SECONDS", 10))

# -----------------------------------------------------------------------------
# Scanner performance tuning (centralised from scanner.py)
# -----------------------------------------------------------------------------

PROVIDER_FETCH_MAX_WORKERS: int = max(1, _env_int("PROVIDER_FETCH_MAX_WORKERS", 3))
SPORT_SCAN_MAX_WORKERS: int = max(1, _env_int("SPORT_SCAN_MAX_WORKERS", 2))
PROVIDER_NETWORK_RETRY_ONCE: bool = _env_bool("PROVIDER_NETWORK_RETRY_ONCE", True)
PROVIDER_NETWORK_RETRY_DELAY_MS: int = max(0, _env_int("PROVIDER_NETWORK_RETRY_DELAY_MS", 250))
ODDS_API_MARKET_BATCH_SIZE: int = max(1, _env_int("ODDS_API_MARKET_BATCH_SIZE", 8))

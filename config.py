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
    "purebet": {"name": "Purebet"},
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
    "basketball_nba": "NBA",
    "basketball_ncaab": "NCAAB",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
    "soccer_usa_mls": "MLS",
}

AMERICAN_SPORTS = {
    "americanfootball_nfl",
    "basketball_nba",
    "basketball_ncaab",
    "baseball_mlb",
    "icehockey_nhl",
}

SOCCER_SPORTS = {
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
}

_DEFAULT_SPORT_KEYS = [
    "americanfootball_nfl",
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

DEFAULT_SPORT_OPTIONS = [
    {"key": key, "label": SPORT_DISPLAY_NAMES.get(key, key)}
    for key in DEFAULT_SPORT_KEYS
]

AMERICAN_MARKETS = ["h2h", "spreads", "totals"]
SOCCER_MARKETS = ["spreads", "totals"]

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
    "dexsport_io": "Dexsport.io",
    "sportbet_one": "Sportbet.one",
    "bookmaker_xyz": "bookmaker.xyz",
    "sx_bet": "SX Bet",
    "overtimemarkets_xyz": "Overtime Markets",
    "polymarket": "Polymarket",
    "purebet": "Purebet",
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
    "dexsport_io": "https://dexsport.com/",
    "sportbet_one": "https://sportbet.one/",
    "bookmaker_xyz": "https://bookmaker.xyz/",
    "sx_bet": "https://sx.bet/",
    "overtimemarkets_xyz": "https://overtimemarkets.xyz/",
    "polymarket": "https://polymarket.com/",
    "purebet": "https://purebet.io/",
}

BOOKMAKER_KEYS: list[str] = []
for key in [
    "pinnacle",
    "polymarket",
    "purebet",
    "betdex",
    "dexsport_io",
    "sportbet_one",
    "bookmaker_xyz",
    "sx_bet",
    "overtimemarkets_xyz",
]:
    if key not in BOOKMAKER_KEYS:
        BOOKMAKER_KEYS.append(key)
for key in EXCHANGE_BOOKMAKERS:
    if key not in BOOKMAKER_KEYS:
        BOOKMAKER_KEYS.append(key)
for book in SHARP_BOOKS:
    key = book.get("key")
    if key and key not in BOOKMAKER_KEYS:
        BOOKMAKER_KEYS.append(key)
for key in SOFT_BOOK_KEYS:
    if key not in BOOKMAKER_KEYS:
        BOOKMAKER_KEYS.append(key)

DEFAULT_BOOKMAKER_KEYS: list[str] = []
_ENV_DEFAULT_BOOKMAKERS = _env_list("DEFAULT_BOOKMAKER_KEYS")
if _ENV_DEFAULT_BOOKMAKERS:
    DEFAULT_BOOKMAKER_KEYS = [key for key in _ENV_DEFAULT_BOOKMAKERS if key in BOOKMAKER_KEYS]

BOOKMAKER_OPTIONS = [
    {"key": key, "label": BOOKMAKER_LABELS.get(key, key)}
    for key in BOOKMAKER_KEYS
]

EDGE_BANDS = [
    (1.0, 3.0, "1-3%"),
    (3.0, 5.0, "3-5%"),
    (5.0, 10.0, "5-10%"),
    (10.0, float("inf"), "10%+"),
]


def markets_for_sport(sport_key: str) -> list[str]:
    if sport_key in AMERICAN_SPORTS:
        return AMERICAN_MARKETS
    if sport_key in SOCCER_SPORTS:
        return SOCCER_MARKETS
    return SOCCER_MARKETS.copy()

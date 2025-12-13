"""Configuration constants for the arbitrage scanner."""

from __future__ import annotations

REGION_CONFIG = {
    "us": {"name": "United States", "default": True},
    "us2": {"name": "United States (Additional)", "default": False},
    "uk": {"name": "United Kingdom", "default": False},
    "eu": {"name": "Europe", "default": True},
    "au": {"name": "Australia", "default": False},
}

DEFAULT_REGION_KEYS = [key for key, meta in REGION_CONFIG.items() if meta.get("default")]

REGION_OPTIONS = [
    {"key": key, "label": meta["name"], "default": meta.get("default", False)}
    for key, meta in REGION_CONFIG.items()
]

EXCHANGE_BOOKMAKERS = {
    "betfair_ex_eu": {"name": "Betfair"},
    "betfair_ex_uk": {"name": "Betfair"},
    "betfair_ex_au": {"name": "Betfair"},
    "sportsbet_ex": {"name": "Sportsbet Exchange"},
    "matchbook": {"name": "Matchbook"},
}

EXCHANGE_KEYS = set(EXCHANGE_BOOKMAKERS.keys())

DEFAULT_COMMISSION = 0.05  # 5%

# -----------------------------------------------------------------------------
# Middles configuration
# -----------------------------------------------------------------------------

MIN_MIDDLE_GAP = 1.5
DEFAULT_MIDDLE_SORT = "ev"
SHOW_POSITIVE_EV_ONLY = True

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

DEFAULT_SPORT_KEYS = [
    "americanfootball_nfl",
    "basketball_nba",
    "baseball_mlb",
    "icehockey_nhl",
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
]

DEFAULT_SPORT_OPTIONS = [
    {"key": key, "label": SPORT_DISPLAY_NAMES.get(key, key)}
    for key in DEFAULT_SPORT_KEYS
]

AMERICAN_MARKETS = ["h2h", "spreads", "totals"]
SOCCER_MARKETS = ["spreads", "totals"]

ROI_BANDS = [
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

DEFAULT_SHARP_BOOK = "pinnacle"
DEFAULT_BANKROLL = 1000.0
MIN_EDGE_PERCENT = 1.0
DEFAULT_KELLY_FRACTION = 0.25
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

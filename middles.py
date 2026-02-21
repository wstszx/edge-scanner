"""Middle-bet detection and probability / EV calculation helpers."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Gap geometry
# ---------------------------------------------------------------------------

def spread_gap_info(favorite_line: float, underdog_line: float) -> Optional[dict]:
    """Return gap metadata for a spread middle, or None if no gap exists.

    ``favorite_line`` should be negative (e.g. -3) and
    ``underdog_line`` positive (e.g. +3.5).
    """
    if favorite_line is None or underdog_line is None:
        return None
    if favorite_line >= 0 or underdog_line <= 0:
        return None
    fav_abs = abs(favorite_line)
    if fav_abs >= underdog_line:
        return None
    gap = round(underdog_line - fav_abs, 2)
    start = math.floor(fav_abs) + 1
    end = math.ceil(underdog_line) - 1
    if end < start:
        return None
    middle_integers = list(range(start, end + 1))
    if not middle_integers:
        return None
    return {
        "gap_points": round(gap, 2),
        "middle_integers": middle_integers,
        "integer_count": len(middle_integers),
    }


_spread_gap_info = spread_gap_info


def total_gap_info(over_line: float, under_line: float) -> Optional[dict]:
    """Return gap metadata for a totals middle, or None if no gap exists."""
    if over_line is None or under_line is None:
        return None
    if over_line >= under_line:
        return None
    gap = under_line - over_line
    lower = math.floor(over_line) + 1
    upper = math.ceil(under_line) - 1
    if upper < lower:
        return None
    middle_integers = list(range(lower, upper + 1))
    if not middle_integers:
        return None
    return {
        "gap_points": round(gap, 2),
        "middle_integers": middle_integers,
        "integer_count": len(middle_integers),
    }


_total_gap_info = total_gap_info


# ---------------------------------------------------------------------------
# Probability estimation
# ---------------------------------------------------------------------------

# Default probability per integer in the window for each market type
PROBABILITY_PER_INTEGER_DEFAULT = 0.030

# NFL/NCAAF key number boost table
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

KEY_NUMBER_SPORTS = {"americanfootball_nfl", "americanfootball_ncaaf"}
MAX_MIDDLE_PROBABILITY = 0.35

# Sport-specific per-integer base probabilities
PROBABILITY_PER_INTEGER = {
    "americanfootball_nfl_spreads": 0.025,
    "americanfootball_ncaaf_spreads": 0.025,
    "basketball_nba_spreads": 0.025,
    "basketball_ncaab_spreads": 0.025,
    "baseball_mlb_spreads": 0.030,
    "icehockey_nhl_spreads": 0.030,
    "americanfootball_nfl_totals": 0.030,
    "americanfootball_ncaaf_totals": 0.030,
    "basketball_nba_totals": 0.020,
    "basketball_ncaab_totals": 0.020,
    "baseball_mlb_totals": 0.045,
    "icehockey_nhl_totals": 0.055,
    "default": PROBABILITY_PER_INTEGER_DEFAULT,
}


def estimate_middle_probability(
    middle_integers: List[int], sport_key: str, market_key: str
) -> float:
    """Estimate the probability that the result lands inside the middle window."""
    if not middle_integers:
        return 0.0
    lookup_key = f"{sport_key}_{market_key}"
    base_prob = PROBABILITY_PER_INTEGER.get(lookup_key, PROBABILITY_PER_INTEGER["default"])
    total = 0.0
    is_key_sport = sport_key in KEY_NUMBER_SPORTS
    for integer in middle_integers:
        abs_int = abs(integer)
        if is_key_sport and abs_int in NFL_KEY_NUMBER_PROBABILITY:
            total += NFL_KEY_NUMBER_PROBABILITY[abs_int]
        else:
            total += base_prob
    return min(total, MAX_MIDDLE_PROBABILITY)


_estimate_middle_probability = estimate_middle_probability


# ---------------------------------------------------------------------------
# Stake and EV calculation
# ---------------------------------------------------------------------------

def calculate_middle_stakes(
    odds_a: float, odds_b: float, total_stake: float
) -> Tuple[float, float]:
    """Split ``total_stake`` between both sides so both pay the same total payout."""
    if total_stake <= 0 or odds_a <= 1 or odds_b <= 1:
        return 0.0, 0.0
    profit_a = odds_a - 1
    profit_b = odds_b - 1
    denominator = profit_a + profit_b
    if denominator <= 0:
        return 0.0, 0.0
    stake_a = total_stake * profit_b / denominator
    stake_a = round(stake_a, 2)
    stake_b = round(total_stake - stake_a, 2)
    return stake_a, stake_b


_calculate_middle_stakes = calculate_middle_stakes


def calculate_middle_outcomes(
    stake_a: float, stake_b: float, odds_a: float, odds_b: float
) -> dict:
    """Return profit/loss for each result scenario."""
    total = stake_a + stake_b
    payout_a = round(stake_a * odds_a, 2)
    payout_b = round(stake_b * odds_b, 2)
    win_both = round((payout_a + payout_b) - total, 2)
    side_a_only = round(payout_a - total, 2)
    side_b_only = round(payout_b - total, 2)
    typical_miss = round((side_a_only + side_b_only) / 2, 2)
    return {
        "win_both_profit": win_both,
        "side_a_wins_profit": side_a_only,
        "side_b_wins_profit": side_b_only,
        "typical_miss_profit": typical_miss,
    }


_calculate_middle_outcomes = calculate_middle_outcomes


def calculate_middle_ev(
    win_both_profit: float,
    side_a_profit: float,
    side_b_profit: float,
    probability: float,
) -> float:
    """Return expected value of the middle bet."""
    probability = max(0.0, min(probability, 1.0))
    miss_probability = 1.0 - probability
    miss_ev = 0.5 * side_a_profit + 0.5 * side_b_profit
    value = (probability * win_both_profit) + (miss_probability * miss_ev)
    return round(value, 2)


_calculate_middle_ev = calculate_middle_ev


def format_middle_zone(
    description_source: str, middle_integers: List[int], is_total: bool
) -> str:
    """Return a human-readable description of the middle window."""
    if not middle_integers:
        return description_source
    middle_integers = sorted(middle_integers)
    if len(middle_integers) == 1:
        range_text = str(middle_integers[0])
    else:
        range_text = f"{middle_integers[0]}-{middle_integers[-1]}"
    if is_total:
        return f"Total {range_text}"
    return f"{description_source} by {range_text}"


_format_middle_zone = format_middle_zone

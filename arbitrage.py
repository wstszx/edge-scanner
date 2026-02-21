"""Arbitrage opportunity detection and stake calculation helpers."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def remove_vig(odds_a: float, odds_b: float) -> Tuple[float, float, float]:
    """Return fair odds for both sides plus vig percent (0-100)."""
    if odds_a <= 1 or odds_b <= 1:
        return odds_a, odds_b, 0.0
    implied_a = 1 / odds_a
    implied_b = 1 / odds_b
    total_implied = implied_a + implied_b
    if total_implied <= 0:
        return odds_a, odds_b, 0.0
    true_prob_a = implied_a / total_implied
    true_prob_b = implied_b / total_implied
    fair_a = 1 / true_prob_a if true_prob_a else odds_a
    fair_b = 1 / true_prob_b if true_prob_b else odds_b
    vig_percent = max(0.0, (total_implied - 1.0) * 100)
    return fair_a, fair_b, vig_percent


# Backwards-compatible private alias used by existing tests and scanner.py
_remove_vig = remove_vig


def kelly_stake(
    true_probability: float, odds: float, bankroll: float, fraction: float
) -> Tuple[float, float, float]:
    """Return (full_kelly_pct, fractional_kelly_pct, recommended_stake).

    Returns (0, 0, 0) when the bet has no edge.
    """
    if bankroll <= 0 or odds <= 1:
        return 0.0, 0.0, 0.0
    p = max(0.0, min(true_probability, 1.0))
    q = 1.0 - p
    b = odds - 1.0
    if b <= 0:
        return 0.0, 0.0, 0.0
    kelly_pct = (b * p - q) / b
    if kelly_pct <= 0:
        return 0.0, 0.0, 0.0
    fraction = max(0.0, min(fraction, 1.0))
    recommended_fraction = kelly_pct * fraction
    stake = round(bankroll * recommended_fraction, 2)
    return round(kelly_pct * 100, 2), round(recommended_fraction * 100, 2), stake


# Backwards-compatible private alias
_kelly_stake = kelly_stake


def calculate_edge_percent(soft_odds: float, fair_odds: float) -> float:
    """Return the edge of soft book odds vs fair book odds as a percentage."""
    if fair_odds <= 0:
        return 0.0
    return (soft_odds / fair_odds - 1.0) * 100


_calculate_edge_percent = calculate_edge_percent


def calculate_ev(true_probability: float, odds: float, stake: float) -> float:
    """Return expected value of a bet."""
    true_probability = max(0.0, min(true_probability, 1.0))
    win_amount = stake * (odds - 1.0)
    lose_amount = stake
    value = (true_probability * win_amount) - ((1.0 - true_probability) * lose_amount)
    return round(value, 2)


_calculate_ev = calculate_ev


def arbitrage_roi(prices: Sequence[float]) -> float:
    """Return ROI percent for a set of outcome prices (arbitrage exists when > 0)."""
    if not prices or any(p <= 1 for p in prices):
        return -100.0
    inverse_sum = sum(1.0 / p for p in prices)
    if inverse_sum <= 0:
        return -100.0
    return round((1.0 / inverse_sum - 1.0) * 100, 4)


def fair_odds_from_prices(prices: Sequence[float]) -> List[float]:
    """Remove vig from a list of outcome prices (any number of outcomes)."""
    if not prices or any(p <= 1 for p in prices):
        return list(prices)
    implied = [1.0 / p for p in prices]
    total = sum(implied)
    if total <= 0:
        return list(prices)
    true_probs = [i / total for i in implied]
    return [1.0 / tp if tp > 0 else p for tp, p in zip(true_probs, prices)]

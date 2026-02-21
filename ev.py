"""Positive Expected Value (+EV) detection helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from arbitrage import (
    calculate_edge_percent,
    calculate_ev,
    kelly_stake,
    remove_vig,
    _calculate_edge_percent,
    _calculate_ev,
    _kelly_stake,
    _remove_vig,
)

__all__ = [
    "apply_commission",
    "build_sharp_reference",
    "calculate_edge_percent",
    "calculate_ev",
    "kelly_stake",
    "remove_vig",
    # private compat aliases
    "_apply_commission",
    "_build_sharp_reference",
    "_calculate_edge_percent",
    "_calculate_ev",
    "_kelly_stake",
    "_remove_vig",
]


def apply_commission(price: float, commission_rate: float, is_exchange: bool) -> float:
    """Reduce an exchange price by the commission rate; pass through bookmaker prices."""
    if not is_exchange:
        return price
    edge = price - 1.0
    if edge <= 0:
        return price
    return 1.0 + edge * (1.0 - commission_rate)


_apply_commission = apply_commission


def build_sharp_reference(
    bookmaker: dict,
    commission_rate: float,
    is_exchange: bool,
    exchange_keys: Optional[set] = None,
) -> Dict[Tuple[str, str], dict]:
    """Build a mapping of (market_key, line_key) → vig-free fair odds from a sharp bookmaker.

    Callers must provide a compatible ``_line_key`` function (from scanner.py) via
    the module-level ``set_line_key_fn`` to avoid a circular import.
    """
    line_fn = _line_key_fn
    if line_fn is None:
        return {}
    outcome_disp_fn = _outcome_display_name_fn
    line_map: Dict[Tuple[str, str], List[dict]] = {}
    for market in bookmaker.get("markets", []):
        market_key = market.get("key")
        if not market_key:
            continue
        for outcome in market.get("outcomes", []):
            price = outcome.get("price")
            if not price or price <= 1:
                continue
            line_key = line_fn(market_key, outcome)
            if not line_key:
                continue
            try:
                price_val = float(price)
            except (TypeError, ValueError):
                continue
            adjusted_price = apply_commission(price_val, commission_rate, is_exchange)
            display_name = outcome_disp_fn(outcome) if outcome_disp_fn else str(outcome.get("name", ""))
            line_map.setdefault((market_key, line_key), []).append(
                {
                    "name": (outcome.get("name") or "").strip().lower(),
                    "display_name": display_name,
                    "price": adjusted_price,
                    "raw_price": price_val,
                    "point": outcome.get("point"),
                }
            )
    references: Dict[Tuple[str, str], dict] = {}
    for key, entries in line_map.items():
        if len(entries) != 2:
            continue
        first, second = entries
        fair_a, fair_b, vig_percent = remove_vig(first["price"], second["price"])
        prob_a = 1 / fair_a if fair_a else 0.0
        prob_b = 1 / fair_b if fair_b else 0.0
        references[key] = {
            "vig_percent": round(vig_percent, 2),
            "outcomes": {
                first["name"]: {
                    "fair_odds": fair_a,
                    "true_probability": prob_a,
                    "sharp_odds": first["raw_price"],
                    "opponent_odds": second["raw_price"],
                    "opponent_name": second["display_name"],
                    "display_name": first["display_name"],
                    "point": first.get("point"),
                },
                second["name"]: {
                    "fair_odds": fair_b,
                    "true_probability": prob_b,
                    "sharp_odds": second["raw_price"],
                    "opponent_odds": first["raw_price"],
                    "opponent_name": first["display_name"],
                    "display_name": second["display_name"],
                    "point": second.get("point"),
                },
            },
        }
    return references


_build_sharp_reference = build_sharp_reference


# ---------------------------------------------------------------------------
# Dependency injection hooks (set by scanner.py at import time to avoid circular deps)
# ---------------------------------------------------------------------------

_line_key_fn = None  # Callable[[str, dict], Optional[str]]
_outcome_display_name_fn = None  # Callable[[dict], str]


def set_helpers(line_key_fn, outcome_display_name_fn) -> None:
    """Register helper functions from scanner.py to avoid circular imports."""
    global _line_key_fn, _outcome_display_name_fn
    _line_key_fn = line_key_fn
    _outcome_display_name_fn = outcome_display_name_fn

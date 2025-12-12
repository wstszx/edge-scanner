"""Core arbitrage scanning logic."""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Sequence, Tuple

import requests

from config import (
    DEFAULT_SPORT_KEYS,
    REGIONS,
    ROI_BANDS,
    SPORT_DISPLAY_NAMES,
    markets_for_sport,
)

BASE_URL = "https://api.the-odds-api.com/v4"


class ScannerError(Exception):
    """Raised for recoverable scanner issues."""


def _iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _request(url: str, params: Dict[str, str]) -> requests.Response:
    try:
        resp = requests.get(url, params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network error
        raise ScannerError(f"Network error: {exc}") from exc
    if resp.status_code >= 400:
        try:
            payload = resp.json()
            message = payload.get("message") or payload.get("error")
        except ValueError:
            message = resp.text or "Unknown error"
        raise ScannerError(message or f"API request failed ({resp.status_code})")
    return resp


def fetch_sports(api_key: str) -> List[dict]:
    url = f"{BASE_URL}/sports/"
    resp = _request(url, {"apiKey": api_key})
    try:
        return resp.json()
    except ValueError as exc:  # pragma: no cover - malformed payload
        raise ScannerError("Failed to parse sports list") from exc


def filter_sports(
    sports: Sequence[dict], requested: Sequence[str], all_sports: bool
) -> List[dict]:
    if all_sports:
        return [s for s in sports if s.get("active") and not s.get("has_outrights")]
    requested_set = set(requested) if requested else set(DEFAULT_SPORT_KEYS)
    return [s for s in sports if s.get("key") in requested_set and s.get("active")]


def fetch_odds_for_sport(api_key: str, sport_key: str, markets: Sequence[str]) -> List[dict]:
    url = f"{BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
        "commenceTimeFrom": _iso_now(),
    }
    resp = _request(url, params)
    try:
        return resp.json()
    except ValueError as exc:  # pragma: no cover
        raise ScannerError(f"Failed to parse odds for {sport_key}") from exc


OutcomeInfo = Dict[str, object]
LineMap = Dict[str, Dict[str, OutcomeInfo]]


def _line_key(market: str, outcome: dict) -> Optional[str]:
    if market == "h2h":
        return "moneyline"
    point = outcome.get("point")
    if point is None:
        return None
    try:
        point_val = float(point)
    except (TypeError, ValueError):
        return None
    if market == "spreads":
        return f"spread_{abs(point_val):.2f}"
    return f"total_{point_val:.2f}"


def _record_best_prices(markets: List[dict], market_key: str) -> LineMap:
    lines: LineMap = {}
    for book in markets:
        bookmaker = book.get("title") or book.get("key")
        for market in book.get("markets", []):
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []):
                price = outcome.get("price")
                if not price or price <= 1:
                    continue
                key = _line_key(market_key, outcome)
                if key is None:
                    continue
                normalized_point = outcome.get("point")
                entry = lines.setdefault(key, {})
                name = outcome.get("name", "")
                existing = entry.get(name)
                if not existing or price > existing["price"]:
                    entry[name] = {
                        "price": float(price),
                        "bookmaker": bookmaker,
                        "name": name,
                        "point": normalized_point,
                    }
    return lines


def _collect_market_entries(game: dict, market_key: str, stake_total: float) -> List[dict]:
    bookmakers = game.get("bookmakers", [])
    markets = [
        {
            "title": book.get("title") or book.get("key"),
            "markets": book.get("markets", []),
        }
        for book in bookmakers
    ]
    best_lines = _record_best_prices(markets, market_key)
    entries = []
    for line_key, offers in best_lines.items():
        # Skip lines that aren't genuine two-way markets (e.g., moneyline with a draw option).
        if len(offers) != 2:
            continue
        outcomes = list(offers.values())
        outcome_payload = [
            {
                "outcome": o["name"],
                "bookmaker": o["bookmaker"],
                "price": o["price"],
                "point": o.get("point"),
            }
            for o in outcomes
        ]
        stake_info = _calculate_stakes(outcomes, stake_total)
        if stake_info["roi_percent"] <= 0:
            continue
        entry = {
            "sport": game.get("sport_key"),
            "sport_display": game.get("sport_display")
            or SPORT_DISPLAY_NAMES.get(game.get("sport_key", ""), game.get("sport_key")),
            "sport_title": game.get("sport_title"),
            "event": f"{game.get('away_team')} vs {game.get('home_team')}",
            "commence_time": game.get("commence_time"),
            "market": market_key,
            "roi_percent": round(stake_info["roi_percent"], 2),
            "best_odds": outcome_payload,
            "stakes": stake_info,
        }
        entries.append(entry)
    return entries


def _calculate_stakes(outcomes: List[dict], stake_total: float) -> dict:
    if stake_total <= 0 or len(outcomes) < 2:
        return {
            "total": 0.0,
            "breakdown": [],
            "guaranteed_profit": 0.0,
            "roi_percent": 0.0,
        }
    inverses = []
    for outcome in outcomes:
        price = outcome.get("price")
        if not price or price <= 0:
            return {
                "total": 0.0,
                "breakdown": [],
                "guaranteed_profit": 0.0,
                "roi_percent": 0.0,
            }
        inverses.append(1 / price)
    inverse_sum = sum(inverses)
    if inverse_sum <= 0:
        return {
            "total": 0.0,
            "breakdown": [],
            "guaranteed_profit": 0.0,
            "roi_percent": 0.0,
        }
    breakdown = []
    for outcome, inv in zip(outcomes, inverses):
        fraction = inv / inverse_sum
        stake_value = round(stake_total * fraction, 2)
        payout = round(stake_value * outcome["price"], 2)
        breakdown.append(
            {
                "outcome": outcome["name"],
                "bookmaker": outcome["bookmaker"],
                "price": outcome["price"],
                "point": outcome.get("point"),
                "stake": stake_value,
                "payout": payout,
                "fraction": fraction,
            }
        )
    guaranteed = round(breakdown[0]["payout"] - stake_total, 2)
    roi = round((guaranteed / stake_total) * 100, 4) if stake_total else 0.0
    return {
        "total": stake_total,
        "breakdown": breakdown,
        "guaranteed_profit": guaranteed,
        "roi_percent": roi,
    }


def _summaries(
    opportunities: List[dict],
    sports_scanned: int,
    events_scanned: int,
    total_profit: float,
) -> dict:
    by_sport: Dict[str, int] = {}
    band_counts: Dict[str, int] = {label: 0 for *_, label in ROI_BANDS}
    for opp in opportunities:
        key = opp.get("sport_display") or opp.get("sport") or "unknown"
        by_sport[key] = by_sport.get(key, 0) + 1
        roi = opp.get("roi_percent", 0)
        for lower, upper, label in ROI_BANDS:
            if lower <= roi < upper:
                band_counts[label] += 1
                break
    return {
        "by_sport": by_sport,
        "by_roi_band": band_counts,
        "sports_scanned": sports_scanned,
        "events_scanned": events_scanned,
        "api_calls_used": sports_scanned,
        "total_guaranteed_profit": round(total_profit, 2),
    }


def run_scan(
    api_key: str,
    sports: Optional[List[str]] = None,
    all_sports: bool = False,
    stake_amount: float = 100.0,
) -> dict:
    if not api_key:
        return {"success": False, "error": "API key is required", "error_code": 400}
    if stake_amount is None or stake_amount <= 0:
        stake_amount = 100.0
    try:
        sports_list = fetch_sports(api_key)
    except ScannerError as exc:
        return {"success": False, "error": str(exc), "error_code": 500}

    filtered = filter_sports(sports_list, sports or DEFAULT_SPORT_KEYS, all_sports)
    if not filtered:
        return {
            "success": True,
            "scan_time": _iso_now(),
            "opportunities_count": 0,
            "opportunities": [],
            "summary": {
                "by_sport": {},
                "by_roi_band": {label: 0 for *_, label in ROI_BANDS},
                "sports_scanned": 0,
                "events_scanned": 0,
                "api_calls_used": 0,
            },
        }

    opportunities: List[dict] = []
    events_scanned = 0
    total_profit = 0.0

    for sport in filtered:
        sport_key = sport.get("key")
        if not sport_key:
            continue
        markets = markets_for_sport(sport_key)
        try:
            events = fetch_odds_for_sport(api_key, sport_key, markets)
        except ScannerError as exc:
            return {"success": False, "error": str(exc), "error_code": 500}
        events_scanned += len(events)
        for game in events:
            game["sport_key"] = sport_key
            game["sport_title"] = sport.get("title")
            game["sport_display"] = SPORT_DISPLAY_NAMES.get(sport_key, sport_key)
            for market_key in markets:
                new_entries = _collect_market_entries(game, market_key, stake_amount)
                for entry in new_entries:
                    total_profit += entry["stakes"].get("guaranteed_profit", 0.0)
                opportunities.extend(new_entries)

    opportunities.sort(key=lambda x: x["roi_percent"], reverse=True)
    summary = _summaries(opportunities, len(filtered), events_scanned, total_profit)
    return {
        "success": True,
        "scan_time": _iso_now(),
        "opportunities_count": len(opportunities),
        "opportunities": opportunities,
        "summary": summary,
        "stake_amount": stake_amount,
    }

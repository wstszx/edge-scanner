from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scanner as scanner_module
from providers import _async_http as async_http
from providers import PROVIDER_CAPABILITIES
from scanner import run_scan


CORE_SPORT_ORDER = [
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
    "americanfootball_nfl",
    "americanfootball_ncaaf",
]

DEFAULT_PROVIDERS = [
    "betdex",
    "bookmaker_xyz",
    "artline",
    "sx_bet",
    "polymarket",
]


def _default_sports() -> list[str]:
    seen: set[str] = set()
    sports: list[str] = []
    for sport_key in CORE_SPORT_ORDER:
        if sport_key not in seen:
            sports.append(sport_key)
            seen.add(sport_key)
    long_tail: set[str] = set()
    for provider_key in DEFAULT_PROVIDERS:
        capability = PROVIDER_CAPABILITIES.get(provider_key)
        if not capability:
            continue
        for sport_key in capability.supported_sport_keys:
            normalized = str(sport_key).strip()
            if normalized and not normalized.startswith("azuro__") and normalized not in seen:
                long_tail.add(normalized)
    for sport_key in sorted(long_tail):
        sports.append(sport_key)
        seen.add(sport_key)
    return sports


DEFAULT_SPORTS = _default_sports()


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _execution_status(item: dict[str, Any]) -> str:
    quality = item.get("execution_quality")
    if isinstance(quality, dict):
        status = str(quality.get("status") or "").strip().lower()
        if status:
            return status
    return "unknown"


def _is_candidate(item: dict[str, Any], *, min_roi: float, allowed_quality: set[str]) -> bool:
    roi = _safe_float(item.get("roi_percent"))
    if roi is None or roi < min_roi:
        return False
    return _execution_status(item) in allowed_quality


def _is_actionable_arbitrage(item: dict[str, Any], *, min_roi: float = 0.0) -> bool:
    roi = _safe_float(item.get("roi_percent"))
    if roi is None or roi <= 0 or roi < min_roi:
        return False
    quality = item.get("execution_quality") if isinstance(item.get("execution_quality"), dict) else {}
    status = str(quality.get("status") or "").strip().lower()
    flags = quality.get("flags") or []
    if status != "high" or flags:
        return False
    stakes = item.get("stakes") if isinstance(item.get("stakes"), dict) else {}
    if stakes.get("limited_by_max_stake"):
        return False
    return True


def _provider_capability_summary(providers: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for provider_key in providers:
        key = str(provider_key).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        capability = PROVIDER_CAPABILITIES.get(key)
        if not capability:
            rows.append(
                {
                    "key": key,
                    "title": key,
                    "liquidity_confidence": "unknown",
                    "live_mode_supported": False,
                    "notes": ["Provider is not registered in local capability metadata."],
                }
            )
            continue
        rows.append(
            {
                "key": capability.key,
                "title": capability.title,
                "liquidity_confidence": capability.liquidity_confidence,
                "live_mode_supported": capability.live_mode_supported,
                "notes": list(capability.notes),
            }
        )
    return rows


def _flatten_provider_sets(provider_sets: Sequence[Sequence[str]]) -> list[str]:
    providers: list[str] = []
    seen: set[str] = set()
    for provider_set in provider_sets:
        for provider_key in provider_set:
            key = str(provider_key).strip()
            if key and key not in seen:
                providers.append(key)
                seen.add(key)
    return providers


def _book_summary(item: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for leg in item.get("best_odds") or []:
        if not isinstance(leg, dict):
            continue
        rows.append(
            {
                "outcome": leg.get("outcome"),
                "bookmaker": leg.get("bookmaker"),
                "bookmaker_key": leg.get("bookmaker_key"),
                "price": leg.get("price"),
                "effective_price": leg.get("effective_price"),
                "fee_rate": leg.get("fee_rate"),
                "max_stake": leg.get("max_stake"),
                "quote_updated_at": leg.get("quote_updated_at"),
                "quote_source": leg.get("quote_source"),
                "is_exchange": leg.get("is_exchange"),
                "book_event_id": leg.get("book_event_id"),
                "book_event_url": leg.get("book_event_url"),
                "market_hash": leg.get("market_hash"),
                "token_id": leg.get("token_id"),
                "asset_id": leg.get("asset_id"),
                "condition_id": leg.get("condition_id"),
                "outcome_index": leg.get("outcome_index"),
                "selection_id": leg.get("selection_id") or leg.get("selectionId"),
                "provider_event_name": leg.get("provider_event_name") or leg.get("providerEventName"),
                "execution_diagnostics": leg.get("execution_diagnostics"),
            }
        )
    return rows


def _leg_quality_flags(legs: Iterable[dict[str, Any]], stake: dict[str, Any] | None = None) -> list[str]:
    flags: set[str] = set()
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        if not leg.get("quote_updated_at"):
            flags.add("missing_quote_time")
        max_stake = _safe_float(leg.get("max_stake"))
        if max_stake is None:
            flags.add("missing_liquidity")
        elif max_stake <= 0:
            flags.add("zero_liquidity")
    if isinstance(stake, dict) and stake.get("limited_by_max_stake"):
        flags.add("stake_limited")
    return sorted(flags)


def _compact_arb(item: dict[str, Any], sport: str, providers: Sequence[str], all_markets: bool) -> dict[str, Any]:
    return {
        "sport": sport,
        "providers": list(providers),
        "all_markets": bool(all_markets),
        "event": item.get("event"),
        "market": item.get("market"),
        "roi_percent": item.get("roi_percent"),
        "gross_roi_percent": item.get("gross_roi_percent"),
        "profit": item.get("profit"),
        "stakes": item.get("stakes"),
        "gross_stakes": item.get("gross_stakes"),
        "execution_quality": item.get("execution_quality"),
        "books": _book_summary(item),
    }


def _compact_middle(item: dict[str, Any], sport: str, providers: Sequence[str], all_markets: bool) -> dict[str, Any]:
    side_a = item.get("side_a") if isinstance(item.get("side_a"), dict) else {}
    side_b = item.get("side_b") if isinstance(item.get("side_b"), dict) else {}
    stake = item.get("stakes") if isinstance(item.get("stakes"), dict) else {}
    books = []
    for side in (side_a, side_b):
        books.append(
            {
                "bookmaker": side.get("bookmaker"),
                "bookmaker_key": side.get("bookmaker_key"),
                "price": side.get("price"),
                "effective_price": side.get("effective_price"),
                "fee_rate": side.get("fee_rate"),
                "line": side.get("line"),
                "max_stake": side.get("max_stake"),
                "quote_updated_at": side.get("quote_updated_at"),
                "quote_source": side.get("quote_source"),
                "is_exchange": side.get("is_exchange"),
                "liquidity_provenance": side.get("liquidity_provenance"),
                "book_event_id": side.get("book_event_id"),
                "book_event_url": side.get("book_event_url"),
                "market_hash": side.get("market_hash"),
                "token_id": side.get("token_id"),
                "asset_id": side.get("asset_id"),
                "condition_id": side.get("condition_id"),
                "outcome_index": side.get("outcome_index"),
                "selection_id": side.get("selection_id") or side.get("selectionId"),
                "provider_event_name": side.get("provider_event_name") or side.get("providerEventName"),
                "execution_diagnostics": side.get("execution_diagnostics"),
            }
        )
    return {
        "sport": sport,
        "providers": list(providers),
        "all_markets": bool(all_markets),
        "event": item.get("event"),
        "market": item.get("market"),
        "middle_zone": item.get("middle_zone"),
        "probability_percent": item.get("probability_percent"),
        "ev_percent": item.get("ev_percent"),
        "gross_ev_percent": item.get("gross_ev_percent"),
        "gap": item.get("gap") or {},
        "stake": {
            "requested_total": stake.get("requested_total"),
            "total": stake.get("total"),
            "max_executable_total": stake.get("max_executable_total"),
            "limited_by_max_stake": bool(stake.get("limited_by_max_stake")),
            "side_a": stake.get("side_a"),
            "side_b": stake.get("side_b"),
        },
        "books": books,
        "risk_flags": _leg_quality_flags(books, stake),
    }


def _is_actionable_middle(item: dict[str, Any]) -> bool:
    ev_percent = _safe_float(item.get("ev_percent"))
    if ev_percent is None or ev_percent <= 0:
        return False
    if item.get("risk_flags"):
        return False
    stake = item.get("stake") if isinstance(item.get("stake"), dict) else {}
    if stake.get("limited_by_max_stake"):
        return False
    return True


def _compact_plus_ev(item: dict[str, Any], sport: str, providers: Sequence[str], all_markets: bool) -> dict[str, Any]:
    bet = item.get("bet") if isinstance(item.get("bet"), dict) else {}
    sharp = item.get("sharp") if isinstance(item.get("sharp"), dict) else {}
    bet_leg = {
        "bookmaker": bet.get("soft_book"),
        "bookmaker_key": bet.get("soft_key"),
        "outcome": bet.get("outcome"),
        "odds": bet.get("soft_odds"),
        "effective_odds": bet.get("effective_odds"),
        "point": bet.get("point"),
        "quote_updated_at": bet.get("quote_updated_at"),
        "quote_source": bet.get("quote_source"),
        "is_exchange": bet.get("is_exchange"),
        "max_stake": bet.get("max_stake"),
        "book_event_id": bet.get("book_event_id"),
        "book_event_url": bet.get("book_event_url"),
        "market_hash": bet.get("market_hash"),
        "token_id": bet.get("token_id"),
        "asset_id": bet.get("asset_id"),
        "condition_id": bet.get("condition_id"),
        "outcome_index": bet.get("outcome_index"),
        "selection_id": bet.get("selection_id") or bet.get("selectionId"),
        "provider_event_name": bet.get("provider_event_name") or bet.get("providerEventName"),
        "execution_diagnostics": bet.get("execution_diagnostics"),
    }
    reference_leg = {
        "bookmaker": sharp.get("book"),
        "bookmaker_key": sharp.get("key"),
        "fair_odds": sharp.get("fair_odds"),
        "true_probability_percent": sharp.get("true_probability_percent"),
        "vig_percent": sharp.get("vig_percent"),
        "quote_updated_at": sharp.get("quote_updated_at"),
    }
    return {
        "sport": sport,
        "providers": list(providers),
        "all_markets": bool(all_markets),
        "event": item.get("event"),
        "market": item.get("market"),
        "market_point": item.get("market_point"),
        "edge_percent": item.get("edge_percent"),
        "net_edge_percent": item.get("net_edge_percent", item.get("edge_percent")),
        "gross_edge_percent": item.get("gross_edge_percent"),
        "ev_per_100": item.get("ev_per_100"),
        "bet": bet_leg,
        "reference": reference_leg,
        "kelly": item.get("kelly") or {},
        "risk_flags": _leg_quality_flags([bet_leg, reference_leg]),
    }


def _positive_count(items: Iterable[dict[str, Any]], metric_key: str) -> int:
    count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        value = _safe_float(item.get(metric_key))
        if value is not None and value > 0:
            count += 1
    return count


def _scan_once(
    sport: str,
    providers: Sequence[str],
    *,
    api_key: str,
    api_bookmakers: Sequence[str],
    all_markets: bool,
    stake: float,
    min_roi: float,
    allowed_quality: set[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    result = run_scan(
        api_key=api_key,
        sports=[sport],
        scan_mode="prematch",
        all_sports=False,
        all_markets=all_markets,
        stake_amount=stake,
        regions=["us", "eu"],
        bookmakers=list(api_bookmakers) + list(providers),
        include_providers=list(providers),
    )
    elapsed = round(time.perf_counter() - started, 3)
    arb_payload = result.get("arbitrage") if isinstance(result, dict) else {}
    opportunities = []
    if isinstance(arb_payload, dict):
        opportunities = [item for item in (arb_payload.get("opportunities") or []) if isinstance(item, dict)]
    middle_payload = result.get("middles") if isinstance(result, dict) else {}
    middle_opportunities: list[dict[str, Any]] = []
    if isinstance(middle_payload, dict):
        middle_opportunities = [
            item for item in (middle_payload.get("opportunities") or []) if isinstance(item, dict)
        ]
    plus_ev_payload = result.get("plus_ev") if isinstance(result, dict) else {}
    plus_ev_opportunities: list[dict[str, Any]] = []
    if isinstance(plus_ev_payload, dict):
        plus_ev_opportunities = [
            item for item in (plus_ev_payload.get("opportunities") or []) if isinstance(item, dict)
        ]
    top_arbitrage = [_compact_arb(item, sport, providers, all_markets) for item in opportunities]
    top_arbitrage.sort(key=lambda item: _safe_float(item.get("roi_percent")) or -999.0, reverse=True)
    candidates = [
        _compact_arb(item, sport, providers, all_markets)
        for item in opportunities
        if _is_candidate(item, min_roi=min_roi, allowed_quality=allowed_quality)
    ]
    candidates.sort(key=lambda item: _safe_float(item.get("roi_percent")) or -999.0, reverse=True)
    actionable_arbitrage = [
        _compact_arb(item, sport, providers, all_markets)
        for item in opportunities
        if _is_actionable_arbitrage(item, min_roi=min_roi)
    ]
    actionable_arbitrage.sort(key=lambda item: _safe_float(item.get("roi_percent")) or -999.0, reverse=True)
    top_middles = [_compact_middle(item, sport, providers, all_markets) for item in middle_opportunities]
    top_middles.sort(key=lambda item: _safe_float(item.get("ev_percent")) or -999.0, reverse=True)
    actionable_middles = [item for item in top_middles if _is_actionable_middle(item)]
    top_plus_ev = [_compact_plus_ev(item, sport, providers, all_markets) for item in plus_ev_opportunities]
    top_plus_ev.sort(key=lambda item: _safe_float(item.get("edge_percent")) or -999.0, reverse=True)
    return {
        "sport": sport,
        "providers": list(providers),
        "api_bookmakers": list(api_bookmakers),
        "all_markets": all_markets,
        "elapsed_seconds": elapsed,
        "success": bool(result.get("success")) if isinstance(result, dict) else False,
        "partial": bool(result.get("partial")) if isinstance(result, dict) else False,
        "arbitrage_count": len(opportunities),
        "positive_candidates": len(candidates),
        "actionable_arbitrage_count": len(actionable_arbitrage),
        "top_candidate": candidates[0] if candidates else None,
        "top_arbitrage": top_arbitrage[:5],
        "actionable_arbitrage": actionable_arbitrage[:5],
        "middle_count": len(middle_opportunities),
        "positive_middle_count": _positive_count(middle_opportunities, "ev_percent"),
        "actionable_middle_count": len(actionable_middles),
        "top_middles": top_middles[:5],
        "actionable_middles": actionable_middles[:5],
        "plus_ev_count": len(plus_ev_opportunities),
        "top_plus_ev": top_plus_ev[:5],
        "provider_capabilities": _provider_capability_summary(providers),
        "scan_diagnostics": result.get("scan_diagnostics") if isinstance(result, dict) else {},
        "cross_provider_match_report_path": (
            result.get("cross_provider_match_report_path") if isinstance(result, dict) else ""
        ),
        "cross_provider_match_report_summary": (
            result.get("cross_provider_match_report_summary") if isinstance(result, dict) else {}
        ),
        "provider_snapshot_paths": (
            result.get("provider_snapshot_paths") if isinstance(result, dict) else {}
        ),
        "custom_providers": result.get("custom_providers") if isinstance(result, dict) else {},
        "sport_errors": result.get("sport_errors") if isinstance(result, dict) else [],
    }


def _provider_sets(value: str) -> list[list[str]]:
    if value == "all":
        return [DEFAULT_PROVIDERS]
    if value == "pairs":
        pairs: list[list[str]] = []
        for idx, first in enumerate(DEFAULT_PROVIDERS):
            for second in DEFAULT_PROVIDERS[idx + 1 :]:
                pairs.append([first, second])
        return pairs
    sets: list[list[str]] = []
    for group in value.split(";"):
        providers = [item.strip() for item in group.split(",") if item.strip()]
        if providers:
            sets.append(providers)
    return sets or [DEFAULT_PROVIDERS]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _shutdown_runtime_resources() -> None:
    async_http.shutdown_shared_clients()
    scanner_module.shutdown_scan_runtime()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hunt for currently executable DEX/exchange-style arbitrage.")
    parser.add_argument("--sports", nargs="*", default=DEFAULT_SPORTS)
    parser.add_argument("--provider-sets", default="all")
    parser.add_argument("--api-key", default=os.getenv("ODDS_API_KEYS") or os.getenv("ODDS_API_KEY") or "")
    parser.add_argument("--api-bookmakers", nargs="*", default=[])
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--min-roi", type=float, default=0.01)
    parser.add_argument("--allow-quality", nargs="*", default=["high", "medium"])
    parser.add_argument("--all-markets", action="store_true")
    parser.add_argument("--stop-on-first", action="store_true")
    parser.add_argument("--out", default=str(Path("data") / "provider_verification" / "real_arb_hunt_latest.json"))
    args = parser.parse_args(argv)

    provider_sets = _provider_sets(args.provider_sets)
    allowed_quality = {str(item).strip().lower() for item in args.allow_quality if str(item).strip()}
    scans: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    actionable_arbitrage_candidates: list[dict[str, Any]] = []
    middle_candidates: list[dict[str, Any]] = []
    actionable_middle_candidates: list[dict[str, Any]] = []
    plus_ev_candidates: list[dict[str, Any]] = []
    try:
        if args.api_only:
            provider_sets = [[]]
        for sport in args.sports:
            for providers in provider_sets:
                api_bookmakers = [str(item).strip() for item in args.api_bookmakers if str(item).strip()]
                row = _scan_once(
                    sport,
                    providers,
                    api_key=str(args.api_key or "").strip(),
                    api_bookmakers=api_bookmakers,
                    all_markets=bool(args.all_markets),
                    stake=float(args.stake),
                    min_roi=float(args.min_roi),
                    allowed_quality=allowed_quality,
                )
                scans.append(row)
                top = row.get("top_candidate")
                if isinstance(top, dict):
                    candidates.append(top)
                actionable_arbitrage_candidates.extend(row.get("actionable_arbitrage") or [])
                middle_candidates.extend(row.get("top_middles") or [])
                actionable_middle_candidates.extend(row.get("actionable_middles") or [])
                plus_ev_candidates.extend(row.get("top_plus_ev") or [])
                print(
                    f"{sport} providers={','.join(providers) or '-'} all_markets={bool(args.all_markets)} "
                    f"api_books={','.join(api_bookmakers) or '-'} "
                    f"success={row['success']} partial={row['partial']} "
                    f"arb={row['arbitrage_count']} candidates={row['positive_candidates']} "
                    f"actionable_arbitrage={row['actionable_arbitrage_count']} "
                    f"middles={row['middle_count']} positive_middles={row['positive_middle_count']} "
                    f"actionable_middles={row['actionable_middle_count']} "
                    f"plus_ev={row['plus_ev_count']} "
                    f"elapsed={row['elapsed_seconds']}s"
                )
                if args.stop_on_first and candidates:
                    actionable_arbitrage_candidates.sort(
                        key=lambda item: _safe_float(item.get("roi_percent")) or -999.0,
                        reverse=True,
                    )
                    middle_candidates.sort(key=lambda item: _safe_float(item.get("ev_percent")) or -999.0, reverse=True)
                    actionable_middle_candidates.sort(
                        key=lambda item: _safe_float(item.get("ev_percent")) or -999.0,
                        reverse=True,
                    )
                    plus_ev_candidates.sort(key=lambda item: _safe_float(item.get("edge_percent")) or -999.0, reverse=True)
                    payload = {
                        "scans": scans,
                        "candidates": candidates,
                        "actionable_arbitrage": actionable_arbitrage_candidates[:20],
                        "top_middles": middle_candidates[:20],
                        "actionable_middles": actionable_middle_candidates[:20],
                        "top_plus_ev": plus_ev_candidates[:20],
                        "provider_capabilities": _provider_capability_summary(_flatten_provider_sets(provider_sets)),
                    }
                    _write_json(Path(args.out), payload)
                    print(f"FOUND candidate; wrote {args.out}")
                    print(json.dumps(candidates[0], ensure_ascii=False, indent=2))
                    return 0
        candidates.sort(key=lambda item: _safe_float(item.get("roi_percent")) or -999.0, reverse=True)
        actionable_arbitrage_candidates.sort(
            key=lambda item: _safe_float(item.get("roi_percent")) or -999.0,
            reverse=True,
        )
        middle_candidates.sort(key=lambda item: _safe_float(item.get("ev_percent")) or -999.0, reverse=True)
        actionable_middle_candidates.sort(
            key=lambda item: _safe_float(item.get("ev_percent")) or -999.0,
            reverse=True,
        )
        plus_ev_candidates.sort(key=lambda item: _safe_float(item.get("edge_percent")) or -999.0, reverse=True)
        payload = {
            "scans": scans,
            "candidates": candidates,
            "actionable_arbitrage": actionable_arbitrage_candidates[:20],
            "top_middles": middle_candidates[:20],
            "actionable_middles": actionable_middle_candidates[:20],
            "top_plus_ev": plus_ev_candidates[:20],
            "provider_capabilities": _provider_capability_summary(DEFAULT_PROVIDERS),
        }
        _write_json(Path(args.out), payload)
        positive_middles = [
            item
            for item in middle_candidates
            if (_safe_float(item.get("ev_percent")) or -999.0) > 0
        ]
        positive_plus_ev = [
            item
            for item in plus_ev_candidates
            if (_safe_float(item.get("edge_percent")) or -999.0) > 0
        ]
        if candidates:
            print(f"FOUND {len(candidates)} candidate(s); wrote {args.out}")
            print(json.dumps(candidates[0], ensure_ascii=False, indent=2))
            return 0
        if positive_middles or positive_plus_ev:
            print(
                f"No arbitrage candidate found; wrote {args.out} "
                f"with positive_middles={len(positive_middles)} "
                f"actionable_middles={len(actionable_middle_candidates)} "
                f"positive_plus_ev={len(positive_plus_ev)}"
            )
        else:
            print(f"No candidate found; wrote {args.out}")
        return 1
    finally:
        _shutdown_runtime_resources()


if __name__ == "__main__":
    raise SystemExit(main())

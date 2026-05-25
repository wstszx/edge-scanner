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
                "max_stake": leg.get("max_stake"),
                "quote_updated_at": leg.get("quote_updated_at"),
                "quote_source": leg.get("quote_source"),
                "is_exchange": leg.get("is_exchange"),
            }
        )
    return rows


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
    candidates = [
        _compact_arb(item, sport, providers, all_markets)
        for item in opportunities
        if _is_candidate(item, min_roi=min_roi, allowed_quality=allowed_quality)
    ]
    candidates.sort(key=lambda item: _safe_float(item.get("roi_percent")) or -999.0, reverse=True)
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
        "top_candidate": candidates[0] if candidates else None,
        "scan_diagnostics": result.get("scan_diagnostics") if isinstance(result, dict) else {},
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
            print(
                f"{sport} providers={','.join(providers) or '-'} all_markets={bool(args.all_markets)} "
                f"api_books={','.join(api_bookmakers) or '-'} "
                f"success={row['success']} partial={row['partial']} "
                f"arb={row['arbitrage_count']} candidates={row['positive_candidates']} "
                f"elapsed={row['elapsed_seconds']}s"
            )
            if args.stop_on_first and candidates:
                payload = {"scans": scans, "candidates": candidates}
                _write_json(Path(args.out), payload)
                print(f"FOUND candidate; wrote {args.out}")
                print(json.dumps(candidates[0], ensure_ascii=False, indent=2))
                return 0
    candidates.sort(key=lambda item: _safe_float(item.get("roi_percent")) or -999.0, reverse=True)
    payload = {"scans": scans, "candidates": candidates}
    _write_json(Path(args.out), payload)
    if candidates:
        print(f"FOUND {len(candidates)} candidate(s); wrote {args.out}")
        print(json.dumps(candidates[0], ensure_ascii=False, indent=2))
        return 0
    print(f"No candidate found; wrote {args.out}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

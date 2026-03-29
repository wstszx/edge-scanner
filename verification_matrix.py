from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from provider_verification import DEFAULT_PROVIDERS, run_live_scan, summarize_provider_statuses, summarize_scan
from config import DEFAULT_STAKE_AMOUNT, SPORT_OPTIONS
from scanner import run_scan


def run_provider_scan(*, sport_key: str, provider_keys: Sequence[str], scan_mode: str, regions: Sequence[str], stake_amount: float, all_markets: bool) -> dict[str, Any]:
    if str(scan_mode).strip().lower() == "live":
        return run_live_scan(
            sport_key=sport_key,
            provider_keys=provider_keys,
            regions=regions,
            stake_amount=stake_amount,
            all_markets=all_markets,
        )
    return run_scan(
        api_key="",
        sports=[sport_key],
        scan_mode="prematch",
        all_sports=False,
        all_markets=all_markets,
        stake_amount=stake_amount,
        regions=list(regions),
        bookmakers=None,
        include_providers=list(provider_keys),
    )


def run_matrix(*, sports: Sequence[str], provider_keys: Sequence[str], scan_modes: Sequence[str], regions: Sequence[str], stake_amount: float, all_markets: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sport_key in sports:
        for scan_mode in scan_modes:
            result = run_provider_scan(
                sport_key=sport_key,
                provider_keys=provider_keys,
                scan_mode=scan_mode,
                regions=regions,
                stake_amount=stake_amount,
                all_markets=all_markets,
            )
            summary = summarize_scan(result, top_n=3)
            provider_statuses = summarize_provider_statuses(
                result,
                provider_keys=provider_keys,
                stake_amount=stake_amount,
            )
            diagnostics = summary.get("scan_diagnostics") or {}
            rows.append({
                "sport_key": sport_key,
                "scan_mode": scan_mode,
                "success": bool(summary.get("success")),
                "partial": bool(summary.get("partial")),
                "reason_code": diagnostics.get("reason_code"),
                "arbitrage_count": int(summary.get("arbitrage_count", 0) or 0),
                "positive_arbitrage_count": int(summary.get("positive_arbitrage_count", 0) or 0),
                "middle_count": int(summary.get("middle_count", 0) or 0),
                "positive_middle_count": int(summary.get("positive_middle_count", 0) or 0),
                "plus_ev_count": int(summary.get("plus_ev_count", 0) or 0),
                "providers": [status.as_dict() for status in provider_statuses],
                "scan_diagnostics": diagnostics,
            })
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sport x mode x provider verification matrix.")
    parser.add_argument("--sports", nargs="*", default=[row["key"] for row in SPORT_OPTIONS])
    parser.add_argument("--scan-modes", nargs="*", default=["prematch", "live"])
    parser.add_argument("--providers", nargs="*", default=list(DEFAULT_PROVIDERS))
    parser.add_argument("--regions", nargs="*", default=["us", "eu"])
    parser.add_argument("--stake", type=float, default=float(DEFAULT_STAKE_AMOUNT))
    parser.add_argument("--all-markets", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> str:
    args = parse_args(argv)
    rows = run_matrix(
        sports=[str(item) for item in args.sports if str(item).strip()],
        provider_keys=[str(item) for item in args.providers if str(item).strip()],
        scan_modes=[str(item) for item in args.scan_modes if str(item).strip()],
        regions=[str(item) for item in args.regions if str(item).strip()],
        stake_amount=float(args.stake),
        all_markets=bool(args.all_markets),
    )
    output = json.dumps(rows, ensure_ascii=False, indent=2)
    return output


if __name__ == "__main__":
    print(main())

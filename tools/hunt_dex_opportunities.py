from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools import hunt_real_arbitrage as hunt


DEFAULT_MIN_EXECUTABLE_STAKE = 25.0


def _normalize_list(values: Sequence[str] | None) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if text and text not in seen:
            rows.append(text)
            seen.add(text)
    return rows


def _provider_has_explicit_liquidity(provider_key: str) -> bool:
    capability = hunt.PROVIDER_CAPABILITIES.get(str(provider_key).strip())
    return bool(capability and capability.liquidity_confidence == "explicit")


def _filter_explicit_providers(providers: Sequence[str], require_explicit_liquidity: bool) -> list[str]:
    normalized = _normalize_list(providers)
    if not require_explicit_liquidity:
        return normalized
    return [provider for provider in normalized if _provider_has_explicit_liquidity(provider)]


def _build_scan_jobs(
    *,
    sports: Sequence[str],
    provider_sets: str,
    include_all_markets: bool,
    all_markets_sports: Sequence[str],
    require_explicit_liquidity: bool = False,
) -> list[dict[str, Any]]:
    normalized_sports = _normalize_list(sports) or list(hunt.DEFAULT_SPORTS)
    all_markets_subset = _normalize_list(all_markets_sports) or normalized_sports
    raw_provider_sets = str(provider_sets or "").strip().lower()
    tokens = [item.strip() for item in raw_provider_sets.split(",") if item.strip()]
    if tokens and all(token not in {"all", "pairs"} for token in tokens):
        tokens = [raw_provider_sets]
    if not tokens:
        tokens = ["all"]

    jobs: list[dict[str, Any]] = []
    for token in tokens:
        if token == "all":
            providers = _filter_explicit_providers(hunt.DEFAULT_PROVIDERS, require_explicit_liquidity)
            if len(providers) < 2:
                continue
            jobs.append(
                {
                    "name": "all:base",
                    "providers": providers,
                    "sports": normalized_sports,
                    "all_markets": False,
                }
            )
            continue
        if token == "pairs":
            for first, second in combinations(hunt.DEFAULT_PROVIDERS, 2):
                providers = _filter_explicit_providers([first, second], require_explicit_liquidity)
                if len(providers) < 2:
                    continue
                jobs.append(
                    {
                        "name": f"pairs:{first}+{second}",
                        "providers": providers,
                        "sports": normalized_sports,
                        "all_markets": False,
                    }
                )
            continue
        providers = _filter_explicit_providers(
            [item.strip() for item in token.replace("+", ",").split(",") if item.strip()],
            require_explicit_liquidity,
        )
        if len(providers) >= 2:
            jobs.append(
                {
                    "name": f"custom:{'+'.join(providers)}",
                    "providers": providers,
                    "sports": normalized_sports,
                    "all_markets": False,
                }
            )

    if include_all_markets:
        providers = _filter_explicit_providers(hunt.DEFAULT_PROVIDERS, require_explicit_liquidity)
        if len(providers) < 2:
            return jobs
        jobs.append(
            {
                "name": "all:all_markets",
                "providers": providers,
                "sports": all_markets_subset,
                "all_markets": True,
            }
        )
    return jobs


def _bookmaker_key(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "bookmaker.xyz": "bookmaker_xyz",
        "bookmaker xyz": "bookmaker_xyz",
        "polymarket": "polymarket",
        "sx bet": "sx_bet",
        "sxbet": "sx_bet",
        "betdex": "betdex",
        "bet dex": "betdex",
    }
    return aliases.get(text, text.replace(".", "_").replace(" ", "_").replace("-", "_"))


def _canonical_event_key(value: object) -> tuple[str, ...]:
    text = str(value or "").strip().lower()
    if not text:
        return ()
    parts = re.split(r"\s+(?:vs|v|versus|@)\s+", text)
    if len(parts) < 2:
        parts = [text]
    teams: list[str] = []
    for part in parts:
        tokens = [token for token in re.sub(r"[^a-z0-9 ]+", " ", part).split() if token]
        if tokens:
            teams.append(tokens[-1])
    return tuple(sorted(teams))


def _book_signature(book: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _bookmaker_key(book.get("bookmaker") or book.get("bookmaker_key")),
        book.get("outcome"),
        book.get("line"),
        book.get("price") or book.get("odds"),
        book.get("effective_price") or book.get("effective_odds"),
    )


def _middle_key(item: dict[str, Any]) -> tuple[Any, ...]:
    books = item.get("books") if isinstance(item.get("books"), list) else []
    return (
        item.get("sport"),
        _canonical_event_key(item.get("event")),
        item.get("market"),
        item.get("middle_zone"),
        tuple(sorted(_book_signature(book) for book in books if isinstance(book, dict))),
    )


def _arb_key(item: dict[str, Any]) -> tuple[Any, ...]:
    books = item.get("books") if isinstance(item.get("books"), list) else []
    return (
        item.get("sport"),
        _canonical_event_key(item.get("event")),
        item.get("market"),
        tuple(sorted(_book_signature(book) for book in books if isinstance(book, dict))),
    )


def _plus_ev_key(item: dict[str, Any]) -> tuple[Any, ...]:
    bet = item.get("bet") if isinstance(item.get("bet"), dict) else {}
    reference = item.get("reference") if isinstance(item.get("reference"), dict) else {}
    return (
        item.get("sport"),
        _canonical_event_key(item.get("event")),
        item.get("market"),
        item.get("market_point"),
        _book_signature(
            {
                "bookmaker": bet.get("bookmaker") or bet.get("bookmaker_key"),
                "outcome": bet.get("outcome"),
                "odds": bet.get("odds"),
                "effective_odds": bet.get("effective_odds"),
            }
        ),
        _book_signature(
            {
                "bookmaker": reference.get("bookmaker") or reference.get("bookmaker_key"),
                "odds": reference.get("fair_odds"),
            }
        ),
    )


def _merge_seen_in_jobs(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    jobs: list[str] = []
    for source in (existing, incoming):
        for job_name in source.get("seen_in_jobs") or [source.get("job_name")]:
            text = str(job_name or "").strip()
            if text and text not in jobs:
                jobs.append(text)
    if jobs:
        existing["seen_in_jobs"] = jobs


def _deduplicate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    key_factory,
    metric_key: str,
) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = key_factory(row)
        current = dict(row)
        current.setdefault("seen_in_jobs", _normalize_list([current.get("job_name")]))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = current
            continue
        existing_metric = hunt._safe_float(existing.get(metric_key)) or -999.0
        current_metric = hunt._safe_float(current.get(metric_key)) or -999.0
        if current_metric > existing_metric:
            _merge_seen_in_jobs(current, existing)
            deduped[key] = current
        else:
            _merge_seen_in_jobs(existing, current)
    result = list(deduped.values())
    result.sort(key=lambda item: hunt._safe_float(item.get(metric_key)) or -999.0, reverse=True)
    return result


def _stake_limit_details(item: dict[str, Any], *, min_executable_stake: float) -> dict[str, Any] | None:
    minimum_required = max(0.0, float(min_executable_stake))
    if minimum_required <= 0:
        return None
    legs_below_minimum: list[dict[str, Any]] = []
    available_stakes: list[float] = []
    for book in item.get("books") or []:
        if not isinstance(book, dict):
            continue
        max_stake = hunt._safe_float(book.get("max_stake"))
        if max_stake is None:
            continue
        available_stakes.append(max_stake)
        if max_stake < minimum_required:
            legs_below_minimum.append(
                {
                    "bookmaker": book.get("bookmaker") or book.get("bookmaker_key"),
                    "max_stake": max_stake,
                }
            )
    if not legs_below_minimum:
        return None
    return {
        "minimum_required": minimum_required,
        "minimum_available": min(available_stakes) if available_stakes else None,
        "legs_below_minimum": legs_below_minimum,
    }


def _risk_flags_from_item(item: dict[str, Any]) -> set[str]:
    reasons = {str(flag).strip() for flag in item.get("risk_flags") or [] if str(flag).strip()}
    quality = item.get("execution_quality") if isinstance(item.get("execution_quality"), dict) else {}
    for flag in quality.get("flags") or []:
        text = str(flag).strip()
        if text:
            reasons.add(text)
    return reasons


def _blocked_reasons(item: dict[str, Any], *, min_executable_stake: float = 0.0) -> list[str]:
    reasons = _risk_flags_from_item(item)
    for book in item.get("books") or []:
        if not isinstance(book, dict):
            continue
        bookmaker_key = _bookmaker_key(book.get("bookmaker") or book.get("bookmaker_key"))
        capability = hunt.PROVIDER_CAPABILITIES.get(bookmaker_key)
        max_stake = hunt._safe_float(book.get("max_stake"))
        if capability and capability.liquidity_confidence == "quote_only" and max_stake is None:
            reasons.add(f"{bookmaker_key}_quote_only")
        diagnostics = book.get("execution_diagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("reason") == "max_bet_below_min_bet":
            reasons.add(f"{bookmaker_key}_max_bet_below_min_bet")
    roi = hunt._safe_float(item.get("roi_percent"))
    if roi is not None and roi <= 0:
        reasons.add("non_positive_roi")
    if _stake_limit_details(item, min_executable_stake=min_executable_stake):
        reasons.add("stake_below_minimum")
    return sorted(reasons)


def _annotate_blocked_reasons(
    rows: Sequence[dict[str, Any]],
    *,
    min_executable_stake: float = 0.0,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        current = dict(row)
        stake_limit = _stake_limit_details(current, min_executable_stake=min_executable_stake)
        current["blocked_reasons"] = _blocked_reasons(current, min_executable_stake=min_executable_stake)
        if stake_limit:
            current["stake_limit"] = stake_limit
        annotated.append(current)
    return annotated


def _blocked_reason_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("blocked_reasons") or [])
    return dict(sorted(counts.items()))


def _parse_quote_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _quote_time_skew_seconds(item: dict[str, Any]) -> int | None:
    quote_times = []
    for book in item.get("books") or []:
        if not isinstance(book, dict):
            continue
        quote_time = _parse_quote_time(book.get("quote_updated_at"))
        if quote_time is not None:
            quote_times.append(quote_time)
    if len(quote_times) < 2:
        return None
    return int((max(quote_times) - min(quote_times)).total_seconds())


def _with_quote_time_skew(item: dict[str, Any]) -> dict[str, Any]:
    current = dict(item)
    current["quote_time_skew_seconds"] = _quote_time_skew_seconds(current)
    return current


def _split_actionable_middles_by_quote_skew(
    rows: Sequence[dict[str, Any]],
    *,
    max_quote_skew_seconds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actionable: list[dict[str, Any]] = []
    risky: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        current = _with_quote_time_skew(row)
        quote_skew = current.get("quote_time_skew_seconds")
        if quote_skew is not None and quote_skew > max_quote_skew_seconds:
            risks = set(current.get("execution_risks") or [])
            risks.add("quote_time_skew")
            current["execution_risks"] = sorted(risks)
            risky.append(current)
        else:
            actionable.append(current)
    return actionable, risky


def _execution_risk_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("execution_risks") or [])
    return dict(sorted(counts.items()))


def _model_only_middles(middles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        item
        for item in middles
        if isinstance(item, dict)
        and (hunt._safe_float(item.get("ev_percent")) or -999.0) > 0
        and not hunt._is_actionable_middle(item)
    ]
    rows.sort(key=lambda item: hunt._safe_float(item.get("ev_percent")) or -999.0, reverse=True)
    return rows


def _model_only_arbitrage(arbitrage_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        item
        for item in arbitrage_rows
        if isinstance(item, dict) and not hunt._is_actionable_arbitrage(item)
    ]
    rows.sort(key=lambda item: hunt._safe_float(item.get("roi_percent")) or -999.0, reverse=True)
    return rows


def _sort_desc(rows: list[dict[str, Any]], key: str) -> None:
    rows.sort(key=lambda item: hunt._safe_float(item.get(key)) or -999.0, reverse=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_counts(counts: dict[str, Any]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in sorted(counts.items())]


def _format_opportunity_row(item: dict[str, Any], *, risk_key: str | None = None) -> str:
    parts = [
        str(item.get("event") or "-"),
        str(item.get("market") or "-"),
        str(item.get("middle_zone") or item.get("market_point") or "-"),
        f"ev={item.get('ev_percent', item.get('roi_percent', item.get('edge_percent', '-')))}",
    ]
    if risk_key:
        risks = item.get(risk_key) or []
        if risks:
            parts.append(f"{risk_key}={','.join(str(risk) for risk in risks)}")
    quote_skew = item.get("quote_time_skew_seconds")
    if quote_skew is not None:
        parts.append(f"quote_skew={quote_skew}s")
    return "- " + " | ".join(parts)


def _render_markdown_summary(payload: dict[str, Any], *, source_json: str) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# DEX Opportunity Summary",
        "",
        f"- source_json: {source_json}",
        f"- scan_count: {summary.get('scan_count', 0)}",
        f"- require_explicit_liquidity: {summary.get('require_explicit_liquidity', False)}",
        f"- max_quote_skew_seconds: {summary.get('max_quote_skew_seconds', 0)}",
        f"- min_executable_stake: {summary.get('min_executable_stake', 0)}",
        f"- actionable_arbitrage_count: {summary.get('actionable_arbitrage_count', 0)}",
        f"- model_only_arbitrage_count: {summary.get('model_only_arbitrage_count', 0)}",
        f"- actionable_middle_count: {summary.get('actionable_middle_count', 0)}",
        f"- execution_risky_middle_count: {summary.get('execution_risky_middle_count', 0)}",
        f"- model_only_middle_count: {summary.get('model_only_middle_count', 0)}",
        f"- plus_ev_count: {summary.get('plus_ev_count', 0)}",
        "",
        "## Execution Risk Counts",
        *_format_counts(summary.get("execution_risk_counts") or {}),
        "",
        "## Blocked Reason Counts",
        *_format_counts(summary.get("blocked_reason_counts") or {}),
        "",
        "## Top Actionable Middles",
    ]
    actionable_middles = [item for item in payload.get("actionable_middles") or [] if isinstance(item, dict)]
    lines.extend(_format_opportunity_row(item) for item in actionable_middles[:5])
    if not actionable_middles:
        lines.append("- none")

    lines.extend(["", "## Top Execution-Risky Middles"])
    risky_middles = [item for item in payload.get("execution_risky_middles") or [] if isinstance(item, dict)]
    lines.extend(_format_opportunity_row(item, risk_key="execution_risks") for item in risky_middles[:5])
    if not risky_middles:
        lines.append("- none")

    lines.extend(["", "## Top Model-Only Arbitrage"])
    model_only_arbitrage = [
        item for item in payload.get("model_only_arbitrage") or [] if isinstance(item, dict)
    ]
    lines.extend(
        _format_opportunity_row(item, risk_key="blocked_reasons") for item in model_only_arbitrage[:5]
    )
    if not model_only_arbitrage:
        lines.append("- none")

    lines.extend(["", "## Top Model-Only Middles"])
    model_only = [item for item in payload.get("model_only_middles") or [] if isinstance(item, dict)]
    lines.extend(_format_opportunity_row(item, risk_key="blocked_reasons") for item in model_only[:5])
    if not model_only:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _write_markdown_summary(path: Path, payload: dict[str, Any], *, source_json: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown_summary(payload, source_json=source_json), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a combined DEX-only opportunity hunt.")
    parser.add_argument("--sports", nargs="*", default=hunt.DEFAULT_SPORTS)
    parser.add_argument("--provider-sets", default="all,pairs")
    parser.add_argument("--include-all-markets", action="store_true")
    parser.add_argument("--all-markets-sports", nargs="*", default=[])
    parser.add_argument("--api-key", default=os.getenv("ODDS_API_KEYS") or os.getenv("ODDS_API_KEY") or "")
    parser.add_argument("--api-bookmakers", nargs="*", default=[])
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--min-roi", type=float, default=0.01)
    parser.add_argument("--allow-quality", nargs="*", default=["high", "medium"])
    parser.add_argument(
        "--max-quote-skew-seconds",
        type=int,
        default=120,
        help="Maximum allowed quote timestamp skew between opportunity legs before demoting a middle.",
    )
    parser.add_argument(
        "--require-explicit-liquidity",
        action="store_true",
        help="Only scan provider sets whose legs have explicit executable liquidity metadata.",
    )
    parser.add_argument(
        "--min-executable-stake",
        type=float,
        default=DEFAULT_MIN_EXECUTABLE_STAKE,
        help="Minimum per-leg stake required before a model-only middle is considered executable-looking.",
    )
    parser.add_argument("--out", default=str(Path("data") / "provider_verification" / "dex_opportunities_latest.json"))
    parser.add_argument(
        "--summary-out",
        default=str(Path("data") / "provider_verification" / "latest_actionable_summary.md"),
    )
    args = parser.parse_args(argv)

    jobs = _build_scan_jobs(
        sports=args.sports,
        provider_sets=str(args.provider_sets),
        include_all_markets=bool(args.include_all_markets),
        all_markets_sports=args.all_markets_sports,
        require_explicit_liquidity=bool(args.require_explicit_liquidity),
    )
    allowed_quality = {str(item).strip().lower() for item in args.allow_quality if str(item).strip()}
    api_bookmakers = _normalize_list(args.api_bookmakers)

    scans: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    top_arbitrage: list[dict[str, Any]] = []
    actionable_arbitrage: list[dict[str, Any]] = []
    top_middles: list[dict[str, Any]] = []
    actionable_middles: list[dict[str, Any]] = []
    plus_ev: list[dict[str, Any]] = []

    try:
        for job in jobs:
            for sport in job["sports"]:
                row = hunt._scan_once(
                    sport,
                    job["providers"],
                    api_key=str(args.api_key or "").strip(),
                    api_bookmakers=api_bookmakers,
                    all_markets=bool(job["all_markets"]),
                    stake=float(args.stake),
                    min_roi=float(args.min_roi),
                    allowed_quality=allowed_quality,
                )
                row["job_name"] = job["name"]
                scans.append(row)
                top_candidate = row.get("top_candidate")
                if isinstance(top_candidate, dict):
                    top_candidate["job_name"] = job["name"]
                    candidates.append(top_candidate)
                for item in row.get("top_arbitrage") or []:
                    if isinstance(item, dict):
                        item["job_name"] = job["name"]
                        top_arbitrage.append(item)
                for item in row.get("actionable_arbitrage") or []:
                    if isinstance(item, dict):
                        item["job_name"] = job["name"]
                        actionable_arbitrage.append(item)
                for item in row.get("top_middles") or []:
                    if isinstance(item, dict):
                        item["job_name"] = job["name"]
                        top_middles.append(item)
                for item in row.get("actionable_middles") or []:
                    if isinstance(item, dict):
                        item["job_name"] = job["name"]
                        actionable_middles.append(item)
                for item in row.get("top_plus_ev") or []:
                    if isinstance(item, dict):
                        item["job_name"] = job["name"]
                        plus_ev.append(item)
                print(
                    f"{job['name']} sport={sport} providers={'+'.join(job['providers'])} "
                    f"all_markets={bool(job['all_markets'])} "
                    f"arb={row['arbitrage_count']} candidates={row['positive_candidates']} "
                    f"actionable_arbitrage={row['actionable_arbitrage_count']} "
                    f"middles={row['middle_count']} actionable_middles={row['actionable_middle_count']} "
                    f"plus_ev={row['plus_ev_count']} elapsed={row['elapsed_seconds']}s"
                )

        candidates = _deduplicate_rows(candidates, key_factory=_arb_key, metric_key="roi_percent")
        top_arbitrage = _deduplicate_rows(top_arbitrage, key_factory=_arb_key, metric_key="roi_percent")
        actionable_arbitrage = _deduplicate_rows(
            actionable_arbitrage,
            key_factory=_arb_key,
            metric_key="roi_percent",
        )
        top_middles = _deduplicate_rows(top_middles, key_factory=_middle_key, metric_key="ev_percent")
        actionable_middles = _deduplicate_rows(
            actionable_middles,
            key_factory=_middle_key,
            metric_key="ev_percent",
        )
        actionable_middles, execution_risky_middles = _split_actionable_middles_by_quote_skew(
            actionable_middles,
            max_quote_skew_seconds=max(0, int(args.max_quote_skew_seconds)),
        )
        plus_ev = _deduplicate_rows(plus_ev, key_factory=_plus_ev_key, metric_key="edge_percent")
        min_executable_stake = max(0.0, float(args.min_executable_stake))
        model_only_arbitrage = _annotate_blocked_reasons(
            _model_only_arbitrage(top_arbitrage),
            min_executable_stake=min_executable_stake,
        )
        model_only_middles = _annotate_blocked_reasons(
            _model_only_middles(top_middles),
            min_executable_stake=min_executable_stake,
        )
        blocked_reason_counts = _blocked_reason_counts([*model_only_arbitrage, *model_only_middles])
        execution_risk_counts = _execution_risk_counts(execution_risky_middles)

        payload = {
            "summary": {
                "scan_count": len(scans),
                "candidate_count": len(candidates),
                "actionable_arbitrage_count": len(actionable_arbitrage),
                "model_only_arbitrage_count": len(model_only_arbitrage),
                "actionable_middle_count": len(actionable_middles),
                "execution_risky_middle_count": len(execution_risky_middles),
                "model_only_middle_count": len(model_only_middles),
                "plus_ev_count": len(plus_ev),
                "blocked_reason_counts": blocked_reason_counts,
                "execution_risk_counts": execution_risk_counts,
                "require_explicit_liquidity": bool(args.require_explicit_liquidity),
                "max_quote_skew_seconds": max(0, int(args.max_quote_skew_seconds)),
                "min_executable_stake": min_executable_stake,
            },
            "jobs": jobs,
            "scans": scans,
            "candidates": candidates[:20],
            "actionable_arbitrage": actionable_arbitrage[:20],
            "model_only_arbitrage": model_only_arbitrage[:20],
            "actionable_middles": actionable_middles[:20],
            "execution_risky_middles": execution_risky_middles[:20],
            "model_only_middles": model_only_middles[:20],
            "top_arbitrage": top_arbitrage[:20],
            "top_middles": top_middles[:20],
            "top_plus_ev": plus_ev[:20],
            "provider_capabilities": hunt._provider_capability_summary(hunt.DEFAULT_PROVIDERS),
        }
        out_path = Path(args.out)
        summary_path = Path(args.summary_out)
        _write_json(out_path, payload)
        _write_markdown_summary(summary_path, payload, source_json=str(out_path))

        if actionable_arbitrage or actionable_middles:
            print(
                f"FOUND actionable opportunities; wrote {args.out} and {args.summary_out} "
                f"actionable_arbitrage={len(actionable_arbitrage)} "
                f"actionable_middles={len(actionable_middles)}"
            )
            return 0
        print(
            f"No actionable opportunity found; wrote {args.out} and {args.summary_out} "
            f"candidates={len(candidates)} "
            f"model_only_arbitrage={len(model_only_arbitrage)} "
            f"execution_risky_middles={len(execution_risky_middles)} "
            f"model_only_middles={len(model_only_middles)} "
            f"plus_ev={len(plus_ev)}"
        )
        return 1
    finally:
        hunt._shutdown_runtime_resources()


if __name__ == "__main__":
    raise SystemExit(main())

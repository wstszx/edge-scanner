from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools import hunt_real_arbitrage as hunt

try:
    from providers import artline as artline_provider
except Exception:  # pragma: no cover - optional during standalone script use
    artline_provider = None


DEFAULT_MIN_EXECUTABLE_STAKE = 25.0
AUTO_EXECUTION_ADAPTER_PROVIDERS = {"polymarket", "sx_bet"}
MANUAL_EXECUTION_ADAPTER_PROVIDERS = {"artline"}
EXECUTION_ADAPTER_PROVIDERS = AUTO_EXECUTION_ADAPTER_PROVIDERS | MANUAL_EXECUTION_ADAPTER_PROVIDERS
PAPER_TRADE_SOFT_BLOCKERS = {
    "artline_max_bet_below_min_bet",
    "artline_not_executable",
    "execution_quality_not_high",
    "limited_by_liquidity",
    "wallet_not_ready",
}
EXECUTION_READY_ENV_BY_PROVIDER = {
    "polymarket": ("POLYMARKET_API_KEY",),
    "sx_bet": ("SX_BET_API_KEY",),
}


def _normalize_list(values: Sequence[str] | None) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        for text in (item.strip() for item in str(value).split(",")):
            if text and text not in seen:
                rows.append(text)
                seen.add(text)
    return rows


def _provider_has_explicit_liquidity(provider_key: str) -> bool:
    capability = hunt.PROVIDER_CAPABILITIES.get(str(provider_key).strip())
    return bool(capability and capability.liquidity_confidence in {"explicit", "estimated"})


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


def _timeout_scan_row(
    sport: str,
    providers: Sequence[str],
    *,
    all_markets: bool,
    timeout_seconds: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "sport": sport,
        "providers": list(providers),
        "api_bookmakers": [],
        "all_markets": bool(all_markets),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "success": False,
        "partial": True,
        "timed_out": True,
        "error": f"scan timed out after {timeout_seconds:g}s",
        "arbitrage_count": 0,
        "positive_candidates": 0,
        "actionable_arbitrage_count": 0,
        "top_candidate": None,
        "actionable_arbitrage": [],
        "top_arbitrage": [],
        "middle_count": 0,
        "positive_middle_count": 0,
        "actionable_middle_count": 0,
        "top_middles": [],
        "actionable_middles": [],
        "plus_ev_count": 0,
        "top_plus_ev": [],
        "provider_capabilities": [],
        "scan_diagnostics": {
            "reason_code": "scan_timeout",
            "raw_provider_events": 0,
            "providers_with_events": 0,
            "overlap_clusters": 0,
            "total_merge_hits": 0,
        },
        "custom_providers": {},
        "sport_errors": [{"sport": sport, "error": f"scan_timeout:{timeout_seconds:g}s"}],
    }


def _run_scan_once_with_timeout(
    sport: str,
    providers: Sequence[str],
    *,
    timeout_seconds: float,
    **kwargs,
) -> dict[str, Any]:
    timeout = max(0.0, float(timeout_seconds or 0.0))
    if timeout <= 0:
        return hunt._scan_once(sport, providers, **kwargs)
    started = time.monotonic()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(hunt._scan_once, sport, providers, **kwargs)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        return _timeout_scan_row(
            sport,
            providers,
            all_markets=bool(kwargs.get("all_markets")),
            timeout_seconds=timeout,
            elapsed_seconds=time.monotonic() - started,
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _bookmaker_key(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "bookmaker.xyz": "bookmaker_xyz",
        "bookmaker xyz": "bookmaker_xyz",
        "artlinebet": "artline",
        "artline bet": "artline",
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
    return (
        item.get("sport"),
        _canonical_event_key(item.get("event")),
        item.get("market"),
        item.get("middle_zone"),
    )


def _sport_family_key(sport: object) -> str:
    text = str(sport or "").strip().lower()
    if text in {"tennis_atp", "tennis_wta"}:
        return "tennis"
    return text


def _arb_key(item: dict[str, Any]) -> tuple[Any, ...]:
    books = item.get("books") if isinstance(item.get("books"), list) else []
    return (
        _sport_family_key(item.get("sport")),
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


def _stake_by_outcome(stakes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in stakes.get("breakdown") or []:
        if not isinstance(item, dict):
            continue
        outcome = str(item.get("outcome") or "").strip()
        if outcome:
            rows[outcome] = item
    return rows


def _middle_side_stakes(item: dict[str, Any]) -> list[float | None]:
    stake = item.get("stake") if isinstance(item.get("stake"), dict) else {}
    rows: list[float | None] = []
    for side_key in ("side_a", "side_b"):
        side = stake.get(side_key) if isinstance(stake.get(side_key), dict) else {}
        rows.append(hunt._safe_float(side.get("stake")))
    if all(value is not None and value > 0 for value in rows):
        return rows

    total_stake = hunt._safe_float(stake.get("total")) or 0.0
    books = [book for book in item.get("books") or [] if isinstance(book, dict)]
    if len(books) < 2 or total_stake <= 0:
        return rows
    odds_a = hunt._safe_float(books[0].get("effective_price") or books[0].get("price"))
    odds_b = hunt._safe_float(books[1].get("effective_price") or books[1].get("price"))
    if odds_a is None or odds_b is None or odds_a <= 1 or odds_b <= 1:
        return rows
    profit_a = odds_a - 1
    profit_b = odds_b - 1
    denominator = profit_a + profit_b
    if denominator <= 0:
        return rows
    side_a_stake = round(total_stake * profit_b / denominator, 2)
    return [side_a_stake, round(total_stake - side_a_stake, 2)]


def _book_execution_blockers(
    book: dict[str, Any],
    *,
    min_executable_stake: float,
    stake: float | None,
) -> list[str]:
    blockers: set[str] = set()
    bookmaker_key = _bookmaker_key(book.get("bookmaker") or book.get("bookmaker_key"))
    capability = hunt.PROVIDER_CAPABILITIES.get(bookmaker_key)
    if bookmaker_key not in EXECUTION_ADAPTER_PROVIDERS:
        blockers.add(f"{bookmaker_key}_execution_adapter_missing")
    max_stake = hunt._safe_float(book.get("max_stake"))
    if capability and capability.liquidity_confidence == "quote_only" and max_stake is None:
        blockers.add(f"{bookmaker_key}_quote_only")
    diagnostics = book.get("execution_diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("reason") == "max_bet_below_min_bet":
        blockers.add(f"{bookmaker_key}_max_bet_below_min_bet")
    if bookmaker_key == "artline":
        if isinstance(diagnostics, dict) and diagnostics.get("executable") is False:
            blockers.add("artline_not_executable")
        if not book.get("book_event_url"):
            blockers.add("artline_missing_event_url")
    if not book.get("quote_updated_at"):
        blockers.add("missing_quote_time")
    if max_stake is None:
        blockers.add("missing_liquidity")
    elif stake is not None and max_stake + 1e-9 < stake:
        blockers.add(f"{bookmaker_key}_stake_exceeds_max")
    if min_executable_stake > 0 and stake is not None and stake < min_executable_stake:
        blockers.add("stake_below_minimum")
    return sorted(blockers)


def _execution_identifier_blockers(bookmaker_key: str, book: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if bookmaker_key == "sx_bet":
        if not book.get("market_hash"):
            blockers.append("sx_bet_missing_market_hash")
        if book.get("outcome_index") is None:
            blockers.append("sx_bet_missing_outcome_index")
    elif bookmaker_key == "polymarket":
        if not (book.get("token_id") or book.get("asset_id")):
            blockers.append("polymarket_missing_token_id")
    elif bookmaker_key == "artline":
        if not book.get("book_event_url"):
            blockers.append("artline_missing_event_url")
    return blockers


def _artline_sport_slug(book: dict[str, Any]) -> str:
    url = str(book.get("book_event_url") or "").strip()
    match = re.search(r"/bookmaker/match/(?:prematch|live)/([^/?#]+)/", url)
    if match:
        return match.group(1)
    sport = str(book.get("sport") or "").strip().lower()
    if sport.startswith("tennis"):
        return "tennis"
    if sport.startswith("icehockey"):
        return "hockey"
    if sport.startswith("soccer"):
        return "football"
    if sport.startswith("basketball"):
        return "basketball"
    if sport.startswith("baseball"):
        return "baseball"
    return sport.replace("_", "-")


def _artline_web_max_bet_preflight(book: dict[str, Any], *, stake: float | None) -> dict[str, Any] | None:
    if artline_provider is None:
        return None
    selection_id = book.get("selection_id") or book.get("selectionId")
    game_id = book.get("book_event_id") or book.get("event_id") or book.get("eventId")
    if not game_id:
        url = str(book.get("book_event_url") or "").strip()
        match = re.search(r"/([^/?#]+)(?:[?#].*)?$", url)
        if match:
            game_id = match.group(1)
    sport_slug = _artline_sport_slug(book)
    live_state = book.get("live_state") if isinstance(book.get("live_state"), dict) else {}
    is_live = bool(live_state.get("is_live")) if isinstance(live_state, dict) else False
    return artline_provider.preflight_web_max_bet(
        sport=sport_slug,
        game_id=game_id,
        selection_id=selection_id,
        is_live=is_live,
        stake=stake,
        event_url=book.get("book_event_url"),
    )


def _paper_trade_preflight(
    *,
    submit_blockers: Sequence[str],
    legs: Sequence[dict[str, Any]],
    positive_metric: bool,
) -> dict[str, Any]:
    manual_web_required = any(
        isinstance(leg.get("draft_order"), dict)
        and (
            leg["draft_order"].get("adapter") in {"manual_artline"}
            or leg["draft_order"].get("order_type") == "manual_web"
        )
        for leg in legs
        if isinstance(leg, dict)
    )
    missing_draft_orders = [
        str(leg.get("bookmaker_key") or leg.get("bookmaker") or "unknown")
        for leg in legs
        if isinstance(leg, dict) and not isinstance(leg.get("draft_order"), dict)
    ]
    liquidity_blocked_legs = [
        leg
        for leg in legs
        if isinstance(leg, dict)
        and "missing_liquidity" in {str(blocker) for blocker in leg.get("blockers") or []}
    ]
    missing_liquidity_is_manual_only = bool(liquidity_blocked_legs) and all(
        isinstance(leg.get("draft_order"), dict)
        and (
            leg["draft_order"].get("adapter") in {"manual_artline"}
            or leg["draft_order"].get("order_type") == "manual_web"
        )
        for leg in liquidity_blocked_legs
    )
    hard_blockers = []
    for blocker in submit_blockers:
        text = str(blocker)
        if not text or text in PAPER_TRADE_SOFT_BLOCKERS:
            continue
        if text == "missing_liquidity" and missing_liquidity_is_manual_only:
            continue
        hard_blockers.append(text)
    if missing_draft_orders:
        hard_blockers.append("missing_draft_order")
    if not positive_metric:
        hard_blockers.append("non_positive_metric")
    hard_blockers = sorted(set(hard_blockers))
    return {
        "paper_trade_ready": not hard_blockers,
        "manual_web_required": manual_web_required,
        "paper_trade_blockers": hard_blockers,
    }


def _draft_order_for_leg(bookmaker_key: str, book: dict[str, Any], *, stake: float | None) -> dict[str, Any] | None:
    if stake is None or stake <= 0:
        return None
    limit_odds = hunt._safe_float(book.get("price"))
    if limit_odds is None or limit_odds <= 1:
        return None
    if bookmaker_key == "sx_bet":
        market_hash = book.get("market_hash")
        outcome_index = book.get("outcome_index")
        if not market_hash or outcome_index is None:
            return None
        return {
            "adapter": "sx_bet",
            "order_type": "limit",
            "market_hash": market_hash,
            "outcome_index": outcome_index,
            "side": "buy",
            "size": round(float(stake), 2),
            "limit_odds": limit_odds,
        }
    if bookmaker_key == "polymarket":
        token_id = book.get("token_id") or book.get("asset_id")
        if not token_id:
            return None
        return {
            "adapter": "polymarket",
            "order_type": "limit",
            "token_id": token_id,
            "asset_id": book.get("asset_id") or token_id,
            "side": "buy",
            "size": round(float(stake), 2),
            "limit_odds": limit_odds,
        }
    if bookmaker_key == "artline":
        event_url = book.get("book_event_url")
        if not event_url:
            return None
        web_preflight = _artline_web_max_bet_preflight(book, stake=stake)
        return {
            "adapter": "manual_artline",
            "order_type": "manual_web",
            "event_url": event_url,
            "event_id": book.get("book_event_id") or book.get("event_id") or book.get("eventId"),
            "selection_id": book.get("selection_id") or book.get("selectionId"),
            "provider_event_name": book.get("provider_event_name") or book.get("providerEventName"),
            "outcome": book.get("outcome"),
            "side": "buy",
            "size": round(float(stake), 2),
            "limit_odds": limit_odds,
            "web_max_bet_preflight": web_preflight,
        }
    return None


def _execution_wallet_ready(providers: Sequence[str]) -> bool:
    required, missing = _execution_wallet_env_status(providers)
    return bool(required) and not missing


def _execution_wallet_env_status(providers: Sequence[str]) -> tuple[list[str], list[str]]:
    required_names: list[str] = []
    for provider in providers:
        provider_key = _bookmaker_key(provider)
        required_names.extend(EXECUTION_READY_ENV_BY_PROVIDER.get(provider_key, ()))
    required = _normalize_list(required_names)
    missing = [name for name in required if not os.getenv(name, "").strip()]
    return sorted(required), sorted(missing)


def _build_execution_ticket(
    item: dict[str, Any],
    *,
    source: str,
    min_executable_stake: float = 0.0,
    max_quote_skew_seconds: int = 120,
    wallet_ready: bool | None = None,
    slippage_bps: int = 0,
) -> dict[str, Any]:
    stakes = item.get("stakes") if isinstance(item.get("stakes"), dict) else {}
    stake_rows = _stake_by_outcome(stakes)
    total_stake = hunt._safe_float(stakes.get("total")) or 0.0
    blockers = set(_blocked_reasons(item, min_executable_stake=min_executable_stake))
    quote_skew = _quote_time_skew_seconds(item)
    max_quote_skew = max(0, int(max_quote_skew_seconds))
    if quote_skew is not None and max_quote_skew > 0 and quote_skew > max_quote_skew:
        blockers.add("quote_time_skew")
    fee_values: list[float] = []
    ticket_providers: list[str] = []
    legs: list[dict[str, Any]] = []
    for book in item.get("books") or []:
        if not isinstance(book, dict):
            continue
        outcome = str(book.get("outcome") or "").strip()
        stake_row = stake_rows.get(outcome, {})
        stake = hunt._safe_float(stake_row.get("stake"))
        bookmaker_key = _bookmaker_key(book.get("bookmaker") or book.get("bookmaker_key"))
        if bookmaker_key in EXECUTION_ADAPTER_PROVIDERS:
            ticket_providers.append(bookmaker_key)
        leg_blockers = _book_execution_blockers(
            book,
            min_executable_stake=min_executable_stake,
            stake=stake,
        )
        leg_blockers = sorted(set([*leg_blockers, *_execution_identifier_blockers(bookmaker_key, book)]))
        draft_order = _draft_order_for_leg(bookmaker_key, book, stake=stake)
        fee_rate = hunt._safe_float(book.get("fee_rate"))
        if fee_rate is not None:
            fee_values.append(fee_rate)
        blockers.update(leg_blockers)
        legs.append(
            {
                "bookmaker": book.get("bookmaker"),
                "bookmaker_key": bookmaker_key,
                "outcome": outcome or None,
                "market": item.get("market"),
                "stake": stake,
                "limit_price": book.get("price"),
                "effective_price": book.get("effective_price"),
                "max_stake": book.get("max_stake"),
                "quote_updated_at": book.get("quote_updated_at"),
                "quote_source": book.get("quote_source"),
                "fee_rate": fee_rate,
                "book_event_url": book.get("book_event_url"),
                "execution_diagnostics": book.get("execution_diagnostics"),
                "market_hash": book.get("market_hash"),
                "token_id": book.get("token_id"),
                "asset_id": book.get("asset_id"),
                "condition_id": book.get("condition_id"),
                "outcome_index": book.get("outcome_index"),
                "selection_id": book.get("selection_id") or book.get("selectionId"),
                "provider_event_name": book.get("provider_event_name") or book.get("providerEventName"),
                "execution_supported": bookmaker_key in EXECUTION_ADAPTER_PROVIDERS,
                "draft_order": draft_order,
                "blockers": leg_blockers,
            }
        )

    roi = hunt._safe_float(item.get("roi_percent"))
    positive_metric = roi is not None and roi > 0
    if not positive_metric:
        blockers.add("non_positive_roi")
    quality = item.get("execution_quality") if isinstance(item.get("execution_quality"), dict) else {}
    if str(quality.get("status") or "").strip().lower() != "high" or quality.get("flags"):
        blockers.add("execution_quality_not_high")
    if bool(stakes.get("limited_by_max_stake")):
        blockers.add("limited_by_liquidity")
    if total_stake <= 0:
        blockers.add("zero_total_stake")
    required_env, missing_env = _execution_wallet_env_status(ticket_providers)
    resolved_wallet_ready = bool(wallet_ready) if wallet_ready is not None else bool(required_env) and not missing_env
    if wallet_ready is True:
        missing_env = []
    if not resolved_wallet_ready:
        blockers.add("wallet_not_ready")

    submit_blockers = sorted(blockers)
    paper_preflight = _paper_trade_preflight(
        submit_blockers=submit_blockers,
        legs=legs,
        positive_metric=positive_metric,
    )
    slippage = max(0, int(slippage_bps))
    preflight = {
        "wallet_ready": resolved_wallet_ready,
        "wallet_required_env": required_env,
        "wallet_missing_env": missing_env,
        "quote_time_skew_seconds": quote_skew,
        "max_quote_skew_seconds": max_quote_skew,
        "slippage_bps": slippage,
        "fee_check": "present" if fee_values else "missing",
    }
    status = "ready" if not submit_blockers else "blocked"
    if status == "ready" and paper_preflight["manual_web_required"]:
        status = "manual_ready"
    return {
        "source": source,
        "dry_run": True,
        "can_submit_live": False,
        "status": status,
        **paper_preflight,
        "event": item.get("event"),
        "sport": item.get("sport"),
        "market": item.get("market"),
        "roi_percent": item.get("roi_percent"),
        "profit": item.get("profit"),
        "total_stake": round(total_stake, 2),
        "requested_total": stakes.get("requested_total"),
        "preflight": preflight,
        "submit_blockers": submit_blockers,
        "legs": legs,
    }


def _build_middle_execution_ticket(
    item: dict[str, Any],
    *,
    source: str,
    min_executable_stake: float = 0.0,
    max_quote_skew_seconds: int = 120,
    wallet_ready: bool | None = None,
    slippage_bps: int = 0,
) -> dict[str, Any]:
    stake = item.get("stake") if isinstance(item.get("stake"), dict) else {}
    side_stakes = _middle_side_stakes(item)
    total_stake = hunt._safe_float(stake.get("total")) or sum(
        value for value in side_stakes if value is not None
    )
    blockers = {str(reason).strip() for reason in item.get("blocked_reasons") or [] if str(reason).strip()}
    blockers.update(_risk_flags_from_item(item))
    quote_skew = _quote_time_skew_seconds(item)
    max_quote_skew = max(0, int(max_quote_skew_seconds))
    if quote_skew is not None and max_quote_skew > 0 and quote_skew > max_quote_skew:
        blockers.add("quote_time_skew")

    fee_values: list[float] = []
    ticket_providers: list[str] = []
    legs: list[dict[str, Any]] = []
    for index, book in enumerate(item.get("books") or []):
        if not isinstance(book, dict):
            continue
        stake_amount = side_stakes[index] if index < len(side_stakes) else None
        bookmaker_key = _bookmaker_key(book.get("bookmaker") or book.get("bookmaker_key"))
        if bookmaker_key in EXECUTION_ADAPTER_PROVIDERS:
            ticket_providers.append(bookmaker_key)
        leg_blockers = _book_execution_blockers(
            book,
            min_executable_stake=min_executable_stake,
            stake=stake_amount,
        )
        leg_blockers = sorted(set([*leg_blockers, *_execution_identifier_blockers(bookmaker_key, book)]))
        draft_order = _draft_order_for_leg(bookmaker_key, book, stake=stake_amount)
        fee_rate = hunt._safe_float(book.get("fee_rate"))
        if fee_rate is not None:
            fee_values.append(fee_rate)
        blockers.update(leg_blockers)
        legs.append(
            {
                "bookmaker": book.get("bookmaker"),
                "bookmaker_key": bookmaker_key,
                "market": item.get("market"),
                "middle_zone": item.get("middle_zone"),
                "line": book.get("line"),
                "stake": stake_amount,
                "limit_price": book.get("price"),
                "effective_price": book.get("effective_price"),
                "max_stake": book.get("max_stake"),
                "quote_updated_at": book.get("quote_updated_at"),
                "quote_source": book.get("quote_source"),
                "fee_rate": fee_rate,
                "book_event_url": book.get("book_event_url"),
                "execution_diagnostics": book.get("execution_diagnostics"),
                "market_hash": book.get("market_hash"),
                "token_id": book.get("token_id"),
                "asset_id": book.get("asset_id"),
                "condition_id": book.get("condition_id"),
                "outcome_index": book.get("outcome_index"),
                "selection_id": book.get("selection_id") or book.get("selectionId"),
                "provider_event_name": book.get("provider_event_name") or book.get("providerEventName"),
                "execution_supported": bookmaker_key in EXECUTION_ADAPTER_PROVIDERS,
                "draft_order": draft_order,
                "blockers": leg_blockers,
            }
        )

    ev = hunt._safe_float(item.get("ev_percent"))
    positive_metric = ev is not None and ev > 0
    if not positive_metric:
        blockers.add("non_positive_ev")
    if bool(stake.get("limited_by_max_stake")):
        blockers.add("limited_by_liquidity")
    if total_stake <= 0:
        blockers.add("zero_total_stake")
    required_env, missing_env = _execution_wallet_env_status(ticket_providers)
    resolved_wallet_ready = bool(wallet_ready) if wallet_ready is not None else bool(required_env) and not missing_env
    if wallet_ready is True:
        missing_env = []
    if not resolved_wallet_ready:
        blockers.add("wallet_not_ready")

    submit_blockers = sorted(blockers)
    paper_preflight = _paper_trade_preflight(
        submit_blockers=submit_blockers,
        legs=legs,
        positive_metric=positive_metric,
    )
    slippage = max(0, int(slippage_bps))
    status = "ready" if not submit_blockers else "blocked"
    if status == "ready" and paper_preflight["manual_web_required"]:
        status = "manual_ready"
    return {
        "source": source,
        "execution_type": "middle",
        "dry_run": True,
        "can_submit_live": False,
        "status": status,
        **paper_preflight,
        "event": item.get("event"),
        "sport": item.get("sport"),
        "market": item.get("market"),
        "middle_zone": item.get("middle_zone"),
        "ev_percent": item.get("ev_percent"),
        "probability_percent": item.get("probability_percent"),
        "total_stake": round(total_stake, 2),
        "requested_total": stake.get("requested_total"),
        "preflight": {
            "wallet_ready": resolved_wallet_ready,
            "wallet_required_env": required_env,
            "wallet_missing_env": missing_env,
            "quote_time_skew_seconds": quote_skew,
            "max_quote_skew_seconds": max_quote_skew,
            "slippage_bps": slippage,
            "fee_check": "present" if fee_values else "missing",
        },
        "submit_blockers": submit_blockers,
        "legs": legs,
    }


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


def _scan_diagnostic_summary(scans: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    scans_with_source_events = 0
    scans_without_source_events = 0
    scans_with_cross_provider_matches = 0
    scans_without_cross_provider_matches = 0
    for scan in scans:
        diagnostics = scan.get("scan_diagnostics") if isinstance(scan, dict) else {}
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        reason = str(diagnostics.get("reason_code") or "unknown").strip() or "unknown"
        reason_counts[reason] += 1
        raw_events = hunt._safe_float(diagnostics.get("raw_provider_events")) or 0.0
        providers_with_events = hunt._safe_float(diagnostics.get("providers_with_events")) or 0.0
        if raw_events > 0 and providers_with_events > 0:
            scans_with_source_events += 1
            overlap_clusters = hunt._safe_float(diagnostics.get("overlap_clusters")) or 0.0
            merge_hits = hunt._safe_float(diagnostics.get("total_merge_hits")) or 0.0
            if overlap_clusters > 0 or merge_hits > 0:
                scans_with_cross_provider_matches += 1
            else:
                scans_without_cross_provider_matches += 1
        else:
            scans_without_source_events += 1
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "scans_with_source_events": scans_with_source_events,
        "scans_without_source_events": scans_without_source_events,
        "scans_with_cross_provider_matches": scans_with_cross_provider_matches,
        "scans_without_cross_provider_matches": scans_without_cross_provider_matches,
    }


def _cross_provider_match_summary(scans: Sequence[dict[str, Any]]) -> dict[str, Any]:
    provider_event_counts: Counter[str] = Counter()
    provider_cluster_presence: Counter[str] = Counter()
    single_provider_cluster_counts: Counter[str] = Counter()
    single_provider_reason_counts: Counter[str] = Counter()
    pair_overlap_clusters: Counter[str] = Counter()
    report_paths: list[str] = []
    total_raw_records = 0
    total_match_clusters = 0
    overlap_clusters = 0
    for scan in scans:
        if not isinstance(scan, dict):
            continue
        path = str(scan.get("cross_provider_match_report_path") or "").strip()
        if path and path not in report_paths:
            report_paths.append(path)
        summary = scan.get("cross_provider_match_report_summary")
        if not isinstance(summary, dict):
            continue
        total_raw_records += int(summary.get("total_raw_records") or 0)
        total_match_clusters += int(summary.get("total_match_clusters") or 0)
        overlap_clusters += int(summary.get("overlap_clusters") or 0)
        provider_event_counts.update(
            {
                str(key): int(value or 0)
                for key, value in (summary.get("provider_event_counts") or {}).items()
            }
        )
        provider_cluster_presence.update(
            {
                str(key): int(value or 0)
                for key, value in (summary.get("provider_cluster_presence") or {}).items()
            }
        )
        single_provider_cluster_counts.update(
            {
                str(key): int(value or 0)
                for key, value in (summary.get("single_provider_cluster_counts") or {}).items()
            }
        )
        single_provider_reason_counts.update(
            {
                str(key): int(value or 0)
                for key, value in (summary.get("single_provider_reason_counts") or {}).items()
            }
        )
        pair_overlap_clusters.update(
            {
                str(key): int(value or 0)
                for key, value in (summary.get("pair_overlap_clusters") or {}).items()
            }
        )
    return {
        "total_raw_records": total_raw_records,
        "total_match_clusters": total_match_clusters,
        "overlap_clusters": overlap_clusters,
        "provider_event_counts": dict(sorted(provider_event_counts.items())),
        "provider_cluster_presence": dict(sorted(provider_cluster_presence.items())),
        "single_provider_cluster_counts": dict(sorted(single_provider_cluster_counts.items())),
        "single_provider_reason_counts": dict(sorted(single_provider_reason_counts.items())),
        "pair_overlap_clusters": dict(sorted(pair_overlap_clusters.items())),
        "report_paths": report_paths[:10],
    }


def _provider_error_label(provider: object, error_code: object, status_code: object, error: object) -> str:
    provider_key = _bookmaker_key(provider) if provider else "unknown_provider"
    code = str(error_code or "").strip()
    if not code and status_code not in (None, ""):
        code = f"http_{status_code}"
    if not code:
        text = str(error or "").strip()
        code = text[:80] if text else "provider_error"
    return f"{provider_key}:{code}"


def _provider_error_counts(scans: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for scan in scans:
        if not isinstance(scan, dict):
            continue
        seen: set[tuple[str, str]] = set()
        for item in scan.get("sport_errors") or []:
            if not isinstance(item, dict):
                continue
            label = _provider_error_label(
                item.get("provider_key") or item.get("provider"),
                item.get("error_code"),
                item.get("status_code"),
                item.get("error"),
            )
            key = (str(item.get("sport_key") or scan.get("sport") or ""), label)
            if key not in seen:
                seen.add(key)
                counts[label] += 1
        custom_providers = scan.get("custom_providers") if isinstance(scan.get("custom_providers"), dict) else {}
        for provider_key, provider_state in custom_providers.items():
            if not isinstance(provider_state, dict):
                continue
            for item in provider_state.get("sports") or []:
                if not isinstance(item, dict) or not item.get("error"):
                    continue
                label = _provider_error_label(
                    provider_key,
                    item.get("error_code"),
                    item.get("status_code"),
                    item.get("error"),
                )
                key = (str(item.get("sport_key") or scan.get("sport") or ""), label)
                if key not in seen:
                    seen.add(key)
                    counts[label] += 1
    return dict(sorted(counts.items()))


def _rank_count_items(counts: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    rows = [
        {"reason": str(reason), "count": int(count or 0)}
        for reason, count in (counts or {}).items()
        if str(reason).strip() and int(count or 0) > 0
    ]
    rows.sort(key=lambda item: (-item["count"], item["reason"]))
    return rows[: max(0, int(limit))]


def _opportunity_funnel_summary(
    *,
    scan_count: int,
    scan_diagnostics: dict[str, Any],
    actionable_arbitrage_count: int,
    actionable_middle_count: int,
    plus_ev_count: int,
    model_only_arbitrage_count: int,
    model_only_middle_count: int,
    execution_ready_ticket_count: int,
    middle_execution_ready_ticket_count: int,
    paper_trade_ready_ticket_count: int = 0,
    paper_trade_manual_ticket_count: int = 0,
    blocked_reason_counts: dict[str, Any],
    execution_risk_counts: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = scan_diagnostics if isinstance(scan_diagnostics, dict) else {}
    ready_ticket_count = int(execution_ready_ticket_count or 0) + int(middle_execution_ready_ticket_count or 0)
    actionable_count = int(actionable_arbitrage_count or 0) + int(actionable_middle_count or 0)
    model_only_count = int(model_only_arbitrage_count or 0) + int(model_only_middle_count or 0)
    if ready_ticket_count > 0:
        conclusion = "execution_ready_opportunity"
    elif int(paper_trade_ready_ticket_count or 0) > 0:
        conclusion = "paper_trade_ready_opportunity"
    elif actionable_count > 0:
        conclusion = "actionable_but_execution_blocked"
    else:
        conclusion = "no_execution_ready_opportunity"
    return {
        "conclusion": conclusion,
        "scan_count": int(scan_count or 0),
        "scans_with_source_events": int(diagnostics.get("scans_with_source_events") or 0),
        "scans_without_source_events": int(diagnostics.get("scans_without_source_events") or 0),
        "scans_with_cross_provider_matches": int(diagnostics.get("scans_with_cross_provider_matches") or 0),
        "scans_without_cross_provider_matches": int(diagnostics.get("scans_without_cross_provider_matches") or 0),
        "actionable_count": actionable_count,
        "ready_ticket_count": ready_ticket_count,
        "paper_trade_ready_ticket_count": int(paper_trade_ready_ticket_count or 0),
        "paper_trade_manual_ticket_count": int(paper_trade_manual_ticket_count or 0),
        "model_only_count": model_only_count,
        "plus_ev_count": int(plus_ev_count or 0),
        "top_scan_reasons": _rank_count_items(diagnostics.get("reason_counts") or {}),
        "top_blockers": _rank_count_items(blocked_reason_counts),
        "execution_risks": _rank_count_items(execution_risk_counts),
    }


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
        f"- execution_ticket_count: {summary.get('execution_ticket_count', 0)}",
        f"- execution_ready_ticket_count: {summary.get('execution_ready_ticket_count', 0)}",
        f"- middle_execution_ticket_count: {summary.get('middle_execution_ticket_count', 0)}",
        f"- middle_execution_ready_ticket_count: {summary.get('middle_execution_ready_ticket_count', 0)}",
        f"- paper_trade_ready_ticket_count: {summary.get('paper_trade_ready_ticket_count', 0)}",
        f"- paper_trade_manual_ticket_count: {summary.get('paper_trade_manual_ticket_count', 0)}",
        f"- actionable_middle_count: {summary.get('actionable_middle_count', 0)}",
        f"- execution_risky_middle_count: {summary.get('execution_risky_middle_count', 0)}",
        f"- model_only_middle_count: {summary.get('model_only_middle_count', 0)}",
        f"- plus_ev_count: {summary.get('plus_ev_count', 0)}",
        "",
        "## Scan Diagnostics",
        *_format_counts(
            (summary.get("scan_diagnostics") or {}).get("reason_counts")
            if isinstance(summary.get("scan_diagnostics"), dict)
            else {}
        ),
        *(
            [
                f"- scans_with_source_events: {summary.get('scan_diagnostics', {}).get('scans_with_source_events', 0)}",
                f"- scans_without_source_events: {summary.get('scan_diagnostics', {}).get('scans_without_source_events', 0)}",
                f"- scans_with_cross_provider_matches: {summary.get('scan_diagnostics', {}).get('scans_with_cross_provider_matches', 0)}",
                f"- scans_without_cross_provider_matches: {summary.get('scan_diagnostics', {}).get('scans_without_cross_provider_matches', 0)}",
            ]
            if isinstance(summary.get("scan_diagnostics"), dict)
            else []
        ),
        "",
        "## Opportunity Funnel",
    ]
    funnel = summary.get("opportunity_funnel") if isinstance(summary.get("opportunity_funnel"), dict) else {}
    if funnel:
        lines.extend(
            [
                f"- conclusion: {funnel.get('conclusion', '-')}",
                f"- scans_with_source_events: {funnel.get('scans_with_source_events', 0)}",
                f"- scans_with_cross_provider_matches: {funnel.get('scans_with_cross_provider_matches', 0)}",
                f"- actionable_count: {funnel.get('actionable_count', 0)}",
                f"- ready_ticket_count: {funnel.get('ready_ticket_count', 0)}",
                f"- paper_trade_ready_ticket_count: {funnel.get('paper_trade_ready_ticket_count', 0)}",
                f"- paper_trade_manual_ticket_count: {funnel.get('paper_trade_manual_ticket_count', 0)}",
                f"- model_only_count: {funnel.get('model_only_count', 0)}",
            ]
        )
        for label, key in (
            ("top_scan_reason", "top_scan_reasons"),
            ("top_blocker", "top_blockers"),
            ("execution_risk", "execution_risks"),
        ):
            for item in funnel.get(key) or []:
                lines.append(f"- {label}: {item.get('reason')} x{item.get('count')}")
    else:
        lines.append("- none")
    lines.extend(
        [
        "",
        "## Cross-Provider Match Summary",
        *_format_counts(
            (summary.get("cross_provider_match_summary") or {}).get("provider_event_counts")
            if isinstance(summary.get("cross_provider_match_summary"), dict)
            else {}
        ),
        *(
            [
                f"- total_raw_records: {summary.get('cross_provider_match_summary', {}).get('total_raw_records', 0)}",
                f"- total_match_clusters: {summary.get('cross_provider_match_summary', {}).get('total_match_clusters', 0)}",
                f"- overlap_clusters: {summary.get('cross_provider_match_summary', {}).get('overlap_clusters', 0)}",
            ]
            if isinstance(summary.get("cross_provider_match_summary"), dict)
            else []
        ),
        *(
            [
                f"- single_provider_reason: {item.get('reason')} x{item.get('count')}"
                for item in _rank_count_items(
                    summary.get("cross_provider_match_summary", {}).get("single_provider_reason_counts") or {}
                )
            ]
            if isinstance(summary.get("cross_provider_match_summary"), dict)
            else []
        ),
        *(
            [
                f"- pair_overlap: {key}: {value}"
                for key, value in sorted(
                    (
                        summary.get("cross_provider_match_summary", {})
                        .get("pair_overlap_clusters", {})
                        .items()
                    )
                )
            ]
            if isinstance(summary.get("cross_provider_match_summary"), dict)
            else []
        ),
        *(
            [
                f"- report_path: {path}"
                for path in (
                    summary.get("cross_provider_match_summary", {}).get("report_paths") or []
                )[:3]
            ]
            if isinstance(summary.get("cross_provider_match_summary"), dict)
            else []
        ),
        "",
        "## Provider Error Counts",
        *_format_counts(summary.get("provider_error_counts") or {}),
        "",
        "## Execution Risk Counts",
        *_format_counts(summary.get("execution_risk_counts") or {}),
        "",
        "## Blocked Reason Counts",
        *_format_counts(summary.get("blocked_reason_counts") or {}),
        "",
        "## Top Actionable Middles",
        ]
    )
    actionable_middles = [item for item in payload.get("actionable_middles") or [] if isinstance(item, dict)]
    lines.extend(_format_opportunity_row(item) for item in actionable_middles[:5])
    if not actionable_middles:
        lines.append("- none")

    lines.extend(["", "## Execution Tickets"])
    tickets = [item for item in payload.get("execution_tickets") or [] if isinstance(item, dict)]
    for ticket in tickets[:5]:
        blockers = ticket.get("submit_blockers") or []
        lines.append(
            "- "
            + " | ".join(
                [
                    str(ticket.get("event") or "-"),
                    str(ticket.get("market") or "-"),
                    f"status={ticket.get('status', '-')}",
                    f"paper_trade_ready={ticket.get('paper_trade_ready', False)}",
                    f"manual_web_required={ticket.get('manual_web_required', False)}",
                    f"roi={ticket.get('roi_percent', '-')}",
                    "blockers=" + (",".join(str(item) for item in blockers) if blockers else "none"),
                ]
            )
        )
    if not tickets:
        lines.append("- none")

    lines.extend(["", "## Middle Execution Tickets"])
    middle_tickets = [item for item in payload.get("middle_execution_tickets") or [] if isinstance(item, dict)]
    for ticket in middle_tickets[:5]:
        blockers = ticket.get("submit_blockers") or []
        lines.append(
            "- "
            + " | ".join(
                [
                    str(ticket.get("event") or "-"),
                    str(ticket.get("market") or "-"),
                    str(ticket.get("middle_zone") or "-"),
                    f"status={ticket.get('status', '-')}",
                    f"paper_trade_ready={ticket.get('paper_trade_ready', False)}",
                    f"manual_web_required={ticket.get('manual_web_required', False)}",
                    f"ev={ticket.get('ev_percent', '-')}",
                    "blockers=" + (",".join(str(item) for item in blockers) if blockers else "none"),
                ]
            )
        )
    if not middle_tickets:
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


def _batch_report_stem(index: int, job_name: object, sport: object) -> str:
    raw = f"{index:03d}_{job_name}_{sport}"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw).strip("._")
    return safe or f"{index:03d}_scan"


def _extend_scan_outputs(
    row: dict[str, Any],
    *,
    job_name: str,
    candidates: list[dict[str, Any]],
    top_arbitrage: list[dict[str, Any]],
    actionable_arbitrage: list[dict[str, Any]],
    top_middles: list[dict[str, Any]],
    actionable_middles: list[dict[str, Any]],
    plus_ev: list[dict[str, Any]],
) -> None:
    top_candidate = row.get("top_candidate")
    if isinstance(top_candidate, dict):
        top_candidate["job_name"] = job_name
        candidates.append(top_candidate)
    for item in row.get("top_arbitrage") or []:
        if isinstance(item, dict):
            item["job_name"] = job_name
            top_arbitrage.append(item)
    for item in row.get("actionable_arbitrage") or []:
        if isinstance(item, dict):
            item["job_name"] = job_name
            actionable_arbitrage.append(item)
    for item in row.get("top_middles") or []:
        if isinstance(item, dict):
            item["job_name"] = job_name
            top_middles.append(item)
    for item in row.get("actionable_middles") or []:
        if isinstance(item, dict):
            item["job_name"] = job_name
            actionable_middles.append(item)
    for item in row.get("top_plus_ev") or []:
        if isinstance(item, dict):
            item["job_name"] = job_name
            plus_ev.append(item)


def _build_payload(
    *,
    jobs: Sequence[dict[str, Any]],
    scans: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    top_arbitrage: Sequence[dict[str, Any]],
    actionable_arbitrage: Sequence[dict[str, Any]],
    top_middles: Sequence[dict[str, Any]],
    actionable_middles: Sequence[dict[str, Any]],
    plus_ev: Sequence[dict[str, Any]],
    require_explicit_liquidity: bool,
    max_quote_skew_seconds: int,
    min_executable_stake: float,
    batch_reports: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates_list = _deduplicate_rows(candidates, key_factory=_arb_key, metric_key="roi_percent")
    top_arbitrage_list = _deduplicate_rows(top_arbitrage, key_factory=_arb_key, metric_key="roi_percent")
    actionable_arbitrage_list = _deduplicate_rows(
        actionable_arbitrage,
        key_factory=_arb_key,
        metric_key="roi_percent",
    )
    top_middles_list = _deduplicate_rows(top_middles, key_factory=_middle_key, metric_key="ev_percent")
    actionable_middles_list = _deduplicate_rows(
        actionable_middles,
        key_factory=_middle_key,
        metric_key="ev_percent",
    )
    actionable_middles_list, execution_risky_middles = _split_actionable_middles_by_quote_skew(
        actionable_middles_list,
        max_quote_skew_seconds=max(0, int(max_quote_skew_seconds)),
    )
    plus_ev_list = _deduplicate_rows(plus_ev, key_factory=_plus_ev_key, metric_key="edge_percent")
    minimum_stake = max(0.0, float(min_executable_stake))
    model_only_arbitrage = _annotate_blocked_reasons(
        _model_only_arbitrage(top_arbitrage_list),
        min_executable_stake=minimum_stake,
    )
    model_only_middles = _annotate_blocked_reasons(
        _model_only_middles(top_middles_list),
        min_executable_stake=minimum_stake,
    )
    max_quote_skew = max(0, int(max_quote_skew_seconds))
    execution_tickets = [
        _build_execution_ticket(
            item,
            source="actionable_arbitrage",
            min_executable_stake=minimum_stake,
            max_quote_skew_seconds=max_quote_skew,
        )
        for item in actionable_arbitrage_list
    ]
    execution_tickets.extend(
        _build_execution_ticket(
            item,
            source="model_only_arbitrage",
            min_executable_stake=minimum_stake,
            max_quote_skew_seconds=max_quote_skew,
        )
        for item in model_only_arbitrage
    )
    execution_ready_ticket_count = sum(1 for item in execution_tickets if item.get("status") == "ready")
    middle_execution_tickets = [
        _build_middle_execution_ticket(
            item,
            source="actionable_middle",
            min_executable_stake=minimum_stake,
            max_quote_skew_seconds=max_quote_skew,
        )
        for item in actionable_middles_list
    ]
    middle_execution_tickets.extend(
        _build_middle_execution_ticket(
            item,
            source="execution_risky_middle",
            min_executable_stake=minimum_stake,
            max_quote_skew_seconds=max_quote_skew,
        )
        for item in execution_risky_middles
    )
    middle_execution_tickets.extend(
        _build_middle_execution_ticket(
            item,
            source="model_only_middle",
            min_executable_stake=minimum_stake,
            max_quote_skew_seconds=max_quote_skew,
        )
        for item in model_only_middles
    )
    middle_execution_ready_ticket_count = sum(
        1 for item in middle_execution_tickets if item.get("status") == "ready"
    )
    all_execution_tickets = [*execution_tickets, *middle_execution_tickets]
    paper_trade_ready_ticket_count = sum(
        1 for item in all_execution_tickets if item.get("paper_trade_ready")
    )
    paper_trade_manual_ticket_count = sum(
        1 for item in all_execution_tickets if item.get("paper_trade_ready") and item.get("manual_web_required")
    )
    blocked_reason_counts = _blocked_reason_counts([*model_only_arbitrage, *model_only_middles])
    execution_risk_counts = _execution_risk_counts(execution_risky_middles)
    scan_diagnostics_summary = _scan_diagnostic_summary(scans)
    cross_provider_match_summary = _cross_provider_match_summary(scans)
    provider_error_counts = _provider_error_counts(scans)
    opportunity_funnel = _opportunity_funnel_summary(
        scan_count=len(scans),
        scan_diagnostics=scan_diagnostics_summary,
        actionable_arbitrage_count=len(actionable_arbitrage_list),
        actionable_middle_count=len(actionable_middles_list),
        plus_ev_count=len(plus_ev_list),
        model_only_arbitrage_count=len(model_only_arbitrage),
        model_only_middle_count=len(model_only_middles),
        execution_ready_ticket_count=execution_ready_ticket_count,
        middle_execution_ready_ticket_count=middle_execution_ready_ticket_count,
        paper_trade_ready_ticket_count=paper_trade_ready_ticket_count,
        paper_trade_manual_ticket_count=paper_trade_manual_ticket_count,
        blocked_reason_counts=blocked_reason_counts,
        execution_risk_counts=execution_risk_counts,
    )
    reports = list(batch_reports or [])
    return {
        "summary": {
            "scan_count": len(scans),
            "candidate_count": len(candidates_list),
            "actionable_arbitrage_count": len(actionable_arbitrage_list),
            "model_only_arbitrage_count": len(model_only_arbitrage),
            "execution_ticket_count": len(execution_tickets),
            "execution_ready_ticket_count": execution_ready_ticket_count,
            "middle_execution_ticket_count": len(middle_execution_tickets),
            "middle_execution_ready_ticket_count": middle_execution_ready_ticket_count,
            "paper_trade_ready_ticket_count": paper_trade_ready_ticket_count,
            "paper_trade_manual_ticket_count": paper_trade_manual_ticket_count,
            "actionable_middle_count": len(actionable_middles_list),
            "execution_risky_middle_count": len(execution_risky_middles),
            "model_only_middle_count": len(model_only_middles),
            "plus_ev_count": len(plus_ev_list),
            "blocked_reason_counts": blocked_reason_counts,
            "execution_risk_counts": execution_risk_counts,
            "provider_error_counts": provider_error_counts,
            "scan_diagnostics": scan_diagnostics_summary,
            "cross_provider_match_summary": cross_provider_match_summary,
            "opportunity_funnel": opportunity_funnel,
            "require_explicit_liquidity": bool(require_explicit_liquidity),
            "max_quote_skew_seconds": max_quote_skew,
            "min_executable_stake": minimum_stake,
            "batch_report_count": len(reports),
        },
        "jobs": list(jobs),
        "scans": list(scans),
        "batch_reports": reports,
        "candidates": candidates_list[:20],
        "actionable_arbitrage": actionable_arbitrage_list[:20],
        "model_only_arbitrage": model_only_arbitrage[:20],
        "execution_tickets": execution_tickets[:20],
        "middle_execution_tickets": middle_execution_tickets[:20],
        "actionable_middles": actionable_middles_list[:20],
        "execution_risky_middles": execution_risky_middles[:20],
        "model_only_middles": model_only_middles[:20],
        "top_arbitrage": top_arbitrage_list[:20],
        "top_middles": top_middles_list[:20],
        "top_plus_ev": plus_ev_list[:20],
        "provider_capabilities": hunt._provider_capability_summary(hunt.DEFAULT_PROVIDERS),
    }


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
    parser.add_argument(
        "--per-scan-timeout-seconds",
        type=float,
        default=0.0,
        help="Maximum seconds to allow each sport/provider job before recording a timeout row; 0 disables.",
    )
    parser.add_argument("--out", default=str(Path("data") / "provider_verification" / "dex_opportunities_latest.json"))
    parser.add_argument(
        "--summary-out",
        default=str(Path("data") / "provider_verification" / "latest_actionable_summary.md"),
    )
    parser.add_argument(
        "--batch-out-dir",
        default="",
        help="Optional directory for one JSON/Markdown report per completed scan row.",
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
    batch_reports: list[dict[str, Any]] = []
    batch_out_dir = Path(args.batch_out_dir) if str(args.batch_out_dir or "").strip() else None

    try:
        scan_index = 0
        for job in jobs:
            for sport in job["sports"]:
                scan_index += 1
                timeout_seconds = max(0.0, float(args.per_scan_timeout_seconds))
                try:
                    row = _run_scan_once_with_timeout(
                        sport,
                        job["providers"],
                        timeout_seconds=timeout_seconds,
                        api_key=str(args.api_key or "").strip(),
                        api_bookmakers=api_bookmakers,
                        all_markets=bool(job["all_markets"]),
                        stake=float(args.stake),
                        min_roi=float(args.min_roi),
                        allowed_quality=allowed_quality,
                    )
                except TimeoutError:
                    row = _timeout_scan_row(
                        sport,
                        job["providers"],
                        all_markets=bool(job["all_markets"]),
                        timeout_seconds=timeout_seconds,
                        elapsed_seconds=timeout_seconds,
                    )
                row["job_name"] = job["name"]
                scans.append(row)
                _extend_scan_outputs(
                    row,
                    job_name=job["name"],
                    candidates=candidates,
                    top_arbitrage=top_arbitrage,
                    actionable_arbitrage=actionable_arbitrage,
                    top_middles=top_middles,
                    actionable_middles=actionable_middles,
                    plus_ev=plus_ev,
                )
                if batch_out_dir is not None:
                    batch_stem = _batch_report_stem(scan_index, job["name"], sport)
                    batch_json_path = batch_out_dir / f"{batch_stem}.json"
                    batch_md_path = batch_out_dir / f"{batch_stem}.md"
                    batch_payload = _build_payload(
                        jobs=[job],
                        scans=[row],
                        candidates=[],
                        top_arbitrage=row.get("top_arbitrage") or [],
                        actionable_arbitrage=row.get("actionable_arbitrage") or [],
                        top_middles=row.get("top_middles") or [],
                        actionable_middles=row.get("actionable_middles") or [],
                        plus_ev=row.get("top_plus_ev") or [],
                        require_explicit_liquidity=bool(args.require_explicit_liquidity),
                        max_quote_skew_seconds=max(0, int(args.max_quote_skew_seconds)),
                        min_executable_stake=max(0.0, float(args.min_executable_stake)),
                    )
                    _write_json(batch_json_path, batch_payload)
                    _write_markdown_summary(batch_md_path, batch_payload, source_json=str(batch_json_path))
                    batch_reports.append(
                        {
                            "sport": sport,
                            "job_name": job["name"],
                            "json": str(batch_json_path),
                            "summary": str(batch_md_path),
                            "success": bool(row.get("success")),
                            "timed_out": bool(row.get("timed_out")),
                            "actionable_arbitrage_count": int(row.get("actionable_arbitrage_count") or 0),
                            "actionable_middle_count": int(row.get("actionable_middle_count") or 0),
                            "plus_ev_count": int(row.get("plus_ev_count") or 0),
                        }
                    )
                print(
                    f"{job['name']} sport={sport} providers={'+'.join(job['providers'])} "
                    f"all_markets={bool(job['all_markets'])} "
                    f"arb={row['arbitrage_count']} candidates={row['positive_candidates']} "
                    f"actionable_arbitrage={row['actionable_arbitrage_count']} "
                    f"middles={row['middle_count']} actionable_middles={row['actionable_middle_count']} "
                    f"plus_ev={row['plus_ev_count']} elapsed={row['elapsed_seconds']}s"
                )

        payload = _build_payload(
            jobs=jobs,
            scans=scans,
            candidates=candidates,
            top_arbitrage=top_arbitrage,
            actionable_arbitrage=actionable_arbitrage,
            top_middles=top_middles,
            actionable_middles=actionable_middles,
            plus_ev=plus_ev,
            require_explicit_liquidity=bool(args.require_explicit_liquidity),
            max_quote_skew_seconds=max(0, int(args.max_quote_skew_seconds)),
            min_executable_stake=max(0.0, float(args.min_executable_stake)),
            batch_reports=batch_reports,
        )
        out_path = Path(args.out)
        summary_path = Path(args.summary_out)
        _write_json(out_path, payload)
        _write_markdown_summary(summary_path, payload, source_json=str(out_path))

        summary = payload.get("summary") or {}
        ready_count = int(summary.get("execution_ready_ticket_count") or 0) + int(
            summary.get("middle_execution_ready_ticket_count") or 0
        )
        paper_ready_count = int(summary.get("paper_trade_ready_ticket_count") or 0)
        actionable_count = int(summary.get("actionable_arbitrage_count") or 0) + int(
            summary.get("actionable_middle_count") or 0
        )
        if actionable_count > 0 or ready_count > 0 or paper_ready_count > 0:
            print(
                f"FOUND reportable opportunities; wrote {args.out} and {args.summary_out} "
                f"actionable_count={actionable_count} "
                f"ready_ticket_count={ready_count} "
                f"paper_trade_ready_ticket_count={paper_ready_count}"
            )
            return 0
        print(
            f"No actionable opportunity found; wrote {args.out} and {args.summary_out} "
            f"candidates={summary.get('candidate_count', 0)} "
            f"model_only_arbitrage={summary.get('model_only_arbitrage_count', 0)} "
            f"execution_risky_middles={summary.get('execution_risky_middle_count', 0)} "
            f"model_only_middles={summary.get('model_only_middle_count', 0)} "
            f"plus_ev={summary.get('plus_ev_count', 0)}"
        )
        return 1
    finally:
        hunt._shutdown_runtime_resources()


if __name__ == "__main__":
    raise SystemExit(main())

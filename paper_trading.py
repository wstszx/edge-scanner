from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_LEDGER_PATH = Path("data") / "paper_trades" / "paper_trades.jsonl"
LEGACY_LEDGER_PATH = Path("data") / "paper_trades" / "middle_paper_trades.jsonl"
EXECUTION_ADAPTER_PROVIDERS = {"polymarket", "sx_bet", "artline"}
QUOTE_ONLY_PROVIDERS = {"bookmaker_xyz"}
MANUAL_WEB_PROVIDERS = {"artline"}
TEAM_ALIASES = {
    "basketball_nba": {
        "hawks": "atlanta hawks",
        "celtics": "boston celtics",
        "nets": "brooklyn nets",
        "hornets": "charlotte hornets",
        "bulls": "chicago bulls",
        "cavaliers": "cleveland cavaliers",
        "mavericks": "dallas mavericks",
        "nuggets": "denver nuggets",
        "pistons": "detroit pistons",
        "warriors": "golden state warriors",
        "rockets": "houston rockets",
        "pacers": "indiana pacers",
        "clippers": "los angeles clippers",
        "lakers": "los angeles lakers",
        "grizzlies": "memphis grizzlies",
        "heat": "miami heat",
        "bucks": "milwaukee bucks",
        "timberwolves": "minnesota timberwolves",
        "pelicans": "new orleans pelicans",
        "knicks": "new york knicks",
        "thunder": "oklahoma city thunder",
        "magic": "orlando magic",
        "76ers": "philadelphia 76ers",
        "sixers": "philadelphia 76ers",
        "suns": "phoenix suns",
        "blazers": "portland trail blazers",
        "trail blazers": "portland trail blazers",
        "kings": "sacramento kings",
        "spurs": "san antonio spurs",
        "raptors": "toronto raptors",
        "jazz": "utah jazz",
        "wizards": "washington wizards",
    }
}


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bookmaker_key(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "polymarket": "polymarket",
        "sx bet": "sx_bet",
        "sxbet": "sx_bet",
    }
    return aliases.get(text, text.replace(".", "_").replace(" ", "_").replace("-", "_"))


def _team_key(value: object, sport: object = None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    chars = [char if char.isalnum() else " " for char in text]
    normalized = " ".join("".join(chars).split())
    aliases = TEAM_ALIASES.get(str(sport or "").strip().lower()) or {}
    return aliases.get(normalized, normalized)


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


def _quote_time_skew_seconds(books: Sequence[dict[str, Any]]) -> int | None:
    quote_times = [
        quote_time
        for book in books
        if isinstance(book, dict)
        for quote_time in [_parse_quote_time(book.get("quote_updated_at"))]
        if quote_time is not None
    ]
    if len(quote_times) < 2:
        return None
    return int((max(quote_times) - min(quote_times)).total_seconds())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _middle_books(item: dict[str, Any]) -> list[dict[str, Any]]:
    books: list[dict[str, Any]] = []
    for side_key in ("side_a", "side_b"):
        side = item.get(side_key) if isinstance(item.get(side_key), dict) else {}
        books.append(
            {
                "team": side.get("team"),
                "bookmaker": side.get("bookmaker"),
                "bookmaker_key": side.get("bookmaker_key"),
                "price": side.get("price"),
                "effective_price": side.get("effective_price"),
                "line": side.get("line"),
                "max_stake": side.get("max_stake"),
                "quote_updated_at": side.get("quote_updated_at"),
                "quote_source": side.get("quote_source"),
                "fee_rate": side.get("fee_rate"),
                "book_event_url": side.get("book_event_url"),
                "execution_diagnostics": side.get("execution_diagnostics"),
                "market_hash": side.get("market_hash"),
                "token_id": side.get("token_id"),
                "asset_id": side.get("asset_id"),
                "condition_id": side.get("condition_id"),
                "outcome_index": side.get("outcome_index"),
                "is_exchange": side.get("is_exchange"),
            }
        )
    return books


def _middle_side_stakes(item: dict[str, Any]) -> list[float | None]:
    stakes = item.get("stakes") if isinstance(item.get("stakes"), dict) else {}
    rows: list[float | None] = []
    for side_key in ("side_a", "side_b"):
        side = stakes.get(side_key) if isinstance(stakes.get(side_key), dict) else {}
        rows.append(_safe_float(side.get("stake")))
    return rows


def _same_team_spread_middle_blocker(
    sport: object,
    market: object,
    first_team: object,
    first_line: object,
    second_team: object,
    second_line: object,
) -> str | None:
    if str(market or "").strip().lower() != "spreads":
        return None
    first = _team_key(first_team, sport)
    second = _team_key(second_team, sport)
    if not first or not second or first != second:
        return None
    first_point = _safe_float(first_line)
    second_point = _safe_float(second_line)
    if first_point is None or second_point is None:
        return None
    if first_point * second_point < 0:
        return "same_team_spread_middle"
    return None


def _arbitrage_books(item: dict[str, Any]) -> list[dict[str, Any]]:
    books: list[dict[str, Any]] = []
    for book in item.get("best_odds") or item.get("books") or []:
        if not isinstance(book, dict):
            continue
        books.append(
            {
                "outcome": book.get("outcome"),
                "bookmaker": book.get("bookmaker"),
                "bookmaker_key": book.get("bookmaker_key"),
                "price": book.get("price"),
                "effective_price": book.get("effective_price"),
                "point": book.get("point"),
                "max_stake": book.get("max_stake"),
                "quote_updated_at": book.get("quote_updated_at") or book.get("last_updated"),
                "quote_source": book.get("quote_source"),
                "fee_rate": book.get("fee_rate"),
                "book_event_url": book.get("book_event_url"),
                "execution_diagnostics": book.get("execution_diagnostics"),
                "market_hash": book.get("market_hash"),
                "token_id": book.get("token_id"),
                "asset_id": book.get("asset_id"),
                "condition_id": book.get("condition_id"),
                "outcome_index": book.get("outcome_index"),
                "is_exchange": book.get("is_exchange"),
            }
        )
    return books


def _arbitrage_stake_by_outcome(item: dict[str, Any]) -> dict[str, float | None]:
    stakes = item.get("stakes") if isinstance(item.get("stakes"), dict) else {}
    rows: dict[str, float | None] = {}
    for row in stakes.get("breakdown") or []:
        if not isinstance(row, dict):
            continue
        outcome = str(row.get("outcome") or "").strip()
        if outcome:
            rows[outcome] = _safe_float(row.get("stake"))
    return rows


def _identifier_blockers(bookmaker_key: str, book: dict[str, Any]) -> list[str]:
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


def _quote_only_blockers(bookmaker_key: str, book: dict[str, Any]) -> list[str]:
    if bookmaker_key in QUOTE_ONLY_PROVIDERS and _safe_float(book.get("max_stake")) is None:
        return [f"{bookmaker_key}_quote_only"]
    return []


def _manual_web_liquidity_risks(bookmaker_key: str, book: dict[str, Any]) -> list[str]:
    if bookmaker_key not in MANUAL_WEB_PROVIDERS:
        return []
    diagnostics = book.get("execution_diagnostics")
    risks: list[str] = []
    if isinstance(diagnostics, dict) and diagnostics.get("reason") == "max_bet_below_min_bet":
        risks.append(f"{bookmaker_key}_api_max_bet_below_min_bet")
    if _safe_float(book.get("max_stake")) is None:
        risks.append("manual_liquidity_unverified")
    return risks


def _max_stake_blocks_paper(bookmaker_key: str) -> bool:
    return bookmaker_key not in MANUAL_WEB_PROVIDERS


def _draft_order(
    bookmaker_key: str,
    book: dict[str, Any],
    stake: float | None,
    *,
    outcome: str | None = None,
) -> dict[str, Any] | None:
    if stake is None or stake <= 0:
        return None
    limit_odds = _safe_float(book.get("price"))
    if limit_odds is None or limit_odds <= 1:
        return None
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
    if bookmaker_key == "sx_bet":
        if not book.get("market_hash") or book.get("outcome_index") is None:
            return None
        return {
            "adapter": "sx_bet",
            "order_type": "limit",
            "market_hash": book.get("market_hash"),
            "outcome_index": book.get("outcome_index"),
            "outcome": outcome,
            "side": "buy",
            "size": round(float(stake), 2),
            "limit_odds": limit_odds,
        }
    if bookmaker_key == "artline":
        event_url = book.get("book_event_url")
        if not event_url:
            return None
        return {
            "adapter": "manual_artline",
            "order_type": "manual_web",
            "event_url": event_url,
            "outcome": outcome,
            "side": "buy",
            "size": round(float(stake), 2),
            "limit_odds": limit_odds,
        }
    return None


def build_arbitrage_execution_ticket(
    item: dict[str, Any],
    *,
    max_quote_skew_seconds: int = 120,
    min_executable_stake: float = 25.0,
) -> dict[str, Any]:
    books = _arbitrage_books(item)
    stakes = item.get("stakes") if isinstance(item.get("stakes"), dict) else {}
    stake_rows = _arbitrage_stake_by_outcome(item)
    blockers: set[str] = set()
    quote_skew = _quote_time_skew_seconds(books)
    max_quote_skew = max(0, int(max_quote_skew_seconds))
    if quote_skew is not None and max_quote_skew > 0 and quote_skew > max_quote_skew:
        blockers.add("quote_time_skew")
    total_stake = _safe_float(stakes.get("total")) or 0.0
    if total_stake <= 0:
        blockers.add("zero_total_stake")
    roi_percent = _safe_float(item.get("roi_percent"))
    if roi_percent is None or roi_percent <= 0:
        blockers.add("non_positive_roi")
    if stakes.get("limited_by_max_stake"):
        blockers.add("limited_by_liquidity")
    quality = item.get("execution_quality") if isinstance(item.get("execution_quality"), dict) else {}
    quality_status = str(quality.get("status") or "").strip().lower()
    quality_flags = [str(flag).strip() for flag in quality.get("flags") or [] if str(flag).strip()]
    if quality_status and quality_status != "high":
        blockers.add("execution_quality_not_high")
    blockers.update(quality_flags)

    legs: list[dict[str, Any]] = []
    for book in books:
        outcome = str(book.get("outcome") or "").strip()
        stake = stake_rows.get(outcome)
        bookmaker_key = _bookmaker_key(book.get("bookmaker") or book.get("bookmaker_key"))
        leg_blockers: set[str] = set()
        if bookmaker_key not in EXECUTION_ADAPTER_PROVIDERS:
            leg_blockers.add(f"{bookmaker_key}_execution_adapter_missing")
        leg_blockers.update(_quote_only_blockers(bookmaker_key, book))
        if not book.get("quote_updated_at"):
            leg_blockers.add("missing_quote_time")
        max_stake = _safe_float(book.get("max_stake"))
        liquidity_risks = _manual_web_liquidity_risks(bookmaker_key, book)
        if max_stake is None and _max_stake_blocks_paper(bookmaker_key):
            leg_blockers.add("missing_liquidity")
        elif (
            max_stake is not None
            and stake is not None
            and max_stake + 1e-9 < stake
            and _max_stake_blocks_paper(bookmaker_key)
        ):
            leg_blockers.add(f"{bookmaker_key}_stake_exceeds_max")
        if min_executable_stake > 0 and stake is not None and stake < min_executable_stake:
            leg_blockers.add("stake_below_minimum")
        if stake is None or stake <= 0:
            leg_blockers.add("missing_stake")
        leg_blockers.update(_identifier_blockers(bookmaker_key, book))
        blockers.update(leg_blockers)
        legs.append(
            {
                "outcome": outcome or None,
                "bookmaker": book.get("bookmaker"),
                "bookmaker_key": bookmaker_key,
                "market": item.get("market"),
                "point": book.get("point"),
                "stake": stake,
                "limit_price": book.get("price"),
                "effective_price": book.get("effective_price"),
                "max_stake": book.get("max_stake"),
                "quote_updated_at": book.get("quote_updated_at"),
                "quote_source": book.get("quote_source"),
                "fee_rate": book.get("fee_rate"),
                "book_event_url": book.get("book_event_url"),
                "execution_diagnostics": book.get("execution_diagnostics"),
                "market_hash": book.get("market_hash"),
                "token_id": book.get("token_id"),
                "asset_id": book.get("asset_id"),
                "condition_id": book.get("condition_id"),
                "outcome_index": book.get("outcome_index"),
                "draft_order": _draft_order(bookmaker_key, book, stake, outcome=outcome or None),
                "manual_liquidity_risks": liquidity_risks,
                "blockers": sorted(leg_blockers),
            }
        )

    submit_blockers = sorted(blockers)
    return {
        "source": "scan_arbitrage",
        "execution_type": "arbitrage",
        "dry_run": True,
        "paper_trade_ready": not submit_blockers,
        "can_submit_live": False,
        "status": "paper_ready" if not submit_blockers else "blocked",
        "event": item.get("event"),
        "sport": item.get("sport"),
        "market": item.get("market"),
        "roi_percent": item.get("roi_percent"),
        "profit": item.get("profit") or stakes.get("guaranteed_profit"),
        "total_stake": round(total_stake, 2),
        "requested_total": stakes.get("requested_total"),
        "opportunity_id": item.get("id") or item.get("event_id"),
        "preflight": {
            "wallet_required_for_paper": False,
            "quote_time_skew_seconds": quote_skew,
            "max_quote_skew_seconds": max_quote_skew,
            "min_executable_stake": max(0.0, float(min_executable_stake)),
        },
        "submit_blockers": submit_blockers,
        "legs": legs,
    }


def build_middle_execution_ticket(
    item: dict[str, Any],
    *,
    max_quote_skew_seconds: int = 120,
    min_executable_stake: float = 25.0,
) -> dict[str, Any]:
    books = _middle_books(item)
    stakes = item.get("stakes") if isinstance(item.get("stakes"), dict) else {}
    side_stakes = _middle_side_stakes(item)
    blockers: set[str] = set()
    quote_skew = _quote_time_skew_seconds(books)
    max_quote_skew = max(0, int(max_quote_skew_seconds))
    if quote_skew is not None and max_quote_skew > 0 and quote_skew > max_quote_skew:
        blockers.add("quote_time_skew")
    total_stake = _safe_float(stakes.get("total")) or 0.0
    if total_stake <= 0:
        blockers.add("zero_total_stake")
    ev_percent = _safe_float(item.get("ev_percent"))
    if ev_percent is None or ev_percent <= 0:
        blockers.add("non_positive_ev")
    if stakes.get("limited_by_max_stake"):
        blockers.add("limited_by_liquidity")
    if len(books) >= 2:
        same_team_blocker = _same_team_spread_middle_blocker(
            item.get("sport"),
            item.get("market"),
            books[0].get("team"),
            books[0].get("line"),
            books[1].get("team"),
            books[1].get("line"),
        )
        if same_team_blocker:
            blockers.add(same_team_blocker)

    legs: list[dict[str, Any]] = []
    for index, book in enumerate(books):
        stake = side_stakes[index] if index < len(side_stakes) else None
        bookmaker_key = _bookmaker_key(book.get("bookmaker") or book.get("bookmaker_key"))
        leg_blockers: set[str] = set()
        if bookmaker_key not in EXECUTION_ADAPTER_PROVIDERS:
            leg_blockers.add(f"{bookmaker_key}_execution_adapter_missing")
        leg_blockers.update(_quote_only_blockers(bookmaker_key, book))
        if not book.get("quote_updated_at"):
            leg_blockers.add("missing_quote_time")
        max_stake = _safe_float(book.get("max_stake"))
        liquidity_risks = _manual_web_liquidity_risks(bookmaker_key, book)
        if max_stake is None and _max_stake_blocks_paper(bookmaker_key):
            leg_blockers.add("missing_liquidity")
        elif (
            max_stake is not None
            and stake is not None
            and max_stake + 1e-9 < stake
            and _max_stake_blocks_paper(bookmaker_key)
        ):
            leg_blockers.add(f"{bookmaker_key}_stake_exceeds_max")
        if min_executable_stake > 0 and stake is not None and stake < min_executable_stake:
            leg_blockers.add("stake_below_minimum")
        leg_blockers.update(_identifier_blockers(bookmaker_key, book))
        blockers.update(leg_blockers)
        legs.append(
            {
                "team": book.get("team"),
                "bookmaker": book.get("bookmaker"),
                "bookmaker_key": bookmaker_key,
                "market": item.get("market"),
                "middle_zone": item.get("middle_zone"),
                "line": book.get("line"),
                "stake": stake,
                "limit_price": book.get("price"),
                "effective_price": book.get("effective_price"),
                "max_stake": book.get("max_stake"),
                "quote_updated_at": book.get("quote_updated_at"),
                "quote_source": book.get("quote_source"),
                "fee_rate": book.get("fee_rate"),
                "book_event_url": book.get("book_event_url"),
                "execution_diagnostics": book.get("execution_diagnostics"),
                "market_hash": book.get("market_hash"),
                "token_id": book.get("token_id"),
                "asset_id": book.get("asset_id"),
                "condition_id": book.get("condition_id"),
                "outcome_index": book.get("outcome_index"),
                "draft_order": _draft_order(bookmaker_key, book, stake),
                "manual_liquidity_risks": liquidity_risks,
                "blockers": sorted(leg_blockers),
            }
        )

    submit_blockers = sorted(blockers)
    return {
        "source": "scan_middle",
        "execution_type": "middle",
        "dry_run": True,
        "paper_trade_ready": not submit_blockers,
        "can_submit_live": False,
        "status": "paper_ready" if not submit_blockers else "blocked",
        "event": item.get("event"),
        "sport": item.get("sport"),
        "market": item.get("market"),
        "middle_zone": item.get("middle_zone"),
        "ev_percent": item.get("ev_percent"),
        "probability_percent": item.get("probability_percent"),
        "total_stake": round(total_stake, 2),
        "requested_total": stakes.get("requested_total"),
        "opportunity_id": item.get("id"),
        "preflight": {
            "wallet_required_for_paper": False,
            "quote_time_skew_seconds": quote_skew,
            "max_quote_skew_seconds": max_quote_skew,
            "min_executable_stake": max(0.0, float(min_executable_stake)),
        },
        "submit_blockers": submit_blockers,
        "legs": legs,
    }


def _plus_ev_book(item: dict[str, Any]) -> dict[str, Any]:
    bet = item.get("bet") if isinstance(item.get("bet"), dict) else {}
    return {
        "outcome": bet.get("outcome"),
        "bookmaker": bet.get("soft_book") or bet.get("bookmaker"),
        "bookmaker_key": bet.get("soft_key") or bet.get("bookmaker_key"),
        "price": bet.get("soft_odds") or bet.get("odds"),
        "effective_price": bet.get("effective_odds"),
        "point": bet.get("point"),
        "max_stake": bet.get("max_stake"),
        "quote_updated_at": bet.get("quote_updated_at"),
        "quote_source": bet.get("quote_source"),
        "fee_rate": bet.get("fee_rate"),
        "book_event_url": bet.get("book_event_url"),
        "execution_diagnostics": bet.get("execution_diagnostics"),
        "market_hash": bet.get("market_hash"),
        "token_id": bet.get("token_id"),
        "asset_id": bet.get("asset_id"),
        "condition_id": bet.get("condition_id"),
        "outcome_index": bet.get("outcome_index"),
        "is_exchange": bet.get("is_exchange"),
    }


def build_plus_ev_execution_ticket(
    item: dict[str, Any],
    *,
    max_quote_skew_seconds: int = 120,
    min_executable_stake: float = 25.0,
) -> dict[str, Any]:
    book = _plus_ev_book(item)
    reference = item.get("reference") if isinstance(item.get("reference"), dict) else {}
    sharp = item.get("sharp") if isinstance(item.get("sharp"), dict) else {}
    bookmaker_key = _bookmaker_key(book.get("bookmaker") or book.get("bookmaker_key"))
    stake = _safe_float((item.get("kelly") if isinstance(item.get("kelly"), dict) else {}).get("recommended_stake"))
    if stake is None or stake <= 0:
        stake = min(_safe_float(book.get("max_stake")) or 0.0, max(0.0, float(min_executable_stake)))
    blockers: set[str] = set()
    if bookmaker_key not in EXECUTION_ADAPTER_PROVIDERS:
        blockers.add(f"{bookmaker_key}_execution_adapter_missing")
    blockers.update(_quote_only_blockers(bookmaker_key, book))
    if not book.get("quote_updated_at"):
        blockers.add("missing_quote_time")
    max_stake = _safe_float(book.get("max_stake"))
    liquidity_risks = _manual_web_liquidity_risks(bookmaker_key, book)
    if max_stake is None and _max_stake_blocks_paper(bookmaker_key):
        blockers.add("missing_liquidity")
    elif (
        max_stake is not None
        and stake is not None
        and max_stake + 1e-9 < stake
        and _max_stake_blocks_paper(bookmaker_key)
    ):
        blockers.add(f"{bookmaker_key}_stake_exceeds_max")
    if min_executable_stake > 0 and (stake is None or stake < min_executable_stake):
        blockers.add("stake_below_minimum")
    if stake is None or stake <= 0:
        blockers.add("missing_stake")
    blockers.update(_identifier_blockers(bookmaker_key, book))
    edge_percent = _safe_float(item.get("net_edge_percent") or item.get("edge_percent"))
    if edge_percent is None or edge_percent <= 0:
        blockers.add("non_positive_edge")
    quality = item.get("execution_quality") if isinstance(item.get("execution_quality"), dict) else {}
    quality_status = str(quality.get("status") or "").strip().lower()
    quality_flags = [str(flag).strip() for flag in quality.get("flags") or [] if str(flag).strip()]
    if quality_status and quality_status != "high":
        blockers.add("execution_quality_not_high")
    blockers.update(quality_flags)
    quote_skew = _quote_time_skew_seconds([book, sharp])
    max_quote_skew = max(0, int(max_quote_skew_seconds))
    if quote_skew is not None and max_quote_skew > 0 and quote_skew > max_quote_skew:
        blockers.add("quote_time_skew")
    leg_blockers = sorted(blockers)
    leg = {
        "outcome": book.get("outcome"),
        "bookmaker": book.get("bookmaker"),
        "bookmaker_key": bookmaker_key,
        "market": item.get("market"),
        "point": book.get("point"),
        "stake": round(float(stake or 0.0), 2),
        "limit_price": book.get("price"),
        "effective_price": book.get("effective_price"),
        "max_stake": book.get("max_stake"),
        "quote_updated_at": book.get("quote_updated_at"),
        "quote_source": book.get("quote_source"),
        "fee_rate": book.get("fee_rate"),
        "book_event_url": book.get("book_event_url"),
        "execution_diagnostics": book.get("execution_diagnostics"),
        "market_hash": book.get("market_hash"),
        "token_id": book.get("token_id"),
        "asset_id": book.get("asset_id"),
        "condition_id": book.get("condition_id"),
        "outcome_index": book.get("outcome_index"),
        "draft_order": _draft_order(bookmaker_key, book, stake, outcome=str(book.get("outcome") or "") or None),
        "manual_liquidity_risks": liquidity_risks,
        "blockers": leg_blockers,
    }
    submit_blockers = sorted(blockers)
    return {
        "source": "scan_plus_ev",
        "execution_type": "plus_ev",
        "dry_run": True,
        "paper_trade_ready": not submit_blockers,
        "can_submit_live": False,
        "status": "paper_ready" if not submit_blockers else "blocked",
        "event": item.get("event"),
        "sport": item.get("sport"),
        "market": item.get("market"),
        "market_point": item.get("market_point"),
        "edge_percent": item.get("net_edge_percent") or item.get("edge_percent"),
        "ev_per_100": item.get("ev_per_100"),
        "total_stake": round(float(stake or 0.0), 2),
        "requested_total": None,
        "opportunity_id": item.get("id"),
        "reference": {
            **reference,
            "fair_odds": sharp.get("fair_odds"),
            "true_probability": sharp.get("true_probability"),
            "book": sharp.get("book"),
            "key": sharp.get("key"),
        },
        "preflight": {
            "wallet_required_for_paper": False,
            "quote_time_skew_seconds": quote_skew,
            "max_quote_skew_seconds": max_quote_skew,
            "min_executable_stake": max(0.0, float(min_executable_stake)),
        },
        "submit_blockers": submit_blockers,
        "legs": [leg],
    }


def _record_key(ticket: dict[str, Any]) -> str:
    legs = [
        {
            "bookmaker_key": leg.get("bookmaker_key"),
            "line": leg.get("line"),
            "limit_price": leg.get("limit_price"),
            "stake": leg.get("stake"),
            "market_hash": leg.get("market_hash"),
            "token_id": leg.get("token_id") or leg.get("asset_id"),
            "outcome_index": leg.get("outcome_index"),
        }
        for leg in ticket.get("legs") or []
        if isinstance(leg, dict)
    ]
    payload = {
        "execution_type": ticket.get("execution_type"),
        "event": ticket.get("event"),
        "sport": ticket.get("sport"),
        "market": ticket.get("market"),
        "middle_zone": ticket.get("middle_zone"),
        "legs": legs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_ledger_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _paper_record_validation_errors(record: dict[str, Any]) -> list[str]:
    ticket = record.get("ticket") if isinstance(record.get("ticket"), dict) else {}
    execution_type = record.get("execution_type") or ticket.get("execution_type")
    if execution_type != "middle":
        return []
    legs = ticket.get("legs") if isinstance(ticket.get("legs"), list) else []
    if len(legs) < 2 or not all(isinstance(leg, dict) for leg in legs[:2]):
        return []
    blocker = _same_team_spread_middle_blocker(
        record.get("sport") or ticket.get("sport"),
        record.get("market") or ticket.get("market"),
        legs[0].get("team"),
        legs[0].get("line"),
        legs[1].get("team"),
        legs[1].get("line"),
    )
    return [blocker] if blocker else []


def _validated_paper_record(record: dict[str, Any]) -> dict[str, Any]:
    errors = _paper_record_validation_errors(record)
    if not errors:
        return record
    next_record = dict(record)
    next_record["status"] = "invalid"
    next_record["validation_errors"] = errors
    ticket = dict(next_record.get("ticket") or {})
    blockers = [
        str(item)
        for item in (ticket.get("submit_blockers") if isinstance(ticket.get("submit_blockers"), list) else [])
        if str(item)
    ]
    for error in errors:
        if error not in blockers:
            blockers.append(error)
    ticket["submit_blockers"] = blockers
    ticket["paper_trade_ready"] = False
    ticket["status"] = "invalid"
    next_record["ticket"] = ticket
    return next_record


def _default_legacy_paths(path: Path) -> list[Path]:
    if path == DEFAULT_LEDGER_PATH:
        return [LEGACY_LEDGER_PATH]
    if path.name == DEFAULT_LEDGER_PATH.name:
        return [path.with_name(LEGACY_LEDGER_PATH.name)]
    return []


def load_paper_trades(
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    legacy_paths: Sequence[str | Path] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    path = Path(ledger_path)
    paths = [path, *([Path(item) for item in legacy_paths] if legacy_paths is not None else _default_legacy_paths(path))]
    latest_by_key: dict[str, dict[str, Any]] = {}
    anonymous_records: list[dict[str, Any]] = []
    for candidate in paths:
        for record in _read_ledger_records(candidate):
            key = str(record.get("paper_trade_key") or "")
            if key:
                if key in latest_by_key and candidate != path:
                    continue
                latest_by_key[key] = _validated_paper_record(record)
            else:
                anonymous_records.append(_validated_paper_record(record))
    records = [*latest_by_key.values(), *anonymous_records]
    records.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    if limit is not None:
        return records[: max(0, int(limit))]
    return records


def load_middle_paper_trades(
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    return load_paper_trades(ledger_path=ledger_path, limit=limit)


def _append_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _settlement_price(leg: dict[str, Any]) -> float:
    price = _safe_float(leg.get("limit_price") or leg.get("effective_price") or leg.get("price"))
    return price if price and price > 1 else 1.0


def _settlement_stake(leg: dict[str, Any]) -> float:
    stake = _safe_float(leg.get("stake"))
    return stake if stake and stake > 0 else 0.0


def _settlement_leg_payload(leg: dict[str, Any], result: str) -> dict[str, Any]:
    stake = _settlement_stake(leg)
    price = _settlement_price(leg)
    if result == "win":
        pnl = round(stake * (price - 1.0), 2)
    elif result == "push":
        pnl = 0.0
    else:
        pnl = round(-stake, 2)
    return {
        "bookmaker": leg.get("bookmaker"),
        "bookmaker_key": leg.get("bookmaker_key"),
        "outcome": leg.get("outcome") or leg.get("team"),
        "line": leg.get("line") if leg.get("line") is not None else leg.get("point"),
        "stake": round(stake, 2),
        "price": price,
        "result": result,
        "pnl": pnl,
    }


def _winner_from_result(result: dict[str, Any]) -> str:
    winner = result.get("winner") or result.get("winning_outcome") or result.get("outcome")
    return _team_key(winner)


def _score_map_from_result(result: dict[str, Any], sport: object) -> dict[str, float]:
    scores = result.get("scores")
    score_map: dict[str, float] = {}
    if isinstance(scores, dict):
        for team, value in scores.items():
            score = _safe_float(value)
            if score is not None:
                score_map[_team_key(team, sport)] = score
    for team_key, score_key in (("home_team", "home_score"), ("away_team", "away_score")):
        team = result.get(team_key)
        score = _safe_float(result.get(score_key))
        if team and score is not None:
            score_map[_team_key(team, sport)] = score
    return score_map


def _settle_moneyline_leg(leg: dict[str, Any], result: dict[str, Any], sport: object) -> dict[str, Any] | None:
    winner = _winner_from_result(result)
    outcome = _team_key(leg.get("outcome") or leg.get("team"), sport)
    if not winner or not outcome:
        return None
    return _settlement_leg_payload(leg, "win" if outcome == winner else "loss")


def _settle_spread_leg(leg: dict[str, Any], result: dict[str, Any], sport: object) -> dict[str, Any] | None:
    scores = _score_map_from_result(result, sport)
    team = _team_key(leg.get("team") or leg.get("outcome"), sport)
    line = _safe_float(leg.get("line") if leg.get("line") is not None else leg.get("point"))
    if not team or line is None or team not in scores or len(scores) < 2:
        return None
    opponent_scores = [score for key, score in scores.items() if key != team]
    if not opponent_scores:
        return None
    margin = scores[team] + line - opponent_scores[0]
    if abs(margin) < 1e-9:
        return _settlement_leg_payload(leg, "push")
    return _settlement_leg_payload(leg, "win" if margin > 0 else "loss")


def _settle_total_leg(leg: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    total = _safe_float(result.get("total_score"))
    if total is None:
        home_score = _safe_float(result.get("home_score"))
        away_score = _safe_float(result.get("away_score"))
        if home_score is not None and away_score is not None:
            total = home_score + away_score
    line = _safe_float(leg.get("line") if leg.get("line") is not None else leg.get("point"))
    outcome = str(leg.get("outcome") or leg.get("team") or "").strip().lower()
    if total is None or line is None or not outcome:
        return None
    if abs(total - line) < 1e-9:
        return _settlement_leg_payload(leg, "push")
    is_over = "over" in outcome
    return _settlement_leg_payload(leg, "win" if (total > line) == is_over else "loss")


def _settle_leg(record: dict[str, Any], leg: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    market = str(record.get("market") or (record.get("ticket") or {}).get("market") or "").strip().lower()
    sport = record.get("sport") or (record.get("ticket") or {}).get("sport")
    if market == "spreads" or (leg.get("line") is not None and not str(leg.get("outcome") or "").lower().startswith(("over", "under"))):
        settled = _settle_spread_leg(leg, result, sport)
        if settled is not None:
            return settled
    if market == "totals":
        return _settle_total_leg(leg, result)
    return _settle_moneyline_leg(leg, result, sport)


def settle_paper_trade_record(
    record: dict[str, Any],
    result: dict[str, Any],
    *,
    settled_at: str | None = None,
) -> dict[str, Any]:
    """Return a settled copy of a paper-trade record using a supplied final result."""
    if not isinstance(record, dict) or not isinstance(result, dict):
        return dict(record or {})
    result_status = str(result.get("status") or result.get("state") or "").strip().lower()
    if result_status and result_status not in {"final", "finished", "settled", "closed", "resolved"}:
        next_record = dict(record)
        next_record["settlement"] = {
            "status": "pending",
            "reason": "result_not_final",
        }
        return next_record
    ticket = record.get("ticket") if isinstance(record.get("ticket"), dict) else {}
    legs = [leg for leg in ticket.get("legs", []) if isinstance(leg, dict)]
    if not legs:
        next_record = dict(record)
        next_record["settlement"] = {
            "status": "pending",
            "reason": "missing_legs",
        }
        return next_record
    settled_legs: list[dict[str, Any]] = []
    for leg in legs:
        settled_leg = _settle_leg(record, leg, result)
        if settled_leg is None:
            next_record = dict(record)
            next_record["settlement"] = {
                "status": "pending",
                "reason": "unmatched_result",
            }
            return next_record
        settled_legs.append(settled_leg)
    total_stake = round(sum(float(leg.get("stake") or 0.0) for leg in settled_legs), 2)
    pnl = round(sum(float(leg.get("pnl") or 0.0) for leg in settled_legs), 2)
    roi_percent = round((pnl / total_stake) * 100.0, 2) if total_stake > 0 else None
    next_record = dict(record)
    next_record["status"] = "settled"
    next_record["settled_at"] = settled_at or _utc_now_iso()
    next_record["settlement"] = {
        "status": "settled",
        "pnl": pnl,
        "roi_percent": roi_percent,
        "total_stake": total_stake,
        "legs": settled_legs,
        "result": {
            key: value
            for key, value in result.items()
            if key not in {"api_key", "token", "secret", "private_key"}
        },
    }
    return next_record


def _result_for_record(record: dict[str, Any], results: dict[str, Any]) -> dict[str, Any] | None:
    keys = [
        record.get("paper_trade_key"),
        record.get("event_id"),
        record.get("event"),
        (record.get("ticket") or {}).get("event") if isinstance(record.get("ticket"), dict) else None,
    ]
    for key in keys:
        token = str(key or "").strip()
        if token and isinstance(results.get(token), dict):
            return results[token]
    return None


def settle_paper_trades(
    results: dict[str, Any],
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    settled_at: str | None = None,
) -> dict[str, Any]:
    """Append settlement updates for open paper-trade records matching supplied result keys."""
    path = Path(ledger_path)
    records = load_paper_trades(ledger_path=path)
    updates: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("status") or "").strip().lower() == "settled":
            continue
        result = _result_for_record(record, results)
        if result is None:
            continue
        settled = settle_paper_trade_record(record, result, settled_at=settled_at)
        if str(settled.get("status") or "").strip().lower() == "settled":
            updates.append(settled)
    _append_records(path, updates)
    return {
        "settled_count": len(updates),
        "records": [*updates, *records][:50],
    }


def record_middle_paper_trades(
    opportunities: Sequence[dict[str, Any]],
    *,
    scan_time: str,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    max_quote_skew_seconds: int = 120,
    min_executable_stake: float = 25.0,
) -> dict[str, Any]:
    tickets = [
        build_middle_execution_ticket(
            item,
            max_quote_skew_seconds=max_quote_skew_seconds,
            min_executable_stake=min_executable_stake,
        )
        for item in opportunities
        if isinstance(item, dict)
    ]
    ready_tickets = [ticket for ticket in tickets if ticket.get("paper_trade_ready")]
    existing = load_paper_trades(ledger_path=ledger_path)
    existing_keys = {str(record.get("paper_trade_key") or "") for record in existing}
    created_at = scan_time or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    new_records: list[dict[str, Any]] = []
    for ticket in ready_tickets:
        key = _record_key(ticket)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_records.append(
            {
                "paper_trade_key": key,
                "created_at": created_at,
                "status": "open",
                "event": ticket.get("event"),
                "sport": ticket.get("sport"),
                "market": ticket.get("market"),
                "middle_zone": ticket.get("middle_zone"),
                "ev_percent": ticket.get("ev_percent"),
                "probability_percent": ticket.get("probability_percent"),
                "total_stake": ticket.get("total_stake"),
                "ticket": ticket,
            }
        )
    _append_records(Path(ledger_path), new_records)
    current_records = [*new_records, *existing]
    return {
        "tickets": tickets,
        "execution_tickets": [],
        "middle_execution_tickets": tickets,
        "created": new_records,
        "records": current_records[:50],
        "summary": {
            "created_count": len(new_records),
            "ready_count": len(ready_tickets),
            "blocked_count": len(tickets) - len(ready_tickets),
            "arbitrage_ready_count": 0,
            "arbitrage_blocked_count": 0,
            "middle_ready_count": len(ready_tickets),
            "middle_blocked_count": len(tickets) - len(ready_tickets),
        },
    }


def record_scan_paper_trades(
    arbitrage_opportunities: Sequence[dict[str, Any]],
    middle_opportunities: Sequence[dict[str, Any]],
    plus_ev_opportunities: Sequence[dict[str, Any]] | None = None,
    *,
    scan_time: str,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    max_quote_skew_seconds: int = 120,
    min_executable_stake: float = 25.0,
) -> dict[str, Any]:
    execution_tickets = [
        build_arbitrage_execution_ticket(
            item,
            max_quote_skew_seconds=max_quote_skew_seconds,
            min_executable_stake=min_executable_stake,
        )
        for item in arbitrage_opportunities
        if isinstance(item, dict)
    ]
    middle_execution_tickets = [
        build_middle_execution_ticket(
            item,
            max_quote_skew_seconds=max_quote_skew_seconds,
            min_executable_stake=min_executable_stake,
        )
        for item in middle_opportunities
        if isinstance(item, dict)
    ]
    plus_ev_execution_tickets = [
        build_plus_ev_execution_ticket(
            item,
            max_quote_skew_seconds=max_quote_skew_seconds,
            min_executable_stake=min_executable_stake,
        )
        for item in (plus_ev_opportunities or [])
        if isinstance(item, dict)
    ]
    tickets = [*execution_tickets, *middle_execution_tickets, *plus_ev_execution_tickets]
    ready_tickets = [ticket for ticket in tickets if ticket.get("paper_trade_ready")]
    existing = load_paper_trades(ledger_path=ledger_path)
    existing_keys = {str(record.get("paper_trade_key") or "") for record in existing}
    created_at = scan_time or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    new_records: list[dict[str, Any]] = []
    for ticket in ready_tickets:
        key = _record_key(ticket)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_records.append(
            {
                "paper_trade_key": key,
                "created_at": created_at,
                "status": "open",
                "execution_type": ticket.get("execution_type"),
                "event": ticket.get("event"),
                "sport": ticket.get("sport"),
                "market": ticket.get("market"),
                "middle_zone": ticket.get("middle_zone"),
                "roi_percent": ticket.get("roi_percent"),
                "ev_percent": ticket.get("ev_percent"),
                "edge_percent": ticket.get("edge_percent"),
                "ev_per_100": ticket.get("ev_per_100"),
                "profit": ticket.get("profit"),
                "probability_percent": ticket.get("probability_percent"),
                "total_stake": ticket.get("total_stake"),
                "ticket": ticket,
            }
        )
    _append_records(Path(ledger_path), new_records)
    current_records = [*new_records, *existing]
    arbitrage_ready = [ticket for ticket in execution_tickets if ticket.get("paper_trade_ready")]
    middle_ready = [ticket for ticket in middle_execution_tickets if ticket.get("paper_trade_ready")]
    plus_ev_ready = [ticket for ticket in plus_ev_execution_tickets if ticket.get("paper_trade_ready")]
    return {
        "tickets": tickets,
        "execution_tickets": execution_tickets,
        "middle_execution_tickets": middle_execution_tickets,
        "plus_ev_execution_tickets": plus_ev_execution_tickets,
        "created": new_records,
        "records": current_records[:50],
        "summary": {
            "created_count": len(new_records),
            "ready_count": len(ready_tickets),
            "blocked_count": len(tickets) - len(ready_tickets),
            "arbitrage_ready_count": len(arbitrage_ready),
            "arbitrage_blocked_count": len(execution_tickets) - len(arbitrage_ready),
            "middle_ready_count": len(middle_ready),
            "middle_blocked_count": len(middle_execution_tickets) - len(middle_ready),
            "plus_ev_ready_count": len(plus_ev_ready),
            "plus_ev_blocked_count": len(plus_ev_execution_tickets) - len(plus_ev_ready),
        },
    }

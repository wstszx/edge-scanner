from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_LEDGER_PATH = Path("data") / "paper_trades" / "middle_paper_trades.jsonl"
EXECUTION_ADAPTER_PROVIDERS = {"polymarket", "sx_bet"}


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
    return blockers


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
        if not book.get("quote_updated_at"):
            leg_blockers.add("missing_quote_time")
        max_stake = _safe_float(book.get("max_stake"))
        if max_stake is None:
            leg_blockers.add("missing_liquidity")
        elif stake is not None and max_stake + 1e-9 < stake:
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
                "market_hash": book.get("market_hash"),
                "token_id": book.get("token_id"),
                "asset_id": book.get("asset_id"),
                "condition_id": book.get("condition_id"),
                "outcome_index": book.get("outcome_index"),
                "draft_order": _draft_order(bookmaker_key, book, stake, outcome=outcome or None),
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

    legs: list[dict[str, Any]] = []
    for index, book in enumerate(books):
        stake = side_stakes[index] if index < len(side_stakes) else None
        bookmaker_key = _bookmaker_key(book.get("bookmaker") or book.get("bookmaker_key"))
        leg_blockers: set[str] = set()
        if bookmaker_key not in EXECUTION_ADAPTER_PROVIDERS:
            leg_blockers.add(f"{bookmaker_key}_execution_adapter_missing")
        if not book.get("quote_updated_at"):
            leg_blockers.add("missing_quote_time")
        max_stake = _safe_float(book.get("max_stake"))
        if max_stake is None:
            leg_blockers.add("missing_liquidity")
        elif stake is not None and max_stake + 1e-9 < stake:
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
                "market_hash": book.get("market_hash"),
                "token_id": book.get("token_id"),
                "asset_id": book.get("asset_id"),
                "condition_id": book.get("condition_id"),
                "outcome_index": book.get("outcome_index"),
                "draft_order": _draft_order(bookmaker_key, book, stake),
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


def load_middle_paper_trades(
    *,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    path = Path(ledger_path)
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
    records.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    if limit is not None:
        return records[: max(0, int(limit))]
    return records


def _append_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


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
    existing = load_middle_paper_trades(ledger_path=ledger_path)
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
    tickets = [*execution_tickets, *middle_execution_tickets]
    ready_tickets = [ticket for ticket in tickets if ticket.get("paper_trade_ready")]
    existing = load_middle_paper_trades(ledger_path=ledger_path)
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
    return {
        "tickets": tickets,
        "execution_tickets": execution_tickets,
        "middle_execution_tickets": middle_execution_tickets,
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
        },
    }

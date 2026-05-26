from pathlib import Path

import paper_trading


def _middle_opportunity() -> dict:
    return {
        "id": "middle-1",
        "sport": "basketball_nba",
        "event": "San Antonio Spurs vs Oklahoma City Thunder",
        "market": "spreads",
        "middle_zone": "Spurs by 4-5",
        "ev_percent": 23.07,
        "probability_percent": 5.0,
        "stakes": {
            "requested_total": 100.0,
            "total": 100.0,
            "limited_by_max_stake": False,
            "max_executable_total": 209.84,
            "side_a": {"stake": 37.59, "payout": 117.22},
            "side_b": {"stake": 62.41, "payout": 117.2},
        },
        "side_a": {
            "team": "San Antonio Spurs",
            "bookmaker": "Polymarket",
            "bookmaker_key": "polymarket",
            "price": 3.125,
            "effective_price": 3.118472,
            "line": -3.5,
            "max_stake": 244.2048,
            "quote_updated_at": "2026-05-26T08:39:57Z",
            "quote_source": "clob_book_best_ask",
            "fee_rate": 0.03,
            "token_id": "poly-token-spurs",
            "asset_id": "poly-token-spurs",
            "outcome_index": 0,
            "is_exchange": True,
        },
        "side_b": {
            "team": "Oklahoma City Thunder",
            "bookmaker": "SX Bet",
            "bookmaker_key": "sx_bet",
            "price": 1.877934,
            "effective_price": 1.877934,
            "line": 5.5,
            "max_stake": 130.96875,
            "quote_updated_at": "2026-05-26T08:39:52Z",
            "quote_source": "rest_snapshot",
            "fee_rate": 0.0,
            "market_hash": "0xsxhash",
            "outcome_index": 1,
            "is_exchange": True,
        },
        "gap": {"points": 1.0, "middle_integers": [4, 5], "integer_count": 2},
        "outcomes": {"win_both_profit": 34.42, "typical_miss_profit": 17.21},
    }


def _arbitrage_opportunity() -> dict:
    return {
        "id": "arb-1",
        "event_id": "event-1",
        "sport": "basketball_nba",
        "event": "Los Angeles Lakers vs Boston Celtics",
        "market": "h2h",
        "roi_percent": 7.36,
        "profit": 7.36,
        "stakes": {
            "requested_total": 100.0,
            "total": 100.0,
            "limited_by_max_stake": False,
            "breakdown": [
                {"outcome": "Los Angeles Lakers", "stake": 48.78, "payout": 107.32},
                {"outcome": "Boston Celtics", "stake": 51.22, "payout": 107.56},
            ],
        },
        "execution_quality": {"status": "high", "flags": []},
        "best_odds": [
            {
                "outcome": "Los Angeles Lakers",
                "bookmaker": "SX Bet",
                "bookmaker_key": "sx_bet",
                "price": 2.2,
                "effective_price": 2.2,
                "max_stake": 150.0,
                "quote_updated_at": "2026-05-26T08:39:57Z",
                "quote_source": "rest_snapshot",
                "market_hash": "0xsxarb",
                "outcome_index": 0,
                "is_exchange": True,
            },
            {
                "outcome": "Boston Celtics",
                "bookmaker": "Polymarket",
                "bookmaker_key": "polymarket",
                "price": 2.1,
                "effective_price": 2.097,
                "max_stake": 160.0,
                "quote_updated_at": "2026-05-26T08:39:52Z",
                "quote_source": "clob_book_best_ask",
                "fee_rate": 0.03,
                "token_id": "poly-token-celtics",
                "asset_id": "poly-token-celtics",
                "outcome_index": 1,
                "is_exchange": True,
            },
        ],
    }


def test_build_arbitrage_execution_ticket_paper_ready() -> None:
    ticket = paper_trading.build_arbitrage_execution_ticket(
        _arbitrage_opportunity(),
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert ticket["status"] == "paper_ready"
    assert ticket["paper_trade_ready"] is True
    assert ticket["submit_blockers"] == []
    assert ticket["preflight"]["quote_time_skew_seconds"] == 5
    assert ticket["execution_type"] == "arbitrage"
    assert [leg["bookmaker_key"] for leg in ticket["legs"]] == ["sx_bet", "polymarket"]
    assert ticket["legs"][0]["draft_order"]["adapter"] == "sx_bet"
    assert ticket["legs"][1]["draft_order"]["adapter"] == "polymarket"


def test_record_scan_paper_trades_writes_arbitrage_and_middle_records(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper_trades.jsonl"

    result = paper_trading.record_scan_paper_trades(
        [_arbitrage_opportunity()],
        [_middle_opportunity()],
        scan_time="2026-05-26T08:40:00Z",
        ledger_path=ledger_path,
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert result["summary"]["created_count"] == 2
    assert result["summary"]["arbitrage_ready_count"] == 1
    assert result["summary"]["middle_ready_count"] == 1
    assert result["execution_tickets"][0]["status"] == "paper_ready"
    assert result["middle_execution_tickets"][0]["status"] == "paper_ready"
    records = paper_trading.load_middle_paper_trades(ledger_path=ledger_path)
    assert {record["execution_type"] for record in records} == {"arbitrage", "middle"}


def test_build_middle_execution_ticket_paper_ready_ignores_wallet_env() -> None:
    ticket = paper_trading.build_middle_execution_ticket(
        _middle_opportunity(),
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert ticket["status"] == "paper_ready"
    assert ticket["paper_trade_ready"] is True
    assert ticket["submit_blockers"] == []
    assert ticket["preflight"]["quote_time_skew_seconds"] == 5
    assert ticket["preflight"]["wallet_required_for_paper"] is False
    assert [leg["bookmaker_key"] for leg in ticket["legs"]] == ["polymarket", "sx_bet"]
    assert ticket["legs"][0]["draft_order"]["adapter"] == "polymarket"
    assert ticket["legs"][1]["draft_order"]["adapter"] == "sx_bet"


def test_build_middle_execution_ticket_blocks_stale_or_small_paper_trade() -> None:
    stale = _middle_opportunity()
    stale["side_b"] = {**stale["side_b"], "quote_updated_at": "2026-05-26T08:31:52Z"}

    ticket = paper_trading.build_middle_execution_ticket(
        stale,
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert ticket["status"] == "blocked"
    assert ticket["paper_trade_ready"] is False
    assert "quote_time_skew" in ticket["submit_blockers"]


def test_record_middle_paper_trades_writes_jsonl_and_dedupes(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper_trades.jsonl"

    first = paper_trading.record_middle_paper_trades(
        [_middle_opportunity()],
        scan_time="2026-05-26T08:40:00Z",
        ledger_path=ledger_path,
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )
    second = paper_trading.record_middle_paper_trades(
        [_middle_opportunity()],
        scan_time="2026-05-26T08:40:10Z",
        ledger_path=ledger_path,
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert first["summary"]["created_count"] == 1
    assert first["summary"]["ready_count"] == 1
    assert first["summary"]["blocked_count"] == 0
    assert second["summary"]["created_count"] == 0
    assert second["summary"]["ready_count"] == 1
    assert second["summary"]["blocked_count"] == 0
    records = paper_trading.load_middle_paper_trades(ledger_path=ledger_path)
    assert len(records) == 1
    assert records[0]["event"] == "San Antonio Spurs vs Oklahoma City Thunder"
    assert records[0]["status"] == "open"
    assert records[0]["ticket"]["paper_trade_ready"] is True

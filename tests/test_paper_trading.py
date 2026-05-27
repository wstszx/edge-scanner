import json
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


def _plus_ev_opportunity(bookmaker_key: str = "polymarket") -> dict:
    title = {
        "polymarket": "Polymarket",
        "sx_bet": "SX Bet",
        "artline": "Artline",
        "bookmaker_xyz": "bookmaker.xyz",
    }.get(bookmaker_key, bookmaker_key)
    bet = {
        "outcome": "Los Angeles Lakers",
        "soft_book": title,
        "soft_key": bookmaker_key,
        "soft_odds": 2.25,
        "effective_odds": 2.24,
        "point": None,
        "max_stake": 150.0,
        "quote_updated_at": "2026-05-26T08:39:57Z",
        "quote_source": "clob_book_best_ask",
        "fee_rate": 0.03 if bookmaker_key == "polymarket" else 0.0,
        "token_id": "poly-token-lakers" if bookmaker_key == "polymarket" else None,
        "asset_id": "poly-token-lakers" if bookmaker_key == "polymarket" else None,
        "market_hash": "0xsxplus" if bookmaker_key == "sx_bet" else None,
        "outcome_index": 0 if bookmaker_key in {"polymarket", "sx_bet"} else None,
    }
    if bookmaker_key in {"artline", "bookmaker_xyz"}:
        bet["max_stake"] = None
        bet["quote_updated_at"] = None
    return {
        "id": "plus-ev-1",
        "sport": "basketball_nba",
        "event": "Los Angeles Lakers vs Boston Celtics",
        "market": "h2h",
        "market_point": None,
        "edge_percent": 8.5,
        "net_edge_percent": 8.5,
        "ev_per_100": 8.5,
        "kelly": {"recommended_stake": 42.5},
        "bet": bet,
        "sharp": {
            "book": "DEX consensus",
            "key": "dex_consensus",
            "fair_odds": 2.06,
            "true_probability": 0.485,
            "quote_updated_at": "2026-05-26T08:39:52Z",
            "quote_source": "dex_consensus",
        },
        "reference": {"method": "dex_liquidity_weighted_consensus", "provider_count": 2},
        "execution_quality": {"status": "high", "flags": []},
    }


def test_build_plus_ev_execution_ticket_paper_ready() -> None:
    ticket = paper_trading.build_plus_ev_execution_ticket(
        _plus_ev_opportunity(),
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert ticket["status"] == "paper_ready"
    assert ticket["paper_trade_ready"] is True
    assert ticket["execution_type"] == "plus_ev"
    assert ticket["edge_percent"] == 8.5
    assert ticket["ev_per_100"] == 8.5
    assert ticket["total_stake"] == 42.5
    assert ticket["legs"][0]["bookmaker_key"] == "polymarket"
    assert ticket["legs"][0]["draft_order"]["adapter"] == "polymarket"
    assert ticket["legs"][0]["token_id"] == "poly-token-lakers"
    assert ticket["reference"]["method"] == "dex_liquidity_weighted_consensus"


def test_build_plus_ev_execution_ticket_blocks_artline_without_manual_event_link() -> None:
    ticket = paper_trading.build_plus_ev_execution_ticket(
        _plus_ev_opportunity("artline"),
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert ticket["status"] == "blocked"
    assert ticket["paper_trade_ready"] is False
    assert "artline_missing_event_url" in ticket["submit_blockers"]
    assert "missing_quote_time" in ticket["submit_blockers"]
    assert ticket["legs"][0]["manual_liquidity_risks"] == ["manual_liquidity_unverified"]
    assert ticket["legs"][0]["draft_order"] is None


def test_build_plus_ev_execution_ticket_allows_manual_artline_leg() -> None:
    item = _plus_ev_opportunity("artline")
    item["bet"].update(
        {
            "max_stake": 150.0,
            "quote_updated_at": "2026-05-26T08:39:57Z",
            "quote_source": "rest_snapshot",
            "book_event_url": "https://artline.bet/bookmaker/match/prematch/tennis/123",
            "execution_diagnostics": {
                "artline_max_bet": 150.0,
                "artline_min_bet": 5.0,
                "executable": True,
            },
        }
    )

    ticket = paper_trading.build_plus_ev_execution_ticket(
        item,
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert ticket["status"] == "paper_ready"
    assert ticket["paper_trade_ready"] is True
    assert ticket["submit_blockers"] == []
    assert ticket["legs"][0]["bookmaker_key"] == "artline"
    assert ticket["legs"][0]["draft_order"]["adapter"] == "manual_artline"
    assert ticket["legs"][0]["draft_order"]["event_url"] == "https://artline.bet/bookmaker/match/prematch/tennis/123"


def test_build_arbitrage_execution_ticket_allows_artline_manual_web_liquidity_risk() -> None:
    item = _arbitrage_opportunity()
    item.update(
        {
            "sport": "tennis_wta",
            "event": "Gerard Campana Lee vs Ivan Biletic",
            "roi_percent": 0.42,
            "profit": 0.42,
            "stakes": {
                "requested_total": 25.0,
                "total": 25.0,
                "limited_by_max_stake": False,
                "breakdown": [
                    {"outcome": "Gerard Campana Lee", "stake": 21.25, "payout": 25.18},
                    {"outcome": "Ivan Biletic", "stake": 3.75, "payout": 26.25},
                ],
            },
            "best_odds": [
                {
                    "outcome": "Gerard Campana Lee",
                    "bookmaker": "Polymarket",
                    "bookmaker_key": "polymarket",
                    "price": 1.176471,
                    "effective_price": 1.172646,
                    "max_stake": 27.2,
                    "quote_updated_at": "2026-05-27T09:27:19Z",
                    "quote_source": "clob_book_best_ask",
                    "fee_rate": 0.03,
                    "token_id": "poly-token",
                    "asset_id": "poly-token",
                    "outcome_index": 1,
                    "is_exchange": True,
                },
                {
                    "outcome": "Ivan Biletic",
                    "bookmaker": "Artline",
                    "bookmaker_key": "artline",
                    "price": 7.0,
                    "effective_price": 7.0,
                    "max_stake": None,
                    "quote_updated_at": "2026-05-27T09:25:56Z",
                    "quote_source": "rest_snapshot",
                    "book_event_id": "385771835504278",
                    "book_event_url": "https://artline.bet/bookmaker/match/prematch/tennis/385771835504278",
                    "selection_id": "3857718355042780064",
                    "provider_event_name": "0_ml_2",
                    "execution_diagnostics": {
                        "artline_max_bet": 0.01,
                        "artline_min_bet": 5.0,
                        "executable": False,
                        "reason": "max_bet_below_min_bet",
                    },
                },
            ],
        }
    )

    ticket = paper_trading.build_arbitrage_execution_ticket(
        item,
        max_quote_skew_seconds=120,
        min_executable_stake=0.0,
    )

    assert ticket["status"] == "paper_ready"
    assert ticket["paper_trade_ready"] is True
    assert ticket["submit_blockers"] == []
    artline_leg = ticket["legs"][1]
    assert artline_leg["draft_order"]["adapter"] == "manual_artline"
    assert artline_leg["draft_order"]["event_url"].endswith("/385771835504278")
    assert artline_leg["selection_id"] == "3857718355042780064"
    assert artline_leg["provider_event_name"] == "0_ml_2"
    assert artline_leg["draft_order"]["selection_id"] == "3857718355042780064"
    assert artline_leg["manual_liquidity_risks"] == [
        "artline_api_max_bet_below_min_bet",
        "manual_liquidity_unverified",
    ]


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


def test_record_scan_paper_trades_writes_plus_ev_records(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper_trades.jsonl"

    result = paper_trading.record_scan_paper_trades(
        [],
        [],
        [_plus_ev_opportunity()],
        scan_time="2026-05-26T08:40:00Z",
        ledger_path=ledger_path,
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert result["summary"]["created_count"] == 1
    assert result["summary"]["plus_ev_ready_count"] == 1
    assert result["plus_ev_execution_tickets"][0]["status"] == "paper_ready"
    records = paper_trading.load_paper_trades(ledger_path=ledger_path)
    assert records[0]["execution_type"] == "plus_ev"
    assert records[0]["edge_percent"] == 8.5


def test_default_ledger_path_is_not_middle_specific() -> None:
    assert paper_trading.DEFAULT_LEDGER_PATH.name == "paper_trades.jsonl"


def test_load_paper_trades_merges_legacy_middle_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper_trades.jsonl"
    legacy_path = tmp_path / "middle_paper_trades.jsonl"
    ledger_path.write_text(
        '{"paper_trade_key":"new","created_at":"2026-05-26T08:41:00Z","event":"new"}\n',
        encoding="utf-8",
    )
    legacy_path.write_text(
        "\n".join(
            [
                '{"paper_trade_key":"old","created_at":"2026-05-26T08:40:00Z","event":"old"}',
                '{"paper_trade_key":"new","created_at":"2026-05-26T08:39:00Z","event":"duplicate"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = paper_trading.load_paper_trades(ledger_path=ledger_path, legacy_paths=[legacy_path])

    assert [record["paper_trade_key"] for record in records] == ["new", "old"]
    assert [record["event"] for record in records] == ["new", "old"]


def test_load_paper_trades_keeps_latest_record_update(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper_trades.jsonl"
    ledger_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_trade_key": "paper-1",
                        "created_at": "2026-05-26T08:40:00Z",
                        "status": "open",
                        "event": "Los Angeles Lakers vs Boston Celtics",
                        "ticket": {"status": "paper_ready", "legs": []},
                    }
                ),
                json.dumps(
                    {
                        "paper_trade_key": "paper-1",
                        "created_at": "2026-05-26T08:40:00Z",
                        "status": "settled",
                        "settled_at": "2026-05-26T10:00:00Z",
                        "event": "Los Angeles Lakers vs Boston Celtics",
                        "ticket": {"status": "paper_ready", "legs": []},
                        "settlement": {"status": "settled", "pnl": 5.0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = paper_trading.load_paper_trades(ledger_path=ledger_path, legacy_paths=[])

    assert len(records) == 1
    assert records[0]["status"] == "settled"
    assert records[0]["settlement"]["pnl"] == 5.0


def test_settle_arbitrage_record_computes_net_pnl() -> None:
    record = {
        "paper_trade_key": "arb-settle",
        "created_at": "2026-05-26T08:40:00Z",
        "status": "open",
        "execution_type": "arbitrage",
        "sport": "basketball_nba",
        "event": "Los Angeles Lakers vs Boston Celtics",
        "market": "h2h",
        "ticket": {
            "execution_type": "arbitrage",
            "legs": [
                {"outcome": "Los Angeles Lakers", "stake": 50.0, "limit_price": 2.1},
                {"outcome": "Boston Celtics", "stake": 50.0, "limit_price": 2.0},
            ],
        },
    }

    settled = paper_trading.settle_paper_trade_record(
        record,
        {"status": "final", "winner": "Los Angeles Lakers"},
        settled_at="2026-05-26T10:00:00Z",
    )

    assert settled["status"] == "settled"
    assert settled["settlement"]["status"] == "settled"
    assert settled["settlement"]["pnl"] == 5.0
    assert [leg["result"] for leg in settled["settlement"]["legs"]] == ["win", "loss"]


def test_settle_spread_middle_record_computes_two_leg_result() -> None:
    record = {
        "paper_trade_key": "middle-settle",
        "created_at": "2026-05-26T08:40:00Z",
        "status": "open",
        "execution_type": "middle",
        "sport": "basketball_nba",
        "event": "Los Angeles Lakers vs Boston Celtics",
        "market": "spreads",
        "ticket": {
            "execution_type": "middle",
            "legs": [
                {"team": "Los Angeles Lakers", "line": 5.5, "stake": 50.0, "limit_price": 1.91},
                {"team": "Boston Celtics", "line": -3.5, "stake": 50.0, "limit_price": 2.0},
            ],
        },
    }

    settled = paper_trading.settle_paper_trade_record(
        record,
        {
            "status": "final",
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
            "home_score": 105,
            "away_score": 101,
        },
        settled_at="2026-05-26T10:00:00Z",
    )

    assert settled["status"] == "settled"
    assert settled["settlement"]["pnl"] == 95.5
    assert [leg["result"] for leg in settled["settlement"]["legs"]] == ["win", "win"]


def test_settle_paper_trades_appends_update_records(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper_trades.jsonl"
    paper_trading.record_scan_paper_trades(
        [_arbitrage_opportunity()],
        [],
        scan_time="2026-05-26T08:40:00Z",
        ledger_path=ledger_path,
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )
    key = paper_trading.load_paper_trades(ledger_path=ledger_path, legacy_paths=[])[0]["paper_trade_key"]

    result = paper_trading.settle_paper_trades(
        {key: {"status": "final", "winner": "Los Angeles Lakers"}},
        ledger_path=ledger_path,
        settled_at="2026-05-26T10:00:00Z",
    )

    assert result["settled_count"] == 1
    records = paper_trading.load_paper_trades(ledger_path=ledger_path, legacy_paths=[])
    assert len(records) == 1
    assert records[0]["status"] == "settled"
    assert records[0]["settlement"]["pnl"] > 0


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


def test_build_middle_execution_ticket_blocks_same_team_alias_spread() -> None:
    invalid = _middle_opportunity()
    invalid["side_a"] = {**invalid["side_a"], "team": "Spurs"}
    invalid["side_b"] = {**invalid["side_b"], "team": "San Antonio Spurs"}

    ticket = paper_trading.build_middle_execution_ticket(
        invalid,
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )

    assert ticket["status"] == "blocked"
    assert ticket["paper_trade_ready"] is False
    assert "same_team_spread_middle" in ticket["submit_blockers"]


def test_load_paper_trades_marks_same_team_alias_middle_record_invalid(tmp_path: Path) -> None:
    ticket = paper_trading.build_middle_execution_ticket(
        _middle_opportunity(),
        max_quote_skew_seconds=120,
        min_executable_stake=25.0,
    )
    ticket["legs"][0]["team"] = "Spurs"
    ticket["legs"][1]["team"] = "San Antonio Spurs"
    record = {
        "paper_trade_key": "bad-middle",
        "created_at": "2026-05-26T08:40:00Z",
        "status": "open",
        "execution_type": "middle",
        "event": ticket["event"],
        "sport": "basketball_nba",
        "market": "spreads",
        "ticket": ticket,
    }
    ledger_path = tmp_path / "paper_trades.jsonl"
    ledger_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    records = paper_trading.load_paper_trades(ledger_path=ledger_path)

    assert records[0]["status"] == "invalid"
    assert records[0]["validation_errors"] == ["same_team_spread_middle"]
    assert records[0]["ticket"]["paper_trade_ready"] is False
    assert records[0]["ticket"]["status"] == "invalid"
    assert "same_team_spread_middle" in records[0]["ticket"]["submit_blockers"]


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

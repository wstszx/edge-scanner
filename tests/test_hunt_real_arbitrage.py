from tools import hunt_real_arbitrage


def test_default_sports_include_supported_dex_provider_long_tail() -> None:
    default_sports = hunt_real_arbitrage.DEFAULT_SPORTS

    assert default_sports[:3] == ["basketball_nba", "basketball_ncaab", "baseball_mlb"]
    assert len(default_sports) == len(set(default_sports))
    for sport_key in [
        "basketball_euroleague",
        "mma_ufc",
        "rugby_union",
        "soccer_england_championship",
        "tennis_atp",
        "tennis_wta",
    ]:
        assert sport_key in default_sports


def test_compact_middle_surfaces_execution_risk_fields() -> None:
    item = {
        "event": "Away vs Home",
        "market": "totals",
        "middle_zone": "Total 45",
        "probability_percent": 4.5,
        "ev_percent": -1.25,
        "gross_ev_percent": 0.5,
        "stakes": {
            "total": 100.0,
            "requested_total": 100.0,
            "limited_by_max_stake": True,
            "max_executable_total": 40.0,
            "side_a": {"stake": 48.78, "bookmaker": "SX Bet"},
            "side_b": {"stake": 51.22, "bookmaker": "Polymarket"},
        },
        "side_a": {
            "bookmaker": "SX Bet",
            "bookmaker_key": "sx_bet",
            "price": 2.1,
            "effective_price": 2.05,
            "line": 44.5,
            "max_stake": 30.0,
            "quote_updated_at": None,
            "is_exchange": True,
            "market_hash": "0xsxhash",
            "outcome_index": 0,
            "execution_diagnostics": {"reason": "max_bet_below_min_bet"},
        },
        "side_b": {
            "bookmaker": "Polymarket",
            "bookmaker_key": "polymarket",
            "price": 2.0,
            "effective_price": 1.96,
            "line": 46.0,
            "max_stake": None,
            "quote_updated_at": "2026-05-25T09:00:00Z",
            "is_exchange": True,
            "token_id": "poly-total-under",
            "asset_id": "poly-total-under",
            "outcome_index": 1,
        },
        "gap": {
            "points": 1.5,
            "middle_integers": [45],
            "includes_key_number": False,
            "key_numbers_crossed": [],
        },
    }

    compact = hunt_real_arbitrage._compact_middle(item, "basketball_nba", ["sx_bet", "polymarket"], False)

    assert compact["sport"] == "basketball_nba"
    assert compact["event"] == "Away vs Home"
    assert compact["market"] == "totals"
    assert compact["ev_percent"] == -1.25
    assert compact["gross_ev_percent"] == 0.5
    assert compact["middle_zone"] == "Total 45"
    assert compact["stake"]["requested_total"] == 100.0
    assert compact["stake"]["limited_by_max_stake"] is True
    assert compact["stake"]["side_a"] == {"stake": 48.78, "bookmaker": "SX Bet"}
    assert compact["stake"]["side_b"] == {"stake": 51.22, "bookmaker": "Polymarket"}
    assert compact["books"][0]["bookmaker"] == "SX Bet"
    assert compact["books"][0]["bookmaker_key"] == "sx_bet"
    assert compact["books"][0]["market_hash"] == "0xsxhash"
    assert compact["books"][0]["outcome_index"] == 0
    assert compact["books"][0]["execution_diagnostics"] == {"reason": "max_bet_below_min_bet"}
    assert compact["books"][1]["bookmaker"] == "Polymarket"
    assert compact["books"][1]["bookmaker_key"] == "polymarket"
    assert compact["books"][1]["token_id"] == "poly-total-under"
    assert compact["books"][1]["asset_id"] == "poly-total-under"
    assert compact["books"][1]["outcome_index"] == 1
    assert "missing_quote_time" in compact["risk_flags"]
    assert "missing_liquidity" in compact["risk_flags"]
    assert "stake_limited" in compact["risk_flags"]


def test_is_actionable_arbitrage_requires_positive_roi_and_clean_execution_quality() -> None:
    actionable = {
        "event": "Away vs Home",
        "roi_percent": 0.75,
        "execution_quality": {"status": "high", "flags": []},
        "stakes": {"limited_by_max_stake": False},
    }
    missing_liquidity = {
        **actionable,
        "execution_quality": {"status": "medium", "flags": ["missing_liquidity"]},
    }
    model_negative = {**actionable, "roi_percent": 0.0}

    assert hunt_real_arbitrage._is_actionable_arbitrage(actionable)
    assert not hunt_real_arbitrage._is_actionable_arbitrage(missing_liquidity)
    assert not hunt_real_arbitrage._is_actionable_arbitrage(model_negative)


def test_scan_once_keeps_top_arbitrage_even_when_not_candidate(monkeypatch) -> None:
    def fake_run_scan(**kwargs):
        return {
            "success": True,
            "partial": False,
            "arbitrage": {
                "opportunities": [
                    {
                        "event": "Away vs Home",
                        "market": "h2h",
                        "roi_percent": -1.5,
                        "best_odds": [
                            {"bookmaker": "Artline", "bookmaker_key": "artline", "price": 2.6},
                            {"bookmaker": "SX Bet", "bookmaker_key": "sx_bet", "price": 1.58},
                        ],
                        "execution_quality": {"status": "low", "flags": ["missing_liquidity"]},
                        "stakes": {"limited_by_max_stake": False},
                    }
                ]
            },
            "middles": {"opportunities": [], "summary": {}},
            "plus_ev": {"opportunities": [], "summary": {}},
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    monkeypatch.setattr(hunt_real_arbitrage, "run_scan", fake_run_scan)

    row = hunt_real_arbitrage._scan_once(
        "basketball_nba",
        ["artline", "sx_bet"],
        api_key="",
        api_bookmakers=[],
        all_markets=False,
        stake=100.0,
        min_roi=0.01,
        allowed_quality={"high", "medium"},
    )

    assert row["arbitrage_count"] == 1
    assert row["positive_candidates"] == 0
    assert row["top_arbitrage"][0]["roi_percent"] == -1.5


def test_compact_arb_surfaces_provider_execution_ids() -> None:
    item = {
        "event": "Away vs Home",
        "market": "h2h",
        "roi_percent": 1.2,
        "best_odds": [
            {
                "outcome": "Home",
                "bookmaker": "SX Bet",
                "bookmaker_key": "sx_bet",
                "price": 2.25,
                "market_hash": "0xsxhash",
                "outcome_index": 0,
            },
            {
                "outcome": "Away",
                "bookmaker": "Polymarket",
                "bookmaker_key": "polymarket",
                "price": 1.85,
                "effective_price": 1.84,
                "fee_rate": 0.03,
                "token_id": "poly-token-away",
                "asset_id": "poly-token-away",
                "outcome_index": 1,
            },
        ],
    }

    compact = hunt_real_arbitrage._compact_arb(item, "basketball_nba", ["sx_bet", "polymarket"], False)

    assert compact["books"][0]["market_hash"] == "0xsxhash"
    assert compact["books"][0]["outcome_index"] == 0
    assert compact["books"][1]["token_id"] == "poly-token-away"
    assert compact["books"][1]["asset_id"] == "poly-token-away"
    assert compact["books"][1]["outcome_index"] == 1
    assert compact["books"][1]["fee_rate"] == 0.03


def test_is_actionable_middle_requires_positive_ev_and_no_execution_risks() -> None:
    actionable = {
        "event": "Away vs Home",
        "ev_percent": 1.25,
        "stake": {"limited_by_max_stake": False},
        "risk_flags": [],
    }
    missing_quote = {**actionable, "risk_flags": ["missing_quote_time"]}
    model_negative = {**actionable, "ev_percent": -0.1}

    assert hunt_real_arbitrage._is_actionable_middle(actionable)
    assert not hunt_real_arbitrage._is_actionable_middle(missing_quote)
    assert not hunt_real_arbitrage._is_actionable_middle(model_negative)


def test_compact_plus_ev_surfaces_fee_adjusted_edge_and_reference() -> None:
    item = {
        "event": "Away vs Home",
        "market": "h2h",
        "edge_percent": 3.2,
        "net_edge_percent": 3.2,
        "gross_edge_percent": 5.1,
        "ev_per_100": 3.2,
        "market_point": None,
        "bet": {
            "outcome": "Home",
            "soft_book": "Polymarket",
            "soft_key": "polymarket",
            "soft_odds": 2.12,
            "effective_odds": 2.08,
            "is_exchange": True,
            "quote_updated_at": None,
        },
        "sharp": {
            "book": "SX Bet",
            "key": "sx_bet",
            "fair_odds": 2.0,
            "true_probability_percent": 50.0,
            "vig_percent": 1.2,
            "quote_updated_at": "2026-05-25T09:00:00Z",
        },
        "kelly": {
            "recommended_stake": 12.34,
            "fraction_percent": 1.23,
        },
    }

    compact = hunt_real_arbitrage._compact_plus_ev(item, "basketball_nba", ["sx_bet", "polymarket"], False)

    assert compact["event"] == "Away vs Home"
    assert compact["market"] == "h2h"
    assert compact["edge_percent"] == 3.2
    assert compact["gross_edge_percent"] == 5.1
    assert compact["bet"]["bookmaker"] == "Polymarket"
    assert compact["bet"]["effective_odds"] == 2.08
    assert compact["reference"]["bookmaker"] == "SX Bet"
    assert compact["reference"]["fair_odds"] == 2.0
    assert compact["kelly"]["recommended_stake"] == 12.34
    assert "missing_quote_time" in compact["risk_flags"]


def test_scan_once_includes_value_and_middle_report_fields(monkeypatch) -> None:
    def fake_run_scan(**kwargs):
        return {
            "success": True,
            "partial": False,
            "arbitrage": {
                "opportunities": [
                    {
                        "event": "Away vs Home",
                        "market": "h2h",
                        "roi_percent": 1.5,
                        "best_odds": [],
                        "execution_quality": {"status": "high", "flags": []},
                        "stakes": {"limited_by_max_stake": False},
                    }
                ]
            },
            "middles": {
                "opportunities": [
                    {
                        "event": "Away vs Home",
                        "market": "spreads",
                        "middle_zone": "Home 3",
                        "probability_percent": 8.0,
                        "ev_percent": 0.75,
                        "side_a": {"bookmaker": "SX Bet", "quote_updated_at": "2026-05-25T09:00:00Z", "max_stake": 100},
                        "side_b": {"bookmaker": "Polymarket", "quote_updated_at": "2026-05-25T09:00:01Z", "max_stake": 100},
                        "stakes": {"total": 100, "limited_by_max_stake": False},
                        "gap": {"middle_integers": [3], "points": 1.0},
                    }
                ],
                "summary": {"positive_count": 1},
            },
            "plus_ev": {
                "opportunities": [
                    {
                        "event": "Away vs Home",
                        "market": "h2h",
                        "edge_percent": 2.5,
                        "bet": {"soft_book": "Bookmaker.xyz", "effective_odds": 2.1},
                        "sharp": {"book": "SX Bet", "fair_odds": 2.0},
                    }
                ],
                "summary": {"count": 1},
            },
            "scan_diagnostics": {"reason_code": "positive_middle_found"},
            "custom_providers": {},
            "sport_errors": [],
        }

    monkeypatch.setattr(hunt_real_arbitrage, "run_scan", fake_run_scan)

    row = hunt_real_arbitrage._scan_once(
        "basketball_nba",
        ["sx_bet", "polymarket"],
        api_key="",
        api_bookmakers=[],
        all_markets=False,
        stake=100.0,
        min_roi=0.01,
        allowed_quality={"high", "medium"},
    )

    assert row["arbitrage_count"] == 1
    assert row["positive_candidates"] == 1
    assert row["actionable_arbitrage_count"] == 1
    assert row["actionable_arbitrage"][0]["roi_percent"] == 1.5
    assert row["middle_count"] == 1
    assert row["positive_middle_count"] == 1
    assert row["actionable_middle_count"] == 1
    assert row["plus_ev_count"] == 1
    assert row["top_middles"][0]["ev_percent"] == 0.75
    assert row["actionable_middles"][0]["ev_percent"] == 0.75
    assert row["top_plus_ev"][0]["edge_percent"] == 2.5


def test_main_reports_positive_middles_when_no_arbitrage_candidate(monkeypatch, tmp_path, capsys) -> None:
    def fake_scan_once(*args, **kwargs):
        return {
            "sport": "basketball_nba",
            "providers": ["sx_bet", "polymarket"],
            "api_bookmakers": [],
            "all_markets": False,
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 0,
            "positive_candidates": 0,
            "actionable_arbitrage_count": 0,
            "actionable_arbitrage": [],
            "top_candidate": None,
            "middle_count": 1,
            "positive_middle_count": 1,
            "actionable_middle_count": 1,
            "top_middles": [{"event": "Away vs Home", "ev_percent": 1.25}],
            "actionable_middles": [{"event": "Away vs Home", "ev_percent": 1.25}],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    monkeypatch.setattr(hunt_real_arbitrage, "_scan_once", fake_scan_once)
    out = tmp_path / "hunt.json"

    exit_code = hunt_real_arbitrage.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "sx_bet,polymarket",
            "--out",
            str(out),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No arbitrage candidate found" in captured.out
    assert "positive_middles=1" in captured.out


def test_main_writes_actionable_middles_and_runs_cleanup(monkeypatch, tmp_path) -> None:
    def fake_scan_once(*args, **kwargs):
        return {
            "sport": "basketball_nba",
            "providers": ["sx_bet", "polymarket"],
            "api_bookmakers": [],
            "all_markets": False,
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 0,
            "positive_candidates": 0,
            "actionable_arbitrage_count": 0,
            "actionable_arbitrage": [],
            "top_candidate": None,
            "middle_count": 1,
            "positive_middle_count": 1,
            "actionable_middle_count": 1,
            "top_middles": [{"event": "Away vs Home", "ev_percent": 1.25, "risk_flags": []}],
            "actionable_middles": [{"event": "Away vs Home", "ev_percent": 1.25, "risk_flags": []}],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    cleanup_calls = []
    monkeypatch.setattr(hunt_real_arbitrage, "_scan_once", fake_scan_once)
    monkeypatch.setattr(hunt_real_arbitrage, "_shutdown_runtime_resources", lambda: cleanup_calls.append("cleanup"))
    out = tmp_path / "hunt.json"

    exit_code = hunt_real_arbitrage.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "sx_bet,polymarket",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 1
    assert cleanup_calls == ["cleanup"]
    assert '"actionable_middles"' in out.read_text(encoding="utf-8")
    assert '"actionable_arbitrage"' in out.read_text(encoding="utf-8")


def test_provider_capability_summary_marks_quote_only_provider() -> None:
    summary = hunt_real_arbitrage._provider_capability_summary(["bookmaker_xyz", "polymarket"])

    by_key = {item["key"]: item for item in summary}
    assert by_key["bookmaker_xyz"]["liquidity_confidence"] == "quote_only"
    assert by_key["polymarket"]["liquidity_confidence"] in {"explicit", "estimated"}


def test_default_dex_providers_include_artline_as_manual_execution_source() -> None:
    assert "artline" in hunt_real_arbitrage.DEFAULT_PROVIDERS
    summary = {
        row["key"]: row
        for row in hunt_real_arbitrage._provider_capability_summary(hunt_real_arbitrage.DEFAULT_PROVIDERS)
    }

    assert summary["artline"]["liquidity_confidence"] == "estimated"

from tools import hunt_dex_opportunities


def test_build_scan_jobs_includes_all_pairs_and_all_markets_subset() -> None:
    jobs = hunt_dex_opportunities._build_scan_jobs(
        sports=["basketball_nba"],
        provider_sets="all,pairs",
        include_all_markets=True,
        all_markets_sports=["basketball_nba"],
    )

    names = [job["name"] for job in jobs]
    assert "all:base" in names
    assert "pairs:betdex+bookmaker_xyz" in names
    assert "pairs:sx_bet+polymarket" in names
    assert "all:all_markets" in names
    assert all(job["sports"] == ["basketball_nba"] for job in jobs)


def test_build_scan_jobs_can_require_explicit_liquidity() -> None:
    jobs = hunt_dex_opportunities._build_scan_jobs(
        sports=["basketball_nba"],
        provider_sets="all,pairs",
        include_all_markets=True,
        all_markets_sports=["basketball_nba"],
        require_explicit_liquidity=True,
    )

    assert jobs
    names = [job["name"] for job in jobs]
    assert "all:base" in names
    assert "pairs:bookmaker_xyz+polymarket" not in names
    assert "pairs:sx_bet+polymarket" in names
    assert "all:all_markets" in names
    assert all("bookmaker_xyz" not in job["providers"] for job in jobs)
    assert all("artline" not in job["providers"] for job in jobs)
    assert all(len(job["providers"]) >= 2 for job in jobs)


def test_deduplicate_rows_merges_repeated_model_only_middles() -> None:
    first = {
        "job_name": "all:base",
        "event": "Away vs Home",
        "market": "spreads",
        "middle_zone": "Home 3",
        "ev_percent": 2.0,
        "risk_flags": ["missing_liquidity"],
        "books": [
            {"bookmaker": "bookmaker.xyz", "line": -1.5, "max_stake": None},
            {"bookmaker": "Polymarket", "line": 2.5, "max_stake": 100.0},
        ],
    }
    duplicate = {**first, "job_name": "pairs:bookmaker_xyz+polymarket", "ev_percent": 1.5}

    rows = hunt_dex_opportunities._deduplicate_rows(
        [first, duplicate],
        key_factory=hunt_dex_opportunities._middle_key,
        metric_key="ev_percent",
    )

    assert len(rows) == 1
    assert rows[0]["ev_percent"] == 2.0
    assert rows[0]["seen_in_jobs"] == ["all:base", "pairs:bookmaker_xyz+polymarket"]


def test_deduplicate_rows_handles_short_and_full_event_names() -> None:
    full_name = {
        "job_name": "pairs:sx_bet+polymarket",
        "event": "New York Knicks vs Cleveland Cavaliers",
        "market": "spreads",
        "middle_zone": "New York Knicks by 1",
        "ev_percent": 2.97,
        "books": [
            {"bookmaker": "SX Bet", "line": -0.5, "price": 1.909308},
            {"bookmaker": "Polymarket", "line": 1.5, "price": 2.12766},
        ],
    }
    short_name = {
        **full_name,
        "job_name": "all:all_markets",
        "event": "Cavaliers vs Knicks",
    }

    rows = hunt_dex_opportunities._deduplicate_rows(
        [full_name, short_name],
        key_factory=hunt_dex_opportunities._middle_key,
        metric_key="ev_percent",
    )

    assert len(rows) == 1
    assert rows[0]["seen_in_jobs"] == ["pairs:sx_bet+polymarket", "all:all_markets"]


def test_annotate_blocked_reasons_marks_quote_only_missing_liquidity() -> None:
    rows = [
        {
            "event": "Away vs Home",
            "market": "spreads",
            "ev_percent": 2.0,
            "risk_flags": ["missing_liquidity"],
            "books": [
                {"bookmaker": "bookmaker.xyz", "line": -1.5, "max_stake": None},
                {"bookmaker": "Polymarket", "line": 2.5, "max_stake": 100.0},
            ],
        }
    ]

    annotated = hunt_dex_opportunities._annotate_blocked_reasons(rows)

    assert annotated[0]["blocked_reasons"] == ["bookmaker_xyz_quote_only", "missing_liquidity"]


def test_annotate_blocked_reasons_marks_artline_max_bet_below_min_bet() -> None:
    rows = [
        {
            "event": "Away vs Home",
            "market": "h2h",
            "ev_percent": 1.0,
            "risk_flags": ["missing_liquidity"],
            "books": [
                {
                    "bookmaker": "Artline",
                    "bookmaker_key": "artline",
                    "price": 2.6,
                    "max_stake": None,
                    "execution_diagnostics": {
                        "artline_max_bet": 0.01,
                        "artline_min_bet": 5.0,
                        "executable": False,
                        "reason": "max_bet_below_min_bet",
                    },
                },
                {"bookmaker": "SX Bet", "price": 1.7, "max_stake": 100.0},
            ],
        }
    ]

    annotated = hunt_dex_opportunities._annotate_blocked_reasons(rows)

    assert annotated[0]["blocked_reasons"] == [
        "artline_max_bet_below_min_bet",
        "artline_quote_only",
        "missing_liquidity",
    ]


def test_annotate_blocked_reasons_marks_legs_below_min_executable_stake() -> None:
    rows = [
        {
            "event": "Away vs Home",
            "market": "spreads",
            "ev_percent": 2.0,
            "risk_flags": [],
            "books": [
                {"bookmaker": "SX Bet", "line": -1.5, "max_stake": 10.0},
                {"bookmaker": "Polymarket", "line": 2.5, "max_stake": 9.92},
            ],
        }
    ]

    annotated = hunt_dex_opportunities._annotate_blocked_reasons(rows, min_executable_stake=25.0)

    assert annotated[0]["blocked_reasons"] == ["stake_below_minimum"]
    assert annotated[0]["stake_limit"] == {
        "minimum_required": 25.0,
        "minimum_available": 9.92,
        "legs_below_minimum": [
            {"bookmaker": "SX Bet", "max_stake": 10.0},
            {"bookmaker": "Polymarket", "max_stake": 9.92},
        ],
    }


def test_quote_time_skew_seconds_uses_oldest_and_newest_book_quotes() -> None:
    item = {
        "books": [
            {"bookmaker": "SX Bet", "quote_updated_at": "2026-05-25T13:51:16Z"},
            {"bookmaker": "Polymarket", "quote_updated_at": "2026-05-25T13:59:27Z"},
        ]
    }

    assert hunt_dex_opportunities._quote_time_skew_seconds(item) == 491


def test_split_actionable_middles_demotes_skewed_quotes() -> None:
    fresh = {
        "event": "Fresh",
        "ev_percent": 1.2,
        "risk_flags": [],
        "books": [
            {"bookmaker": "SX Bet", "quote_updated_at": "2026-05-25T13:59:00Z"},
            {"bookmaker": "Polymarket", "quote_updated_at": "2026-05-25T13:59:30Z"},
        ],
    }
    stale = {
        "event": "Stale",
        "ev_percent": 2.0,
        "risk_flags": [],
        "books": [
            {"bookmaker": "SX Bet", "quote_updated_at": "2026-05-25T13:51:16Z"},
            {"bookmaker": "Polymarket", "quote_updated_at": "2026-05-25T13:59:27Z"},
        ],
    }

    actionable, risky = hunt_dex_opportunities._split_actionable_middles_by_quote_skew(
        [fresh, stale],
        max_quote_skew_seconds=120,
    )

    assert [item["event"] for item in actionable] == ["Fresh"]
    assert [item["event"] for item in risky] == ["Stale"]
    assert risky[0]["execution_risks"] == ["quote_time_skew"]
    assert risky[0]["quote_time_skew_seconds"] == 491


def test_runner_aggregates_actionable_and_model_only_results(monkeypatch, tmp_path) -> None:
    def fake_scan_once(sport, providers, **kwargs):
        all_markets = bool(kwargs.get("all_markets"))
        return {
            "sport": sport,
            "providers": list(providers),
            "api_bookmakers": [],
            "all_markets": all_markets,
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 1,
            "positive_candidates": 1,
            "actionable_arbitrage_count": 1 if all_markets else 0,
            "top_candidate": {"event": "Away vs Home", "roi_percent": 0.5},
            "actionable_arbitrage": [{"event": "Away vs Home", "roi_percent": 0.5}] if all_markets else [],
            "middle_count": 1,
            "positive_middle_count": 1,
            "actionable_middle_count": 0,
            "top_middles": [{"event": "Away vs Home", "ev_percent": 2.0, "risk_flags": ["missing_liquidity"]}],
            "actionable_middles": [],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "provider_capabilities": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    cleanup_calls = []
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_scan_once", fake_scan_once)
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_shutdown_runtime_resources", lambda: cleanup_calls.append("cleanup"))
    out = tmp_path / "dex_hunt.json"
    summary_out = tmp_path / "summary.md"

    exit_code = hunt_dex_opportunities.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "all",
            "--include-all-markets",
            "--all-markets-sports",
            "basketball_nba",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert exit_code == 0
    assert cleanup_calls == ["cleanup"]
    payload = out.read_text(encoding="utf-8")
    assert '"actionable_arbitrage"' in payload
    assert '"model_only_arbitrage"' in payload
    assert '"model_only_middles"' in payload


def test_runner_writes_model_only_arbitrage_blocked_reasons(monkeypatch, tmp_path) -> None:
    def fake_scan_once(sport, providers, **kwargs):
        return {
            "sport": sport,
            "providers": list(providers),
            "api_bookmakers": [],
            "all_markets": False,
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 1,
            "positive_candidates": 0,
            "actionable_arbitrage_count": 0,
            "top_candidate": None,
            "top_arbitrage": [
                {
                    "event": "Away vs Home",
                    "market": "h2h",
                    "roi_percent": -1.59,
                    "execution_quality": {
                        "status": "low",
                        "flags": ["missing_quote_time", "missing_liquidity"],
                    },
                    "books": [
                        {
                            "bookmaker": "Artline",
                            "bookmaker_key": "artline",
                            "price": 2.6,
                            "max_stake": None,
                            "execution_diagnostics": {
                                "artline_max_bet": 0.01,
                                "artline_min_bet": 5.0,
                                "executable": False,
                                "reason": "max_bet_below_min_bet",
                            },
                        },
                        {"bookmaker": "SX Bet", "bookmaker_key": "sx_bet", "price": 1.58, "max_stake": 10.0},
                    ],
                }
            ],
            "actionable_arbitrage": [],
            "middle_count": 0,
            "positive_middle_count": 0,
            "actionable_middle_count": 0,
            "top_middles": [],
            "actionable_middles": [],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "provider_capabilities": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_scan_once", fake_scan_once)
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_shutdown_runtime_resources", lambda: None)
    out = tmp_path / "dex_hunt.json"
    summary_out = tmp_path / "summary.md"

    exit_code = hunt_dex_opportunities.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "artline,sx_bet",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ]
    )

    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["summary"]["model_only_arbitrage_count"] == 1
    assert payload["summary"]["blocked_reason_counts"]["artline_max_bet_below_min_bet"] == 1
    assert payload["model_only_arbitrage"][0]["blocked_reasons"] == [
        "artline_max_bet_below_min_bet",
        "artline_quote_only",
        "missing_liquidity",
        "missing_quote_time",
        "non_positive_roi",
        "stake_below_minimum",
    ]


def test_runner_writes_deduped_blocked_reason_summary(monkeypatch, tmp_path) -> None:
    def fake_scan_once(sport, providers, **kwargs):
        return {
            "sport": sport,
            "providers": list(providers),
            "api_bookmakers": [],
            "all_markets": bool(kwargs.get("all_markets")),
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 0,
            "positive_candidates": 0,
            "actionable_arbitrage_count": 0,
            "top_candidate": None,
            "actionable_arbitrage": [],
            "middle_count": 1,
            "positive_middle_count": 1,
            "actionable_middle_count": 0,
            "top_middles": [
                {
                    "event": "Away vs Home",
                    "market": "spreads",
                    "middle_zone": "Home 3",
                    "ev_percent": 2.0,
                    "risk_flags": ["missing_liquidity"],
                    "books": [
                        {"bookmaker": "bookmaker.xyz", "line": -1.5, "max_stake": None},
                        {"bookmaker": "Polymarket", "line": 2.5, "max_stake": 100.0},
                    ],
                }
            ],
            "actionable_middles": [],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "provider_capabilities": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    cleanup_calls = []
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_scan_once", fake_scan_once)
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_shutdown_runtime_resources", lambda: cleanup_calls.append("cleanup"))
    out = tmp_path / "dex_hunt.json"
    summary_out = tmp_path / "summary.md"

    exit_code = hunt_dex_opportunities.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "all,pairs",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert exit_code == 1
    assert cleanup_calls == ["cleanup"]
    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["model_only_middle_count"] == 1
    assert payload["summary"]["blocked_reason_counts"] == {
        "bookmaker_xyz_quote_only": 1,
        "missing_liquidity": 1,
    }
    assert payload["model_only_middles"][0]["seen_in_jobs"]


def test_runner_applies_min_executable_stake_to_model_only_middles(monkeypatch, tmp_path) -> None:
    def fake_scan_once(sport, providers, **kwargs):
        return {
            "sport": sport,
            "providers": list(providers),
            "api_bookmakers": [],
            "all_markets": bool(kwargs.get("all_markets")),
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 0,
            "positive_candidates": 0,
            "actionable_arbitrage_count": 0,
            "top_candidate": None,
            "actionable_arbitrage": [],
            "middle_count": 1,
            "positive_middle_count": 1,
            "actionable_middle_count": 0,
            "top_middles": [
                {
                    "event": "Away vs Home",
                    "market": "spreads",
                    "middle_zone": "Home 3",
                    "ev_percent": 2.0,
                    "risk_flags": ["stake_limited"],
                    "books": [
                        {"bookmaker": "SX Bet", "line": -1.5, "max_stake": 10.0},
                        {"bookmaker": "Polymarket", "line": 2.5, "max_stake": 9.92},
                    ],
                }
            ],
            "actionable_middles": [],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "provider_capabilities": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_scan_once", fake_scan_once)
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_shutdown_runtime_resources", lambda: None)
    out = tmp_path / "dex_hunt.json"
    summary_out = tmp_path / "summary.md"

    exit_code = hunt_dex_opportunities.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "sx_bet,polymarket",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
            "--min-executable-stake",
            "25",
        ]
    )

    assert exit_code == 1
    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["min_executable_stake"] == 25.0
    assert payload["summary"]["blocked_reason_counts"] == {
        "stake_below_minimum": 1,
        "stake_limited": 1,
    }
    assert payload["model_only_middles"][0]["stake_limit"]["minimum_required"] == 25.0
    assert "- min_executable_stake: 25.0" in summary_out.read_text(encoding="utf-8")


def test_runner_defaults_min_executable_stake_for_dex_reports(monkeypatch, tmp_path) -> None:
    def fake_scan_once(sport, providers, **kwargs):
        return {
            "sport": sport,
            "providers": list(providers),
            "api_bookmakers": [],
            "all_markets": bool(kwargs.get("all_markets")),
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 0,
            "positive_candidates": 0,
            "actionable_arbitrage_count": 0,
            "top_candidate": None,
            "actionable_arbitrage": [],
            "middle_count": 1,
            "positive_middle_count": 1,
            "actionable_middle_count": 0,
            "top_middles": [
                {
                    "event": "Away vs Home",
                    "market": "spreads",
                    "middle_zone": "Home 3",
                    "ev_percent": 2.0,
                    "risk_flags": ["stake_limited"],
                    "books": [
                        {"bookmaker": "SX Bet", "line": -1.5, "max_stake": 10.0},
                        {"bookmaker": "Polymarket", "line": 2.5, "max_stake": 9.92},
                    ],
                }
            ],
            "actionable_middles": [],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "provider_capabilities": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_scan_once", fake_scan_once)
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_shutdown_runtime_resources", lambda: None)
    out = tmp_path / "dex_hunt.json"
    summary_out = tmp_path / "summary.md"

    exit_code = hunt_dex_opportunities.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "sx_bet,polymarket",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert exit_code == 1
    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["min_executable_stake"] == 25.0
    assert payload["summary"]["blocked_reason_counts"]["stake_below_minimum"] == 1


def test_runner_allows_disabling_min_executable_stake(monkeypatch, tmp_path) -> None:
    def fake_scan_once(sport, providers, **kwargs):
        return {
            "sport": sport,
            "providers": list(providers),
            "api_bookmakers": [],
            "all_markets": bool(kwargs.get("all_markets")),
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 0,
            "positive_candidates": 0,
            "actionable_arbitrage_count": 0,
            "top_candidate": None,
            "actionable_arbitrage": [],
            "middle_count": 1,
            "positive_middle_count": 1,
            "actionable_middle_count": 0,
            "top_middles": [
                {
                    "event": "Away vs Home",
                    "market": "spreads",
                    "middle_zone": "Home 3",
                    "ev_percent": 2.0,
                    "risk_flags": ["stake_limited"],
                    "books": [
                        {"bookmaker": "SX Bet", "line": -1.5, "max_stake": 10.0},
                        {"bookmaker": "Polymarket", "line": 2.5, "max_stake": 9.92},
                    ],
                }
            ],
            "actionable_middles": [],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "provider_capabilities": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_scan_once", fake_scan_once)
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_shutdown_runtime_resources", lambda: None)
    out = tmp_path / "dex_hunt.json"
    summary_out = tmp_path / "summary.md"

    exit_code = hunt_dex_opportunities.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "sx_bet,polymarket",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
            "--min-executable-stake",
            "0",
        ]
    )

    assert exit_code == 1
    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["min_executable_stake"] == 0.0
    assert "stake_below_minimum" not in payload["summary"]["blocked_reason_counts"]


def test_runner_demotes_skewed_actionable_middle_to_execution_risky(monkeypatch, tmp_path) -> None:
    def fake_scan_once(sport, providers, **kwargs):
        middle = {
            "event": "Away vs Home",
            "market": "spreads",
            "middle_zone": "Home 3",
            "ev_percent": 2.0,
            "risk_flags": [],
            "books": [
                {"bookmaker": "SX Bet", "line": -1.5, "max_stake": 100.0, "quote_updated_at": "2026-05-25T13:51:16Z"},
                {
                    "bookmaker": "Polymarket",
                    "line": 2.5,
                    "max_stake": 100.0,
                    "quote_updated_at": "2026-05-25T13:59:27Z",
                },
            ],
        }
        return {
            "sport": sport,
            "providers": list(providers),
            "api_bookmakers": [],
            "all_markets": bool(kwargs.get("all_markets")),
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 0,
            "positive_candidates": 0,
            "actionable_arbitrage_count": 0,
            "top_candidate": None,
            "actionable_arbitrage": [],
            "middle_count": 1,
            "positive_middle_count": 1,
            "actionable_middle_count": 1,
            "top_middles": [middle],
            "actionable_middles": [middle],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "provider_capabilities": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_scan_once", fake_scan_once)
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_shutdown_runtime_resources", lambda: None)
    out = tmp_path / "dex_hunt.json"
    summary_out = tmp_path / "summary.md"

    exit_code = hunt_dex_opportunities.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "sx_bet,polymarket",
            "--max-quote-skew-seconds",
            "120",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert exit_code == 1
    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["actionable_middle_count"] == 0
    assert payload["summary"]["execution_risky_middle_count"] == 1
    assert payload["summary"]["execution_risk_counts"] == {"quote_time_skew": 1}
    assert payload["execution_risky_middles"][0]["quote_time_skew_seconds"] == 491


def test_render_markdown_summary_includes_counts_and_top_risks() -> None:
    payload = {
        "summary": {
            "actionable_arbitrage_count": 0,
            "actionable_middle_count": 1,
            "execution_risky_middle_count": 2,
            "model_only_middle_count": 3,
            "blocked_reason_counts": {"bookmaker_xyz_quote_only": 3},
            "execution_risk_counts": {"quote_time_skew": 2},
            "require_explicit_liquidity": True,
            "max_quote_skew_seconds": 120,
        },
        "actionable_middles": [
            {"event": "Fresh vs Current", "market": "spreads", "middle_zone": "Fresh by 1", "ev_percent": 1.23}
        ],
        "execution_risky_middles": [
            {
                "event": "Stale vs Current",
                "market": "spreads",
                "middle_zone": "Stale by 1",
                "ev_percent": 2.34,
                "execution_risks": ["quote_time_skew"],
                "quote_time_skew_seconds": 491,
            }
        ],
        "model_only_middles": [
            {
                "event": "Quote vs Only",
                "market": "totals",
                "middle_zone": "Total 200",
                "ev_percent": 3.45,
                "blocked_reasons": ["bookmaker_xyz_quote_only"],
            }
        ],
    }

    markdown = hunt_dex_opportunities._render_markdown_summary(payload, source_json="report.json")

    assert "# DEX Opportunity Summary" in markdown
    assert "actionable_middle_count: 1" in markdown
    assert "execution_risky_middle_count: 2" in markdown
    assert "quote_time_skew: 2" in markdown
    assert "bookmaker_xyz_quote_only: 3" in markdown
    assert "Fresh vs Current" in markdown
    assert "source_json: report.json" in markdown


def test_runner_writes_markdown_summary(monkeypatch, tmp_path) -> None:
    def fake_scan_once(sport, providers, **kwargs):
        return {
            "sport": sport,
            "providers": list(providers),
            "api_bookmakers": [],
            "all_markets": bool(kwargs.get("all_markets")),
            "elapsed_seconds": 0.01,
            "success": True,
            "partial": False,
            "arbitrage_count": 0,
            "positive_candidates": 0,
            "actionable_arbitrage_count": 0,
            "top_candidate": None,
            "actionable_arbitrage": [],
            "middle_count": 0,
            "positive_middle_count": 0,
            "actionable_middle_count": 0,
            "top_middles": [],
            "actionable_middles": [],
            "plus_ev_count": 0,
            "top_plus_ev": [],
            "provider_capabilities": [],
            "scan_diagnostics": {},
            "custom_providers": {},
            "sport_errors": [],
        }

    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_scan_once", fake_scan_once)
    monkeypatch.setattr(hunt_dex_opportunities.hunt, "_shutdown_runtime_resources", lambda: None)
    out = tmp_path / "dex_hunt.json"
    summary_out = tmp_path / "summary.md"

    exit_code = hunt_dex_opportunities.main(
        [
            "--sports",
            "basketball_nba",
            "--provider-sets",
            "sx_bet,polymarket",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert exit_code == 1
    assert out.exists()
    assert summary_out.exists()
    assert "DEX Opportunity Summary" in summary_out.read_text(encoding="utf-8")

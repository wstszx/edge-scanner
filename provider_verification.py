from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from config import DEFAULT_STAKE_AMOUNT
from providers.dexsport_io import DEXSPORT_SOURCE
from providers.sportbet_one import SPORTBET_ONE_SOURCE
from scanner import run_scan

DEFAULT_PROVIDERS = [
    "betdex",
    "bookmaker_xyz",
    "dexsport_io",
    "sportbet_one",
    "sx_bet",
    "polymarket",
    "purebet",
]

DEFAULT_PROVIDER_TESTS = [
    "tests/test_provider_market_segmentation.py",
    "tests/test_provider_sport_coverage.py",
    "tests/test_purebet_market_parsing.py",
    "tests/test_polymarket_realtime.py",
    "tests/test_bookmaker_xyz_cache.py",
    "tests/test_scanner_regressions.py",
]

OFFICIAL_DOCS: dict[str, list[str]] = {
    "betdex": [
        "https://docs.betdex.com/",
        "https://docs.api.monacoprotocol.xyz/",
    ],
    "bookmaker_xyz": [
        "https://docs.bookmaker.xyz/guides/sportsbook",
        "https://docs.azuro.org/developers/onchain-data",
    ],
    "dexsport_io": [
        "https://dexsport.io/",
    ],
    "sportbet_one": [
        "https://sportbet.one/",
    ],
    "sx_bet": [
        "https://api.docs.sx.bet/",
    ],
    "polymarket": [
        "https://docs.polymarket.com/developers/gamma-markets-api/get-events",
        "https://docs.polymarket.com/developers/CLOB/introduction",
    ],
    "purebet": [
        "https://docs.purebet.io/",
    ],
}

PROVIDER_TITLES = {
    "betdex": "BetDEX",
    "bookmaker_xyz": "bookmaker.xyz",
    "dexsport_io": "Dexsport.io",
    "sportbet_one": "Sportbet.one",
    "sx_bet": "SX Bet",
    "polymarket": "Polymarket",
    "purebet": "Purebet",
}


@dataclass
class ProviderStatus:
    key: str
    name: str
    status: str
    enabled: bool
    events_merged: int
    errors: list[str]
    notes: list[str]
    docs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "status": self.status,
            "enabled": self.enabled,
            "events_merged": self.events_merged,
            "errors": list(self.errors),
            "notes": list(self.notes),
            "docs": list(self.docs),
        }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _safe_provider_title(provider_key: str) -> str:
    return PROVIDER_TITLES.get(provider_key, provider_key)


def _output_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _compact_text(value: object) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _provider_error_list(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    sports = summary.get("sports")
    if not isinstance(sports, list):
        return errors
    for item in sports:
        if not isinstance(item, dict):
            continue
        error = _compact_text(item.get("error"))
        if error:
            errors.append(error)
    return errors


def _provider_notes(provider_key: str, summary: dict[str, Any], stake_amount: float) -> list[str]:
    notes: list[str] = []
    if provider_key == "dexsport_io" and DEXSPORT_SOURCE in {"bookmaker_xyz", "api"}:
        notes.append(
            "Current code path proxies bookmaker.xyz data instead of a dedicated Dexsport API."
        )
    if provider_key == "sportbet_one" and SPORTBET_ONE_SOURCE in {"bookmaker_xyz", "api"}:
        notes.append(
            "Current code path proxies bookmaker.xyz data instead of a dedicated Sportbet.one API."
        )
    if provider_key == "purebet":
        errors = _provider_error_list(summary)
        if errors:
            notes.append(
                "Purebet failures should be checked against docs.purebet.io before changing parsing code."
            )
    if summary.get("events_merged", 0) == 0 and not _provider_error_list(summary):
        notes.append("Returned zero merged events in this scan.")
    if stake_amount <= 0:
        notes.append("Requested stake is non-positive; liquidity checks may be misleading.")
    return notes


def summarize_provider_statuses(
    result: dict[str, Any],
    provider_keys: Sequence[str],
    stake_amount: float,
) -> list[ProviderStatus]:
    custom_providers = result.get("custom_providers")
    if not isinstance(custom_providers, dict):
        custom_providers = {}

    statuses: list[ProviderStatus] = []
    for provider_key in provider_keys:
        summary = custom_providers.get(provider_key)
        if not isinstance(summary, dict):
            if provider_key == "purebet":
                purebet_summary = result.get("purebet")
                summary = purebet_summary if isinstance(purebet_summary, dict) else {}
            else:
                summary = {}
        enabled = bool(summary.get("enabled", provider_key in custom_providers))
        events_merged = int(summary.get("events_merged", 0) or 0)
        errors = _provider_error_list(summary)
        notes = _provider_notes(provider_key, summary, stake_amount)
        if errors:
            status = "error"
        elif events_merged > 0:
            status = "ok"
        elif enabled:
            status = "warning"
        else:
            status = "disabled"
        statuses.append(
            ProviderStatus(
                key=provider_key,
                name=_safe_provider_title(provider_key),
                status=status,
                enabled=enabled,
                events_merged=events_merged,
                errors=errors,
                notes=notes,
                docs=OFFICIAL_DOCS.get(provider_key, []),
            )
        )
    return statuses


def _stake_limit_note(stakes: dict[str, Any]) -> Optional[str]:
    if not isinstance(stakes, dict):
        return None
    limited = bool(stakes.get("limited_by_max_stake"))
    requested_total = float(stakes.get("requested_total", 0.0) or 0.0)
    executed_total = float(stakes.get("total", 0.0) or 0.0)
    if limited:
        return (
            f"liquidity-limited: requested={requested_total:.2f}, executable={executed_total:.2f}"
        )
    return None


def _min_positive_max_stake(books: Sequence[dict[str, Any]]) -> Optional[float]:
    stakes = []
    for row in books:
        if not isinstance(row, dict):
            continue
        value = _safe_float(row.get("max_stake"))
        if value is None or value <= 0:
            continue
        stakes.append(value)
    if not stakes:
        return None
    return min(stakes)


def _arb_highlights(items: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for item in list(items)[:limit]:
        books = []
        for row in item.get("best_odds") or []:
            if not isinstance(row, dict):
                continue
            books.append(
                {
                    "bookmaker": row.get("bookmaker"),
                    "price": row.get("price"),
                    "max_stake": row.get("max_stake"),
                }
            )
        note = _stake_limit_note(item.get("stakes") or {})
        highlights.append(
            {
                "event": item.get("event"),
                "market": item.get("market"),
                "roi_percent": item.get("roi_percent"),
                "books": books,
                "note": note,
            }
        )
    return highlights


def _middle_highlights(items: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for item in list(items)[:limit]:
        highlights.append(
            {
                "event": item.get("event"),
                "market": item.get("market"),
                "middle_zone": item.get("middle_zone"),
                "probability_percent": item.get("probability_percent"),
                "ev_percent": item.get("ev_percent"),
                "books": [
                    (item.get("side_a") or {}).get("bookmaker"),
                    (item.get("side_b") or {}).get("bookmaker"),
                ],
            }
        )
    return highlights


def _plus_ev_highlights(items: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for item in list(items)[:limit]:
        highlights.append(
            {
                "event": item.get("event"),
                "market": item.get("market"),
                "edge_percent": item.get("edge_percent"),
                "fair_odds": item.get("fair_odds"),
                "bookmaker": item.get("bookmaker"),
                "sharp_book": item.get("sharp_book"),
            }
        )
    return highlights


def summarize_scan(result: dict[str, Any], top_n: int) -> dict[str, Any]:
    arbitrage = result.get("arbitrage") if isinstance(result.get("arbitrage"), dict) else {}
    middles = result.get("middles") if isinstance(result.get("middles"), dict) else {}
    plus_ev = result.get("plus_ev") if isinstance(result.get("plus_ev"), dict) else {}
    return {
        "success": bool(result.get("success")),
        "partial": bool(result.get("partial")),
        "sport_errors": result.get("sport_errors") or [],
        "arbitrage_count": int(arbitrage.get("opportunities_count", 0) or 0),
        "middle_count": int(middles.get("opportunities_count", 0) or 0),
        "plus_ev_count": int(plus_ev.get("opportunities_count", 0) or 0),
        "timings": result.get("timings") or {},
        "top_arbitrage": _arb_highlights(arbitrage.get("opportunities") or [], top_n),
        "top_middles": _middle_highlights(middles.get("opportunities") or [], top_n),
        "top_plus_ev": _plus_ev_highlights(plus_ev.get("opportunities") or [], top_n),
    }


def run_provider_tests(timeout_seconds: int) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *DEFAULT_PROVIDER_TESTS]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "ran": True,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def run_live_scan(
    sport_key: str,
    provider_keys: Sequence[str],
    regions: Sequence[str],
    stake_amount: float,
    all_markets: bool,
) -> dict[str, Any]:
    include_purebet = "purebet" in provider_keys
    include_providers = [key for key in provider_keys if key != "purebet"]
    return run_scan(
        api_key="",
        sports=[sport_key],
        all_sports=False,
        all_markets=all_markets,
        stake_amount=stake_amount,
        regions=list(regions),
        bookmakers=None,
        include_purebet=include_purebet,
        include_providers=include_providers,
    )


def build_report(
    sport_key: str,
    provider_keys: Sequence[str],
    regions: Sequence[str],
    stake_amount: float,
    top_n: int,
    tests_result: Optional[dict[str, Any]],
    scan_result: dict[str, Any],
) -> dict[str, Any]:
    provider_statuses = summarize_provider_statuses(
        scan_result,
        provider_keys=provider_keys,
        stake_amount=stake_amount,
    )
    scan_summary = summarize_scan(scan_result, top_n=top_n)
    return {
        "generated_at": _utc_now(),
        "sport_key": sport_key,
        "regions": list(regions),
        "stake_amount": stake_amount,
        "providers": [status.as_dict() for status in provider_statuses],
        "tests": tests_result or {"ran": False},
        "scan": scan_summary,
    }


def _markdown_provider_table(provider_rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| Provider | Status | Events Merged | Notes | Docs |",
        "|---|---|---:|---|---|",
    ]
    for row in provider_rows:
        notes = "; ".join(row.get("notes") or row.get("errors") or []) or "-"
        docs = "<br>".join(row.get("docs") or []) or "-"
        lines.append(
            f"| {row.get('name')} | {row.get('status')} | {row.get('events_merged')} | {notes} | {docs} |"
        )
    return lines


def _markdown_top_items(
    title: str,
    rows: Sequence[dict[str, Any]],
    metric_key: str,
) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.append("- none")
        lines.append("")
        return lines
    for row in rows:
        metric = row.get(metric_key)
        event = row.get("event") or "unknown event"
        market = row.get("market") or "unknown market"
        note = _compact_text(row.get("note"))
        line = f"- {event} | {market} | {metric_key}={metric}"
        if note:
            line += f" | {note}"
        lines.append(line)
    lines.append("")
    return lines


def report_to_markdown(report: dict[str, Any]) -> str:
    tests = report.get("tests") or {}
    scan = report.get("scan") or {}
    lines = [
        "# Provider Verification Report",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- sport_key: {report.get('sport_key')}",
        f"- regions: {', '.join(report.get('regions') or [])}",
        f"- stake_amount: {report.get('stake_amount')}",
        "",
        "## Test Status",
        "",
        f"- ran: {tests.get('ran')}",
        f"- ok: {tests.get('ok')}",
        f"- returncode: {tests.get('returncode')}",
        "",
        "## Provider Status",
        "",
    ]
    lines.extend(_markdown_provider_table(report.get("providers") or []))
    lines.extend(
        [
            "",
            "## Scan Summary",
            "",
            f"- success: {scan.get('success')}",
            f"- partial: {scan.get('partial')}",
            f"- arbitrage_count: {scan.get('arbitrage_count')}",
            f"- middle_count: {scan.get('middle_count')}",
            f"- plus_ev_count: {scan.get('plus_ev_count')}",
            "",
        ]
    )
    lines.extend(_markdown_top_items("Top Arbitrage", scan.get("top_arbitrage") or [], "roi_percent"))
    lines.extend(_markdown_top_items("Top Middles", scan.get("top_middles") or [], "ev_percent"))
    lines.extend(_markdown_top_items("Top Plus EV", scan.get("top_plus_ev") or [], "edge_percent"))
    sport_errors = scan.get("sport_errors") or []
    lines.append("## Sport Errors")
    lines.append("")
    if not sport_errors:
        lines.append("- none")
    else:
        for item in sport_errors:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('sport_key') or item.get('sport')}: {_compact_text(item.get('error'))}"
            )
    lines.append("")
    return "\n".join(lines)


def collect_provider_alerts(report: dict[str, Any]) -> list[str]:
    alerts: list[str] = []
    tests = report.get("tests") or {}
    if tests.get("ran") and tests.get("ok") is False:
        alerts.append(f"tests failed: returncode={tests.get('returncode')}")
    for row in report.get("providers") or []:
        if not isinstance(row, dict):
            continue
        status = _compact_text(row.get("status"))
        if status in {"error", "warning"}:
            details = "; ".join((row.get("errors") or []) + (row.get("notes") or []))
            if details:
                alerts.append(f"{row.get('name')} [{status}]: {details}")
            else:
                alerts.append(f"{row.get('name')} [{status}]")
    return alerts


def collect_result_alerts(
    report: dict[str, Any],
    roi_threshold: float = 20.0,
    max_stake_threshold: float = 50.0,
) -> list[str]:
    alerts: list[str] = []
    scan = report.get("scan") or {}
    if not scan.get("success"):
        alerts.append("scan failed")
    if scan.get("partial"):
        alerts.append("scan returned partial results")
    for item in scan.get("sport_errors") or []:
        if not isinstance(item, dict):
            continue
        alerts.append(
            f"sport error [{item.get('sport_key') or item.get('sport')}]: {_compact_text(item.get('error'))}"
        )

    for row in scan.get("top_arbitrage") or []:
        if not isinstance(row, dict):
            continue
        reasons: list[str] = []
        roi = _safe_float(row.get("roi_percent"))
        if roi is not None and roi >= roi_threshold:
            reasons.append(f"high ROI {roi:.2f}%")
        low_max_stake = _min_positive_max_stake(row.get("books") or [])
        if low_max_stake is not None and low_max_stake < max_stake_threshold:
            reasons.append(f"low max stake {low_max_stake:.2f}")
        note = _compact_text(row.get("note"))
        if note:
            reasons.append(note)
        if reasons:
            alerts.append(
                f"arbitrage suspect [{row.get('event')} | {row.get('market')}]: {', '.join(reasons)}"
            )

    for row in scan.get("top_middles") or []:
        if not isinstance(row, dict):
            continue
        ev_percent = _safe_float(row.get("ev_percent"))
        if ev_percent is not None and ev_percent < 0:
            alerts.append(
                f"middle negative EV [{row.get('event')} | {row.get('market')}]: {ev_percent:.2f}%"
            )

    return alerts


def build_console_summary(report: dict[str, Any], written: Optional[dict[str, str]] = None) -> str:
    scan = report.get("scan") or {}
    lines = [
        "Provider Verification Summary",
        f"sport={report.get('sport_key')} regions={','.join(report.get('regions') or [])} "
        f"success={scan.get('success')} partial={scan.get('partial')}",
        (
            f"counts: arbitrage={scan.get('arbitrage_count')} "
            f"middles={scan.get('middle_count')} plus_ev={scan.get('plus_ev_count')}"
        ),
    ]
    provider_alerts = collect_provider_alerts(report)
    result_alerts = collect_result_alerts(report)
    if provider_alerts:
        lines.append("provider alerts:")
        lines.extend(f"- {item}" for item in provider_alerts)
    if result_alerts:
        lines.append("result alerts:")
        lines.extend(f"- {item}" for item in result_alerts)
    if not provider_alerts and not result_alerts:
        lines.append("no provider or result alerts")
    if written:
        latest_json = written.get("latest_json")
        latest_markdown = written.get("latest_markdown")
        if latest_json or latest_markdown:
            lines.append("artifacts:")
            if latest_json:
                lines.append(f"- latest_json={latest_json}")
            if latest_markdown:
                lines.append(f"- latest_markdown={latest_markdown}")
    return "\n".join(lines)


def report_has_alerts(report: dict[str, Any]) -> bool:
    return bool(collect_provider_alerts(report) or collect_result_alerts(report))


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    token = _output_token()
    json_path = out_dir / f"provider_verification_{token}.json"
    md_path = out_dir / f"provider_verification_{token}.md"
    latest_json_path = out_dir / "provider_verification_latest.json"
    latest_md_path = out_dir / "provider_verification_latest.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = report_to_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    latest_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_md_path.write_text(markdown, encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json_path),
        "latest_markdown": str(latest_md_path),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable provider verification checks and write a summary report.",
    )
    parser.add_argument(
        "--sport",
        default="basketball_nba",
        help="Sport key used for the live provider-only scan.",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=list(DEFAULT_PROVIDERS),
        help="Provider keys to verify. Default verifies all registered custom providers.",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        default=["us", "eu"],
        help="Regions passed to the live scan.",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=float(DEFAULT_STAKE_AMOUNT),
        help="Stake amount used for scan summaries.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of top opportunities to include in the report.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the provider-focused pytest subset.",
    )
    parser.add_argument(
        "--all-markets",
        action="store_true",
        help="Enable allMarkets for the live scan.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("data") / "provider_verification"),
        help="Directory where JSON and Markdown reports are written.",
    )
    parser.add_argument(
        "--pytest-timeout",
        type=int,
        default=180,
        help="Timeout in seconds for the provider-focused pytest subset.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a concise console summary instead of the full JSON payload.",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Print the full JSON payload to stdout even when summary mode is preferred elsewhere.",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with code 2 when provider or result alerts are detected.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    providers = [str(item).strip() for item in args.providers if str(item).strip()]
    tests_result = None if args.skip_tests else run_provider_tests(args.pytest_timeout)
    scan_result = run_live_scan(
        sport_key=args.sport,
        provider_keys=providers,
        regions=args.regions,
        stake_amount=float(args.stake),
        all_markets=bool(args.all_markets),
    )
    report = build_report(
        sport_key=args.sport,
        provider_keys=providers,
        regions=args.regions,
        stake_amount=float(args.stake),
        top_n=max(1, int(args.top)),
        tests_result=tests_result,
        scan_result=scan_result,
    )
    written = write_report(report, Path(args.out_dir))
    payload = {"report": report, "written": written}
    if args.summary_only and not args.json_stdout:
        print(build_console_summary(report, written=written))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.fail_on_alert and report_has_alerts(report):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

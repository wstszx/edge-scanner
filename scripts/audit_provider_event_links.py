from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from providers import betdex, bookmaker_xyz, polymarket, sx_bet


DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "provider_link_audits"

PROVIDER_AUDIT_CONFIG = {
    "polymarket": {
        "fetcher": polymarket.fetch_events_async,
        "sport_key": "basketball_nba",
        "markets": ["h2h"],
    },
    "betdex": {
        "fetcher": betdex.fetch_events_async,
        "sport_key": "icehockey_nhl",
        "markets": ["h2h"],
    },
    "sx_bet": {
        "fetcher": sx_bet.fetch_events_async,
        "sport_key": "icehockey_nhl",
        "markets": ["h2h"],
    },
    "bookmaker_xyz": {
        "fetcher": bookmaker_xyz.fetch_events_async,
        "sport_key": "icehockey_nhl",
        "markets": ["h2h"],
    },
}


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_match_text(value: str) -> str:
    value = _safe_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _team_hits(page_text: str, teams: list[str]) -> dict[str, bool]:
    normalized_page = _normalize_match_text(page_text)
    return {
        team: bool(team) and _normalize_match_text(team) in normalized_page
        for team in teams
    }


async def _fetch_provider_samples(provider_key: str, sample_count: int) -> list[dict[str, Any]]:
    config = PROVIDER_AUDIT_CONFIG[provider_key]
    fetcher = config["fetcher"]
    events = await fetcher(
        config["sport_key"],
        list(config["markets"]),
        ["us"],
    )
    samples: list[dict[str, Any]] = []
    for event in events:
        bookmakers = event.get("bookmakers") if isinstance(event, dict) else []
        book = bookmakers[0] if isinstance(bookmakers, list) and bookmakers else {}
        samples.append(
            {
                "provider": provider_key,
                "sport_key": _safe_text(event.get("sport_key")),
                "home_team": _safe_text(event.get("home_team")),
                "away_team": _safe_text(event.get("away_team")),
                "event_id": _safe_text(book.get("event_id") or event.get("id")),
                "event_url": _safe_text(book.get("event_url")),
            }
        )
        if len(samples) >= sample_count:
            break
    return samples


def _classify_page_status(page_text: str, teams: list[str]) -> tuple[str, dict[str, bool], str]:
    normalized_page = _normalize_match_text(page_text)
    excerpt = re.sub(r"\s+", " ", page_text).strip()[:1200]
    team_hits = _team_hits(page_text, teams)

    if "security verification" in normalized_page and "cloudflare" in normalized_page:
        status = "blocked"
    elif "oops something went wrong" in normalized_page and "doesn t exist" in normalized_page:
        status = "wrong_page"
    elif "sports ice hockey" in normalized_page and "/ / /" in excerpt and teams and not any(team_hits.values()):
        status = "wrong_page"
    elif teams and all(team_hits.values()):
        status = "ok"
    else:
        status = "mismatch"
    return status, team_hits, excerpt


async def _audit_sample(browser, sample: dict[str, Any], wait_ms: int) -> dict[str, Any]:
    event_url = _safe_text(sample.get("event_url"))
    teams = [_safe_text(sample.get("home_team")), _safe_text(sample.get("away_team"))]
    report = dict(sample)
    if not event_url:
        report.update({"status": "missing_url", "team_hits": {}, "page_excerpt": ""})
        return report

    attempt_reports: list[dict[str, Any]] = []
    for attempt in range(2):
        page = await browser.new_page()
        try:
            await page.goto(event_url, wait_until="domcontentloaded", timeout=90000)
            await page.wait_for_timeout(wait_ms)
            page_text = await page.locator("body").inner_text()
            status, team_hits, excerpt = _classify_page_status(page_text, teams)
            attempt_report = {
                "status": status,
                "team_hits": team_hits,
                "page_excerpt": excerpt,
                "page_title": await page.title(),
                "attempt": attempt + 1,
            }
        except PlaywrightTimeoutError as exc:
            attempt_report = {
                "status": "timeout",
                "error": str(exc),
                "team_hits": {},
                "page_excerpt": "",
                "page_title": "",
                "attempt": attempt + 1,
            }
        except Exception as exc:  # pragma: no cover - defensive guard for live audit failures
            attempt_report = {
                "status": "navigation_error",
                "error": str(exc),
                "team_hits": {},
                "page_excerpt": "",
                "page_title": "",
                "attempt": attempt + 1,
            }
        finally:
            await page.close()
        attempt_reports.append(attempt_report)
        if attempt_report["status"] in {"ok", "blocked"}:
            break

    final_attempt = attempt_reports[-1]
    report.update(final_attempt)
    report["attempts"] = attempt_reports
    return report


async def _run_audit(sample_count: int, wait_ms: int) -> dict[str, Any]:
    samples_by_provider: dict[str, list[dict[str, Any]]] = {}
    for provider_key in PROVIDER_AUDIT_CONFIG:
        try:
            samples_by_provider[provider_key] = await _fetch_provider_samples(provider_key, sample_count)
        except Exception as exc:
            samples_by_provider[provider_key] = [
                {
                    "provider": provider_key,
                    "status": "fetch_error",
                    "error": str(exc),
                    "event_url": "",
                    "home_team": "",
                    "away_team": "",
                    "event_id": "",
                    "sport_key": PROVIDER_AUDIT_CONFIG[provider_key]["sport_key"],
                }
            ]

    report: dict[str, Any] = {
        "generated_at": _utc_now(),
        "sample_count": sample_count,
        "wait_ms": wait_ms,
        "providers": {},
    }

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        for provider_key, samples in samples_by_provider.items():
            provider_reports: list[dict[str, Any]] = []
            for sample in samples:
                if sample.get("status") == "fetch_error":
                    provider_reports.append(sample)
                    continue
                provider_reports.append(await _audit_sample(browser, sample, wait_ms))
            report["providers"][provider_key] = provider_reports
        await browser.close()

    return report


def _write_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"provider_link_audit_{timestamp}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit custom provider event links against live pages.")
    parser.add_argument("--samples-per-provider", type=int, default=2)
    parser.add_argument("--wait-ms", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = asyncio.run(_run_audit(max(1, args.samples_per_provider), max(0, args.wait_ms)))
    output_path = _write_report(report, args.output_dir)
    print(f"saved report to {output_path}")
    for provider_key, entries in report["providers"].items():
        statuses = [entry.get("status", "unknown") for entry in entries]
        print(f"{provider_key}: {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

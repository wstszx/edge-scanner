from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import SPORT_OPTIONS
from providers import PROVIDER_FETCHERS

DEFAULT_PROVIDER_KEYS = [
    "artline",
    "betdex",
    "bookmaker_xyz",
    "sx_bet",
    "polymarket",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _canonical_matchup(home_team: object, away_team: object) -> tuple[str, str]:
    pair = sorted([_normalize_text(home_team).lower(), _normalize_text(away_team).lower()])
    return pair[0], pair[1]


def _event_market_keys(event: dict[str, Any]) -> list[str]:
    bookmakers = event.get("bookmakers") if isinstance(event.get("bookmakers"), list) else []
    if not bookmakers or not isinstance(bookmakers[0], dict):
        return []
    markets = bookmakers[0].get("markets") if isinstance(bookmakers[0].get("markets"), list) else []
    keys: list[str] = []
    seen = set()
    for market in markets:
        if not isinstance(market, dict):
            continue
        key = _normalize_text(market.get("key"))
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def build_report_data(samples: dict[str, Sequence[dict[str, Any]]], *, scan_mode: str = "live") -> dict[str, Any]:
    providers: dict[str, dict[str, Any]] = {}
    matchup_sets: dict[str, set[tuple[str, str]]] = {}
    matchup_markets: dict[str, dict[tuple[str, str], set[str]]] = {}

    for provider_key, events in samples.items():
        explicit_live_events = 0
        scheduled_or_prematch_events = 0
        unique_matchups: set[tuple[str, str]] = set()
        sample_events: list[dict[str, Any]] = []
        markets_by_matchup: dict[tuple[str, str], set[str]] = {}

        for event in events or []:
            if not isinstance(event, dict):
                continue
            live_state = event.get("live_state") if isinstance(event.get("live_state"), dict) else {}
            if live_state.get("is_live") is True:
                explicit_live_events += 1
            elif str(live_state.get("status") or "").lower() in {"scheduled", "prematch", "pre_play", "preplay"}:
                scheduled_or_prematch_events += 1

            matchup = _canonical_matchup(event.get("home_team"), event.get("away_team"))
            if all(matchup):
                unique_matchups.add(matchup)
                markets_by_matchup.setdefault(matchup, set()).update(_event_market_keys(event))

            if len(sample_events) < 5:
                sample_events.append(
                    {
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "commence_time": event.get("commence_time"),
                        "live_state": live_state,
                        "market_keys": _event_market_keys(event),
                    }
                )

        providers[provider_key] = {
            "raw_events": len(list(events or [])),
            "explicit_live_events": explicit_live_events,
            "scheduled_or_prematch_events": scheduled_or_prematch_events,
            "unique_matchups": len(unique_matchups),
            "sample_events": sample_events,
        }
        matchup_sets[provider_key] = unique_matchups
        matchup_markets[provider_key] = markets_by_matchup

    overlap_pairs: list[dict[str, Any]] = []
    provider_keys = sorted(providers.keys())
    for idx, left in enumerate(provider_keys):
        for right in provider_keys[idx + 1:]:
            overlap = matchup_sets.get(left, set()) & matchup_sets.get(right, set())
            common_market_keys: set[str] = set()
            for matchup in overlap:
                left_keys = matchup_markets.get(left, {}).get(matchup, set())
                right_keys = matchup_markets.get(right, {}).get(matchup, set())
                common_market_keys.update(left_keys & right_keys)
            overlap_pairs.append(
                {
                    "providers": [left, right],
                    "overlap_matchups": len(overlap),
                    "common_market_keys": sorted(common_market_keys),
                }
            )

    return {
        "generated_at": _utc_now(),
        "scan_mode": scan_mode,
        "providers": providers,
        "overlap_pairs": overlap_pairs,
    }


def build_report_data_from_scan_result(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("scan_diagnostics") if isinstance(result, dict) else {}
    provider_breakdown = diagnostics.get("provider_breakdown") if isinstance(diagnostics, dict) else []
    providers: dict[str, dict[str, Any]] = {}
    for row in provider_breakdown if isinstance(provider_breakdown, list) else []:
        if not isinstance(row, dict):
            continue
        provider_key = _normalize_text(row.get("provider_key"))
        if not provider_key:
            continue
        providers[provider_key] = {
            "raw_events": int(row.get("raw_events", 0) or 0),
            "explicit_live_events": None,
            "scheduled_or_prematch_events": None,
            "unique_matchups": int(row.get("appended_new", 0) or 0),
            "sample_events": [],
        }
    return {
        "generated_at": _utc_now(),
        "scan_mode": _normalize_text(result.get("scan_mode") or "live"),
        "providers": providers,
        "overlap_pairs": [],
        "scan_diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
    }


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Availability Report",
        "",
        f"Generated at: {report.get('generated_at')}",
        f"Scan mode: {report.get('scan_mode')}",
        "",
        "## Providers",
        "",
        "| Provider | Raw Events | Explicit Live | Scheduled/Prematch | Unique Matchups |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    providers = report.get("providers") if isinstance(report.get("providers"), dict) else {}
    for provider_key in sorted(providers.keys()):
        row = providers[provider_key] if isinstance(providers[provider_key], dict) else {}
        lines.append(
            f"| {provider_key} | {int(row.get('raw_events', 0) or 0)} | {int(row.get('explicit_live_events', 0) or 0)} | {int(row.get('scheduled_or_prematch_events', 0) or 0)} | {int(row.get('unique_matchups', 0) or 0)} |"
        )

    lines.extend(["", "## Overlap Pairs", ""])
    for item in report.get("overlap_pairs") or []:
        if not isinstance(item, dict):
            continue
        providers_pair = item.get("providers") or []
        lines.append(f"- {providers_pair}: overlap={item.get('overlap_matchups')}, common_markets={item.get('common_market_keys')}")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"live_availability_{token}.json"
    markdown_path = output_dir / f"live_availability_{token}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def write_live_scan_report(result: dict[str, Any], output_dir: Path | str = "data") -> dict[str, str]:
    report = build_report_data_from_scan_result(result)
    return write_report(report, Path(output_dir))


async def _sample_live_provider_events(
    *,
    sports: Sequence[str],
    provider_keys: Sequence[str],
    regions: Sequence[str],
    markets: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in provider_keys}
    for provider_key in provider_keys:
        fetch = PROVIDER_FETCHERS.get(provider_key)
        if not callable(fetch):
            continue
        for sport_key in sports:
            try:
                events = await fetch(
                    sport_key,
                    markets,
                    regions=regions,
                    bookmakers=[provider_key],
                    context={"live": True},
                )
            except Exception:
                continue
            if not isinstance(events, list):
                continue
            samples[provider_key].extend(item for item in events if isinstance(item, dict))
    return samples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate live availability report.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--sports", nargs="*", default=[row["key"] for row in SPORT_OPTIONS])
    parser.add_argument("--providers", nargs="*", default=list(DEFAULT_PROVIDER_KEYS))
    parser.add_argument("--regions", nargs="*", default=["us", "eu"])
    parser.add_argument("--markets", nargs="*", default=["h2h", "spreads", "totals"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> str:
    args = parse_args(argv)
    samples = asyncio.run(
        _sample_live_provider_events(
            sports=[str(item) for item in args.sports if str(item).strip()],
            provider_keys=[str(item) for item in args.providers if str(item).strip()],
            regions=[str(item) for item in args.regions if str(item).strip()],
            markets=[str(item) for item in args.markets if str(item).strip()],
        )
    )
    report = build_report_data(samples, scan_mode="live")
    paths = write_report(report, Path(args.output_dir))
    return json.dumps(paths, ensure_ascii=False)


if __name__ == "__main__":
    print(main())

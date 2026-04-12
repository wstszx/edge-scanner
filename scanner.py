"""Core arbitrage scanning logic."""

from __future__ import annotations

import atexit
import asyncio
import copy
import contextvars
import datetime as dt
import concurrent.futures
import difflib
import importlib
import inspect
import itertools
import json
import math
import os
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

try:  # Optional when running scanner standalone
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from settings import apply_config_env

if load_dotenv:
    load_dotenv()
apply_config_env()

from config import (
    DEFAULT_BANKROLL,
    DEFAULT_COMMISSION,
    EXCHANGE_CONFIG_WARNINGS,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIDDLE_SORT,
    DEFAULT_REGION_KEYS,
    DEFAULT_SHARP_BOOK,
    DEFAULT_SPORT_KEYS,
    DEFAULT_STAKE_AMOUNT,
    EDGE_BANDS,
    EXCHANGE_BOOKMAKERS,
    EXCHANGE_KEYS,
    KEY_NUMBER_SPORTS,
    MAX_MIDDLE_PROBABILITY,
    MIN_EDGE_PERCENT,
    MIN_MIDDLE_GAP,
    NFL_KEY_NUMBER_PROBABILITY,
    PROBABILITY_PER_INTEGER,
    REGION_CONFIG,
    ROI_BANDS,
    SHARP_BOOKS,
    SHOW_POSITIVE_EV_ONLY,
    SOFT_BOOK_KEYS,
    SPORT_DISPLAY_NAMES,
    derive_required_regions,
    markets_for_sport,
    normalize_supported_bookmakers,
)
from providers import PROVIDER_FETCHERS, PROVIDER_TITLES, resolve_provider_key

BASE_URL = "https://api.the-odds-api.com/v4"
SCAN_MODE_PREMATCH = "prematch"
SCAN_MODE_LIVE = "live"
DEFAULT_LIVE_PROVIDER_KEYS = (
    "sx_bet",
    "betdex",
    "polymarket",
    "bookmaker_xyz",
    "artline",
)
LIVE_SUPPORTED_PROVIDER_KEYS = DEFAULT_LIVE_PROVIDER_KEYS
MIDDLE_MARKETS = {"spreads", "totals"}
PREMATCH_QUOTE_MAX_AGE_SECONDS_RAW = os.getenv("PREMATCH_QUOTE_MAX_AGE_SECONDS", "7200").strip()
ODDS_API_ALL_MARKETS_RAW = os.getenv("ODDS_API_ALL_MARKETS", "").strip()
ODDS_API_MARKET_BATCH_SIZE_RAW = os.getenv("ODDS_API_MARKET_BATCH_SIZE", "8").strip()
ODDS_API_INVALID_MARKET_STATUS_CODES = {400, 422}
CUSTOM_PROVIDER_SNAPSHOT_ENABLED = os.getenv(
    "CUSTOM_PROVIDER_SNAPSHOT_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
CUSTOM_PROVIDER_SNAPSHOT_DIR = os.getenv(
    "CUSTOM_PROVIDER_SNAPSHOT_DIR",
    str(Path("data") / "provider_snapshots"),
).strip()
CROSS_PROVIDER_MATCH_REPORT_ENABLED = os.getenv(
    "CROSS_PROVIDER_MATCH_REPORT_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
CROSS_PROVIDER_MATCH_REPORT_FILENAME = os.getenv(
    "CROSS_PROVIDER_MATCH_REPORT_FILENAME",
    "cross_provider_match_report.json",
).strip()
CROSS_PROVIDER_MATCH_TOLERANCE_MINUTES_RAW = os.getenv(
    "CROSS_PROVIDER_MATCH_TOLERANCE_MINUTES",
    "180",
).strip()
SCAN_REQUEST_LOG_ENABLED = os.getenv("SCAN_REQUEST_LOG_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SCAN_REQUEST_LOG_DIR = os.getenv(
    "SCAN_REQUEST_LOG_DIR",
    str(Path("data") / "request_logs"),
).strip()
SCAN_REQUEST_LOG_MAX_BODY_CHARS_RAW = os.getenv("SCAN_REQUEST_LOG_MAX_BODY_CHARS", "2000").strip()
SCAN_REQUEST_LOG_RETENTION_FILES_RAW = os.getenv(
    "SCAN_REQUEST_LOG_RETENTION_FILES",
    "20",
).strip()
try:
    SCAN_REQUEST_LOG_MAX_BODY_CHARS = max(0, int(float(SCAN_REQUEST_LOG_MAX_BODY_CHARS_RAW)))
except (TypeError, ValueError):
    SCAN_REQUEST_LOG_MAX_BODY_CHARS = 2000
try:
    SCAN_REQUEST_LOG_RETENTION_FILES = max(
        0,
        int(float(SCAN_REQUEST_LOG_RETENTION_FILES_RAW)),
    )
except (TypeError, ValueError):
    SCAN_REQUEST_LOG_RETENTION_FILES = 20

LIVE_EVENT_MAX_FUTURE_SECONDS_RAW = os.getenv("LIVE_EVENT_MAX_FUTURE_SECONDS", "0").strip()
LIVE_QUOTE_MAX_AGE_SECONDS_RAW = os.getenv("LIVE_QUOTE_MAX_AGE_SECONDS", "60").strip()
LIVE_STATE_CLOCK_TOLERANCE_SECONDS_RAW = os.getenv(
    "LIVE_STATE_CLOCK_TOLERANCE_SECONDS",
    "180",
).strip()

LIVE_STATE_IN_PLAY_TOKENS = {
    "active",
    "in_play",
    "inplay",
    "live",
    "open",
    "running",
    "started",
    "tradeable",
    "trading",
}
LIVE_STATE_NOT_LIVE_TOKENS = {
    "created",
    "interrupted",
    "not_started",
    "pending",
    "paused",
    "pre_play",
    "preplay",
    "scheduled",
    "suspended",
    "upcoming",
}
LIVE_STATE_TERMINAL_TOKENS = {
    "abandoned",
    "canceled",
    "cancelled",
    "closed",
    "complete",
    "completed",
    "ended",
    "expired",
    "final",
    "finished",
    "postponed",
    "resolved",
    "settled",
    "suspended_final",
}

COMMON_EXTRA_MARKETS = [
    "h2h_lay",
    "h2h_3_way",
    "alternate_spreads",
    "alternate_totals",
    "spreads_h1",
    "spreads_h2",
    "totals_h1",
    "totals_h2",
    "spreads_q1",
    "spreads_q2",
    "spreads_q3",
    "spreads_q4",
    "totals_q1",
    "totals_q2",
    "totals_q3",
    "totals_q4",
    "team_totals",
    "team_totals_h1",
    "team_totals_h2",
]

FOOTBALL_EXTRA_MARKETS = [
    "h2h_q1",
    "h2h_q2",
    "h2h_q3",
    "h2h_q4",
    "h2h_h1",
    "h2h_h2",
    "player_pass_tds",
    "player_pass_yds",
    "player_pass_completions",
    "player_pass_attempts",
    "player_pass_interceptions",
    "player_rush_yds",
    "player_rush_attempts",
    "player_receptions",
    "player_reception_yds",
    "player_anytime_td",
    "player_first_td",
]

BASKETBALL_EXTRA_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_turnovers",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
    "player_rebounds_assists",
]

BASEBALL_EXTRA_MARKETS = [
    "pitcher_strikeouts",
    "pitcher_outs",
    "pitcher_hits_allowed",
    "pitcher_walks",
    "pitcher_earned_runs",
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs_scored",
    "batter_home_runs",
]

HOCKEY_EXTRA_MARKETS = [
    "player_points",
    "player_assists",
    "player_goals",
    "player_shots_on_goal",
    "player_blocks",
    "player_power_play_points",
    "player_goal_scorer_anytime",
]

SOCCER_EXTRA_MARKETS = [
    "h2h",
    "h2h_3_way",
    "draw_no_bet",
    "double_chance",
    "both_teams_to_score",
    "team_totals",
]
SOFT_BOOK_KEY_SET = set(SOFT_BOOK_KEYS)
SHARP_BOOK_MAP = {book["key"]: book for book in SHARP_BOOKS}
PROVIDER_FETCH_MAX_WORKERS_RAW = os.getenv("PROVIDER_FETCH_MAX_WORKERS", "8").strip()
SPORT_SCAN_MAX_WORKERS_RAW = os.getenv("SPORT_SCAN_MAX_WORKERS", "4").strip()
PROVIDER_NETWORK_RETRY_ONCE_RAW = os.getenv("PROVIDER_NETWORK_RETRY_ONCE", "1").strip()
PROVIDER_NETWORK_RETRY_DELAY_MS_RAW = os.getenv("PROVIDER_NETWORK_RETRY_DELAY_MS", "250").strip()
PROVIDER_PROXY_MIRROR_DEDUPE = os.getenv("PROVIDER_PROXY_MIRROR_DEDUPE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
EVENT_TIME_TOLERANCE_MINUTES = os.getenv("EVENT_TIME_TOLERANCE_MINUTES", "15").strip()
EVENT_MAX_PAST_MINUTES_RAW = os.getenv("EVENT_MAX_PAST_MINUTES", "30").strip()
EVENT_MATCH_FUZZY_THRESHOLD_RAW = os.getenv("EVENT_MATCH_FUZZY_THRESHOLD", "0.85").strip()
PROXY_PROVIDER_MIRRORS = {}


class ScannerError(Exception):
    """Raised for recoverable scanner issues."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ScanAsyncRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive() and self._loop is not None:
                return
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="scanner-async-runtime",
                daemon=True,
            )
            self._thread.start()
        self._ready.wait(timeout=2.0)
        if self._loop is None:
            raise RuntimeError("Scanner async runtime failed to start")

    def submit(self, coroutine) -> dict:
        self.start()
        loop = self._loop
        if loop is None:
            raise RuntimeError("Scanner async runtime is unavailable")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        if thread and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_seconds)))
        with self._lock:
            self._loop = None
            self._thread = None
            self._ready.clear()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._lock:
            self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
            loop.close()
            with self._lock:
                self._loop = None


_SCAN_ASYNC_RUNTIME = _ScanAsyncRuntime()


def _get_scan_async_runtime() -> _ScanAsyncRuntime:
    return _SCAN_ASYNC_RUNTIME


def shutdown_scan_runtime() -> None:
    _SCAN_ASYNC_RUNTIME.stop()


atexit.register(shutdown_scan_runtime)


def _iso_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _provider_snapshot_filename(provider_key: object) -> str:
    token = re.sub(r"[^a-z0-9._-]+", "_", str(provider_key or "").strip().lower())
    return token or "provider"


def _cleanup_old_request_logs(target_dir: Path, keep_path: Optional[Path] = None) -> None:
    if SCAN_REQUEST_LOG_RETENTION_FILES <= 0:
        return
    try:
        log_paths = sorted(
            target_dir.glob("requests_*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return

    kept = 0
    for path in log_paths:
        if keep_path is not None and path == keep_path:
            kept += 1
            continue
        kept += 1
        if kept <= SCAN_REQUEST_LOG_RETENTION_FILES:
            continue
        try:
            path.unlink()
        except OSError:
            continue


def _cross_provider_match_tolerance_minutes() -> int:
    try:
        return max(0, int(float(CROSS_PROVIDER_MATCH_TOLERANCE_MINUTES_RAW)))
    except (TypeError, ValueError):
        return 180


def _event_match_tolerance_minutes() -> int:
    try:
        return max(0, int(float(EVENT_TIME_TOLERANCE_MINUTES)))
    except (TypeError, ValueError):
        return 15


def _normalize_match_team_token(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(fc|cf|afc|sc)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


SPORT_TEAM_ALIASES: Dict[str, Dict[str, str]] = {
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


def _canonicalize_team_name(value: Optional[str], sport_key: Optional[str] = None) -> str:
    normalized = _normalize_team_name(value)
    if not normalized:
        return ""
    aliases = SPORT_TEAM_ALIASES.get(_normalize_line_component(sport_key)) or {}
    return aliases.get(normalized, normalized)


def _canonicalize_outcome_name(value: object, sport_key: Optional[str] = None) -> str:
    normalized = _normalize_line_component(value)
    if not normalized:
        return ""
    aliases = SPORT_TEAM_ALIASES.get(_normalize_line_component(sport_key)) or {}
    return aliases.get(normalized, normalized)


def _cross_provider_pair_norm(home_team: object, away_team: object) -> str:
    home = _normalize_match_team_token(home_team)
    away = _normalize_match_team_token(away_team)
    if not home or not away:
        return ""
    first, second = sorted((home, away))
    return f"{first} vs {second}"


def _parse_commence_ts(commence_time: object) -> Optional[int]:
    text = str(commence_time or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            parsed = dt.datetime.fromisoformat(text[:-1] + "+00:00")
        else:
            parsed = dt.datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return int(parsed.astimezone(dt.timezone.utc).timestamp())
    except ValueError:
        return None


def _event_market_count(event: dict) -> int:
    bookmakers = event.get("bookmakers") if isinstance(event.get("bookmakers"), list) else []
    total = 0
    for bookmaker in bookmakers:
        if not isinstance(bookmaker, dict):
            continue
        markets = bookmaker.get("markets")
        if not isinstance(markets, list):
            continue
        total += sum(1 for market in markets if isinstance(market, dict))
    return total


def _cross_provider_candidate_reason(
    candidate: Optional[dict],
    *,
    candidate_count: int,
    tolerance_seconds: int,
) -> str:
    if candidate_count <= 0:
        return "no_other_provider_candidates"
    if not isinstance(candidate, dict):
        return "no_close_candidate"
    if candidate.get("same_pair_norm") and not candidate.get("within_tolerance"):
        return "same_pair_time_mismatch"
    pair_similarity = float(candidate.get("pair_similarity", 0.0) or 0.0)
    fuzzy_threshold = _event_match_fuzzy_threshold()
    if candidate.get("within_tolerance") and pair_similarity >= fuzzy_threshold:
        return "name_variation_within_time"
    if pair_similarity >= fuzzy_threshold:
        return "similar_pair_time_mismatch"
    if pair_similarity >= 0.5:
        return "possible_name_mismatch"
    return "no_close_candidate"


def _nearest_cross_provider_candidate(
    source: dict,
    sport_records: Sequence[dict],
    *,
    tolerance_seconds: int,
) -> dict:
    other_records = [
        record
        for record in sport_records
        if isinstance(record, dict)
        and str(record.get("provider") or "").strip()
        and str(record.get("provider") or "").strip() != str(source.get("provider") or "").strip()
    ]
    sport_key = _normalize_line_component(source.get("sport_key"))
    source_home_norm = _canonicalize_team_name(source.get("home_team"), sport_key)
    source_away_norm = _canonicalize_team_name(source.get("away_team"), sport_key)
    source_pair_norm = str(source.get("pair_norm") or "").strip()
    source_ts = source.get("commence_ts")
    source_time = str(source.get("commence_time") or "").strip()

    best_candidate: Optional[dict] = None
    best_rank: Optional[Tuple[int, int, float, float]] = None
    for record in other_records:
        record_sport_key = _normalize_line_component(record.get("sport_key")) or sport_key
        home_norm = _canonicalize_team_name(record.get("home_team"), record_sport_key)
        away_norm = _canonicalize_team_name(record.get("away_team"), record_sport_key)
        if not home_norm and not away_norm:
            continue
        direct_home = _team_similarity(source_home_norm, home_norm)
        direct_away = _team_similarity(source_away_norm, away_norm)
        direct_score = min(direct_home, direct_away)
        reverse_home = _team_similarity(source_home_norm, away_norm)
        reverse_away = _team_similarity(source_away_norm, home_norm)
        reverse_score = min(reverse_home, reverse_away)
        pair_similarity = max(direct_score, reverse_score)

        candidate_ts = record.get("commence_ts")
        time_delta_seconds: Optional[int]
        if source_ts is not None and candidate_ts is not None:
            time_delta_seconds = abs(int(source_ts) - int(candidate_ts))
        elif source_time and source_time == str(record.get("commence_time") or "").strip():
            time_delta_seconds = 0
        else:
            time_delta_seconds = None

        same_pair_norm = bool(source_pair_norm) and source_pair_norm == str(record.get("pair_norm") or "").strip()
        within_tolerance = bool(
            time_delta_seconds is not None and time_delta_seconds <= tolerance_seconds
        )
        rank = (
            1 if same_pair_norm else 0,
            1 if within_tolerance else 0,
            float(pair_similarity),
            -float(time_delta_seconds) if time_delta_seconds is not None else float("-inf"),
        )
        if best_rank is not None and rank <= best_rank:
            continue
        best_rank = rank
        best_candidate = {
            "provider": record.get("provider"),
            "event_id": record.get("event_id"),
            "commence_time": record.get("commence_time"),
            "pair_norm": record.get("pair_norm"),
            "home_team": record.get("home_team"),
            "away_team": record.get("away_team"),
            "markets_count": record.get("markets_count"),
            "same_pair_norm": same_pair_norm,
            "within_tolerance": within_tolerance,
            "time_delta_seconds": time_delta_seconds,
            "time_delta_minutes": (
                round(time_delta_seconds / 60.0, 1)
                if time_delta_seconds is not None
                else None
            ),
            "pair_similarity": round(pair_similarity, 4),
            "pair_similarity_percent": round(pair_similarity * 100.0, 1),
            "orientation": "reversed" if reverse_score > direct_score else "direct",
        }

    reason_code = _cross_provider_candidate_reason(
        best_candidate,
        candidate_count=len(other_records),
        tolerance_seconds=tolerance_seconds,
    )
    return {
        "reason_code": reason_code,
        "candidate_count": len(other_records),
        "closest_candidate": best_candidate,
    }


def _build_cross_provider_match_report(scan_time: str, snapshots: Dict[str, dict]) -> Optional[dict]:
    providers = [str(key) for key in snapshots.keys()]
    provider_event_counts = {provider: 0 for provider in providers}
    records: List[dict] = []
    for provider_key, payload in snapshots.items():
        provider = str(provider_key)
        events = payload.get("events") if isinstance(payload, dict) else []
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            sport_key = _normalize_line_component(event.get("sport_key"))
            home_team = str(event.get("home_team") or "").strip()
            away_team = str(event.get("away_team") or "").strip()
            commence_time = str(event.get("commence_time") or "").strip()
            event_id = str(event.get("id") or event.get("event_id") or "").strip()
            pair_norm = _cross_provider_pair_norm(home_team, away_team)
            record = {
                "provider": provider,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "event_id": event_id,
                "commence_time": commence_time,
                "commence_ts": _parse_commence_ts(commence_time),
                "pair_norm": pair_norm,
                "markets_count": _event_market_count(event),
            }
            records.append(record)
            provider_event_counts[provider] = provider_event_counts.get(provider, 0) + 1

    if not records:
        return {
            "saved_at": scan_time,
            "summary": {
                "providers_considered": providers,
                "provider_event_counts": provider_event_counts,
                "total_raw_records": 0,
                "total_match_clusters": 0,
                "overlap_clusters": 0,
                "clusters_by_provider_count": {},
                "provider_cluster_presence": {},
                "pair_overlap_clusters": {},
                "single_provider_cluster_counts": {},
                "tolerance_minutes": _cross_provider_match_tolerance_minutes(),
                "event_match_tolerance_minutes": _event_match_tolerance_minutes(),
                "event_match_fuzzy_threshold": _event_match_fuzzy_threshold(),
            },
            "clusters": [],
            "single_provider_samples": [],
        }

    grouped: Dict[Tuple[str, str], List[dict]] = {}
    records_by_sport: Dict[str, List[dict]] = {}
    for idx, record in enumerate(records):
        sport_key = record.get("sport_key") or "unknown_sport"
        pair_norm = record.get("pair_norm") or f"__single__{idx}"
        grouped.setdefault((sport_key, pair_norm), []).append(record)
        records_by_sport.setdefault(sport_key, []).append(record)

    tolerance_seconds = _cross_provider_match_tolerance_minutes() * 60
    cluster_id = 0
    clusters: List[dict] = []
    for (sport_key, pair_norm), items in grouped.items():
        ordered = sorted(
            items,
            key=lambda item: (
                item.get("commence_ts") is None,
                item.get("commence_ts") or 0,
                str(item.get("commence_time") or ""),
                str(item.get("provider") or ""),
            ),
        )
        local_clusters: List[dict] = []
        for record in ordered:
            ts = record.get("commence_ts")
            placed = None
            for candidate in local_clusters:
                rep_ts = candidate.get("representative_ts")
                if ts is None or rep_ts is None:
                    if record.get("commence_time") and record.get("commence_time") == candidate.get(
                        "representative_time_utc"
                    ):
                        placed = candidate
                        break
                    continue
                if abs(int(ts) - int(rep_ts)) <= tolerance_seconds:
                    placed = candidate
                    break
            if placed is None:
                cluster_id += 1
                placed = {
                    "cluster_id": cluster_id,
                    "sport_key": sport_key,
                    "pair_norm": pair_norm,
                    "representative_ts": ts,
                    "representative_time_utc": record.get("commence_time") or "",
                    "events": [],
                }
                local_clusters.append(placed)
            if placed.get("representative_ts") is None and ts is not None:
                placed["representative_ts"] = ts
                placed["representative_time_utc"] = record.get("commence_time") or ""
            placed["events"].append(
                {
                    "provider": record.get("provider"),
                    "event_id": record.get("event_id"),
                    "commence_time": record.get("commence_time"),
                    "home_team": record.get("home_team"),
                    "away_team": record.get("away_team"),
                    "markets_count": record.get("markets_count"),
                }
            )
        clusters.extend(local_clusters)

    clusters_by_provider_count: Dict[str, int] = {}
    provider_cluster_presence: Dict[str, int] = {}
    pair_overlap_clusters: Dict[str, int] = {}
    single_provider_cluster_counts: Dict[str, int] = {}

    for cluster in clusters:
        events = cluster.get("events") if isinstance(cluster.get("events"), list) else []
        providers_in_cluster = sorted(
            {
                str(item.get("provider"))
                for item in events
                if isinstance(item, dict) and str(item.get("provider") or "").strip()
            }
        )
        provider_count = len(providers_in_cluster)
        cluster["providers"] = providers_in_cluster
        cluster["provider_count"] = provider_count
        if "representative_ts" in cluster:
            cluster.pop("representative_ts", None)
        events.sort(key=lambda item: (str(item.get("provider") or ""), str(item.get("commence_time") or "")))
        key = str(provider_count)
        clusters_by_provider_count[key] = clusters_by_provider_count.get(key, 0) + 1
        for provider in providers_in_cluster:
            provider_cluster_presence[provider] = provider_cluster_presence.get(provider, 0) + 1
        if provider_count == 1:
            only_provider = providers_in_cluster[0]
            single_provider_cluster_counts[only_provider] = (
                single_provider_cluster_counts.get(only_provider, 0) + 1
            )
        if provider_count >= 2:
            for left, right in itertools.combinations(providers_in_cluster, 2):
                pair_key = f"{left}__{right}"
                pair_overlap_clusters[pair_key] = pair_overlap_clusters.get(pair_key, 0) + 1

    overlap_clusters = [cluster for cluster in clusters if int(cluster.get("provider_count", 0) or 0) >= 2]
    overlap_clusters.sort(
        key=lambda cluster: (
            -int(cluster.get("provider_count", 0) or 0),
            str(cluster.get("sport_key") or ""),
            str(cluster.get("representative_time_utc") or ""),
            str(cluster.get("pair_norm") or ""),
        )
    )

    single_provider_samples: List[dict] = []
    single_provider_reason_counts: Dict[str, int] = {}
    for cluster in clusters:
        if int(cluster.get("provider_count", 0) or 0) != 1:
            continue
        events = cluster.get("events") if isinstance(cluster.get("events"), list) else []
        if not events:
            continue
        first = events[0] if isinstance(events[0], dict) else {}
        source_record = dict(first)
        source_record["pair_norm"] = cluster.get("pair_norm")
        source_record["commence_ts"] = _parse_commence_ts(first.get("commence_time"))
        near_match = _nearest_cross_provider_candidate(
            source_record,
            records_by_sport.get(str(cluster.get("sport_key") or "unknown_sport"), []),
            tolerance_seconds=tolerance_seconds,
        )
        reason_code = str(near_match.get("reason_code") or "no_close_candidate")
        single_provider_reason_counts[reason_code] = (
            single_provider_reason_counts.get(reason_code, 0) + 1
        )
        sample = {
            "cluster_id": cluster.get("cluster_id"),
            "sport_key": cluster.get("sport_key"),
            "pair_norm": cluster.get("pair_norm"),
            "provider": first.get("provider"),
            "event_id": first.get("event_id"),
            "commence_time": first.get("commence_time"),
            "markets_count": first.get("markets_count"),
            "reason_code": reason_code,
        }
        closest_candidate = near_match.get("closest_candidate")
        if isinstance(closest_candidate, dict):
            sample["closest_candidate"] = closest_candidate
        single_provider_samples.append(sample)
        if len(single_provider_samples) >= 120:
            break

    return {
        "saved_at": scan_time,
        "summary": {
            "providers_considered": providers,
            "provider_event_counts": provider_event_counts,
            "total_raw_records": len(records),
            "total_match_clusters": len(clusters),
            "overlap_clusters": len(overlap_clusters),
            "clusters_by_provider_count": clusters_by_provider_count,
            "provider_cluster_presence": provider_cluster_presence,
            "pair_overlap_clusters": pair_overlap_clusters,
            "single_provider_cluster_counts": single_provider_cluster_counts,
            "single_provider_reason_counts": single_provider_reason_counts,
            "tolerance_minutes": _cross_provider_match_tolerance_minutes(),
            "event_match_tolerance_minutes": _event_match_tolerance_minutes(),
            "event_match_fuzzy_threshold": _event_match_fuzzy_threshold(),
        },
        "clusters": overlap_clusters,
        "single_provider_samples": single_provider_samples,
    }


def _persist_cross_provider_match_report(scan_time: str, snapshots: Dict[str, dict]) -> str:
    if not CROSS_PROVIDER_MATCH_REPORT_ENABLED:
        return ""
    if not CUSTOM_PROVIDER_SNAPSHOT_ENABLED:
        return ""
    if not CUSTOM_PROVIDER_SNAPSHOT_DIR:
        return ""
    if not snapshots:
        return ""
    report = _build_cross_provider_match_report(scan_time, snapshots)
    if not isinstance(report, dict):
        return ""
    filename = re.sub(
        r"[^a-z0-9._-]+",
        "_",
        str(CROSS_PROVIDER_MATCH_REPORT_FILENAME or "").strip().lower(),
    )
    if not filename:
        filename = "cross_provider_match_report.json"
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    try:
        target_dir = Path(CUSTOM_PROVIDER_SNAPSHOT_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        tmp_path = path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
        return str(path)
    except OSError:
        return ""


def _persist_provider_snapshots(scan_time: str, snapshots: Dict[str, dict]) -> Dict[str, str]:
    saved_paths: Dict[str, str] = {}
    if not CUSTOM_PROVIDER_SNAPSHOT_ENABLED:
        return saved_paths
    if not CUSTOM_PROVIDER_SNAPSHOT_DIR:
        return saved_paths
    if not snapshots:
        return saved_paths
    try:
        target_dir = Path(CUSTOM_PROVIDER_SNAPSHOT_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return saved_paths

    for provider_key, payload in snapshots.items():
        if not isinstance(payload, dict):
            continue
        path = target_dir / f"{_provider_snapshot_filename(provider_key)}.json"
        tmp_path = path.with_suffix(".json.tmp")
        document = {
            "saved_at": scan_time,
            "provider_key": provider_key,
            "provider_name": payload.get("provider_name")
            or PROVIDER_TITLES.get(str(provider_key), str(provider_key)),
            "sports": payload.get("sports") if isinstance(payload.get("sports"), list) else [],
            "events": payload.get("events") if isinstance(payload.get("events"), list) else [],
        }
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
            tmp_path.replace(path)
            saved_paths[str(provider_key)] = str(path)
        except OSError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            continue
    return saved_paths


_REQUEST_LOG_SENSITIVE_KEYS = {
    "apikey",
    "api_key",
    "authorization",
    "token",
    "secret",
    "password",
    "cookie",
    "xapikey",
    "x_api_key",
    "session",
}
_REQUEST_TRACE_LOCK = threading.RLock()
_REQUEST_TRACE_ACTIVE: List["_ScanRequestLogger"] = []
_REQUEST_TRACE_PATCHED = False
_REQUEST_TRACE_CONTEXT = contextvars.ContextVar("scan_request_logger", default=None)
_REQUESTS_SESSION_REQUEST_ORIGINAL = requests.sessions.Session.request


def _is_sensitive_log_key(key: object) -> bool:
    token = re.sub(r"[^a-z0-9]+", "", str(key or "").strip().lower())
    if not token:
        return False
    if token in _REQUEST_LOG_SENSITIVE_KEYS:
        return True
    return any(part in token for part in ("apikey", "authorization", "token", "secret", "password"))


def _sanitize_for_request_log(value: object, key_hint: Optional[str] = None) -> object:
    if key_hint and _is_sensitive_log_key(key_hint):
        return "***redacted***"
    if isinstance(value, dict):
        return {str(key): _sanitize_for_request_log(item, key_hint=str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_request_log(item, key_hint=key_hint) for item in value]
    if isinstance(value, bytes):
        return _truncate_request_log_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        return _truncate_request_log_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate_request_log_text(str(value))


def _truncate_request_log_text(text: str, limit: Optional[int] = None) -> str:
    capped = SCAN_REQUEST_LOG_MAX_BODY_CHARS if limit is None else max(0, int(limit))
    value = str(text or "")
    if capped <= 0:
        return ""
    if len(value) <= capped:
        return value
    return f"{value[:capped]}...<truncated {len(value) - capped} chars>"


def _sanitize_request_log_url(url: object) -> str:
    raw_url = str(url or "")
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return raw_url
    if not parsed.query:
        return raw_url
    redacted_params = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if _is_sensitive_log_key(key):
            redacted_params.append((key, "***redacted***"))
        else:
            redacted_params.append((key, value))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(redacted_params, doseq=True),
            parsed.fragment,
        )
    )


def _response_preview_for_request_log(response: requests.Response) -> dict:
    content_type = str(response.headers.get("Content-Type", ""))
    body_preview = ""
    size_bytes: Optional[int] = None
    if SCAN_REQUEST_LOG_MAX_BODY_CHARS > 0:
        body_bytes = b""
        try:
            body_bytes = response.content or b""
        except Exception:
            body_bytes = b""
        size_bytes = len(body_bytes)
        lower_ct = content_type.lower()
        if "json" in lower_ct:
            try:
                payload = response.json()
                encoded = json.dumps(
                    _sanitize_for_request_log(payload),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                body_preview = _truncate_request_log_text(encoded)
            except ValueError:
                body_preview = _truncate_request_log_text(response.text)
        elif lower_ct.startswith("text/") or "xml" in lower_ct or "javascript" in lower_ct:
            body_preview = _truncate_request_log_text(response.text)
        else:
            body_preview = f"<binary:{size_bytes or 0} bytes>"
    return {
        "status_code": response.status_code,
        "ok": bool(response.ok),
        "headers": _sanitize_for_request_log(dict(response.headers)),
        "content_type": content_type,
        "size_bytes": size_bytes,
        "body_preview": body_preview,
    }


def _build_request_log_entry(
    method: object,
    url: object,
    kwargs: dict,
    elapsed_ms: float,
    response: Optional[requests.Response] = None,
    error: Optional[str] = None,
) -> dict:
    headers = kwargs.get("headers")
    request_payload = {
        "method": str(method or "").upper() or "GET",
        "url": _sanitize_request_log_url(url),
        "params": _sanitize_for_request_log(kwargs.get("params")),
        "headers": _sanitize_for_request_log(dict(headers)) if isinstance(headers, dict) else _sanitize_for_request_log(headers),
        "json": _sanitize_for_request_log(kwargs.get("json")),
        "data": _sanitize_for_request_log(kwargs.get("data")),
        "timeout": _sanitize_for_request_log(kwargs.get("timeout")),
    }
    entry = {
        "type": "request",
        "time": _iso_now(),
        "elapsed_ms": round(float(elapsed_ms or 0.0), 2),
        "request": request_payload,
    }
    if response is not None:
        entry["response"] = _response_preview_for_request_log(response)
    if error:
        entry["error"] = _truncate_request_log_text(str(error), limit=max(512, SCAN_REQUEST_LOG_MAX_BODY_CHARS))
    return entry


class _ScanRequestLogger:
    def __init__(self, scan_time: str) -> None:
        self.scan_time = scan_time
        self.path = ""
        self.error = ""
        self.requests_logged = 0
        self.owner_thread_id = threading.get_ident()
        self.enabled = SCAN_REQUEST_LOG_ENABLED and bool(SCAN_REQUEST_LOG_DIR)
        self._lock = threading.Lock()
        self._handle = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            target_dir = Path(SCAN_REQUEST_LOG_DIR)
            target_dir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = target_dir / f"requests_{stamp}_{uuid.uuid4().hex[:6]}.jsonl"
            self._handle = path.open("w", encoding="utf-8")
            self.path = str(path)
            self._write(
                {
                    "type": "meta",
                    "scan_time": self.scan_time,
                    "created_at": _iso_now(),
                    "max_body_chars": SCAN_REQUEST_LOG_MAX_BODY_CHARS,
                }
            )
            _cleanup_old_request_logs(target_dir, keep_path=path)
        except OSError as exc:
            self.enabled = False
            self.error = f"Failed to create request log file: {exc}"
            self._handle = None
            self.path = ""

    def _write(self, payload: dict) -> None:
        if not self._handle:
            return
        with self._lock:
            self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._handle.flush()

    def log_request(self, payload: dict) -> None:
        if not self._handle:
            return
        with self._lock:
            self.requests_logged += 1
            enriched = dict(payload)
            enriched["seq"] = self.requests_logged
            self._handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            self._handle.flush()

    def log_meta(self, payload: dict) -> None:
        if not self._handle:
            return
        self._write(payload)

    def close(self) -> None:
        if not self._handle:
            return
        with self._lock:
            try:
                self._handle.write(
                    json.dumps(
                        {
                            "type": "summary",
                            "closed_at": _iso_now(),
                            "requests_logged": self.requests_logged,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self._handle.flush()
            finally:
                try:
                    self._handle.close()
                finally:
                    self._handle = None


def _active_request_loggers() -> List["_ScanRequestLogger"]:
    with _REQUEST_TRACE_LOCK:
        return list(_REQUEST_TRACE_ACTIVE)


def _set_current_request_logger(logger: Optional["_ScanRequestLogger"]) -> None:
    _REQUEST_TRACE_CONTEXT.set(logger)


def _current_request_logger() -> Optional["_ScanRequestLogger"]:
    logger = _REQUEST_TRACE_CONTEXT.get()
    if isinstance(logger, _ScanRequestLogger):
        return logger
    return None


def _select_request_logger(active: Sequence["_ScanRequestLogger"]) -> Optional["_ScanRequestLogger"]:
    current = _current_request_logger()
    if current is not None:
        for logger in active:
            if logger is current:
                return logger
        return None
    if len(active) == 1:
        return active[0]
    return None


def _instrumented_session_request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
    started_at = time.perf_counter()
    response = None
    error = None
    try:
        response = _REQUESTS_SESSION_REQUEST_ORIGINAL(self, method, url, **kwargs)
        return response
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        active = _active_request_loggers()
        logger = _select_request_logger(active)
        if logger is not None:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            try:
                entry = _build_request_log_entry(
                    method=method,
                    url=url,
                    kwargs=kwargs,
                    elapsed_ms=elapsed_ms,
                    response=response,
                    error=error,
                )
                logger.log_request(entry)
            except Exception:
                pass


def _ensure_request_logging_patch() -> None:
    global _REQUEST_TRACE_PATCHED
    with _REQUEST_TRACE_LOCK:
        if _REQUEST_TRACE_PATCHED:
            return
        requests.sessions.Session.request = _instrumented_session_request
        _REQUEST_TRACE_PATCHED = True


def _activate_request_logger(logger: _ScanRequestLogger) -> None:
    _set_current_request_logger(logger)
    if not logger.enabled:
        return
    _ensure_request_logging_patch()
    with _REQUEST_TRACE_LOCK:
        for idx in range(len(_REQUEST_TRACE_ACTIVE) - 1, -1, -1):
            stale = _REQUEST_TRACE_ACTIVE[idx]
            if stale is logger:
                _REQUEST_TRACE_ACTIVE.pop(idx)
        _REQUEST_TRACE_ACTIVE.append(logger)


def _deactivate_request_logger(logger: _ScanRequestLogger) -> None:
    with _REQUEST_TRACE_LOCK:
        for idx in range(len(_REQUEST_TRACE_ACTIVE) - 1, -1, -1):
            if _REQUEST_TRACE_ACTIVE[idx] is logger:
                _REQUEST_TRACE_ACTIVE.pop(idx)
                break
    if _current_request_logger() is logger:
        _set_current_request_logger(None)


def _run_with_request_logger(
    logger: Optional["_ScanRequestLogger"],
    func,
    *args,
    **kwargs,
):
    previous = _current_request_logger()
    _set_current_request_logger(logger)
    try:
        return func(*args, **kwargs)
    finally:
        _set_current_request_logger(previous)


def _submit_with_request_logger(executor, func, *args, **kwargs):
    logger = _current_request_logger()
    return executor.submit(_run_with_request_logger, logger, func, *args, **kwargs)


async def _run_async_with_request_logger(
    logger: Optional["_ScanRequestLogger"],
    func,
    *args,
    **kwargs,
):
    previous = _current_request_logger()
    _set_current_request_logger(logger)
    try:
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
    finally:
        _set_current_request_logger(previous)


async def _call_with_request_logger_async(func, *args, **kwargs):
    logger = _current_request_logger()
    if inspect.iscoroutinefunction(func):
        return await _run_async_with_request_logger(logger, func, *args, **kwargs)
    return await asyncio.to_thread(_run_with_request_logger, logger, func, *args, **kwargs)


async def _await_if_needed(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _attach_request_log_info(result: dict, logger: _ScanRequestLogger) -> dict:
    if not isinstance(result, dict):
        return result
    if logger.path:
        result["request_log"] = {
            "enabled": True,
            "path": logger.path,
            "requests_logged": logger.requests_logged,
        }
        if logger.error:
            result["request_log"]["error"] = logger.error
    elif SCAN_REQUEST_LOG_ENABLED:
        result["request_log"] = {"enabled": False}
        if logger.error:
            result["request_log"]["error"] = logger.error
    return result


def _activate_provider_scan_caches(enabled_provider_keys: Sequence[str]) -> List[object]:
    active_cache_modules: List[object] = []
    seen = set()
    for provider_key in enabled_provider_keys or []:
        normalized_key = str(provider_key).strip().lower()
        if not normalized_key or normalized_key in seen:
            continue
        seen.add(normalized_key)
        try:
            module = importlib.import_module(f"providers.{normalized_key}")
            enable = getattr(module, "enable_scan_cache", None)
            if not callable(enable):
                continue
            enable()
            active_cache_modules.append(module)
        except Exception:
            continue
    return active_cache_modules


def _deactivate_provider_scan_caches(active_cache_modules: Sequence[object]) -> None:
    for module in active_cache_modules or []:
        try:
            disable = getattr(module, "disable_scan_cache", None)
            if callable(disable):
                disable()
        except Exception:
            continue


def _normalize_provider_keys(provider_keys: Optional[Sequence[str]]) -> Optional[List[str]]:
    if provider_keys is None:
        return None
    normalized: List[str] = []
    seen = set()
    for value in provider_keys:
        key = resolve_provider_key(value)
        if not key or key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized


def _dedupe_proxy_provider_keys(
    enabled_provider_keys: Sequence[str],
    explicit_provider_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    ordered = [str(key) for key in enabled_provider_keys if str(key) in PROVIDER_FETCHERS]
    if not PROVIDER_PROXY_MIRROR_DEDUPE:
        deduped: List[str] = []
        seen = set()
        for key in ordered:
            if key in seen:
                continue
            deduped.append(key)
            seen.add(key)
        return deduped

    explicit_set = {
        str(key)
        for key in (explicit_provider_keys or [])
        if isinstance(key, str)
    }
    enabled_set = set(ordered)
    for mirror_key, upstream_key in PROXY_PROVIDER_MIRRORS.items():
        if mirror_key not in enabled_set:
            continue
        if upstream_key not in enabled_set:
            continue
        if mirror_key in explicit_set:
            continue
        enabled_set.remove(mirror_key)

    deduped: List[str] = []
    seen = set()
    for key in ordered:
        if key not in enabled_set or key in seen:
            continue
        deduped.append(key)
        seen.add(key)
    return deduped


def _resolve_enabled_provider_keys(
    include_providers: Optional[Sequence[str]],
) -> List[str]:
    enabled_by_key = {
        key: False
        for key in PROVIDER_FETCHERS
    }
    explicit_providers = _normalize_provider_keys(include_providers) or []
    for key in explicit_providers:
        enabled_by_key[key] = True
    return [key for key in PROVIDER_FETCHERS if enabled_by_key.get(key)]


def _default_live_provider_keys() -> List[str]:
    defaults = [key for key in DEFAULT_LIVE_PROVIDER_KEYS if key in PROVIDER_FETCHERS]
    if defaults:
        return defaults
    supported = [key for key in LIVE_SUPPORTED_PROVIDER_KEYS if key in PROVIDER_FETCHERS]
    if supported:
        return supported
    return list(PROVIDER_FETCHERS.keys())


def _empty_provider_summary(provider_key: str, enabled: bool) -> dict:
    return {
        "key": provider_key,
        "name": PROVIDER_TITLES.get(provider_key, provider_key),
        "enabled": enabled,
        "events_merged": 0,
        "sports": [],
    }


def _clamp_commission(rate: Optional[float]) -> float:
    if rate is None:
        return DEFAULT_COMMISSION
    return max(0.0, min(rate, 0.2))


def _parse_token_list(raw: str) -> List[str]:
    if not raw:
        return []
    if raw.startswith("["):
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if isinstance(payload, list):
            out = []
            seen = set()
            for item in payload:
                token = str(item).strip()
                if token and token not in seen:
                    out.append(token)
                    seen.add(token)
            return out
    out = []
    seen = set()
    for token in re.split(r"[,\s]+", raw):
        cleaned = token.strip()
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return out


def _odds_api_market_batch_size() -> int:
    try:
        return max(1, int(float(ODDS_API_MARKET_BATCH_SIZE_RAW)))
    except (TypeError, ValueError):
        return 8


def _default_all_markets_for_sport(sport_key: str) -> List[str]:
    if sport_key.startswith("americanfootball_"):
        return COMMON_EXTRA_MARKETS + FOOTBALL_EXTRA_MARKETS
    if sport_key.startswith("basketball_"):
        return COMMON_EXTRA_MARKETS + BASKETBALL_EXTRA_MARKETS
    if sport_key.startswith("baseball_"):
        return COMMON_EXTRA_MARKETS + BASEBALL_EXTRA_MARKETS
    if sport_key.startswith("icehockey_"):
        return COMMON_EXTRA_MARKETS + HOCKEY_EXTRA_MARKETS
    if sport_key.startswith("soccer_"):
        return COMMON_EXTRA_MARKETS + SOCCER_EXTRA_MARKETS
    return COMMON_EXTRA_MARKETS.copy()


def _requested_api_markets(
    sport_key: str,
    base_markets: Sequence[str],
    all_markets: bool,
) -> List[str]:
    requested: List[str] = []
    seen = set()
    for market in base_markets or []:
        key = str(market).strip()
        if key and key not in seen:
            requested.append(key)
            seen.add(key)
    if not all_markets:
        return requested

    configured = _parse_token_list(ODDS_API_ALL_MARKETS_RAW)
    extras = configured or _default_all_markets_for_sport(sport_key)
    for market in extras:
        key = str(market).strip()
        if key and key not in seen:
            requested.append(key)
            seen.add(key)

    # Soccer three-way markets are usually omitted in base defaults; include when full scan is enabled.
    if sport_key.startswith("soccer_") and "h2h_3_way" not in seen:
        requested.append("h2h_3_way")
    return requested


def _normalize_requested_sport_keys(sports: Optional[Sequence[str]]) -> List[str]:
    source = sports or DEFAULT_SPORT_KEYS
    normalized: List[str] = []
    seen = set()
    for item in source:
        key = _normalize_line_component(item)
        if not key or key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized or list(DEFAULT_SPORT_KEYS)


def _normalize_scan_mode(value: object) -> str:
    text = _normalize_line_component(value)
    if text in {SCAN_MODE_PREMATCH, SCAN_MODE_LIVE}:
        return text
    return SCAN_MODE_PREMATCH


def _provider_requested_markets(
    sport_key: str,
    requested_markets: Sequence[str],
    all_markets: bool,
) -> List[str]:
    merged: List[str] = []
    seen = set()
    for market in requested_markets or []:
        key = str(market).strip()
        if not key or key in seen:
            continue
        merged.append(key)
        seen.add(key)
    if sport_key.startswith("soccer_"):
        for extra in ("h2h", "h2h_3_way"):
            if extra not in seen:
                merged.append(extra)
                seen.add(extra)
        if all_markets:
            for extra in ("both_teams_to_score",):
                if extra not in seen:
                    merged.append(extra)
                    seen.add(extra)
    return merged


def _sport_stub(sport_key: str) -> dict:
    return {
        "key": sport_key,
        "title": SPORT_DISPLAY_NAMES.get(sport_key, sport_key),
        "active": True,
    }


def _chunked(values: Sequence[str], size: int) -> List[List[str]]:
    if size <= 0:
        size = 1
    out: List[List[str]] = []
    for index in range(0, len(values), size):
        out.append(list(values[index : index + size]))
    return out


def _normalize_line_component(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _outcome_description_token(outcome: dict) -> str:
    for key in ("description", "participant", "player", "label"):
        token = _normalize_line_component(outcome.get(key))
        if token:
            return token
    return ""


def _safe_point_key(value: object) -> Optional[object]:
    if value is None:
        return None
    numeric = _safe_float(value)
    if numeric is None:
        token = _normalize_line_component(value)
        return token or None
    return round(float(numeric), 6)


def _outcome_signature(outcome: dict) -> Tuple[str, str, Optional[object]]:
    return (
        _normalize_line_component(outcome.get("name")),
        _outcome_description_token(outcome),
        _safe_point_key(outcome.get("point")),
    )


def _merge_market_outcomes(target_market: dict, incoming_market: dict) -> None:
    target_outcomes = target_market.get("outcomes")
    incoming_outcomes = incoming_market.get("outcomes")
    if not isinstance(incoming_outcomes, list):
        return
    if not isinstance(target_outcomes, list):
        target_market["outcomes"] = [item for item in incoming_outcomes if isinstance(item, dict)]
        return

    index: Dict[Tuple[str, str, Optional[object]], int] = {}
    for idx, outcome in enumerate(target_outcomes):
        if not isinstance(outcome, dict):
            continue
        index[_outcome_signature(outcome)] = idx

    for outcome in incoming_outcomes:
        if not isinstance(outcome, dict):
            continue
        signature = _outcome_signature(outcome)
        existing_index = index.get(signature)
        if existing_index is None:
            target_outcomes.append(outcome)
            index[signature] = len(target_outcomes) - 1
            continue
        existing = target_outcomes[existing_index]
        if not isinstance(existing, dict):
            target_outcomes[existing_index] = outcome
            continue
        existing_price = _safe_float(existing.get("price"))
        incoming_price = _safe_float(outcome.get("price"))
        if incoming_price is not None and (
            existing_price is None or incoming_price > existing_price
        ):
            merged = dict(existing)
            merged.update(outcome)
            target_outcomes[existing_index] = merged
            continue
        merged = dict(outcome)
        merged.update(existing)
        target_outcomes[existing_index] = merged


def _merge_bookmaker_markets(target_book: dict, incoming_book: dict) -> None:
    target_markets = target_book.get("markets")
    incoming_markets = incoming_book.get("markets")
    if not isinstance(incoming_markets, list):
        return
    if not isinstance(target_markets, list):
        target_book["markets"] = [item for item in incoming_markets if isinstance(item, dict)]
        return

    market_index: Dict[str, dict] = {}
    for market in target_markets:
        if not isinstance(market, dict):
            continue
        key = _normalize_line_component(market.get("key"))
        if key and key not in market_index:
            market_index[key] = market

    for market in incoming_markets:
        if not isinstance(market, dict):
            continue
        key = _normalize_line_component(market.get("key"))
        if not key:
            target_markets.append(market)
            continue
        existing = market_index.get(key)
        if existing is None:
            target_markets.append(market)
            market_index[key] = market
            continue
        _merge_market_outcomes(existing, market)


def _merge_event_bookmakers(target_event: dict, incoming_event: dict) -> None:
    target_books = target_event.get("bookmakers")
    incoming_books = incoming_event.get("bookmakers")
    if not isinstance(incoming_books, list):
        return
    if not isinstance(target_books, list):
        target_event["bookmakers"] = [item for item in incoming_books if isinstance(item, dict)]
        return

    book_index: Dict[str, dict] = {}
    for book in target_books:
        if not isinstance(book, dict):
            continue
        key = _normalize_line_component(book.get("key") or book.get("title"))
        if key and key not in book_index:
            book_index[key] = book

    for book in incoming_books:
        if not isinstance(book, dict):
            continue
        key = _normalize_line_component(book.get("key") or book.get("title"))
        if not key:
            target_books.append(book)
            continue
        existing = book_index.get(key)
        if existing is None:
            target_books.append(book)
            book_index[key] = book
            continue
        _merge_bookmaker_markets(existing, book)


def _event_merge_key(event: dict) -> Optional[Tuple[str, str]]:
    event_id = str(event.get("id") or "").strip()
    if event_id:
        return ("id", event_id)
    sport_key = _normalize_line_component(event.get("sport_key"))
    home = _normalize_line_component(event.get("home_team"))
    away = _normalize_line_component(event.get("away_team"))
    commence = str(event.get("commence_time") or "").strip()
    if sport_key and home and away and commence:
        return ("fixture", f"{sport_key}|{home}|{away}|{commence}")
    return None


def _merge_odds_event_lists(target: List[dict], incoming: List[dict]) -> List[dict]:
    if not incoming:
        return target
    if not target:
        return list(incoming)
    event_index: Dict[Tuple[str, str], dict] = {}
    for event in target:
        if not isinstance(event, dict):
            continue
        key = _event_merge_key(event)
        if key and key not in event_index:
            event_index[key] = event

    for event in incoming:
        if not isinstance(event, dict):
            continue
        key = _event_merge_key(event)
        if key is None:
            target.append(event)
            continue
        existing = event_index.get(key)
        if existing is None:
            target.append(event)
            event_index[key] = event
            continue
        _merge_event_bookmakers(existing, event)
    return target


def _event_match_fuzzy_threshold() -> float:
    try:
        return max(0.0, min(float(EVENT_MATCH_FUZZY_THRESHOLD_RAW), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _provider_fetch_max_workers() -> int:
    try:
        return max(1, int(float(PROVIDER_FETCH_MAX_WORKERS_RAW)))
    except (TypeError, ValueError):
        return 8


def _sport_scan_max_workers() -> int:
    try:
        return max(1, int(float(SPORT_SCAN_MAX_WORKERS_RAW)))
    except (TypeError, ValueError):
        return 4


def _provider_network_retry_once_enabled() -> bool:
    return str(PROVIDER_NETWORK_RETRY_ONCE_RAW or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _provider_network_retry_delay_seconds() -> float:
    try:
        return max(0.0, float(PROVIDER_NETWORK_RETRY_DELAY_MS_RAW) / 1000.0)
    except (TypeError, ValueError):
        return 0.25


def _is_transient_provider_network_error(message: object) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    indicators = (
        "connection reset",
        "connectionreseterror",
        "connection aborted",
        "remote host",
        "read timed out",
        "timed out",
        "temporarily unavailable",
    )
    return any(token in text for token in indicators)


def _normalize_regions(regions: Optional[Sequence[str]]) -> List[str]:
    if not regions:
        return list(DEFAULT_REGION_KEYS)
    valid = []
    seen = set()
    for region in regions:
        if region not in REGION_CONFIG or region in seen:
            continue
        valid.append(region)
        seen.add(region)
    return valid or list(DEFAULT_REGION_KEYS)


def _normalize_bookmakers(bookmakers: Optional[Sequence[str]]) -> List[str]:
    return normalize_supported_bookmakers(list(bookmakers) if bookmakers else [])


def _normalize_api_keys(api_key: Optional[Sequence[str] | str]) -> List[str]:
    if not api_key:
        return []
    if isinstance(api_key, str):
        raw_keys = [item.strip() for item in re.split(r"[,\s]+", api_key) if item.strip()]
    else:
        raw_keys = []
        for key in api_key:
            if not isinstance(key, str):
                continue
            cleaned = key.strip()
            if cleaned:
                raw_keys.append(cleaned)
    normalized = []
    seen = set()
    for key in raw_keys:
        if key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized


def _event_team_key(event: dict) -> Optional[Tuple[str, str, str]]:
    sport = (event.get("sport_key") or "").strip().lower()
    home = (event.get("home_team") or "").strip().lower()
    away = (event.get("away_team") or "").strip().lower()
    if not (sport and home and away):
        return None
    return (sport, home, away)


def _event_team_key_normalized(event: dict) -> Optional[Tuple[str, str, str]]:
    sport = (event.get("sport_key") or "").strip().lower()
    home = _canonicalize_team_name(event.get("home_team"), sport)
    away = _canonicalize_team_name(event.get("away_team"), sport)
    if not (sport and home and away):
        return None
    return (sport, home, away)


def _event_identity(event: dict) -> Optional[Tuple[str, str, str, str]]:
    key = _event_team_key(event)
    if not key:
        return None
    commence = _normalize_commence_time(event.get("commence_time"))
    if not commence:
        return None
    return (*key, commence)


def _event_time_seconds(event: dict) -> Optional[int]:
    commence = _normalize_commence_time(event.get("commence_time"))
    if not commence:
        return None
    try:
        if commence.endswith("Z"):
            parsed = dt.datetime.fromisoformat(commence[:-1] + "+00:00")
        else:
            parsed = dt.datetime.fromisoformat(commence)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp())


def _live_quote_max_age_seconds() -> int:
    try:
        return max(0, int(float(LIVE_QUOTE_MAX_AGE_SECONDS_RAW)))
    except (TypeError, ValueError):
        return 60


def _event_live_state(event: dict) -> dict:
    payload = event.get("live_state")
    return payload if isinstance(payload, dict) else {}


def _normalize_live_state_token(value: object) -> str:
    text = _normalize_text(str(value) if value is not None else "")
    return re.sub(r"[\s-]+", "_", text)


def _live_state_tokens(state: Optional[dict]) -> List[str]:
    if not isinstance(state, dict):
        return []
    tokens: List[str] = []
    seen = set()
    for key in (
        "status",
        "game_status",
        "gameState",
        "state",
        "in_play_status",
        "market_status",
        "provider_status",
    ):
        token = _normalize_live_state_token(state.get(key))
        if token and token not in seen:
            tokens.append(token)
            seen.add(token)
    return tokens


def _live_state_token(state: Optional[dict]) -> str:
    tokens = _live_state_tokens(state)
    return tokens[0] if tokens else ""


def _event_live_state_token(event: dict) -> str:
    return _live_state_token(_event_live_state(event))


def _live_state_is_explicitly_live(state: Optional[dict]) -> Optional[bool]:
    if not isinstance(state, dict):
        return None
    tokens = _live_state_tokens(state)
    if any(token in LIVE_STATE_NOT_LIVE_TOKENS or token in LIVE_STATE_TERMINAL_TOKENS for token in tokens):
        return False
    explicit_true = False
    for key in ("is_live", "live", "is_in_play", "in_play"):
        if key in state:
            if not bool(state.get(key)):
                return False
            explicit_true = True
    if explicit_true:
        return True
    if any(token in LIVE_STATE_IN_PLAY_TOKENS for token in tokens):
        return True
    return None


def _event_is_explicitly_live(event: dict) -> Optional[bool]:
    if not isinstance(event, dict):
        return None

    saw_explicit_false = False
    event_live_state = _event_live_state(event)
    explicit_event_state = _live_state_is_explicitly_live(event_live_state)
    if explicit_event_state is True:
        return True
    if explicit_event_state is False:
        saw_explicit_false = True

    bookmakers = event.get("bookmakers")
    if isinstance(bookmakers, list):
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                continue
            bookmaker_live_state = bookmaker.get("live_state")
            explicit_bookmaker_state = _live_state_is_explicitly_live(bookmaker_live_state)
            if explicit_bookmaker_state is True:
                return True
            if explicit_bookmaker_state is False:
                saw_explicit_false = True

    if saw_explicit_false:
        return False
    return None


def _event_max_past_seconds() -> int:
    try:
        minutes = int(float(EVENT_MAX_PAST_MINUTES_RAW))
    except (TypeError, ValueError):
        minutes = 30
    return max(0, minutes) * 60


def _live_event_max_future_seconds() -> int:
    try:
        return max(0, int(float(LIVE_EVENT_MAX_FUTURE_SECONDS_RAW)))
    except (TypeError, ValueError):
        return 0


def _filter_events_for_scan(events: Sequence[dict]) -> Tuple[List[dict], dict]:
    now_epoch = int(time.time())
    max_past_seconds = _event_max_past_seconds()
    min_epoch = now_epoch - max_past_seconds
    filtered: List[dict] = []
    dropped_past = 0
    dropped_missing_time = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        event_epoch = _event_time_seconds(event)
        if event_epoch is None:
            dropped_missing_time += 1
            continue
        if event_epoch < min_epoch:
            dropped_past += 1
            continue
        filtered.append(event)
    return filtered, {
        "dropped_past": dropped_past,
        "dropped_missing_time": dropped_missing_time,
    }


def _filter_live_events_for_scan(events: Sequence[dict]) -> Tuple[List[dict], dict]:
    now_epoch = int(time.time())
    max_past_seconds = _event_max_past_seconds()
    max_future_seconds = _live_event_max_future_seconds()
    explicit_live_future_seconds = max(max_future_seconds, max_past_seconds)
    min_epoch = now_epoch - max_past_seconds
    max_epoch = now_epoch + max_future_seconds
    filtered: List[dict] = []
    dropped_past = 0
    dropped_future = 0
    dropped_missing_time = 0
    dropped_not_live_state = 0
    dropped_terminal_state = 0
    suspicious_explicit_live_future = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        event_epoch = _event_time_seconds(event)
        explicit_live = _event_is_explicitly_live(event)
        if explicit_live is not None:
            if explicit_live:
                if event_epoch is not None:
                    if event_epoch > now_epoch + explicit_live_future_seconds:
                        suspicious_explicit_live_future += 1
                filtered.append(event)
                continue
            state_token = _event_live_state_token(event)
            if state_token in LIVE_STATE_TERMINAL_TOKENS:
                dropped_terminal_state += 1
            else:
                dropped_not_live_state += 1
            continue
        if event_epoch is None:
            dropped_missing_time += 1
            continue
        if event_epoch < min_epoch:
            dropped_past += 1
            continue
        if event_epoch > max_epoch:
            dropped_future += 1
            continue
        filtered.append(event)
    return filtered, {
        "dropped_past": dropped_past,
        "dropped_future": dropped_future,
        "dropped_missing_time": dropped_missing_time,
        "dropped_not_live_state": dropped_not_live_state,
        "dropped_terminal_state": dropped_terminal_state,
        "suspicious_explicit_live_future": suspicious_explicit_live_future,
    }


def _filter_events_for_scan_mode(events: Sequence[dict], scan_mode: str) -> Tuple[List[dict], dict]:
    if _normalize_scan_mode(scan_mode) == SCAN_MODE_LIVE:
        return _filter_live_events_for_scan(events)
    return _filter_events_for_scan(events)


def _merge_bookmakers(target: List[dict], incoming: List[dict]) -> None:
    book_index: Dict[str, dict] = {}
    for book in target:
        if not isinstance(book, dict):
            continue
        key = _normalize_line_component(book.get("key") or book.get("title"))
        if key and key not in book_index:
            book_index[key] = book
    for book in incoming:
        if not isinstance(book, dict):
            continue
        key = _normalize_line_component(book.get("key") or book.get("title"))
        if not key:
            target.append(book)
            continue
        existing = book_index.get(key)
        if existing is None:
            target.append(book)
            book_index[key] = book
            continue
        _merge_bookmaker_markets(existing, book)


def _index_event_for_merge(
    event: dict,
    index: Dict[Tuple[str, str, str, str], dict],
    by_team: Dict[Tuple[str, str, str], List[Tuple[int, dict]]],
    normalized_by_sport: Dict[str, List[Tuple[int, dict, str, str]]],
) -> None:
    identity = _event_identity(event)
    if identity and identity not in index:
        index[identity] = event

    epoch = _event_time_seconds(event)
    if epoch is None:
        return

    team_key = _event_team_key(event)
    if team_key:
        by_team.setdefault(team_key, []).append((epoch, event))

    normalized_key = _event_team_key_normalized(event)
    if normalized_key:
        sport, home_norm, away_norm = normalized_key
        normalized_by_sport.setdefault(sport, []).append((epoch, event, home_norm, away_norm))


def _merge_events(base_events: List[dict], extra_events: List[dict]) -> List[dict]:
    return _merge_events_with_stats(base_events, extra_events)


def _empty_event_merge_stats() -> dict:
    return {
        "incoming_events": 0,
        "matched_existing": 0,
        "matched_identity": 0,
        "matched_team": 0,
        "matched_reverse_team": 0,
        "matched_fuzzy": 0,
        "appended_new": 0,
    }


def _merge_events_with_stats(
    base_events: List[dict],
    extra_events: List[dict],
    stats: Optional[dict] = None,
) -> List[dict]:
    index: Dict[Tuple[str, str, str, str], dict] = {}
    by_team: Dict[Tuple[str, str, str], List[Tuple[int, dict]]] = {}
    normalized_by_sport: Dict[str, List[Tuple[int, dict, str, str]]] = {}
    tolerance_minutes = _event_match_tolerance_minutes()
    tolerance_seconds = max(0, tolerance_minutes) * 60
    fuzzy_threshold = _event_match_fuzzy_threshold()
    if isinstance(stats, dict):
        for key, default in _empty_event_merge_stats().items():
            stats[key] = int(stats.get(key, default) or 0)
        stats["incoming_events"] += len(extra_events)
    for event in base_events:
        _index_event_for_merge(event, index, by_team, normalized_by_sport)
    for extra in extra_events:
        identity = _event_identity(extra)
        match_reason = ""
        if identity and identity in index:
            base = index[identity]
            base_books = base.setdefault("bookmakers", [])
            extra_books = extra.get("bookmakers") or []
            if isinstance(base_books, list) and isinstance(extra_books, list):
                _merge_bookmakers(base_books, extra_books)
            match_reason = "matched_identity"
            if isinstance(stats, dict):
                stats["matched_existing"] += 1
                stats[match_reason] += 1
            continue
        matched_event = None
        if tolerance_seconds > 0:
            team_key = _event_team_key(extra)
            if team_key and team_key in by_team:
                extra_epoch = _event_time_seconds(extra)
                if extra_epoch is not None:
                    best_diff = None
                    for base_epoch, base_event in by_team[team_key]:
                        diff = abs(base_epoch - extra_epoch)
                        if diff <= tolerance_seconds and (best_diff is None or diff < best_diff):
                            best_diff = diff
                            matched_event = base_event
                            match_reason = "matched_team"
            # Some providers flip home/away labels for the same fixture.
            if matched_event is None and team_key:
                reverse_team_key = (team_key[0], team_key[2], team_key[1])
                if reverse_team_key in by_team:
                    extra_epoch = _event_time_seconds(extra)
                    if extra_epoch is not None:
                        best_diff = None
                        for base_epoch, base_event in by_team[reverse_team_key]:
                            diff = abs(base_epoch - extra_epoch)
                            if diff <= tolerance_seconds and (best_diff is None or diff < best_diff):
                                best_diff = diff
                                matched_event = base_event
                                match_reason = "matched_reverse_team"
        if matched_event is None and tolerance_seconds > 0 and fuzzy_threshold > 0:
            normalized_key = _event_team_key_normalized(extra)
            extra_epoch = _event_time_seconds(extra)
            if normalized_key and extra_epoch is not None:
                sport, home_norm, away_norm = normalized_key
                best_score = None
                for base_epoch, base_event, base_home, base_away in normalized_by_sport.get(
                    sport, []
                ):
                    if abs(base_epoch - extra_epoch) > tolerance_seconds:
                        continue
                    # Evaluate both orientations to tolerate provider home/away flips.
                    score_home = _team_similarity(home_norm, base_home)
                    score_away = _team_similarity(away_norm, base_away)
                    direct_score = min(score_home, score_away)
                    reverse_home = _team_similarity(home_norm, base_away)
                    reverse_away = _team_similarity(away_norm, base_home)
                    reverse_score = min(reverse_home, reverse_away)
                    score = max(direct_score, reverse_score)
                    if score < fuzzy_threshold:
                        continue
                    if reverse_score > direct_score:
                        avg_score = (reverse_home + reverse_away) / 2.0
                    else:
                        avg_score = (score_home + score_away) / 2.0
                    if best_score is None or avg_score > best_score:
                        best_score = avg_score
                        matched_event = base_event
                        match_reason = "matched_fuzzy"
        if matched_event is not None:
            base_books = matched_event.setdefault("bookmakers", [])
            extra_books = extra.get("bookmakers") or []
            if isinstance(base_books, list) and isinstance(extra_books, list):
                _merge_bookmakers(base_books, extra_books)
            if isinstance(stats, dict):
                stats["matched_existing"] += 1
                if match_reason in stats:
                    stats[match_reason] += 1
            continue
        base_events.append(extra)
        _index_event_for_merge(extra, index, by_team, normalized_by_sport)
        if isinstance(stats, dict):
            stats["appended_new"] += 1
    return base_events


def _sum_stale_filter_counts(items: Sequence[dict]) -> int:
    total = 0
    for entry in items or []:
        if not isinstance(entry, dict):
            continue
        for key in (
            "dropped_past",
            "dropped_future",
            "dropped_missing_time",
            "suspicious_explicit_live_future",
        ):
            try:
                total += int(entry.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
    return total


def _build_scan_diagnostics(
    *,
    provider_summaries: Dict[str, dict],
    cross_provider_report: Optional[dict],
    events_scanned: int,
    arbitrage_count: int,
    positive_arbitrage_count: int,
    middle_count: int,
    positive_middle_count: int,
    plus_ev_count: int,
    sport_errors: Sequence[dict],
    stale_event_filters: Sequence[dict],
) -> dict:
    report_summary = (
        cross_provider_report.get("summary")
        if isinstance(cross_provider_report, dict)
        and isinstance(cross_provider_report.get("summary"), dict)
        else {}
    )
    total_raw_records = int(report_summary.get("total_raw_records", 0) or 0)
    total_match_clusters = int(report_summary.get("total_match_clusters", 0) or 0)
    overlap_clusters = int(report_summary.get("overlap_clusters", 0) or 0)
    provider_breakdown: List[dict] = []
    enabled_provider_count = 0
    providers_with_events = 0
    providers_with_errors = 0
    providers_with_match_hits = 0
    total_merge_hits = 0
    total_fuzzy_matches = 0
    total_new_events = 0

    for provider_key, provider_summary in (provider_summaries or {}).items():
        if not isinstance(provider_summary, dict):
            continue
        sports = provider_summary.get("sports")
        if not isinstance(sports, list):
            sports = []
        enabled = bool(provider_summary.get("enabled"))
        raw_events = 0
        error_count = 0
        matched_existing = 0
        matched_identity = 0
        matched_team = 0
        matched_reverse_team = 0
        matched_fuzzy = 0
        appended_new = 0
        for sport_entry in sports:
            if not isinstance(sport_entry, dict):
                continue
            raw_events += int(sport_entry.get("events_returned", 0) or 0)
            if sport_entry.get("error"):
                error_count += 1
            merge_stats = sport_entry.get("merge_stats")
            if not isinstance(merge_stats, dict):
                merge_stats = {}
            matched_existing += int(merge_stats.get("matched_existing", 0) or 0)
            matched_identity += int(merge_stats.get("matched_identity", 0) or 0)
            matched_team += int(merge_stats.get("matched_team", 0) or 0)
            matched_reverse_team += int(merge_stats.get("matched_reverse_team", 0) or 0)
            matched_fuzzy += int(merge_stats.get("matched_fuzzy", 0) or 0)
            appended_new += int(merge_stats.get("appended_new", 0) or 0)
        if enabled:
            enabled_provider_count += 1
        if raw_events > 0:
            providers_with_events += 1
        if error_count > 0:
            providers_with_errors += 1
        if matched_existing > 0:
            providers_with_match_hits += 1
        total_merge_hits += matched_existing
        total_fuzzy_matches += matched_fuzzy
        total_new_events += appended_new
        if enabled or raw_events > 0 or error_count > 0:
            provider_breakdown.append(
                {
                    "provider_key": provider_key,
                    "provider_name": provider_summary.get("name")
                    or PROVIDER_TITLES.get(provider_key, provider_key),
                    "enabled": enabled,
                    "raw_events": raw_events,
                    "events_merged": int(provider_summary.get("events_merged", 0) or 0),
                    "matched_existing": matched_existing,
                    "matched_identity": matched_identity,
                    "matched_team": matched_team,
                    "matched_reverse_team": matched_reverse_team,
                    "matched_fuzzy": matched_fuzzy,
                    "appended_new": appended_new,
                    "error_count": error_count,
                    "sport_count": len(sports),
                }
            )

    provider_breakdown.sort(
        key=lambda item: (
            -int(item.get("raw_events", 0) or 0),
            -int(item.get("matched_existing", 0) or 0),
            str(item.get("provider_name") or item.get("provider_key") or ""),
        )
    )
    stale_filter_drop_total = _sum_stale_filter_counts(stale_event_filters)
    sport_error_count = len([item for item in sport_errors or [] if isinstance(item, dict)])
    if positive_arbitrage_count > 0:
        reason_code = "arbitrage_found"
    elif positive_middle_count > 0:
        reason_code = "positive_middle_found"
    elif sport_error_count and events_scanned == 0 and total_raw_records == 0:
        reason_code = "fetch_errors"
    elif total_raw_records == 0 and enabled_provider_count > 0:
        reason_code = "no_source_events"
    elif stale_filter_drop_total > 0 and events_scanned == 0:
        reason_code = "events_filtered_by_time"
    elif enabled_provider_count >= 2 and total_raw_records > 0 and overlap_clusters == 0:
        reason_code = "no_cross_provider_overlap"
    elif total_raw_records > 0 and total_merge_hits == 0 and enabled_provider_count >= 2:
        reason_code = "low_merge_overlap"
    elif sport_error_count > 0:
        reason_code = "partial_errors"
    elif events_scanned > 0 and (overlap_clusters > 0 or total_merge_hits > 0):
        reason_code = "matched_but_no_arbitrage"
    else:
        reason_code = "no_arbitrage_after_merge"

    return {
        "reason_code": reason_code,
        "enabled_provider_count": enabled_provider_count,
        "providers_with_events": providers_with_events,
        "providers_with_errors": providers_with_errors,
        "providers_with_match_hits": providers_with_match_hits,
        "raw_provider_events": total_raw_records,
        "merged_events_scanned": int(events_scanned or 0),
        "total_match_clusters": total_match_clusters,
        "overlap_clusters": overlap_clusters,
        "single_provider_clusters": max(0, total_match_clusters - overlap_clusters),
        "total_merge_hits": total_merge_hits,
        "total_fuzzy_matches": total_fuzzy_matches,
        "total_new_events": total_new_events,
        "arbitrage_count": int(arbitrage_count or 0),
        "positive_arbitrage_count": int(positive_arbitrage_count or 0),
        "middle_count": int(middle_count or 0),
        "positive_middle_count": int(positive_middle_count or 0),
        "plus_ev_count": int(plus_ev_count or 0),
        "sport_error_count": sport_error_count,
        "stale_filter_drop_total": stale_filter_drop_total,
        "provider_breakdown": provider_breakdown,
    }


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip().lower()


def _normalize_team_name(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    tokens = [token for token in normalized.split() if token]
    drop_tokens = {
        "fc",
        "cf",
        "sc",
        "ac",
        "afc",
        "u23",
        "u21",
        "u20",
        "u19",
        "u18",
        "u17",
        "women",
        "woman",
        "ladies",
        "reserves",
        "ii",
        "iii",
    }
    cleaned = [token for token in tokens if token not in drop_tokens]
    return " ".join(cleaned)


def _team_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _epoch_to_iso(value: float) -> Optional[str]:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    if timestamp > 1e12:
        timestamp /= 1000.0
    try:
        return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError):
        return None


def _normalize_commence_time(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _epoch_to_iso(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return _epoch_to_iso(int(text))
        try:
            if text.endswith("Z"):
                dt.datetime.fromisoformat(text[:-1])  # validate format
                return text
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return (
            parsed.astimezone(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    return None


def _parse_timestamp_seconds(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp <= 0:
            return None
        if timestamp > 1e12:
            timestamp /= 1000.0
        return timestamp
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        numeric = _safe_float(text)
        if numeric is not None and re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            timestamp = float(numeric)
            if timestamp <= 0:
                return None
            if timestamp > 1e12:
                timestamp /= 1000.0
            return timestamp
        normalized = _normalize_commence_time(text)
        if not normalized:
            return None
        try:
            if normalized.endswith("Z"):
                parsed = dt.datetime.fromisoformat(normalized[:-1] + "+00:00")
            else:
                parsed = dt.datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()
    return None


def _quote_updated_at_seconds(
    game: Optional[dict],
    bookmaker: Optional[dict],
    market: Optional[dict],
    outcome: Optional[dict],
) -> Optional[float]:
    live_state = _event_live_state(game or {}) if isinstance(game, dict) else {}
    for source in (outcome, market, bookmaker, live_state, game):
        if not isinstance(source, dict):
            continue
        for key in ("last_updated", "updated_at", "updatedAt", "timestamp", "ts"):
            timestamp = _parse_timestamp_seconds(source.get(key))
            if timestamp is not None:
                return timestamp
    return None


def _quote_freshness_timestamp_seconds(
    game: Optional[dict],
    bookmaker: Optional[dict],
    market: Optional[dict],
    outcome: Optional[dict],
    *,
    prefer_observed_at: bool = True,
) -> Optional[float]:
    live_state = _event_live_state(game or {}) if isinstance(game, dict) else {}
    sources = (outcome, market, bookmaker, live_state, game)
    if prefer_observed_at:
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in (
                "quote_observed_at",
                "quoteObservedAt",
                "observed_at",
                "observedAt",
                "last_seen_at",
                "lastSeenAt",
            ):
                timestamp = _parse_timestamp_seconds(source.get(key))
                if timestamp is not None:
                    return timestamp
    return _quote_updated_at_seconds(game, bookmaker, market, outcome)


def _live_quote_age_seconds(
    game: Optional[dict],
    bookmaker: Optional[dict],
    market: Optional[dict],
    outcome: Optional[dict],
    now_epoch: Optional[float] = None,
    *,
    prefer_observed_at: bool = True,
) -> Optional[float]:
    timestamp = _quote_freshness_timestamp_seconds(
        game,
        bookmaker,
        market,
        outcome,
        prefer_observed_at=prefer_observed_at,
    )
    if timestamp is None:
        return None
    now_value = float(now_epoch if now_epoch is not None else time.time())
    return max(0.0, now_value - timestamp)


def _prematch_quote_max_age_seconds() -> float:
    try:
        return max(0.0, float(PREMATCH_QUOTE_MAX_AGE_SECONDS_RAW))
    except (TypeError, ValueError):
        return 0.0


def _is_live_quote_fresh(
    game: Optional[dict],
    bookmaker: Optional[dict],
    market: Optional[dict],
    outcome: Optional[dict],
    scan_mode: str,
    now_epoch: Optional[float] = None,
) -> bool:
    normalized_mode = _normalize_scan_mode(scan_mode)
    if normalized_mode == SCAN_MODE_LIVE:
        max_age = _live_quote_max_age_seconds()
        prefer_observed_at = True
    elif normalized_mode == SCAN_MODE_PREMATCH:
        max_age = _prematch_quote_max_age_seconds()
        prefer_observed_at = False
    else:
        return True
    if max_age <= 0:
        return True
    age = _live_quote_age_seconds(
        game,
        bookmaker,
        market,
        outcome,
        now_epoch=now_epoch,
        prefer_observed_at=prefer_observed_at,
    )
    if age is None:
        return True
    return age <= max_age


def _bookmaker_live_state(game: Optional[dict], bookmaker: Optional[dict]) -> dict:
    if isinstance(bookmaker, dict):
        payload = bookmaker.get("live_state")
        if isinstance(payload, dict):
            return payload
    return _event_live_state(game or {}) if isinstance(game, dict) else {}


def _live_state_clock_tolerance_seconds() -> int:
    try:
        return max(0, int(float(LIVE_STATE_CLOCK_TOLERANCE_SECONDS_RAW)))
    except (TypeError, ValueError):
        return 180


def _parse_live_score_value(value: object) -> Optional[int]:
    numeric = _safe_float(value)
    if numeric is not None and numeric >= 0:
        return int(round(float(numeric)))
    text = _normalize_text(value)
    if text and re.fullmatch(r"\d+", text):
        return int(text)
    return None


def _extract_live_score_pair(state: Optional[dict]) -> Optional[Tuple[int, int]]:
    if not isinstance(state, dict):
        return None
    direct_pairs = (
        (state.get("home_score"), state.get("away_score")),
        (state.get("homeScore"), state.get("awayScore")),
        (state.get("scoreHome"), state.get("scoreAway")),
    )
    for home_value, away_value in direct_pairs:
        home_score = _parse_live_score_value(home_value)
        away_score = _parse_live_score_value(away_value)
        if home_score is not None and away_score is not None:
            return tuple(sorted((home_score, away_score)))
    score_payload = state.get("score") or state.get("scores")
    if isinstance(score_payload, dict):
        dict_pairs = (
            (score_payload.get("home"), score_payload.get("away")),
            (score_payload.get("home_score"), score_payload.get("away_score")),
            (score_payload.get("homeScore"), score_payload.get("awayScore")),
            (score_payload.get("team1"), score_payload.get("team2")),
        )
        for home_value, away_value in dict_pairs:
            home_score = _parse_live_score_value(home_value)
            away_score = _parse_live_score_value(away_value)
            if home_score is not None and away_score is not None:
                return tuple(sorted((home_score, away_score)))
    if isinstance(score_payload, (list, tuple)) and len(score_payload) >= 2:
        first_score = _parse_live_score_value(score_payload[0])
        second_score = _parse_live_score_value(score_payload[1])
        if first_score is not None and second_score is not None:
            return tuple(sorted((first_score, second_score)))
    text = _normalize_text(score_payload)
    if text:
        match = re.search(r"(\d+)\s*[-:]\s*(\d+)", text)
        if match:
            return tuple(sorted((int(match.group(1)), int(match.group(2)))))
    return None


def _normalize_live_period_token(value: object) -> str:
    token = _normalize_live_state_token(value)
    if not token:
        return ""
    token = re.sub(r"(\d)(st|nd|rd|th)", r"\1", token)
    token = (
        token.replace("first", "1")
        .replace("second", "2")
        .replace("third", "3")
        .replace("fourth", "4")
    )
    if token in {"half_time", "halftime", "ht"}:
        return "ht"
    if token in {"ot", "overtime", "extra_time"}:
        return "ot"
    patterns = (
        (r"(?:^|_)(?:q|quarter)_?(\d+)(?:_|$)", "q"),
        (r"(?:^|_)(?:half|h)_?(\d+)(?:_|$)", "h"),
        (r"(?:^|_)(?:period|p)_?(\d+)(?:_|$)", "p"),
        (r"(?:^|_)(?:inning)_?(\d+)(?:_|$)", "inning_"),
        (r"(?:^|_)(?:set)_?(\d+)(?:_|$)", "set_"),
        (r"(?:^|_)(?:map)_?(\d+)(?:_|$)", "map_"),
    )
    for pattern, prefix in patterns:
        match = re.search(pattern, token)
        if match:
            return f"{prefix}{match.group(1)}"
    if re.fullmatch(r"\d+", token):
        return f"segment_{token}"
    return token


def _extract_live_period_token(state: Optional[dict]) -> str:
    if not isinstance(state, dict):
        return ""
    for key in (
        "period",
        "game_period",
        "gamePeriod",
        "quarter",
        "inning",
        "set",
        "map",
        "phase",
        "stage",
    ):
        token = _normalize_live_period_token(state.get(key))
        if token:
            return token
    return _normalize_live_period_token(_live_state_token(state))


def _live_period_tokens_match(first: str, second: str) -> bool:
    if not first or not second:
        return True
    if first == second:
        return True
    generic_match = re.fullmatch(r"segment_(\d+)", first)
    if generic_match and re.search(rf"(?:^|_){generic_match.group(1)}$", second):
        return True
    generic_match = re.fullmatch(r"segment_(\d+)", second)
    if generic_match and re.search(rf"(?:^|_){generic_match.group(1)}$", first):
        return True
    return False


def _parse_live_clock_seconds(value: object) -> Optional[float]:
    numeric = _safe_float(value)
    if numeric is not None and numeric >= 0:
        return float(numeric)
    text = _normalize_text(value)
    if not text:
        return None
    match = re.search(r"(?:(\d+):)?(\d{1,2}):(\d{2})$", text)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        return float((hours * 3600) + (minutes * 60) + seconds)
    match = re.search(r"(\d{1,2}):(\d{2})$", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return float((minutes * 60) + seconds)
    return None


def _extract_live_clock_seconds(state: Optional[dict]) -> Optional[float]:
    if not isinstance(state, dict):
        return None
    for key in ("clock", "matchClock", "timeRemaining", "time_remaining"):
        seconds = _parse_live_clock_seconds(state.get(key))
        if seconds is not None:
            return seconds
    return None


def _selected_live_state_payload(
    states: Sequence[object],
    fallback: Optional[dict] = None,
) -> Optional[dict]:
    candidates = [state for state in states if isinstance(state, dict) and state]
    if isinstance(fallback, dict) and fallback:
        candidates.append(fallback)
    if not candidates:
        return None

    def _richness(state: dict) -> Tuple[int, float]:
        richness = 0
        if _live_state_token(state):
            richness += 1
        if _extract_live_period_token(state):
            richness += 1
        if _extract_live_score_pair(state) is not None:
            richness += 1
        if _extract_live_clock_seconds(state) is not None:
            richness += 1
        updated_at = 0.0
        for key in ("updated_at", "updatedAt", "last_updated", "timestamp", "ts"):
            parsed = _parse_timestamp_seconds(state.get(key))
            if parsed is not None:
                updated_at = parsed
                break
        return richness, updated_at

    best = max(candidates, key=_richness)
    return copy.deepcopy(best)


def _live_states_are_compatible(states: Sequence[object], scan_mode: str) -> bool:
    if _normalize_scan_mode(scan_mode) != SCAN_MODE_LIVE:
        return True
    normalized_states = [state for state in states if isinstance(state, dict) and state]
    if len(normalized_states) < 2:
        return True
    explicit_values = []
    for state in normalized_states:
        explicit = _live_state_is_explicitly_live(state)
        if explicit is False:
            return False
        if explicit is not None:
            explicit_values.append(explicit)
    if explicit_values and any(value != explicit_values[0] for value in explicit_values):
        return False

    period_token = ""
    for state in normalized_states:
        current_period = _extract_live_period_token(state)
        if not current_period:
            continue
        if not period_token:
            period_token = current_period
            continue
        if not _live_period_tokens_match(period_token, current_period):
            return False

    score_pair = None
    for state in normalized_states:
        current_score = _extract_live_score_pair(state)
        if current_score is None:
            continue
        if score_pair is None:
            score_pair = current_score
            continue
        if current_score != score_pair:
            return False

    if period_token:
        clock_values = [
            value
            for value in (_extract_live_clock_seconds(state) for state in normalized_states)
            if value is not None
        ]
        if len(clock_values) >= 2:
            tolerance = _live_state_clock_tolerance_seconds()
            if (max(clock_values) - min(clock_values)) > tolerance:
                return False
    return True


def _ensure_sharp_region(regions: List[str], sharp_key: str) -> List[str]:
    """Always include the region required for the sharp reference (usually EU)."""
    required_region = SHARP_BOOK_MAP.get(sharp_key, {}).get("region", "eu")
    normalized = []
    seen = set()
    for region in regions:
        if region not in seen:
            normalized.append(region)
            seen.add(region)
    if required_region and required_region not in seen and required_region in REGION_CONFIG:
        normalized.append(required_region)
    return normalized


def _apply_commission(price: float, commission_rate: float, is_exchange: bool) -> float:
    if not is_exchange:
        return price
    edge = price - 1.0
    if edge <= 0:
        return price
    return 1.0 + edge * (1.0 - commission_rate)


def _sharp_priority(selected_key: str) -> List[dict]:
    priority = []
    seen = set()
    if selected_key in SHARP_BOOK_MAP:
        priority.append(SHARP_BOOK_MAP[selected_key])
        seen.add(selected_key)
    for book in SHARP_BOOKS:
        key = book.get("key")
        if key and key not in seen:
            priority.append(book)
            seen.add(key)
    return priority


def _points_match(point_a, point_b, tolerance: float = 1e-6) -> bool:
    if point_a is None and point_b is None:
        return True
    if point_a is None or point_b is None:
        return False
    try:
        return abs(float(point_a) - float(point_b)) <= tolerance
    except (TypeError, ValueError):
        return False


def _spread_gap_info(favorite_line: float, underdog_line: float) -> Optional[dict]:
    if favorite_line is None or underdog_line is None:
        return None
    if favorite_line >= 0 or underdog_line <= 0:
        return None
    fav_abs = abs(favorite_line)
    if fav_abs >= underdog_line:
        return None
    gap = round(underdog_line - fav_abs, 2)
    start = math.floor(fav_abs) + 1
    end = math.ceil(underdog_line) - 1
    if end < start:
        return None
    middle_integers = list(range(start, end + 1))
    if not middle_integers:
        return None
    return {
        "gap_points": round(gap, 2),
        "middle_integers": middle_integers,
        "integer_count": len(middle_integers),
    }


def _total_gap_info(over_line: float, under_line: float) -> Optional[dict]:
    if over_line is None or under_line is None:
        return None
    if over_line >= under_line:
        return None
    gap = under_line - over_line
    lower = math.floor(over_line) + 1
    upper = math.ceil(under_line) - 1
    if upper < lower:
        return None
    middle_integers = list(range(lower, upper + 1))
    if not middle_integers:
        return None
    return {
        "gap_points": round(gap, 2),
        "middle_integers": middle_integers,
        "integer_count": len(middle_integers),
    }


def _build_sharp_reference(
    bookmaker: dict,
    commission_rate: float,
    is_exchange: bool,
    *,
    game: Optional[dict] = None,
    scan_mode: str = SCAN_MODE_PREMATCH,
) -> Dict[Tuple[str, str], dict]:
    """Return mapping of (market_key, line_key) to vig-free odds."""
    line_map: Dict[Tuple[str, str], List[dict]] = {}
    now_epoch = time.time()
    bookmaker_live_state = _selected_live_state_payload(
        [_bookmaker_live_state(game, bookmaker)],
        fallback=_event_live_state(game or {}) if isinstance(game, dict) else None,
    )
    for market in bookmaker.get("markets", []):
        market_key = market.get("key")
        if not market_key:
            continue
        for outcome in market.get("outcomes", []):
            if not _is_live_quote_fresh(
                game,
                bookmaker,
                market,
                outcome,
                scan_mode,
                now_epoch=now_epoch,
            ):
                continue
            price_val = _safe_float(outcome.get("price"))
            if price_val is None or price_val <= 1:
                continue
            line_key = _line_key(market_key, outcome)
            if not line_key:
                continue
            adjusted_price = _apply_commission(price_val, commission_rate, is_exchange)
            name_norm = _normalize_line_component(outcome.get("name"))
            if not name_norm:
                continue
            line_map.setdefault((market_key, line_key), []).append(
                {
                    "name": name_norm,
                    "display_name": _outcome_display_name(outcome),
                    "price": adjusted_price,
                    "raw_price": price_val,
                    "raw_percentage_odds": outcome.get("raw_percentage_odds"),
                    "point": outcome.get("point"),
                    "quote_updated_at": _epoch_to_iso(
                        _quote_updated_at_seconds(game, bookmaker, market, outcome) or 0.0
                    ),
                    "live_state": bookmaker_live_state,
                }
            )
    references: Dict[Tuple[str, str], dict] = {}
    for key, entries in line_map.items():
        if len(entries) != 2:
            continue
        first, second = entries
        fair_a, fair_b, vig_percent = _remove_vig(first["price"], second["price"])
        prob_a = 1 / fair_a if fair_a else 0.0
        prob_b = 1 / fair_b if fair_b else 0.0
        references[key] = {
            "vig_percent": round(vig_percent, 2),
            "outcomes": {
                first["name"]: {
                    "fair_odds": fair_a,
                    "true_probability": prob_a,
                    "sharp_odds": first["raw_price"],
                    "opponent_odds": second["raw_price"],
                    "opponent_name": second["display_name"],
                    "display_name": first["display_name"],
                    "point": first.get("point"),
                    "raw_percentage_odds": first.get("raw_percentage_odds"),
                    "quote_updated_at": first.get("quote_updated_at"),
                    "live_state": first.get("live_state"),
                },
                second["name"]: {
                    "fair_odds": fair_b,
                    "true_probability": prob_b,
                    "sharp_odds": second["raw_price"],
                    "opponent_odds": first["raw_price"],
                    "opponent_name": first["display_name"],
                    "display_name": second["display_name"],
                    "point": second.get("point"),
                    "raw_percentage_odds": second.get("raw_percentage_odds"),
                    "quote_updated_at": second.get("quote_updated_at"),
                    "live_state": second.get("live_state"),
                },
            },
        }
    return references


def _two_way_outcomes(
    bookmaker: dict,
    *,
    game: Optional[dict] = None,
    scan_mode: str = SCAN_MODE_PREMATCH,
) -> Dict[Tuple[str, str], List[dict]]:
    line_map: Dict[Tuple[str, str], List[dict]] = {}
    now_epoch = time.time()
    sport_key = _normalize_line_component(game.get("sport_key")) if isinstance(game, dict) else ""
    bookmaker_live_state = _selected_live_state_payload(
        [_bookmaker_live_state(game, bookmaker)],
        fallback=_event_live_state(game or {}) if isinstance(game, dict) else None,
    )
    for market in bookmaker.get("markets", []):
        market_key = market.get("key")
        if not market_key:
            continue
        for outcome in market.get("outcomes", []):
            if not _is_live_quote_fresh(
                game,
                bookmaker,
                market,
                outcome,
                scan_mode,
                now_epoch=now_epoch,
            ):
                continue
            display_price = _safe_float(outcome.get("price"))
            if display_price is None or display_price <= 1:
                continue
            line_key = _line_key(market_key, outcome)
            if not line_key:
                continue
            name_norm = _canonicalize_outcome_name(outcome.get("name"), sport_key)
            if not name_norm:
                continue
            line_map.setdefault((market_key, line_key), []).append(
                {
                    "name": name_norm,
                    "display_name": _outcome_display_name(outcome),
                    "price": display_price,
                    "point": outcome.get("point"),
                    "raw_percentage_odds": outcome.get("raw_percentage_odds"),
                    "quote_updated_at": _epoch_to_iso(
                        _quote_updated_at_seconds(game, bookmaker, market, outcome) or 0.0
                    ),
                    "live_state": bookmaker_live_state,
                }
            )
    return {key: entries for key, entries in line_map.items() if len(entries) == 2}


def _estimate_middle_probability(
    middle_integers: List[int], sport_key: str, market_key: str
) -> float:
    if not middle_integers:
        return 0.0
    lookup_key = f"{sport_key}_{market_key}"
    base_prob = PROBABILITY_PER_INTEGER.get(lookup_key, PROBABILITY_PER_INTEGER["default"])
    total = 0.0
    is_key_sport = sport_key in KEY_NUMBER_SPORTS
    for integer in middle_integers:
        abs_int = abs(integer)
        if is_key_sport and abs_int in NFL_KEY_NUMBER_PROBABILITY:
            total += NFL_KEY_NUMBER_PROBABILITY[abs_int]
        else:
            total += base_prob
    return min(total, MAX_MIDDLE_PROBABILITY)


def _calculate_middle_stakes(odds_a: float, odds_b: float, total_stake: float) -> Tuple[float, float]:
    if total_stake <= 0 or odds_a <= 1 or odds_b <= 1:
        return 0.0, 0.0
    profit_a = odds_a - 1
    profit_b = odds_b - 1
    denominator = profit_a + profit_b
    if denominator <= 0:
        return 0.0, 0.0
    stake_a = total_stake * profit_b / denominator
    stake_a = round(stake_a, 2)
    stake_b = round(total_stake - stake_a, 2)
    return stake_a, stake_b


def _calculate_middle_outcomes(
    stake_a: float, stake_b: float, odds_a: float, odds_b: float
) -> dict:
    total = stake_a + stake_b
    payout_a = round(stake_a * odds_a, 2)
    payout_b = round(stake_b * odds_b, 2)
    win_both = round((payout_a + payout_b) - total, 2)
    side_a_only = round(payout_a - total, 2)
    side_b_only = round(payout_b - total, 2)
    typical_miss = round((side_a_only + side_b_only) / 2, 2)
    return {
        "win_both_profit": win_both,
        "side_a_wins_profit": side_a_only,
        "side_b_wins_profit": side_b_only,
        "typical_miss_profit": typical_miss,
    }


def _calculate_middle_ev(
    win_both_profit: float,
    side_a_profit: float,
    side_b_profit: float,
    probability: float,
) -> float:
    probability = max(0.0, min(probability, 1.0))
    miss_probability = 1.0 - probability
    miss_ev = 0.5 * side_a_profit + 0.5 * side_b_profit
    value = (probability * win_both_profit) + (miss_probability * miss_ev)
    return round(value, 2)


def _format_middle_zone(
    description_source: str, middle_integers: List[int], is_total: bool
) -> str:
    if not middle_integers:
        return description_source
    middle_integers = sorted(middle_integers)
    if len(middle_integers) == 1:
        range_text = str(middle_integers[0])
    else:
        range_text = f"{middle_integers[0]}-{middle_integers[-1]}"
    if is_total:
        return f"Total {range_text}"
    return f"{description_source} by {range_text}"


def _request(url: str, params: Dict[str, str]) -> requests.Response:
    try:
        resp = requests.get(url, params=params, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network error
        raise ScannerError(f"Network error: {exc}") from exc
    if resp.status_code >= 400:
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                message = payload.get("message") or payload.get("error")
            else:
                message = resp.text or str(payload)
        except ValueError:
            message = resp.text or "Unknown error"
        raise ScannerError(message or f"API request failed ({resp.status_code})", status_code=resp.status_code)
    return resp


def _should_rotate_key(error: ScannerError) -> bool:
    return error.status_code in {401, 403, 429}


class ApiKeyPool:
    def __init__(self, keys: Sequence[str]) -> None:
        normalized = []
        seen = set()
        for key in keys:
            if not isinstance(key, str):
                continue
            cleaned = key.strip()
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        self._keys = normalized
        self._cycle = itertools.cycle(self._keys)
        self.calls_made = 0
        self._lock = threading.Lock()

    def request(self, url: str, params: Dict[str, str]) -> requests.Response:
        if not self._keys:
            raise ScannerError("API key is required", status_code=401)
        last_error: Optional[ScannerError] = None
        for _ in range(len(self._keys)):
            with self._lock:
                key = next(self._cycle)
                self.calls_made += 1
            try:
                return _request(url, {**params, "apiKey": key})
            except ScannerError as exc:
                last_error = exc
                if not _should_rotate_key(exc):
                    raise
        if last_error:
            raise last_error
        raise ScannerError("API key is required", status_code=401)


def fetch_sports(api_pool: ApiKeyPool) -> List[dict]:
    url = f"{BASE_URL}/sports/"
    resp = api_pool.request(url, {})
    try:
        return resp.json()
    except ValueError as exc:  # pragma: no cover - malformed payload
        raise ScannerError("Failed to parse sports list") from exc


def filter_sports(
    sports: Sequence[dict], requested: Sequence[str], all_sports: bool
) -> List[dict]:
    if all_sports:
        return [s for s in sports if s.get("active") and not s.get("has_outrights")]
    requested_set = set(requested) if requested else set(DEFAULT_SPORT_KEYS)
    return [s for s in sports if s.get("key") in requested_set and s.get("active")]


def fetch_odds_for_sport(
    api_pool: ApiKeyPool,
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    url = f"{BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "regions": ",".join(regions),
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
        "commenceTimeFrom": _iso_now(),
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    resp = api_pool.request(url, params)
    try:
        return resp.json()
    except ValueError as exc:  # pragma: no cover
        raise ScannerError(f"Failed to parse odds for {sport_key}") from exc


def _fetch_odds_for_market_batch(
    api_pool: ApiKeyPool,
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]],
    invalid_markets: List[str],
) -> List[dict]:
    if not markets:
        return []
    try:
        return fetch_odds_for_sport(
            api_pool,
            sport_key,
            markets,
            regions,
            bookmakers=bookmakers,
        )
    except ScannerError as exc:
        if exc.status_code not in ODDS_API_INVALID_MARKET_STATUS_CODES:
            raise
        if len(markets) == 1:
            invalid_markets.append(markets[0])
            return []
        midpoint = len(markets) // 2
        left = _fetch_odds_for_market_batch(
            api_pool,
            sport_key,
            markets[:midpoint],
            regions,
            bookmakers,
            invalid_markets,
        )
        right = _fetch_odds_for_market_batch(
            api_pool,
            sport_key,
            markets[midpoint:],
            regions,
            bookmakers,
            invalid_markets,
        )
        return _merge_odds_event_lists(left, right)


def fetch_odds_for_sport_multi_market(
    api_pool: ApiKeyPool,
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> Tuple[List[dict], List[str]]:
    cleaned_markets = []
    seen = set()
    for market in markets or []:
        token = str(market).strip()
        if token and token not in seen:
            cleaned_markets.append(token)
            seen.add(token)
    if not cleaned_markets:
        return [], []

    invalid_markets: List[str] = []
    merged_events: List[dict] = []
    for batch in _chunked(cleaned_markets, _odds_api_market_batch_size()):
        batch_events = _fetch_odds_for_market_batch(
            api_pool,
            sport_key,
            batch,
            regions,
            bookmakers,
            invalid_markets,
        )
        merged_events = _merge_odds_event_lists(merged_events, batch_events)
    return merged_events, invalid_markets


OutcomeInfo = Dict[str, object]
LineOfferMap = Dict[str, Dict[str, List[OutcomeInfo]]]


def _is_spread_market_key(market_key: object) -> bool:
    token = _normalize_line_component(market_key)
    if not token:
        return False
    return token == "spreads" or token.startswith("spreads_") or token == "alternate_spreads"


def _is_total_market_key(market_key: object) -> bool:
    token = _normalize_line_component(market_key)
    if not token:
        return False
    return token == "totals" or token.startswith("totals_") or token == "alternate_totals"


def _is_h2h_market_key(market_key: object) -> bool:
    token = _normalize_line_component(market_key)
    return token in {"h2h", "h2h_3_way"}


def _line_key(market: str, outcome: dict) -> Optional[str]:
    descriptor = _outcome_description_token(outcome)
    if market == "h2h":
        return f"moneyline:{descriptor}" if descriptor else "moneyline"
    point = outcome.get("point")
    if point is None:
        if descriptor:
            return f"{market}_{descriptor}"
        return f"{market}_nopoint"
    try:
        point_val = float(point)
    except (TypeError, ValueError):
        return None
    suffix = f":{descriptor}" if descriptor else ""
    if _is_spread_market_key(market):
        return f"spread_{abs(point_val):.2f}{suffix}"
    if _is_total_market_key(market):
        return f"total_{point_val:.2f}{suffix}"
    return f"{market}_{point_val:.2f}{suffix}"


def _outcome_display_name(outcome: dict) -> str:
    name = str(outcome.get("name") or "").strip()
    description = str(outcome.get("description") or "").strip()
    if name and description:
        return f"{name} ({description})"
    return name or description


def _record_line_offers(
    markets: List[dict],
    market_key: str,
    commission_rate: float,
    *,
    game: Optional[dict] = None,
    scan_mode: str = SCAN_MODE_PREMATCH,
) -> LineOfferMap:
    lines: LineOfferMap = {}
    now_epoch = time.time()
    sport_key = _normalize_line_component(game.get("sport_key")) if isinstance(game, dict) else ""
    for book in markets:
        bookmaker = book.get("title") or book.get("key")
        bookmaker_key = str(book.get("key") or bookmaker or "").strip()
        bookmaker_key_normalized = _normalize_line_component(bookmaker_key)
        book_live_state = _selected_live_state_payload(
            [_bookmaker_live_state(game, book)],
            fallback=_event_live_state(game or {}) if isinstance(game, dict) else None,
        )
        book_event_id = (
            book.get("event_id")
            or book.get("eventId")
            or book.get("id")
        )
        book_event_url = (
            book.get("event_url")
            or book.get("eventUrl")
            or book.get("url")
        )
        for market in book.get("markets", []):
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []):
                if not _is_live_quote_fresh(game, book, market, outcome, scan_mode, now_epoch=now_epoch):
                    continue
                display_price = _safe_float(outcome.get("price"))
                if display_price is None or display_price <= 1:
                    continue
                key = _line_key(market_key, outcome)
                if key is None:
                    continue
                normalized_point = outcome.get("point")
                stake_value = _safe_float(
                    outcome.get("stake")
                    or outcome.get("max_stake")
                    or outcome.get("liquidity")
                )
                entry = lines.setdefault(key, {})
                name = str(outcome.get("name") or "").strip()
                outcome_key = _canonicalize_outcome_name(name, sport_key)
                if not outcome_key:
                    continue
                display_name = _outcome_display_name(outcome)
                is_exchange = bookmaker_key_normalized in EXCHANGE_KEYS
                effective_price = _apply_commission(display_price, commission_rate, is_exchange)
                entry.setdefault(outcome_key, []).append(
                    {
                        "effective_price": effective_price,
                        "display_price": display_price,
                        "raw_percentage_odds": outcome.get("raw_percentage_odds"),
                        "quote_updated_at": _epoch_to_iso(
                            _quote_updated_at_seconds(game, book, market, outcome) or 0.0
                        ),
                        "quote_source": outcome.get("quote_source") or outcome.get("source"),
                        "live_state": book_live_state,
                        "bookmaker": bookmaker,
                        "bookmaker_key": bookmaker_key_normalized or bookmaker_key,
                        "name": outcome_key,
                        "display_name": display_name,
                        "description": outcome.get("description"),
                        "point": normalized_point,
                        "max_stake": stake_value,
                        "is_exchange": is_exchange,
                        "book_event_id": book_event_id,
                        "book_event_url": book_event_url,
                    }
                )
    normalized: LineOfferMap = {}
    for line_key, outcome_map in lines.items():
        normalized_outcomes: Dict[str, List[OutcomeInfo]] = {}
        for name, offers in outcome_map.items():
            if not isinstance(offers, list) or not offers:
                continue
            best_by_bookmaker: Dict[str, OutcomeInfo] = {}
            for idx, offer in enumerate(offers):
                if not isinstance(offer, dict):
                    continue
                book_identity = str(
                    offer.get("bookmaker_key") or offer.get("bookmaker") or ""
                ).strip().lower()
                if not book_identity:
                    book_identity = f"unknown:{idx}"
                existing = best_by_bookmaker.get(book_identity)
                offer_price = _safe_float(offer.get("effective_price")) or 0.0
                existing_price = _safe_float(existing.get("effective_price")) if isinstance(existing, dict) else None
                if existing_price is None or offer_price > existing_price:
                    best_by_bookmaker[book_identity] = offer
            ranked = sorted(
                best_by_bookmaker.values(),
                key=lambda item: (
                    _safe_float(item.get("effective_price")) or 0.0,
                    _safe_float(item.get("display_price")) or 0.0,
                ),
                reverse=True,
            )
            if ranked:
                normalized_outcomes[name] = ranked
        if normalized_outcomes:
            normalized[line_key] = normalized_outcomes
    return normalized


def _available_markets(game: dict) -> List[str]:
    keys = set()
    for book in game.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key")
            if key:
                keys.add(key)
    return sorted(keys)


def _collect_market_entries(
    game: dict,
    market_key: str,
    stake_total: float,
    commission_rate: float,
    scan_mode: str = SCAN_MODE_PREMATCH,
) -> List[dict]:
    bookmakers = game.get("bookmakers", [])
    markets = [
        {
            "key": book.get("key"),
            "title": book.get("title") or book.get("key"),
            "event_id": book.get("event_id") or book.get("eventId") or book.get("id"),
            "event_url": book.get("event_url") or book.get("eventUrl") or book.get("url"),
            "live_state": copy.deepcopy(book.get("live_state")) if isinstance(book.get("live_state"), dict) else None,
            "markets": book.get("markets", []),
        }
        for book in bookmakers
    ]
    line_offers = _record_line_offers(
        markets,
        market_key,
        commission_rate,
        game=game,
        scan_mode=scan_mode,
    )
    entries = []
    for line_key, offers in line_offers.items():
        offer_count = len(offers)
        if offer_count < 2:
            continue
        # Only moneyline-style markets can be genuine 3-way (home/draw/away).
        if offer_count > 3 or (offer_count > 2 and not _is_h2h_market_key(market_key)):
            continue
        outcome_names = sorted(offers.keys())
        candidate_lists = [offers.get(name) or [] for name in outcome_names]
        if any(not candidates for candidates in candidate_lists):
            continue

        best_outcomes: Optional[List[OutcomeInfo]] = None
        best_net_stake_info: Optional[dict] = None
        best_score: Optional[Tuple[float, float]] = None
        for combination in itertools.product(*candidate_lists):
            outcomes = list(combination)
            if not _live_states_are_compatible(
                [outcome.get("live_state") for outcome in outcomes],
                scan_mode,
            ):
                continue
            # Arbitrage output is meant to represent cross-book opportunities.
            # Skip lines where all selected legs come from the same bookmaker.
            bookmaker_keys = {
                str(outcome.get("bookmaker_key") or outcome.get("bookmaker") or "").strip().lower()
                for outcome in outcomes
                if str(outcome.get("bookmaker_key") or outcome.get("bookmaker") or "").strip()
            }
            if len(bookmaker_keys) < 2:
                continue
            if _is_spread_market_key(market_key):
                try:
                    point_values = [float(o.get("point")) for o in outcomes]
                except (TypeError, ValueError):
                    continue
                if any(p is None for p in point_values):
                    continue
                # Require opposite sides of the spread (one positive, one negative) and different teams.
                # Only the first two outcomes are checked; prior filtering ensures spread markets
                # always have exactly 2 outcomes (offer_count > 2 skips non-h2h markets above).
                if point_values[0] * point_values[1] >= 0:
                    continue
                selection_names = {o.get("name", "").strip().lower() for o in outcomes}
                if len(selection_names) < 2:
                    continue
            net_stake_info = _calculate_stakes(outcomes, stake_total, price_field="effective_price")
            guaranteed_profit = _safe_float(net_stake_info.get("guaranteed_profit"))
            if guaranteed_profit is None:
                continue
            net_roi = _safe_float(net_stake_info.get("roi_percent")) or 0.0
            score = (guaranteed_profit, net_roi)
            if best_score is None or score > best_score:
                best_outcomes = outcomes
                best_net_stake_info = net_stake_info
                best_score = score

        if not best_outcomes or not isinstance(best_net_stake_info, dict):
            continue
        outcomes = best_outcomes
        net_stake_info = best_net_stake_info
        has_exchange = any(o.get("is_exchange") for o in outcomes)
        outcome_payload = [
            {
                "outcome": o.get("display_name") or o["name"],
                "bookmaker": o["bookmaker"],
                "bookmaker_key": o.get("bookmaker_key"),
                "price": o["display_price"],
                "effective_price": o["effective_price"],
                "point": o.get("point"),
                "max_stake": o.get("max_stake"),
                "is_exchange": o.get("is_exchange", False),
                "book_event_id": o.get("book_event_id"),
                "book_event_url": o.get("book_event_url"),
                "quote_updated_at": o.get("quote_updated_at"),
                "quote_source": o.get("quote_source"),
                "raw_percentage_odds": o.get("raw_percentage_odds"),
            }
            for o in outcomes
        ]
        gross_stake_info = (
            _calculate_stakes(
                outcomes,
                _safe_float(net_stake_info.get("total")) or stake_total,
                price_field="display_price",
            )
            if has_exchange
            else None
        )
        net_roi = net_stake_info["roi_percent"]
        gross_roi = gross_stake_info["roi_percent"] if gross_stake_info else net_roi
        exchange_names = {
            o.get("bookmaker")
            or EXCHANGE_BOOKMAKERS.get(o.get("bookmaker_key"), {}).get("name")
            for o in outcomes
            if o.get("is_exchange")
        }
        exchange_books = sorted(name for name in exchange_names if name)
        entry = {
            "sport": game.get("sport_key"),
            "sport_display": game.get("sport_display")
            or SPORT_DISPLAY_NAMES.get(game.get("sport_key", ""), game.get("sport_key")),
            "sport_title": game.get("sport_title"),
            "event_id": game.get("id"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "event": f"{game.get('away_team')} vs {game.get('home_team')}",
            "commence_time": game.get("commence_time"),
            "live_state": _selected_live_state_payload(
                [outcome.get("live_state") for outcome in outcomes],
                fallback=_event_live_state(game),
            ),
            "market": market_key,
            "roi_percent": round(net_roi, 2),
            "gross_roi_percent": round(gross_roi, 2) if has_exchange else round(net_roi, 2),
            "best_odds": outcome_payload,
            "stakes": net_stake_info,
            "gross_stakes": gross_stake_info,
            "has_exchange": has_exchange,
            "exchange_books": exchange_books,
        }
        entries.append(entry)
    return entries


def _collect_middle_opportunities(
    game: dict,
    market_key: str,
    stake_total: float,
    commission_rate: float,
    scan_mode: str = SCAN_MODE_PREMATCH,
) -> List[dict]:
    if market_key not in MIDDLE_MARKETS or stake_total <= 0:
        return []
    bookmakers = game.get("bookmakers", [])
    offers = []
    now_epoch = time.time()
    for book in bookmakers:
        bookmaker_title = book.get("title") or book.get("key")
        bookmaker_key_raw = book.get("key") or bookmaker_title
        bookmaker_key = _normalize_line_component(bookmaker_key_raw) or str(bookmaker_key_raw or "").strip()
        book_live_state = _selected_live_state_payload(
            [_bookmaker_live_state(game, book)],
            fallback=_event_live_state(game),
        )
        markets = book.get("markets", [])
        for market in markets:
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []):
                if not _is_live_quote_fresh(game, book, market, outcome, scan_mode, now_epoch=now_epoch):
                    continue
                point = outcome.get("point")
                display_price = _safe_float(outcome.get("price"))
                if point is None or display_price is None or display_price <= 1:
                    continue
                name = outcome.get("name") or ""
                try:
                    point_value = float(point)
                except (TypeError, ValueError):
                    continue
                is_exchange = bookmaker_key in EXCHANGE_KEYS
                effective_price = _apply_commission(display_price, commission_rate, is_exchange)
                offers.append(
                    {
                        "pair_key": f"{bookmaker_key}:{name}:{point_value}",
                        "bookmaker": bookmaker_title,
                        "bookmaker_key": bookmaker_key,
                        "team": name,
                        "line": point_value,
                        "display_price": display_price,
                        "effective_price": effective_price,
                        "quote_updated_at": _epoch_to_iso(
                            _quote_updated_at_seconds(game, book, market, outcome) or 0.0
                        ),
                        "raw_percentage_odds": outcome.get("raw_percentage_odds"),
                        "live_state": book_live_state,
                        "max_stake": _safe_float(
                            outcome.get("stake")
                            or outcome.get("max_stake")
                            or outcome.get("liquidity")
                        ),
                        "is_exchange": is_exchange,
                    }
                )
    entries: List[dict] = []
    seen_pairs = set()
    for offer_a, offer_b in itertools.combinations(offers, 2):
        if offer_a["bookmaker_key"] == offer_b["bookmaker_key"]:
            continue
        if not _live_states_are_compatible(
            [offer_a.get("live_state"), offer_b.get("live_state")],
            scan_mode,
        ):
            continue
        pair_signature = tuple(sorted([offer_a["pair_key"], offer_b["pair_key"]]))
        if pair_signature in seen_pairs:
            continue
        seen_pairs.add(pair_signature)
        middle_entry = _build_middle_entry(game, market_key, offer_a, offer_b, stake_total)
        if middle_entry:
            entries.append(middle_entry)
    return entries


def _collect_plus_ev_opportunities(
    game: dict,
    markets: Sequence[str],
    sharp_priority: List[dict],
    commission_rate: float,
    min_edge_percent: float,
    bankroll: float,
    kelly_fraction: float,
    scan_mode: str = SCAN_MODE_PREMATCH,
) -> List[dict]:
    bookmakers = game.get("bookmakers", [])
    sharp_meta = None
    sharp_bookmaker = None
    sharp_key = ""
    for candidate in sharp_priority:
        candidate_key = _normalize_line_component(candidate.get("key"))
        if not candidate_key:
            continue
        for book in bookmakers:
            if _normalize_line_component(book.get("key")) == candidate_key:
                sharp_meta = candidate
                sharp_bookmaker = book
                sharp_key = candidate_key
                break
        if sharp_bookmaker:
            break
    if not sharp_bookmaker or not sharp_meta:
        return []
    is_sharp_exchange = sharp_meta.get("type") == "exchange"
    sharp_reference = _build_sharp_reference(
        sharp_bookmaker,
        commission_rate,
        is_sharp_exchange,
        game=game,
        scan_mode=scan_mode,
    )
    if not sharp_reference:
        return []
    opportunities: List[dict] = []
    for book in bookmakers:
        key = _normalize_line_component(book.get("key"))
        if not key:
            continue
        if key == sharp_key:
            continue
        if SOFT_BOOK_KEY_SET and key not in SOFT_BOOK_KEY_SET:
            continue
        bookmaker_title = book.get("title") or key
        is_exchange = key in EXCHANGE_KEYS
        soft_lines = _two_way_outcomes(
            book,
            game=game,
            scan_mode=scan_mode,
        )
        if not soft_lines:
            continue
        for (market_key, line_key), entries in soft_lines.items():
            if market_key not in markets:
                continue
            reference = sharp_reference.get((market_key, line_key))
            if not reference:
                continue
            for entry in entries:
                name_norm = entry["name"]
                sharp_outcome = reference["outcomes"].get(name_norm)
                if not sharp_outcome:
                    continue
                if not _live_states_are_compatible(
                    [entry.get("live_state"), sharp_outcome.get("live_state")],
                    scan_mode,
                ):
                    continue
                soft_point = entry.get("point")
                sharp_point = sharp_outcome.get("point")
                has_point = sharp_point is not None or soft_point is not None
                if has_point and not _points_match(soft_point, sharp_point):
                    continue
                display_price = entry["price"]
                effective_price = _apply_commission(display_price, commission_rate, is_exchange)
                fair_odds = sharp_outcome["fair_odds"]
                gross_edge = _calculate_edge_percent(display_price, fair_odds)
                net_edge = _calculate_edge_percent(effective_price, fair_odds)
                if net_edge < min_edge_percent:
                    continue
                true_probability = sharp_outcome["true_probability"]
                ev_per_100 = _calculate_ev(true_probability, effective_price, 100.0)
                full_pct, fraction_pct, recommended = _kelly_stake(
                    true_probability, effective_price, bankroll, kelly_fraction
                )
                opportunity = {
                    "id": str(uuid.uuid4()),
                    "sport": game.get("sport_key"),
                    "sport_display": game.get("sport_display")
                    or SPORT_DISPLAY_NAMES.get(game.get("sport_key", ""), game.get("sport_key")),
                    "event": f"{game.get('away_team')} vs {game.get('home_team')}",
                    "commence_time": game.get("commence_time"),
                    "live_state": _selected_live_state_payload(
                        [entry.get("live_state"), sharp_outcome.get("live_state")],
                        fallback=_event_live_state(game),
                    ),
                    "market": market_key,
                    "market_point": sharp_outcome.get("point"),
                    "bet": {
                        "outcome": entry.get("display_name") or "",
                        "soft_book": bookmaker_title,
                        "soft_key": key,
                        "soft_odds": display_price,
                        "effective_odds": effective_price,
                        "is_exchange": is_exchange,
                        "point": soft_point,
                        "quote_updated_at": entry.get("quote_updated_at"),
                        "raw_percentage_odds": entry.get("raw_percentage_odds"),
                    },
                    "sharp": {
                        "book": sharp_meta.get("name") or sharp_bookmaker.get("title") or sharp_meta.get("key"),
                        "key": sharp_meta.get("key"),
                        "odds": sharp_outcome["sharp_odds"],
                        "opponent_odds": sharp_outcome["opponent_odds"],
                        "opponent": sharp_outcome["opponent_name"],
                        "fair_odds": sharp_outcome["fair_odds"],
                        "true_probability": sharp_outcome["true_probability"],
                        "true_probability_percent": round(sharp_outcome["true_probability"] * 100, 2),
                        "vig_percent": reference.get("vig_percent"),
                        "quote_updated_at": sharp_outcome.get("quote_updated_at"),
                    },
                    "edge_percent": round(net_edge, 2),
                    "net_edge_percent": round(net_edge, 2),
                    "gross_edge_percent": round(gross_edge, 2),
                    "ev_per_100": round(ev_per_100, 2),
                    "kelly": {
                        "full_percent": full_pct,
                        "fraction_percent": fraction_pct,
                        "recommended_stake": recommended,
                    },
                    "has_exchange": is_exchange,
                }
                opportunities.append(opportunity)
    return opportunities


def _build_middle_entry(
    game: dict,
    market_key: str,
    offer_a: dict,
    offer_b: dict,
    stake_total: float,
) -> Optional[dict]:
    sport_key = game.get("sport_key") or ""
    home_team = game.get("home_team") or "Home"
    away_team = game.get("away_team") or "Away"
    favorite_offer: Optional[dict] = None
    underdog_offer: Optional[dict] = None
    over_offer: Optional[dict] = None
    under_offer: Optional[dict] = None
    gap_info: Optional[dict] = None
    is_total = market_key == "totals"

    if market_key == "spreads":
        if offer_a["line"] < 0 and offer_b["line"] > 0:
            favorite_offer, underdog_offer = offer_a, offer_b
        elif offer_b["line"] < 0 and offer_a["line"] > 0:
            favorite_offer, underdog_offer = offer_b, offer_a
        else:
            return None
        # Require opposite teams so we're not pairing two prices on the same side
        team_a = (favorite_offer.get("team") or "").strip().lower()
        team_b = (underdog_offer.get("team") or "").strip().lower()
        if not team_a or not team_b or team_a == team_b:
            return None
        gap_info = _spread_gap_info(favorite_offer["line"], underdog_offer["line"])
        if not gap_info:
            return None
        side_a = favorite_offer
        side_b = underdog_offer
        descriptor_name = side_a["team"]
    else:
        names = {
            offer_a["team"].strip().lower(): offer_a,
            offer_b["team"].strip().lower(): offer_b,
        }
        for name, offer in names.items():
            if "over" in name:
                over_offer = offer
            elif "under" in name:
                under_offer = offer
        if not over_offer or not under_offer:
            return None
        if over_offer["line"] >= under_offer["line"]:
            return None
        gap_info = _total_gap_info(over_offer["line"], under_offer["line"])
        side_a = over_offer
        side_b = under_offer
        descriptor_name = "Total"

    if not gap_info:
        return None
    middle_integers = gap_info["middle_integers"]
    if not middle_integers:
        return None
    middle_probability = _estimate_middle_probability(middle_integers, sport_key, market_key)
    if middle_probability <= 0:
        return None

    pricing_outcomes = [
        {
            "name": side_a["team"],
            "display_name": side_a["team"],
            "bookmaker": side_a["bookmaker"],
            "price": side_a["display_price"],
            "display_price": side_a["display_price"],
            "effective_price": side_a["effective_price"],
            "point": side_a["line"],
            "max_stake": side_a.get("max_stake"),
            "is_exchange": side_a["is_exchange"],
        },
        {
            "name": side_b["team"],
            "display_name": side_b["team"],
            "bookmaker": side_b["bookmaker"],
            "price": side_b["display_price"],
            "display_price": side_b["display_price"],
            "effective_price": side_b["effective_price"],
            "point": side_b["line"],
            "max_stake": side_b.get("max_stake"),
            "is_exchange": side_b["is_exchange"],
        },
    ]
    net_stake_info = _calculate_stakes(
        pricing_outcomes, stake_total, price_field="effective_price"
    )
    net_breakdown = net_stake_info.get("breakdown") if isinstance(net_stake_info, dict) else []
    if not isinstance(net_breakdown, list) or len(net_breakdown) != 2:
        return None
    stake_a = _safe_float(net_breakdown[0].get("stake")) or 0.0
    stake_b = _safe_float(net_breakdown[1].get("stake")) or 0.0
    used_total = _safe_float(net_stake_info.get("total")) or 0.0
    if used_total <= 0 or stake_a <= 0 or stake_b <= 0:
        return None
    outcomes = _calculate_middle_outcomes(stake_a, stake_b, side_a["effective_price"], side_b["effective_price"])
    ev_dollars = _calculate_middle_ev(
        outcomes["win_both_profit"],
        outcomes["side_a_wins_profit"],
        outcomes["side_b_wins_profit"],
        middle_probability,
    )
    ev_percent = round((ev_dollars / used_total) * 100, 2) if used_total else 0.0
    has_exchange = side_a["is_exchange"] or side_b["is_exchange"]
    gross_ev_percent = None
    if has_exchange:
        gross_stake_info = _calculate_stakes(
            pricing_outcomes, used_total, price_field="display_price"
        )
        gross_breakdown = (
            gross_stake_info.get("breakdown")
            if isinstance(gross_stake_info, dict)
            else []
        )
        if isinstance(gross_breakdown, list) and len(gross_breakdown) == 2:
            gross_stake_a = _safe_float(gross_breakdown[0].get("stake")) or 0.0
            gross_stake_b = _safe_float(gross_breakdown[1].get("stake")) or 0.0
            gross_outcomes = _calculate_middle_outcomes(
                gross_stake_a, gross_stake_b, side_a["display_price"], side_b["display_price"]
            )
            gross_ev = _calculate_middle_ev(
                gross_outcomes["win_both_profit"],
                gross_outcomes["side_a_wins_profit"],
                gross_outcomes["side_b_wins_profit"],
                middle_probability,
            )
            gross_total = _safe_float(gross_stake_info.get("total")) or used_total
            gross_ev_percent = round((gross_ev / gross_total) * 100, 2) if gross_total else 0.0

    key_numbers_crossed: List[int] = []
    includes_key_number = False
    if sport_key in KEY_NUMBER_SPORTS and market_key == "spreads":
        key_numbers_crossed = [
            integer for integer in middle_integers if integer in NFL_KEY_NUMBER_PROBABILITY
        ]
        includes_key_number = bool(key_numbers_crossed)

    middle_zone = _format_middle_zone(descriptor_name, middle_integers, is_total)
    best_odds = (
        {
            "team": side_a["team"],
            "line": side_a["line"],
            "price": side_a["display_price"],
            "effective_price": side_a["effective_price"],
            "bookmaker": side_a["bookmaker"],
            "max_stake": side_a.get("max_stake"),
            "is_exchange": side_a["is_exchange"],
            "quote_updated_at": side_a.get("quote_updated_at"),
            "raw_percentage_odds": side_a.get("raw_percentage_odds"),
        },
        {
            "team": side_b["team"],
            "line": side_b["line"],
            "price": side_b["display_price"],
            "effective_price": side_b["effective_price"],
            "bookmaker": side_b["bookmaker"],
            "max_stake": side_b.get("max_stake"),
            "is_exchange": side_b["is_exchange"],
            "quote_updated_at": side_b.get("quote_updated_at"),
            "raw_percentage_odds": side_b.get("raw_percentage_odds"),
        },
    )
    stakes_payload = {
        "requested_total": net_stake_info.get("requested_total", stake_total),
        "total": used_total,
        "limited_by_max_stake": bool(net_stake_info.get("limited_by_max_stake")),
        "max_executable_total": net_stake_info.get("max_executable_total"),
        "side_a": {
            "stake": stake_a,
            "payout": round(stake_a * side_a["effective_price"], 2),
        },
        "side_b": {
            "stake": stake_b,
            "payout": round(stake_b * side_b["effective_price"], 2),
        },
    }

    opportunity = {
        "id": str(uuid.uuid4()),
        "sport": sport_key,
        "sport_display": game.get("sport_display")
        or SPORT_DISPLAY_NAMES.get(sport_key, sport_key),
        "event": f"{game.get('away_team', away_team)} vs {game.get('home_team', home_team)}",
        "commence_time": game.get("commence_time"),
        "live_state": _selected_live_state_payload(
            [side_a.get("live_state"), side_b.get("live_state")],
            fallback=_event_live_state(game),
        ),
        "market": market_key,
        "side_a": best_odds[0],
        "side_b": best_odds[1],
        "gap": {
            "points": gap_info["gap_points"],
            "middle_integers": middle_integers,
            "integer_count": gap_info["integer_count"],
            "includes_key_number": includes_key_number,
            "key_numbers_crossed": key_numbers_crossed,
        },
        "middle_zone": middle_zone,
        "middle_probability": middle_probability,
        "probability_percent": round(middle_probability * 100, 2),
        "stakes": stakes_payload,
        "outcomes": outcomes,
        "ev_dollars": ev_dollars,
        "ev_percent": ev_percent,
        "has_exchange": has_exchange,
        "gross_ev_percent": gross_ev_percent,
        "sport_title": game.get("sport_title"),
    }
    return opportunity


def _calculate_stakes(outcomes: List[dict], stake_total: float, price_field: str) -> dict:
    if stake_total <= 0 or len(outcomes) < 2:
        return {
            "total": 0.0,
            "requested_total": max(0.0, round(float(stake_total or 0.0), 2)),
            "max_executable_total": 0.0,
            "limited_by_max_stake": False,
            "breakdown": [],
            "guaranteed_profit": 0.0,
            "roi_percent": 0.0,
        }
    requested_total = max(0.0, round(float(stake_total or 0.0), 2))
    inverses: List[float] = []
    for outcome in outcomes:
        price = outcome.get(price_field)
        if not price or price <= 0:
            return {
                "total": 0.0,
                "requested_total": requested_total,
                "max_executable_total": 0.0,
                "limited_by_max_stake": False,
                "breakdown": [],
                "guaranteed_profit": 0.0,
                "roi_percent": 0.0,
            }
        inverses.append(1 / price)
    inverse_sum = sum(inverses)
    if inverse_sum <= 0:
        return {
            "total": 0.0,
            "requested_total": requested_total,
            "max_executable_total": 0.0,
            "limited_by_max_stake": False,
            "breakdown": [],
            "guaranteed_profit": 0.0,
            "roi_percent": 0.0,
        }
    fractions = [inv / inverse_sum for inv in inverses]
    cap_total: Optional[float] = None
    max_stakes: List[Optional[float]] = []
    for outcome, fraction in zip(outcomes, fractions):
        cap_raw = None
        for cap_key in ("max_stake", "stake", "liquidity"):
            if cap_key in outcome and outcome.get(cap_key) is not None:
                cap_raw = outcome.get(cap_key)
                break
        cap_value = _safe_float(cap_raw)
        if cap_value is None:
            max_stakes.append(None)
            continue
        cap = max(0.0, float(cap_value))
        max_stakes.append(cap)
        if fraction > 0:
            leg_total_limit = cap / fraction
            cap_total = leg_total_limit if cap_total is None else min(cap_total, leg_total_limit)

    executable_total = requested_total
    if cap_total is not None:
        executable_total = min(executable_total, cap_total)
    executable_total = max(0.0, math.floor(executable_total * 100.0 + 1e-9) / 100.0)
    if executable_total <= 0:
        return {
            "total": 0.0,
            "requested_total": requested_total,
            "max_executable_total": round(cap_total, 2) if cap_total is not None else requested_total,
            "limited_by_max_stake": cap_total is not None and requested_total > 0,
            "breakdown": [],
            "guaranteed_profit": 0.0,
            "roi_percent": 0.0,
        }

    total_cents = int(round(executable_total * 100))
    raw_cents = [fraction * total_cents for fraction in fractions]
    stake_cents = [int(math.floor(value + 1e-9)) for value in raw_cents]
    remaining = total_cents - sum(stake_cents)
    order = sorted(
        range(len(stake_cents)),
        key=lambda idx: (raw_cents[idx] - stake_cents[idx]),
        reverse=True,
    )
    for idx in order:
        if remaining <= 0:
            break
        cap = max_stakes[idx]
        if cap is not None:
            cap_cents = int(math.floor(cap * 100 + 1e-9))
            if stake_cents[idx] + 1 > cap_cents:
                continue
        stake_cents[idx] += 1
        remaining -= 1

    actual_total = round(sum(stake_cents) / 100.0, 2)
    if actual_total <= 0:
        return {
            "total": 0.0,
            "requested_total": requested_total,
            "max_executable_total": round(cap_total, 2) if cap_total is not None else requested_total,
            "limited_by_max_stake": cap_total is not None and requested_total > 0,
            "breakdown": [],
            "guaranteed_profit": 0.0,
            "roi_percent": 0.0,
        }
    breakdown = []
    exact_payouts = []
    for outcome, fraction, stake_cent in zip(outcomes, fractions, stake_cents):
        stake_value = round(stake_cent / 100.0, 2)
        price_used = outcome.get(price_field)
        display_price = outcome.get("display_price", price_used)
        payout_exact = stake_value * price_used
        payout = round(payout_exact, 2)
        exact_payouts.append(payout_exact)
        breakdown.append(
            {
                "outcome": outcome.get("display_name") or outcome["name"],
                "bookmaker": outcome["bookmaker"],
                "price": display_price,
                "effective_price": outcome.get("effective_price", price_used),
                "point": outcome.get("point"),
                "stake": stake_value,
                "max_stake": outcome.get("max_stake"),
                "payout": payout,
                "fraction": fraction,
                "is_exchange": outcome.get("is_exchange", False),
            }
        )
    min_payout = min(item["payout"] for item in breakdown) if breakdown else 0.0
    min_payout_exact = min(exact_payouts) if exact_payouts else 0.0
    guaranteed = round(min_payout_exact - actual_total, 2)
    roi = round(((min_payout_exact - actual_total) / actual_total) * 100, 4) if actual_total else 0.0
    return {
        "total": actual_total,
        "requested_total": requested_total,
        "max_executable_total": round(cap_total, 2) if cap_total is not None else requested_total,
        "limited_by_max_stake": bool(cap_total is not None and actual_total + 1e-9 < requested_total),
        "breakdown": breakdown,
        "guaranteed_profit": guaranteed,
        "roi_percent": roi,
    }


def _summaries(
    opportunities: List[dict],
    sports_scanned: int,
    events_scanned: int,
    total_profit: float,
    api_calls_used: int,
) -> dict:
    by_sport: Dict[str, int] = {}
    band_counts: Dict[str, int] = {label: 0 for *_, label in ROI_BANDS}
    positive_count = 0
    for opp in opportunities:
        key = opp.get("sport_display") or opp.get("sport") or "unknown"
        by_sport[key] = by_sport.get(key, 0) + 1
        roi = opp.get("roi_percent", 0)
        if _safe_float(roi) is not None and float(roi) > 0:
            positive_count += 1
        for lower, upper, label in ROI_BANDS:
            if lower <= roi < upper:
                band_counts[label] += 1
                break
    return {
        "positive_count": positive_count,
        "by_sport": by_sport,
        "by_roi_band": band_counts,
        "sports_scanned": sports_scanned,
        "events_scanned": events_scanned,
        "api_calls_used": api_calls_used,
        "total_guaranteed_profit": round(total_profit, 2),
    }


def _middle_summary(opportunities: List[dict]) -> dict:
    count = len(opportunities)
    positive = [opp for opp in opportunities if opp.get("ev_percent", 0) > 0]
    avg_ev = (
        round(sum(opp.get("ev_percent", 0) for opp in opportunities) / count, 2)
        if count
        else 0.0
    )
    best = max(opportunities, key=lambda o: o.get("ev_percent", 0), default=None)
    by_sport: Dict[str, int] = {}
    key_numbers: Dict[str, int] = {}
    for opp in opportunities:
        sport = opp.get("sport_display") or opp.get("sport") or "unknown"
        by_sport[sport] = by_sport.get(sport, 0) + 1
        for key_number in opp.get("gap", {}).get("key_numbers_crossed", []):
            key_numbers[str(key_number)] = key_numbers.get(str(key_number), 0) + 1
    return {
        "count": count,
        "positive_count": len(positive),
        "average_ev_percent": avg_ev,
        "best_ev": {
            "ev_percent": best.get("ev_percent") if best else None,
            "event": best.get("event") if best else None,
            "sport": best.get("sport_display") if best else None,
        }
        if best
        else None,
        "by_sport": by_sport,
        "key_numbers": key_numbers,
    }


def _deduplicate_middles(opportunities: List[dict]) -> List[dict]:
    best_by_key: Dict[tuple, dict] = {}
    for opp in opportunities:
        key = (
            opp.get("sport"),
            opp.get("event"),
            opp.get("commence_time"),
            opp.get("market"),
            tuple(opp.get("gap", {}).get("middle_integers", [])),
        )
        if key not in best_by_key or opp.get("ev_percent", 0) > best_by_key[key].get("ev_percent", 0):
            best_by_key[key] = opp
    return list(best_by_key.values())


def _plus_ev_summary(opportunities: List[dict]) -> dict:
    count = len(opportunities)
    avg_edge = (
        round(sum(opp.get("edge_percent", 0) for opp in opportunities) / count, 2)
        if count
        else 0.0
    )
    best = max(opportunities, key=lambda o: o.get("edge_percent", 0), default=None)
    total_ev = round(sum(opp.get("ev_per_100", 0) for opp in opportunities), 2)
    by_sport: Dict[str, int] = {}
    by_edge_band: Dict[str, int] = {label: 0 for *_, label in EDGE_BANDS}
    for opp in opportunities:
        sport = opp.get("sport_display") or opp.get("sport") or "Other"
        by_sport[sport] = by_sport.get(sport, 0) + 1
        edge = opp.get("edge_percent", 0)
        for lower, upper, label in EDGE_BANDS:
            if lower <= edge < upper:
                by_edge_band[label] += 1
                break
    return {
        "count": count,
        "average_edge_percent": avg_edge,
        "total_ev_per_100": total_ev,
        "best_edge": {
            "edge_percent": best.get("edge_percent") if best else None,
            "event": best.get("event") if best else None,
            "sport": best.get("sport_display") if best else None,
        }
        if best
        else None,
        "by_sport": by_sport,
        "by_edge_band": by_edge_band,
    }


def _deduplicate_plus_ev(opportunities: List[dict]) -> List[dict]:
    best_by_key: Dict[tuple, dict] = {}
    for opp in opportunities:
        bet = opp.get("bet", {})
        point = bet.get("point")
        if point is None:
            point = opp.get("market_point")
        try:
            normalized_point = float(point) if point is not None else None
        except (TypeError, ValueError):
            normalized_point = None
        key = (
            opp.get("event"),
            opp.get("sport"),
            opp.get("commence_time"),
            opp.get("market"),
            (bet.get("outcome") or "").strip().lower(),
            normalized_point,
        )
        existing = best_by_key.get(key)
        if not existing or (opp.get("edge_percent", 0) > existing.get("edge_percent", 0)):
            best_by_key[key] = opp
    return list(best_by_key.values())


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _build_scan_timings(
    scan_started_at: float,
    steps: List[dict],
    sports: List[dict],
    top_n: int = 10,
) -> dict:
    ranked_steps = sorted(
        (step for step in steps if isinstance(step.get("ms"), (int, float))),
        key=lambda item: item.get("ms", 0.0),
        reverse=True,
    )
    ranked_sports = sorted(
        sports,
        key=lambda item: item.get("total_ms", 0.0),
        reverse=True,
    )
    total_ms = _elapsed_ms(scan_started_at)
    return {
        "total_ms": total_ms,
        "total_seconds": round(total_ms / 1000.0, 2),
        "slowest_steps": ranked_steps[: max(1, top_n)],
        "sports": ranked_sports,
    }


async def _fetch_provider_events_for_sport(
    provider_key: str,
    sport_key: str,
    provider_markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]],
    provider_context: Optional[dict] = None,
) -> dict:
    provider_title = PROVIDER_TITLES.get(provider_key, provider_key)
    fetch_provider_events = PROVIDER_FETCHERS.get(provider_key)
    provider_fetch_started_at = time.perf_counter()
    provider_events: List[dict] = []
    provider_stats: dict = {}
    provider_error: Optional[str] = None
    if not callable(fetch_provider_events):
        provider_error = "Provider fetcher is not callable"
    else:
        try:
            call_kwargs = {"bookmakers": bookmakers}
            if provider_context:
                supports_context = False
                try:
                    parameters = inspect.signature(fetch_provider_events).parameters.values()
                    supports_context = any(
                        parameter.name == "context" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters
                    )
                except (TypeError, ValueError):
                    supports_context = False
                if supports_context:
                    call_kwargs["context"] = provider_context
            provider_events = await _call_with_request_logger_async(
                fetch_provider_events,
                sport_key,
                provider_markets,
                regions,
                **call_kwargs,
            )
            if not isinstance(provider_events, list):
                provider_events = []
            provider_stats = getattr(fetch_provider_events, "last_stats", {}) or {}
        except Exception as exc:
            provider_error = str(exc)
    return {
        "key": provider_key,
        "name": provider_title,
        "events": provider_events,
        "stats": provider_stats,
        "error": provider_error,
        "ms": _elapsed_ms(provider_fetch_started_at),
    }


async def _scan_single_sport(
    sport: dict,
    scan_mode: str,
    all_markets: bool,
    should_fetch_api: bool,
    api_pool: ApiKeyPool,
    normalized_regions: Sequence[str],
    api_bookmakers: Sequence[str],
    provider_target_sport_keys: Sequence[str],
    enabled_provider_keys: Sequence[str],
    normalized_bookmakers: Sequence[str],
    stake_amount: float,
    commission_rate: float,
    sharp_priority: Sequence[dict],
    min_edge_percent: float,
    bankroll: float,
    kelly_fraction: float,
    progress_callback=None,
) -> dict:
    sport_started_at = time.perf_counter()
    sport_key = sport.get("key")
    if not sport_key:
        return {
            "skipped": True,
            "sport_key": "",
            "sport_timing": None,
            "timing_steps": [],
            "api_market_skips": [],
            "sport_errors": [],
            "provider_updates": {},
            "provider_snapshot_updates": {},
            "events_scanned": 0,
            "total_profit": 0.0,
            "arb_opportunities": [],
            "middle_opportunities": [],
            "plus_ev_opportunities": [],
            "stale_event_filters": [],
            "successful": 0,
        }

    sport_name = sport.get("title") or SPORT_DISPLAY_NAMES.get(sport_key, sport_key)
    normalized_scan_mode = _normalize_scan_mode(scan_mode)
    def _prepare_provider_live_funnel(events_payload: Sequence[dict]) -> Tuple[List[dict], List[dict], int, int, dict]:
        raw_events = list(events_payload)
        filtered_events = raw_events
        stats: dict = {}
        if normalized_scan_mode == SCAN_MODE_LIVE:
            filtered_events, stats = _filter_live_events_for_scan(raw_events)
        return raw_events, filtered_events, len(raw_events), len(filtered_events), stats
    sport_timing = {
        "sport_key": sport_key,
        "sport": sport_name,
        "scan_mode": normalized_scan_mode,
        "api_fetch_ms": 0.0,
        "provider_fetch_ms": 0.0,
        "analysis_ms": 0.0,
        "events_scanned": 0,
        "providers": [],
        "total_ms": 0.0,
    }
    timing_steps: List[dict] = []
    api_market_skips: List[dict] = []
    sport_errors: List[dict] = []
    provider_updates: Dict[str, dict] = {}
    provider_snapshot_updates: Dict[str, dict] = {}

    base_markets = markets_for_sport(sport_key)
    requested_markets = _requested_api_markets(
        sport_key,
        base_markets,
        all_markets=all_markets,
    )
    provider_markets = _provider_requested_markets(
        sport_key,
        requested_markets,
        all_markets=all_markets,
    )

    def _build_partial_sport_result() -> dict:
        return {
            "sport_key": sport_key,
            "arb_opportunities": list(arb_opportunities),
            "middle_opportunities": list(middle_opportunities),
            "plus_ev_opportunities": list(plus_ev_opportunities),
        }

    events: List[dict] = []
    if should_fetch_api:
        api_fetch_started_at = time.perf_counter()
        api_error = None
        try:
            events, invalid_markets = await _call_with_request_logger_async(
                fetch_odds_for_sport_multi_market,
                api_pool,
                sport_key,
                requested_markets,
                normalized_regions,
                bookmakers=api_bookmakers or None,
            )
            if invalid_markets:
                api_market_skips.append(
                    {
                        "sport_key": sport_key,
                        "markets": sorted(set(invalid_markets)),
                    }
                )
        except ScannerError as exc:
            api_error = str(exc)
            sport_errors.append(
                {
                    "sport_key": sport_key,
                    "sport": sport.get("title")
                    or SPORT_DISPLAY_NAMES.get(sport_key, sport_key),
                    "error": str(exc),
                }
            )
        api_fetch_ms = _elapsed_ms(api_fetch_started_at)
        sport_timing["api_fetch_ms"] = api_fetch_ms
        api_step = {
            "name": "api_fetch",
            "label": f"[{sport_key}] API odds fetch",
            "sport_key": sport_key,
            "ms": api_fetch_ms,
            "events_returned": len(events),
        }
        if api_error:
            api_step["error"] = api_error
        timing_steps.append(api_step)

    if sport_key not in set(provider_target_sport_keys):
        provider_markets = []

    provider_keys_to_fetch = [
        key for key in enabled_provider_keys if callable(PROVIDER_FETCHERS.get(key))
    ]
    provider_context = {"scan_mode": normalized_scan_mode, "live": normalized_scan_mode == SCAN_MODE_LIVE}
    async def _provider_job(provider_key: str) -> Tuple[str, dict]:
        try:
            result = await _fetch_provider_events_for_sport(
                provider_key=provider_key,
                sport_key=sport_key,
                provider_markets=provider_markets,
                regions=normalized_regions,
                bookmakers=normalized_bookmakers,
                provider_context=provider_context,
            )
        except Exception as exc:  # pragma: no cover - defensive
            result = {
                "key": provider_key,
                "name": PROVIDER_TITLES.get(provider_key, provider_key),
                "events": [],
                "stats": {},
                "error": str(exc),
                "ms": 0.0,
            }
        return provider_key, result

    async def _process_provider_result(provider_key: str, provider_result: dict) -> None:
        nonlocal events
        provider_title = str(provider_result.get("name") or PROVIDER_TITLES.get(provider_key, provider_key))
        provider_events = provider_result.get("events")
        if not isinstance(provider_events, list):
            provider_events = []
        raw_provider_events, filtered_provider_events, events_fetched_raw, events_after_live_filter, live_filter_stats = (
            _prepare_provider_live_funnel(provider_events)
        )
        provider_events = filtered_provider_events
        stats = provider_result.get("stats")
        if not isinstance(stats, dict):
            stats = {}
        provider_error = _normalize_text(provider_result.get("error")) or None
        provider_fetch_ms = float(provider_result.get("ms") or 0.0)
        if (
            provider_error
            and _provider_network_retry_once_enabled()
            and _is_transient_provider_network_error(provider_error)
        ):
            retry_delay = _provider_network_retry_delay_seconds()
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)
            retry_result = await _fetch_provider_events_for_sport(
                provider_key=provider_key,
                sport_key=sport_key,
                provider_markets=provider_markets,
                regions=normalized_regions,
                bookmakers=normalized_bookmakers,
                provider_context=provider_context,
            )
            provider_fetch_ms += float(retry_result.get("ms") or 0.0)
            retry_events = retry_result.get("events")
            if isinstance(retry_events, list):
                provider_events = retry_events
            else:
                provider_events = []
            retry_stats = retry_result.get("stats")
            stats = retry_stats if isinstance(retry_stats, dict) else {}
            (
                raw_provider_events,
                filtered_provider_events,
                events_fetched_raw,
                events_after_live_filter,
                live_filter_stats,
            ) = _prepare_provider_live_funnel(provider_events)
            provider_events = filtered_provider_events
            retry_error = _normalize_text(retry_result.get("error")) or None
            if retry_error:
                provider_error = f"{provider_error}; retry failed: {retry_error}"
            else:
                provider_error = None
                if isinstance(stats, dict):
                    stats["network_retry_recovered"] = True

        provider_update = provider_updates.setdefault(
            provider_key,
            {"events_merged": 0, "sports": []},
        )
        provider_snapshot_update = provider_snapshot_updates.setdefault(
            provider_key,
            {"provider_name": provider_title, "sports": [], "events": []},
        )
        sport_snapshot = {
            "sport_key": sport_key,
            "requested_markets": list(provider_markets),
            "regions": list(normalized_regions),
        }
        if provider_error:
            provider_update["sports"].append(
                {
                    "sport_key": sport_key,
                    "error": provider_error,
                    "requested_markets": list(provider_markets),
                }
            )
            sport_snapshot["error"] = provider_error
            provider_snapshot_update["sports"].append(sport_snapshot)
            sport_errors.append(
                {
                    "sport_key": sport_key,
                    "sport": sport.get("title")
                    or SPORT_DISPLAY_NAMES.get(sport_key, sport_key),
                    "error": f"{provider_title}: {provider_error}",
                }
            )
        else:
            merge_stats = _empty_event_merge_stats()
            events_before_merge = len(events)
            if provider_events:
                events = _merge_events_with_stats(events, provider_events, stats=merge_stats)
            events_after_merge = len(events)
            sport_summary = {
                "sport_key": sport_key,
                "events_returned": events_fetched_raw,
                "stats": stats,
                "requested_markets": list(provider_markets),
                "events_before_merge": events_before_merge,
                "events_after_merge": events_after_merge,
                "merge_stats": merge_stats,
            }
            if normalized_scan_mode == SCAN_MODE_LIVE:
                sport_summary.update(
                    {
                        "events_fetched_raw": events_fetched_raw,
                        "events_after_live_filter": events_after_live_filter,
                        "live_filter_stats": {
                            key: int(live_filter_stats.get(key, 0) or 0)
                            for key in (
                                "dropped_not_live_state",
                                "dropped_terminal_state",
                                "dropped_past",
                                "dropped_future",
                                "dropped_missing_time",
                                "suspicious_explicit_live_future",
                            )
                        },
                    }
                )
            provider_update["sports"].append(sport_summary)
            sport_snapshot.update(
                {
                    "events_returned": events_fetched_raw,
                    "stats": stats,
                    "events_before_merge": events_before_merge,
                    "events_after_merge": events_after_merge,
                    "merge_stats": merge_stats,
                }
            )
            provider_snapshot_update["sports"].append(sport_snapshot)
            if raw_provider_events:
                # Keep raw provider snapshots isolated from later event merges.
                provider_snapshot_update["events"].extend(copy.deepcopy(raw_provider_events))
            if provider_events:
                provider_update["events_merged"] += len(provider_events)

        sport_timing["provider_fetch_ms"] += provider_fetch_ms
        provider_timing = {
            "key": provider_key,
            "name": provider_title,
            "ms": provider_fetch_ms,
            "events_returned": events_fetched_raw,
        }
        if provider_error:
            provider_timing["error"] = provider_error
        sport_timing["providers"].append(provider_timing)
        provider_step = {
            "name": "provider_fetch",
            "label": f"[{sport_key}] Provider {provider_title}",
            "sport_key": sport_key,
            "provider_key": provider_key,
            "provider": provider_title,
            "ms": provider_fetch_ms,
            "events_returned": events_fetched_raw,
        }
        if provider_error:
            provider_step["error"] = provider_error
        timing_steps.append(provider_step)
        if callable(progress_callback):
            partial_provider_arb: List[dict] = []
            partial_provider_middle: List[dict] = []
            partial_provider_plus_ev: List[dict] = []
            progress_events, _ = _filter_events_for_scan_mode(copy.deepcopy(events), normalized_scan_mode)
            for game in progress_events:
                game["sport_key"] = sport_key
                game["sport_title"] = sport.get("title")
                game["sport_display"] = SPORT_DISPLAY_NAMES.get(sport_key, sport_key)
                if normalized_scan_mode == SCAN_MODE_LIVE:
                    current_markets = _available_markets(game)
                else:
                    current_markets = _available_markets(game) if all_markets else base_markets
                for market_key in current_markets:
                    partial_provider_arb.extend(
                        _collect_market_entries(
                            game,
                            market_key,
                            stake_amount,
                            commission_rate,
                            scan_mode=scan_mode,
                        )
                    )
                    if market_key in MIDDLE_MARKETS:
                        partial_provider_middle.extend(
                            _collect_middle_opportunities(
                                game,
                                market_key,
                                stake_amount,
                                commission_rate,
                                scan_mode=scan_mode,
                            )
                        )
                partial_provider_plus_ev.extend(
                    _collect_plus_ev_opportunities(
                        game,
                        current_markets,
                        sharp_priority,
                        commission_rate,
                        min_edge_percent,
                        bankroll,
                        kelly_fraction,
                        scan_mode=scan_mode,
                    )
                )
            progress_callback(
                {
                    "type": "provider_completed",
                    "sport_key": sport_key,
                    "sport": sport_name,
                    "provider_key": provider_key,
                    "provider": provider_title,
                    "ms": provider_fetch_ms,
                    "events_returned": len(provider_events),
                    "error": provider_error,
                    "result": {
                        "sport_key": sport_key,
                        "arb_opportunities": partial_provider_arb,
                        "middle_opportunities": partial_provider_middle,
                        "plus_ev_opportunities": partial_provider_plus_ev,
                    },
                }
            )

    if provider_markets and provider_keys_to_fetch:
        max_workers = min(_provider_fetch_max_workers(), len(provider_keys_to_fetch))
        if max_workers <= 1:
            for provider_key in provider_keys_to_fetch:
                resolved_key, result = await _provider_job(provider_key)
                await _process_provider_result(resolved_key, result)
        else:
            provider_semaphore = asyncio.Semaphore(max_workers)

            async def _limited_provider_job(provider_key: str) -> Tuple[str, dict]:
                async with provider_semaphore:
                    return await _provider_job(provider_key)

            provider_tasks = [
                asyncio.create_task(_limited_provider_job(provider_key))
                for provider_key in provider_keys_to_fetch
            ]
            for task in asyncio.as_completed(provider_tasks):
                resolved_key, result = await task
                await _process_provider_result(resolved_key, result)

    stale_event_filters: List[dict] = []
    events, time_filter_stats = _filter_events_for_scan_mode(events, normalized_scan_mode)
    dropped_past = int(time_filter_stats.get("dropped_past", 0) or 0)
    dropped_future = int(time_filter_stats.get("dropped_future", 0) or 0)
    dropped_missing = int(time_filter_stats.get("dropped_missing_time", 0) or 0)
    suspicious_explicit_live_future = int(
        time_filter_stats.get("suspicious_explicit_live_future", 0) or 0
    )
    if dropped_past or dropped_future or dropped_missing or suspicious_explicit_live_future:
        filter_payload = {
            "sport_key": sport_key,
            "scan_mode": normalized_scan_mode,
            "dropped_past": dropped_past,
            "dropped_missing_time": dropped_missing,
        }
        if dropped_future:
            filter_payload["dropped_future"] = dropped_future
        if suspicious_explicit_live_future:
            filter_payload["suspicious_explicit_live_future"] = suspicious_explicit_live_future
        stale_event_filters.append(filter_payload)

    arb_opportunities: List[dict] = []
    middle_opportunities: List[dict] = []
    plus_ev_opportunities: List[dict] = []
    total_profit = 0.0
    analysis_started_at = time.perf_counter()
    for game in events:
        game["sport_key"] = sport_key
        game["sport_title"] = sport.get("title")
        game["sport_display"] = SPORT_DISPLAY_NAMES.get(sport_key, sport_key)
        if normalized_scan_mode == SCAN_MODE_LIVE:
            # Live providers may return sport-specific markets beyond the prematch defaults
            # (for example soccer h2h/h2h_3_way even when base defaults are spreads/totals).
            # Analyze the markets actually present on the merged live event so we don't miss
            # valid provider-only in-play opportunities.
            arb_markets = _available_markets(game)
        else:
            arb_markets = _available_markets(game) if all_markets else base_markets
        for market_key in arb_markets:
            new_entries = _collect_market_entries(
                game,
                market_key,
                stake_amount,
                commission_rate,
                scan_mode=scan_mode,
            )
            for entry in new_entries:
                total_profit += entry["stakes"].get("guaranteed_profit", 0.0)
            arb_opportunities.extend(new_entries)
            if market_key in MIDDLE_MARKETS:
                middle_entries = _collect_middle_opportunities(
                    game,
                    market_key,
                    stake_amount,
                    commission_rate,
                    scan_mode=scan_mode,
                )
                middle_opportunities.extend(middle_entries)
        plus_entries = _collect_plus_ev_opportunities(
            game,
            arb_markets,
            sharp_priority,
            commission_rate,
            min_edge_percent,
            bankroll,
            kelly_fraction,
            scan_mode=scan_mode,
        )
        plus_ev_opportunities.extend(plus_entries)
    analysis_ms = _elapsed_ms(analysis_started_at)
    sport_timing["analysis_ms"] = analysis_ms
    sport_timing["events_scanned"] = len(events)
    timing_steps.append(
        {
            "name": "analyze_events",
            "label": f"[{sport_key}] Analyze events",
            "sport_key": sport_key,
            "ms": analysis_ms,
            "events_processed": len(events),
        }
    )
    sport_timing["total_ms"] = _elapsed_ms(sport_started_at)
    timing_steps.append(
        {
            "name": "sport_total",
            "label": f"[{sport_key}] Total",
            "sport_key": sport_key,
            "ms": sport_timing["total_ms"],
            "events_scanned": len(events),
        }
    )
    return {
        "skipped": False,
        "sport_key": sport_key,
        "sport_timing": sport_timing,
        "timing_steps": timing_steps,
        "api_market_skips": api_market_skips,
        "sport_errors": sport_errors,
        "provider_updates": provider_updates,
        "provider_snapshot_updates": provider_snapshot_updates,
        "events_scanned": len(events),
        "total_profit": total_profit,
        "arb_opportunities": arb_opportunities,
        "middle_opportunities": middle_opportunities,
        "plus_ev_opportunities": plus_ev_opportunities,
        "stale_event_filters": stale_event_filters,
        "successful": 1,
    }


async def run_scan_async(
    api_key: str | Sequence[str],
    sports: Optional[List[str]] = None,
    scan_mode: str = SCAN_MODE_PREMATCH,
    all_sports: bool = False,
    all_markets: bool = False,
    stake_amount: float = DEFAULT_STAKE_AMOUNT,
    regions: Optional[Sequence[str]] = None,
    bookmakers: Optional[Sequence[str]] = None,
    commission_rate: float = DEFAULT_COMMISSION,
    sharp_book: str = DEFAULT_SHARP_BOOK,
    min_edge_percent: float = MIN_EDGE_PERCENT,
    bankroll: float = DEFAULT_BANKROLL,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    include_providers: Optional[Sequence[str]] = None,
    progress_callback=None,
) -> dict:
    scan_started_at = time.perf_counter()
    timing_steps: List[dict] = []
    sport_timings: List[dict] = []
    request_logger = _ScanRequestLogger(scan_time=_iso_now())
    request_logger.start()
    _activate_request_logger(request_logger)
    active_provider_scan_caches: List[object] = []

    def _finish(result: dict) -> dict:
        _deactivate_provider_scan_caches(active_provider_scan_caches)
        _deactivate_request_logger(request_logger)
        request_logger.close()
        return _attach_request_log_info(result, request_logger)

    setup_started_at = time.perf_counter()
    api_keys = _normalize_api_keys(api_key)
    normalized_scan_mode = _normalize_scan_mode(scan_mode)
    if stake_amount is None or stake_amount <= 0:
        stake_amount = DEFAULT_STAKE_AMOUNT
    all_markets = bool(all_markets)
    requested_sport_keys = _normalize_requested_sport_keys(sports)
    requested_provider_keys = _normalize_provider_keys(include_providers) or []
    enabled_provider_keys = _resolve_enabled_provider_keys(include_providers)
    normalized_bookmakers = _normalize_bookmakers(bookmakers)
    if regions is None:
        normalized_regions = derive_required_regions(
            normalized_bookmakers,
            sharp_book=sharp_book or DEFAULT_SHARP_BOOK,
        )
    else:
        normalized_regions = _normalize_regions(regions)
        normalized_regions = _ensure_sharp_region(
            normalized_regions,
            sharp_book or DEFAULT_SHARP_BOOK,
        )
    provider_bookmaker_keys = _normalize_provider_keys(normalized_bookmakers) or []
    api_bookmakers = [
        book for book in normalized_bookmakers if resolve_provider_key(book) is None
    ]
    explicit_provider_selection = set(requested_provider_keys) | set(provider_bookmaker_keys)
    enabled_provider_set = set(enabled_provider_keys)
    if provider_bookmaker_keys:
        enabled_provider_set.update(provider_bookmaker_keys)
        enabled_provider_keys = [key for key in PROVIDER_FETCHERS if key in enabled_provider_set]
    enabled_provider_keys = _dedupe_proxy_provider_keys(
        enabled_provider_keys,
        explicit_provider_keys=list(explicit_provider_selection),
    )
    enabled_provider_set = set(enabled_provider_keys)
    if normalized_scan_mode == SCAN_MODE_LIVE:
        if not enabled_provider_keys:
            enabled_provider_keys = _dedupe_proxy_provider_keys(
                _default_live_provider_keys(),
                explicit_provider_keys=list(explicit_provider_selection),
            )
            enabled_provider_set = set(enabled_provider_keys)
        live_bookmaker_filter = _dedupe_proxy_provider_keys(
            list(explicit_provider_selection) or list(enabled_provider_keys),
            explicit_provider_keys=list(explicit_provider_selection),
        )
        normalized_bookmakers = live_bookmaker_filter or list(enabled_provider_keys)
        provider_bookmaker_keys = _normalize_provider_keys(normalized_bookmakers) or list(enabled_provider_keys)
        api_bookmakers = []
    if callable(progress_callback):
        progress_callback(
            {
                "type": "scan_started",
                "scan_mode": normalized_scan_mode,
                "sports_total": len(sports or []),
                "providers_total": len(enabled_provider_keys),
            }
        )
    active_provider_scan_caches = _activate_provider_scan_caches(enabled_provider_keys)
    provider_target_sport_keys = set(requested_sport_keys) if enabled_provider_keys else set()
    provider_only_via_bookmakers = bool(normalized_bookmakers) and not api_bookmakers
    provider_only_via_missing_api_key = (
        bool(enabled_provider_keys)
        and not normalized_bookmakers
        and not api_bookmakers
        and not api_keys
    )
    should_fetch_api = not (provider_only_via_bookmakers or provider_only_via_missing_api_key)
    if normalized_scan_mode == SCAN_MODE_LIVE:
        should_fetch_api = False
    warnings: List[str] = []
    api_disabled_reason = ""
    warnings.extend(EXCHANGE_CONFIG_WARNINGS)
    if normalized_scan_mode == SCAN_MODE_LIVE:
        api_disabled_reason = "live_mode_provider_only"
    elif provider_only_via_bookmakers:
        api_disabled_reason = "provider_only_bookmakers_selected"
        warnings.append(
            "Odds API fetch skipped because only custom provider bookmakers were selected."
        )
    elif provider_only_via_missing_api_key:
        api_disabled_reason = "provider_only_without_api_key"
        warnings.append(
            "Odds API fetch skipped because no API key was provided and custom providers were used."
        )
    request_logger.log_meta(
        {
            "type": "scan_config",
            "time": _iso_now(),
            "scan_mode": normalized_scan_mode,
            "requested_sports": list(requested_sport_keys),
            "bookmakers": list(normalized_bookmakers),
            "enabled_provider_keys": list(enabled_provider_keys),
            "should_fetch_api": bool(should_fetch_api),
            "api_disabled_reason": api_disabled_reason,
            "warnings": list(warnings),
        }
    )
    timing_steps.append(
        {
            "name": "prepare_inputs",
            "label": "Prepare scan inputs",
            "ms": _elapsed_ms(setup_started_at),
            "scan_mode": normalized_scan_mode,
            "sports_requested": len(requested_sport_keys),
            "providers_enabled": len(enabled_provider_keys),
        }
    )
    if not normalized_regions:
        return _finish({
            "success": False,
            "error": "At least one region must be selected",
            "error_code": 400,
            "timings": _build_scan_timings(scan_started_at, timing_steps, sport_timings),
        })
    if should_fetch_api and not api_keys:
        return _finish({
            "success": False,
            "error": "API key is required",
            "error_code": 400,
            "timings": _build_scan_timings(scan_started_at, timing_steps, sport_timings),
        })
    if not should_fetch_api and not enabled_provider_keys:
        return _finish({
            "success": False,
            "error": "Live mode requires at least one enabled custom provider"
            if normalized_scan_mode == SCAN_MODE_LIVE
            else "No enabled providers selected",
            "error_code": 400,
            "scan_mode": normalized_scan_mode,
            "timings": _build_scan_timings(scan_started_at, timing_steps, sport_timings),
        })
    commission_rate = _clamp_commission(commission_rate)
    api_pool = ApiKeyPool(api_keys)
    filtered_api_sports: List[dict] = []
    api_sports_fetch_error = ""
    if should_fetch_api:
        fetch_sports_started_at = time.perf_counter()
        try:
            sports_list = await _call_with_request_logger_async(fetch_sports, api_pool)
            timing_steps.append(
                {
                    "name": "fetch_sports",
                    "label": "Fetch sports catalog",
                    "ms": _elapsed_ms(fetch_sports_started_at),
                    "sports_returned": len(sports_list),
                }
            )
            filter_started_at = time.perf_counter()
            filtered_api_sports = filter_sports(sports_list, requested_sport_keys, all_sports)
            timing_steps.append(
                {
                    "name": "filter_sports",
                    "label": "Filter sports for scan",
                    "ms": _elapsed_ms(filter_started_at),
                    "sports_selected": len(filtered_api_sports),
                }
            )
        except ScannerError as exc:
            timing_steps.append(
                {
                    "name": "fetch_sports",
                    "label": "Fetch sports catalog",
                    "ms": _elapsed_ms(fetch_sports_started_at),
                    "error": str(exc),
                }
            )
            if not enabled_provider_keys:
                return _finish({
                    "success": False,
                    "error": str(exc),
                    "error_code": 500,
                    "timings": _build_scan_timings(scan_started_at, timing_steps, sport_timings),
                })
            api_sports_fetch_error = str(exc)
            should_fetch_api = False

    scan_sports_by_key: Dict[str, dict] = {}
    for sport in filtered_api_sports:
        if not isinstance(sport, dict):
            continue
        sport_key = _normalize_line_component(sport.get("key"))
        if not sport_key:
            continue
        if sport_key not in scan_sports_by_key:
            scan_sports_by_key[sport_key] = sport
    # When scanning all active sports, providers should track the same active sport set.
    if enabled_provider_keys and all_sports and scan_sports_by_key:
        provider_target_sport_keys = set(scan_sports_by_key.keys())
    for sport_key in requested_sport_keys:
        if sport_key not in scan_sports_by_key and sport_key in provider_target_sport_keys:
            scan_sports_by_key[sport_key] = _sport_stub(sport_key)
    filtered = list(scan_sports_by_key.values())

    provider_summaries = {
        key: _empty_provider_summary(key, key in enabled_provider_set)
        for key in PROVIDER_FETCHERS
    }
    if not filtered:
        arb_summary = _summaries([], 0, 0, 0.0, api_pool.calls_made)
        middle_summary = _middle_summary([])
        plus_ev_summary = _plus_ev_summary([])
        scan_diagnostics = _build_scan_diagnostics(
            provider_summaries=provider_summaries,
            cross_provider_report=None,
            events_scanned=0,
            arbitrage_count=0,
            positive_arbitrage_count=0,
            middle_count=0,
            positive_middle_count=0,
            plus_ev_count=0,
            sport_errors=[],
            stale_event_filters=[],
        )
        empty_result = {
            "success": True,
            "scan_mode": normalized_scan_mode,
            "scan_time": _iso_now(),
            "warnings": warnings,
            "api_disabled_reason": api_disabled_reason,
            "api_market_skips": [],
            "provider_snapshot_paths": {},
            "cross_provider_match_report_path": "",
            "cross_provider_match_report_summary": {},
            "arbitrage": {
                "opportunities": [],
                "opportunities_count": 0,
                "summary": arb_summary,
                "stake_amount": stake_amount,
            },
            "middles": {
                "opportunities": [],
                "opportunities_count": 0,
                "summary": middle_summary,
                "stake_amount": stake_amount,
                "defaults": {
                    "min_gap": MIN_MIDDLE_GAP,
                    "sort": DEFAULT_MIDDLE_SORT,
                    "positive_only": SHOW_POSITIVE_EV_ONLY,
                },
            },
            "plus_ev": {
                "opportunities": [],
                "opportunities_count": 0,
                "summary": plus_ev_summary,
                "defaults": {
                    "sharp_book": sharp_book or DEFAULT_SHARP_BOOK,
                    "min_edge_percent": min_edge_percent,
                    "bankroll": bankroll,
                    "kelly_fraction": kelly_fraction,
                },
            },
            "sport_errors": [],
            "partial": False,
            "regions": normalized_regions,
            "commission_rate": commission_rate,
            "custom_providers": provider_summaries,
            "scan_diagnostics": scan_diagnostics,
            "timings": _build_scan_timings(scan_started_at, timing_steps, sport_timings),
        }
        if callable(progress_callback):
            progress_callback({"type": "scan_completed", "result": empty_result})
        return _finish(empty_result)

    arb_opportunities: List[dict] = []
    middle_opportunities: List[dict] = []
    plus_ev_opportunities: List[dict] = []
    api_market_skips: List[dict] = []
    provider_snapshots: Dict[str, dict] = {}
    events_scanned = 0
    total_profit = 0.0
    sport_errors: List[dict] = []
    if api_sports_fetch_error:
        sport_errors.append(
            {
                "sport_key": "odds_api",
                "sport": "Odds API",
                "error": f"Failed to load active sports list: {api_sports_fetch_error}",
            }
        )
    successful_sports = 0
    sharp_priority = _sharp_priority(sharp_book or DEFAULT_SHARP_BOOK)
    stale_event_filters: List[dict] = []

    sport_results_by_index: Dict[int, dict] = {}
    sport_workers = min(_sport_scan_max_workers(), len(filtered)) if filtered else 1
    if sport_workers <= 1:
        for idx, sport in enumerate(filtered):
            sport_result = await _await_if_needed(
                _scan_single_sport(
                    sport=sport,
                    scan_mode=normalized_scan_mode,
                    all_markets=all_markets,
                    should_fetch_api=should_fetch_api,
                    api_pool=api_pool,
                    normalized_regions=normalized_regions,
                    api_bookmakers=api_bookmakers,
                    provider_target_sport_keys=provider_target_sport_keys,
                    enabled_provider_keys=enabled_provider_keys,
                    normalized_bookmakers=normalized_bookmakers,
                    stake_amount=stake_amount,
                    commission_rate=commission_rate,
                    sharp_priority=sharp_priority,
                    min_edge_percent=min_edge_percent,
                    bankroll=bankroll,
                    kelly_fraction=kelly_fraction,
                    progress_callback=progress_callback,
                )
            )
            sport_results_by_index[idx] = sport_result
            if callable(progress_callback) and isinstance(sport_result, dict) and not sport_result.get("skipped"):
                progress_callback(
                    {
                        "type": "sport_completed",
                        "sport_key": sport_result.get("sport_key"),
                        "result": sport_result,
                    }
                )
    else:
        sport_semaphore = asyncio.Semaphore(sport_workers)

        async def _sport_job(idx: int, sport: dict) -> Tuple[int, Optional[dict], Optional[Exception]]:
            async with sport_semaphore:
                try:
                    result = await _await_if_needed(
                        _scan_single_sport(
                            sport=sport,
                            scan_mode=normalized_scan_mode,
                            all_markets=all_markets,
                            should_fetch_api=should_fetch_api,
                            api_pool=api_pool,
                            normalized_regions=normalized_regions,
                            api_bookmakers=api_bookmakers,
                            provider_target_sport_keys=provider_target_sport_keys,
                            enabled_provider_keys=enabled_provider_keys,
                            normalized_bookmakers=normalized_bookmakers,
                            stake_amount=stake_amount,
                            commission_rate=commission_rate,
                            sharp_priority=sharp_priority,
                            min_edge_percent=min_edge_percent,
                            bankroll=bankroll,
                            kelly_fraction=kelly_fraction,
                            progress_callback=progress_callback,
                        )
                    )
                    return idx, result, None
                except Exception as exc:  # pragma: no cover - defensive
                    return idx, None, exc

        sport_tasks = [
            asyncio.create_task(_sport_job(idx, sport))
            for idx, sport in enumerate(filtered)
        ]
        for task in asyncio.as_completed(sport_tasks):
            idx, result, error = await task
            sport = filtered[idx]
            if error is None and isinstance(result, dict):
                sport_results_by_index[idx] = result
                if callable(progress_callback) and not result.get("skipped"):
                    progress_callback(
                        {
                            "type": "sport_completed",
                            "sport_key": result.get("sport_key"),
                            "result": result,
                        }
                    )
                continue
            sport_key = _normalize_line_component(sport.get("key"))
            sport_name = sport.get("title") or SPORT_DISPLAY_NAMES.get(sport_key, sport_key)
            sport_errors.append(
                {
                    "sport_key": sport_key,
                    "sport": sport_name,
                    "error": f"Sport worker failed: {error}",
                }
            )

    for idx in range(len(filtered)):
        result = sport_results_by_index.get(idx)
        if not isinstance(result, dict) or result.get("skipped"):
            continue
        successful_sports += int(result.get("successful", 0) or 0)
        events_scanned += int(result.get("events_scanned", 0) or 0)
        total_profit += float(result.get("total_profit", 0.0) or 0.0)
        arb_opportunities.extend(result.get("arb_opportunities") or [])
        middle_opportunities.extend(result.get("middle_opportunities") or [])
        plus_ev_opportunities.extend(result.get("plus_ev_opportunities") or [])
        api_market_skips.extend(result.get("api_market_skips") or [])
        stale_event_filters.extend(result.get("stale_event_filters") or [])
        sport_errors.extend(result.get("sport_errors") or [])
        sport_timing = result.get("sport_timing")
        if isinstance(sport_timing, dict):
            sport_timings.append(sport_timing)
        timing_steps.extend(result.get("timing_steps") or [])
        provider_updates = result.get("provider_updates") or {}
        for provider_key, update in provider_updates.items():
            if not isinstance(update, dict):
                continue
            provider_summary = provider_summaries.setdefault(
                provider_key,
                _empty_provider_summary(provider_key, True),
            )
            provider_summary["events_merged"] += int(update.get("events_merged", 0) or 0)
            provider_summary["sports"].extend(update.get("sports") or [])

        snapshot_updates = result.get("provider_snapshot_updates") or {}
        for provider_key, update in snapshot_updates.items():
            if not isinstance(update, dict):
                continue
            provider_title = _normalize_text(update.get("provider_name")) or PROVIDER_TITLES.get(
                provider_key, provider_key
            )
            provider_snapshot = provider_snapshots.setdefault(
                provider_key,
                {
                    "provider_name": provider_title,
                    "sports": [],
                    "events": [],
                },
            )
            provider_snapshot["sports"].extend(update.get("sports") or [])
            provider_snapshot["events"].extend(update.get("events") or [])

    finalize_started_at = time.perf_counter()
    api_calls_used = api_pool.calls_made
    arb_opportunities.sort(key=lambda x: x["roi_percent"], reverse=True)
    middle_opportunities.sort(key=lambda x: x["ev_percent"], reverse=True)
    middle_opportunities = _deduplicate_middles(middle_opportunities)
    plus_ev_opportunities = _deduplicate_plus_ev(plus_ev_opportunities)
    plus_ev_opportunities.sort(key=lambda x: x.get("edge_percent", 0), reverse=True)
    arb_summary = _summaries(
        arb_opportunities, successful_sports, events_scanned, total_profit, api_calls_used
    )
    middle_summary = _middle_summary(middle_opportunities)
    plus_ev_summary = _plus_ev_summary(plus_ev_opportunities)
    timing_steps.append(
        {
            "name": "finalize_results",
            "label": "Finalize and rank results",
            "ms": _elapsed_ms(finalize_started_at),
            "arbitrage_count": len(arb_opportunities),
            "middle_count": len(middle_opportunities),
            "plus_ev_count": len(plus_ev_opportunities),
        }
    )
    timings = _build_scan_timings(scan_started_at, timing_steps, sport_timings)
    scan_time = _iso_now()
    provider_snapshot_paths = _persist_provider_snapshots(scan_time, provider_snapshots)
    cross_provider_report = _build_cross_provider_match_report(scan_time, provider_snapshots)
    cross_provider_match_report_path = _persist_cross_provider_match_report(
        scan_time,
        provider_snapshots,
    )
    cross_provider_report_summary = (
        cross_provider_report.get("summary")
        if isinstance(cross_provider_report, dict)
        and isinstance(cross_provider_report.get("summary"), dict)
        else {}
    )
    scan_diagnostics = _build_scan_diagnostics(
        provider_summaries=provider_summaries,
        cross_provider_report=cross_provider_report,
        events_scanned=events_scanned,
        arbitrage_count=len(arb_opportunities),
        positive_arbitrage_count=int(arb_summary.get("positive_count", 0) or 0),
        middle_count=len(middle_opportunities),
        positive_middle_count=int(middle_summary.get("positive_count", 0) or 0),
        plus_ev_count=len(plus_ev_opportunities),
        sport_errors=sport_errors,
        stale_event_filters=stale_event_filters,
    )
    final_result = {
        "success": True,
        "scan_mode": normalized_scan_mode,
        "scan_time": scan_time,
        "warnings": warnings,
        "api_disabled_reason": api_disabled_reason,
        "api_market_skips": api_market_skips,
        "provider_snapshot_paths": provider_snapshot_paths,
        "cross_provider_match_report_path": cross_provider_match_report_path,
        "cross_provider_match_report_summary": cross_provider_report_summary,
        "arbitrage": {
            "opportunities": arb_opportunities,
            "opportunities_count": len(arb_opportunities),
            "summary": arb_summary,
            "stake_amount": stake_amount,
        },
        "middles": {
            "opportunities": middle_opportunities,
            "opportunities_count": len(middle_opportunities),
            "summary": middle_summary,
            "stake_amount": stake_amount,
            "defaults": {
                "min_gap": MIN_MIDDLE_GAP,
                "sort": DEFAULT_MIDDLE_SORT,
                "positive_only": SHOW_POSITIVE_EV_ONLY,
            },
        },
        "plus_ev": {
            "opportunities": plus_ev_opportunities,
            "opportunities_count": len(plus_ev_opportunities),
            "summary": plus_ev_summary,
            "defaults": {
                "sharp_book": sharp_book or DEFAULT_SHARP_BOOK,
                "min_edge_percent": min_edge_percent,
                "bankroll": bankroll,
                "kelly_fraction": kelly_fraction,
            },
        },
        "sport_errors": sport_errors,
        "partial": bool(sport_errors),
        "regions": normalized_regions,
        "commission_rate": commission_rate,
        "stale_event_filters": stale_event_filters,
        "custom_providers": provider_summaries,
        "scan_diagnostics": scan_diagnostics,
        "timings": timings,
    }
    if callable(progress_callback):
        progress_callback(
            {
                "type": "scan_completed",
                "result": final_result,
            }
        )
    return _finish(final_result)


def run_scan(
    api_key: str | Sequence[str],
    sports: Optional[List[str]] = None,
    scan_mode: str = SCAN_MODE_PREMATCH,
    all_sports: bool = False,
    all_markets: bool = False,
    stake_amount: float = DEFAULT_STAKE_AMOUNT,
    regions: Optional[Sequence[str]] = None,
    bookmakers: Optional[Sequence[str]] = None,
    commission_rate: float = DEFAULT_COMMISSION,
    sharp_book: str = DEFAULT_SHARP_BOOK,
    min_edge_percent: float = MIN_EDGE_PERCENT,
    bankroll: float = DEFAULT_BANKROLL,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    include_providers: Optional[Sequence[str]] = None,
    progress_callback=None,
) -> dict:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _get_scan_async_runtime().submit(
            run_scan_async(
                api_key=api_key,
                sports=sports,
                scan_mode=scan_mode,
                all_sports=all_sports,
                all_markets=all_markets,
                stake_amount=stake_amount,
                regions=regions,
                bookmakers=bookmakers,
                commission_rate=commission_rate,
                sharp_book=sharp_book,
                min_edge_percent=min_edge_percent,
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                include_providers=include_providers,
                progress_callback=progress_callback,
            )
        )
    raise RuntimeError("run_scan() cannot be used inside an active event loop; use await run_scan_async()")


def _remove_vig(odds_a: float, odds_b: float) -> Tuple[float, float, float]:
    """Return fair odds for both sides plus vig percent."""
    if odds_a <= 1 or odds_b <= 1:
        return odds_a, odds_b, 0.0
    implied_a = 1 / odds_a
    implied_b = 1 / odds_b
    total_implied = implied_a + implied_b
    if total_implied <= 0:
        return odds_a, odds_b, 0.0
    true_prob_a = implied_a / total_implied
    true_prob_b = implied_b / total_implied
    fair_a = 1 / true_prob_a if true_prob_a else odds_a
    fair_b = 1 / true_prob_b if true_prob_b else odds_b
    vig_percent = max(0.0, (total_implied - 1.0) * 100)
    return fair_a, fair_b, vig_percent


def _calculate_edge_percent(soft_odds: float, fair_odds: float) -> float:
    if fair_odds <= 0:
        return 0.0
    return (soft_odds / fair_odds - 1.0) * 100


def _calculate_ev(true_probability: float, odds: float, stake: float) -> float:
    true_probability = max(0.0, min(true_probability, 1.0))
    win_amount = stake * (odds - 1.0)
    lose_amount = stake
    value = (true_probability * win_amount) - ((1.0 - true_probability) * lose_amount)
    return round(value, 2)


def _kelly_stake(
    true_probability: float, odds: float, bankroll: float, fraction: float
) -> Tuple[float, float, float]:
    if bankroll <= 0 or odds <= 1:
        return 0.0, 0.0, 0.0
    p = max(0.0, min(true_probability, 1.0))
    q = 1.0 - p
    b = odds - 1.0
    if b <= 0:
        return 0.0, 0.0, 0.0
    kelly_fraction = (b * p - q) / b
    if kelly_fraction <= 0:
        return 0.0, 0.0, 0.0
    fraction = max(0.0, min(fraction, 1.0))
    recommended_fraction = kelly_fraction * fraction
    stake = round(bankroll * recommended_fraction, 2)
    return round(kelly_fraction * 100, 2), round(recommended_fraction * 100, 2), stake

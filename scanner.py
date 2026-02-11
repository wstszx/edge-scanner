"""Core arbitrage scanning logic."""

from __future__ import annotations

import datetime as dt
import concurrent.futures
import difflib
import itertools
import json
import math
import os
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

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
    markets_for_sport,
)
from providers import PROVIDER_FETCHERS, PROVIDER_TITLES, resolve_provider_key

BASE_URL = "https://api.the-odds-api.com/v4"
MIDDLE_MARKETS = {"spreads", "totals"}
ALLOWED_PLUS_EV_MARKETS = {"h2h", "spreads", "totals"}
SOFT_BOOK_KEY_SET = set(SOFT_BOOK_KEYS)
SHARP_BOOK_MAP = {book["key"]: book for book in SHARP_BOOKS}
PUREBET_BOOK_KEY = "purebet"
PUREBET_TITLE = "Purebet"
PUREBET_ENV_ENABLED = os.getenv("PUREBET_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
PUREBET_SOURCE = os.getenv("PUREBET_SOURCE", "api").strip().lower()
PUREBET_SAMPLE_PATH = os.getenv(
    "PUREBET_SAMPLE_PATH", str(Path("data") / "purebet_sample.json")
).strip()
PUREBET_API_BASE = os.getenv("PUREBET_API_BASE", "").strip()
PUREBET_DEFAULT_BASE = "https://v3api.purebet.io"
PUREBET_LIVE = os.getenv("PUREBET_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}
PUREBET_USER_AGENT = os.getenv(
    "PUREBET_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
).strip()
PUREBET_ORIGIN = os.getenv("PUREBET_ORIGIN", "https://purebet.io").strip()
PUREBET_REFERER = os.getenv("PUREBET_REFERER", "https://purebet.io/").strip()
PUREBET_MIN_STAKE_RAW = os.getenv("PUREBET_MIN_STAKE", "50").strip()
PUREBET_MAX_AGE_SECONDS_RAW = os.getenv("PUREBET_MAX_AGE_SECONDS", "60").strip()
PUREBET_FUZZY_THRESHOLD_RAW = os.getenv("PUREBET_FUZZY_MATCH_THRESHOLD", "0.85").strip()
PUREBET_MARKET_WORKERS_RAW = os.getenv("PUREBET_MARKET_WORKERS", "8").strip()
PUREBET_MARKET_RETRIES_RAW = os.getenv("PUREBET_MARKET_RETRIES", "2").strip()
PUREBET_RETRY_BACKOFF_RAW = os.getenv("PUREBET_RETRY_BACKOFF", "0.4").strip()
PUREBET_MARKETS_ENABLED = os.getenv("PUREBET_MARKETS_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
PUREBET_LEAGUE_SYNC_ENABLED = os.getenv(
    "PUREBET_LEAGUE_SYNC_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
PUREBET_LEAGUE_SYNC_TTL_RAW = os.getenv("PUREBET_LEAGUE_SYNC_TTL", "600").strip()
EVENT_TIME_TOLERANCE_MINUTES = os.getenv("EVENT_TIME_TOLERANCE_MINUTES", "15").strip()
PUREBET_DEFAULT_LEAGUE_MAP = {
    487: "basketball_nba",
    493: "basketball_ncaab",
    889: "americanfootball_nfl",
    1980: "soccer_epl",
    2196: "soccer_spain_la_liga",
    1842: "soccer_germany_bundesliga",
    2436: "soccer_italy_serie_a",
    2036: "soccer_france_ligue_one",
    2663: "soccer_usa_mls",
}
PUREBET_LEAGUE_MAP_RAW = os.getenv("PUREBET_LEAGUE_MAP", "").strip()
PUREBET_SUPPORTED_MARKETS = {"h2h", "spreads", "totals"}
PUREBET_ACTIVE_LEAGUES_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "mapping": {},
    "meta": {},
}


class ScannerError(Exception):
    """Raised for recoverable scanner issues."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _resolve_purebet_enabled(value: Optional[bool]) -> bool:
    if value is None:
        return PUREBET_ENV_ENABLED
    return bool(value)


def _provider_env_enabled(provider_key: str, default: bool = False) -> bool:
    env_key = f"{re.sub(r'[^A-Za-z0-9]', '_', provider_key).upper()}_ENABLED"
    raw = os.getenv(env_key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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


def _resolve_enabled_provider_keys(
    include_purebet: Optional[bool],
    include_providers: Optional[Sequence[str]],
) -> List[str]:
    enabled_by_key = {
        key: _provider_env_enabled(
            key,
            default=PUREBET_ENV_ENABLED if key == PUREBET_BOOK_KEY else False,
        )
        for key in PROVIDER_FETCHERS
    }
    explicit_providers = _normalize_provider_keys(include_providers)
    if explicit_providers is not None:
        enabled_by_key = {key: False for key in PROVIDER_FETCHERS}
        for key in explicit_providers:
            enabled_by_key[key] = True
    if include_purebet is not None and PUREBET_BOOK_KEY in enabled_by_key:
        enabled_by_key[PUREBET_BOOK_KEY] = bool(include_purebet)
    return [key for key in PROVIDER_FETCHERS if enabled_by_key.get(key)]


def _empty_purebet_summary(enabled: bool) -> dict:
    return {
        "enabled": enabled,
        "events_merged": 0,
        "details": {"requested": 0, "success": 0, "failed": 0, "empty": 0, "retries": 0},
        "league_sync": {
            "live_updates": 0,
            "cache_hits": 0,
            "stale_cache_uses": 0,
            "dynamic_added": 0,
            "unresolved": 0,
        },
        "sports": [],
    }


def _empty_provider_summary(provider_key: str, enabled: bool) -> dict:
    return {
        "key": provider_key,
        "name": PROVIDER_TITLES.get(provider_key, provider_key),
        "enabled": enabled,
        "events_merged": 0,
        "sports": [],
    }


def _purebet_public_base() -> str:
    base = (PUREBET_ORIGIN or "").strip() or "https://purebet.io"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _purebet_event_url(event_id: object) -> str:
    raw = f"{event_id or ''}".strip()
    if not raw:
        return ""
    return f"{_purebet_public_base()}/event/{quote(raw, safe='')}"


def _clamp_commission(rate: float) -> float:
    if rate is None:
        return DEFAULT_COMMISSION
    return max(0.0, min(rate, 0.2))


def _purebet_headers() -> Dict[str, str]:
    headers = {}
    if PUREBET_ORIGIN:
        headers["Origin"] = PUREBET_ORIGIN
    if PUREBET_REFERER:
        headers["Referer"] = PUREBET_REFERER
    if PUREBET_USER_AGENT:
        headers["User-Agent"] = PUREBET_USER_AGENT
    return headers


def _purebet_min_stake() -> float:
    try:
        return max(0.0, float(PUREBET_MIN_STAKE_RAW))
    except (TypeError, ValueError):
        return 0.0


def _purebet_max_age_seconds() -> int:
    try:
        return max(0, int(float(PUREBET_MAX_AGE_SECONDS_RAW)))
    except (TypeError, ValueError):
        return 0


def _purebet_fuzzy_threshold() -> float:
    try:
        return max(0.0, min(float(PUREBET_FUZZY_THRESHOLD_RAW), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _purebet_market_workers() -> int:
    try:
        return max(1, int(float(PUREBET_MARKET_WORKERS_RAW)))
    except (TypeError, ValueError):
        return 8


def _purebet_market_retries() -> int:
    try:
        return max(0, int(float(PUREBET_MARKET_RETRIES_RAW)))
    except (TypeError, ValueError):
        return 2


def _purebet_retry_backoff() -> float:
    try:
        return max(0.0, float(PUREBET_RETRY_BACKOFF_RAW))
    except (TypeError, ValueError):
        return 0.4


def _purebet_league_sync_ttl() -> int:
    try:
        return max(0, int(float(PUREBET_LEAGUE_SYNC_TTL_RAW)))
    except (TypeError, ValueError):
        return 600


def _purebet_get_json(
    url: str,
    params: Dict[str, object],
    headers: Dict[str, str],
    retries: int,
    backoff_seconds: float,
    timeout: int = 30,
) -> Tuple[object, int]:
    """Return (json_payload, retries_used). Raises ScannerError on final failure."""
    last_error: Optional[ScannerError] = None
    retriable_status = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_error = ScannerError(f"Purebet network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if response.status_code >= 400:
            if response.status_code in retriable_status and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise ScannerError(
                f"Purebet API request failed ({response.status_code})",
                status_code=response.status_code,
            )
        try:
            return response.json(), attempt
        except ValueError as exc:
            last_error = ScannerError("Failed to parse Purebet API response")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
    if last_error:
        raise last_error
    raise ScannerError("Purebet request failed")


def _normalize_regions(regions: Optional[Sequence[str]]) -> List[str]:
    if not regions:
        return list(DEFAULT_REGION_KEYS)
    valid = [region for region in regions if region in REGION_CONFIG]
    return valid or list(DEFAULT_REGION_KEYS)


def _normalize_bookmakers(bookmakers: Optional[Sequence[str]]) -> List[str]:
    if not bookmakers:
        return []
    normalized = []
    seen = set()
    for book in bookmakers:
        if not isinstance(book, str):
            continue
        key = book.strip()
        if not key or key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized


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


def _load_event_list(path: str) -> List[dict]:
    if not path:
        raise ScannerError("Purebet source file path is empty")
    path_obj = Path(path)
    if not path_obj.exists():
        raise ScannerError(f"Purebet source file not found: {path}")
    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ScannerError(f"Failed to read Purebet source file: {exc}") from exc
    if not isinstance(payload, list):
        raise ScannerError("Purebet source file must be a JSON array of events")
    return [item for item in payload if isinstance(item, dict)]


def _normalize_purebet_events(events: List[dict]) -> List[dict]:
    normalized: List[dict] = []
    for event in events:
        event_id = (
            event.get("event")
            or event.get("_id")
            or event.get("eventId")
            or event.get("id")
        )
        event_url = _purebet_event_url(event_id)
        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        for book in bookmakers:
            if not isinstance(book, dict):
                continue
            if not book.get("key"):
                book["key"] = PUREBET_BOOK_KEY
            if not book.get("title"):
                book["title"] = PUREBET_TITLE
            if event_id and not (book.get("event_id") or book.get("eventId") or book.get("id")):
                book["event_id"] = event_id
            if event_url and not (book.get("event_url") or book.get("eventUrl") or book.get("url")):
                book["event_url"] = event_url
        normalized.append(event)
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
    home = _normalize_team_name(event.get("home_team"))
    away = _normalize_team_name(event.get("away_team"))
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


def _merge_bookmakers(target: List[dict], incoming: List[dict]) -> None:
    seen = set()
    for book in target:
        key = (book.get("key") or book.get("title") or "").strip().lower()
        if key:
            seen.add(key)
    for book in incoming:
        key = (book.get("key") or book.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        target.append(book)
        seen.add(key)


def _merge_events(base_events: List[dict], extra_events: List[dict]) -> List[dict]:
    index: Dict[Tuple[str, str, str, str], dict] = {}
    by_team: Dict[Tuple[str, str, str], List[Tuple[int, dict]]] = {}
    normalized_by_sport: Dict[str, List[Tuple[int, dict, str, str]]] = {}
    try:
        tolerance_minutes = int(EVENT_TIME_TOLERANCE_MINUTES)
    except ValueError:
        tolerance_minutes = 15
    tolerance_seconds = max(0, tolerance_minutes) * 60
    fuzzy_threshold = _purebet_fuzzy_threshold()
    for event in base_events:
        identity = _event_identity(event)
        if identity and identity not in index:
            index[identity] = event
        team_key = _event_team_key(event)
        if team_key:
            epoch = _event_time_seconds(event)
            if epoch is not None:
                by_team.setdefault(team_key, []).append((epoch, event))
        normalized_key = _event_team_key_normalized(event)
        if normalized_key:
            sport, home_norm, away_norm = normalized_key
            epoch = _event_time_seconds(event)
            if epoch is not None:
                normalized_by_sport.setdefault(sport, []).append(
                    (epoch, event, home_norm, away_norm)
                )
    for extra in extra_events:
        identity = _event_identity(extra)
        if identity and identity in index:
            base = index[identity]
            base_books = base.setdefault("bookmakers", [])
            extra_books = extra.get("bookmakers") or []
            if isinstance(base_books, list) and isinstance(extra_books, list):
                _merge_bookmakers(base_books, extra_books)
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
                    score_home = _team_similarity(home_norm, base_home)
                    score_away = _team_similarity(away_norm, base_away)
                    score = min(score_home, score_away)
                    if score < fuzzy_threshold:
                        continue
                    avg_score = (score_home + score_away) / 2.0
                    if best_score is None or avg_score > best_score:
                        best_score = avg_score
                        matched_event = base_event
        if matched_event is not None:
            base_books = matched_event.setdefault("bookmakers", [])
            extra_books = extra.get("bookmakers") or []
            if isinstance(base_books, list) and isinstance(extra_books, list):
                _merge_bookmakers(base_books, extra_books)
            continue
        base_events.append(extra)
        if identity:
            index[identity] = extra
        team_key = _event_team_key(extra)
        if team_key:
            epoch = _event_time_seconds(extra)
            if epoch is not None:
                by_team.setdefault(team_key, []).append((epoch, extra))
    return base_events


def _base_purebet_league_map() -> Dict[str, str]:
    mapping = {str(key): value for key, value in PUREBET_DEFAULT_LEAGUE_MAP.items()}
    if not PUREBET_LEAGUE_MAP_RAW:
        return mapping
    try:
        payload = json.loads(PUREBET_LEAGUE_MAP_RAW)
    except ValueError:
        return mapping
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str) and value.strip():
                mapping[str(key)] = value.strip()
    return mapping


def _infer_sport_key_from_active_league(league: dict) -> Optional[str]:
    if not isinstance(league, dict):
        return None
    sport_name = _normalize_text(league.get("sportName"))
    sport_id = str(league.get("sport") or "")
    league_name = _normalize_text(league.get("name"))
    abbr = _normalize_text(league.get("abbr"))
    country = _normalize_text(league.get("country"))
    haystack = " ".join(value for value in (league_name, abbr, country) if value)
    if not haystack:
        return None

    if "nfl" in haystack:
        return "americanfootball_nfl"
    if "nba" in haystack:
        return "basketball_nba"
    if "ncaa" in haystack and "basket" in haystack:
        return "basketball_ncaab"

    is_soccer = sport_name == "soccer" or sport_id == "29"
    if is_soccer:
        if "premier league" in haystack and ("eng" in abbr or "england" in haystack):
            return "soccer_epl"
        if "la liga" in haystack or ("spain" in haystack and "liga" in haystack):
            return "soccer_spain_la_liga"
        if "bundesliga" in haystack or ("germany" in haystack and "bundesliga" in haystack):
            return "soccer_germany_bundesliga"
        if "serie a" in haystack or ("italy" in haystack and "serie" in haystack):
            return "soccer_italy_serie_a"
        if "ligue 1" in haystack or ("france" in haystack and "ligue" in haystack):
            return "soccer_france_ligue_one"
        if "mls" in haystack or "major league soccer" in haystack:
            return "soccer_usa_mls"

    is_basketball = sport_name == "basketball" or sport_id == "4"
    if is_basketball:
        if "nba" in haystack:
            return "basketball_nba"
        if "ncaa" in haystack:
            return "basketball_ncaab"
    return None


def _build_dynamic_purebet_league_map(
    leagues: Sequence[dict], base_mapping: Dict[str, str]
) -> Tuple[Dict[str, str], dict]:
    dynamic: Dict[str, str] = {}
    unresolved_samples: List[str] = []
    total = 0
    inferred = 0
    unresolved = 0
    for league in leagues:
        if not isinstance(league, dict):
            continue
        league_id = league.get("_id") or league.get("id")
        if league_id is None:
            continue
        total += 1
        key = str(league_id)
        if key in base_mapping:
            continue
        inferred_sport = _infer_sport_key_from_active_league(league)
        if inferred_sport:
            dynamic[key] = inferred_sport
            inferred += 1
            continue
        unresolved += 1
        if len(unresolved_samples) < 5:
            name = league.get("name") or league.get("abbr") or key
            unresolved_samples.append(f"{key}:{name}")
    return dynamic, {
        "total": total,
        "inferred": inferred,
        "unresolved": unresolved,
        "unresolved_samples": unresolved_samples,
    }


def _load_purebet_league_map(
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    stats: Optional[dict] = None,
) -> Dict[str, str]:
    mapping = _base_purebet_league_map()
    if stats is not None:
        stats["league_sync_enabled"] = bool(PUREBET_LEAGUE_SYNC_ENABLED)
    if not PUREBET_LEAGUE_SYNC_ENABLED:
        if stats is not None:
            stats["league_sync_source"] = "disabled"
        return mapping
    if not base_url:
        if stats is not None:
            stats["league_sync_source"] = "no_base_url"
        return mapping
    now = time.time()
    ttl = _purebet_league_sync_ttl()
    cache_valid = ttl > 0 and now < float(PUREBET_ACTIVE_LEAGUES_CACHE.get("expires_at", 0.0))
    cached_mapping = PUREBET_ACTIVE_LEAGUES_CACHE.get("mapping") or {}
    cached_meta = PUREBET_ACTIVE_LEAGUES_CACHE.get("meta") or {}
    if cache_valid and isinstance(cached_mapping, dict):
        mapping.update(cached_mapping)
        if stats is not None:
            stats["league_sync_source"] = "cache"
            stats["league_sync_total_leagues"] = int(cached_meta.get("total", 0) or 0)
            stats["league_sync_dynamic_added"] = len(cached_mapping)
            stats["league_sync_unresolved"] = int(cached_meta.get("unresolved", 0) or 0)
            stats["league_sync_unresolved_samples"] = list(cached_meta.get("unresolved_samples", []))
        return mapping

    leagues_url = f"{base_url.rstrip('/')}/activeLeagues"
    retries = _purebet_market_retries()
    backoff = _purebet_retry_backoff()
    try:
        payload, retries_used = _purebet_get_json(
            leagues_url,
            {},
            headers or _purebet_headers(),
            retries=retries,
            backoff_seconds=backoff,
            timeout=30,
        )
    except ScannerError as exc:
        if isinstance(cached_mapping, dict):
            mapping.update(cached_mapping)
        if stats is not None:
            stats["league_sync_source"] = "stale_cache" if cached_mapping else "error"
            stats["league_sync_error"] = str(exc)
            stats["league_sync_dynamic_added"] = len(cached_mapping) if isinstance(cached_mapping, dict) else 0
        return mapping

    if not isinstance(payload, list):
        if isinstance(cached_mapping, dict):
            mapping.update(cached_mapping)
        if stats is not None:
            stats["league_sync_source"] = "stale_cache" if cached_mapping else "invalid_payload"
            stats["league_sync_error"] = "Purebet activeLeagues response must be a JSON array"
            stats["league_sync_dynamic_added"] = len(cached_mapping) if isinstance(cached_mapping, dict) else 0
        return mapping

    dynamic_mapping, meta = _build_dynamic_purebet_league_map(payload, mapping)
    expires_at = now + ttl if ttl > 0 else now
    PUREBET_ACTIVE_LEAGUES_CACHE["expires_at"] = expires_at
    PUREBET_ACTIVE_LEAGUES_CACHE["mapping"] = dynamic_mapping
    PUREBET_ACTIVE_LEAGUES_CACHE["meta"] = meta
    mapping.update(dynamic_mapping)
    if stats is not None:
        stats["league_sync_source"] = "live"
        stats["league_sync_retries"] = retries_used
        stats["league_sync_total_leagues"] = int(meta.get("total", 0) or 0)
        stats["league_sync_dynamic_added"] = len(dynamic_mapping)
        stats["league_sync_unresolved"] = int(meta.get("unresolved", 0) or 0)
        stats["league_sync_unresolved_samples"] = list(meta.get("unresolved_samples", []))
    return mapping


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
        return dt.datetime.utcfromtimestamp(timestamp).replace(microsecond=0).isoformat() + "Z"
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
                dt.datetime.fromisoformat(text[:-1])
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


def _resolve_purebet_sport_key(event: dict, league_map: Dict[str, str]) -> Optional[str]:
    for key in ("sport_key", "sportKey", "sport"):
        raw = event.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    league_id = event.get("leagueId") or event.get("league") or event.get("league_id")
    if league_id is None:
        return None
    return league_map.get(str(league_id))


def _is_moneyline_market(market_type: str) -> bool:
    if not market_type:
        return False
    for token in ("moneyline", "ml", "h2h", "match_winner", "matchwinner", "winner", "win"):
        if token in market_type:
            return True
    return False


def _normalize_purebet_market_type(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    if text in {"ah", "asian handicap", "asian_handicap", "handicap"}:
        return "AH"
    if text in {"ou", "over/under", "over_under", "totals", "total"}:
        return "OU"
    if text in {"1x2", "match winner", "match_winner", "matchwinner", "winner", "h2h", "ml", "moneyline"}:
        return "H2H"
    if text in {"btts", "both teams to score", "both_teams_to_score"}:
        return "BTTS"
    return text.upper()


def _parse_purebet_market_value(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text:
            return None
        if "/" in text:
            parts = [part for part in text.split("/") if part]
            if parts:
                text = parts[0]
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _purebet_is_recent(last_updated, max_age_seconds: int, now_epoch: int) -> bool:
    if max_age_seconds <= 0:
        return True
    timestamp = _safe_float(last_updated)
    if timestamp is None:
        return True
    if timestamp > 1e12:
        timestamp /= 1000.0
    age = now_epoch - int(timestamp)
    if age < 0:
        return True
    return age <= max_age_seconds


def _select_purebet_side(
    entries: Optional[Sequence[dict]], min_stake: float, max_age_seconds: int, now_epoch: int
) -> Optional[dict]:
    if not isinstance(entries, list):
        return None
    best = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        odds = _safe_float(entry.get("odds") or entry.get("price") or entry.get("decimalOdds"))
        if odds is None or odds <= 1:
            continue
        stake = _safe_float(entry.get("stake") or entry.get("maxStake") or entry.get("liquidity"))
        if min_stake > 0 and (stake is None or stake < min_stake):
            continue
        last_updated = entry.get("lastUpdated") or entry.get("last_update") or entry.get("timestamp")
        if not _purebet_is_recent(last_updated, max_age_seconds, now_epoch):
            continue
        if best is None or odds > best["odds"]:
            best = {"odds": odds, "stake": stake, "last_updated": last_updated}
    return best


def _normalize_purebet_markets_payload(
    payload: Sequence[dict],
    home: str,
    away: str,
    supported_markets: Sequence[str],
) -> List[dict]:
    if not isinstance(payload, list):
        return []
    supported = set(supported_markets or [])
    if not supported:
        return []
    min_stake = _purebet_min_stake()
    max_age_seconds = _purebet_max_age_seconds()
    now_epoch = int(dt.datetime.utcnow().timestamp())
    normalized: List[dict] = []
    for market in payload:
        if not isinstance(market, dict):
            continue
        period = market.get("period")
        if period not in (None, 1, "1"):
            continue
        market_type = _normalize_purebet_market_type(
            market.get("marketType") or market.get("type") or market.get("market_type")
        )
        if not market_type:
            continue
        if market_type == "AH" and "spreads" not in supported:
            continue
        if market_type == "OU" and "totals" not in supported:
            continue
        if market_type == "H2H" and "h2h" not in supported:
            continue
        if market_type == "BTTS":
            continue
        side0 = _select_purebet_side(
            market.get("side0odds"), min_stake, max_age_seconds, now_epoch
        )
        side1 = _select_purebet_side(
            market.get("side1odds"), min_stake, max_age_seconds, now_epoch
        )
        side2 = _select_purebet_side(
            market.get("side2odds"), min_stake, max_age_seconds, now_epoch
        )
        if not side0 or not side1:
            continue
        market_value = _parse_purebet_market_value(
            market.get("marketValue") or market.get("value") or market.get("market_value")
        )
        if market_type == "AH":
            if market_value is None:
                continue
            home_point = float(market_value)
            away_point = -home_point
            normalized.append(
                {
                    "key": "spreads",
                    "outcomes": [
                        {
                            "name": home,
                            "price": side0["odds"],
                            "point": home_point,
                            "stake": side0.get("stake"),
                            "last_updated": side0.get("last_updated"),
                        },
                        {
                            "name": away,
                            "price": side1["odds"],
                            "point": away_point,
                            "stake": side1.get("stake"),
                            "last_updated": side1.get("last_updated"),
                        },
                    ],
                }
            )
        elif market_type == "OU":
            if market_value is None:
                continue
            total = float(market_value)
            normalized.append(
                {
                    "key": "totals",
                    "outcomes": [
                        {
                            "name": "Over",
                            "price": side0["odds"],
                            "point": total,
                            "stake": side0.get("stake"),
                            "last_updated": side0.get("last_updated"),
                        },
                        {
                            "name": "Under",
                            "price": side1["odds"],
                            "point": total,
                            "stake": side1.get("stake"),
                            "last_updated": side1.get("last_updated"),
                        },
                    ],
                }
            )
        elif market_type == "H2H":
            outcomes = [
                {
                    "name": home,
                    "price": side0["odds"],
                    "stake": side0.get("stake"),
                    "last_updated": side0.get("last_updated"),
                },
                {
                    "name": away,
                    "price": side1["odds"],
                    "stake": side1.get("stake"),
                    "last_updated": side1.get("last_updated"),
                },
            ]
            if side2:
                outcomes.append(
                    {
                        "name": "Draw",
                        "price": side2["odds"],
                        "stake": side2.get("stake"),
                        "last_updated": side2.get("last_updated"),
                    }
                )
            normalized.append({"key": "h2h", "outcomes": outcomes})
    return normalized


def _fetch_purebet_event_markets(
    base_url: str,
    event: dict,
    supported_markets: Sequence[str],
    headers: Dict[str, str],
    retries: int,
    backoff_seconds: float,
) -> dict:
    event_id = event.get("id")
    if not event_id:
        return {"event_id": None, "markets": [], "retries_used": 0, "error": "missing_event_id"}
    market_url = f"{base_url.rstrip('/')}/markets"
    try:
        payload, retries_used = _purebet_get_json(
            market_url,
            {"event": event_id},
            headers,
            retries=retries,
            backoff_seconds=backoff_seconds,
            timeout=30,
        )
    except ScannerError as exc:
        return {
            "event_id": event_id,
            "markets": [],
            "retries_used": retries,
            "error": str(exc),
        }
    if not isinstance(payload, list):
        return {
            "event_id": event_id,
            "markets": [],
            "retries_used": retries_used,
            "error": "invalid_markets_payload",
        }
    markets = _normalize_purebet_markets_payload(
        payload,
        event.get("home_team") or "",
        event.get("away_team") or "",
        supported_markets,
    )
    return {
        "event_id": event_id,
        "markets": markets,
        "retries_used": retries_used,
        "error": None,
    }


def _normalize_purebet_h2h_markets(odds: Sequence[dict], home: str, away: str) -> List[dict]:
    groups: Dict[str, dict] = {}
    for item in odds:
        if not isinstance(item, dict):
            continue
        market = item.get("market") if isinstance(item.get("market"), dict) else {}
        market_id = market.get("id") or item.get("marketId") or "default"
        market_type = _normalize_text(
            market.get("type") or market.get("marketType") or item.get("marketType")
        )
        side = market.get("side") if "side" in market else item.get("side")
        try:
            side_val = int(side)
        except (TypeError, ValueError):
            continue
        if side_val not in (0, 1):
            continue
        price = _safe_float(item.get("odds") or item.get("price") or item.get("decimalOdds"))
        if price is None or price <= 1:
            continue
        point = market.get("point") if isinstance(market, dict) else None
        if point is None:
            point = item.get("point")
        if point not in (None, 0, 0.0, "0", "0.0"):
            if not _is_moneyline_market(market_type):
                continue
        group = groups.setdefault(str(market_id), {"type": market_type, "sides": {}})
        if market_type and not group["type"]:
            group["type"] = market_type
        group["sides"][side_val] = {"price": price}

    candidates = [
        group
        for group in groups.values()
        if 0 in group["sides"] and 1 in group["sides"]
    ]
    if not candidates:
        return []
    moneyline = [group for group in candidates if _is_moneyline_market(group["type"])]
    if moneyline:
        candidates = moneyline
    best = max(
        candidates,
        key=lambda group: min(group["sides"][0]["price"], group["sides"][1]["price"]),
    )
    return [
        {
            "key": "h2h",
            "outcomes": [
                {"name": home, "price": best["sides"][0]["price"]},
                {"name": away, "price": best["sides"][1]["price"]},
            ],
        }
    ]


def _normalize_purebet_v3_events(
    payload: Sequence[dict],
    sport_key: str,
    markets: Sequence[str],
    base_url: Optional[str] = None,
    league_map: Optional[Dict[str, str]] = None,
    allow_empty_markets: bool = False,
) -> List[dict]:
    supported_markets = PUREBET_SUPPORTED_MARKETS.intersection(markets or [])
    if not supported_markets:
        return []
    if league_map is None:
        league_map = _load_purebet_league_map()
    normalized: List[dict] = []
    for event in payload:
        if not isinstance(event, dict):
            continue
        event_sport_key = _resolve_purebet_sport_key(event, league_map)
        if not event_sport_key:
            continue
        if sport_key and event_sport_key != sport_key:
            continue
        home = (event.get("homeTeam") or event.get("home_team") or "").strip()
        away = (event.get("awayTeam") or event.get("away_team") or "").strip()
        commence = _normalize_commence_time(
            event.get("startTime") or event.get("start_time") or event.get("start")
        )
        if not (home and away and commence):
            continue
        event_id = (
            event.get("event")
            or event.get("_id")
            or event.get("eventId")
            or event.get("id")
        )
        event_url = _purebet_event_url(event_id)
        markets_out: List[dict] = []
        odds = event.get("odds")
        if isinstance(odds, list) and "h2h" in supported_markets:
            markets_out.extend(_normalize_purebet_h2h_markets(odds, home, away))
        if not markets_out and not allow_empty_markets:
            continue
        normalized.append(
            {
                "id": event_id,
                "sport_key": event_sport_key,
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "league_id": event.get("leagueId") or event.get("league") or event.get("league_id"),
                "bookmakers": [
                    {
                        "key": PUREBET_BOOK_KEY,
                        "title": PUREBET_TITLE,
                        "event_id": event_id,
                        "event_url": event_url,
                        "markets": markets_out,
                    }
                ],
            }
        )
    return normalized


def fetch_purebet_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    source = PUREBET_SOURCE or "api"
    stats = {
        "source": source,
        "events_payload_count": 0,
        "events_normalized_count": 0,
        "events_returned_count": 0,
        "details_enabled": False,
        "details_requested": 0,
        "details_success": 0,
        "details_failed": 0,
        "details_empty": 0,
        "details_retries": 0,
        "details_workers": 0,
        "details_error_samples": [],
        "league_sync_enabled": False,
        "league_sync_source": "not_used",
        "league_sync_total_leagues": 0,
        "league_sync_dynamic_added": 0,
        "league_sync_unresolved": 0,
        "league_sync_unresolved_samples": [],
    }
    fetch_purebet_events.last_stats = stats
    if bookmakers:
        lowered = {str(book).strip().lower() for book in bookmakers if isinstance(book, str)}
        if PUREBET_BOOK_KEY not in lowered and PUREBET_TITLE.lower() not in lowered:
            stats["events_returned_count"] = 0
            return []
    if source == "file":
        events = _normalize_purebet_events(_load_event_list(PUREBET_SAMPLE_PATH))
        stats["events_payload_count"] = len(events)
        stats["events_normalized_count"] = len(events)
    else:
        base_url = PUREBET_API_BASE or PUREBET_DEFAULT_BASE
        if not base_url:
            raise ScannerError(
                "Purebet API base URL not configured. Set PUREBET_API_BASE or use PUREBET_SOURCE=file."
            )
        url = f"{base_url.rstrip('/')}/events"
        params = {"live": "true" if PUREBET_LIVE else "false"}
        payload, retries_used = _purebet_get_json(
            url,
            params,
            _purebet_headers(),
            retries=_purebet_market_retries(),
            backoff_seconds=_purebet_retry_backoff(),
            timeout=30,
        )
        stats["details_retries"] += retries_used
        if not isinstance(payload, list):
            raise ScannerError("Purebet API response must be a JSON array of events")
        stats["events_payload_count"] = len(payload)
        supported_markets = PUREBET_SUPPORTED_MARKETS.intersection(markets or [])
        needs_details = (
            PUREBET_MARKETS_ENABLED
            and bool({"spreads", "totals"}.intersection(supported_markets))
        )
        stats["details_enabled"] = bool(needs_details)
        league_map = _load_purebet_league_map(
            base_url=base_url,
            headers=_purebet_headers(),
            stats=stats,
        )
        events = _normalize_purebet_v3_events(
            payload,
            sport_key,
            markets,
            base_url=base_url,
            league_map=league_map,
            allow_empty_markets=needs_details,
        )
        stats["events_normalized_count"] = len(events)
        if needs_details and events:
            workers = min(_purebet_market_workers(), len(events))
            retries = _purebet_market_retries()
            backoff = _purebet_retry_backoff()
            stats["details_workers"] = workers
            stats["details_requested"] = len(events)
            event_map = {str(event.get("id")): event for event in events if event.get("id")}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _fetch_purebet_event_markets,
                        base_url,
                        event,
                        supported_markets,
                        _purebet_headers(),
                        retries,
                        backoff,
                    )
                    for event in events
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        stats["details_failed"] += 1
                        if len(stats["details_error_samples"]) < 5:
                            stats["details_error_samples"].append(str(exc))
                        continue
                    stats["details_retries"] += int(result.get("retries_used") or 0)
                    event_id = result.get("event_id")
                    event_obj = event_map.get(str(event_id))
                    if event_obj is None:
                        continue
                    error = result.get("error")
                    extra_markets = result.get("markets") or []
                    if error:
                        stats["details_failed"] += 1
                        if len(stats["details_error_samples"]) < 5:
                            stats["details_error_samples"].append(str(error))
                        continue
                    if not extra_markets:
                        stats["details_empty"] += 1
                        continue
                    stats["details_success"] += 1
                    bookmakers_list = event_obj.get("bookmakers")
                    if not isinstance(bookmakers_list, list) or not bookmakers_list:
                        fallback_event_id = event_obj.get("id")
                        fallback_event_url = _purebet_event_url(fallback_event_id)
                        event_obj["bookmakers"] = [
                            {
                                "key": PUREBET_BOOK_KEY,
                                "title": PUREBET_TITLE,
                                "event_id": fallback_event_id,
                                "event_url": fallback_event_url,
                                "markets": [],
                            }
                        ]
                        bookmakers_list = event_obj["bookmakers"]
                    book = bookmakers_list[0]
                    markets_list = book.get("markets")
                    if not isinstance(markets_list, list):
                        markets_list = []
                    if any(m.get("key") == "h2h" for m in extra_markets):
                        markets_list = [m for m in markets_list if m.get("key") != "h2h"]
                    markets_list.extend(extra_markets)
                    book["markets"] = markets_list
    if sport_key:
        events = [event for event in events if event.get("sport_key") == sport_key]
    events = [
        event
        for event in events
        if any(
            isinstance(book, dict)
            and isinstance(book.get("markets"), list)
            and book.get("markets")
            for book in (event.get("bookmakers") or [])
        )
    ]
    if bookmakers:
        filtered = []
        for event in events:
            books = event.get("bookmakers") or []
            if not isinstance(books, list):
                continue
            kept = [
                book
                for book in books
                if (book.get("key") or "").strip() in bookmakers
                or (book.get("title") or "").strip() in bookmakers
            ]
            if kept:
                event["bookmakers"] = kept
                filtered.append(event)
        events = filtered
    stats["events_returned_count"] = len(events)
    fetch_purebet_events.last_stats = stats
    return events


_legacy_fetch_purebet_events = fetch_purebet_events


def fetch_purebet_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    fetcher = PROVIDER_FETCHERS.get(PUREBET_BOOK_KEY)
    if not callable(fetcher):
        raise ScannerError("Purebet provider is not registered")
    events = fetcher(
        sport_key,
        markets,
        regions,
        bookmakers=bookmakers,
    )
    fetch_purebet_events.last_stats = getattr(fetcher, "last_stats", {}) or {}
    return events


fetch_purebet_events.last_stats = {}


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
    bookmaker: dict, commission_rate: float, is_exchange: bool
) -> Dict[Tuple[str, str], dict]:
    """Return mapping of (market_key, line_key) to vig-free odds."""
    line_map: Dict[Tuple[str, str], List[dict]] = {}
    for market in bookmaker.get("markets", []):
        market_key = market.get("key")
        if market_key not in ALLOWED_PLUS_EV_MARKETS:
            continue
        for outcome in market.get("outcomes", []):
            price = outcome.get("price")
            if not price or price <= 1:
                continue
            line_key = _line_key(market_key, outcome)
            if not line_key:
                continue
            try:
                price_val = float(price)
            except (TypeError, ValueError):
                continue
            adjusted_price = _apply_commission(price_val, commission_rate, is_exchange)
            line_map.setdefault((market_key, line_key), []).append(
                {
                    "name": (outcome.get("name") or "").strip().lower(),
                    "display_name": outcome.get("name") or "",
                    "price": adjusted_price,
                    "raw_price": price_val,
                    "point": outcome.get("point"),
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
                },
                second["name"]: {
                    "fair_odds": fair_b,
                    "true_probability": prob_b,
                    "sharp_odds": second["raw_price"],
                    "opponent_odds": first["raw_price"],
                    "opponent_name": first["display_name"],
                    "display_name": second["display_name"],
                    "point": second.get("point"),
                },
            },
        }
    return references


def _two_way_outcomes(bookmaker: dict) -> Dict[Tuple[str, str], List[dict]]:
    line_map: Dict[Tuple[str, str], List[dict]] = {}
    for market in bookmaker.get("markets", []):
        market_key = market.get("key")
        if market_key not in ALLOWED_PLUS_EV_MARKETS:
            continue
        for outcome in market.get("outcomes", []):
            price = outcome.get("price")
            if not price or price <= 1:
                continue
            line_key = _line_key(market_key, outcome)
            if not line_key:
                continue
            try:
                display_price = float(price)
            except (TypeError, ValueError):
                continue
            line_map.setdefault((market_key, line_key), []).append(
                {
                    "name": (outcome.get("name") or "").strip().lower(),
                    "display_name": outcome.get("name") or "",
                    "price": display_price,
                    "point": outcome.get("point"),
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
            message = payload.get("message") or payload.get("error")
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

    def request(self, url: str, params: Dict[str, str]) -> requests.Response:
        if not self._keys:
            raise ScannerError("API key is required", status_code=401)
        last_error: Optional[ScannerError] = None
        for _ in range(len(self._keys)):
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


OutcomeInfo = Dict[str, object]
LineMap = Dict[str, Dict[str, OutcomeInfo]]


def _line_key(market: str, outcome: dict) -> Optional[str]:
    if market == "h2h":
        return "moneyline"
    point = outcome.get("point")
    if point is None:
        return f"{market}_nopoint"
    try:
        point_val = float(point)
    except (TypeError, ValueError):
        return None
    if market == "spreads":
        return f"spread_{abs(point_val):.2f}"
    if market == "totals":
        return f"total_{point_val:.2f}"
    return f"{market}_{point_val:.2f}"


def _record_best_prices(
    markets: List[dict], market_key: str, commission_rate: float
) -> LineMap:
    lines: LineMap = {}
    for book in markets:
        bookmaker = book.get("title") or book.get("key")
        bookmaker_key = str(book.get("key") or bookmaker or "").strip()
        bookmaker_key_normalized = bookmaker_key.lower()
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
                price = outcome.get("price")
                if not price or price <= 1:
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
                name = outcome.get("name", "")
                existing = entry.get(name)
                display_price = float(price)
                is_exchange = bookmaker_key_normalized in EXCHANGE_KEYS
                effective_price = _apply_commission(display_price, commission_rate, is_exchange)
                if not existing or effective_price > existing["effective_price"]:
                    entry[name] = {
                        "effective_price": effective_price,
                        "display_price": display_price,
                        "bookmaker": bookmaker,
                        "bookmaker_key": bookmaker_key,
                        "name": name,
                        "point": normalized_point,
                        "max_stake": stake_value,
                        "is_exchange": is_exchange,
                        "book_event_id": book_event_id,
                        "book_event_url": book_event_url,
                    }
    return lines


def _available_markets(game: dict) -> List[str]:
    keys = set()
    for book in game.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key")
            if key:
                keys.add(key)
    return sorted(keys)


def _collect_market_entries(
    game: dict, market_key: str, stake_total: float, commission_rate: float
) -> List[dict]:
    bookmakers = game.get("bookmakers", [])
    markets = [
        {
            "key": book.get("key"),
            "title": book.get("title") or book.get("key"),
            "event_id": book.get("event_id") or book.get("eventId") or book.get("id"),
            "event_url": book.get("event_url") or book.get("eventUrl") or book.get("url"),
            "markets": book.get("markets", []),
        }
        for book in bookmakers
    ]
    best_lines = _record_best_prices(markets, market_key, commission_rate)
    entries = []
    for line_key, offers in best_lines.items():
        # Skip lines that aren't genuine two-way markets (e.g., moneyline with a draw option).
        if len(offers) != 2:
            continue
        outcomes = list(offers.values())
        if market_key == "spreads":
            try:
                point_values = [float(o.get("point")) for o in outcomes]
            except (TypeError, ValueError):
                continue
            if any(p is None for p in point_values):
                continue
            # Require opposite sides of the spread (one positive, one negative) and different teams.
            if point_values[0] * point_values[1] >= 0:
                continue
            outcome_names = {o.get("name", "").strip().lower() for o in outcomes}
            if len(outcome_names) < 2:
                continue
        has_exchange = any(o.get("is_exchange") for o in outcomes)
        outcome_payload = [
            {
                "outcome": o["name"],
                "bookmaker": o["bookmaker"],
                "bookmaker_key": o.get("bookmaker_key"),
                "price": o["display_price"],
                "effective_price": o["effective_price"],
                "point": o.get("point"),
                "max_stake": o.get("max_stake"),
                "is_exchange": o.get("is_exchange", False),
                "book_event_id": o.get("book_event_id"),
                "book_event_url": o.get("book_event_url"),
            }
            for o in outcomes
        ]
        net_stake_info = _calculate_stakes(outcomes, stake_total, price_field="effective_price")
        gross_stake_info = (
            _calculate_stakes(outcomes, stake_total, price_field="display_price") if has_exchange else None
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
    game: dict, market_key: str, stake_total: float, commission_rate: float
) -> List[dict]:
    if market_key not in MIDDLE_MARKETS or stake_total <= 0:
        return []
    bookmakers = game.get("bookmakers", [])
    offers = []
    for book in bookmakers:
        bookmaker_title = book.get("title") or book.get("key")
        bookmaker_key = book.get("key") or bookmaker_title
        markets = book.get("markets", [])
        for market in markets:
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []):
                point = outcome.get("point")
                price = outcome.get("price")
                if point is None or not price or price <= 1:
                    continue
                name = outcome.get("name") or ""
                try:
                    point_value = float(point)
                except (TypeError, ValueError):
                    continue
                display_price = float(price)
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
                        "is_exchange": is_exchange,
                    }
                )
    entries: List[dict] = []
    seen_pairs = set()
    for offer_a, offer_b in itertools.combinations(offers, 2):
        if offer_a["bookmaker_key"] == offer_b["bookmaker_key"]:
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
) -> List[dict]:
    bookmakers = game.get("bookmakers", [])
    sharp_meta = None
    sharp_bookmaker = None
    for candidate in sharp_priority:
        for book in bookmakers:
            if book.get("key") == candidate.get("key"):
                sharp_meta = candidate
                sharp_bookmaker = book
                break
        if sharp_bookmaker:
            break
    if not sharp_bookmaker or not sharp_meta:
        return []
    is_sharp_exchange = sharp_meta.get("type") == "exchange"
    sharp_reference = _build_sharp_reference(sharp_bookmaker, commission_rate, is_sharp_exchange)
    if not sharp_reference:
        return []
    opportunities: List[dict] = []
    for book in bookmakers:
        key = book.get("key")
        if not key:
            continue
        if key == sharp_meta.get("key"):
            continue
        if SOFT_BOOK_KEY_SET and key not in SOFT_BOOK_KEY_SET:
            continue
        bookmaker_title = book.get("title") or key
        is_exchange = key in EXCHANGE_KEYS
        soft_lines = _two_way_outcomes(book)
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

    stake_a, stake_b = _calculate_middle_stakes(
        side_a["effective_price"], side_b["effective_price"], stake_total
    )
    if stake_a <= 0 or stake_b <= 0:
        return None
    outcomes = _calculate_middle_outcomes(stake_a, stake_b, side_a["effective_price"], side_b["effective_price"])
    ev_dollars = _calculate_middle_ev(
        outcomes["win_both_profit"],
        outcomes["side_a_wins_profit"],
        outcomes["side_b_wins_profit"],
        middle_probability,
    )
    ev_percent = round((ev_dollars / stake_total) * 100, 2) if stake_total else 0.0
    has_exchange = side_a["is_exchange"] or side_b["is_exchange"]
    gross_ev_percent = None
    if has_exchange:
        gross_stake_a, gross_stake_b = _calculate_middle_stakes(
            side_a["display_price"], side_b["display_price"], stake_total
        )
        if gross_stake_a > 0 and gross_stake_b > 0:
            gross_outcomes = _calculate_middle_outcomes(
                gross_stake_a, gross_stake_b, side_a["display_price"], side_b["display_price"]
            )
            gross_ev = _calculate_middle_ev(
                gross_outcomes["win_both_profit"],
                gross_outcomes["side_a_wins_profit"],
                gross_outcomes["side_b_wins_profit"],
                middle_probability,
            )
            gross_ev_percent = round((gross_ev / stake_total) * 100, 2) if stake_total else 0.0

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
            "is_exchange": side_a["is_exchange"],
        },
        {
            "team": side_b["team"],
            "line": side_b["line"],
            "price": side_b["display_price"],
            "effective_price": side_b["effective_price"],
            "bookmaker": side_b["bookmaker"],
            "is_exchange": side_b["is_exchange"],
        },
    )
    stakes_payload = {
        "total": stake_total,
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
            "breakdown": [],
            "guaranteed_profit": 0.0,
            "roi_percent": 0.0,
        }
    inverses = []
    for outcome in outcomes:
        price = outcome.get(price_field)
        if not price or price <= 0:
            return {
                "total": 0.0,
                "breakdown": [],
                "guaranteed_profit": 0.0,
                "roi_percent": 0.0,
            }
        inverses.append(1 / price)
    inverse_sum = sum(inverses)
    if inverse_sum <= 0:
        return {
            "total": 0.0,
            "breakdown": [],
            "guaranteed_profit": 0.0,
            "roi_percent": 0.0,
        }
    breakdown = []
    for outcome, inv in zip(outcomes, inverses):
        fraction = inv / inverse_sum
        stake_value = round(stake_total * fraction, 2)
        price_used = outcome.get(price_field)
        display_price = outcome.get("display_price", price_used)
        payout = round(stake_value * price_used, 2)
        breakdown.append(
            {
                "outcome": outcome["name"],
                "bookmaker": outcome["bookmaker"],
                "price": display_price,
                "effective_price": outcome.get("effective_price", price_used),
                "point": outcome.get("point"),
                "stake": stake_value,
                "payout": payout,
                "fraction": fraction,
                "is_exchange": outcome.get("is_exchange", False),
            }
        )
    min_payout = min(item["payout"] for item in breakdown) if breakdown else 0.0
    guaranteed = round(min_payout - stake_total, 2)
    roi = round((guaranteed / stake_total) * 100, 4) if stake_total else 0.0
    return {
        "total": stake_total,
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
    for opp in opportunities:
        key = opp.get("sport_display") or opp.get("sport") or "unknown"
        by_sport[key] = by_sport.get(key, 0) + 1
        roi = opp.get("roi_percent", 0)
        for lower, upper, label in ROI_BANDS:
            if lower <= roi < upper:
                band_counts[label] += 1
                break
    return {
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
            opp.get("event"),
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
            opp.get("market"),
            (bet.get("outcome") or "").strip().lower(),
            normalized_point,
        )
        existing = best_by_key.get(key)
        if not existing or (opp.get("edge_percent", 0) > existing.get("edge_percent", 0)):
            best_by_key[key] = opp
    return list(best_by_key.values())


def run_scan(
    api_key: str | Sequence[str],
    sports: Optional[List[str]] = None,
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
    include_purebet: Optional[bool] = None,
    include_providers: Optional[Sequence[str]] = None,
) -> dict:
    api_keys = _normalize_api_keys(api_key)
    if not api_keys:
        return {"success": False, "error": "API key is required", "error_code": 400}
    if stake_amount is None or stake_amount <= 0:
        stake_amount = DEFAULT_STAKE_AMOUNT
    all_markets = bool(all_markets)
    enabled_provider_keys = _resolve_enabled_provider_keys(include_purebet, include_providers)
    enabled_provider_set = set(enabled_provider_keys)
    normalized_regions = _normalize_regions(regions)
    normalized_regions = _ensure_sharp_region(normalized_regions, sharp_book or DEFAULT_SHARP_BOOK)
    normalized_bookmakers = _normalize_bookmakers(bookmakers)
    provider_bookmaker_keys = _normalize_provider_keys(normalized_bookmakers) or []
    if provider_bookmaker_keys:
        enabled_provider_set.update(provider_bookmaker_keys)
        enabled_provider_keys = [key for key in PROVIDER_FETCHERS if key in enabled_provider_set]
    include_purebet = PUREBET_BOOK_KEY in enabled_provider_set
    if not normalized_regions:
        return {
            "success": False,
            "error": "At least one region must be selected",
            "error_code": 400,
        }
    commission_rate = _clamp_commission(commission_rate)
    api_pool = ApiKeyPool(api_keys)
    try:
        sports_list = fetch_sports(api_pool)
    except ScannerError as exc:
        return {"success": False, "error": str(exc), "error_code": 500}

    filtered = filter_sports(sports_list, sports or DEFAULT_SPORT_KEYS, all_sports)
    provider_summaries = {
        key: _empty_provider_summary(key, key in enabled_provider_set)
        for key in PROVIDER_FETCHERS
    }
    purebet_summary = _empty_purebet_summary(include_purebet)
    if not filtered:
        arb_summary = _summaries([], 0, 0, 0.0, api_pool.calls_made)
        middle_summary = _middle_summary([])
        return {
            "success": True,
            "scan_time": _iso_now(),
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
            "sport_errors": [],
            "partial": False,
            "regions": normalized_regions,
            "commission_rate": commission_rate,
            "purebet": purebet_summary,
            "custom_providers": provider_summaries,
        }

    arb_opportunities: List[dict] = []
    middle_opportunities: List[dict] = []
    plus_ev_opportunities: List[dict] = []
    events_scanned = 0
    total_profit = 0.0
    sport_errors: List[dict] = []
    successful_sports = 0
    sharp_priority = _sharp_priority(sharp_book or DEFAULT_SHARP_BOOK)

    for sport in filtered:
        sport_key = sport.get("key")
        if not sport_key:
            continue
        base_markets = markets_for_sport(sport_key)
        try:
            events = fetch_odds_for_sport(
                api_pool,
                sport_key,
                base_markets,
                normalized_regions,
                bookmakers=normalized_bookmakers,
            )
        except ScannerError as exc:
            sport_errors.append(
                {
                    "sport_key": sport_key,
                    "sport": sport.get("title")
                    or SPORT_DISPLAY_NAMES.get(sport_key, sport_key),
                    "error": str(exc),
                }
            )
            continue
        for provider_key in enabled_provider_keys:
            fetch_provider_events = PROVIDER_FETCHERS.get(provider_key)
            if not callable(fetch_provider_events):
                continue
            provider_summary = provider_summaries.setdefault(
                provider_key,
                _empty_provider_summary(provider_key, True),
            )
            provider_title = provider_summary.get("name") or PROVIDER_TITLES.get(
                provider_key, provider_key
            )
            try:
                provider_events = fetch_provider_events(
                    sport_key,
                    base_markets,
                    normalized_regions,
                    bookmakers=normalized_bookmakers,
                )
                stats = getattr(fetch_provider_events, "last_stats", {}) or {}
                provider_summary["sports"].append(
                    {
                        "sport_key": sport_key,
                        "events_returned": len(provider_events),
                        "stats": stats,
                    }
                )
                if provider_key == PUREBET_BOOK_KEY:
                    purebet_summary["sports"].append(
                        {
                            "sport_key": sport_key,
                            "events_payload": stats.get("events_payload_count", 0),
                            "events_normalized": stats.get("events_normalized_count", 0),
                            "events_returned": stats.get("events_returned_count", 0),
                            "details_enabled": stats.get("details_enabled", False),
                            "details_requested": stats.get("details_requested", 0),
                            "details_success": stats.get("details_success", 0),
                            "details_failed": stats.get("details_failed", 0),
                            "details_empty": stats.get("details_empty", 0),
                            "details_retries": stats.get("details_retries", 0),
                            "details_workers": stats.get("details_workers", 0),
                            "league_sync_source": stats.get("league_sync_source"),
                            "league_sync_total_leagues": stats.get("league_sync_total_leagues", 0),
                            "league_sync_dynamic_added": stats.get("league_sync_dynamic_added", 0),
                            "league_sync_unresolved": stats.get("league_sync_unresolved", 0),
                            "league_sync_unresolved_samples": stats.get("league_sync_unresolved_samples", []),
                            "errors": stats.get("details_error_samples", []),
                        }
                    )
                    purebet_summary["details"]["requested"] += int(
                        stats.get("details_requested", 0) or 0
                    )
                    purebet_summary["details"]["success"] += int(
                        stats.get("details_success", 0) or 0
                    )
                    purebet_summary["details"]["failed"] += int(
                        stats.get("details_failed", 0) or 0
                    )
                    purebet_summary["details"]["empty"] += int(
                        stats.get("details_empty", 0) or 0
                    )
                    purebet_summary["details"]["retries"] += int(
                        stats.get("details_retries", 0) or 0
                    )
                    league_source = stats.get("league_sync_source")
                    if league_source == "live":
                        purebet_summary["league_sync"]["live_updates"] += 1
                    elif league_source == "cache":
                        purebet_summary["league_sync"]["cache_hits"] += 1
                    elif league_source == "stale_cache":
                        purebet_summary["league_sync"]["stale_cache_uses"] += 1
                    purebet_summary["league_sync"]["dynamic_added"] += int(
                        stats.get("league_sync_dynamic_added", 0) or 0
                    )
                    purebet_summary["league_sync"]["unresolved"] += int(
                        stats.get("league_sync_unresolved", 0) or 0
                    )
                if provider_events:
                    provider_summary["events_merged"] += len(provider_events)
                    if provider_key == PUREBET_BOOK_KEY:
                        purebet_summary["events_merged"] += len(provider_events)
                    events = _merge_events(events, provider_events)
            except Exception as exc:
                provider_summary["sports"].append(
                    {
                        "sport_key": sport_key,
                        "error": str(exc),
                    }
                )
                if provider_key == PUREBET_BOOK_KEY:
                    purebet_summary["sports"].append(
                        {
                            "sport_key": sport_key,
                            "error": str(exc),
                        }
                    )
                sport_errors.append(
                    {
                        "sport_key": sport_key,
                        "sport": sport.get("title")
                        or SPORT_DISPLAY_NAMES.get(sport_key, sport_key),
                        "error": f"{provider_title}: {exc}",
                    }
                )
        successful_sports += 1
        events_scanned += len(events)
        for game in events:
            game["sport_key"] = sport_key
            game["sport_title"] = sport.get("title")
            game["sport_display"] = SPORT_DISPLAY_NAMES.get(sport_key, sport_key)
            arb_markets = _available_markets(game) if all_markets else base_markets
            for market_key in arb_markets:
                new_entries = _collect_market_entries(
                    game, market_key, stake_amount, commission_rate
                )
                for entry in new_entries:
                    total_profit += entry["stakes"].get("guaranteed_profit", 0.0)
                arb_opportunities.extend(new_entries)
                if market_key in MIDDLE_MARKETS:
                    middle_entries = _collect_middle_opportunities(
                        game, market_key, stake_amount, commission_rate
                    )
                    middle_opportunities.extend(middle_entries)
            plus_entries = _collect_plus_ev_opportunities(
                game,
                base_markets,
                sharp_priority,
                commission_rate,
                min_edge_percent,
                bankroll,
                kelly_fraction,
            )
            plus_ev_opportunities.extend(plus_entries)

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
    return {
        "success": True,
        "scan_time": _iso_now(),
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
        "purebet": purebet_summary,
        "custom_providers": provider_summaries,
    }
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
        return 0.0, 0.0
    p = max(0.0, min(true_probability, 1.0))
    q = 1.0 - p
    b = odds - 1.0
    if b <= 0:
        return 0.0, 0.0
    kelly_fraction = (b * p - q) / b
    if kelly_fraction <= 0:
        return 0.0, 0.0
    fraction = max(0.0, min(fraction, 1.0))
    recommended_fraction = kelly_fraction * fraction
    stake = round(bankroll * recommended_fraction, 2)
    return round(kelly_fraction * 100, 2), round(recommended_fraction * 100, 2), stake

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import threading
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import httpx
import requests

from ._async_http import get_shared_client, request_json

PROVIDER_KEY = "sx_bet"
PROVIDER_TITLE = "SX Bet"

SX_BET_SOURCE = os.getenv("SX_BET_SOURCE", "api").strip().lower()
SX_BET_API_BASE = os.getenv("SX_BET_API_BASE", "https://api.sx.bet").strip()
SX_BET_PUBLIC_BASE = os.getenv("SX_BET_PUBLIC_BASE", "https://sx.bet").strip()
SX_BET_BASE_TOKEN = os.getenv(
    "SX_BET_BASE_TOKEN",
    "0x6629Ce1Cf35Cc1329ebB4F63202F3f197b3F050B",
).strip()
SX_BET_TIMEOUT_RAW = os.getenv("SX_BET_TIMEOUT_SECONDS", "20").strip()
SX_BET_RETRIES_RAW = os.getenv("SX_BET_RETRIES", "2").strip()
SX_BET_RETRY_BACKOFF_RAW = os.getenv("SX_BET_RETRY_BACKOFF", "0.5").strip()
SX_BET_PAGE_TTL_RAW = os.getenv("SX_BET_PAGE_CACHE_TTL", "8").strip()
SX_BET_ODDS_CACHE_TTL_RAW = os.getenv("SX_BET_ODDS_CACHE_TTL", "4").strip()
SX_BET_ORDER_CACHE_TTL_RAW = os.getenv("SX_BET_ORDER_CACHE_TTL", "2").strip()
SX_BET_BASE_TOKEN_DECIMALS_RAW = os.getenv("SX_BET_BASE_TOKEN_DECIMALS", "6").strip()
SX_BET_FIXTURE_SOURCE_RAW = os.getenv("SX_BET_FIXTURE_SOURCE", "auto").strip().lower()
SX_BET_MARKETS_ACTIVE_MAX_PAGES_RAW = os.getenv("SX_BET_MARKETS_ACTIVE_MAX_PAGES", "4").strip()
SX_BET_MARKETS_ACTIVE_LEAGUE_WORKERS_RAW = os.getenv(
    "SX_BET_MARKETS_ACTIVE_LEAGUE_WORKERS",
    "8",
).strip()
SX_BET_LEAGUES_CACHE_TTL_RAW = os.getenv("SX_BET_LEAGUES_CACHE_TTL", "300").strip()
SX_BET_MARKETS_ACTIVE_ONLY_MAIN_LINE = os.getenv(
    "SX_BET_MARKETS_ACTIVE_ONLY_MAIN_LINE",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
SX_BET_USER_AGENT = os.getenv(
    "SX_BET_USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
).strip()
SX_BET_BEARER_TOKEN = os.getenv("SX_BET_BEARER_TOKEN", "").strip()
SX_BET_API_KEY = os.getenv("SX_BET_API_KEY", "").strip()
SX_BET_COOKIE = os.getenv("SX_BET_COOKIE", "").strip()
SX_BET_LEAGUE_MAP_RAW = os.getenv("SX_BET_LEAGUE_MAP", "").strip()
SX_BET_WS_ENABLED = os.getenv("SX_BET_WS_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SX_BET_WS_STARTUP_WAIT_SECONDS_RAW = os.getenv("SX_BET_WS_STARTUP_WAIT_SECONDS", "1.5").strip()
SX_BET_WS_QUOTE_MAX_AGE_SECONDS_RAW = os.getenv("SX_BET_WS_QUOTE_MAX_AGE_SECONDS", "5").strip()
SX_BET_FIXTURE_STATE_BATCH_SIZE = 20

SX_SPORT_ID_MAP: Dict[str, int] = {
    "basketball_nba": 1,
    "basketball_ncaab": 1,
    "americanfootball_nfl": 8,
    "americanfootball_ncaaf": 8,
    "baseball_mlb": 3,
    "icehockey_nhl": 2,
    "soccer_epl": 5,
    "soccer_spain_la_liga": 5,
    "soccer_germany_bundesliga": 5,
    "soccer_italy_serie_a": 5,
    "soccer_france_ligue_one": 5,
    "soccer_usa_mls": 5,
}

SPORT_LEAGUE_HINTS: Dict[str, Sequence[str]] = {
    "basketball_nba": ("nba",),
    "basketball_ncaab": ("ncaa", "college"),
    "americanfootball_nfl": ("nfl",),
    "americanfootball_ncaaf": ("ncaaf", "college football", "ncaa football"),
    "baseball_mlb": ("mlb", "major league baseball"),
    "icehockey_nhl": ("nhl", "national hockey league"),
    "soccer_epl": ("premier league", "epl"),
    "soccer_spain_la_liga": ("la liga",),
    "soccer_germany_bundesliga": ("bundesliga",),
    "soccer_italy_serie_a": ("serie a",),
    "soccer_france_ligue_one": ("ligue 1", "ligue one"),
    "soccer_usa_mls": ("mls", "major league soccer"),
}
SX_PUBLIC_SPORT_SLUGS_BY_PREFIX: Dict[str, str] = {
    "basketball": "basketball",
    "baseball": "baseball",
    "icehockey": "hockey",
    "soccer": "soccer",
    "americanfootball": "football",
}
SX_PUBLIC_SPORT_SLUGS_BY_ID: Dict[int, str] = {
    1: "basketball",
    2: "hockey",
    3: "baseball",
    5: "soccer",
    8: "football",
}
SX_PUBLIC_LEAGUE_SLUG_OVERRIDES: Dict[str, str] = {
    "major league soccer": "mls",
}

UPCOMING_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "entries": {},
}
ODDS_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "entries": {},
}

SX_MARKET_TYPE_ID_BASE_KEY: Dict[int, str] = {
    2: "totals",
    3: "spreads",
    28: "totals",
    52: "h2h",
    226: "h2h",
    236: "totals",
    342: "spreads",
    1618: "h2h",
}
ORDERS_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "entries": {},
}
LEAGUES_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "entries": {},
}
REALTIME_MANAGER: Optional["SXBetRealtimeManager"] = None
REALTIME_MANAGER_LOCK = threading.Lock()


class ProviderError(Exception):
    """Raised for provider-specific recoverable issues."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _int_or_default(value: str, default: int, min_value: int = 0) -> int:
    try:
        return max(min_value, int(float(value)))
    except (TypeError, ValueError):
        return default


def _float_or_default(value: str, default: float, min_value: float = 0.0) -> float:
    try:
        return max(min_value, float(value))
    except (TypeError, ValueError):
        return default


def _sx_ws_startup_wait_seconds() -> float:
    return _float_or_default(SX_BET_WS_STARTUP_WAIT_SECONDS_RAW, 1.5, min_value=0.0)


def _sx_ws_quote_max_age_seconds() -> float:
    return _float_or_default(SX_BET_WS_QUOTE_MAX_AGE_SECONDS_RAW, 5.0, min_value=0.0)


def _as_int(value: object) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fixture_source_mode(context: Optional[dict] = None) -> str:
    if isinstance(context, dict) and bool(context.get("live")):
        return "markets_active"
    mode = _normalize_text(SX_BET_FIXTURE_SOURCE_RAW).lower()
    if mode in {"summary", "summary_upcoming", "legacy"}:
        return "summary"
    if mode in {"markets_active", "market_active", "markets"}:
        return "markets_active"
    return "auto"


def _markets_active_max_pages() -> int:
    return _int_or_default(SX_BET_MARKETS_ACTIVE_MAX_PAGES_RAW, 4, min_value=1)


def _markets_active_league_workers() -> int:
    return _int_or_default(SX_BET_MARKETS_ACTIVE_LEAGUE_WORKERS_RAW, 8, min_value=1)


def _api_base() -> str:
    base = (SX_BET_API_BASE or "").strip() or "https://api.sx.bet"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _public_base() -> str:
    base = (SX_BET_PUBLIC_BASE or "").strip() or "https://sx.bet"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if SX_BET_USER_AGENT:
        headers["User-Agent"] = SX_BET_USER_AGENT
    if SX_BET_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {SX_BET_BEARER_TOKEN}"
    if SX_BET_API_KEY:
        headers["X-API-Key"] = SX_BET_API_KEY
    if SX_BET_COOKIE:
        headers["Cookie"] = SX_BET_COOKIE
    return headers


def _sx_best_odds_ws_enabled() -> bool:
    return bool(SX_BET_WS_ENABLED) and bool(_normalize_text(SX_BET_API_KEY)) and bool(_normalize_text(SX_BET_BASE_TOKEN))


def _sx_token_url() -> str:
    return f"{_api_base()}/user/token"


def _request_json(
    path: str,
    params: Optional[Dict[str, object]] = None,
    retries: Optional[int] = None,
    backoff_seconds: Optional[float] = None,
) -> Tuple[object, int]:
    if retries is None:
        retries = _int_or_default(SX_BET_RETRIES_RAW, 2, min_value=0)
    if backoff_seconds is None:
        backoff_seconds = _float_or_default(SX_BET_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    timeout = _int_or_default(SX_BET_TIMEOUT_RAW, 20, min_value=1)
    url = f"{_api_base()}/{path.lstrip('/')}"
    retriable_status = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    last_error: Optional[ProviderError] = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params or {}, headers=_headers(), timeout=timeout)
        except requests.RequestException as exc:
            last_error = ProviderError(f"SX Bet network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if response.status_code >= 400:
            if response.status_code in retriable_status and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise ProviderError(
                f"SX Bet API request failed ({response.status_code})",
                status_code=response.status_code,
            )
        try:
            return response.json(), attempt
        except ValueError as exc:
            last_error = ProviderError("Failed to parse SX Bet API response")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
    if last_error:
        raise last_error
    raise ProviderError("SX Bet request failed")


async def _request_json_async(
    client: httpx.AsyncClient,
    path: str,
    params: Optional[Dict[str, object]] = None,
    retries: Optional[int] = None,
    backoff_seconds: Optional[float] = None,
) -> Tuple[object, int]:
    if retries is None:
        retries = _int_or_default(SX_BET_RETRIES_RAW, 2, min_value=0)
    if backoff_seconds is None:
        backoff_seconds = _float_or_default(SX_BET_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    timeout = _int_or_default(SX_BET_TIMEOUT_RAW, 20, min_value=1)
    url = f"{_api_base()}/{path.lstrip('/')}"
    return await request_json(
        client,
        "GET",
        url,
        params=params or {},
        headers=_headers(),
        timeout=float(timeout),
        retries=retries,
        backoff_seconds=backoff_seconds,
        error_cls=ProviderError,
        network_error_prefix="SX Bet network error",
        parse_error_message="Failed to parse SX Bet API response",
        status_error_message=lambda status_code: f"SX Bet API request failed ({status_code})",
    )


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_commence_time(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1e12:
            timestamp /= 1000.0
        try:
            return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (OSError, OverflowError, ValueError):
            return None
    text = _normalize_text(value)
    if not text:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return f"{text}T00:00:00Z"
    try:
        if text.endswith("Z"):
            parsed = dt.datetime.fromisoformat(text[:-1] + "+00:00")
        else:
            parsed = dt.datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return (
            parsed.astimezone(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except ValueError:
        return None


def _normalize_status_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize_text(value).lower()).strip("_")


def _merge_live_state_payload(existing: Optional[dict], incoming: Optional[dict]) -> Optional[dict]:
    if not isinstance(existing, dict) or not existing:
        return dict(incoming) if isinstance(incoming, dict) and incoming else None
    if not isinstance(incoming, dict) or not incoming:
        return dict(existing)
    merged = dict(existing)
    incoming_status = _normalize_status_token(incoming.get("status"))
    existing_status = _normalize_status_token(merged.get("status"))
    if incoming.get("is_live") is True:
        merged["is_live"] = True
        if incoming_status:
            merged["status"] = incoming_status
    elif "is_live" not in merged and "is_live" in incoming:
        merged["is_live"] = bool(incoming.get("is_live"))
    if incoming_status in {"final", "finished", "closed", "resolved", "settled", "cancelled", "canceled"}:
        merged["status"] = incoming_status
        merged["is_live"] = False
    elif not existing_status and incoming_status:
        merged["status"] = incoming_status
    for key in ("provider_status", "updated_at", "market_status", "in_play_status", "live_enabled"):
        if not merged.get(key) and incoming.get(key) not in (None, ""):
            merged[key] = incoming.get(key)
    return merged


def _sx_live_state_payload(*payloads: object) -> Optional[dict]:
    merged: Optional[dict] = None
    now_utc = dt.datetime.now(dt.timezone.utc)
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        current: Dict[str, object] = {}
        live_enabled = payload.get("liveEnabled")
        if live_enabled is not None:
            current["live_enabled"] = bool(live_enabled)
        in_play_status = _normalize_status_token(
            payload.get("inPlayStatus")
            or payload.get("gameState")
            or payload.get("matchStatus")
        )
        if in_play_status:
            current["in_play_status"] = in_play_status
            if in_play_status in {"preplay", "prematch", "scheduled", "not_started"}:
                current["is_live"] = False
                current["status"] = "scheduled"
            elif in_play_status in {"inplay", "live", "playing"}:
                current["is_live"] = True
                current["status"] = "live"
        explicit_live = (
            _bool_or_none(payload.get("isLive"))
            if payload.get("isLive") is not None
            else _bool_or_none(payload.get("isInPlay"))
        )
        if explicit_live is None and payload.get("live") is not None:
            explicit_live = _bool_or_none(payload.get("live"))
        if explicit_live is not None:
            current["is_live"] = explicit_live
            current["status"] = "live" if explicit_live else "scheduled"
        status_token = _normalize_status_token(
            payload.get("orderStatus")
            or payload.get("marketStatus")
            or payload.get("status")
        )
        if status_token:
            current["provider_status"] = status_token
            if status_token in {"settled", "closed", "resolved", "cancelled", "canceled", "finished", "final"}:
                current["status"] = "final"
                current["is_live"] = False
            elif status_token in {"active", "open"}:
                current["market_status"] = status_token
        commence_time = _normalize_commence_time(
            payload.get("gameTime") or payload.get("startsAt") or payload.get("startTime")
        )
        if commence_time and current.get("status") != "final":
            commence_dt = _normalize_commence_time(commence_time)
            if commence_dt:
                try:
                    parsed_commence = dt.datetime.fromisoformat(str(commence_dt).replace("Z", "+00:00"))
                except ValueError:
                    parsed_commence = None
                if parsed_commence is not None:
                    if parsed_commence > now_utc:
                        current["is_live"] = False
                        current["status"] = "scheduled"
                    elif current.get("is_live") is None and status_token in {"active", "open"}:
                        current["is_live"] = True
                        current["status"] = "live"
        updated_at = payload.get("updatedAt") or payload.get("lastUpdated") or payload.get("modifiedAt")
        if updated_at not in (None, ""):
            current["updated_at"] = updated_at
        merged = _merge_live_state_payload(merged, current)
    return merged


def _sx_fixture_status_live_state_payload(status_code: object) -> Optional[dict]:
    normalized = _as_int(status_code)
    if normalized == 2:
        return {'is_live': True, 'status': 'live'}
    if normalized in {1, 9}:
        return {'is_live': False, 'status': 'scheduled'}
    if normalized in {3, 4, 5, 6, 7, 8}:
        return {'is_live': False, 'status': 'final'}
    return None


def _sx_live_scores_live_state_payload(payload: object) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    current_period = _normalize_text(payload.get('currentPeriod'))
    period_time = _normalize_text(payload.get('periodTime'))
    if not current_period and not period_time:
        return None
    live_state = {'is_live': True, 'status': 'live'}
    if current_period:
        live_state['period'] = current_period
    if period_time:
        live_state['clock'] = period_time
    return live_state


def _fixture_has_live_evidence(fixture: object) -> bool:
    if not isinstance(fixture, dict):
        return False
    live_state = fixture.get("live_state")
    if not isinstance(live_state, dict):
        return False
    if live_state.get("is_live") is True:
        return True
    status = _normalize_status_token(
        live_state.get("status")
        or live_state.get("in_play_status")
        or live_state.get("provider_status")
    )
    if status in {"live", "inplay", "playing"}:
        return True
    if status in {"final", "finished", "closed", "resolved", "settled", "cancelled", "canceled"}:
        return False
    provider_status = _normalize_status_token(live_state.get("provider_status"))
    market_status = _normalize_status_token(live_state.get("market_status"))
    live_enabled = bool(live_state.get("live_enabled") or live_state.get("liveEnabled"))
    if live_enabled and (provider_status in {"active", "open"} or market_status in {"active", "open"}):
        return True
    return False


async def _load_fixture_status_map_async(
    client: httpx.AsyncClient,
    event_ids: Sequence[str],
    retries: int,
    backoff_seconds: float,
) -> Dict[str, dict]:
    normalized_ids = list(
        dict.fromkeys(
            item for item in (_normalize_text(event_id) for event_id in event_ids) if item
        )
    )
    if not normalized_ids:
        return {}
    merged: Dict[str, dict] = {}
    for chunk in _chunked(normalized_ids, SX_BET_FIXTURE_STATE_BATCH_SIZE):
        payload, _ = await _request_json_async(
            client,
            'fixture/status',
            params={'sportXEventIds': ','.join(chunk)},
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        if not isinstance(payload, dict):
            continue
        data = payload.get('data')
        if isinstance(data, dict):
            merged.update(dict(data))
    return merged


async def _load_live_scores_map_async(
    client: httpx.AsyncClient,
    event_ids: Sequence[str],
    retries: int,
    backoff_seconds: float,
) -> Dict[str, dict]:
    normalized_ids = list(
        dict.fromkeys(
            item for item in (_normalize_text(event_id) for event_id in event_ids) if item
        )
    )
    if not normalized_ids:
        return {}
    mapped: Dict[str, dict] = {}
    for chunk in _chunked(normalized_ids, SX_BET_FIXTURE_STATE_BATCH_SIZE):
        payload, _ = await _request_json_async(
            client,
            'live-scores',
            params={'sportXEventIds': ','.join(chunk)},
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        if not isinstance(payload, dict):
            continue
        data = payload.get('data')
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            event_id = _normalize_text(row.get('sportXEventId'))
            if not event_id:
                continue
            mapped[event_id] = dict(row)
    return mapped


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    token = _normalize_text(value).lower()
    if token in {"true", "1", "yes", "y"}:
        return True
    if token in {"false", "0", "no", "n"}:
        return False
    return None


def _token_amount_from_raw(value: object, decimals: int) -> Optional[float]:
    text = _normalize_text(value)
    if not text:
        return None
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            integer_value = int(text)
        except ValueError:
            return None
        scale = 10 ** max(0, int(decimals))
        if scale <= 1:
            return float(integer_value)
        return float(integer_value) / float(scale)
    return _safe_float(text)


def _normalize_token(value: object) -> str:
    text = _normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_market_key(value: object) -> str:
    token = _normalize_text(value).lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


def _cache_lookup(cache_key: str, ttl_seconds: int) -> Optional[List[dict]]:
    now = time.time()
    cache_valid = ttl_seconds > 0 and now < float(UPCOMING_CACHE.get("expires_at", 0.0))
    cache_entries = UPCOMING_CACHE.get("entries")
    if not cache_valid or not isinstance(cache_entries, dict):
        return None
    cached = cache_entries.get(cache_key)
    if isinstance(cached, list):
        return cached
    return None


def _cache_store(cache_key: str, rows: List[dict], ttl_seconds: int) -> None:
    entries = UPCOMING_CACHE.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    entries[cache_key] = rows
    UPCOMING_CACHE["entries"] = entries
    now = time.time()
    UPCOMING_CACHE["expires_at"] = now + ttl_seconds if ttl_seconds > 0 else now


def _market_period_suffix(*values: object) -> str:
    token = "_".join(_normalize_market_key(item) for item in values if _normalize_market_key(item))
    if not token:
        return ""
    if any(pattern in token for pattern in ("1st_5", "first_5", "five_innings", "first_five_innings")):
        return "h1"
    if any(pattern in token for pattern in ("second_half", "2nd_half", "half_2", "h2")):
        return "h2"
    if any(pattern in token for pattern in ("first_half", "1st_half", "half_time", "halftime", "half_1", "h1")):
        return "h1"
    quarter_patterns = {
        "q1": ("q1", "quarter_1", "1st_quarter", "first_quarter"),
        "q2": ("q2", "quarter_2", "2nd_quarter", "second_quarter"),
        "q3": ("q3", "quarter_3", "3rd_quarter", "third_quarter"),
        "q4": ("q4", "quarter_4", "4th_quarter", "fourth_quarter"),
    }
    for suffix, patterns in quarter_patterns.items():
        if any(pattern in token for pattern in patterns):
            return suffix
    return ""


def _scoped_market_alias(base_key: str, *period_hints: object) -> str:
    suffix = _market_period_suffix(*period_hints)
    return f"{base_key}_{suffix}" if suffix else base_key


def _base_market_key(value: object) -> str:
    key = _normalize_market_key(value)
    for prefix in ("h2h_", "spreads_", "totals_"):
        if key.startswith(prefix):
            return prefix[:-1]
    return key


def _requested_market_keys(markets: Sequence[str]) -> set[str]:
    requested = {_normalize_market_key(item) for item in (markets or []) if _normalize_market_key(item)}
    if "both_teams_to_score" in requested:
        requested.add("btts")
    if "btts" in requested:
        requested.add("both_teams_to_score")
    return requested


def _market_signature(market: dict) -> str:
    key = _normalize_text(market.get("key"))
    outcomes = market.get("outcomes") if isinstance(market.get("outcomes"), list) else []
    parts: List[str] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        name = _normalize_token(outcome.get("name"))
        point = _safe_float(outcome.get("point"))
        if point is None:
            parts.append(name)
        else:
            parts.append(f"{name}:{point:.6f}")
    return f"{key}:{'|'.join(sorted(parts))}"


def _market_score(market: dict) -> float:
    outcomes = market.get("outcomes") if isinstance(market.get("outcomes"), list) else []
    prices = [float(item.get("price")) for item in outcomes if _safe_float(item.get("price"))]
    if len(prices) < 2:
        return 0.0
    return min(prices)


def _parse_number(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return _safe_float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", _normalize_text(value))
    if not match:
        return None
    return _safe_float(match.group(0))


def _parse_number_pair(value: object) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _safe_float(value[0]), _safe_float(value[1])
    parts = re.findall(r"[-+]?\d+(?:\.\d+)?", _normalize_text(value))
    if len(parts) < 2:
        return None, None
    return _safe_float(parts[0]), _safe_float(parts[1])


def _spread_name_point(value: object) -> Tuple[str, Optional[float]]:
    label = _normalize_text(value)
    if not label:
        return "", None
    match = re.match(r"^(.*?)([-+]\d+(?:\.\d+)?)$", label)
    if match:
        return _normalize_text(match.group(1)), _safe_float(match.group(2))
    return label, _parse_number(label)


def _total_side_point(value: object) -> Tuple[Optional[str], Optional[float]]:
    label = _normalize_text(value)
    if not label:
        return None, None
    lower = label.lower()
    side: Optional[str] = None
    if lower.startswith("over"):
        side = "Over"
    elif lower.startswith("under"):
        side = "Under"
    return side, _parse_number(label)


def _market_type_base_hint(market: dict) -> Optional[str]:
    type_id = _as_int(market.get("type"))
    if type_id is None:
        type_id = _as_int(market.get("marketTypeId"))
    if type_id is None:
        return None
    return SX_MARKET_TYPE_ID_BASE_KEY.get(type_id)


def _infer_market_base_key(market: dict) -> Optional[str]:
    outcome_one = _normalize_text(market.get("outcomeOneName")).lower()
    outcome_two = _normalize_text(market.get("outcomeTwoName")).lower()
    home_team = _normalize_text(market.get("teamOneName")).lower()
    away_team = _normalize_text(market.get("teamTwoName")).lower()
    market_name = _normalize_text(market.get("marketName") or market.get("name")).lower()
    tokens = " ".join(
        part
        for part in (outcome_one, outcome_two, market_name)
        if part
    )
    if not tokens:
        return None

    if 'tie' in {outcome_one, outcome_two} and 'not tie' in {outcome_one, outcome_two}:
        return 'h2h_3_way'

    if "both teams to score" in tokens or re.search(r"\byes\b", tokens) and re.search(r"\bno\b", tokens):
        return "btts"

    if (
        outcome_one.startswith("over")
        and outcome_two.startswith("under")
    ) or (
        outcome_one.startswith("under")
        and outcome_two.startswith("over")
    ):
        return "totals"

    line = _safe_float(market.get("line"))
    if line is not None:
        plus_minus_pattern = re.compile(r"[-+]\d+(?:\.\d+)?")
        if plus_minus_pattern.search(outcome_one) and plus_minus_pattern.search(outcome_two):
            return "spreads"

    home_match = bool(home_team and home_team == outcome_one)
    away_match = bool(away_team and away_team == outcome_two)
    home_match_swapped = bool(home_team and home_team == outcome_two)
    away_match_swapped = bool(away_team and away_team == outcome_one)
    if (home_match and away_match) or (home_match_swapped and away_match_swapped):
        return "h2h"
    return None


def _market_type_aliases(market: dict) -> List[str]:
    type_hint = _market_type_base_hint(market)
    raw_type = _normalize_text(market.get("marketType") or market.get("type")).upper()
    raw_name = _normalize_text(market.get("marketName") or market.get("name"))
    outcome_one_name = _normalize_text(market.get('outcomeOneName'))
    outcome_two_name = _normalize_text(market.get('outcomeTwoName'))
    aliases: List[str] = []
    period_hints = (raw_type, raw_name, type_hint, outcome_one_name, outcome_two_name)

    for raw in (raw_type, raw_name, type_hint):
        key = _normalize_market_key(raw)
        if key:
            aliases.append(key)

    type_lc = raw_type.lower()
    name_lc = raw_name.lower()

    if any(token in type_lc for token in ("money_line", "moneyline", "match_odds", "h2h")) or any(
        token in name_lc for token in ("moneyline", "match winner", "winner", "to win")
    ):
        aliases.append(_scoped_market_alias("h2h", *period_hints))
    if any(token in type_lc for token in ("spread", "handicap")) or any(
        token in name_lc for token in ("spread", "handicap")
    ):
        aliases.append(_scoped_market_alias("spreads", *period_hints))
    if any(token in type_lc for token in ("total", "over_under", "overunder")) or any(
        token in name_lc for token in ("total", "over/under", "over under")
    ):
        aliases.append(_scoped_market_alias("totals", *period_hints))
    if "btts" in type_lc or "both teams to score" in name_lc:
        aliases.extend(["btts", "both_teams_to_score"])

    if type_hint in {"h2h", "spreads", "totals"}:
        aliases.append(_scoped_market_alias(type_hint, *period_hints))
    elif type_hint == "btts":
        aliases.extend(["btts", "both_teams_to_score"])

    inferred = _infer_market_base_key(market)
    if inferred in {"h2h", "spreads", "totals"}:
        aliases.append(_scoped_market_alias(inferred, *period_hints))
    elif inferred == "btts":
        aliases.extend(["btts", "both_teams_to_score"])

    if inferred == 'h2h_3_way':
        aliases.append('h2h_3_way')

    out: List[str] = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            out.append(alias)
            seen.add(alias)
    return out


def _market_value_pair(market: dict) -> Tuple[Optional[float], Optional[float]]:
    for key in ("marketValue", "value", "marketLine", "line", "handicap", "spread"):
        if key not in market:
            continue
        left, right = _parse_number_pair(market.get(key))
        if left is not None and right is not None:
            return left, right
    return None, None


def _market_value_single(market: dict) -> Optional[float]:
    for key in ("marketValue", "value", "marketLine", "line", "handicap", "spread", "total", "point"):
        if key not in market:
            continue
        parsed = _parse_number(market.get(key))
        if parsed is not None:
            return parsed
    return None


def _normalize_fixture_market(
    market: dict,
    requested_markets: set[str],
    home_team: str,
    away_team: str,
) -> Optional[dict]:
    if not isinstance(market, dict):
        return None

    target_market_key = None
    for alias in _market_type_aliases(market):
        if alias in requested_markets:
            target_market_key = alias
            break
    if not target_market_key:
        return None
    if target_market_key == "btts":
        target_market_key = (
            "both_teams_to_score" if "both_teams_to_score" in requested_markets else "btts"
        )
    target_market_base = _base_market_key(target_market_key)

    summary_entry = _summary_display_odds_entry(market)
    odds_one = summary_entry.get("odds_one")
    odds_two = summary_entry.get("odds_two")
    market_hash = _normalize_text(market.get("marketHash"))
    outcome_one_label = _normalize_text(market.get("outcomeOneName")) or home_team
    outcome_two_label = _normalize_text(market.get("outcomeTwoName")) or away_team
    home_token = _normalize_token(home_team)
    away_token = _normalize_token(away_team)
    label_one_token = _normalize_token(outcome_one_label)
    label_two_token = _normalize_token(outcome_two_label)

    description = _normalize_text(market.get("marketName") or market.get("name"))
    if not description:
        description = _normalize_text(market.get("marketType") or market.get("type"))

    candidate = {
        "market_key": target_market_key,
        "market_hash": market_hash,
        "odds_one": odds_one,
        "odds_two": odds_two,
        "outcome_one_raw_percentage_odds": summary_entry.get("raw_percentage_one"),
        "outcome_two_raw_percentage_odds": summary_entry.get("raw_percentage_two"),
        "outcome_one_quote_source": None,
        "outcome_two_quote_source": None,
        "description": description,
        "outcome_one_name": outcome_one_label,
        "outcome_two_name": outcome_two_label,
        "outcome_one_point": None,
        "outcome_two_point": None,
    }

    if target_market_base == "h2h":
        if target_market_key == 'h2h_3_way':
            if label_one_token == 'tie' and label_two_token == 'not tie':
                return {
                    **candidate,
                    'outcome_one_name': 'Draw',
                    'outcome_two_name': 'Not Draw',
                }
            if label_two_token == 'tie' and label_one_token == 'not tie':
                return {
                    **candidate,
                    'outcome_one_name': 'Not Draw',
                    'outcome_two_name': 'Draw',
                }
            return None
        if "draw" in {label_one_token, label_two_token}:
            return None
        if label_one_token == away_token and label_two_token == home_token:
            candidate["outcome_one_name"] = away_team
            candidate["outcome_two_name"] = home_team
        elif label_one_token == home_token and label_two_token == away_token:
            candidate["outcome_one_name"] = home_team
            candidate["outcome_two_name"] = away_team
        else:
            candidate["outcome_one_name"] = home_team
            candidate["outcome_two_name"] = away_team
        return candidate

    if target_market_base == "spreads":
        name_one, point_one = _spread_name_point(outcome_one_label)
        name_two, point_two = _spread_name_point(outcome_two_label)
        value_one, value_two = _market_value_pair(market)
        shared = _market_value_single(market)
        if point_one is None and value_one is not None:
            point_one = value_one
        if point_two is None and value_two is not None:
            point_two = value_two
        if point_one is None and point_two is None and shared is not None:
            point_one = shared
            point_two = -shared
        if point_one is None or point_two is None:
            return None
        if _normalize_token(name_one) == away_token:
            candidate["outcome_one_name"] = away_team
        elif _normalize_token(name_one) == home_token:
            candidate["outcome_one_name"] = home_team
        else:
            candidate["outcome_one_name"] = home_team
        if _normalize_token(name_two) == home_token:
            candidate["outcome_two_name"] = home_team
        elif _normalize_token(name_two) == away_token:
            candidate["outcome_two_name"] = away_team
        else:
            candidate["outcome_two_name"] = away_team
        candidate["outcome_one_point"] = round(float(point_one), 6)
        candidate["outcome_two_point"] = round(float(point_two), 6)
        return candidate

    if target_market_base == "totals":
        side_one, point_one = _total_side_point(outcome_one_label)
        side_two, point_two = _total_side_point(outcome_two_label)
        if side_one is None and side_two is None:
            side_one, side_two = "Over", "Under"
        elif side_one is None:
            side_one = "Under" if side_two == "Over" else "Over"
        elif side_two is None:
            side_two = "Under" if side_one == "Over" else "Over"
        if side_one == side_two:
            return None
        shared = point_one if point_one is not None else point_two
        if shared is None:
            shared = _market_value_single(market)
        if shared is None:
            return None
        total_point = round(float(shared), 6)
        candidate["outcome_one_name"] = side_one
        candidate["outcome_two_name"] = side_two
        candidate["outcome_one_point"] = total_point
        candidate["outcome_two_point"] = total_point
        return candidate

    value_one, value_two = _market_value_pair(market)
    shared = _market_value_single(market)
    if value_one is not None:
        candidate["outcome_one_point"] = round(float(value_one), 6)
    elif shared is not None:
        candidate["outcome_one_point"] = round(float(shared), 6)
    if value_two is not None:
        candidate["outcome_two_point"] = round(float(value_two), 6)
    elif shared is not None:
        candidate["outcome_two_point"] = round(float(shared), 6)
    return candidate


def _parse_manual_league_map() -> Dict[str, Set[str]]:
    if not SX_BET_LEAGUE_MAP_RAW:
        return {}
    try:
        payload = json.loads(SX_BET_LEAGUE_MAP_RAW)
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    mapping: Dict[str, Set[str]] = {}
    for key, value in payload.items():
        sport_key = _normalize_text(key)
        if not sport_key:
            continue
        league_ids: Set[str] = set()
        if isinstance(value, list):
            for item in value:
                item_text = _normalize_text(item)
                if item_text:
                    league_ids.add(item_text)
        else:
            item_text = _normalize_text(value)
            if item_text:
                league_ids.add(item_text)
        if league_ids:
            mapping[sport_key] = league_ids
    return mapping


def _fixture_matches_sport(
    sport_key: str,
    fixture: dict,
    manual_league_map: Dict[str, Set[str]],
) -> bool:
    manual_ids = manual_league_map.get(sport_key)
    league_id = _normalize_text(fixture.get("leagueId"))
    league_label = _normalize_text(fixture.get("leagueLabel")).lower()
    if manual_ids is not None:
        return bool(league_id and league_id in manual_ids)
    hints = SPORT_LEAGUE_HINTS.get(sport_key)
    if not hints:
        return True
    return any(hint in league_label for hint in hints)


def _moneyline_decimal_from_summary(value) -> Optional[float]:
    """
    SX summary fields can be:
    - decimal odds (legacy payloads)
    - probability 0..1
    - scaled implied odds from orders endpoints (1e20 scale)
    """
    raw = _safe_float(value)
    if raw is None:
        return None
    if raw <= 0:
        return None

    # Current SX payloads typically use the implied-odds format (1e20 scale).
    # Interpret those with the same parser used for orders/odds endpoints.
    if raw > 10000:
        return _moneyline_decimal_from_percentage(raw)

    # Basis-point style probabilities occasionally appear in older payloads.
    if raw > 100:
        return _moneyline_decimal_from_percentage(raw)

    # Preserve legacy decimal/probability support.
    if raw > 1:
        return raw
    converted = 1.0 / raw
    return converted if converted > 1 else None


def _moneyline_decimal_from_percentage(value) -> Optional[float]:
    """
    SX odds endpoints can return probabilities in multiple scales:
    - 0..1 => direct probability
    - 1..100 => percentage probability
    - >10000 => 1e20-scaled probability (official orders/odds docs format)
    """
    raw = _safe_float(value)
    if raw is None or raw <= 0:
        return None
    if raw > 10000:
        probability = raw / 1e20
    elif raw > 1:
        if raw <= 100:
            probability = raw / 100.0
        elif raw <= 10000:
            probability = raw / 10000.0
        else:
            return None
    else:
        probability = raw
    if probability <= 0 or probability >= 1:
        return None
    odds = 1.0 / probability
    return odds if odds > 1 else None


def _moneyline_decimal_from_american(value) -> Optional[float]:
    raw = _safe_float(value)
    if raw is None or raw == 0:
        return None
    if raw > 0:
        odds = 1.0 + (raw / 100.0)
    else:
        odds = 1.0 + (100.0 / abs(raw))
    return odds if odds > 1 else None


def _moneyline_decimal_from_outcome_payload(payload) -> Optional[float]:
    if not isinstance(payload, dict):
        return _moneyline_decimal_from_percentage(payload)

    for key in ("percentageOdds", "probability", "prob", "impliedProbability"):
        odds = _moneyline_decimal_from_percentage(payload.get(key))
        if odds is not None:
            return odds

    for key in ("decimalOdds", "decimal", "price", "odds"):
        odds = _safe_float(payload.get(key))
        if odds is not None and odds > 1:
            return odds

    return _moneyline_decimal_from_american(payload.get("americanOdds"))


def _taker_decimal_from_maker_payload(payload) -> Optional[float]:
    if isinstance(payload, dict):
        for key in ("percentageOdds", "probability", "prob", "impliedProbability"):
            odds = _taker_decimal_from_maker_percentage(payload.get(key))
            if odds is not None:
                return odds
    return _taker_decimal_from_maker_percentage(payload)


def _outcome_updated_at(payload) -> Optional[object]:
    if not isinstance(payload, dict):
        return None
    return payload.get("updatedAt") or payload.get("updated_at")


def _outcome_percentage_raw(payload) -> Optional[object]:
    if not isinstance(payload, dict):
        return None
    value = payload.get("percentageOdds")
    return value if value not in (None, "") else None


def _summary_best_odds_use_maker_view(value) -> bool:
    raw = _safe_float(value)
    if raw is None or raw <= 0:
        return False
    return raw < 1 or raw > 100


def _summary_display_odds_entry(market: dict) -> dict:
    raw_one = market.get("bestOddsOutcomeOne")
    raw_two = market.get("bestOddsOutcomeTwo")

    # SX summary payloads mirror the maker-side raw odds used by the orders APIs.
    # Platform prices are the taker view, which is the opposite outcome after conversion.
    if _summary_best_odds_use_maker_view(raw_one) or _summary_best_odds_use_maker_view(raw_two):
        return {
            "odds_one": _taker_decimal_from_maker_percentage(raw_two),
            "odds_two": _taker_decimal_from_maker_percentage(raw_one),
            "raw_percentage_one": raw_two if raw_two not in (None, "") else None,
            "raw_percentage_two": raw_one if raw_one not in (None, "") else None,
        }

    return {
        "odds_one": _moneyline_decimal_from_summary(raw_one),
        "odds_two": _moneyline_decimal_from_summary(raw_two),
        "raw_percentage_one": None,
        "raw_percentage_two": None,
    }


def _best_odds_entry_from_payloads(outcome_one_payload, outcome_two_payload) -> dict:
    taker_odds_one = _taker_decimal_from_maker_payload(outcome_two_payload)
    taker_odds_two = _taker_decimal_from_maker_payload(outcome_one_payload)

    if taker_odds_one is not None or taker_odds_two is not None:
        return {
            "odds_one": taker_odds_one,
            "odds_two": taker_odds_two,
            "updated_at_one": _outcome_updated_at(outcome_two_payload),
            "updated_at_two": _outcome_updated_at(outcome_one_payload),
            "raw_percentage_one": _outcome_percentage_raw(outcome_two_payload),
            "raw_percentage_two": _outcome_percentage_raw(outcome_one_payload),
            "source_one": "rest_snapshot" if taker_odds_one is not None else None,
            "source_two": "rest_snapshot" if taker_odds_two is not None else None,
        }

    odds_one = _moneyline_decimal_from_outcome_payload(outcome_one_payload)
    odds_two = _moneyline_decimal_from_outcome_payload(outcome_two_payload)
    return {
        "odds_one": odds_one,
        "odds_two": odds_two,
        "updated_at_one": _outcome_updated_at(outcome_one_payload),
        "updated_at_two": _outcome_updated_at(outcome_two_payload),
        "raw_percentage_one": None,
        "raw_percentage_two": None,
        "source_one": "rest_snapshot" if odds_one is not None else None,
        "source_two": "rest_snapshot" if odds_two is not None else None,
    }


def _probability_from_percentage(value) -> Optional[float]:
    raw = _safe_float(value)
    if raw is None or raw <= 0:
        return None
    if raw > 10000:
        probability = raw / 1e20
    elif raw > 1:
        if raw <= 100:
            probability = raw / 100.0
        elif raw <= 10000:
            probability = raw / 10000.0
        else:
            return None
    else:
        probability = raw
    if probability <= 0 or probability >= 1:
        return None
    return probability


def _taker_decimal_from_maker_percentage(value) -> Optional[float]:
    maker_probability = _probability_from_percentage(value)
    if maker_probability is None:
        return None
    taker_probability = 1.0 - maker_probability
    if taker_probability <= 0 or taker_probability >= 1:
        return None
    odds = 1.0 / taker_probability
    return odds if odds > 1 else None


def _slugify_public_segment(value: object) -> str:
    token = _normalize_token(value)
    if not token:
        return ""
    return token.replace(" ", "-")


def _public_sport_slug(event: dict) -> str:
    sport_key = _normalize_text(event.get("sport_key")).lower()
    for prefix, slug in SX_PUBLIC_SPORT_SLUGS_BY_PREFIX.items():
        if sport_key.startswith(prefix):
            return slug
    sport_id = _as_int(event.get("sport_id") or event.get("sportId"))
    if sport_id is not None and sport_id in SX_PUBLIC_SPORT_SLUGS_BY_ID:
        return SX_PUBLIC_SPORT_SLUGS_BY_ID[sport_id]
    sport_label = _normalize_text(event.get("sport_label") or event.get("sportLabel"))
    return _slugify_public_segment(sport_label)


def _public_league_slug(event: dict) -> str:
    league_label = _normalize_text(event.get("league_label") or event.get("leagueLabel"))
    if not league_label:
        return ""
    override = SX_PUBLIC_LEAGUE_SLUG_OVERRIDES.get(league_label.strip().lower())
    if override:
        return override
    return _slugify_public_segment(league_label)


def _event_url(event: object) -> str:
    if isinstance(event, dict):
        event_id = _normalize_text(event.get("event_id") or event.get("eventId") or event.get("id"))
        sport_slug = _public_sport_slug(event)
        league_slug = _public_league_slug(event)
        if event_id and sport_slug and league_slug:
            return f"{_public_base()}/{sport_slug}/{league_slug}/game-lines/{event_id}"
        if event_id:
            return f"{_public_base()}/event/{event_id}"
        return ""
    raw = _normalize_text(event)
    if not raw:
        return ""
    return f"{_public_base()}/event/{raw}"


def _chunked(values: Sequence[str], size: int) -> List[List[str]]:
    out: List[List[str]] = []
    if size <= 0:
        size = 1
    for index in range(0, len(values), size):
        out.append(list(values[index : index + size]))
    return out


def _extract_orders_payload(payload: object) -> List[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        nested_orders = data.get("orders")
        if isinstance(nested_orders, list):
            return [item for item in nested_orders if isinstance(item, dict)]
    root_orders = payload.get("orders")
    if isinstance(root_orders, list):
        return [item for item in root_orders if isinstance(item, dict)]
    return []


class SXBetRealtimeManager:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_requested = threading.Event()
        self._ready_event = threading.Event()
        self._state_lock = threading.RLock()
        self._best_odds_by_market: Dict[str, dict] = {}
        self._started = False
        self._connected = False
        self._messages_received = 0
        self._last_message_at = 0.0
        self._last_error = ""
        self._last_channel = ""

    def ensure_started(self) -> bool:
        if not _sx_best_odds_ws_enabled():
            return False
        with self._state_lock:
            if self._started and self._thread and self._thread.is_alive():
                return True
            self._stop_requested.clear()
            self._ready_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop_thread,
                name="sx-bet-ws",
                daemon=True,
            )
            self._thread.start()
            self._started = True
        return True

    def wait_until_ready(self, timeout_seconds: float) -> bool:
        if timeout_seconds <= 0:
            return False
        self._ready_event.wait(timeout=max(0.0, float(timeout_seconds)))
        with self._state_lock:
            return self._connected

    def stop(self, timeout_seconds: float = 2.0) -> None:
        with self._state_lock:
            thread = self._thread
            loop = self._loop
        self._stop_requested.set()
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass
        if thread and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_seconds)))
        with self._state_lock:
            self._thread = None
            self._loop = None
            self._started = False
            self._connected = False
            self._ready_event.clear()

    def snapshot(self) -> dict:
        with self._state_lock:
            age_seconds = None
            if self._last_message_at > 0:
                age_seconds = round(max(0.0, time.time() - self._last_message_at), 2)
            return {
                "started": self._started,
                "connected": self._connected,
                "messages_received": int(self._messages_received or 0),
                "best_odds_cached": len(self._best_odds_by_market),
                "last_message_age_seconds": age_seconds,
                "last_error": _normalize_text(self._last_error),
                "channel": _normalize_text(self._last_channel),
            }

    def get_best_odds_map(
        self,
        market_hashes: Sequence[str],
        max_age_seconds: float,
    ) -> Dict[str, dict]:
        now = time.time()
        results: Dict[str, dict] = {}
        with self._state_lock:
            for market_hash in (_normalize_text(item) for item in market_hashes or []):
                if not market_hash:
                    continue
                entry = self._best_odds_by_market.get(market_hash)
                if not isinstance(entry, dict):
                    continue
                observed_at = self._updated_at_seconds(
                    entry.get("observed_at"),
                    default_seconds=self._updated_at_seconds(entry.get("updated_at"), default_seconds=0.0),
                )
                if max_age_seconds > 0 and observed_at > 0 and (now - observed_at) > max_age_seconds:
                    continue
                results[market_hash] = dict(entry)
        return results

    def merge_best_odds_map(self, odds_map: Dict[str, dict], source: str = "snapshot") -> int:
        merged = 0
        default_seconds = time.time()
        with self._state_lock:
            for market_hash_raw, payload in (odds_map or {}).items():
                market_hash = _normalize_text(market_hash_raw)
                if not market_hash or not isinstance(payload, dict):
                    continue
                entry = self._best_odds_by_market.setdefault(
                    market_hash,
                    {
                        "odds_one": None,
                        "odds_two": None,
                        "updated_at_one": None,
                        "updated_at_two": None,
                        "observed_at_one": None,
                        "observed_at_two": None,
                        "observed_at": 0.0,
                        "updated_at": 0.0,
                    },
                )
                changed = False
                changed |= self._merge_best_odds_side(
                    entry,
                    side_key="one",
                    odds_value=payload.get("odds_one"),
                    updated_at_raw=payload.get("updated_at_one") or payload.get("updated_at"),
                    source=source,
                    default_seconds=default_seconds,
                )
                changed |= self._merge_best_odds_side(
                    entry,
                    side_key="two",
                    odds_value=payload.get("odds_two"),
                    updated_at_raw=payload.get("updated_at_two") or payload.get("updated_at"),
                    source=source,
                    default_seconds=default_seconds,
                )
                if changed:
                    merged += 1
        return merged

    @staticmethod
    def _updated_at_seconds(value: object, default_seconds: float = 0.0) -> float:
        seconds = _safe_float(value)
        if seconds is None:
            return float(default_seconds)
        if seconds > 1e12:
            seconds /= 1000.0
        return float(seconds)

    def _merge_best_odds_side(
        self,
        entry: dict,
        side_key: str,
        odds_value: object,
        updated_at_raw: object,
        source: str,
        default_seconds: float,
    ) -> bool:
        odds_decimal = _safe_float(odds_value)
        if odds_decimal is None or odds_decimal <= 1:
            return False
        updated_at_seconds = self._updated_at_seconds(updated_at_raw, default_seconds=default_seconds)
        current_updated_at_raw = entry.get(f"updated_at_{side_key}")
        current_updated_at_seconds = self._updated_at_seconds(current_updated_at_raw, default_seconds=0.0)
        has_existing_odds = _safe_float(entry.get(f"odds_{side_key}")) is not None
        if has_existing_odds:
            if current_updated_at_seconds > 0 and updated_at_seconds > 0 and updated_at_seconds < current_updated_at_seconds:
                return False
            if updated_at_raw in (None, "") and current_updated_at_seconds > 0:
                return False
        entry[f"odds_{side_key}"] = round(float(odds_decimal), 6)
        entry[f"updated_at_{side_key}"] = updated_at_raw if updated_at_raw not in (None, "") else updated_at_seconds
        entry[f"source_{side_key}"] = _normalize_text(source) or "snapshot"
        entry[f"observed_at_{side_key}"] = float(default_seconds)
        entry["updated_at"] = max(
            self._updated_at_seconds(entry.get("updated_at"), default_seconds=0.0),
            updated_at_seconds,
        )
        entry["observed_at"] = max(
            self._updated_at_seconds(entry.get("observed_at"), default_seconds=0.0),
            float(default_seconds),
        )
        return True

    def _run_loop_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._state_lock:
            self._loop = loop
        try:
            loop.run_until_complete(self._run_forever_async())
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            asyncio.set_event_loop(None)
            with self._state_lock:
                self._loop = None
                self._connected = False
                self._started = False

    async def _run_forever_async(self) -> None:
        while not self._stop_requested.is_set():
            try:
                await self._run_session_async()
            except Exception as exc:
                with self._state_lock:
                    self._connected = False
                    self._last_error = str(exc)
                if self._stop_requested.is_set():
                    break
                await asyncio.sleep(1.0)

    async def _run_session_async(self) -> None:
        client = None
        channel = None
        try:
            client = self._create_ably_client()
            channel_name = f"best_odds:{_normalize_text(SX_BET_BASE_TOKEN)}"
            channel = client.channels.get(channel_name)
            await channel.subscribe(self._handle_best_odds_message)
            with self._state_lock:
                self._last_channel = channel_name
                self._last_error = ""
            while not self._stop_requested.is_set():
                state = getattr(client.connection, "state", None)
                state_token = _normalize_text(getattr(state, "value", state)).lower()
                error_reason = getattr(client.connection, "error_reason", None)
                with self._state_lock:
                    self._connected = state_token == "connected"
                    if self._connected:
                        self._last_error = ""
                    elif error_reason:
                        self._last_error = str(error_reason)
                if state_token == "connected":
                    self._ready_event.set()
                elif state_token in {"closed", "failed"}:
                    break
                await asyncio.sleep(0.25)
        finally:
            with self._state_lock:
                self._connected = False
            self._ready_event.clear()
            try:
                if channel is not None:
                    channel.unsubscribe()
            except Exception:
                pass
            try:
                if client is not None:
                    await client.close()
            except Exception:
                pass

    def _create_ably_client(self):
        try:
            from ably.realtime.realtime import AblyRealtime
        except Exception as exc:
            raise ProviderError(f"SX Bet realtime requires the 'ably' package: {exc}") from exc
        auth_headers = {"X-Api-Key": SX_BET_API_KEY}
        if SX_BET_USER_AGENT:
            auth_headers["User-Agent"] = SX_BET_USER_AGENT
        return AblyRealtime(
            loop=asyncio.get_running_loop(),
            use_token_auth=True,
            auth_url=_sx_token_url(),
            auth_headers=auth_headers,
            auto_connect=True,
        )

    def _handle_best_odds_message(self, message) -> None:
        rows = self._decode_realtime_rows(getattr(message, "data", None))
        if not rows:
            return
        now = time.time()
        with self._state_lock:
            self._messages_received += len(rows)
            self._last_message_at = now
            for row in rows:
                market_hash = _normalize_text(row.get("marketHash"))
                if not market_hash:
                    continue
                maker_outcome_one = _bool_or_none(row.get("isMakerBettingOutcomeOne"))
                taker_odds = _taker_decimal_from_maker_percentage(row.get("percentageOdds"))
                if maker_outcome_one is None or taker_odds is None:
                    continue
                entry = self._best_odds_by_market.setdefault(
                    market_hash,
                    {
                        "odds_one": None,
                        "odds_two": None,
                        "updated_at_one": None,
                        "updated_at_two": None,
                        "observed_at_one": None,
                        "observed_at_two": None,
                        "observed_at": 0.0,
                        "updated_at": 0.0,
                    },
                )
                updated_at_raw = row.get("updatedAt") or row.get("updated_at") or row.get("timestamp") or row.get("ts")
                if maker_outcome_one:
                    self._merge_best_odds_side(
                        entry,
                        side_key="two",
                        odds_value=taker_odds,
                        updated_at_raw=updated_at_raw,
                        source="ws",
                        default_seconds=now,
                    )
                else:
                    self._merge_best_odds_side(
                        entry,
                        side_key="one",
                        odds_value=taker_odds,
                        updated_at_raw=updated_at_raw,
                        source="ws",
                        default_seconds=now,
                    )

    def _decode_realtime_rows(self, payload: object) -> List[dict]:
        data = payload
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError:
                return []
        if isinstance(data, str):
            text = data.strip()
            if not text:
                return []
            try:
                data = json.loads(text)
            except ValueError:
                return []
        if isinstance(data, dict):
            inner = data.get("data")
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []


def _get_realtime_manager() -> SXBetRealtimeManager:
    global REALTIME_MANAGER
    with REALTIME_MANAGER_LOCK:
        if REALTIME_MANAGER is None:
            REALTIME_MANAGER = SXBetRealtimeManager()
        return REALTIME_MANAGER


def ensure_realtime_started(wait_timeout: Optional[float] = None) -> dict:
    if not _sx_best_odds_ws_enabled():
        return {
            "enabled": False,
            "started": False,
            "ready": False,
            "status": {},
        }
    manager = _get_realtime_manager()
    started = manager.ensure_started()
    wait_seconds = _sx_ws_startup_wait_seconds() if wait_timeout is None else max(0.0, float(wait_timeout))
    ready = manager.wait_until_ready(wait_seconds) if wait_seconds > 0 else False
    return {
        "enabled": True,
        "started": started,
        "ready": ready,
        "status": manager.snapshot(),
    }


def stop_realtime(timeout_seconds: float = 2.0) -> None:
    manager = _get_realtime_manager()
    manager.stop(timeout_seconds=timeout_seconds)


def realtime_status() -> dict:
    if not _sx_best_odds_ws_enabled():
        return {
            "enabled": False,
            "started": False,
            "ready": False,
            "status": {},
        }
    manager = _get_realtime_manager()
    status = manager.snapshot()
    ready = bool(status.get("connected")) and (
        status.get("last_message_age_seconds") is None
        or (float(status.get("last_message_age_seconds") or 0.0) <= max(5.0, _sx_ws_quote_max_age_seconds()))
    )
    return {
        "enabled": True,
        "started": bool(status.get("started")),
        "ready": ready,
        "status": status,
    }


def _extract_markets_active_rows(payload: object) -> Tuple[List[dict], str]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], ""
    if not isinstance(payload, dict):
        raise ProviderError("SX Bet markets/active response must be a JSON object")
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)], ""
    if not isinstance(data, dict):
        return [], ""
    rows: List[dict] = []
    for key in ("markets", "items", "rows", "data"):
        candidate = data.get(key)
        if isinstance(candidate, list):
            rows = [item for item in candidate if isinstance(item, dict)]
            break
    next_key = _normalize_text(data.get("nextKey") or data.get("next") or data.get("cursor"))
    return rows, next_key


def _load_active_league_ids(
    sport_id: int,
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[int], int]:
    ttl = _int_or_default(SX_BET_LEAGUES_CACHE_TTL_RAW, 300, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(LEAGUES_CACHE.get("expires_at", 0.0))
    cache_entries = LEAGUES_CACHE.get("entries")
    if cache_valid and isinstance(cache_entries, dict):
        cached = cache_entries.get(str(sport_id))
        if isinstance(cached, list):
            league_ids = [int(item) for item in cached if _as_int(item) is not None]
            if league_ids:
                return league_ids, 0

    payload, retries_used = _request_json(
        "leagues/active",
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    if not isinstance(payload, dict):
        raise ProviderError("SX Bet leagues/active response must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        return [], retries_used
    league_ids: List[int] = []
    seen = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        row_sport_id = _as_int(row.get("sportId"))
        if row_sport_id != sport_id:
            continue
        league_id = _as_int(row.get("leagueId"))
        if league_id is None or league_id in seen:
            continue
        seen.add(league_id)
        league_ids.append(league_id)
    entries = cache_entries if isinstance(cache_entries, dict) else {}
    entries[str(sport_id)] = list(league_ids)
    LEAGUES_CACHE["entries"] = entries
    LEAGUES_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    return league_ids, retries_used


async def _load_active_league_ids_async(
    client: httpx.AsyncClient,
    sport_id: int,
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[int], int]:
    ttl = _int_or_default(SX_BET_LEAGUES_CACHE_TTL_RAW, 300, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(LEAGUES_CACHE.get("expires_at", 0.0))
    cache_entries = LEAGUES_CACHE.get("entries")
    if cache_valid and isinstance(cache_entries, dict):
        cached = cache_entries.get(str(sport_id))
        if isinstance(cached, list):
            league_ids = [int(item) for item in cached if _as_int(item) is not None]
            if league_ids:
                return league_ids, 0

    payload, retries_used = await _request_json_async(
        client,
        "leagues/active",
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    if not isinstance(payload, dict):
        raise ProviderError("SX Bet leagues/active response must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        return [], retries_used
    league_ids: List[int] = []
    seen = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        row_sport_id = _as_int(row.get("sportId"))
        if row_sport_id != sport_id:
            continue
        league_id = _as_int(row.get("leagueId"))
        if league_id is None or league_id in seen:
            continue
        seen.add(league_id)
        league_ids.append(league_id)
    entries = cache_entries if isinstance(cache_entries, dict) else {}
    entries[str(sport_id)] = list(league_ids)
    LEAGUES_CACHE["entries"] = entries
    LEAGUES_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    return league_ids, retries_used


def _load_markets_active_rows_for_league(
    sport_id: int,
    league_id: int,
    base_token: str,
    max_pages: int,
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    seen_market_hashes = set()
    pages_fetched = 0
    retries_used_total = 0
    next_key = ""
    seen_next_keys = set()

    for _ in range(max(1, max_pages)):
        params: Dict[str, object] = {
            "baseToken": base_token,
            "leagueId": league_id,
            "pageSize": 50,
        }
        if next_key:
            params["nextKey"] = next_key
        payload, retries_used = _request_json(
            "markets/active",
            params=params,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        retries_used_total += retries_used
        page_rows, page_next_key = _extract_markets_active_rows(payload)
        pages_fetched += 1
        if not page_rows:
            break

        added = 0
        for row in page_rows:
            if not isinstance(row, dict):
                continue
            row_sport = _as_int(row.get("sportId"))
            if row_sport is not None and row_sport != sport_id:
                continue
            market_hash = _normalize_text(row.get("marketHash"))
            if not market_hash:
                market_hash = (
                    f"{league_id}:"
                    f"{_normalize_text(row.get('sportXeventId'))}:"
                    f"{_normalize_text(row.get('type'))}:"
                    f"{_normalize_text(row.get('line'))}:"
                    f"{_normalize_text(row.get('outcomeOneName'))}:"
                    f"{_normalize_text(row.get('outcomeTwoName'))}"
                )
            if market_hash in seen_market_hashes:
                continue
            seen_market_hashes.add(market_hash)
            rows.append(dict(row))
            added += 1

        if added == 0:
            break
        normalized_next_key = _normalize_text(page_next_key)
        if not normalized_next_key:
            break
        if normalized_next_key == next_key or normalized_next_key in seen_next_keys:
            break
        seen_next_keys.add(normalized_next_key)
        next_key = normalized_next_key

    return rows, {"pages_fetched": pages_fetched, "retries_used": retries_used_total}


async def _load_markets_active_rows_for_league_async(
    client: httpx.AsyncClient,
    sport_id: int,
    league_id: int,
    base_token: str,
    max_pages: int,
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    seen_market_hashes = set()
    pages_fetched = 0
    retries_used_total = 0
    next_key = ""
    seen_next_keys = set()

    for _ in range(max(1, max_pages)):
        params: Dict[str, object] = {
            "baseToken": base_token,
            "leagueId": league_id,
            "pageSize": 50,
        }
        if next_key:
            params["nextKey"] = next_key
        payload, retries_used = await _request_json_async(
            client,
            "markets/active",
            params=params,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        retries_used_total += retries_used
        page_rows, page_next_key = _extract_markets_active_rows(payload)
        pages_fetched += 1
        if not page_rows:
            break

        added = 0
        for row in page_rows:
            if not isinstance(row, dict):
                continue
            row_sport = _as_int(row.get("sportId"))
            if row_sport is not None and row_sport != sport_id:
                continue
            market_hash = _normalize_text(row.get("marketHash"))
            if not market_hash:
                market_hash = (
                    f"{league_id}:"
                    f"{_normalize_text(row.get('sportXeventId'))}:"
                    f"{_normalize_text(row.get('type'))}:"
                    f"{_normalize_text(row.get('line'))}:"
                    f"{_normalize_text(row.get('outcomeOneName'))}:"
                    f"{_normalize_text(row.get('outcomeTwoName'))}"
                )
            if market_hash in seen_market_hashes:
                continue
            seen_market_hashes.add(market_hash)
            rows.append(dict(row))
            added += 1

        if added == 0:
            break
        normalized_next_key = _normalize_text(page_next_key)
        if not normalized_next_key:
            break
        if normalized_next_key == next_key or normalized_next_key in seen_next_keys:
            break
        seen_next_keys.add(normalized_next_key)
        next_key = normalized_next_key

    return rows, {"pages_fetched": pages_fetched, "retries_used": retries_used_total}


def _build_fixtures_from_markets_active(
    rows: Sequence[dict],
    sport_id: int,
    only_main_line: bool,
) -> Tuple[List[dict], dict]:
    events_by_id: Dict[str, dict] = {}
    main_line_scopes: Set[str] = set()
    if only_main_line:
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not bool(row.get("mainLine")):
                continue
            event_id = _normalize_text(row.get("sportXeventId") or row.get("eventId") or row.get("id"))
            if not event_id:
                continue
            type_scope = _normalize_text(row.get("type") or row.get("marketType") or row.get("marketName"))
            main_line_scopes.add(f"{event_id}:{type_scope}")
    meta = {
        "markets_rows_total": 0,
        "markets_rows_main_line_filtered": 0,
        "markets_rows_missing_event": 0,
        "markets_rows_missing_team": 0,
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        meta["markets_rows_total"] += 1
        row_sport = _as_int(row.get("sportId"))
        if row_sport is not None and row_sport != sport_id:
            continue
        team_one = _normalize_text(row.get("teamOneName") or row.get("teamOne"))
        team_two = _normalize_text(row.get("teamTwoName") or row.get("teamTwo"))
        if not (team_one and team_two):
            meta["markets_rows_missing_team"] += 1
            continue
        event_id = _normalize_text(row.get("sportXeventId") or row.get("eventId") or row.get("id"))
        if not event_id:
            meta["markets_rows_missing_event"] += 1
            continue
        if only_main_line and isinstance(row.get("mainLine"), bool) and not bool(row.get("mainLine")):
            type_scope = _normalize_text(row.get("type") or row.get("marketType") or row.get("marketName"))
            scope_key = f"{event_id}:{type_scope}"
            if scope_key in main_line_scopes:
                meta["markets_rows_main_line_filtered"] += 1
                continue
        fixture = events_by_id.setdefault(
            event_id,
            {
                "id": event_id,
                "eventId": event_id,
                "teamOne": team_one,
                "teamTwo": team_two,
                "gameTime": row.get("gameTime"),
                "sportId": row.get("sportId"),
                "sportLabel": row.get("sportLabel"),
                "leagueLabel": row.get("leagueLabel"),
                "leagueId": row.get("leagueId"),
                "live_state": _sx_live_state_payload(row),
                "markets": [],
            },
        )
        fixture["live_state"] = _merge_live_state_payload(
            fixture.get("live_state") if isinstance(fixture.get("live_state"), dict) else None,
            _sx_live_state_payload(row),
        )
        fixture_markets = fixture.get("markets")
        if not isinstance(fixture_markets, list):
            fixture_markets = []
            fixture["markets"] = fixture_markets
        fixture_markets.append(dict(row))
    fixtures = list(events_by_id.values())
    meta["fixtures_built"] = len(fixtures)
    return fixtures, meta


def _load_upcoming_fixtures_summary(
    sport_id: int,
    base_token: str,
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[dict], dict]:
    ttl = _int_or_default(SX_BET_PAGE_TTL_RAW, 8, min_value=0)
    cache_key = f"summary:{base_token}:{sport_id}"
    cached = _cache_lookup(cache_key, ttl)
    if cached is not None:
        return cached, {"pages_fetched": 0, "retries_used": 0, "cache": "hit", "fixture_source": "summary"}

    payload, retries_used = _request_json(
        f"summary/upcoming/{base_token}/{sport_id}",
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    if not isinstance(payload, dict):
        raise ProviderError("SX Bet summary/upcoming response must be a JSON object")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    sports = data.get("sports") if isinstance(data.get("sports"), list) else []
    fixtures: List[dict] = []
    for sport in sports:
        if not isinstance(sport, dict):
            continue
        leagues = sport.get("leagues")
        if not isinstance(leagues, list):
            continue
        for league in leagues:
            if not isinstance(league, dict):
                continue
            league_fixtures = league.get("fixtures")
            if not isinstance(league_fixtures, list):
                continue
            for fixture in league_fixtures:
                if not isinstance(fixture, dict):
                    continue
                fixture_obj = dict(fixture)
                if not fixture_obj.get("sportId"):
                    fixture_obj["sportId"] = sport.get("sportId")
                if not fixture_obj.get("sportLabel"):
                    fixture_obj["sportLabel"] = (
                        sport.get("sportLabel")
                        or sport.get("label")
                        or sport.get("name")
                    )
                if not fixture_obj.get("leagueLabel"):
                    fixture_obj["leagueLabel"] = league.get("leagueLabel")
                if not fixture_obj.get("leagueId"):
                    fixture_obj["leagueId"] = league.get("leagueId")
                fixtures.append(fixture_obj)
    _cache_store(cache_key, fixtures, ttl)
    return fixtures, {"pages_fetched": 1, "retries_used": retries_used, "cache": "miss", "fixture_source": "summary"}


async def _load_upcoming_fixtures_summary_async(
    client: httpx.AsyncClient,
    sport_id: int,
    base_token: str,
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[dict], dict]:
    ttl = _int_or_default(SX_BET_PAGE_TTL_RAW, 8, min_value=0)
    cache_key = f"summary:{base_token}:{sport_id}"
    cached = _cache_lookup(cache_key, ttl)
    if cached is not None:
        return cached, {"pages_fetched": 0, "retries_used": 0, "cache": "hit", "fixture_source": "summary"}

    payload, retries_used = await _request_json_async(
        client,
        f"summary/upcoming/{base_token}/{sport_id}",
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    if not isinstance(payload, dict):
        raise ProviderError("SX Bet summary/upcoming response must be a JSON object")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    sports = data.get("sports") if isinstance(data.get("sports"), list) else []
    fixtures: List[dict] = []
    for sport in sports:
        if not isinstance(sport, dict):
            continue
        leagues = sport.get("leagues")
        if not isinstance(leagues, list):
            continue
        for league in leagues:
            if not isinstance(league, dict):
                continue
            league_fixtures = league.get("fixtures")
            if not isinstance(league_fixtures, list):
                continue
            for fixture in league_fixtures:
                if not isinstance(fixture, dict):
                    continue
                fixture_obj = dict(fixture)
                if not fixture_obj.get("sportId"):
                    fixture_obj["sportId"] = sport.get("sportId")
                if not fixture_obj.get("sportLabel"):
                    fixture_obj["sportLabel"] = (
                        sport.get("sportLabel")
                        or sport.get("label")
                        or sport.get("name")
                    )
                if not fixture_obj.get("leagueLabel"):
                    fixture_obj["leagueLabel"] = league.get("leagueLabel")
                if not fixture_obj.get("leagueId"):
                    fixture_obj["leagueId"] = league.get("leagueId")
                fixtures.append(fixture_obj)
    _cache_store(cache_key, fixtures, ttl)
    return fixtures, {"pages_fetched": 1, "retries_used": retries_used, "cache": "miss", "fixture_source": "summary"}


def _load_upcoming_fixtures_markets_active(
    sport_id: int,
    base_token: str,
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[dict], dict]:
    ttl = _int_or_default(SX_BET_PAGE_TTL_RAW, 8, min_value=0)
    max_pages = _markets_active_max_pages()
    cache_key = (
        f"markets_active:{base_token}:{sport_id}:"
        f"{int(bool(SX_BET_MARKETS_ACTIVE_ONLY_MAIN_LINE))}:{max_pages}"
    )
    cached = _cache_lookup(cache_key, ttl)
    if cached is not None:
        return cached, {"pages_fetched": 0, "retries_used": 0, "cache": "hit", "fixture_source": "markets_active"}

    league_ids, retries_used = _load_active_league_ids(
        sport_id=sport_id,
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    all_rows: List[dict] = []
    pages_fetched = 0
    failed_leagues = 0
    for league_id in league_ids:
        try:
            rows, league_meta = _load_markets_active_rows_for_league(
                sport_id=sport_id,
                league_id=league_id,
                base_token=base_token,
                max_pages=max_pages,
                retries=retries,
                backoff_seconds=backoff_seconds,
            )
        except ProviderError:
            failed_leagues += 1
            continue
        all_rows.extend(rows)
        pages_fetched += int(league_meta.get("pages_fetched", 0) or 0)
        retries_used += int(league_meta.get("retries_used", 0) or 0)
    if league_ids and failed_leagues >= len(league_ids):
        raise ProviderError("SX Bet markets/active failed for all active leagues")

    fixtures, markets_meta = _build_fixtures_from_markets_active(
        rows=all_rows,
        sport_id=sport_id,
        only_main_line=bool(SX_BET_MARKETS_ACTIVE_ONLY_MAIN_LINE),
    )
    _cache_store(cache_key, fixtures, ttl)
    return fixtures, {
        "pages_fetched": pages_fetched,
        "retries_used": retries_used,
        "cache": "miss",
        "fixture_source": "markets_active",
        "leagues_requested": len(league_ids),
        "leagues_failed": failed_leagues,
        **markets_meta,
    }


async def _load_upcoming_fixtures_markets_active_async(
    client: httpx.AsyncClient,
    sport_id: int,
    base_token: str,
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[dict], dict]:
    ttl = _int_or_default(SX_BET_PAGE_TTL_RAW, 8, min_value=0)
    max_pages = _markets_active_max_pages()
    cache_key = (
        f"markets_active:{base_token}:{sport_id}:"
        f"{int(bool(SX_BET_MARKETS_ACTIVE_ONLY_MAIN_LINE))}:{max_pages}"
    )
    cached = _cache_lookup(cache_key, ttl)
    if cached is not None:
        return cached, {"pages_fetched": 0, "retries_used": 0, "cache": "hit", "fixture_source": "markets_active"}

    league_ids, retries_used = await _load_active_league_ids_async(
        client=client,
        sport_id=sport_id,
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    all_rows: List[dict] = []
    pages_fetched = 0
    failed_leagues = 0

    async def _fetch_for_league(league_id: int) -> Tuple[List[dict], dict]:
        return await _load_markets_active_rows_for_league_async(
            client=client,
            sport_id=sport_id,
            league_id=league_id,
            base_token=base_token,
            max_pages=max_pages,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )

    if league_ids:
        workers = min(_markets_active_league_workers(), len(league_ids))
        if workers <= 1:
            for league_id in league_ids:
                try:
                    rows, league_meta = await _fetch_for_league(league_id)
                except ProviderError:
                    failed_leagues += 1
                    continue
                all_rows.extend(rows)
                pages_fetched += int(league_meta.get("pages_fetched", 0) or 0)
                retries_used += int(league_meta.get("retries_used", 0) or 0)
        else:
            semaphore = asyncio.Semaphore(workers)

            async def _limited_fetch(league_id: int) -> Tuple[List[dict], dict]:
                async with semaphore:
                    return await _fetch_for_league(league_id)

            tasks = [
                asyncio.create_task(_limited_fetch(league_id))
                for league_id in league_ids
            ]
            for task in asyncio.as_completed(tasks):
                try:
                    rows, league_meta = await task
                except ProviderError:
                    failed_leagues += 1
                    continue
                all_rows.extend(rows)
                pages_fetched += int(league_meta.get("pages_fetched", 0) or 0)
                retries_used += int(league_meta.get("retries_used", 0) or 0)
    if league_ids and failed_leagues >= len(league_ids):
        raise ProviderError("SX Bet markets/active failed for all active leagues")

    fixtures, markets_meta = _build_fixtures_from_markets_active(
        rows=all_rows,
        sport_id=sport_id,
        only_main_line=bool(SX_BET_MARKETS_ACTIVE_ONLY_MAIN_LINE),
    )
    _cache_store(cache_key, fixtures, ttl)
    return fixtures, {
        "pages_fetched": pages_fetched,
        "retries_used": retries_used,
        "cache": "miss",
        "fixture_source": "markets_active",
        "leagues_requested": len(league_ids),
        "leagues_failed": failed_leagues,
        **markets_meta,
    }


def _load_upcoming_fixtures(
    sport_id: int,
    base_token: str,
    retries: int,
    backoff_seconds: float,
    context: Optional[dict] = None,
) -> Tuple[List[dict], dict]:
    source_mode = _fixture_source_mode(context)
    if source_mode == "summary":
        return _load_upcoming_fixtures_summary(
            sport_id=sport_id,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
    if source_mode == "markets_active":
        return _load_upcoming_fixtures_markets_active(
            sport_id=sport_id,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
    try:
        fixtures, meta = _load_upcoming_fixtures_markets_active(
            sport_id=sport_id,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
    except ProviderError as exc:
        fixtures, meta = _load_upcoming_fixtures_summary(
            sport_id=sport_id,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        meta["fallback_used"] = True
        meta["fallback_reason"] = str(exc)
        return fixtures, meta
    meta["fallback_used"] = False
    return fixtures, meta


async def _load_upcoming_fixtures_async(
    client: httpx.AsyncClient,
    sport_id: int,
    base_token: str,
    retries: int,
    backoff_seconds: float,
    context: Optional[dict] = None,
) -> Tuple[List[dict], dict]:
    source_mode = _fixture_source_mode(context)
    if source_mode == "summary":
        return await _load_upcoming_fixtures_summary_async(
            client=client,
            sport_id=sport_id,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
    if source_mode == "markets_active":
        return await _load_upcoming_fixtures_markets_active_async(
            client=client,
            sport_id=sport_id,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
    try:
        fixtures, meta = await _load_upcoming_fixtures_markets_active_async(
            client=client,
            sport_id=sport_id,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
    except ProviderError as exc:
        fixtures, meta = await _load_upcoming_fixtures_summary_async(
            client=client,
            sport_id=sport_id,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        meta["fallback_used"] = True
        meta["fallback_reason"] = str(exc)
        return fixtures, meta
    meta["fallback_used"] = False
    return fixtures, meta


def _load_best_odds_map(
    market_hashes: Sequence[str],
    base_token: str,
    retries: int,
    backoff_seconds: float,
    cache_ttl_seconds: Optional[int] = None,
) -> Tuple[Dict[str, dict], int, dict]:
    ttl = (
        max(0, int(cache_ttl_seconds))
        if cache_ttl_seconds is not None
        else _int_or_default(SX_BET_ODDS_CACHE_TTL_RAW, 4, min_value=0)
    )
    now = time.time()
    cache_valid = ttl > 0 and now < float(ODDS_CACHE.get("expires_at", 0.0))
    cache_entries = ODDS_CACHE.get("entries") if isinstance(ODDS_CACHE.get("entries"), dict) else {}
    odds_map: Dict[str, dict] = {}
    unresolved: List[str] = []
    for market_hash in market_hashes:
        key = f"{base_token}:{market_hash}"
        if cache_valid and key in cache_entries:
            cached = cache_entries[key]
            if isinstance(cached, dict):
                odds_map[market_hash] = dict(cached)
            elif isinstance(cached, (list, tuple)) and len(cached) >= 2:
                odds_map[market_hash] = {
                    "odds_one": _safe_float(cached[0]),
                    "odds_two": _safe_float(cached[1]),
                    "updated_at_one": None,
                    "updated_at_two": None,
                    "source_one": "rest_cache",
                    "source_two": "rest_cache",
                }
        else:
            unresolved.append(market_hash)

    retries_used_total = 0
    lookup_meta = {
        "best_odds_items": 0,
        "best_odds_with_any_odds": 0,
        "best_odds_with_both_odds": 0,
        "best_odds_null_count": 0,
        "best_odds_missing_market_hash": 0,
    }
    for chunk in _chunked(unresolved, 100):
        payload, retries_used = _request_json(
            "orders/odds/best",
            params={"marketHashes": ",".join(chunk), "baseToken": base_token},
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        retries_used_total += retries_used
        if not isinstance(payload, dict):
            continue
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        best_odds = data.get("bestOdds") if isinstance(data.get("bestOdds"), list) else []
        for item in best_odds:
            if not isinstance(item, dict):
                continue
            lookup_meta["best_odds_items"] += 1
            market_hash = _normalize_text(item.get("marketHash"))
            if not market_hash:
                lookup_meta["best_odds_missing_market_hash"] += 1
                continue
            out_one = item.get("outcomeOne") if isinstance(item.get("outcomeOne"), dict) else {}
            out_two = item.get("outcomeTwo") if isinstance(item.get("outcomeTwo"), dict) else {}
            odds_entry = _best_odds_entry_from_payloads(out_one, out_two)
            observed_at = time.time()
            if odds_entry.get("odds_one") is not None:
                odds_entry["observed_at_one"] = observed_at
            if odds_entry.get("odds_two") is not None:
                odds_entry["observed_at_two"] = observed_at
            if odds_entry.get("odds_one") is not None or odds_entry.get("odds_two") is not None:
                odds_entry["observed_at"] = observed_at
            odds_one = odds_entry.get("odds_one")
            odds_two = odds_entry.get("odds_two")
            odds_map[market_hash] = odds_entry
            if odds_one is None and odds_two is None:
                lookup_meta["best_odds_null_count"] += 1
            elif odds_one is not None and odds_two is not None:
                lookup_meta["best_odds_with_both_odds"] += 1
                lookup_meta["best_odds_with_any_odds"] += 1
            else:
                lookup_meta["best_odds_with_any_odds"] += 1
            cache_entries[f"{base_token}:{market_hash}"] = odds_entry

    ODDS_CACHE["entries"] = cache_entries
    ODDS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    return odds_map, retries_used_total, lookup_meta


async def _load_best_odds_map_async(
    client: httpx.AsyncClient,
    market_hashes: Sequence[str],
    base_token: str,
    retries: int,
    backoff_seconds: float,
    cache_ttl_seconds: Optional[int] = None,
) -> Tuple[Dict[str, dict], int, dict]:
    ttl = (
        max(0, int(cache_ttl_seconds))
        if cache_ttl_seconds is not None
        else _int_or_default(SX_BET_ODDS_CACHE_TTL_RAW, 4, min_value=0)
    )
    now = time.time()
    cache_valid = ttl > 0 and now < float(ODDS_CACHE.get("expires_at", 0.0))
    cache_entries = ODDS_CACHE.get("entries") if isinstance(ODDS_CACHE.get("entries"), dict) else {}
    odds_map: Dict[str, dict] = {}
    unresolved: List[str] = []
    for market_hash in market_hashes:
        key = f"{base_token}:{market_hash}"
        if cache_valid and key in cache_entries:
            cached = cache_entries[key]
            if isinstance(cached, dict):
                odds_map[market_hash] = dict(cached)
            elif isinstance(cached, (list, tuple)) and len(cached) >= 2:
                odds_map[market_hash] = {
                    "odds_one": _safe_float(cached[0]),
                    "odds_two": _safe_float(cached[1]),
                    "updated_at_one": None,
                    "updated_at_two": None,
                    "source_one": "rest_cache",
                    "source_two": "rest_cache",
                }
        else:
            unresolved.append(market_hash)

    retries_used_total = 0
    lookup_meta = {
        "best_odds_items": 0,
        "best_odds_with_any_odds": 0,
        "best_odds_with_both_odds": 0,
        "best_odds_null_count": 0,
        "best_odds_missing_market_hash": 0,
    }
    for chunk in _chunked(unresolved, 100):
        payload, retries_used = await _request_json_async(
            client,
            "orders/odds/best",
            params={"marketHashes": ",".join(chunk), "baseToken": base_token},
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        retries_used_total += retries_used
        if not isinstance(payload, dict):
            continue
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        best_odds = data.get("bestOdds") if isinstance(data.get("bestOdds"), list) else []
        for item in best_odds:
            if not isinstance(item, dict):
                continue
            lookup_meta["best_odds_items"] += 1
            market_hash = _normalize_text(item.get("marketHash"))
            if not market_hash:
                lookup_meta["best_odds_missing_market_hash"] += 1
                continue
            out_one = item.get("outcomeOne") if isinstance(item.get("outcomeOne"), dict) else {}
            out_two = item.get("outcomeTwo") if isinstance(item.get("outcomeTwo"), dict) else {}
            odds_entry = _best_odds_entry_from_payloads(out_one, out_two)
            observed_at = time.time()
            if odds_entry.get("odds_one") is not None:
                odds_entry["observed_at_one"] = observed_at
            if odds_entry.get("odds_two") is not None:
                odds_entry["observed_at_two"] = observed_at
            if odds_entry.get("odds_one") is not None or odds_entry.get("odds_two") is not None:
                odds_entry["observed_at"] = observed_at
            odds_one = odds_entry.get("odds_one")
            odds_two = odds_entry.get("odds_two")
            odds_map[market_hash] = odds_entry
            if odds_one is None and odds_two is None:
                lookup_meta["best_odds_null_count"] += 1
            elif odds_one is not None and odds_two is not None:
                lookup_meta["best_odds_with_both_odds"] += 1
                lookup_meta["best_odds_with_any_odds"] += 1
            else:
                lookup_meta["best_odds_with_any_odds"] += 1
            cache_entries[f"{base_token}:{market_hash}"] = odds_entry

    ODDS_CACHE["entries"] = cache_entries
    ODDS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    return odds_map, retries_used_total, lookup_meta


def _load_best_stake_map(
    market_hashes: Sequence[str],
    base_token: str,
    retries: int,
    backoff_seconds: float,
    base_token_decimals: int,
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], int, dict]:
    ttl = _int_or_default(SX_BET_ORDER_CACHE_TTL_RAW, 2, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(ORDERS_CACHE.get("expires_at", 0.0))
    cache_entries = ORDERS_CACHE.get("entries") if isinstance(ORDERS_CACHE.get("entries"), dict) else {}

    stake_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    unresolved: List[str] = []
    for market_hash in market_hashes:
        cache_key = f"{base_token}:{market_hash}"
        cached = cache_entries.get(cache_key) if cache_valid else None
        if isinstance(cached, (list, tuple)) and len(cached) >= 2:
            stake_one = _safe_float(cached[0])
            stake_two = _safe_float(cached[1])
            stake_map[market_hash] = (
                round(float(stake_one), 6) if stake_one is not None and stake_one > 0 else None,
                round(float(stake_two), 6) if stake_two is not None and stake_two > 0 else None,
            )
        else:
            unresolved.append(market_hash)

    retries_used_total = 0
    lookup_meta = {
        "orders_rows": 0,
        "orders_missing_market_hash": 0,
    }
    best_by_market: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
    for chunk in _chunked(unresolved, 100):
        payload, retries_used = _request_json(
            "orders",
            params={"marketHashes": ",".join(chunk), "baseToken": base_token},
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        retries_used_total += retries_used
        for order in _extract_orders_payload(payload):
            lookup_meta["orders_rows"] += 1
            market_hash = _normalize_text(order.get("marketHash"))
            if not market_hash:
                lookup_meta["orders_missing_market_hash"] += 1
                continue
            maker_outcome_one = _bool_or_none(order.get("isMakerBettingOutcomeOne"))
            if maker_outcome_one is None:
                continue
            total_size = _token_amount_from_raw(order.get("totalBetSize"), base_token_decimals)
            if total_size is None or total_size <= 0:
                continue
            filled_size = _token_amount_from_raw(order.get("fillAmount"), base_token_decimals) or 0.0
            remaining_size = max(0.0, float(total_size) - float(filled_size))
            if remaining_size <= 0:
                continue
            odds = _moneyline_decimal_from_percentage(order.get("percentageOdds"))
            if odds is None or odds <= 1:
                continue
            side = 1 if maker_outcome_one else 2
            market_state = best_by_market.setdefault(
                market_hash,
                {
                    1: {"best_odds": None, "stake": 0.0},
                    2: {"best_odds": None, "stake": 0.0},
                },
            )
            side_state = market_state[side]
            best_odds = _safe_float(side_state.get("best_odds"))
            if best_odds is None or odds > best_odds + 1e-9:
                side_state["best_odds"] = float(odds)
                side_state["stake"] = float(remaining_size)
            elif abs(odds - best_odds) <= 1e-9:
                previous_stake = _safe_float(side_state.get("stake")) or 0.0
                side_state["stake"] = previous_stake + float(remaining_size)

    for market_hash in unresolved:
        market_state = best_by_market.get(market_hash) if isinstance(best_by_market.get(market_hash), dict) else {}
        side_one_state = market_state.get(1) if isinstance(market_state, dict) else {}
        side_two_state = market_state.get(2) if isinstance(market_state, dict) else {}
        stake_one_raw = _safe_float(side_one_state.get("stake")) if isinstance(side_one_state, dict) else None
        stake_two_raw = _safe_float(side_two_state.get("stake")) if isinstance(side_two_state, dict) else None
        stake_one = round(float(stake_one_raw), 6) if stake_one_raw is not None and stake_one_raw > 0 else None
        stake_two = round(float(stake_two_raw), 6) if stake_two_raw is not None and stake_two_raw > 0 else None
        stake_pair = (stake_one, stake_two)
        stake_map[market_hash] = stake_pair
        cache_entries[f"{base_token}:{market_hash}"] = stake_pair

    ORDERS_CACHE["entries"] = cache_entries
    ORDERS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    return stake_map, retries_used_total, lookup_meta


async def _load_best_stake_map_async(
    client: httpx.AsyncClient,
    market_hashes: Sequence[str],
    base_token: str,
    retries: int,
    backoff_seconds: float,
    base_token_decimals: int,
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], int, dict]:
    ttl = _int_or_default(SX_BET_ORDER_CACHE_TTL_RAW, 2, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(ORDERS_CACHE.get("expires_at", 0.0))
    cache_entries = ORDERS_CACHE.get("entries") if isinstance(ORDERS_CACHE.get("entries"), dict) else {}

    stake_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    unresolved: List[str] = []
    for market_hash in market_hashes:
        cache_key = f"{base_token}:{market_hash}"
        cached = cache_entries.get(cache_key) if cache_valid else None
        if isinstance(cached, (list, tuple)) and len(cached) >= 2:
            stake_one = _safe_float(cached[0])
            stake_two = _safe_float(cached[1])
            stake_map[market_hash] = (
                round(float(stake_one), 6) if stake_one is not None and stake_one > 0 else None,
                round(float(stake_two), 6) if stake_two is not None and stake_two > 0 else None,
            )
        else:
            unresolved.append(market_hash)

    retries_used_total = 0
    lookup_meta = {
        "orders_rows": 0,
        "orders_missing_market_hash": 0,
    }
    best_by_market: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
    for chunk in _chunked(unresolved, 100):
        payload, retries_used = await _request_json_async(
            client,
            "orders",
            params={"marketHashes": ",".join(chunk), "baseToken": base_token},
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        retries_used_total += retries_used
        for order in _extract_orders_payload(payload):
            lookup_meta["orders_rows"] += 1
            market_hash = _normalize_text(order.get("marketHash"))
            if not market_hash:
                lookup_meta["orders_missing_market_hash"] += 1
                continue
            maker_outcome_one = _bool_or_none(order.get("isMakerBettingOutcomeOne"))
            if maker_outcome_one is None:
                continue
            total_size = _token_amount_from_raw(order.get("totalBetSize"), base_token_decimals)
            if total_size is None or total_size <= 0:
                continue
            filled_size = _token_amount_from_raw(order.get("fillAmount"), base_token_decimals) or 0.0
            remaining_size = max(0.0, float(total_size) - float(filled_size))
            if remaining_size <= 0:
                continue
            odds = _moneyline_decimal_from_percentage(order.get("percentageOdds"))
            if odds is None or odds <= 1:
                continue
            side = 1 if maker_outcome_one else 2
            market_state = best_by_market.setdefault(
                market_hash,
                {
                    1: {"best_odds": None, "stake": 0.0},
                    2: {"best_odds": None, "stake": 0.0},
                },
            )
            side_state = market_state[side]
            best_odds = _safe_float(side_state.get("best_odds"))
            if best_odds is None or odds > best_odds + 1e-9:
                side_state["best_odds"] = float(odds)
                side_state["stake"] = float(remaining_size)
            elif abs(odds - best_odds) <= 1e-9:
                previous_stake = _safe_float(side_state.get("stake")) or 0.0
                side_state["stake"] = previous_stake + float(remaining_size)

    for market_hash in unresolved:
        market_state = best_by_market.get(market_hash) if isinstance(best_by_market.get(market_hash), dict) else {}
        side_one_state = market_state.get(1) if isinstance(market_state, dict) else {}
        side_two_state = market_state.get(2) if isinstance(market_state, dict) else {}
        stake_one_raw = _safe_float(side_one_state.get("stake")) if isinstance(side_one_state, dict) else None
        stake_two_raw = _safe_float(side_two_state.get("stake")) if isinstance(side_two_state, dict) else None
        stake_one = round(float(stake_one_raw), 6) if stake_one_raw is not None and stake_one_raw > 0 else None
        stake_two = round(float(stake_two_raw), 6) if stake_two_raw is not None and stake_two_raw > 0 else None
        stake_pair = (stake_one, stake_two)
        stake_map[market_hash] = stake_pair
        cache_entries[f"{base_token}:{market_hash}"] = stake_pair

    ORDERS_CACHE["entries"] = cache_entries
    ORDERS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    return stake_map, retries_used_total, lookup_meta


def _set_last_stats(stats: dict) -> None:
    fetch_events.last_stats = stats
    fetch_events_async.last_stats = stats


async def fetch_events_async(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
    context: Optional[dict] = None,
) -> List[dict]:
    _ = regions
    live_context = isinstance(context, dict) and bool(context.get("live"))
    realtime_manager = _get_realtime_manager() if _sx_best_odds_ws_enabled() else None
    stats = {
        "provider": PROVIDER_KEY,
        "source": SX_BET_SOURCE or "api",
        "fixture_source_mode": _fixture_source_mode(context),
        "fixture_source_used": "",
        "fixture_source_fallback": False,
        "fixture_source_fallback_reason": "",
        "sport_id": None,
        "fixtures_payload_count": 0,
        "fixtures_sport_filtered_count": 0,
        "fixtures_filtered_out_by_league_count": 0,
        "fixtures_missing_team_count": 0,
        "fixtures_missing_commence_count": 0,
        "fixtures_no_markets_count": 0,
        "fixtures_live_context_filtered_count": 0,
        "fixtures_market_found_count": 0,
        "candidates_total_count": 0,
        "events_returned_count": 0,
        "odds_lookup_requested": 0,
        "odds_lookup_resolved": 0,
        "odds_lookup_resolved_partial": 0,
        "odds_lookup_unresolved_after_lookup": 0,
        "odds_lookup_response_count": 0,
        "odds_lookup_missing_hash_entries": 0,
        "odds_lookup_null_entries": 0,
        "odds_lookup_sample_missing_hashes": [],
        "odds_lookup_sample_null_hashes": [],
        "orders_lookup_requested": 0,
        "orders_lookup_with_any_stake": 0,
        "orders_lookup_without_stake": 0,
        "orders_lookup_response_count": 0,
        "orders_lookup_missing_hash_entries": 0,
        "orders_lookup_missing_market_hash_rows": 0,
        "orders_lookup_sample_missing_hashes": [],
        "dropped_missing_odds_count": 0,
        "dropped_invalid_odds_count": 0,
        "sample_dropped_market_hashes": [],
        "sample_filtered_leagues": [],
        "pages_fetched": 0,
        "retries_used": 0,
        "payload_cache": "miss",
        "leagues_requested": 0,
        "leagues_failed": 0,
        "markets_rows_total": 0,
        "markets_rows_main_line_filtered": 0,
        "markets_rows_missing_event": 0,
        "markets_rows_missing_team": 0,
        "realtime_ws_enabled": _sx_best_odds_ws_enabled(),
        "realtime_connected": False,
        "realtime_ready": False,
        "realtime_best_odds_cached": 0,
        "realtime_messages_received": 0,
        "realtime_last_message_age_seconds": None,
        "realtime_stream_hits": 0,
        "realtime_snapshot_seeded": 0,
        "realtime_odds_hits": 0,
        "realtime_odds_missed": 0,
    }
    _set_last_stats(stats)

    requested_markets = _requested_market_keys(markets)
    if not requested_markets:
        return []

    if bookmakers:
        lowered = {str(book).strip().lower() for book in bookmakers if isinstance(book, str)}
        if PROVIDER_KEY not in lowered and PROVIDER_TITLE.lower() not in lowered:
            return []

    if (SX_BET_SOURCE or "api").lower() != "api":
        raise ProviderError("SX Bet provider currently supports SX_BET_SOURCE=api only")

    sport_id = SX_SPORT_ID_MAP.get(sport_key)
    stats["sport_id"] = sport_id
    if sport_id is None:
        return []

    base_token = _normalize_text(SX_BET_BASE_TOKEN)
    if not base_token:
        raise ProviderError("SX_BET_BASE_TOKEN is required")

    retries = _int_or_default(SX_BET_RETRIES_RAW, 2, min_value=0)
    backoff = _float_or_default(SX_BET_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    base_token_decimals = _int_or_default(SX_BET_BASE_TOKEN_DECIMALS_RAW, 6, min_value=0)
    manual_league_map = _parse_manual_league_map()
    timeout = _int_or_default(SX_BET_TIMEOUT_RAW, 20, min_value=1)
    client = await get_shared_client(PROVIDER_KEY, timeout=float(timeout), follow_redirects=True)
    realtime_odds_map: Dict[str, dict] = {}
    if realtime_manager is not None:
        realtime_manager.ensure_started()
        wait_seconds = _sx_ws_startup_wait_seconds() if live_context else 0.0
        ready = realtime_manager.wait_until_ready(wait_seconds) if wait_seconds > 0 else False
        realtime_snapshot = realtime_manager.snapshot()
        stats["realtime_connected"] = bool(realtime_snapshot.get("connected"))
        stats["realtime_ready"] = bool(ready or realtime_snapshot.get("connected"))
        stats["realtime_best_odds_cached"] = int(realtime_snapshot.get("best_odds_cached", 0) or 0)
        stats["realtime_messages_received"] = int(realtime_snapshot.get("messages_received", 0) or 0)
        stats["realtime_last_message_age_seconds"] = realtime_snapshot.get("last_message_age_seconds")
    fixtures, meta = await _load_upcoming_fixtures_async(
        client=client,
        sport_id=sport_id,
        base_token=base_token,
        retries=retries,
        backoff_seconds=backoff,
        context=context,
    )
    stats["payload_cache"] = meta.get("cache", "miss")
    stats["pages_fetched"] = int(meta.get("pages_fetched", 0) or 0)
    stats["retries_used"] += int(meta.get("retries_used", 0) or 0)
    stats["fixture_source_used"] = _normalize_text(meta.get("fixture_source")) or (
        "summary" if _fixture_source_mode() == "summary" else "markets_active"
    )
    stats["fixture_source_fallback"] = bool(meta.get("fallback_used"))
    if stats["fixture_source_fallback"]:
        stats["fixture_source_fallback_reason"] = _normalize_text(meta.get("fallback_reason"))
    stats["leagues_requested"] = int(meta.get("leagues_requested", 0) or 0)
    stats["leagues_failed"] = int(meta.get("leagues_failed", 0) or 0)
    stats["markets_rows_total"] = int(meta.get("markets_rows_total", 0) or 0)
    stats["markets_rows_main_line_filtered"] = int(meta.get("markets_rows_main_line_filtered", 0) or 0)
    stats["markets_rows_missing_event"] = int(meta.get("markets_rows_missing_event", 0) or 0)
    stats["markets_rows_missing_team"] = int(meta.get("markets_rows_missing_team", 0) or 0)
    stats["fixtures_payload_count"] = len(fixtures)

    if live_context and _normalize_text(stats.get('fixture_source_used')) == 'markets_active':
        fixture_status_map = await _load_fixture_status_map_async(
            client,
            [fixture.get('eventId') or fixture.get('id') for fixture in fixtures if isinstance(fixture, dict)],
            retries=retries,
            backoff_seconds=backoff,
        )
        live_scores_map = await _load_live_scores_map_async(
            client,
            [fixture.get('eventId') or fixture.get('id') for fixture in fixtures if isinstance(fixture, dict)],
            retries=retries,
            backoff_seconds=backoff,
        )
        for fixture in fixtures:
            if not isinstance(fixture, dict):
                continue
            fixture_event_id = _normalize_text(fixture.get('eventId') or fixture.get('id'))
            if not fixture_event_id:
                continue
            fixture_status_row = fixture_status_map.get(fixture_event_id)
            if not isinstance(fixture_status_row, dict):
                fixture_status_row = {}
            fixture['live_state'] = _merge_live_state_payload(
                fixture.get('live_state') if isinstance(fixture.get('live_state'), dict) else None,
                _sx_fixture_status_live_state_payload(fixture_status_row.get('status')),
            )
            fixture['live_state'] = _merge_live_state_payload(
                fixture.get('live_state') if isinstance(fixture.get('live_state'), dict) else None,
                _sx_live_scores_live_state_payload(live_scores_map.get(fixture_event_id)),
            )

    candidates: List[dict] = []
    unresolved_hashes: List[str] = []
    candidate_hashes: List[str] = []
    for fixture in fixtures:
        if (
            live_context
            and _normalize_text(stats.get("fixture_source_used")) == "markets_active"
            and not _fixture_has_live_evidence(fixture)
        ):
            stats["fixtures_live_context_filtered_count"] += 1
            continue
        if not _fixture_matches_sport(sport_key, fixture, manual_league_map):
            stats["fixtures_filtered_out_by_league_count"] += 1
            if True:
                league_label = _normalize_text(fixture.get("leagueLabel"))
                if league_label and len(stats["sample_filtered_leagues"]) < 5:
                    if league_label not in stats["sample_filtered_leagues"]:
                        stats["sample_filtered_leagues"].append(league_label)
                continue
        else:
            stats["fixtures_sport_filtered_count"] += 1
            team_one = _normalize_text(fixture.get("teamOne"))
            team_two = _normalize_text(fixture.get("teamTwo"))
            if not (team_one and team_two):
                stats["fixtures_missing_team_count"] += 1
                continue
            commence = _normalize_commence_time(fixture.get("gameTime"))
            if not commence:
                stats["fixtures_missing_commence_count"] += 1
                continue

            markets_list = fixture.get("markets")
            if not isinstance(markets_list, list):
                stats["fixtures_no_markets_count"] += 1
                continue
            fixture_event_id = _normalize_text(fixture.get("eventId") or fixture.get("id"))
            fixture_id = _normalize_text(fixture.get("id") or fixture.get("eventId"))
            for market in markets_list:
                normalized_market = _normalize_fixture_market(
                    market=market,
                    requested_markets=requested_markets,
                    home_team=team_one,
                    away_team=team_two,
                )
                if not normalized_market:
                    continue
                stats["fixtures_market_found_count"] += 1
                stats["candidates_total_count"] += 1
                market_hash = normalized_market.get("market_hash")
                if market_hash:
                    candidate_hashes.append(str(market_hash))
                if (
                    (normalized_market.get("odds_one") is None or normalized_market.get("odds_two") is None)
                    and market_hash
                ):
                    unresolved_hashes.append(str(market_hash))
                candidate_live_state = _merge_live_state_payload(
                    fixture.get('live_state') if isinstance(fixture.get('live_state'), dict) else None,
                    _sx_live_state_payload(market),
                )
                candidates.append(
                    {
                        "id": fixture_id or str(market_hash or ""),
                        "event_id": fixture_event_id or str(market_hash or ""),
                        "sport_id": fixture.get("sportId"),
                        "sport_label": fixture.get("sportLabel"),
                        "league_label": fixture.get("leagueLabel"),
                        "home_team": team_one,
                        "away_team": team_two,
                        "commence_time": commence,
                        "live_state": _sx_live_state_payload(fixture, market),
                        **normalized_market,
                    }
                )
                candidates[-1]['live_state'] = candidate_live_state

    if True:
        odds_map: Dict[str, dict] = {}
        stake_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        odds_task = None
        stake_task = None
        candidate_unique_hashes = list(dict.fromkeys(candidate_hashes))
        if realtime_manager is not None and candidate_hashes:
            realtime_odds_map = realtime_manager.get_best_odds_map(
                candidate_unique_hashes,
                max_age_seconds=_sx_ws_quote_max_age_seconds(),
            )
            stats["realtime_stream_hits"] = len(realtime_odds_map)
            stats["realtime_odds_hits"] = len(realtime_odds_map)
            stats["realtime_odds_missed"] = max(0, len(candidate_unique_hashes) - len(realtime_odds_map))
        odds_lookup_hashes: List[str] = []
        if stats.get("fixture_source_used") == "summary" and candidate_hashes:
            odds_lookup_hashes = [
                market_hash
                for market_hash in candidate_unique_hashes
                if market_hash not in realtime_odds_map
            ]
        elif unresolved_hashes:
            odds_lookup_hashes = [
                market_hash
                for market_hash in list(dict.fromkeys(unresolved_hashes))
                if market_hash not in realtime_odds_map
            ]
        if odds_lookup_hashes:
            unique_hashes = odds_lookup_hashes
            stats["odds_lookup_requested"] = len(unique_hashes)
            odds_task = asyncio.create_task(
                _load_best_odds_map_async(
                    client=client,
                    market_hashes=unique_hashes,
                    base_token=base_token,
                    retries=retries,
                    backoff_seconds=backoff,
                    cache_ttl_seconds=0 if live_context else None,
                )
            )
        if candidate_hashes:
            unique_hashes = list(dict.fromkeys(candidate_hashes))
            stats["orders_lookup_requested"] = len(unique_hashes)
            stake_task = asyncio.create_task(
                _load_best_stake_map_async(
                    client=client,
                    market_hashes=unique_hashes,
                    base_token=base_token,
                    retries=retries,
                    backoff_seconds=backoff,
                    base_token_decimals=base_token_decimals,
                )
            )

        if odds_task is not None:
            odds_map, retries_used, lookup_meta = await odds_task
            stats["retries_used"] += retries_used
            stats["odds_lookup_response_count"] = int(lookup_meta.get("best_odds_items", 0) or 0)
            stats["odds_lookup_null_entries"] = int(lookup_meta.get("best_odds_null_count", 0) or 0)
            resolved_partial = int(lookup_meta.get("best_odds_with_any_odds", 0) or 0) - int(
                lookup_meta.get("best_odds_with_both_odds", 0) or 0
            )
            stats["odds_lookup_resolved_partial"] = max(0, resolved_partial)
            lookup_unique_hashes = odds_lookup_hashes
            missing_hashes = [market_hash for market_hash in lookup_unique_hashes if market_hash not in odds_map]
            null_hashes = [
                market_hash
                for market_hash in lookup_unique_hashes
                if market_hash in odds_map
                and odds_map[market_hash].get("odds_one") is None
                and odds_map[market_hash].get("odds_two") is None
            ]
            stats["odds_lookup_missing_hash_entries"] = len(missing_hashes)
            stats["odds_lookup_sample_missing_hashes"] = missing_hashes[:5]
            stats["odds_lookup_sample_null_hashes"] = null_hashes[:5]
            stats["odds_lookup_resolved"] = sum(
                1
                for market_hash in lookup_unique_hashes
                if market_hash in odds_map
                and odds_map[market_hash].get("odds_one") is not None
                and odds_map[market_hash].get("odds_two") is not None
            )
            stats["odds_lookup_unresolved_after_lookup"] = len(lookup_unique_hashes) - int(
                stats.get("odds_lookup_resolved", 0) or 0
            )
            if realtime_manager is not None and odds_map:
                stats["realtime_snapshot_seeded"] = realtime_manager.merge_best_odds_map(
                    odds_map,
                    source="rest_snapshot",
                )
                realtime_odds_map = realtime_manager.get_best_odds_map(
                    candidate_unique_hashes,
                    max_age_seconds=_sx_ws_quote_max_age_seconds(),
                )
                stats["realtime_odds_hits"] = len(realtime_odds_map)
                stats["realtime_odds_missed"] = max(0, len(candidate_unique_hashes) - len(realtime_odds_map))

        if stake_task is not None:
            stake_map, retries_used, order_meta = await stake_task
            stats["retries_used"] += retries_used
            stats["orders_lookup_response_count"] = int(order_meta.get("orders_rows", 0) or 0)
            stats["orders_lookup_missing_market_hash_rows"] = int(
                order_meta.get("orders_missing_market_hash", 0) or 0
            )
            missing_hashes = [market_hash for market_hash in candidate_unique_hashes if market_hash not in stake_map]
            stats["orders_lookup_missing_hash_entries"] = len(missing_hashes)
            stats["orders_lookup_sample_missing_hashes"] = missing_hashes[:5]
            with_any_stake = sum(
                1
                for market_hash in candidate_unique_hashes
                if market_hash in stake_map
                and (
                    stake_map[market_hash][0] is not None
                    or stake_map[market_hash][1] is not None
                )
            )
            stats["orders_lookup_with_any_stake"] = with_any_stake
            stats["orders_lookup_without_stake"] = len(candidate_unique_hashes) - with_any_stake

        if realtime_manager is not None:
            realtime_snapshot = realtime_manager.snapshot()
            stats["realtime_connected"] = bool(realtime_snapshot.get("connected"))
            stats["realtime_ready"] = bool(stats.get("realtime_ready")) or bool(realtime_snapshot.get("connected"))
            stats["realtime_best_odds_cached"] = int(realtime_snapshot.get("best_odds_cached", 0) or 0)
            stats["realtime_messages_received"] = int(realtime_snapshot.get("messages_received", 0) or 0)
            stats["realtime_last_message_age_seconds"] = realtime_snapshot.get("last_message_age_seconds")

    events_by_id: Dict[str, dict] = {}
    for candidate in candidates:
        odds_one = candidate.get("odds_one")
        odds_two = candidate.get("odds_two")
        outcome_one_last_updated = candidate.get("outcome_one_last_updated")
        outcome_two_last_updated = candidate.get("outcome_two_last_updated")
        outcome_one_observed_at = candidate.get("outcome_one_observed_at")
        outcome_two_observed_at = candidate.get("outcome_two_observed_at")
        outcome_one_quote_source = candidate.get("outcome_one_quote_source")
        outcome_two_quote_source = candidate.get("outcome_two_quote_source")
        outcome_one_raw_percentage = candidate.get("outcome_one_raw_percentage_odds")
        outcome_two_raw_percentage = candidate.get("outcome_two_raw_percentage_odds")
        market_hash = candidate.get("market_hash")
        if market_hash:
            mapped = odds_map.get(market_hash)
            if mapped:
                mapped_odds_one = _safe_float(mapped.get("odds_one"))
                mapped_odds_two = _safe_float(mapped.get("odds_two"))
                if mapped_odds_one is not None:
                    odds_one = mapped_odds_one
                if mapped_odds_two is not None:
                    odds_two = mapped_odds_two
                outcome_one_last_updated = outcome_one_last_updated or mapped.get("updated_at_one")
                outcome_two_last_updated = outcome_two_last_updated or mapped.get("updated_at_two")
                outcome_one_observed_at = outcome_one_observed_at or mapped.get("observed_at_one")
                outcome_two_observed_at = outcome_two_observed_at or mapped.get("observed_at_two")
                if mapped.get("raw_percentage_one") not in (None, ""):
                    outcome_one_raw_percentage = mapped.get("raw_percentage_one")
                if mapped.get("raw_percentage_two") not in (None, ""):
                    outcome_two_raw_percentage = mapped.get("raw_percentage_two")
                if mapped.get("source_one") not in (None, ""):
                    outcome_one_quote_source = mapped.get("source_one")
                if mapped.get("source_two") not in (None, ""):
                    outcome_two_quote_source = mapped.get("source_two")
            realtime_mapped = realtime_odds_map.get(market_hash)
            if realtime_mapped:
                realtime_odds_one = _safe_float(realtime_mapped.get("odds_one"))
                realtime_odds_two = _safe_float(realtime_mapped.get("odds_two"))
                if realtime_odds_one is not None:
                    odds_one = realtime_odds_one
                if realtime_odds_two is not None:
                    odds_two = realtime_odds_two
                if realtime_mapped.get("updated_at_one") not in (None, ""):
                    outcome_one_last_updated = realtime_mapped.get("updated_at_one")
                if realtime_mapped.get("updated_at_two") not in (None, ""):
                    outcome_two_last_updated = realtime_mapped.get("updated_at_two")
                if realtime_mapped.get("observed_at_one") not in (None, ""):
                    outcome_one_observed_at = realtime_mapped.get("observed_at_one")
                if realtime_mapped.get("observed_at_two") not in (None, ""):
                    outcome_two_observed_at = realtime_mapped.get("observed_at_two")
                if realtime_mapped.get("source_one") not in (None, ""):
                    outcome_one_quote_source = realtime_mapped.get("source_one")
                if realtime_mapped.get("source_two") not in (None, ""):
                    outcome_two_quote_source = realtime_mapped.get("source_two")
        default_fixture_source = f"fixture_{_normalize_text(stats.get('fixture_source_used')) or 'summary'}"
        if outcome_one_quote_source in (None, ""):
            outcome_one_quote_source = default_fixture_source
        if outcome_two_quote_source in (None, ""):
            outcome_two_quote_source = default_fixture_source
        if odds_one is None or odds_two is None:
            stats["dropped_missing_odds_count"] += 1
            if market_hash and len(stats["sample_dropped_market_hashes"]) < 5:
                if market_hash not in stats["sample_dropped_market_hashes"]:
                    stats["sample_dropped_market_hashes"].append(market_hash)
            continue
        if odds_one <= 1 or odds_two <= 1:
            stats["dropped_invalid_odds_count"] += 1
            continue

        stake_one: Optional[float] = None
        stake_two: Optional[float] = None
        if market_hash:
            mapped_stakes = stake_map.get(str(market_hash))
            if mapped_stakes:
                stake_one = _safe_float(mapped_stakes[0])
                stake_two = _safe_float(mapped_stakes[1])

        event_id = candidate.get("event_id") or candidate.get("id")
        home_team = candidate.get("home_team")
        away_team = candidate.get("away_team")
        if not (event_id and home_team and away_team):
            continue
        event = events_by_id.setdefault(
            str(event_id),
            {
                "id": candidate.get("id") or event_id,
                "event_id": str(event_id),
                "sport_key": sport_key,
                "sport_id": candidate.get("sport_id"),
                "sport_label": candidate.get("sport_label"),
                "league_label": candidate.get("league_label"),
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": candidate.get("commence_time"),
                "live_state": candidate.get("live_state"),
                "markets_by_sig": {},
            },
        )
        event["live_state"] = _merge_live_state_payload(
            event.get("live_state") if isinstance(event.get("live_state"), dict) else None,
            candidate.get("live_state") if isinstance(candidate.get("live_state"), dict) else None,
        )

        outcomes = [
            {
                "name": _normalize_text(candidate.get("outcome_one_name")) or home_team,
                "price": round(float(odds_one), 6),
            },
            {
                "name": _normalize_text(candidate.get("outcome_two_name")) or away_team,
                "price": round(float(odds_two), 6),
            },
        ]
        point_one = _safe_float(candidate.get("outcome_one_point"))
        point_two = _safe_float(candidate.get("outcome_two_point"))
        if point_one is not None:
            outcomes[0]["point"] = round(float(point_one), 6)
        if point_two is not None:
            outcomes[1]["point"] = round(float(point_two), 6)
        if stake_one is not None and stake_one > 0:
            outcomes[0]["stake"] = round(float(stake_one), 6)
        if stake_two is not None and stake_two > 0:
            outcomes[1]["stake"] = round(float(stake_two), 6)
        if outcome_one_raw_percentage not in (None, ""):
            outcomes[0]["raw_percentage_odds"] = outcome_one_raw_percentage
        if outcome_two_raw_percentage not in (None, ""):
            outcomes[1]["raw_percentage_odds"] = outcome_two_raw_percentage
        if outcome_one_last_updated not in (None, ""):
            outcomes[0]["last_updated"] = outcome_one_last_updated
        if outcome_two_last_updated not in (None, ""):
            outcomes[1]["last_updated"] = outcome_two_last_updated
        if outcome_one_observed_at not in (None, ""):
            outcomes[0]["observed_at"] = outcome_one_observed_at
        if outcome_two_observed_at not in (None, ""):
            outcomes[1]["observed_at"] = outcome_two_observed_at
        if outcome_one_quote_source not in (None, ""):
            outcomes[0]["quote_source"] = outcome_one_quote_source
        if outcome_two_quote_source not in (None, ""):
            outcomes[1]["quote_source"] = outcome_two_quote_source

        description = _normalize_text(candidate.get("description"))
        if description and _base_market_key(candidate.get("market_key")) not in {"h2h", "spreads", "totals"}:
            outcomes[0]["description"] = description
            outcomes[1]["description"] = description

        market_payload = {
            "key": _normalize_text(candidate.get("market_key")).lower(),
            "outcomes": outcomes,
        }
        signature = _market_signature(market_payload)
        previous = event["markets_by_sig"].get(signature)
        if previous is None or _market_score(market_payload) > _market_score(previous):
            event["markets_by_sig"][signature] = market_payload

    events_out: List[dict] = []
    for event in events_by_id.values():
        market_list = list(event.get("markets_by_sig", {}).values())
        if not market_list:
            continue
        events_out.append(
            {
                "id": event["id"],
                "sport_key": sport_key,
                "home_team": event["home_team"],
                "away_team": event["away_team"],
                "commence_time": event["commence_time"],
                "live_state": event.get("live_state"),
                "bookmakers": [
                    {
                        "key": PROVIDER_KEY,
                        "title": PROVIDER_TITLE,
                        "event_id": event["event_id"],
                        "event_url": _event_url(event),
                        "live_state": event.get("live_state"),
                        "markets": market_list,
                    }
                ],
            }
        )

    stats["events_returned_count"] = len(events_out)
    _set_last_stats(stats)
    return events_out


def fetch_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
    context: Optional[dict] = None,
) -> List[dict]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            fetch_events_async(
                sport_key,
                markets,
                regions,
                bookmakers=bookmakers,
                context=context,
            )
        )
    raise RuntimeError("fetch_events() cannot be used inside an active event loop; use await fetch_events_async()")


fetch_events.last_stats = {
    "provider": PROVIDER_KEY,
    "source": SX_BET_SOURCE or "api",
    "fixture_source_mode": _fixture_source_mode(),
    "fixture_source_used": "",
    "events_returned_count": 0,
}
fetch_events_async.last_stats = dict(fetch_events.last_stats)

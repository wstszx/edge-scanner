from __future__ import annotations

import asyncio
import concurrent.futures
import datetime as dt
import json
import os
import re
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import httpx
import requests

from ._async_http import get_shared_client, request_json

PROVIDER_KEY = "polymarket"
PROVIDER_TITLE = "Polymarket"

POLYMARKET_SOURCE = os.getenv("POLYMARKET_SOURCE", "api").strip().lower()
POLYMARKET_API_BASE = os.getenv("POLYMARKET_API_BASE", "https://gamma-api.polymarket.com").strip()
POLYMARKET_CLOB_BASE = os.getenv("POLYMARKET_CLOB_BASE", "https://clob.polymarket.com").strip()
POLYMARKET_PUBLIC_BASE = os.getenv("POLYMARKET_PUBLIC_BASE", "https://polymarket.com").strip()
POLYMARKET_GAME_TAG_ID = os.getenv("POLYMARKET_GAME_TAG_ID", "100639").strip()
POLYMARKET_PAGE_SIZE_RAW = os.getenv("POLYMARKET_PAGE_SIZE", "200").strip()
POLYMARKET_MAX_PAGES_RAW = os.getenv("POLYMARKET_MAX_PAGES", "8").strip()
POLYMARKET_RETRIES_RAW = os.getenv("POLYMARKET_RETRIES", "2").strip()
POLYMARKET_RETRY_BACKOFF_RAW = os.getenv("POLYMARKET_RETRY_BACKOFF", "0.5").strip()
POLYMARKET_TIMEOUT_RAW = os.getenv("POLYMARKET_TIMEOUT_SECONDS", "20").strip()
POLYMARKET_SPORT_TAG_CACHE_TTL_RAW = os.getenv("POLYMARKET_SPORT_TAG_CACHE_TTL", "600").strip()
POLYMARKET_EVENTS_CACHE_TTL_RAW = os.getenv("POLYMARKET_EVENTS_CACHE_TTL", "12").strip()
POLYMARKET_CLOB_BOOK_CACHE_TTL_RAW = os.getenv("POLYMARKET_CLOB_BOOK_CACHE_TTL", "4").strip()
POLYMARKET_CLOB_MAX_BOOKS_RAW = os.getenv("POLYMARKET_CLOB_MAX_BOOKS", "300").strip()
POLYMARKET_CLOB_BOOK_WORKERS_RAW = os.getenv("POLYMARKET_CLOB_BOOK_WORKERS", "8").strip()
POLYMARKET_HTTP_CLOB_DEPTH_ENABLED = os.getenv(
    "POLYMARKET_HTTP_CLOB_DEPTH_ENABLED",
    "0",
).strip().lower() not in {"0", "false", "no", "off"}
POLYMARKET_USER_AGENT = os.getenv(
    "POLYMARKET_USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
).strip()
POLYMARKET_BEARER_TOKEN = os.getenv("POLYMARKET_BEARER_TOKEN", "").strip()
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "").strip()
POLYMARKET_COOKIE = os.getenv("POLYMARKET_COOKIE", "").strip()
POLYMARKET_WS_ENABLED = os.getenv("POLYMARKET_WS_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
POLYMARKET_MARKET_WS_ENABLED = os.getenv("POLYMARKET_MARKET_WS_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
POLYMARKET_SPORTS_WS_ENABLED = os.getenv("POLYMARKET_SPORTS_WS_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
POLYMARKET_MARKET_WS_URL = os.getenv(
    "POLYMARKET_MARKET_WS_URL",
    "wss://ws-subscriptions-clob.polymarket.com/ws/market",
).strip()
POLYMARKET_SPORTS_WS_URL = os.getenv(
    "POLYMARKET_SPORTS_WS_URL",
    "wss://sports-api.polymarket.com/ws",
).strip()
POLYMARKET_WS_WARMUP_MS_RAW = os.getenv("POLYMARKET_WS_WARMUP_MS", "250").strip()
POLYMARKET_WS_QUOTE_WAIT_SECONDS_RAW = os.getenv("POLYMARKET_WS_QUOTE_WAIT_SECONDS", "4").strip()
POLYMARKET_WS_BOOK_MAX_AGE_SECONDS_RAW = os.getenv(
    "POLYMARKET_WS_BOOK_MAX_AGE_SECONDS",
    "30",
).strip()
POLYMARKET_WS_CONNECT_TIMEOUT_RAW = os.getenv("POLYMARKET_WS_CONNECT_TIMEOUT_SECONDS", "15").strip()
POLYMARKET_WS_RECONNECT_MAX_DELAY_RAW = os.getenv("POLYMARKET_WS_RECONNECT_MAX_DELAY_SECONDS", "15").strip()
POLYMARKET_WS_STARTUP_WAIT_SECONDS_RAW = os.getenv("POLYMARKET_WS_STARTUP_WAIT_SECONDS", "1.5").strip()
POLYMARKET_REALTIME_SHARED_DIR = os.getenv(
    "POLYMARKET_REALTIME_SHARED_DIR",
    os.path.join("data", "polymarket_realtime"),
).strip()
POLYMARKET_REALTIME_OWNER_STALE_SECONDS_RAW = os.getenv(
    "POLYMARKET_REALTIME_OWNER_STALE_SECONDS",
    "30",
).strip()
POLYMARKET_REALTIME_SNAPSHOT_FLUSH_SECONDS_RAW = os.getenv(
    "POLYMARKET_REALTIME_SNAPSHOT_FLUSH_SECONDS",
    "1.0",
).strip()
POLYMARKET_REALTIME_SHARED_READ_TTL_MS_RAW = os.getenv(
    "POLYMARKET_REALTIME_SHARED_READ_TTL_MS",
    "500",
).strip()
POLYMARKET_REALTIME_CLEANUP_SECONDS_RAW = os.getenv(
    "POLYMARKET_REALTIME_CLEANUP_SECONDS",
    "30",
).strip()
POLYMARKET_REALTIME_SUBSCRIPTION_TTL_SECONDS_RAW = os.getenv(
    "POLYMARKET_REALTIME_SUBSCRIPTION_TTL_SECONDS",
    "900",
).strip()
POLYMARKET_REALTIME_SPORT_RESULT_TTL_SECONDS_RAW = os.getenv(
    "POLYMARKET_REALTIME_SPORT_RESULT_TTL_SECONDS",
    "10800",
).strip()
POLYMARKET_REALTIME_MAX_BOOK_ENTRIES_RAW = os.getenv(
    "POLYMARKET_REALTIME_MAX_BOOK_ENTRIES",
    "4000",
).strip()
POLYMARKET_REALTIME_MAX_SPORT_RESULTS_RAW = os.getenv(
    "POLYMARKET_REALTIME_MAX_SPORT_RESULTS",
    "2000",
).strip()

SPORT_TAG_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "mapping": {},
}
EVENTS_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "events": [],
}
CLOB_BOOK_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "entries": {},
}
CLOB_QUOTE_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "entries": {},
}
SCAN_CACHE_CONTEXT: Dict[str, object] = {
    "active": False,
    "events": None,
    "events_meta": {},
    "clob_entries": {},
    "clob_quotes": {},
}
SCAN_CACHE_LOCK = threading.RLock()
SPORT_TAG_CACHE_ASYNC_LOCK: Optional[asyncio.Lock] = None
EVENTS_CACHE_ASYNC_LOCK: Optional[asyncio.Lock] = None
SCAN_CACHE_ASYNC_LOCK: Optional[asyncio.Lock] = None
REALTIME_MANAGER: Optional["PolymarketRealtimeManager"] = None
REALTIME_MANAGER_LOCK = threading.Lock()

SPORT_ALIASES: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("nfl", "american-football"),
    "americanfootball_ncaaf": ("ncaaf", "ncaa-football", "college-football", "college-football-playoff"),
    "basketball_nba": ("nba",),
    "basketball_ncaab": ("ncaab", "ncaa", "ncaa-basketball"),
    "baseball_mlb": ("mlb", "baseball"),
    "icehockey_nhl": ("nhl", "hockey"),
    "soccer_epl": ("epl", "premier-league", "premier-league-uk", "premierleague"),
    "soccer_spain_la_liga": ("la-liga", "laliga"),
    "soccer_germany_bundesliga": ("bundesliga",),
    "soccer_italy_serie_a": ("serie-a", "serie-a-tim", "seriea"),
    "soccer_france_ligue_one": ("ligue-1", "ligue-1-uber-eats", "ligue1"),
    "soccer_usa_mls": ("mls", "major-league-soccer"),
}


class ProviderError(Exception):
    """Raised for provider-specific recoverable issues."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _sport_tag_cache_async_lock() -> asyncio.Lock:
    global SPORT_TAG_CACHE_ASYNC_LOCK
    if SPORT_TAG_CACHE_ASYNC_LOCK is None:
        SPORT_TAG_CACHE_ASYNC_LOCK = asyncio.Lock()
    return SPORT_TAG_CACHE_ASYNC_LOCK


def _events_cache_async_lock() -> asyncio.Lock:
    global EVENTS_CACHE_ASYNC_LOCK
    if EVENTS_CACHE_ASYNC_LOCK is None:
        EVENTS_CACHE_ASYNC_LOCK = asyncio.Lock()
    return EVENTS_CACHE_ASYNC_LOCK


def _scan_cache_async_lock() -> asyncio.Lock:
    global SCAN_CACHE_ASYNC_LOCK
    if SCAN_CACHE_ASYNC_LOCK is None:
        SCAN_CACHE_ASYNC_LOCK = asyncio.Lock()
    return SCAN_CACHE_ASYNC_LOCK


def enable_scan_cache() -> None:
    with SCAN_CACHE_LOCK:
        SCAN_CACHE_CONTEXT["active"] = True
        SCAN_CACHE_CONTEXT["events"] = None
        SCAN_CACHE_CONTEXT["events_meta"] = {}
        SCAN_CACHE_CONTEXT["clob_entries"] = {}
        SCAN_CACHE_CONTEXT["clob_quotes"] = {}


def disable_scan_cache() -> None:
    with SCAN_CACHE_LOCK:
        SCAN_CACHE_CONTEXT["active"] = False
        SCAN_CACHE_CONTEXT["events"] = None
        SCAN_CACHE_CONTEXT["events_meta"] = {}
        SCAN_CACHE_CONTEXT["clob_entries"] = {}
        SCAN_CACHE_CONTEXT["clob_quotes"] = {}


def _websocket_realtime_enabled() -> bool:
    return bool(POLYMARKET_WS_ENABLED) and (
        bool(POLYMARKET_MARKET_WS_ENABLED) or bool(POLYMARKET_SPORTS_WS_ENABLED)
    )


def _market_websocket_enabled() -> bool:
    return bool(POLYMARKET_WS_ENABLED) and bool(POLYMARKET_MARKET_WS_ENABLED)


def _sports_websocket_enabled() -> bool:
    return bool(POLYMARKET_WS_ENABLED) and bool(POLYMARKET_SPORTS_WS_ENABLED)


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


def _http_clob_depth_enabled() -> bool:
    return bool(POLYMARKET_HTTP_CLOB_DEPTH_ENABLED)


def _scan_cache_active() -> bool:
    with SCAN_CACHE_LOCK:
        return bool(SCAN_CACHE_CONTEXT.get("active"))


def _ws_warmup_seconds() -> float:
    return _float_or_default(POLYMARKET_WS_WARMUP_MS_RAW, 250.0, min_value=0.0) / 1000.0


def _ws_book_max_age_seconds() -> float:
    return _float_or_default(POLYMARKET_WS_BOOK_MAX_AGE_SECONDS_RAW, 30.0, min_value=0.0)


def _ws_quote_wait_seconds() -> float:
    return _float_or_default(POLYMARKET_WS_QUOTE_WAIT_SECONDS_RAW, 4.0, min_value=0.0)


def _ws_connect_timeout_seconds() -> float:
    return _float_or_default(POLYMARKET_WS_CONNECT_TIMEOUT_RAW, 15.0, min_value=1.0)


def _ws_reconnect_max_delay_seconds() -> float:
    return _float_or_default(POLYMARKET_WS_RECONNECT_MAX_DELAY_RAW, 15.0, min_value=1.0)


def _ws_startup_wait_seconds() -> float:
    return _float_or_default(POLYMARKET_WS_STARTUP_WAIT_SECONDS_RAW, 1.5, min_value=0.0)


def _realtime_owner_stale_seconds() -> float:
    return _float_or_default(POLYMARKET_REALTIME_OWNER_STALE_SECONDS_RAW, 30.0, min_value=1.0)


def _realtime_snapshot_flush_seconds() -> float:
    return _float_or_default(POLYMARKET_REALTIME_SNAPSHOT_FLUSH_SECONDS_RAW, 1.0, min_value=0.1)


def _realtime_shared_read_ttl_seconds() -> float:
    return _float_or_default(POLYMARKET_REALTIME_SHARED_READ_TTL_MS_RAW, 500.0, min_value=0.0) / 1000.0


def _realtime_cleanup_seconds() -> float:
    return _float_or_default(POLYMARKET_REALTIME_CLEANUP_SECONDS_RAW, 30.0, min_value=1.0)


def _realtime_subscription_ttl_seconds() -> float:
    return _float_or_default(POLYMARKET_REALTIME_SUBSCRIPTION_TTL_SECONDS_RAW, 900.0, min_value=30.0)


def _realtime_sport_result_ttl_seconds() -> float:
    return _float_or_default(POLYMARKET_REALTIME_SPORT_RESULT_TTL_SECONDS_RAW, 10800.0, min_value=60.0)


def _realtime_max_book_entries() -> int:
    return _int_or_default(POLYMARKET_REALTIME_MAX_BOOK_ENTRIES_RAW, 4000, min_value=100)


def _realtime_max_sport_results() -> int:
    return _int_or_default(POLYMARKET_REALTIME_MAX_SPORT_RESULTS_RAW, 2000, min_value=100)


def _realtime_shared_dir() -> str:
    return _normalize_text(POLYMARKET_REALTIME_SHARED_DIR)


def _realtime_owner_path() -> str:
    base_dir = _realtime_shared_dir()
    if not base_dir:
        return ""
    return os.path.join(base_dir, "owner.json")


def _realtime_snapshot_path() -> str:
    base_dir = _realtime_shared_dir()
    if not base_dir:
        return ""
    return os.path.join(base_dir, "snapshot.json")


def _api_base() -> str:
    base = (POLYMARKET_API_BASE or "").strip() or "https://gamma-api.polymarket.com"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _clob_base() -> str:
    base = (POLYMARKET_CLOB_BASE or "").strip() or "https://clob.polymarket.com"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _public_base() -> str:
    base = (POLYMARKET_PUBLIC_BASE or "").strip() or "https://polymarket.com"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if POLYMARKET_USER_AGENT:
        headers["User-Agent"] = POLYMARKET_USER_AGENT
    if POLYMARKET_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {POLYMARKET_BEARER_TOKEN}"
    if POLYMARKET_API_KEY:
        headers["X-API-Key"] = POLYMARKET_API_KEY
    if POLYMARKET_COOKIE:
        headers["Cookie"] = POLYMARKET_COOKIE
    return headers


def _request_json(
    path: str,
    params: Dict[str, object],
    retries: int,
    backoff_seconds: float,
) -> Tuple[object, int]:
    url = f"{_api_base()}/{path.lstrip('/')}"
    timeout = _int_or_default(POLYMARKET_TIMEOUT_RAW, 20, min_value=1)
    retriable_status = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    last_error: Optional[ProviderError] = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        except requests.RequestException as exc:
            last_error = ProviderError(f"Polymarket network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if response.status_code >= 400:
            if response.status_code in retriable_status and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise ProviderError(
                f"Polymarket API request failed ({response.status_code})",
                status_code=response.status_code,
            )
        try:
            return response.json(), attempt
        except ValueError as exc:
            last_error = ProviderError("Failed to parse Polymarket API response")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
    if last_error:
        raise last_error
    raise ProviderError("Polymarket request failed")


async def _request_json_async(
    client: httpx.AsyncClient,
    path: str,
    params: Dict[str, object],
    retries: int,
    backoff_seconds: float,
) -> Tuple[object, int]:
    url = f"{_api_base()}/{path.lstrip('/')}"
    timeout = _int_or_default(POLYMARKET_TIMEOUT_RAW, 20, min_value=1)
    return await request_json(
        client,
        "GET",
        url,
        params=params,
        headers=_headers(),
        timeout=float(timeout),
        retries=retries,
        backoff_seconds=backoff_seconds,
        error_cls=ProviderError,
        network_error_prefix="Polymarket network error",
        parse_error_message="Failed to parse Polymarket API response",
        status_error_message=lambda status_code: f"Polymarket API request failed ({status_code})",
    )


def _request_clob_json(
    path: str,
    params: Dict[str, object],
    retries: int,
    backoff_seconds: float,
) -> Tuple[object, int]:
    url = f"{_clob_base()}/{path.lstrip('/')}"
    timeout = _int_or_default(POLYMARKET_TIMEOUT_RAW, 20, min_value=1)
    retriable_status = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    last_error: Optional[ProviderError] = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        except requests.RequestException as exc:
            last_error = ProviderError(f"Polymarket CLOB network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if response.status_code >= 400:
            if response.status_code in retriable_status and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise ProviderError(
                f"Polymarket CLOB request failed ({response.status_code})",
                status_code=response.status_code,
            )
        try:
            return response.json(), attempt
        except ValueError as exc:
            last_error = ProviderError("Failed to parse Polymarket CLOB response")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
    if last_error:
        raise last_error
    raise ProviderError("Polymarket CLOB request failed")


async def _request_clob_json_async(
    client: httpx.AsyncClient,
    path: str,
    params: Dict[str, object],
    retries: int,
    backoff_seconds: float,
) -> Tuple[object, int]:
    url = f"{_clob_base()}/{path.lstrip('/')}"
    timeout = _int_or_default(POLYMARKET_TIMEOUT_RAW, 20, min_value=1)
    return await request_json(
        client,
        "GET",
        url,
        params=params,
        headers=_headers(),
        timeout=float(timeout),
        retries=retries,
        backoff_seconds=backoff_seconds,
        error_cls=ProviderError,
        network_error_prefix="Polymarket CLOB network error",
        parse_error_message="Failed to parse Polymarket CLOB response",
        status_error_message=lambda status_code: f"Polymarket CLOB request failed ({status_code})",
    )


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_token(value: object) -> str:
    text = _normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _normalize_market_key(value: object) -> str:
    text = _normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _requested_market_keys(markets: Sequence[str]) -> set[str]:
    requested = {_normalize_market_key(item) for item in (markets or []) if _normalize_market_key(item)}
    if "both_teams_to_score" in requested:
        requested.add("btts")
    if "btts" in requested:
        requested.add("both_teams_to_score")
    return requested


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_commence_time(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            timestamp = float(value)
            if timestamp > 1e12:
                timestamp /= 1000.0
            return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OSError, OverflowError):
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


def _parse_datetime_utc(value: object) -> Optional[dt.datetime]:
    normalized = _normalize_commence_time(value)
    if not normalized:
        return None
    try:
        parsed = dt.datetime.fromisoformat(normalized[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed.astimezone(dt.timezone.utc)


def _flag_is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = _normalize_text(value).lower()
    return text in {"1", "true", "yes", "y", "on"}


def _flag_is_false(value: object) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    text = _normalize_text(value).lower()
    return text in {"0", "false", "no", "n", "off"}


def _market_accepting_orders(market: object) -> bool:
    if not isinstance(market, dict):
        return False
    if market.get("acceptingOrders") is None:
        return False
    return _flag_is_true(market.get("acceptingOrders"))


def _event_has_tradeable_live_markets(event: object) -> bool:
    if not isinstance(event, dict):
        return False
    for market in (event.get("markets") or []):
        if not isinstance(market, dict):
            continue
        if _flag_is_true(market.get("closed")):
            continue
        if _flag_is_true(market.get("archived")):
            continue
        if _flag_is_true(market.get("ended")):
            continue
        if market.get("active") is not None and _flag_is_false(market.get("active")):
            continue
        if _market_accepting_orders(market):
            return True
    return False


def _event_is_tradeable(event: dict, now_utc: dt.datetime) -> bool:
    if not isinstance(event, dict):
        return False
    if _flag_is_true(event.get("closed")):
        return False
    if _flag_is_true(event.get("archived")):
        return False
    if event.get("active") is not None and _flag_is_false(event.get("active")):
        return False
    if _flag_is_true(event.get("ended")):
        return False
    end_at = _parse_datetime_utc(event.get("endDate"))
    if end_at is not None and end_at < now_utc and not _event_has_tradeable_live_markets(event):
        return False
    return True


def _market_is_tradeable(market: dict, now_utc: dt.datetime) -> bool:
    if not isinstance(market, dict):
        return False
    if _flag_is_true(market.get("closed")):
        return False
    if _flag_is_true(market.get("archived")):
        return False
    if market.get("active") is not None and _flag_is_false(market.get("active")):
        return False
    if _flag_is_true(market.get("ended")):
        return False
    if market.get("acceptingOrders") is not None and _flag_is_false(market.get("acceptingOrders")):
        return False
    end_at = _parse_datetime_utc(market.get("endDate"))
    if end_at is not None and end_at < now_utc and not _market_accepting_orders(market):
        return False
    return True


def _load_sport_tag_mapping(retries: int, backoff_seconds: float) -> Dict[str, set]:
    ttl = _int_or_default(POLYMARKET_SPORT_TAG_CACHE_TTL_RAW, 600, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(SPORT_TAG_CACHE.get("expires_at", 0.0))
    cached_mapping = SPORT_TAG_CACHE.get("mapping")
    if cache_valid and isinstance(cached_mapping, dict):
        return cached_mapping

    payload, _ = _request_json(
        "sports",
        {},
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    if not isinstance(payload, list):
        raise ProviderError("Polymarket sports endpoint must return a JSON array")
    raw_mapping: Dict[str, set] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        sport_token = _normalize_token(item.get("sport"))
        if not sport_token:
            continue
        tag_values = str(item.get("tags") or "")
        tag_ids = {
            part.strip()
            for part in tag_values.split(",")
            if part and part.strip().isdigit()
        }
        # Remove cross-cutting tags shared by nearly all sports markets.
        tag_ids.discard("1")
        tag_ids.discard("100639")
        bucket = raw_mapping.setdefault(sport_token, set())
        bucket.update(tag_ids)

    tag_frequency: Dict[str, int] = {}
    for tag_ids in raw_mapping.values():
        for tag_id in tag_ids:
            tag_frequency[tag_id] = tag_frequency.get(tag_id, 0) + 1

    mapping: Dict[str, set] = {}
    for sport_token, tag_ids in raw_mapping.items():
        specific = {tag_id for tag_id in tag_ids if tag_frequency.get(tag_id, 0) <= 3}
        mapping[sport_token] = specific or set(tag_ids)

    SPORT_TAG_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    SPORT_TAG_CACHE["mapping"] = mapping
    return mapping


async def _load_sport_tag_mapping_async(
    client: httpx.AsyncClient,
    retries: int,
    backoff_seconds: float,
) -> Dict[str, set]:
    ttl = _int_or_default(POLYMARKET_SPORT_TAG_CACHE_TTL_RAW, 600, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(SPORT_TAG_CACHE.get("expires_at", 0.0))
    cached_mapping = SPORT_TAG_CACHE.get("mapping")
    if cache_valid and isinstance(cached_mapping, dict):
        return cached_mapping

    async with _sport_tag_cache_async_lock():
        now = time.time()
        cache_valid = ttl > 0 and now < float(SPORT_TAG_CACHE.get("expires_at", 0.0))
        cached_mapping = SPORT_TAG_CACHE.get("mapping")
        if cache_valid and isinstance(cached_mapping, dict):
            return cached_mapping

        payload, _ = await _request_json_async(
            client,
            "sports",
            {},
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        if not isinstance(payload, list):
            raise ProviderError("Polymarket sports endpoint must return a JSON array")
        raw_mapping: Dict[str, set] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            sport_token = _normalize_token(item.get("sport"))
            if not sport_token:
                continue
            tag_values = str(item.get("tags") or "")
            tag_ids = {
                part.strip()
                for part in tag_values.split(",")
                if part and part.strip().isdigit()
            }
            tag_ids.discard("1")
            tag_ids.discard("100639")
            bucket = raw_mapping.setdefault(sport_token, set())
            bucket.update(tag_ids)

        tag_frequency: Dict[str, int] = {}
        for tag_ids in raw_mapping.values():
            for tag_id in tag_ids:
                tag_frequency[tag_id] = tag_frequency.get(tag_id, 0) + 1

        mapping: Dict[str, set] = {}
        for sport_token, tag_ids in raw_mapping.items():
            specific = {tag_id for tag_id in tag_ids if tag_frequency.get(tag_id, 0) <= 3}
            mapping[sport_token] = specific or set(tag_ids)

        SPORT_TAG_CACHE["expires_at"] = now + ttl if ttl > 0 else now
        SPORT_TAG_CACHE["mapping"] = mapping
        return mapping


def _event_tags(event: dict) -> Tuple[set, set]:
    tag_ids = set()
    tag_slugs = set()
    for tag in (event.get("tags") or []):
        if not isinstance(tag, dict):
            continue
        tag_id = _normalize_text(tag.get("id"))
        tag_slug = _normalize_token(tag.get("slug"))
        if tag_id:
            tag_ids.add(tag_id)
        if tag_slug:
            tag_slugs.add(tag_slug)
    return tag_ids, tag_slugs


def _event_is_sports(event: dict) -> bool:
    tag_ids, tag_slugs = _event_tags(event)
    return "1" in tag_ids or "sports" in tag_slugs


def _event_matches_sport(
    event: dict,
    sport_key: str,
    sport_tag_mapping: Dict[str, set],
) -> bool:
    if not sport_key:
        return True
    aliases = SPORT_ALIASES.get(sport_key)
    if not aliases:
        return False
    tag_ids, tag_slugs = _event_tags(event)
    alias_tokens = {_normalize_token(alias) for alias in aliases}
    if tag_slugs.intersection(alias_tokens):
        return True
    expected_tag_ids = set()
    for alias in alias_tokens:
        expected_tag_ids.update(sport_tag_mapping.get(alias, set()))
    if expected_tag_ids and tag_ids.intersection(expected_tag_ids):
        return True
    return False


def _clean_team_name(value: object) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    # Drop suffix markers commonly used in market titles.
    text = re.sub(r"\s*\(([A-Za-z]{1,5})\)\s*$", "", text).strip()
    text = re.sub(r"\s*[-–:]\s*more markets\s*$", "", text, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", text)


def _team_token(value: object) -> str:
    text = _clean_team_name(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_matchup_from_text(text: object) -> Optional[Tuple[str, str]]:
    raw = _normalize_text(text)
    if not raw:
        return None
    patterns = [
        r"^\s*(.+?)\s+vs\.?\s+(.+?)\s*$",
        r"^\s*(.+?)\s+v\.?\s+(.+?)\s*$",
        r"^\s*(.+?)\s+@\s+(.+?)\s*$",
        r"^\s*(.+?)\s+at\s+(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        home = _clean_team_name(match.group(1))
        away = _clean_team_name(match.group(2))
        if home and away and _team_token(home) != _team_token(away):
            return home, away
    return None


def _extract_matchup(event: dict) -> Optional[Tuple[str, str]]:
    direct = _extract_matchup_from_text(event.get("title"))
    if direct:
        return direct
    for market in (event.get("markets") or []):
        if not isinstance(market, dict):
            continue
        direct = _extract_matchup_from_text(market.get("question"))
        if direct:
            return direct
    return None


def _question_matches_matchup(question: object, home_team: str, away_team: str) -> bool:
    matchup = _extract_matchup_from_text(question)
    if not matchup:
        return False
    left_token = _team_token(matchup[0])
    right_token = _team_token(matchup[1])
    home_token = _team_token(home_team)
    away_token = _team_token(away_team)
    return {left_token, right_token} == {home_token, away_token}


def _parse_list(value) -> List[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []


def _parse_clob_token_ids(value: object) -> List[str]:
    token_ids: List[str] = []
    for item in _parse_list(value):
        token_id = _normalize_text(item)
        if token_id and token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids


def _match_market_clob_token_ids(
    market: dict,
    home_team: str,
    away_team: str,
    requested_markets: set[str],
) -> List[str]:
    if not isinstance(market, dict):
        return []

    home_token = _team_token(home_team)
    away_token = _team_token(away_team)
    if not home_token or not away_token:
        return []

    outcomes = [_normalize_text(item) for item in _parse_list(market.get("outcomes"))]
    prices = _parse_list(market.get("outcomePrices"))
    if len(outcomes) != 2 or len(prices) != 2:
        return []

    token_ids = _parse_clob_token_ids(market.get("clobTokenIds"))[:2]
    if len(token_ids) != 2:
        return []

    outcome_tokens = [_team_token(outcomes[0]), _team_token(outcomes[1])]
    if (
        "h2h" in requested_markets
        and set(outcome_tokens) == {home_token, away_token}
        and _question_matches_matchup(market.get("question"), home_team, away_team)
    ):
        return token_ids

    normalized_outcomes = [token.lower() for token in outcome_tokens]
    if normalized_outcomes not in (["yes", "no"], ["no", "yes"]):
        return []

    question = _normalize_text(market.get("question")).lower()
    if {"btts", "both_teams_to_score"} & requested_markets and (
        "both teams to score" in question or "btts" in question
    ):
        return token_ids

    if "draw" in question:
        return token_ids if "h2h_3_way" in requested_markets else []

    if {"h2h", "h2h_3_way"} & requested_markets:
        team_match = re.match(
            r"^\s*will\s+(.+?)\s+win(?:\s+on\s+\d{4}-\d{2}-\d{2})?\??\s*$",
            _normalize_text(market.get("question")),
            flags=re.IGNORECASE,
        )
        if team_match:
            team_token = _team_token(team_match.group(1))
            if team_token == home_token or team_token == away_token:
                return token_ids

    return []


def _event_match_clob_token_ids(
    event: dict,
    home_team: str,
    away_team: str,
    requested_markets: set[str],
    now_utc: dt.datetime,
) -> List[str]:
    token_ids: List[str] = []
    for market in (event.get("markets") or []):
        if not isinstance(market, dict):
            continue
        if not _market_is_tradeable(market, now_utc):
            continue
        token_ids.extend(
            _match_market_clob_token_ids(
                market,
                home_team,
                away_team,
                requested_markets,
            )
        )
    return list(dict.fromkeys(token_ids))


def _price_to_decimal_odds(price) -> Optional[float]:
    probability = _safe_float(price)
    if probability is None or probability <= 0 or probability >= 1:
        return None
    odds = 1.0 / probability
    if odds <= 1:
        return None
    return round(odds, 6)


def _book_ask_depth_notional(payload: object) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    asks = payload.get("asks")
    if not isinstance(asks, list):
        return None
    total = 0.0
    for level in asks:
        if not isinstance(level, dict):
            continue
        price = _safe_float(level.get("price"))
        size = _safe_float(level.get("size"))
        if price is None or size is None:
            continue
        if price <= 0 or size <= 0:
            continue
        total += float(price) * float(size)
    if total <= 0:
        return None
    return round(total, 6)


def _book_best_ask_quote(payload: object) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    asks = payload.get("asks")
    if not isinstance(asks, list):
        return None
    best_price = None
    best_size = None
    for level in asks:
        if not isinstance(level, dict):
            continue
        price = _safe_float(level.get("price"))
        size = _safe_float(level.get("size"))
        if price is None or size is None:
            continue
        if price <= 0 or price >= 1 or size <= 0:
            continue
        if best_price is None or float(price) < float(best_price):
            best_price = float(price)
            best_size = float(size)
    if best_price is None or best_size is None:
        return None
    decimal_odds = _price_to_decimal_odds(best_price)
    if decimal_odds is None:
        return None
    return {
        "probability": round(float(best_price), 6),
        "decimal_odds": decimal_odds,
        "stake": round(float(best_price) * float(best_size), 6),
        "quote_source": "clob_book_best_ask",
    }


def _price_level_map(levels: object) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not isinstance(levels, list):
        return mapping
    for level in levels:
        if not isinstance(level, dict):
            continue
        price = _safe_float(level.get("price"))
        size = _safe_float(level.get("size"))
        if price is None or size is None or price <= 0:
            continue
        mapping[f"{float(price):.10f}"] = max(0.0, float(size))
    return mapping


def _depth_from_level_map(levels: Dict[str, float]) -> Optional[float]:
    total = 0.0
    for price_key, size in (levels or {}).items():
        price = _safe_float(price_key)
        size_value = _safe_float(size)
        if price is None or size_value is None or price <= 0 or size_value <= 0:
            continue
        total += float(price) * float(size_value)
    if total <= 0:
        return None
    return round(total, 6)


def _best_ask_quote_from_level_map(levels: Dict[str, float]) -> Optional[dict]:
    best_price = None
    best_size = None
    for price_key, size in (levels or {}).items():
        price = _safe_float(price_key)
        size_value = _safe_float(size)
        if price is None or size_value is None:
            continue
        if price <= 0 or price >= 1 or size_value <= 0:
            continue
        if best_price is None or float(price) < float(best_price):
            best_price = float(price)
            best_size = float(size_value)
    if best_price is None or best_size is None:
        return None
    decimal_odds = _price_to_decimal_odds(best_price)
    if decimal_odds is None:
        return None
    return {
        "probability": round(float(best_price), 6),
        "decimal_odds": decimal_odds,
        "stake": round(float(best_price) * float(best_size), 6),
        "quote_source": "ws_book_best_ask",
    }


def _normalized_clob_token_ids(
    token_ids: Sequence[str],
) -> Tuple[List[str], int, int]:
    unique_token_ids = list(
        dict.fromkeys(_normalize_text(token_id) for token_id in token_ids if _normalize_text(token_id))
    )
    total_requested = len(unique_token_ids)
    max_books = _int_or_default(POLYMARKET_CLOB_MAX_BOOKS_RAW, 300, min_value=1)
    if len(unique_token_ids) > max_books:
        unique_token_ids = unique_token_ids[:max_books]
    truncated_count = max(0, total_requested - len(unique_token_ids))
    return unique_token_ids, total_requested, truncated_count


def _sports_result_slug(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("gameId", "metadataGameId", "slug", "marketSlug", "eventSlug", "gameSlug"):
        value = _normalize_text(payload.get(key))
        if value:
            return value
    return ""


def _sports_result_game_id(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("gameId", "metadataGameId"):
        value = _normalize_text(payload.get(key))
        if value:
            return value
    return ""


def _sports_result_is_tradeable(payload: object) -> bool:
    if not isinstance(payload, dict):
        return True
    if _flag_is_true(payload.get("ended")):
        return False
    if _flag_is_true(payload.get("closed")):
        return False
    if _flag_is_true(payload.get("archived")):
        return False
    status = _normalize_text(
        payload.get("status")
        or payload.get("gameStatus")
        or payload.get("matchStatus")
        or payload.get("state")
    ).lower()
    if status in {
        "final",
        "finished",
        "full_time",
        "full-time",
        "ended",
        "cancelled",
        "canceled",
        "postponed",
        "abandoned",
        "suspended_final",
    }:
        return False
    return True


def _event_sports_result_keys(event: object) -> List[str]:
    if not isinstance(event, dict):
        return []
    keys: List[str] = []
    for value in (
        event.get("gameId"),
        event.get("metadataGameId"),
        event.get("slug"),
    ):
        token = _normalize_text(value)
        if token and token not in keys:
            keys.append(token)
    return keys


def _event_identity(event: object) -> str:
    if not isinstance(event, dict):
        return ""
    for value in (
        event.get("id"),
        event.get("gameId"),
        event.get("metadataGameId"),
        event.get("slug"),
        event.get("ticker"),
    ):
        token = _normalize_text(value)
        if token:
            return token
    return ""


def _merge_events(primary: Sequence[dict], supplemental: Sequence[dict]) -> List[dict]:
    merged: List[dict] = []
    seen: set[str] = set()
    for event in list(primary or []) + list(supplemental or []):
        if not isinstance(event, dict):
            continue
        identity = _event_identity(event)
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        merged.append(event)
    return merged


def _event_live_state_payload(event: object, realtime_state: object = None) -> Optional[dict]:
    payload: Dict[str, object] = {}
    if isinstance(realtime_state, dict):
        realtime_event_state = realtime_state.get("eventState") if isinstance(realtime_state.get("eventState"), dict) else {}
        status = _normalize_text(
            realtime_state.get("status")
            or realtime_state.get("gameStatus")
            or realtime_state.get("matchStatus")
            or realtime_state.get("state")
            or realtime_event_state.get("status")
            or realtime_event_state.get("gameStatus")
            or realtime_event_state.get("matchStatus")
            or realtime_event_state.get("state")
        ).lower()
        live_flag = _flag_is_true(realtime_state.get("live")) or _flag_is_true(realtime_event_state.get("live"))
        ended_flag = _flag_is_true(realtime_state.get("ended")) or _flag_is_true(realtime_event_state.get("ended"))
        closed_flag = _flag_is_true(realtime_state.get("closed")) or _flag_is_true(realtime_event_state.get("closed"))
        if live_flag and status in {"", "inprogress", "in_progress"}:
            status = "live"
        elif (ended_flag or closed_flag) and not status:
            status = "final"
        if status:
            payload["status"] = status
        updated_at = _normalize_text(
            realtime_state.get("updated_at")
            or realtime_state.get("updatedAt")
            or realtime_state.get("timestamp")
            or realtime_state.get("ts")
            or realtime_event_state.get("updated_at")
            or realtime_event_state.get("updatedAt")
            or realtime_event_state.get("timestamp")
            or realtime_event_state.get("ts")
            or realtime_event_state.get("createdAt")
        )
        if updated_at:
            payload["updated_at"] = updated_at
        for source in (realtime_state, realtime_event_state):
            for key in ("score", "homeScore", "awayScore", "period", "clock", "elapsed"):
                value = source.get(key)
                if value not in (None, ""):
                    payload[key] = value
        if live_flag:
            payload["is_live"] = True
        if "ended" in realtime_state or "ended" in realtime_event_state:
            payload["ended"] = ended_flag
        if "closed" in realtime_state or "closed" in realtime_event_state:
            payload["closed"] = closed_flag
    if not payload and isinstance(event, dict):
        status = _normalize_text(
            event.get("status")
            or event.get("gameStatus")
            or event.get("matchStatus")
            or event.get("state")
        ).lower()
        if status:
            payload["status"] = status
        updated_at = _normalize_text(
            event.get("updatedAt")
            or event.get("updated_at")
            or event.get("lastUpdated")
            or event.get("startTime")
        )
        if updated_at:
            payload["updated_at"] = updated_at
    return payload or None


def _read_json_file(path: object) -> Optional[dict]:
    target = _normalize_text(path)
    if not target or not os.path.exists(target):
        return None
    try:
        with open(target, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_file_atomic(path: object, payload: dict) -> None:
    target = _normalize_text(path)
    if not target or not isinstance(payload, dict):
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temp_path = f"{target}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temp_path, target)


def _remove_file_quietly(path: object) -> None:
    target = _normalize_text(path)
    if not target:
        return
    try:
        os.remove(target)
    except OSError:
        return


def _timestamp_to_iso(value: object) -> str:
    timestamp = _safe_float(value)
    if timestamp is None or timestamp <= 0:
        return ""
    try:
        return (
            dt.datetime.fromtimestamp(float(timestamp), tz=dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OSError, OverflowError, ValueError):
        return ""


def _shared_snapshot_is_fresh(age_seconds: object) -> bool:
    age = _safe_float(age_seconds)
    return age is not None and age <= _realtime_owner_stale_seconds()


def _realtime_channel_is_ready(status: object, channel: str) -> bool:
    if not isinstance(status, dict):
        return False
    if bool(status.get("shared_snapshot_stale")):
        return False
    if not bool(status.get(f"{channel}_connected")):
        return False
    age = _safe_float(status.get(f"{channel}_last_message_age_seconds"))
    if age is None:
        return True
    max_age = _ws_book_max_age_seconds() if channel == "market" else _realtime_owner_stale_seconds()
    return age <= max_age


def _sport_result_payload_is_fresh(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    updated_at = _safe_float(payload.get("updated_at")) or 0.0
    if updated_at <= 0:
        return True
    return (time.time() - updated_at) <= _realtime_sport_result_ttl_seconds()


class PolymarketRealtimeManager:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_ready = threading.Event()
        self._state_lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._owner_id = f"{os.getpid()}-{int(time.time() * 1000)}"
        self._owner_active = False
        self._market_ws = None
        self._sports_ws = None
        self._market_connected = False
        self._sports_connected = False
        self._market_books: Dict[str, dict] = {}
        self._sports_results: Dict[str, dict] = {}
        self._subscribed_asset_ids: Dict[str, float] = {}
        self._market_subscription_initialized = False
        self._messages_received = {"market": 0, "sports": 0}
        self._connect_attempts = {"market": 0, "sports": 0}
        self._last_errors = {"market": "", "sports": ""}
        self._last_message_at = {"market": 0.0, "sports": 0.0}
        self._last_owner_heartbeat_at = 0.0
        self._last_snapshot_write_at = 0.0
        self._last_cleanup_at = 0.0
        self._shared_snapshot_cache = {
            "loaded_at": 0.0,
            "mtime": 0.0,
            "payload": {},
        }
        self._started = False

    def ensure_started(self) -> bool:
        if not _websocket_realtime_enabled():
            return False
        with self._state_lock:
            if self._started and self._thread and self._thread.is_alive():
                return True
        if not self._try_acquire_owner():
            with self._state_lock:
                self._owner_active = False
            return False
        with self._state_lock:
            self._stop_requested.clear()
            self._loop_ready.clear()
            self._owner_active = True
            self._thread = threading.Thread(
                target=self._run_loop,
                name="polymarket-ws",
                daemon=True,
            )
            self._thread.start()
            self._started = True
        self._loop_ready.wait(timeout=2.0)
        return self._loop is not None

    def wait_until_ready(self, timeout_seconds: float) -> bool:
        if timeout_seconds <= 0:
            return False
        deadline = time.time() + float(timeout_seconds)
        while time.time() < deadline:
            snapshot = self.snapshot()
            if _market_websocket_enabled() and snapshot.get("market_connected"):
                return True
            if _sports_websocket_enabled() and snapshot.get("sports_connected"):
                return True
            time.sleep(0.05)
        return False

    def stop(self, timeout_seconds: float = 2.0) -> None:
        with self._state_lock:
            thread = self._thread
        if not thread:
            with self._state_lock:
                self._started = False
                self._loop = None
                self._owner_active = False
            self._release_owner()
            return
        self._stop_requested.set()
        if thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_seconds)))
        with self._state_lock:
            self._started = False
            self._thread = None
            self._loop = None
            self._market_ws = None
            self._sports_ws = None
            self._market_connected = False
            self._sports_connected = False
            self._market_subscription_initialized = False
            self._owner_active = False
        self._release_owner()

    def _try_acquire_owner(self) -> bool:
        owner_path = _realtime_owner_path()
        if not owner_path:
            return False
        now = time.time()
        stale_after = _realtime_owner_stale_seconds()
        payload = {
            "owner_id": self._owner_id,
            "pid": os.getpid(),
            "heartbeat_at": now,
            "saved_at": _timestamp_to_iso(now),
        }
        os.makedirs(os.path.dirname(owner_path), exist_ok=True)

        def _create_owner_file() -> bool:
            try:
                fd = os.open(owner_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return False
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            except OSError:
                return False
            return True

        if _create_owner_file():
            return True
        current = _read_json_file(owner_path) or {}
        current_owner = _normalize_text(current.get("owner_id"))
        if current_owner == self._owner_id:
            self._refresh_owner_heartbeat(force=True)
            return True
        heartbeat_at = _safe_float(current.get("heartbeat_at")) or 0.0
        if heartbeat_at > 0 and (now - heartbeat_at) <= stale_after:
            return False
        _remove_file_quietly(owner_path)
        return _create_owner_file()

    def _release_owner(self) -> None:
        owner_path = _realtime_owner_path()
        current = _read_json_file(owner_path) or {}
        if _normalize_text(current.get("owner_id")) != self._owner_id:
            return
        _remove_file_quietly(owner_path)

    def _refresh_owner_heartbeat(self, force: bool = False) -> None:
        if not self._owner_active:
            return
        now = time.time()
        if not force and (now - self._last_owner_heartbeat_at) < min(5.0, _realtime_owner_stale_seconds() / 2.0):
            return
        owner_path = _realtime_owner_path()
        if not owner_path:
            return
        payload = {
            "owner_id": self._owner_id,
            "pid": os.getpid(),
            "heartbeat_at": now,
            "saved_at": _timestamp_to_iso(now),
        }
        try:
            _write_json_file_atomic(owner_path, payload)
        except OSError:
            return
        self._last_owner_heartbeat_at = now

    def _load_shared_snapshot(self, force: bool = False) -> dict:
        snapshot_path = _realtime_snapshot_path()
        if not snapshot_path or not os.path.exists(snapshot_path):
            return {}
        try:
            mtime = os.path.getmtime(snapshot_path)
        except OSError:
            return {}
        now = time.time()
        cached = self._shared_snapshot_cache
        if (
            not force
            and float(cached.get("loaded_at") or 0.0) > 0
            and (now - float(cached.get("loaded_at") or 0.0)) <= _realtime_shared_read_ttl_seconds()
            and float(cached.get("mtime") or 0.0) == float(mtime)
        ):
            payload = cached.get("payload")
            return payload if isinstance(payload, dict) else {}
        payload = _read_json_file(snapshot_path) or {}
        self._shared_snapshot_cache = {
            "loaded_at": now,
            "mtime": float(mtime),
            "payload": payload,
        }
        return payload

    def _local_status_payload(self) -> dict:
        now = time.time()
        book_ages = [
            max(0.0, now - float(_safe_float(entry.get("updated_at")) or 0.0))
            for entry in self._market_books.values()
            if isinstance(entry, dict) and (_safe_float(entry.get("updated_at")) or 0.0) > 0
        ]
        result_ages = [
            max(0.0, now - float(_safe_float(entry.get("updated_at")) or 0.0))
            for entry in self._sports_results.values()
            if isinstance(entry, dict) and (_safe_float(entry.get("updated_at")) or 0.0) > 0
        ]
        return {
            "started": self._started,
            "owner_active": self._owner_active,
            "owner_id": self._owner_id,
            "pid": os.getpid(),
            "market_connected": self._market_connected,
            "sports_connected": self._sports_connected,
            "subscribed_assets": len(self._subscribed_asset_ids),
            "market_books_cached": len(self._market_books),
            "sports_results_cached": len(self._sports_results),
            "market_messages_received": int(self._messages_received.get("market", 0) or 0),
            "sports_messages_received": int(self._messages_received.get("sports", 0) or 0),
            "market_connect_attempts": int(self._connect_attempts.get("market", 0) or 0),
            "sports_connect_attempts": int(self._connect_attempts.get("sports", 0) or 0),
            "market_last_error": _normalize_text(self._last_errors.get("market")),
            "sports_last_error": _normalize_text(self._last_errors.get("sports")),
            "market_last_message_at": _timestamp_to_iso(self._last_message_at.get("market")),
            "sports_last_message_at": _timestamp_to_iso(self._last_message_at.get("sports")),
            "market_last_message_age_seconds": round(max(0.0, now - float(self._last_message_at.get("market") or 0.0)), 2)
            if float(self._last_message_at.get("market") or 0.0) > 0
            else None,
            "sports_last_message_age_seconds": round(max(0.0, now - float(self._last_message_at.get("sports") or 0.0)), 2)
            if float(self._last_message_at.get("sports") or 0.0) > 0
            else None,
            "market_oldest_book_age_seconds": round(max(book_ages), 2) if book_ages else None,
            "sports_oldest_result_age_seconds": round(max(result_ages), 2) if result_ages else None,
            "snapshot_saved_at": _timestamp_to_iso(self._last_snapshot_write_at),
            "cleanup_ran_at": _timestamp_to_iso(self._last_cleanup_at),
        }

    def _write_shared_snapshot(self, force: bool = False) -> None:
        if not self._owner_active:
            return
        now = time.time()
        if not force and (now - self._last_snapshot_write_at) < _realtime_snapshot_flush_seconds():
            return
        snapshot_path = _realtime_snapshot_path()
        if not snapshot_path:
            return
        with self._state_lock:
            payload = {
                "saved_at": _timestamp_to_iso(now),
                "status": self._local_status_payload(),
                "market_books": {
                    asset_id: {
                        "asks": dict(entry.get("asks") or {}),
                        "updated_at": entry.get("updated_at"),
                    }
                    for asset_id, entry in self._market_books.items()
                    if isinstance(entry, dict)
                },
                "sports_results": {
                    slug: dict(entry)
                    for slug, entry in self._sports_results.items()
                    if isinstance(entry, dict)
                },
            }
        _write_json_file_atomic(snapshot_path, payload)
        self._last_snapshot_write_at = now

    def subscribe_assets(self, asset_ids: Sequence[str]) -> int:
        normalized = [
            token_id
            for token_id in (_normalize_text(item) for item in asset_ids or [])
            if token_id
        ]
        if not normalized:
            return 0
        if not self._owner_active:
            self.ensure_started()
        if not self._owner_active:
            return 0
        now = time.time()
        with self._state_lock:
            new_ids = [token_id for token_id in normalized if token_id not in self._subscribed_asset_ids]
            for token_id in normalized:
                self._subscribed_asset_ids[token_id] = now
            loop = self._loop
        if loop is not None and _market_websocket_enabled():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_market_subscription(new_ids),
                    loop,
                )
            except Exception:
                pass
        return len(new_ids)

    def wait_for_assets(self, asset_ids: Sequence[str], timeout_seconds: float) -> bool:
        if timeout_seconds <= 0:
            return False
        deadline = time.time() + float(timeout_seconds)
        token_ids = [token_id for token_id in (_normalize_text(item) for item in asset_ids or []) if token_id]
        if not token_ids:
            return False
        while time.time() < deadline:
            snapshot = self.get_depth_map(token_ids, max_age_seconds=_ws_book_max_age_seconds())
            if snapshot:
                return True
            time.sleep(0.05)
        return False

    def wait_for_quotes(self, asset_ids: Sequence[str], timeout_seconds: float) -> bool:
        if timeout_seconds <= 0:
            return False
        deadline = time.time() + float(timeout_seconds)
        token_ids = [token_id for token_id in (_normalize_text(item) for item in asset_ids or []) if token_id]
        if not token_ids:
            return False
        while time.time() < deadline:
            snapshot = self.get_quote_map(token_ids, max_age_seconds=_ws_book_max_age_seconds())
            if snapshot:
                return True
            time.sleep(0.05)
        return False

    def get_depth_map(
        self,
        asset_ids: Sequence[str],
        max_age_seconds: float,
    ) -> Dict[str, Optional[float]]:
        now = time.time()
        depths: Dict[str, Optional[float]] = {}
        missing: List[str] = []
        with self._state_lock:
            for asset_id in (_normalize_text(item) for item in asset_ids or []):
                if not asset_id:
                    continue
                if self._owner_active:
                    self._subscribed_asset_ids[asset_id] = now
                entry = self._market_books.get(asset_id)
                if not isinstance(entry, dict):
                    missing.append(asset_id)
                    continue
                updated_at = _safe_float(entry.get("updated_at")) or 0.0
                if max_age_seconds > 0 and updated_at > 0 and (now - updated_at) > max_age_seconds:
                    missing.append(asset_id)
                    continue
                ask_levels = entry.get("asks")
                if not isinstance(ask_levels, dict):
                    missing.append(asset_id)
                    continue
                depth = _depth_from_level_map(ask_levels)
                if depth is not None:
                    depths[asset_id] = depth
                else:
                    missing.append(asset_id)
        if missing:
            shared_snapshot = self._load_shared_snapshot()
            shared_books = shared_snapshot.get("market_books") if isinstance(shared_snapshot, dict) else {}
            if isinstance(shared_books, dict):
                for asset_id in missing:
                    entry = shared_books.get(asset_id)
                    if not isinstance(entry, dict):
                        continue
                    updated_at = _safe_float(entry.get("updated_at")) or 0.0
                    if max_age_seconds > 0 and updated_at > 0 and (now - updated_at) > max_age_seconds:
                        continue
                    ask_levels = entry.get("asks")
                    if not isinstance(ask_levels, dict):
                        continue
                    depth = _depth_from_level_map(ask_levels)
                    if depth is not None:
                        depths[asset_id] = depth
        return depths

    def get_quote_map(
        self,
        asset_ids: Sequence[str],
        max_age_seconds: float,
    ) -> Dict[str, dict]:
        now = time.time()
        quotes: Dict[str, dict] = {}
        missing: List[str] = []
        with self._state_lock:
            for asset_id in (_normalize_text(item) for item in asset_ids or []):
                if not asset_id:
                    continue
                if self._owner_active:
                    self._subscribed_asset_ids[asset_id] = now
                entry = self._market_books.get(asset_id)
                if not isinstance(entry, dict):
                    missing.append(asset_id)
                    continue
                updated_at = _safe_float(entry.get("updated_at")) or 0.0
                if max_age_seconds > 0 and updated_at > 0 and (now - updated_at) > max_age_seconds:
                    missing.append(asset_id)
                    continue
                ask_levels = entry.get("asks")
                if not isinstance(ask_levels, dict):
                    missing.append(asset_id)
                    continue
                quote = _best_ask_quote_from_level_map(ask_levels)
                if quote is not None:
                    quotes[asset_id] = dict(quote)
                else:
                    missing.append(asset_id)
        if missing:
            shared_snapshot = self._load_shared_snapshot()
            shared_books = shared_snapshot.get("market_books") if isinstance(shared_snapshot, dict) else {}
            if isinstance(shared_books, dict):
                for asset_id in missing:
                    entry = shared_books.get(asset_id)
                    if not isinstance(entry, dict):
                        continue
                    updated_at = _safe_float(entry.get("updated_at")) or 0.0
                    if max_age_seconds > 0 and updated_at > 0 and (now - updated_at) > max_age_seconds:
                        continue
                    ask_levels = entry.get("asks")
                    if not isinstance(ask_levels, dict):
                        continue
                    quote = _best_ask_quote_from_level_map(ask_levels)
                    if quote is not None:
                        quotes[asset_id] = dict(quote)
        return quotes

    def get_sport_result(self, slug: object) -> Optional[dict]:
        token = _normalize_text(slug)
        if not token:
            return None
        with self._state_lock:
            payload = self._sports_results.get(token)
            if _sport_result_payload_is_fresh(payload):
                return dict(payload)
        shared_snapshot = self._load_shared_snapshot()
        shared_results = shared_snapshot.get("sports_results") if isinstance(shared_snapshot, dict) else {}
        if not isinstance(shared_results, dict):
            return None
        payload = shared_results.get(token)
        return dict(payload) if _sport_result_payload_is_fresh(payload) else None

    def get_sport_results(self, max_age_seconds: float = 0.0) -> Dict[str, dict]:
        now = time.time()
        results: Dict[str, dict] = {}

        def _merge_entries(entries: object) -> None:
            if not isinstance(entries, dict):
                return
            for key, payload in entries.items():
                token = _normalize_text(key)
                if not token or not isinstance(payload, dict):
                    continue
                if not _sport_result_payload_is_fresh(payload):
                    continue
                updated_at = _safe_float(payload.get("updated_at")) or 0.0
                if max_age_seconds > 0 and updated_at > 0 and (now - updated_at) > max_age_seconds:
                    continue
                existing = results.get(token)
                existing_updated_at = _safe_float((existing or {}).get("updated_at")) or 0.0
                if existing is not None and existing_updated_at >= updated_at:
                    continue
                results[token] = dict(payload)

        with self._state_lock:
            _merge_entries(self._sports_results)
        shared_snapshot = self._load_shared_snapshot()
        shared_results = shared_snapshot.get("sports_results") if isinstance(shared_snapshot, dict) else {}
        _merge_entries(shared_results)
        return results

    def snapshot(self) -> dict:
        with self._state_lock:
            status = self._local_status_payload()
        if status.get("owner_active") or status.get("started"):
            return status
        shared_snapshot = self._load_shared_snapshot()
        shared_status = shared_snapshot.get("status") if isinstance(shared_snapshot, dict) else {}
        if isinstance(shared_status, dict) and shared_status:
            saved_at = _normalize_text(shared_snapshot.get("saved_at"))
            saved_dt = _parse_datetime_utc(saved_at)
            age_seconds = None
            if saved_dt is not None:
                age_seconds = round(
                    max(0.0, (dt.datetime.now(dt.timezone.utc) - saved_dt).total_seconds()),
                    2,
                )
            snapshot_fresh = _shared_snapshot_is_fresh(age_seconds)
            status.update(
                {
                    "shared_snapshot_loaded": True,
                    "shared_snapshot_saved_at": saved_at,
                    "shared_snapshot_age_seconds": age_seconds,
                    "shared_snapshot_stale": not snapshot_fresh,
                    "shared_owner_active": snapshot_fresh,
                }
            )
            status.update(shared_status)
            if not snapshot_fresh:
                status["started"] = False
                status["owner_active"] = False
                status["market_connected"] = False
                status["sports_connected"] = False
            return status
        return status

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._state_lock:
            self._loop = loop
        self._loop_ready.set()
        try:
            loop.run_until_complete(self._run_forever())
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
            with self._state_lock:
                self._loop = None
                self._market_ws = None
                self._sports_ws = None
                self._market_connected = False
                self._sports_connected = False
                self._started = False
            self._release_owner()

    async def _run_forever(self) -> None:
        tasks = []
        if _market_websocket_enabled():
            tasks.append(asyncio.create_task(self._market_loop()))
        if _sports_websocket_enabled():
            tasks.append(asyncio.create_task(self._sports_loop()))
        tasks.append(asyncio.create_task(self._maintenance_loop()))
        try:
            await self._wait_for_stop()
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _wait_for_stop(self) -> None:
        while not self._stop_requested.is_set():
            await asyncio.sleep(0.2)

    async def _market_loop(self) -> None:
        from websockets.asyncio.client import connect

        delay = 1.0
        while not self._stop_requested.is_set():
            try:
                with self._state_lock:
                    self._connect_attempts["market"] = int(self._connect_attempts.get("market", 0) or 0) + 1
                async with connect(
                    POLYMARKET_MARKET_WS_URL,
                    open_timeout=_ws_connect_timeout_seconds(),
                    ping_interval=20.0,
                    ping_timeout=20.0,
                    close_timeout=5.0,
                    max_queue=1000,
                ) as websocket:
                    with self._state_lock:
                        self._market_ws = websocket
                        self._market_connected = True
                        self._market_subscription_initialized = False
                        self._last_errors["market"] = ""
                        subscribed = [
                            asset_id
                            for asset_id, last_requested_at in self._subscribed_asset_ids.items()
                            if (time.time() - float(last_requested_at or 0.0)) <= _realtime_subscription_ttl_seconds()
                        ]
                    if subscribed:
                        await self._send_market_initial_subscription(subscribed)
                    heartbeat_task = asyncio.create_task(self._market_heartbeat(websocket))
                    delay = 1.0
                    try:
                        while not self._stop_requested.is_set():
                            raw = await websocket.recv()
                            await self._handle_market_message(raw)
                    finally:
                        heartbeat_task.cancel()
                        await asyncio.gather(heartbeat_task, return_exceptions=True)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                with self._state_lock:
                    self._market_connected = False
                    self._market_ws = None
                    self._last_errors["market"] = str(exc)
                if self._stop_requested.is_set():
                    break
                await asyncio.sleep(delay)
                delay = min(_ws_reconnect_max_delay_seconds(), delay * 2.0)

    async def _sports_loop(self) -> None:
        from websockets.asyncio.client import connect

        delay = 1.0
        while not self._stop_requested.is_set():
            try:
                with self._state_lock:
                    self._connect_attempts["sports"] = int(self._connect_attempts.get("sports", 0) or 0) + 1
                async with connect(
                    POLYMARKET_SPORTS_WS_URL,
                    open_timeout=_ws_connect_timeout_seconds(),
                    ping_interval=20.0,
                    ping_timeout=20.0,
                    close_timeout=5.0,
                    max_queue=1000,
                ) as websocket:
                    with self._state_lock:
                        self._sports_ws = websocket
                        self._sports_connected = True
                        self._last_errors["sports"] = ""
                    delay = 1.0
                    while not self._stop_requested.is_set():
                        raw = await websocket.recv()
                        if isinstance(raw, str) and raw.strip().lower() == "ping":
                            await websocket.send("pong")
                            continue
                        await self._handle_sports_message(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                with self._state_lock:
                    self._sports_connected = False
                    self._sports_ws = None
                    self._last_errors["sports"] = str(exc)
                if self._stop_requested.is_set():
                    break
                await asyncio.sleep(delay)
                delay = min(_ws_reconnect_max_delay_seconds(), delay * 2.0)

    async def _wait_for_market_subscriptions(self) -> List[str]:
        while not self._stop_requested.is_set():
            with self._state_lock:
                now = time.time()
                subscribed = [
                    asset_id
                    for asset_id, last_requested_at in self._subscribed_asset_ids.items()
                    if (now - float(last_requested_at or 0.0)) <= _realtime_subscription_ttl_seconds()
                ]
            if subscribed:
                return subscribed
            await asyncio.sleep(0.2)
        return []

    async def _market_heartbeat(self, websocket) -> None:
        while not self._stop_requested.is_set():
            await asyncio.sleep(10.0)
            await websocket.send("PING")

    async def _maintenance_loop(self) -> None:
        while not self._stop_requested.is_set():
            await asyncio.sleep(1.0)
            with self._state_lock:
                owner_active = self._owner_active
            if not owner_active:
                continue
            self._refresh_owner_heartbeat()
            now = time.time()
            if (now - self._last_cleanup_at) >= _realtime_cleanup_seconds():
                await self._prune_realtime_state()
            self._write_shared_snapshot()

    async def _prune_realtime_state(self) -> None:
        now = time.time()
        unsubscribe_ids: List[str] = []
        with self._state_lock:
            expired_assets = [
                asset_id
                for asset_id, last_requested_at in self._subscribed_asset_ids.items()
                if (now - float(last_requested_at or 0.0)) > _realtime_subscription_ttl_seconds()
            ]
            for asset_id in expired_assets:
                self._subscribed_asset_ids.pop(asset_id, None)
            unsubscribe_ids.extend(expired_assets)

            expired_books = [
                asset_id
                for asset_id, entry in self._market_books.items()
                if (now - float(_safe_float(entry.get("updated_at")) or 0.0)) > _realtime_subscription_ttl_seconds()
            ]
            for asset_id in expired_books:
                self._market_books.pop(asset_id, None)

            expired_results = [
                slug
                for slug, entry in self._sports_results.items()
                if (now - float(_safe_float(entry.get("updated_at")) or 0.0)) > _realtime_sport_result_ttl_seconds()
            ]
            for slug in expired_results:
                self._sports_results.pop(slug, None)

            if len(self._market_books) > _realtime_max_book_entries():
                overflow = len(self._market_books) - _realtime_max_book_entries()
                ordered = sorted(
                    self._market_books.items(),
                    key=lambda item: _safe_float((item[1] or {}).get("updated_at")) or 0.0,
                )
                for asset_id, _ in ordered[:overflow]:
                    self._market_books.pop(asset_id, None)
                    self._subscribed_asset_ids.pop(asset_id, None)
                    unsubscribe_ids.append(asset_id)
            if len(self._sports_results) > _realtime_max_sport_results():
                overflow = len(self._sports_results) - _realtime_max_sport_results()
                ordered = sorted(
                    self._sports_results.items(),
                    key=lambda item: _safe_float((item[1] or {}).get("updated_at")) or 0.0,
                )
                for slug, _ in ordered[:overflow]:
                    self._sports_results.pop(slug, None)
            self._last_cleanup_at = now
        unsubscribe_ids = list(dict.fromkeys(_normalize_text(item) for item in unsubscribe_ids if _normalize_text(item)))
        if unsubscribe_ids:
            await self._send_market_unsubscribe(unsubscribe_ids)

    async def _send_market_initial_subscription(self, asset_ids: Sequence[str]) -> None:
        if not asset_ids:
            return
        with self._state_lock:
            websocket = self._market_ws
            connected = self._market_connected
        if websocket is None or not connected:
            return
        payload = {
            "assets_ids": [
                token_id
                for token_id in (_normalize_text(item) for item in asset_ids)
                if token_id
            ],
            "type": "market",
            "custom_feature_enabled": True,
        }
        if not payload["assets_ids"]:
            return
        await websocket.send(json.dumps(payload, separators=(",", ":")))
        with self._state_lock:
            self._market_subscription_initialized = True

    async def _send_market_subscription(self, asset_ids: Sequence[str]) -> None:
        if not asset_ids:
            return
        with self._state_lock:
            websocket = self._market_ws
            connected = self._market_connected
            initialized = self._market_subscription_initialized
        if websocket is None or not connected:
            return
        if not initialized:
            await self._send_market_initial_subscription(asset_ids)
            return
        payload = {
            "assets_ids": [
                token_id
                for token_id in (_normalize_text(item) for item in asset_ids)
                if token_id
            ],
            "operation": "subscribe",
            "custom_feature_enabled": True,
        }
        if not payload["assets_ids"]:
            return
        await websocket.send(json.dumps(payload, separators=(",", ":")))

    async def _send_market_unsubscribe(self, asset_ids: Sequence[str]) -> None:
        if not asset_ids:
            return
        with self._state_lock:
            websocket = self._market_ws
            connected = self._market_connected
        if websocket is None or not connected:
            return
        payload = {
            "assets_ids": [
                token_id
                for token_id in (_normalize_text(item) for item in asset_ids)
                if token_id
            ],
            "operation": "unsubscribe",
        }
        if not payload["assets_ids"]:
            return
        await websocket.send(json.dumps(payload, separators=(",", ":")))

    async def _handle_market_message(self, raw: object) -> None:
        messages = self._decode_ws_messages(raw)
        if not messages:
            return
        now = time.time()
        with self._state_lock:
            self._messages_received["market"] = int(self._messages_received.get("market", 0) or 0) + len(messages)
            self._last_message_at["market"] = now
            for payload in messages:
                if not isinstance(payload, dict):
                    continue
                event_type = _normalize_text(payload.get("event_type") or payload.get("type")).lower()
                asset_id = _normalize_text(
                    payload.get("asset_id")
                    or payload.get("assetId")
                    or payload.get("market")
                    or payload.get("token_id")
                    or payload.get("tokenId")
                )
                if not asset_id:
                    continue
                entry = self._market_books.setdefault(
                    asset_id,
                    {"bids": {}, "asks": {}, "updated_at": 0.0},
                )
                if event_type == "book":
                    entry["bids"] = _price_level_map(payload.get("bids"))
                    entry["asks"] = _price_level_map(payload.get("asks"))
                elif event_type == "price_change":
                    changes = payload.get("price_changes") if isinstance(payload.get("price_changes"), list) else []
                    for change in changes:
                        if not isinstance(change, dict):
                            continue
                        price = _safe_float(change.get("price"))
                        size = _safe_float(change.get("size"))
                        side = _normalize_text(change.get("side")).lower()
                        if price is None:
                            continue
                        bucket = entry["asks"] if side == "sell" else entry["bids"] if side == "buy" else None
                        if bucket is None:
                            continue
                        price_key = f"{float(price):.10f}"
                        if size is None or size <= 0:
                            bucket.pop(price_key, None)
                        else:
                            bucket[price_key] = float(size)
                entry["updated_at"] = now

    async def _handle_sports_message(self, raw: object) -> None:
        messages = self._decode_ws_messages(raw)
        if not messages:
            return
        now = time.time()
        with self._state_lock:
            self._messages_received["sports"] = int(self._messages_received.get("sports", 0) or 0) + len(messages)
            self._last_message_at["sports"] = now
            for payload in messages:
                slug = _sports_result_slug(payload)
                if not slug:
                    continue
                copied = dict(payload)
                copied["updated_at"] = now
                self._sports_results[slug] = copied

    def _decode_ws_messages(self, raw: object) -> List[dict]:
        payload = raw
        if isinstance(raw, bytes):
            try:
                payload = raw.decode("utf-8")
            except UnicodeDecodeError:
                return []
        if isinstance(payload, str):
            text = payload.strip()
            if not text or text.lower() in {"ping", "pong"}:
                return []
            try:
                payload = json.loads(text)
            except ValueError:
                return []
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []


def _get_realtime_manager() -> PolymarketRealtimeManager:
    global REALTIME_MANAGER
    with REALTIME_MANAGER_LOCK:
        if REALTIME_MANAGER is None:
            REALTIME_MANAGER = PolymarketRealtimeManager()
        return REALTIME_MANAGER


def ensure_realtime_started(wait_timeout: Optional[float] = None) -> dict:
    if not _websocket_realtime_enabled():
        return {
            "enabled": False,
            "started": False,
            "ready": False,
            "status": {},
        }
    manager = _get_realtime_manager()
    started = manager.ensure_started()
    wait_seconds = _ws_startup_wait_seconds() if wait_timeout is None else max(0.0, float(wait_timeout))
    ready = manager.wait_until_ready(wait_seconds) if wait_seconds > 0 else False
    status = manager.snapshot()
    return {
        "enabled": True,
        "started": started,
        "ready": ready,
        "status": status,
    }


def stop_realtime(timeout_seconds: float = 2.0) -> None:
    manager = _get_realtime_manager()
    manager.stop(timeout_seconds=timeout_seconds)


def realtime_status() -> dict:
    if not _websocket_realtime_enabled():
        return {
            "enabled": False,
            "started": False,
            "ready": False,
            "status": {},
        }
    manager = _get_realtime_manager()
    status = manager.snapshot()
    ready = bool(
        (_market_websocket_enabled() and _realtime_channel_is_ready(status, "market"))
        or (_sports_websocket_enabled() and _realtime_channel_is_ready(status, "sports"))
    )
    return {
        "enabled": True,
        "started": bool(status.get("started") or status.get("owner_active") or status.get("shared_owner_active")),
        "ready": ready,
        "status": status,
    }


def _apply_realtime_stats(stats: dict, snapshot: Optional[dict]) -> None:
    if not isinstance(stats, dict) or not isinstance(snapshot, dict):
        return
    stats["realtime_market_connected"] = bool(snapshot.get("market_connected"))
    stats["realtime_sports_connected"] = bool(snapshot.get("sports_connected"))
    stats["realtime_owner_active"] = bool(snapshot.get("owner_active"))
    stats["realtime_owner_id"] = _normalize_text(snapshot.get("owner_id"))
    stats["realtime_market_subscribed_assets"] = int(snapshot.get("subscribed_assets", 0) or 0)
    stats["realtime_market_messages_received"] = int(snapshot.get("market_messages_received", 0) or 0)
    stats["realtime_sports_messages_received"] = int(snapshot.get("sports_messages_received", 0) or 0)
    stats["realtime_market_books_cached"] = int(snapshot.get("market_books_cached", 0) or 0)
    stats["realtime_sports_results_cached"] = int(snapshot.get("sports_results_cached", 0) or 0)
    stats["realtime_market_last_message_age_seconds"] = snapshot.get("market_last_message_age_seconds")
    stats["realtime_sports_last_message_age_seconds"] = snapshot.get("sports_last_message_age_seconds")
    stats["realtime_shared_snapshot_loaded"] = bool(snapshot.get("shared_snapshot_loaded"))
    stats["realtime_shared_snapshot_age_seconds"] = snapshot.get("shared_snapshot_age_seconds")


def _load_clob_depth_map(
    token_ids: Sequence[str],
    retries: int,
    backoff_seconds: float,
) -> Tuple[Dict[str, Optional[float]], dict]:
    unique_token_ids = list(dict.fromkeys(_normalize_text(token_id) for token_id in token_ids if _normalize_text(token_id)))
    total_requested = len(unique_token_ids)
    max_books = _int_or_default(POLYMARKET_CLOB_MAX_BOOKS_RAW, 300, min_value=1)
    if len(unique_token_ids) > max_books:
        unique_token_ids = unique_token_ids[:max_books]
    truncated_count = max(0, total_requested - len(unique_token_ids))

    ttl = _int_or_default(POLYMARKET_CLOB_BOOK_CACHE_TTL_RAW, 4, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(CLOB_BOOK_CACHE.get("expires_at", 0.0))
    cache_entries = CLOB_BOOK_CACHE.get("entries") if isinstance(CLOB_BOOK_CACHE.get("entries"), dict) else {}
    scan_cache_entries: Dict[str, object] = {}
    if _scan_cache_active():
        with SCAN_CACHE_LOCK:
            cached_scan_entries = SCAN_CACHE_CONTEXT.get("clob_entries")
            if isinstance(cached_scan_entries, dict):
                scan_cache_entries = cached_scan_entries
            else:
                scan_cache_entries = {}
                SCAN_CACHE_CONTEXT["clob_entries"] = scan_cache_entries

    depth_by_token: Dict[str, Optional[float]] = {}
    unresolved: List[str] = []
    for token_id in unique_token_ids:
        if token_id in scan_cache_entries:
            depth = _safe_float(scan_cache_entries.get(token_id))
            depth_by_token[token_id] = round(float(depth), 6) if depth is not None and depth > 0 else None
        elif cache_valid and token_id in cache_entries:
            depth = _safe_float(cache_entries.get(token_id))
            depth_by_token[token_id] = round(float(depth), 6) if depth is not None and depth > 0 else None
        else:
            unresolved.append(token_id)

    retries_used = 0
    books_fetched = 0
    book_errors = 0
    if unresolved:
        max_workers = _int_or_default(POLYMARKET_CLOB_BOOK_WORKERS_RAW, 8, min_value=1)
        worker_count = min(max_workers, len(unresolved))

        def _fetch_single_book(token_id: str) -> Tuple[str, Optional[float], int, bool]:
            try:
                payload, retry_count = _request_clob_json(
                    "book",
                    {"token_id": token_id},
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                )
            except ProviderError:
                return token_id, None, 0, True
            return token_id, _book_ask_depth_notional(payload), retry_count, False

        if worker_count <= 1:
            for token_id in unresolved:
                resolved_token_id, depth, retry_count, had_error = _fetch_single_book(token_id)
                retries_used += retry_count
                if had_error:
                    depth_by_token[resolved_token_id] = None
                    cache_entries[resolved_token_id] = None
                    scan_cache_entries[resolved_token_id] = None
                    book_errors += 1
                    continue
                books_fetched += 1
                depth_by_token[resolved_token_id] = depth
                cache_entries[resolved_token_id] = depth
                scan_cache_entries[resolved_token_id] = depth
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_fetch_single_book, token_id) for token_id in unresolved]
                for future in concurrent.futures.as_completed(futures):
                    resolved_token_id, depth, retry_count, had_error = future.result()
                    retries_used += retry_count
                    if had_error:
                        depth_by_token[resolved_token_id] = None
                        cache_entries[resolved_token_id] = None
                        scan_cache_entries[resolved_token_id] = None
                        book_errors += 1
                        continue
                    books_fetched += 1
                    depth_by_token[resolved_token_id] = depth
                    cache_entries[resolved_token_id] = depth
                    scan_cache_entries[resolved_token_id] = depth

    CLOB_BOOK_CACHE["entries"] = cache_entries
    CLOB_BOOK_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    if _scan_cache_active():
        with SCAN_CACHE_LOCK:
            SCAN_CACHE_CONTEXT["clob_entries"] = scan_cache_entries

    books_with_depth = sum(
        1
        for token_id in unique_token_ids
        if token_id in depth_by_token and _safe_float(depth_by_token[token_id]) is not None
    )
    return depth_by_token, {
        "token_count_requested": total_requested,
        "token_count_considered": len(unique_token_ids),
        "token_count_truncated": truncated_count,
        "books_fetched": books_fetched,
        "books_with_depth": books_with_depth,
        "book_errors": book_errors,
        "retries_used": retries_used,
    }


async def _load_clob_depth_map_async(
    client: httpx.AsyncClient,
    token_ids: Sequence[str],
    retries: int,
    backoff_seconds: float,
) -> Tuple[Dict[str, Optional[float]], dict]:
    unique_token_ids = list(
        dict.fromkeys(_normalize_text(token_id) for token_id in token_ids if _normalize_text(token_id))
    )
    total_requested = len(unique_token_ids)
    max_books = _int_or_default(POLYMARKET_CLOB_MAX_BOOKS_RAW, 300, min_value=1)
    if len(unique_token_ids) > max_books:
        unique_token_ids = unique_token_ids[:max_books]
    truncated_count = max(0, total_requested - len(unique_token_ids))

    ttl = _int_or_default(POLYMARKET_CLOB_BOOK_CACHE_TTL_RAW, 4, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(CLOB_BOOK_CACHE.get("expires_at", 0.0))
    cache_entries = CLOB_BOOK_CACHE.get("entries") if isinstance(CLOB_BOOK_CACHE.get("entries"), dict) else {}
    scan_cache_entries: Dict[str, object] = {}
    if _scan_cache_active():
        with SCAN_CACHE_LOCK:
            cached_scan_entries = SCAN_CACHE_CONTEXT.get("clob_entries")
            if isinstance(cached_scan_entries, dict):
                scan_cache_entries = cached_scan_entries
            else:
                scan_cache_entries = {}
                SCAN_CACHE_CONTEXT["clob_entries"] = scan_cache_entries

    depth_by_token: Dict[str, Optional[float]] = {}
    unresolved: List[str] = []
    for token_id in unique_token_ids:
        if token_id in scan_cache_entries:
            depth = _safe_float(scan_cache_entries.get(token_id))
            depth_by_token[token_id] = round(float(depth), 6) if depth is not None and depth > 0 else None
        elif cache_valid and token_id in cache_entries:
            depth = _safe_float(cache_entries.get(token_id))
            depth_by_token[token_id] = round(float(depth), 6) if depth is not None and depth > 0 else None
        else:
            unresolved.append(token_id)

    retries_used = 0
    books_fetched = 0
    book_errors = 0
    if unresolved:
        max_workers = _int_or_default(POLYMARKET_CLOB_BOOK_WORKERS_RAW, 8, min_value=1)
        worker_count = min(max_workers, len(unresolved))

        async def _fetch_single_book(token_id: str) -> Tuple[str, Optional[float], int, bool]:
            try:
                payload, retry_count = await _request_clob_json_async(
                    client,
                    "book",
                    {"token_id": token_id},
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                )
            except ProviderError:
                return token_id, None, 0, True
            return token_id, _book_ask_depth_notional(payload), retry_count, False

        if worker_count <= 1:
            for token_id in unresolved:
                resolved_token_id, depth, retry_count, had_error = await _fetch_single_book(token_id)
                retries_used += retry_count
                if had_error:
                    depth_by_token[resolved_token_id] = None
                    cache_entries[resolved_token_id] = None
                    scan_cache_entries[resolved_token_id] = None
                    book_errors += 1
                    continue
                books_fetched += 1
                depth_by_token[resolved_token_id] = depth
                cache_entries[resolved_token_id] = depth
                scan_cache_entries[resolved_token_id] = depth
        else:
            semaphore = asyncio.Semaphore(worker_count)

            async def _limited_fetch(token_id: str) -> Tuple[str, Optional[float], int, bool]:
                async with semaphore:
                    return await _fetch_single_book(token_id)

            tasks = [asyncio.create_task(_limited_fetch(token_id)) for token_id in unresolved]
            for task in asyncio.as_completed(tasks):
                resolved_token_id, depth, retry_count, had_error = await task
                retries_used += retry_count
                if had_error:
                    depth_by_token[resolved_token_id] = None
                    cache_entries[resolved_token_id] = None
                    scan_cache_entries[resolved_token_id] = None
                    book_errors += 1
                    continue
                books_fetched += 1
                depth_by_token[resolved_token_id] = depth
                cache_entries[resolved_token_id] = depth
                scan_cache_entries[resolved_token_id] = depth

    CLOB_BOOK_CACHE["entries"] = cache_entries
    CLOB_BOOK_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    if _scan_cache_active():
        with SCAN_CACHE_LOCK:
            SCAN_CACHE_CONTEXT["clob_entries"] = scan_cache_entries

    books_with_depth = sum(
        1
        for token_id in unique_token_ids
        if token_id in depth_by_token and _safe_float(depth_by_token[token_id]) is not None
    )
    return depth_by_token, {
        "token_count_requested": total_requested,
        "token_count_considered": len(unique_token_ids),
        "token_count_truncated": truncated_count,
        "books_fetched": books_fetched,
        "books_with_depth": books_with_depth,
        "book_errors": book_errors,
        "retries_used": retries_used,
    }


async def _load_clob_quote_map_async(
    client: httpx.AsyncClient,
    token_ids: Sequence[str],
    retries: int,
    backoff_seconds: float,
) -> Tuple[Dict[str, dict], dict]:
    unique_token_ids = list(
        dict.fromkeys(_normalize_text(token_id) for token_id in token_ids if _normalize_text(token_id))
    )
    total_requested = len(unique_token_ids)
    max_books = _int_or_default(POLYMARKET_CLOB_MAX_BOOKS_RAW, 300, min_value=1)
    if len(unique_token_ids) > max_books:
        unique_token_ids = unique_token_ids[:max_books]
    truncated_count = max(0, total_requested - len(unique_token_ids))

    ttl = _int_or_default(POLYMARKET_CLOB_BOOK_CACHE_TTL_RAW, 4, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(CLOB_QUOTE_CACHE.get("expires_at", 0.0))
    cache_entries = CLOB_QUOTE_CACHE.get("entries") if isinstance(CLOB_QUOTE_CACHE.get("entries"), dict) else {}
    scan_cache_entries: Dict[str, object] = {}
    if _scan_cache_active():
        with SCAN_CACHE_LOCK:
            cached_scan_entries = SCAN_CACHE_CONTEXT.get("clob_quotes")
            if isinstance(cached_scan_entries, dict):
                scan_cache_entries = cached_scan_entries
            else:
                scan_cache_entries = {}
                SCAN_CACHE_CONTEXT["clob_quotes"] = scan_cache_entries

    quote_by_token: Dict[str, dict] = {}
    unresolved: List[str] = []
    for token_id in unique_token_ids:
        cached_quote = None
        if token_id in scan_cache_entries:
            cached_quote = scan_cache_entries.get(token_id)
        elif cache_valid and token_id in cache_entries:
            cached_quote = cache_entries.get(token_id)
        if isinstance(cached_quote, dict) and _safe_float(cached_quote.get("decimal_odds")):
            quote_by_token[token_id] = dict(cached_quote)
        else:
            unresolved.append(token_id)

    retries_used = 0
    books_fetched = 0
    book_errors = 0
    if unresolved:
        max_workers = _int_or_default(POLYMARKET_CLOB_BOOK_WORKERS_RAW, 8, min_value=1)
        worker_count = min(max_workers, len(unresolved))

        async def _fetch_single_book(token_id: str) -> Tuple[str, Optional[dict], int, bool]:
            try:
                payload, retry_count = await _request_clob_json_async(
                    client,
                    "book",
                    {"token_id": token_id},
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                )
            except ProviderError:
                return token_id, None, 0, True
            return token_id, _book_best_ask_quote(payload), retry_count, False

        if worker_count <= 1:
            for token_id in unresolved:
                resolved_token_id, quote, retry_count, had_error = await _fetch_single_book(token_id)
                retries_used += retry_count
                if had_error:
                    cache_entries[resolved_token_id] = None
                    scan_cache_entries[resolved_token_id] = None
                    book_errors += 1
                    continue
                books_fetched += 1
                if isinstance(quote, dict):
                    quote_by_token[resolved_token_id] = dict(quote)
                    cache_entries[resolved_token_id] = dict(quote)
                    scan_cache_entries[resolved_token_id] = dict(quote)
                else:
                    cache_entries[resolved_token_id] = None
                    scan_cache_entries[resolved_token_id] = None
        else:
            semaphore = asyncio.Semaphore(worker_count)

            async def _limited_fetch(token_id: str) -> Tuple[str, Optional[dict], int, bool]:
                async with semaphore:
                    return await _fetch_single_book(token_id)

            tasks = [asyncio.create_task(_limited_fetch(token_id)) for token_id in unresolved]
            for task in asyncio.as_completed(tasks):
                resolved_token_id, quote, retry_count, had_error = await task
                retries_used += retry_count
                if had_error:
                    cache_entries[resolved_token_id] = None
                    scan_cache_entries[resolved_token_id] = None
                    book_errors += 1
                    continue
                books_fetched += 1
                if isinstance(quote, dict):
                    quote_by_token[resolved_token_id] = dict(quote)
                    cache_entries[resolved_token_id] = dict(quote)
                    scan_cache_entries[resolved_token_id] = dict(quote)
                else:
                    cache_entries[resolved_token_id] = None
                    scan_cache_entries[resolved_token_id] = None

    CLOB_QUOTE_CACHE["entries"] = cache_entries
    CLOB_QUOTE_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    if _scan_cache_active():
        with SCAN_CACHE_LOCK:
            SCAN_CACHE_CONTEXT["clob_quotes"] = scan_cache_entries

    books_with_quotes = sum(
        1
        for token_id in unique_token_ids
        if token_id in quote_by_token and _safe_float(quote_by_token[token_id].get("decimal_odds")) is not None
    )
    return quote_by_token, {
        "token_count_requested": total_requested,
        "token_count_considered": len(unique_token_ids),
        "token_count_truncated": truncated_count,
        "books_fetched": books_fetched,
        "books_with_quotes": books_with_quotes,
        "book_errors": book_errors,
        "retries_used": retries_used,
    }


def _market_outcome_row(
    name: str,
    price: float,
    stake: Optional[float] = None,
    raw_percentage_odds: object = None,
    quote_source: object = None,
) -> dict:
    row = {"name": name, "price": round(float(price), 6)}
    if stake is not None and stake > 0:
        row["stake"] = round(float(stake), 6)
    if raw_percentage_odds not in (None, ""):
        row["raw_percentage_odds"] = raw_percentage_odds
    source = _normalize_text(quote_source)
    if source:
        row["quote_source"] = source
    return row


def _pick_match_markets(
    event: dict,
    home_team: str,
    away_team: str,
    requested_markets: set[str],
    now_utc: dt.datetime,
    clob_quote_map: Optional[Dict[str, dict]] = None,
) -> List[dict]:
    home_token = _team_token(home_team)
    away_token = _team_token(away_team)
    if not home_token or not away_token:
        return []

    direct_candidates: List[dict] = []
    yes_by_team: Dict[str, dict] = {}
    draw_yes: Optional[dict] = None
    btts_yes: Optional[dict] = None
    btts_no: Optional[dict] = None
    has_draw_prompt = False

    for market in (event.get("markets") or []):
        if not isinstance(market, dict):
            continue
        if not _market_is_tradeable(market, now_utc):
            continue
        outcomes = [_normalize_text(item) for item in _parse_list(market.get("outcomes"))]
        prices = _parse_list(market.get("outcomePrices"))
        if len(outcomes) != 2 or len(prices) != 2:
            continue

        token_ids = _parse_clob_token_ids(market.get("clobTokenIds"))
        quote_details: List[dict] = []
        for idx in range(2):
            raw_probability = prices[idx]
            quote_payload = (
                clob_quote_map.get(token_ids[idx])
                if isinstance(clob_quote_map, dict) and idx < len(token_ids) and isinstance(clob_quote_map.get(token_ids[idx]), dict)
                else None
            )
            decimal_odds = _safe_float(quote_payload.get("decimal_odds")) if isinstance(quote_payload, dict) else None
            stake = _safe_float(quote_payload.get("stake")) if isinstance(quote_payload, dict) else None
            quote_source = _normalize_text(quote_payload.get("quote_source")) if isinstance(quote_payload, dict) else ""
            if decimal_odds is None or decimal_odds <= 1:
                decimal_odds = _price_to_decimal_odds(raw_probability)
                stake = None
                quote_source = "gamma_outcome_price"
            if decimal_odds is None:
                quote_details = []
                break
            quote_details.append(
                {
                    "odds": decimal_odds,
                    "stake": round(float(stake), 6) if stake is not None and stake > 0 else None,
                    "raw_percentage_odds": raw_probability,
                    "quote_source": quote_source,
                }
            )
        if len(quote_details) != 2:
            continue

        outcome_tokens = [_team_token(outcomes[0]), _team_token(outcomes[1])]
        question = _normalize_text(market.get("question")).lower()

        if "draw" in question:
            has_draw_prompt = True

        if (
            set(outcome_tokens) == {home_token, away_token}
            and _question_matches_matchup(market.get("question"), home_team, away_team)
        ):
            direct_candidates.append(
                {
                    "outcomes": [
                        _market_outcome_row(
                            outcomes[0],
                            quote_details[0]["odds"],
                            quote_details[0]["stake"],
                            quote_details[0]["raw_percentage_odds"],
                            quote_details[0]["quote_source"],
                        ),
                        _market_outcome_row(
                            outcomes[1],
                            quote_details[1]["odds"],
                            quote_details[1]["stake"],
                            quote_details[1]["raw_percentage_odds"],
                            quote_details[1]["quote_source"],
                        ),
                    ],
                    "volume": _safe_float(market.get("volumeNum") or market.get("volume")) or 0.0,
                }
            )
            continue

        normalized_outcomes = [token.lower() for token in outcome_tokens]
        if normalized_outcomes == ["yes", "no"] or normalized_outcomes == ["no", "yes"]:
            yes_index = 0 if normalized_outcomes[0] == "yes" else 1
            yes_quote = quote_details[yes_index]
            no_quote = quote_details[1 - yes_index]
            if "both teams to score" in question or "btts" in question:
                btts_yes = dict(yes_quote)
                btts_no = dict(no_quote)
                continue
            if "draw" in question:
                draw_yes = dict(yes_quote)
                continue
            team_match = re.match(
                r"^\s*will\s+(.+?)\s+win(?:\s+on\s+\d{4}-\d{2}-\d{2})?\??\s*$",
                _normalize_text(market.get("question")),
                flags=re.IGNORECASE,
            )
            if team_match:
                team_token = _team_token(team_match.group(1))
                if team_token == home_token:
                    yes_by_team[home_token] = dict(yes_quote)
                elif team_token == away_token:
                    yes_by_team[away_token] = dict(yes_quote)

    collected: List[dict] = []
    if direct_candidates:
        best = max(direct_candidates, key=lambda item: item.get("volume", 0.0))
        outcomes = best["outcomes"]
        # Normalize outcome order to home/away.
        ordered = {_team_token(item["name"]): item for item in outcomes}
        home_outcome = ordered.get(home_token)
        away_outcome = ordered.get(away_token)
        if "h2h" in requested_markets:
            if home_outcome and away_outcome:
                collected.append(
                    {
                        "key": "h2h",
                        "outcomes": [
                            _market_outcome_row(
                                home_team,
                                home_outcome["price"],
                                _safe_float(home_outcome.get("stake")),
                                home_outcome.get("raw_percentage_odds"),
                                home_outcome.get("quote_source"),
                            ),
                            _market_outcome_row(
                                away_team,
                                away_outcome["price"],
                                _safe_float(away_outcome.get("stake")),
                                away_outcome.get("raw_percentage_odds"),
                                away_outcome.get("quote_source"),
                            ),
                        ],
                    }
                )
            else:
                collected.append(
                    {
                        "key": "h2h",
                        "outcomes": outcomes,
                    }
                )
        return collected

    if home_token in yes_by_team and away_token in yes_by_team:
        home_data = yes_by_team[home_token]
        away_data = yes_by_team[away_token]
        if "h2h" in requested_markets and not has_draw_prompt:
            collected.append(
                {
                    "key": "h2h",
                        "outcomes": [
                            _market_outcome_row(
                                home_team,
                                home_data["odds"],
                                _safe_float(home_data.get("stake")),
                                home_data.get("raw_percentage_odds"),
                                home_data.get("quote_source"),
                            ),
                            _market_outcome_row(
                                away_team,
                                away_data["odds"],
                                _safe_float(away_data.get("stake")),
                                away_data.get("raw_percentage_odds"),
                                away_data.get("quote_source"),
                            ),
                        ],
                    }
                )
        if "h2h_3_way" in requested_markets and isinstance(draw_yes, dict):
            collected.append(
                {
                    "key": "h2h_3_way",
                        "outcomes": [
                            _market_outcome_row(
                                home_team,
                                home_data["odds"],
                                _safe_float(home_data.get("stake")),
                                home_data.get("raw_percentage_odds"),
                                home_data.get("quote_source"),
                            ),
                            _market_outcome_row(
                                "Draw",
                                draw_yes["odds"],
                                _safe_float(draw_yes.get("stake")),
                                draw_yes.get("raw_percentage_odds"),
                                draw_yes.get("quote_source"),
                            ),
                            _market_outcome_row(
                                away_team,
                                away_data["odds"],
                                _safe_float(away_data.get("stake")),
                                away_data.get("raw_percentage_odds"),
                                away_data.get("quote_source"),
                            ),
                        ],
                    }
                )

    if (
        {"btts", "both_teams_to_score"} & requested_markets
        and isinstance(btts_yes, dict)
        and isinstance(btts_no, dict)
    ):
        market_key = "both_teams_to_score" if "both_teams_to_score" in requested_markets else "btts"
        collected.append(
            {
                "key": market_key,
                "outcomes": [
                    _market_outcome_row(
                        "Yes",
                        btts_yes["odds"],
                        _safe_float(btts_yes.get("stake")),
                        btts_yes.get("raw_percentage_odds"),
                        btts_yes.get("quote_source"),
                    ),
                    _market_outcome_row(
                        "No",
                        btts_no["odds"],
                        _safe_float(btts_no.get("stake")),
                        btts_no.get("raw_percentage_odds"),
                        btts_no.get("quote_source"),
                    ),
                ],
            }
        )
    return collected


def _event_url(event: dict) -> str:
    slug = _normalize_text(event.get("slug"))
    if slug:
        return f"{_public_base()}/event/{quote(slug, safe='')}"
    event_id = _normalize_text(event.get("id"))
    if event_id:
        return f"{_public_base()}/event/{quote(event_id, safe='')}"
    return ""


def _load_active_game_events(
    retries: int,
    backoff_seconds: float,
    page_size: int,
    max_pages: int,
) -> Tuple[List[dict], dict]:
    if _scan_cache_active():
        with SCAN_CACHE_LOCK:
            cached_events = SCAN_CACHE_CONTEXT.get("events")
            cached_meta = dict(SCAN_CACHE_CONTEXT.get("events_meta") or {})
        if isinstance(cached_events, list):
            return cached_events, {**cached_meta, "pages_fetched": 0, "retries_used": 0, "cache": "scan_hit"}
    ttl = _int_or_default(POLYMARKET_EVENTS_CACHE_TTL_RAW, 12, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(EVENTS_CACHE.get("expires_at", 0.0))
    cached_events = EVENTS_CACHE.get("events")
    if cache_valid and isinstance(cached_events, list):
        return cached_events, {"pages_fetched": 0, "retries_used": 0, "cache": "hit"}

    all_events: List[dict] = []
    total_retries = 0
    pages_fetched = 0
    offset = 0
    now_iso = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    for _page in range(max_pages):
        payload, retries_used = _request_json(
            "events",
            {
                "tag_id": POLYMARKET_GAME_TAG_ID or "100639",
                "active": "true",
                "closed": "false",
                "archived": "false",
                "end_date_min": now_iso,
                "order": "id",
                "ascending": "false",
                "limit": page_size,
                "offset": offset,
            },
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        pages_fetched += 1
        total_retries += retries_used
        if not isinstance(payload, list):
            raise ProviderError("Polymarket events endpoint must return a JSON array")
        if not payload:
            break
        all_events.extend(item for item in payload if isinstance(item, dict))
        if len(payload) < page_size:
            break
        offset += page_size

    EVENTS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    EVENTS_CACHE["events"] = all_events
    meta = {"pages_fetched": pages_fetched, "retries_used": total_retries, "cache": "miss"}
    if _scan_cache_active():
        with SCAN_CACHE_LOCK:
            SCAN_CACHE_CONTEXT["events"] = list(all_events)
            SCAN_CACHE_CONTEXT["events_meta"] = dict(meta)
    return all_events, meta


async def _load_active_game_events_async(
    client: httpx.AsyncClient,
    retries: int,
    backoff_seconds: float,
    page_size: int,
    max_pages: int,
) -> Tuple[List[dict], dict]:
    if _scan_cache_active():
        with SCAN_CACHE_LOCK:
            cached_events = SCAN_CACHE_CONTEXT.get("events")
            cached_meta = dict(SCAN_CACHE_CONTEXT.get("events_meta") or {})
        if isinstance(cached_events, list):
            return cached_events, {**cached_meta, "pages_fetched": 0, "retries_used": 0, "cache": "scan_hit"}
    ttl = _int_or_default(POLYMARKET_EVENTS_CACHE_TTL_RAW, 12, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(EVENTS_CACHE.get("expires_at", 0.0))
    cached_events = EVENTS_CACHE.get("events")
    if cache_valid and isinstance(cached_events, list):
        return cached_events, {"pages_fetched": 0, "retries_used": 0, "cache": "hit"}

    async with _scan_cache_async_lock():
        if _scan_cache_active():
            with SCAN_CACHE_LOCK:
                cached_events = SCAN_CACHE_CONTEXT.get("events")
                cached_meta = dict(SCAN_CACHE_CONTEXT.get("events_meta") or {})
            if isinstance(cached_events, list):
                return cached_events, {**cached_meta, "pages_fetched": 0, "retries_used": 0, "cache": "scan_hit"}
    async with _events_cache_async_lock():
        now = time.time()
        cache_valid = ttl > 0 and now < float(EVENTS_CACHE.get("expires_at", 0.0))
        cached_events = EVENTS_CACHE.get("events")
        if cache_valid and isinstance(cached_events, list):
            return cached_events, {"pages_fetched": 0, "retries_used": 0, "cache": "hit"}

        all_events: List[dict] = []
        total_retries = 0
        pages_fetched = 0
        offset = 0
        now_iso = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        for _page in range(max_pages):
            payload, retries_used = await _request_json_async(
                client,
                "events",
                {
                    "tag_id": POLYMARKET_GAME_TAG_ID or "100639",
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "end_date_min": now_iso,
                    "order": "id",
                    "ascending": "false",
                    "limit": page_size,
                    "offset": offset,
                },
                retries=retries,
                backoff_seconds=backoff_seconds,
            )
            pages_fetched += 1
            total_retries += retries_used
            if not isinstance(payload, list):
                raise ProviderError("Polymarket events endpoint must return a JSON array")
            if not payload:
                break
            all_events.extend(item for item in payload if isinstance(item, dict))
            if len(payload) < page_size:
                break
            offset += page_size

        EVENTS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
        EVENTS_CACHE["events"] = all_events
        meta = {
            "pages_fetched": pages_fetched,
            "retries_used": total_retries,
            "cache": "miss",
        }
        if _scan_cache_active():
            with SCAN_CACHE_LOCK:
                SCAN_CACHE_CONTEXT["events"] = list(all_events)
                SCAN_CACHE_CONTEXT["events_meta"] = dict(meta)
        return all_events, meta


async def _load_game_events_by_ids_async(
    client: httpx.AsyncClient,
    game_ids: Sequence[object],
    existing_events: Optional[Sequence[dict]],
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[dict], dict]:
    known_game_ids = {
        _normalize_text(event.get("gameId"))
        for event in (existing_events or [])
        if isinstance(event, dict) and _normalize_text(event.get("gameId"))
    }
    unique_game_ids: List[str] = []
    for item in game_ids or []:
        token = _normalize_text(item)
        if not token or token in known_game_ids or token in unique_game_ids:
            continue
        unique_game_ids.append(token)
    if not unique_game_ids:
        return [], {"game_ids_requested": 0, "lookups": 0, "events_added": 0, "retries_used": 0, "lookup_errors": 0}

    semaphore = asyncio.Semaphore(min(8, len(unique_game_ids)))

    async def _fetch_one(game_id: str) -> Tuple[List[dict], int, int]:
        async with semaphore:
            try:
                payload, retries_used = await _request_json_async(
                    client,
                    "events",
                    {
                        "game_id": game_id,
                        "tag_id": POLYMARKET_GAME_TAG_ID or "100639",
                        "active": "true",
                        "closed": "false",
                        "archived": "false",
                        "limit": 20,
                    },
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                )
            except Exception:
                return [], 0, 1
            if not isinstance(payload, list):
                return [], 0, 1
            return [item for item in payload if isinstance(item, dict)], retries_used, 0

    fetched = await asyncio.gather(*(_fetch_one(game_id) for game_id in unique_game_ids))
    retries_used = 0
    lookup_errors = 0
    supplemental_events: List[dict] = []
    for events, retries_for_request, request_errors in fetched:
        retries_used += int(retries_for_request or 0)
        lookup_errors += int(request_errors or 0)
        supplemental_events.extend(events)
    merged = _merge_events([], supplemental_events)
    return merged, {
        "game_ids_requested": len(unique_game_ids),
        "lookups": len(unique_game_ids),
        "events_added": len(merged),
        "retries_used": retries_used,
        "lookup_errors": lookup_errors,
    }


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
    _ = regions  # Reserved for future region-specific routing.
    _ = context
    stats = {
        "provider": PROVIDER_KEY,
        "source": POLYMARKET_SOURCE or "api",
        "skipped_unsupported_sport": False,
        "events_payload_count": 0,
        "events_sports_count": 0,
        "events_sport_filtered_count": 0,
        "events_status_filtered_count": 0,
        "events_matchup_count": 0,
        "events_with_market_count": 0,
        "events_returned_count": 0,
        "clob_tokens_requested": 0,
        "clob_tokens_considered": 0,
        "clob_tokens_truncated": 0,
        "clob_http_fallback_enabled": True,
        "clob_http_depth_enabled": _http_clob_depth_enabled(),
        "clob_http_fallback_skipped": 0,
        "clob_books_fetched": 0,
        "clob_books_with_depth": 0,
        "clob_books_with_quotes": 0,
        "clob_book_errors": 0,
        "pages_fetched": 0,
        "retries_used": 0,
        "payload_cache": "miss",
        "realtime_ws_enabled": _websocket_realtime_enabled(),
        "realtime_market_connected": False,
        "realtime_sports_connected": False,
        "realtime_market_books_hit": 0,
        "realtime_market_books_missed": 0,
        "realtime_market_books_pending": 0,
        "realtime_market_new_subscriptions": 0,
        "realtime_market_subscribed_assets": 0,
        "realtime_market_messages_received": 0,
        "realtime_market_books_cached": 0,
        "realtime_market_last_message_age_seconds": None,
        "realtime_sports_messages_received": 0,
        "realtime_sports_results_cached": 0,
        "realtime_sports_last_message_age_seconds": None,
        "realtime_sports_game_ids_observed": 0,
        "realtime_sports_event_lookups": 0,
        "realtime_sports_events_supplemented": 0,
        "realtime_sports_event_lookup_errors": 0,
        "realtime_sports_state_hits": 0,
        "realtime_sports_state_filtered_count": 0,
        "realtime_owner_active": False,
        "realtime_owner_id": "",
        "realtime_shared_snapshot_loaded": False,
        "realtime_shared_snapshot_age_seconds": None,
    }
    _set_last_stats(stats)

    supported_markets = _requested_market_keys(markets)
    if not ({"h2h", "h2h_3_way", "btts", "both_teams_to_score"} & supported_markets):
        return []

    if bookmakers:
        lowered = {str(book).strip().lower() for book in bookmakers if isinstance(book, str)}
        if PROVIDER_KEY not in lowered and PROVIDER_TITLE.lower() not in lowered:
            return []

    if (POLYMARKET_SOURCE or "api").lower() != "api":
        raise ProviderError("Polymarket provider currently supports POLYMARKET_SOURCE=api only")

    if not SPORT_ALIASES.get(sport_key):
        stats["skipped_unsupported_sport"] = True
        _set_last_stats(stats)
        return []

    retries = _int_or_default(POLYMARKET_RETRIES_RAW, 2, min_value=0)
    backoff = _float_or_default(POLYMARKET_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    timeout = _int_or_default(POLYMARKET_TIMEOUT_RAW, 20, min_value=1)
    page_size = _int_or_default(POLYMARKET_PAGE_SIZE_RAW, 200, min_value=1)
    max_pages = _int_or_default(POLYMARKET_MAX_PAGES_RAW, 8, min_value=1)
    realtime_manager = _get_realtime_manager() if _websocket_realtime_enabled() else None
    if realtime_manager is not None:
        realtime_manager.ensure_started()
        _apply_realtime_stats(stats, realtime_manager.snapshot())

    client = await get_shared_client(PROVIDER_KEY, timeout=float(timeout), follow_redirects=True)
    sport_tag_mapping, payload_bundle = await asyncio.gather(
        _load_sport_tag_mapping_async(
            client=client,
            retries=retries,
            backoff_seconds=backoff,
        ),
        _load_active_game_events_async(
            client=client,
            retries=retries,
            backoff_seconds=backoff,
            page_size=page_size,
            max_pages=max_pages,
        ),
    )
    payload, payload_meta = payload_bundle
    stats["payload_cache"] = payload_meta.get("cache", "miss")
    stats["pages_fetched"] += int(payload_meta.get("pages_fetched", 0) or 0)
    stats["retries_used"] += int(payload_meta.get("retries_used", 0) or 0)
    if realtime_manager is not None and _sports_websocket_enabled():
        recent_sport_results = realtime_manager.get_sport_results(
            max_age_seconds=max(120.0, _realtime_owner_stale_seconds())
        )
        live_game_ids: List[str] = []
        for sport_result in recent_sport_results.values():
            if not _sports_result_is_tradeable(sport_result):
                continue
            game_id = _sports_result_game_id(sport_result)
            if game_id and game_id not in live_game_ids:
                live_game_ids.append(game_id)
        stats["realtime_sports_game_ids_observed"] = len(live_game_ids)
        supplemental_events, supplemental_meta = await _load_game_events_by_ids_async(
            client=client,
            game_ids=live_game_ids,
            existing_events=payload,
            retries=retries,
            backoff_seconds=backoff,
        )
        stats["realtime_sports_event_lookups"] = int(supplemental_meta.get("lookups", 0) or 0)
        stats["realtime_sports_events_supplemented"] = int(supplemental_meta.get("events_added", 0) or 0)
        stats["realtime_sports_event_lookup_errors"] = int(supplemental_meta.get("lookup_errors", 0) or 0)
        stats["retries_used"] += int(supplemental_meta.get("retries_used", 0) or 0)
        if supplemental_events:
            payload = _merge_events(payload, supplemental_events)

    events_out: List[dict] = []
    now_utc = dt.datetime.now(dt.timezone.utc)
    stats["events_payload_count"] = len(payload)
    filtered_events: List[Tuple[dict, str, str, Optional[dict]]] = []
    clob_token_ids: List[str] = []
    if True:
        for event in payload:
            if not isinstance(event, dict):
                continue
            if not _event_is_sports(event):
                continue
            stats["events_sports_count"] += 1
            if not _event_matches_sport(event, sport_key, sport_tag_mapping):
                continue
            stats["events_sport_filtered_count"] += 1
            if not _event_is_tradeable(event, now_utc):
                stats["events_status_filtered_count"] += 1
                continue
            realtime_state = None
            filtered_by_realtime_state = False
            if realtime_manager is not None and _sports_websocket_enabled():
                for sport_result_key in _event_sports_result_keys(event):
                    realtime_state = realtime_manager.get_sport_result(sport_result_key)
                    if isinstance(realtime_state, dict):
                        stats["realtime_sports_state_hits"] += 1
                        if not _sports_result_is_tradeable(realtime_state):
                            stats["events_status_filtered_count"] += 1
                            stats["realtime_sports_state_filtered_count"] += 1
                            filtered_by_realtime_state = True
                            realtime_state = None
                            break
                        break
                if filtered_by_realtime_state:
                    continue

            matchup = _extract_matchup(event)
            if not matchup:
                continue
            home_team, away_team = matchup
            stats["events_matchup_count"] += 1
            filtered_events.append((event, home_team, away_team, realtime_state))
            clob_token_ids.extend(
                _event_match_clob_token_ids(
                    event,
                    home_team,
                    away_team,
                    supported_markets,
                    now_utc,
                )
            )

        clob_quote_map: Dict[str, dict] = {}
        if clob_token_ids:
            normalized_token_ids, total_requested, truncated_count = _normalized_clob_token_ids(clob_token_ids)
            stats["clob_tokens_requested"] = total_requested
            stats["clob_tokens_considered"] = len(normalized_token_ids)
            stats["clob_tokens_truncated"] = truncated_count
            unresolved_token_ids = list(normalized_token_ids)
            if realtime_manager is not None and _market_websocket_enabled():
                stats["realtime_market_new_subscriptions"] = realtime_manager.subscribe_assets(normalized_token_ids)
                quote_wait_seconds = max(_ws_warmup_seconds(), _ws_quote_wait_seconds())
                if unresolved_token_ids and quote_wait_seconds > 0:
                    await asyncio.to_thread(
                        realtime_manager.wait_for_quotes,
                        unresolved_token_ids,
                        quote_wait_seconds,
                    )
                realtime_quote_map = realtime_manager.get_quote_map(
                    normalized_token_ids,
                    max_age_seconds=_ws_book_max_age_seconds(),
                )
                if realtime_quote_map:
                    clob_quote_map.update(realtime_quote_map)
                unresolved_token_ids = [
                    token_id for token_id in normalized_token_ids if token_id not in clob_quote_map
                ]
                stats["realtime_market_books_hit"] = len(clob_quote_map)
                stats["realtime_market_books_missed"] = len(unresolved_token_ids)
                stats["realtime_market_books_pending"] = len(unresolved_token_ids)
                _apply_realtime_stats(stats, realtime_manager.snapshot())
            if unresolved_token_ids:
                rest_quote_map, clob_meta = await _load_clob_quote_map_async(
                    client=client,
                    token_ids=unresolved_token_ids,
                    retries=retries,
                    backoff_seconds=backoff,
                )
                clob_quote_map.update(rest_quote_map)
                stats["retries_used"] += int(clob_meta.get("retries_used", 0) or 0)
                stats["clob_books_fetched"] = int(clob_meta.get("books_fetched", 0) or 0)
                stats["clob_book_errors"] = int(clob_meta.get("book_errors", 0) or 0)
            stats["clob_http_fallback_skipped"] = sum(
                1
                for token_id in normalized_token_ids
                if token_id not in clob_quote_map
            )
            stats["clob_books_with_quotes"] = sum(
                1
                for token_id in normalized_token_ids
                if token_id in clob_quote_map and _safe_float(clob_quote_map[token_id].get("decimal_odds")) is not None
            )
            stats["clob_books_with_depth"] = stats["clob_books_with_quotes"]

        for event, home_team, away_team, realtime_state in filtered_events:
            market_list = _pick_match_markets(
                event,
                home_team,
                away_team,
                supported_markets,
                now_utc,
                clob_quote_map=clob_quote_map,
            )
            if not market_list:
                continue
            stats["events_with_market_count"] += 1

            commence = _normalize_commence_time(event.get("startTime"))
            if not commence:
                commence = _normalize_commence_time(
                    (event.get("markets") or [{}])[0].get("gameStartTime")
                    if isinstance(event.get("markets"), list) and event.get("markets")
                    else None
                )
            if not commence:
                commence = _normalize_commence_time(
                    event.get("eventDate")
                    or event.get("startDate")
                    or event.get("creationDate")
                    or event.get("createdAt")
                )
            if not commence:
                continue

            event_id = _normalize_text(event.get("id") or event.get("slug"))
            if not event_id:
                continue

            events_out.append(
                {
                    "id": event_id,
                    "sport_key": sport_key,
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": commence,
                    "live_state": _event_live_state_payload(event, realtime_state),
                    "bookmakers": [
                        {
                            "key": PROVIDER_KEY,
                            "title": PROVIDER_TITLE,
                            "event_id": event_id,
                            "event_url": _event_url(event),
                            "live_state": _event_live_state_payload(event, realtime_state),
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
    "source": POLYMARKET_SOURCE or "api",
    "skipped_unsupported_sport": False,
    "clob_http_fallback_enabled": True,
    "clob_http_depth_enabled": _http_clob_depth_enabled(),
    "clob_http_fallback_skipped": 0,
    "events_returned_count": 0,
}
fetch_events_async.last_stats = dict(fetch_events.last_stats)


from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests

from ._async_http import get_shared_client, request_json

PROVIDER_KEY = "betdex"
PROVIDER_TITLE = "BetDEX"

BETDEX_SOURCE = os.getenv("BETDEX_SOURCE", "api").strip().lower()
BETDEX_SAMPLE_PATH = os.getenv("BETDEX_SAMPLE_PATH", str(Path("data") / "betdex_sample.json")).strip()
BETDEX_SESSION_URL = os.getenv("BETDEX_SESSION_URL", "https://www.betdex.com/api/session").strip()
BETDEX_MONACO_API_BASE = os.getenv("BETDEX_MONACO_API_BASE", "https://production.api.monacoprotocol.xyz").strip()
BETDEX_PUBLIC_BASE = os.getenv("BETDEX_PUBLIC_BASE", "https://www.betdex.com").strip()
BETDEX_APP_ID = os.getenv("BETDEX_APP_ID", "").strip()
BETDEX_API_KEY = os.getenv("BETDEX_API_KEY", "").strip()
BETDEX_TIMEOUT_RAW = os.getenv("BETDEX_TIMEOUT_SECONDS", "20").strip()
BETDEX_RETRIES_RAW = os.getenv("BETDEX_RETRIES", "2").strip()
BETDEX_RETRY_BACKOFF_RAW = os.getenv("BETDEX_RETRY_BACKOFF", "0.5").strip()
BETDEX_SESSION_CACHE_TTL_RAW = os.getenv("BETDEX_SESSION_CACHE_TTL_SECONDS", "900").strip()
BETDEX_SESSION_EXPIRY_SKEW_RAW = os.getenv("BETDEX_SESSION_EXPIRY_SKEW_SECONDS", "30").strip()
BETDEX_EVENTS_PAGE_SIZE_RAW = os.getenv("BETDEX_EVENTS_PAGE_SIZE", "250").strip()
BETDEX_EVENTS_MAX_PAGES_RAW = os.getenv("BETDEX_EVENTS_MAX_PAGES", "8").strip()
BETDEX_MARKETS_PAGE_SIZE_RAW = os.getenv("BETDEX_MARKETS_PAGE_SIZE", "500").strip()
BETDEX_MARKETS_MAX_PAGES_RAW = os.getenv("BETDEX_MARKETS_MAX_PAGES", "8").strip()
BETDEX_EVENT_BATCH_SIZE_RAW = os.getenv("BETDEX_EVENT_BATCH_SIZE", "60").strip()
BETDEX_PRICE_BATCH_SIZE_RAW = os.getenv("BETDEX_PRICE_BATCH_SIZE", "120").strip()
BETDEX_MARKET_STATUSES_RAW = os.getenv("BETDEX_MARKET_STATUSES", "Open").strip()
BETDEX_BACK_PRICE_SIDE_RAW = os.getenv("BETDEX_BACK_PRICE_SIDE", "against").strip()
BETDEX_USER_AGENT = os.getenv(
    "BETDEX_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
).strip()

SPORT_SUBCATEGORY_DEFAULTS: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("AMFOOT",),
    "americanfootball_ncaaf": ("AMFOOT",),
    "basketball_nba": ("BBALL",),
    "basketball_ncaab": ("BBALL",),
    "basketball_euroleague": ("BBALL",),
    "basketball_germany_bbl": ("BBALL",),
    "baseball_mlb": ("BASEBALL",),
    "icehockey_nhl": ("ICEHKY",),
    "soccer_epl": ("FOOTBALL",),
    "soccer_england_championship": ("FOOTBALL",),
    "soccer_england_league_one": ("FOOTBALL",),
    "soccer_england_league_two": ("FOOTBALL",),
    "soccer_spain_la_liga": ("FOOTBALL",),
    "soccer_germany_bundesliga": ("FOOTBALL",),
    "soccer_italy_serie_a": ("FOOTBALL",),
    "soccer_france_ligue_one": ("FOOTBALL",),
    "soccer_brazil_serie_a": ("FOOTBALL",),
    "soccer_netherlands_eredivisie": ("FOOTBALL",),
    "soccer_argentina_liga_profesional": ("FOOTBALL",),
    "soccer_usa_mls": ("FOOTBALL",),
    "mma_ufc": ("MMA",),
    "rugby_union": ("RUGBY",),
    "tennis_atp": ("TENNIS",),
    "tennis_wta": ("TENNIS",),
}

SPORT_LEAGUE_HINTS: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("nfl",),
    "americanfootball_ncaaf": ("ncaaf", "ncaa football", "college football"),
    "basketball_nba": ("nba",),
    "basketball_ncaab": ("ncaab", "ncaa", "college"),
    "basketball_euroleague": ("euroleague",),
    "basketball_germany_bbl": ("germany bbl", "bbl",),
    "baseball_mlb": ("mlb",),
    "icehockey_nhl": ("nhl",),
    "soccer_epl": ("premier league", "epl"),
    "soccer_england_championship": ("english championship", "championship"),
    "soccer_england_league_one": ("english football league 1", "league 1"),
    "soccer_england_league_two": ("english football league 2", "league 2"),
    "soccer_spain_la_liga": ("la liga",),
    "soccer_germany_bundesliga": ("bundesliga",),
    "soccer_italy_serie_a": ("serie a",),
    "soccer_france_ligue_one": ("ligue 1", "ligue one"),
    "soccer_brazil_serie_a": ("brasileiro serie a", "brasileiro s?rie a",),
    "soccer_netherlands_eredivisie": ("eredivisie",),
    "soccer_argentina_liga_profesional": ("argentina liga profesional", "liga profesional",),
    "soccer_usa_mls": ("mls", "major league soccer"),
    "mma_ufc": ("ufc",),
    "rugby_union": ("rugby", "rugby union", "six nations"),
    "tennis_atp": ("atp",),
    "tennis_wta": ("wta",),
}


class ProviderError(Exception):
    """Raised for provider-specific recoverable issues."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


ACCESS_TOKEN_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "token": "",
}
ACCESS_TOKEN_CACHE_LOCK = threading.RLock()
ACCESS_TOKEN_CACHE_ASYNC_LOCK: Optional[asyncio.Lock] = None


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


def _session_cache_async_lock() -> asyncio.Lock:
    global ACCESS_TOKEN_CACHE_ASYNC_LOCK
    if ACCESS_TOKEN_CACHE_ASYNC_LOCK is None:
        ACCESS_TOKEN_CACHE_ASYNC_LOCK = asyncio.Lock()
    return ACCESS_TOKEN_CACHE_ASYNC_LOCK


def _session_cache_ttl_seconds() -> float:
    return _float_or_default(BETDEX_SESSION_CACHE_TTL_RAW, 900.0, min_value=0.0)


def _session_expiry_skew_seconds() -> float:
    return _float_or_default(BETDEX_SESSION_EXPIRY_SKEW_RAW, 30.0, min_value=0.0)


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_token(value: object) -> str:
    text = _normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _session_url() -> str:
    value = _normalize_text(BETDEX_SESSION_URL) or "https://www.betdex.com/api/session"
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = f"https://{value}"
    return value


def _official_session_url() -> str:
    return f"{_api_base()}/sessions"


def _configured_session_auth_mode() -> str:
    has_app_id = bool(_normalize_text(BETDEX_APP_ID))
    has_api_key = bool(_normalize_text(BETDEX_API_KEY))
    if has_app_id and has_api_key:
        return "official"
    if has_app_id or has_api_key:
        return "misconfigured"
    return "public"


def _session_auth_mode() -> str:
    mode = _configured_session_auth_mode()
    if mode == "misconfigured":
        raise ProviderError("BetDEX official auth requires both BETDEX_APP_ID and BETDEX_API_KEY")
    return mode


def _api_base() -> str:
    value = _normalize_text(BETDEX_MONACO_API_BASE) or "https://production.api.monacoprotocol.xyz"
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = f"https://{value}"
    return value.rstrip("/")


def _public_base() -> str:
    value = _normalize_text(BETDEX_PUBLIC_BASE) or "https://www.betdex.com"
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = f"https://{value}"
    return value.rstrip("/")


def _headers(access_token: Optional[str] = None) -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if BETDEX_USER_AGENT:
        headers["User-Agent"] = BETDEX_USER_AGENT
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def _request_json(
    url: str,
    params: Optional[object],
    access_token: Optional[str],
    retries: int,
    backoff_seconds: float,
    timeout: int,
    method: str = "GET",
    json_payload: object = None,
) -> Tuple[object, int]:
    retriable = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    last_error: Optional[ProviderError] = None
    for attempt in range(attempts):
        try:
            response = requests.request(
                method,
                url,
                params=params,
                headers=_headers(access_token),
                json=json_payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = ProviderError(f"BetDEX network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if response.status_code >= 400:
            if response.status_code in retriable and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise ProviderError(
                f"BetDEX request failed ({response.status_code})",
                status_code=response.status_code,
            )
        try:
            return response.json(), attempt
        except ValueError as exc:
            last_error = ProviderError("Failed to parse BetDEX response as JSON")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
    if last_error:
        raise last_error
    raise ProviderError("BetDEX request failed")


async def _request_json_async(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[object],
    access_token: Optional[str],
    retries: int,
    backoff_seconds: float,
    timeout: int,
    method: str = "GET",
    json_payload: object = None,
) -> Tuple[object, int]:
    return await request_json(
        client,
        method,
        url,
        params=params,
        headers=_headers(access_token),
        json_payload=json_payload,
        timeout=float(timeout),
        retries=retries,
        backoff_seconds=backoff_seconds,
        error_cls=ProviderError,
        network_error_prefix="BetDEX network error",
        parse_error_message="Failed to parse BetDEX response as JSON",
        status_error_message=lambda status_code: f"BetDEX request failed ({status_code})",
    )

def _chunked(values: Sequence[str], size: int) -> List[List[str]]:
    out: List[List[str]] = []
    step = max(1, size)
    for i in range(0, len(values), step):
        out.append(list(values[i : i + step]))
    return out


def _doc_ref_ids(value: object) -> List[str]:
    if not isinstance(value, dict):
        return []
    raw = value.get("_ids")
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        text = _normalize_text(item)
        if text:
            out.append(text)
    return out


def _doc_ref_first_id(value: object) -> Optional[str]:
    ids = _doc_ref_ids(value)
    return ids[0] if ids else None


def _parse_env_list(raw: str) -> List[str]:
    if not raw:
        return []
    if raw.startswith("["):
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if isinstance(payload, list):
            return [_normalize_text(item) for item in payload if _normalize_text(item)]
    return [item for item in re.split(r"[,\s]+", raw) if item]


def _canonical_price_side(value: object) -> str:
    token = re.sub(r"[^a-z]+", "", _normalize_text(value).lower())
    mapping = {
        "for": "for",
        "back": "for",
        "buy": "for",
        "against": "against",
        "lay": "against",
        "sell": "against",
    }
    return mapping.get(token, "")


def _back_price_side() -> str:
    configured = _canonical_price_side(BETDEX_BACK_PRICE_SIDE_RAW)
    return configured or "against"


def _normalize_commence_time(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts /= 1000.0
            return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OSError, OverflowError):
            return None
    text = _normalize_text(value)
    if not text:
        return None
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


def _timestamp_epoch(value: object) -> Optional[float]:
    normalized = _normalize_commence_time(value)
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


def _latest_timestamp_value(*values: object) -> Optional[str]:
    best_value = None
    best_epoch = None
    for value in values:
        epoch = _timestamp_epoch(value)
        if epoch is None:
            continue
        if best_epoch is None or epoch > best_epoch:
            best_epoch = epoch
            best_value = _normalize_commence_time(value)
    return best_value


def _betdex_live_state_payload(event: object, markets: Optional[Sequence[dict]] = None) -> Optional[dict]:
    if not isinstance(event, dict):
        return None
    payload: Dict[str, object] = {}
    actual_start = _normalize_commence_time(event.get("actualStartTime"))
    actual_end = _normalize_commence_time(event.get("actualEndTime"))
    in_play_status = ""
    market_status = ""
    suspended = False
    market_updated_values: List[object] = []
    for market in markets or []:
        if not isinstance(market, dict):
            continue
        token = _normalize_status_token(market.get("inPlayStatus"))
        if token == "inplay":
            in_play_status = "in_play"
        elif not in_play_status and token:
            in_play_status = token
        if not market_status:
            market_status = _normalize_status_token(market.get("status"))
        suspended = suspended or bool(market.get("suspended"))
        market_updated_values.extend([market.get("modifiedAt"), market.get("createdAt"), market.get("settledAt")])
    if actual_end:
        payload["status"] = "final"
        payload["is_live"] = False
    elif actual_start:
        payload["status"] = "live"
        payload["is_live"] = True
    elif in_play_status in {"in_play", "inplay"}:
        payload["status"] = "live"
        payload["is_live"] = True
    elif in_play_status in {"pre_play", "preplay"}:
        payload["status"] = "scheduled"
        payload["is_live"] = False
    if in_play_status:
        payload["in_play_status"] = in_play_status
    if suspended:
        payload["market_status"] = "suspended"
    elif market_status:
        payload["market_status"] = market_status
    updated_at = _latest_timestamp_value(
        actual_end,
        event.get("modifiedAt"),
        actual_start,
        event.get("createdAt"),
        *market_updated_values,
    )
    if updated_at:
        payload["updated_at"] = updated_at
    return payload or None


def _session_entry(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ProviderError("BetDEX session endpoint returned an invalid payload")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or not sessions or not isinstance(sessions[0], dict):
        raise ProviderError("BetDEX session endpoint returned no sessions")
    return sessions[0]


def _extract_access_token(payload: object) -> str:
    session = _session_entry(payload)
    token = _normalize_text(session.get("accessToken"))
    if not token:
        raise ProviderError("BetDEX session endpoint returned an empty access token")
    return token


def _parse_iso_datetime(value: object) -> Optional[float]:
    text = _normalize_text(value)
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def _extract_access_token_expires_at(payload: object) -> Optional[float]:
    session = _session_entry(payload)
    expires_at = _parse_iso_datetime(session.get("accessExpiresAt"))
    if expires_at is None:
        return None
    return max(0.0, expires_at - _session_expiry_skew_seconds())


def _session_error(exc: ProviderError) -> ProviderError:
    if _configured_session_auth_mode() != "public":
        return exc
    text = _normalize_text(exc).lower()
    if exc.status_code == 429 or "parse betdex response as json" in text or "invalid payload" in text:
        return ProviderError(
            "BetDEX public website session is blocked by Vercel Security Checkpoint. "
            "Configure BETDEX_APP_ID and BETDEX_API_KEY to use the official Monaco POST /sessions API, "
            "or disable BetDEX.",
            status_code=exc.status_code,
        )
    return exc


def _cached_access_token() -> str:
    with ACCESS_TOKEN_CACHE_LOCK:
        token = _normalize_text(ACCESS_TOKEN_CACHE.get("token"))
        expires_at = float(ACCESS_TOKEN_CACHE.get("expires_at", 0.0) or 0.0)
    if token and time.time() < expires_at:
        return token
    return ""


def _store_access_token(token: str, expires_at: Optional[float] = None) -> None:
    if expires_at is None:
        ttl = _session_cache_ttl_seconds()
        expires_at = time.time() + ttl if ttl > 0 else time.time()
    with ACCESS_TOKEN_CACHE_LOCK:
        ACCESS_TOKEN_CACHE["token"] = token
        ACCESS_TOKEN_CACHE["expires_at"] = expires_at


def _supports_sport(sport_key: str) -> bool:
    key = _normalize_text(sport_key).lower()
    return bool(SPORT_SUBCATEGORY_DEFAULTS.get(key) or SPORT_LEAGUE_HINTS.get(key))


def _fetch_access_token(retries: int, backoff_seconds: float, timeout: int) -> Tuple[str, int, str]:
    cached = _cached_access_token()
    if cached:
        return cached, 0, "hit"
    mode = _session_auth_mode()
    try:
        if mode == "official":
            payload, retries_used = _request_json(
                _official_session_url(),
                params=None,
                access_token=None,
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout=timeout,
                method="POST",
                json_payload={
                    "appId": _normalize_text(BETDEX_APP_ID),
                    "apiKey": _normalize_text(BETDEX_API_KEY),
                },
            )
        else:
            payload, retries_used = _request_json(
                _session_url(),
                params=None,
                access_token=None,
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout=timeout,
            )
    except ProviderError as exc:
        raise _session_error(exc) from exc
    token = _extract_access_token(payload)
    _store_access_token(token, expires_at=_extract_access_token_expires_at(payload))
    return token, retries_used, "miss"


async def _fetch_access_token_async(
    client: httpx.AsyncClient,
    retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Tuple[str, int, str]:
    cached = _cached_access_token()
    if cached:
        return cached, 0, "hit"
    async with _session_cache_async_lock():
        cached = _cached_access_token()
        if cached:
            return cached, 0, "hit"
        mode = _session_auth_mode()
        try:
            if mode == "official":
                payload, retries_used = await _request_json_async(
                    client,
                    _official_session_url(),
                    params=None,
                    access_token=None,
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                    timeout=timeout,
                    method="POST",
                    json_payload={
                        "appId": _normalize_text(BETDEX_APP_ID),
                        "apiKey": _normalize_text(BETDEX_API_KEY),
                    },
                )
            else:
                payload, retries_used = await _request_json_async(
                    client,
                    _session_url(),
                    params=None,
                    access_token=None,
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                    timeout=timeout,
                )
        except ProviderError as exc:
            raise _session_error(exc) from exc
        token = _extract_access_token(payload)
        _store_access_token(token, expires_at=_extract_access_token_expires_at(payload))
        return token, retries_used, "miss"


def _fetch_events_dataset(
    token: str,
    subcategory_ids: Optional[Sequence[str]],
    retries: int,
    backoff_seconds: float,
    timeout: int,
    page_size: int,
    max_pages: int,
) -> Tuple[dict, dict]:
    events: Dict[str, dict] = {}
    event_groups: Dict[str, dict] = {}
    participants: Dict[str, dict] = {}
    pages_fetched = 0
    retries_used = 0

    subcategories = [item for item in (subcategory_ids or []) if _normalize_text(item)]
    if not subcategories:
        subcategories = [None]

    for subcategory in subcategories:
        for page in range(max_pages):
            params: Dict[str, object] = {"active": "true", "page": page, "size": page_size}
            if subcategory:
                params["subcategoryIds"] = subcategory
            payload, attempt = _request_json(
                f"{_api_base()}/events",
                params=params,
                access_token=token,
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout=timeout,
            )
            pages_fetched += 1
            retries_used += attempt
            if not isinstance(payload, dict):
                break
            page_events = payload.get("events") if isinstance(payload.get("events"), list) else []
            for event in page_events:
                if not isinstance(event, dict):
                    continue
                event_id = _normalize_text(event.get("id"))
                if event_id and event_id not in events:
                    events[event_id] = event
            for group in payload.get("eventGroups") or []:
                if not isinstance(group, dict):
                    continue
                group_id = _normalize_text(group.get("id"))
                if group_id and group_id not in event_groups:
                    event_groups[group_id] = group
            for participant in payload.get("participants") or []:
                if not isinstance(participant, dict):
                    continue
                participant_id = _normalize_text(participant.get("id"))
                if participant_id and participant_id not in participants:
                    participants[participant_id] = participant

            page_meta = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}
            page_data = page_meta.get("_page") if isinstance(page_meta.get("_page"), dict) else {}
            total_pages = _int_or_default(str(page_data.get("_totalPages") or "1"), 1, min_value=1)
            if not page_events:
                break
            if len(page_events) < page_size:
                break
            if page + 1 >= total_pages:
                break

    return {
        "events": list(events.values()),
        "eventGroups": list(event_groups.values()),
        "participants": list(participants.values()),
    }, {"pages_fetched": pages_fetched, "retries_used": retries_used}


async def _fetch_events_dataset_async(
    client: httpx.AsyncClient,
    token: str,
    subcategory_ids: Optional[Sequence[str]],
    retries: int,
    backoff_seconds: float,
    timeout: int,
    page_size: int,
    max_pages: int,
) -> Tuple[dict, dict]:
    events: Dict[str, dict] = {}
    event_groups: Dict[str, dict] = {}
    participants: Dict[str, dict] = {}
    pages_fetched = 0
    retries_used = 0

    subcategories = [item for item in (subcategory_ids or []) if _normalize_text(item)]
    if not subcategories:
        subcategories = [None]

    for subcategory in subcategories:
        for page in range(max_pages):
            params: Dict[str, object] = {"active": "true", "page": page, "size": page_size}
            if subcategory:
                params["subcategoryIds"] = subcategory
            payload, attempt = await _request_json_async(
                client,
                f"{_api_base()}/events",
                params=params,
                access_token=token,
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout=timeout,
            )
            pages_fetched += 1
            retries_used += attempt
            if not isinstance(payload, dict):
                break
            page_events = payload.get("events") if isinstance(payload.get("events"), list) else []
            for event in page_events:
                if not isinstance(event, dict):
                    continue
                event_id = _normalize_text(event.get("id"))
                if event_id and event_id not in events:
                    events[event_id] = event
            for group in payload.get("eventGroups") or []:
                if not isinstance(group, dict):
                    continue
                group_id = _normalize_text(group.get("id"))
                if group_id and group_id not in event_groups:
                    event_groups[group_id] = group
            for participant in payload.get("participants") or []:
                if not isinstance(participant, dict):
                    continue
                participant_id = _normalize_text(participant.get("id"))
                if participant_id and participant_id not in participants:
                    participants[participant_id] = participant

            page_meta = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}
            page_data = page_meta.get("_page") if isinstance(page_meta.get("_page"), dict) else {}
            total_pages = _int_or_default(str(page_data.get("_totalPages") or "1"), 1, min_value=1)
            if not page_events:
                break
            if len(page_events) < page_size:
                break
            if page + 1 >= total_pages:
                break

    return {
        "events": list(events.values()),
        "eventGroups": list(event_groups.values()),
        "participants": list(participants.values()),
    }, {"pages_fetched": pages_fetched, "retries_used": retries_used}


def _fetch_markets_dataset(
    token: str,
    event_ids: Sequence[str],
    retries: int,
    backoff_seconds: float,
    timeout: int,
    batch_size: int,
    page_size: int,
    max_pages: int,
    statuses: Sequence[str],
) -> Tuple[dict, dict]:
    markets: Dict[str, dict] = {}
    outcomes: Dict[str, dict] = {}
    pages_fetched = 0
    retries_used = 0

    for event_batch in _chunked(event_ids, batch_size):
        for page in range(max_pages):
            params = [("eventIds", event_id) for event_id in event_batch]
            params.extend([("published", "true"), ("page", page), ("size", page_size)])
            for status in statuses:
                params.append(("statuses", status))
            payload, attempt = _request_json(
                f"{_api_base()}/markets",
                params=params,
                access_token=token,
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout=timeout,
            )
            pages_fetched += 1
            retries_used += attempt
            if not isinstance(payload, dict):
                break
            page_markets = payload.get("markets") if isinstance(payload.get("markets"), list) else []
            for market in page_markets:
                if not isinstance(market, dict):
                    continue
                market_id = _normalize_text(market.get("id"))
                if market_id and market_id not in markets:
                    markets[market_id] = market
            for outcome in payload.get("marketOutcomes") or []:
                if not isinstance(outcome, dict):
                    continue
                outcome_id = _normalize_text(outcome.get("id"))
                if outcome_id and outcome_id not in outcomes:
                    outcomes[outcome_id] = outcome

            page_meta = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}
            page_data = page_meta.get("_page") if isinstance(page_meta.get("_page"), dict) else {}
            total_pages = _int_or_default(str(page_data.get("_totalPages") or "1"), 1, min_value=1)
            if not page_markets:
                break
            if len(page_markets) < page_size:
                break
            if page + 1 >= total_pages:
                break

    return {
        "markets": list(markets.values()),
        "marketOutcomes": list(outcomes.values()),
    }, {"pages_fetched": pages_fetched, "retries_used": retries_used}


async def _fetch_markets_dataset_async(
    client: httpx.AsyncClient,
    token: str,
    event_ids: Sequence[str],
    retries: int,
    backoff_seconds: float,
    timeout: int,
    batch_size: int,
    page_size: int,
    max_pages: int,
    statuses: Sequence[str],
) -> Tuple[dict, dict]:
    markets: Dict[str, dict] = {}
    outcomes: Dict[str, dict] = {}
    pages_fetched = 0
    retries_used = 0

    for event_batch in _chunked(event_ids, batch_size):
        for page in range(max_pages):
            params = [("eventIds", event_id) for event_id in event_batch]
            params.extend([("published", "true"), ("page", page), ("size", page_size)])
            for status in statuses:
                params.append(("statuses", status))
            payload, attempt = await _request_json_async(
                client,
                f"{_api_base()}/markets",
                params=params,
                access_token=token,
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout=timeout,
            )
            pages_fetched += 1
            retries_used += attempt
            if not isinstance(payload, dict):
                break
            page_markets = payload.get("markets") if isinstance(payload.get("markets"), list) else []
            for market in page_markets:
                if not isinstance(market, dict):
                    continue
                market_id = _normalize_text(market.get("id"))
                if market_id and market_id not in markets:
                    markets[market_id] = market
            for outcome in payload.get("marketOutcomes") or []:
                if not isinstance(outcome, dict):
                    continue
                outcome_id = _normalize_text(outcome.get("id"))
                if outcome_id and outcome_id not in outcomes:
                    outcomes[outcome_id] = outcome

            page_meta = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}
            page_data = page_meta.get("_page") if isinstance(page_meta.get("_page"), dict) else {}
            total_pages = _int_or_default(str(page_data.get("_totalPages") or "1"), 1, min_value=1)
            if not page_markets:
                break
            if len(page_markets) < page_size:
                break
            if page + 1 >= total_pages:
                break

    return {
        "markets": list(markets.values()),
        "marketOutcomes": list(outcomes.values()),
    }, {"pages_fetched": pages_fetched, "retries_used": retries_used}


def _fetch_prices_by_market(
    token: str,
    market_ids: Sequence[str],
    retries: int,
    backoff_seconds: float,
    timeout: int,
    batch_size: int,
) -> Tuple[Dict[str, dict], dict]:
    out: Dict[str, dict] = {}
    pages_fetched = 0
    retries_used = 0
    side_counts = {"for": 0, "against": 0, "unknown": 0}
    for market_batch in _chunked(market_ids, batch_size):
        params = [("marketIds", market_id) for market_id in market_batch]
        payload, attempt = _request_json(
            f"{_api_base()}/market-prices",
            params=params,
            access_token=token,
            retries=retries,
            backoff_seconds=backoff_seconds,
            timeout=timeout,
        )
        observed_at = time.time()
        pages_fetched += 1
        retries_used += attempt
        if not isinstance(payload, dict):
            continue
        for entry in payload.get("prices") or []:
            if not isinstance(entry, dict):
                continue
            market_id = _normalize_text(entry.get("marketId"))
            rows = entry.get("prices")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    row["observed_at"] = observed_at
                    side = _canonical_price_side(row.get("side"))
                    if side == "for":
                        side_counts["for"] += 1
                    elif side == "against":
                        side_counts["against"] += 1
                    else:
                        side_counts["unknown"] += 1
            if market_id:
                entry["observed_at"] = observed_at
                out[market_id] = entry
    return out, {
        "pages_fetched": pages_fetched,
        "retries_used": retries_used,
        "price_side_counts": side_counts,
    }


async def _fetch_prices_by_market_async(
    client: httpx.AsyncClient,
    token: str,
    market_ids: Sequence[str],
    retries: int,
    backoff_seconds: float,
    timeout: int,
    batch_size: int,
) -> Tuple[Dict[str, dict], dict]:
    out: Dict[str, dict] = {}
    pages_fetched = 0
    retries_used = 0
    side_counts = {"for": 0, "against": 0, "unknown": 0}
    for market_batch in _chunked(market_ids, batch_size):
        params = [("marketIds", market_id) for market_id in market_batch]
        payload, attempt = await _request_json_async(
            client,
            f"{_api_base()}/market-prices",
            params=params,
            access_token=token,
            retries=retries,
            backoff_seconds=backoff_seconds,
            timeout=timeout,
        )
        observed_at = time.time()
        pages_fetched += 1
        retries_used += attempt
        if not isinstance(payload, dict):
            continue
        for entry in payload.get("prices") or []:
            if not isinstance(entry, dict):
                continue
            market_id = _normalize_text(entry.get("marketId"))
            rows = entry.get("prices")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    row["observed_at"] = observed_at
                    side = _canonical_price_side(row.get("side"))
                    if side == "for":
                        side_counts["for"] += 1
                    elif side == "against":
                        side_counts["against"] += 1
                    else:
                        side_counts["unknown"] += 1
            if market_id:
                entry["observed_at"] = observed_at
                out[market_id] = entry
    return out, {
        "pages_fetched": pages_fetched,
        "retries_used": retries_used,
        "price_side_counts": side_counts,
    }

def _clean_team_name(value: object) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _extract_matchup_from_text(value: object) -> Optional[Tuple[str, str]]:
    text = _normalize_text(value)
    if not text:
        return None
    patterns = [
        r"^\s*(.+?)\s+v(?:s\.?)?\s+(.+?)\s*$",
        r"^\s*(.+?)\s+@\s+(.+?)\s*$",
        r"^\s*(.+?)\s+at\s+(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        first = _clean_team_name(match.group(1))
        second = _clean_team_name(match.group(2))
        if first and second and _normalize_token(first) != _normalize_token(second):
            return first, second
    return None


def _extract_matchup(event: dict, participants_by_id: Dict[str, dict]) -> Optional[Tuple[str, str]]:
    parsed = _extract_matchup_from_text(event.get("name"))
    if parsed:
        return parsed
    names = []
    for participant_id in _doc_ref_ids(event.get("participants")):
        participant = participants_by_id.get(participant_id)
        if not isinstance(participant, dict):
            continue
        name = _clean_team_name(participant.get("name"))
        if name:
            names.append(name)
    if len(names) >= 2 and _normalize_token(names[0]) != _normalize_token(names[1]):
        return names[0], names[1]
    return None


def _event_matches_sport(
    event: dict,
    sport_key: str,
    event_groups_by_id: Dict[str, dict],
    expected_subcategories: Sequence[str],
    league_hints: Sequence[str],
    strict_subcategory: bool,
) -> bool:
    if not sport_key:
        return True
    expected = {_normalize_text(item).upper() for item in expected_subcategories if _normalize_text(item)}
    hints = [_normalize_token(item) for item in league_hints if _normalize_token(item)]
    event_group = event_groups_by_id.get(_doc_ref_first_id(event.get("eventGroup")) or "")
    group_name = _normalize_text(event_group.get("name")) if isinstance(event_group, dict) else ""
    subcategory_id = (
        _normalize_text(_doc_ref_first_id(event_group.get("subcategory"))).upper()
        if isinstance(event_group, dict)
        else ""
    )
    if strict_subcategory and expected and subcategory_id not in expected:
        return False
    if hints:
        haystack = _normalize_token(" ".join(
            item
            for item in (
                _normalize_text(event.get("name")),
                _normalize_text(event.get("code")),
                group_name,
            )
            if item
        ))
        return any(hint in haystack for hint in hints)
    return True


def _best_back_prices(entry: Optional[dict], back_side: Optional[str] = None) -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    if not isinstance(entry, dict):
        return best
    prices = entry.get("prices")
    if not isinstance(prices, list):
        return best
    target_side = _canonical_price_side(back_side) if back_side is not None else _back_price_side()
    for item in prices:
        if not isinstance(item, dict):
            continue
        if _canonical_price_side(item.get("side")) != target_side:
            continue
        outcome_id = _normalize_text(item.get("outcomeId"))
        price = _safe_float(item.get("price"))
        amount = _safe_float(item.get("amount"))
        if not outcome_id or price is None or price <= 1:
            continue
        existing = best.get(outcome_id)
        existing_price = _safe_float(existing.get("price")) if isinstance(existing, dict) else None
        existing_amount = _safe_float(existing.get("amount")) if isinstance(existing, dict) else None
        is_better_price = existing_price is None or price > existing_price
        is_tie_better_size = (
            existing_price is not None
            and abs(price - existing_price) <= 1e-12
            and (amount or 0.0) > (existing_amount or 0.0)
        )
        if is_better_price or is_tie_better_size:
            best[outcome_id] = {
                "price": float(price),
                "amount": amount,
                "observed_at": item.get("observed_at")
                if item.get("observed_at") not in (None, "")
                else entry.get("observed_at"),
            }
    return best


def _market_type_id(market: dict) -> str:
    return _normalize_text(_doc_ref_first_id(market.get("marketType"))).upper()


def _normalize_market_key(value: object) -> str:
    token = _normalize_text(value).lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


def _market_period_suffix(*values: object) -> str:
    tokens = "_".join(_normalize_market_key(item) for item in values if _normalize_market_key(item))
    if not tokens:
        return ""
    # Check more specific patterns first.
    if any(token in tokens for token in ("second_half", "2nd_half", "half_2", "h2")):
        return "h2"
    if any(token in tokens for token in ("first_half", "1st_half", "half_time", "halftime", "half_1", "h1")):
        return "h1"
    quarter_patterns = {
        "q1": ("q1", "quarter_1", "1st_quarter", "first_quarter"),
        "q2": ("q2", "quarter_2", "2nd_quarter", "second_quarter"),
        "q3": ("q3", "quarter_3", "3rd_quarter", "third_quarter"),
        "q4": ("q4", "quarter_4", "4th_quarter", "fourth_quarter"),
    }
    for suffix, patterns in quarter_patterns.items():
        if any(pattern in tokens for pattern in patterns):
            return suffix
    return ""


def _scoped_market_key(base_key: str, *period_hints: object) -> str:
    suffix = _market_period_suffix(*period_hints)
    if suffix:
        return f"{base_key}_{suffix}"
    return base_key


def _requested_market_keys(markets: Sequence[str]) -> set[str]:
    requested = {_normalize_market_key(item) for item in (markets or []) if _normalize_market_key(item)}
    if "both_teams_to_score" in requested:
        requested.add("btts")
    if "btts" in requested:
        requested.add("both_teams_to_score")
    return requested


def _market_aliases_for_type(market_type: str, market_name: object = None) -> List[str]:
    aliases: List[str] = []
    token = _normalize_market_key(market_type)
    if token:
        aliases.append(token)
    period_hints = (market_type, market_name)
    if "HANDICAP" in market_type:
        aliases.append(_scoped_market_key("spreads", *period_hints))
    if "OVER_UNDER" in market_type:
        aliases.append(_scoped_market_key("totals", *period_hints))
    if (
        "MONEYLINE" in market_type
        or "FULL_TIME_RESULT" in market_type
        or "MATCH_RESULT" in market_type
        or "WINNER" in market_type
    ):
        aliases.append(_scoped_market_key("h2h", *period_hints))
    if "BTTS" in market_type:
        aliases.extend(["btts", "both_teams_to_score"])
    out = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            out.append(alias)
            seen.add(alias)
    return out


def _parse_market_value_pair(value: object) -> Tuple[Optional[float], Optional[float]]:
    parts = re.findall(r"[-+]?\d+(?:\.\d+)?", _normalize_text(value))
    if len(parts) < 2:
        return None, None
    return _safe_float(parts[0]), _safe_float(parts[1])


def _parse_market_value_single(value: object) -> Optional[float]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", _normalize_text(value))
    if not match:
        return None
    return _safe_float(match.group(0))


def _parse_spread_title(title: str) -> Tuple[str, Optional[float]]:
    clean = _clean_team_name(title)
    match = re.match(r"^(.*?)([-+]\d+(?:\.\d+)?)$", clean)
    if not match:
        return clean, None
    return _clean_team_name(match.group(1)), _safe_float(match.group(2))


def _parse_total_title(title: str) -> Tuple[Optional[str], Optional[float]]:
    clean = _clean_team_name(title)
    lower = clean.lower()
    side: Optional[str] = None
    if lower.startswith("over"):
        side = "Over"
    elif lower.startswith("under"):
        side = "Under"
    return side, _parse_market_value_single(clean)


def _market_signature(market: dict) -> str:
    key = _normalize_text(market.get("key"))
    outcomes = market.get("outcomes") if isinstance(market.get("outcomes"), list) else []
    parts = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        name = _normalize_token(outcome.get("name"))
        point = _safe_float(outcome.get("point"))
        parts.append(f"{name}:{point:.6f}" if point is not None else name)
    return f"{key}:{'|'.join(sorted(parts))}"


def _score_market(market: dict) -> float:
    outcomes = market.get("outcomes") if isinstance(market.get("outcomes"), list) else []
    prices = [float(item.get("price")) for item in outcomes if _safe_float(item.get("price"))]
    if len(prices) < 2:
        return 0.0
    return min(prices)


def _event_url(event: dict, event_groups_by_id: Dict[str, dict], default_market_id: str = "") -> str:
    code = _normalize_text(event.get("code"))
    event_id = _normalize_text(event.get("id"))
    if not code and not event_id:
        return ""
    group_id = _doc_ref_first_id(event.get("eventGroup"))
    event_group = event_groups_by_id.get(group_id or "")
    subcategory_id = _doc_ref_first_id(event_group.get("subcategory")) if isinstance(event_group, dict) else None
    
    target_id = event_id if event_id else code
    base_url = ""
    if group_id and subcategory_id:
        base_url = (
            f"{_public_base()}/events/{quote(subcategory_id.lower(), safe='')}"
            f"/{quote(group_id.lower(), safe='')}/{quote(target_id, safe='')}"
        )
    else:
        base_url = f"{_public_base()}/events/{quote(target_id, safe='')}"
        
    if default_market_id:
        return f"{base_url}?market={default_market_id}"
    return base_url


def _load_event_list(path: str) -> List[dict]:
    if not path:
        return []
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _normalize_file_events(events: Sequence[dict], sport_key: str, requested_markets: set[str]) -> List[dict]:
    output: List[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if sport_key and event.get("sport_key") != sport_key:
            continue
        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        normalized_books = []
        for book in bookmakers:
            if not isinstance(book, dict):
                continue
            markets = book.get("markets")
            if not isinstance(markets, list):
                continue
            kept = []
            for market in markets:
                if not isinstance(market, dict):
                    continue
                if _normalize_text(market.get("key")).lower() in requested_markets:
                    kept.append(market)
            if not kept:
                continue
            normalized = dict(book)
            normalized["key"] = PROVIDER_KEY
            normalized["title"] = PROVIDER_TITLE
            normalized["markets"] = kept
            normalized_books.append(normalized)
        if not normalized_books:
            continue
        normalized_event = dict(event)
        normalized_event["bookmakers"] = normalized_books
        output.append(normalized_event)
    return output

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
    _ = context
    requested_markets = _requested_market_keys(markets)

    stats = {
        "provider": PROVIDER_KEY,
        "source": BETDEX_SOURCE or "api",
        "back_price_side": _back_price_side(),
        "session_auth_mode": _configured_session_auth_mode(),
        "session_cache": "miss",
        "skipped_unsupported_sport": False,
        "events_payload_count": 0,
        "events_sport_filtered_count": 0,
        "markets_payload_count": 0,
        "prices_payload_count": 0,
        "price_rows_for": 0,
        "price_rows_against": 0,
        "price_rows_unknown": 0,
        "events_with_market_count": 0,
        "events_returned_count": 0,
        "pages_fetched": 0,
        "retries_used": 0,
    }
    _set_last_stats(stats)

    if not requested_markets:
        return []

    if bookmakers:
        lowered = {_normalize_text(book).lower() for book in bookmakers if _normalize_text(book)}
        if PROVIDER_KEY not in lowered and PROVIDER_TITLE.lower() not in lowered:
            return []

    source = (BETDEX_SOURCE or "api").lower()
    if source == "file":
        payload = _load_event_list(BETDEX_SAMPLE_PATH)
        stats["events_payload_count"] = len(payload)
        normalized = _normalize_file_events(payload, sport_key=sport_key, requested_markets=requested_markets)
        stats["events_returned_count"] = len(normalized)
        _set_last_stats(stats)
        return normalized
    if source != "api":
        raise ProviderError("BetDEX provider supports BETDEX_SOURCE=api or BETDEX_SOURCE=file")

    if not _supports_sport(sport_key):
        stats["skipped_unsupported_sport"] = True
        _set_last_stats(stats)
        return []

    retries = _int_or_default(BETDEX_RETRIES_RAW, 2, min_value=0)
    backoff = _float_or_default(BETDEX_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    timeout = _int_or_default(BETDEX_TIMEOUT_RAW, 20, min_value=1)
    events_page_size = _int_or_default(BETDEX_EVENTS_PAGE_SIZE_RAW, 250, min_value=1)
    events_max_pages = _int_or_default(BETDEX_EVENTS_MAX_PAGES_RAW, 8, min_value=1)
    markets_page_size = _int_or_default(BETDEX_MARKETS_PAGE_SIZE_RAW, 500, min_value=1)
    markets_max_pages = _int_or_default(BETDEX_MARKETS_MAX_PAGES_RAW, 8, min_value=1)
    event_batch_size = _int_or_default(BETDEX_EVENT_BATCH_SIZE_RAW, 60, min_value=1)
    price_batch_size = _int_or_default(BETDEX_PRICE_BATCH_SIZE_RAW, 120, min_value=1)
    status_filters = [item for item in _parse_env_list(BETDEX_MARKET_STATUSES_RAW) if item] or ["Open"]

    expected_subcategories = SPORT_SUBCATEGORY_DEFAULTS.get(sport_key, ())
    league_hints = SPORT_LEAGUE_HINTS.get(sport_key, ())
    client = await get_shared_client(PROVIDER_KEY, timeout=float(timeout), follow_redirects=True)
    token, retries_used, session_cache = await _fetch_access_token_async(
        client,
        retries=retries,
        backoff_seconds=backoff,
        timeout=timeout,
    )
    stats["retries_used"] += retries_used
    stats["session_cache"] = session_cache

    events_dataset, events_meta = await _fetch_events_dataset_async(
        client=client,
        token=token,
        subcategory_ids=expected_subcategories,
        retries=retries,
        backoff_seconds=backoff,
        timeout=timeout,
        page_size=events_page_size,
        max_pages=events_max_pages,
    )
    stats["pages_fetched"] += int(events_meta.get("pages_fetched", 0) or 0)
    stats["retries_used"] += int(events_meta.get("retries_used", 0) or 0)
    stats["events_payload_count"] = len(events_dataset.get("events", []))

    event_groups_by_id = {
            _normalize_text(item.get("id")): item
            for item in events_dataset.get("eventGroups", [])
            if isinstance(item, dict) and _normalize_text(item.get("id"))
        }
    if True:
        participants_by_id = {
            _normalize_text(item.get("id")): item
            for item in events_dataset.get("participants", [])
            if isinstance(item, dict) and _normalize_text(item.get("id"))
        }

        filtered_events: List[dict] = []
        for event in events_dataset.get("events", []):
            if not isinstance(event, dict):
                continue
            if not event.get("active", True):
                continue
            if not _event_matches_sport(
                event,
                sport_key=sport_key,
                event_groups_by_id=event_groups_by_id,
                expected_subcategories=expected_subcategories,
                league_hints=league_hints,
                strict_subcategory=True,
            ):
                continue
            filtered_events.append(event)

        if not filtered_events and expected_subcategories:
            all_events_dataset, all_events_meta = await _fetch_events_dataset_async(
                client=client,
                token=token,
                subcategory_ids=None,
                retries=retries,
                backoff_seconds=backoff,
                timeout=timeout,
                page_size=events_page_size,
                max_pages=events_max_pages,
            )
            stats["pages_fetched"] += int(all_events_meta.get("pages_fetched", 0) or 0)
            stats["retries_used"] += int(all_events_meta.get("retries_used", 0) or 0)
            if len(all_events_dataset.get("events", [])) > stats["events_payload_count"]:
                stats["events_payload_count"] = len(all_events_dataset.get("events", []))
            event_groups_by_id = {
                _normalize_text(item.get("id")): item
                for item in all_events_dataset.get("eventGroups", [])
                if isinstance(item, dict) and _normalize_text(item.get("id"))
            }
            participants_by_id = {
                _normalize_text(item.get("id")): item
                for item in all_events_dataset.get("participants", [])
                if isinstance(item, dict) and _normalize_text(item.get("id"))
            }
            for event in all_events_dataset.get("events", []):
                if not isinstance(event, dict):
                    continue
                if not event.get("active", True):
                    continue
                if not _event_matches_sport(
                    event,
                    sport_key=sport_key,
                    event_groups_by_id=event_groups_by_id,
                    expected_subcategories=expected_subcategories,
                    league_hints=league_hints,
                    strict_subcategory=False,
                ):
                    continue
                filtered_events.append(event)

        if not filtered_events:
            _set_last_stats(stats)
            return []

        stats["events_sport_filtered_count"] = len(filtered_events)

        event_ids = list(dict.fromkeys(_normalize_text(event.get("id")) for event in filtered_events if _normalize_text(event.get("id"))))
        markets_dataset, markets_meta = await _fetch_markets_dataset_async(
            client=client,
            token=token,
            event_ids=event_ids,
            retries=retries,
            backoff_seconds=backoff,
            timeout=timeout,
            batch_size=event_batch_size,
            page_size=markets_page_size,
            max_pages=markets_max_pages,
            statuses=status_filters,
        )
        stats["pages_fetched"] += int(markets_meta.get("pages_fetched", 0) or 0)
        stats["retries_used"] += int(markets_meta.get("retries_used", 0) or 0)
        stats["markets_payload_count"] = len(markets_dataset.get("markets", []))

        outcomes_by_id = {
            _normalize_text(item.get("id")): item
            for item in markets_dataset.get("marketOutcomes", [])
            if isinstance(item, dict) and _normalize_text(item.get("id"))
        }

        markets_by_event_id: Dict[str, List[dict]] = {}
        market_ids: List[str] = []
        for market in markets_dataset.get("markets", []):
            if not isinstance(market, dict):
                continue
            if not market.get("published", True) or market.get("suspended", False):
                continue
            event_id = _doc_ref_first_id(market.get("event"))
            market_id = _normalize_text(market.get("id"))
            if not event_id or not market_id:
                continue
            markets_by_event_id.setdefault(event_id, []).append(market)
            market_ids.append(market_id)
        market_ids = list(dict.fromkeys(market_ids))

        prices_by_market_id: Dict[str, dict] = {}
        back_side = _back_price_side()
        if market_ids:
            prices_by_market_id, prices_meta = await _fetch_prices_by_market_async(
                client=client,
                token=token,
                market_ids=market_ids,
                retries=retries,
                backoff_seconds=backoff,
                timeout=timeout,
                batch_size=price_batch_size,
            )
            stats["pages_fetched"] += int(prices_meta.get("pages_fetched", 0) or 0)
            stats["retries_used"] += int(prices_meta.get("retries_used", 0) or 0)
            side_counts = prices_meta.get("price_side_counts") if isinstance(prices_meta, dict) else {}
            if isinstance(side_counts, dict):
                stats["price_rows_for"] = int(side_counts.get("for", 0) or 0)
                stats["price_rows_against"] = int(side_counts.get("against", 0) or 0)
                stats["price_rows_unknown"] = int(side_counts.get("unknown", 0) or 0)
        stats["prices_payload_count"] = len(prices_by_market_id)

        event_by_id = {
            _normalize_text(event.get("id")): event
            for event in filtered_events
            if isinstance(event, dict) and _normalize_text(event.get("id"))
        }

        events_out: List[dict] = []
        for event_id in event_ids:
            event = event_by_id.get(event_id)
            if not isinstance(event, dict):
                continue
            event_markets = markets_by_event_id.get(event_id, [])
            if not event_markets:
                continue
            matchup = _extract_matchup(event, participants_by_id)
            if not matchup:
                continue
            home_team, away_team = matchup
            commence = _normalize_commence_time(event.get("expectedStartTime") or event.get("createdAt"))
            if not commence:
                continue
            event_live_state = _betdex_live_state_payload(event, event_markets)

            best_h2h: Optional[dict] = None
            by_signature: Dict[str, dict] = {}

            for market in event_markets:
                market_id = _normalize_text(market.get("id"))
                market_type = _market_type_id(market)
                market_name = _normalize_text(market.get("name") or market.get("marketName"))
                target_h2h_key = _scoped_market_key("h2h", market_type, market_name)
                target_spread_key = _scoped_market_key("spreads", market_type, market_name)
                target_total_key = _scoped_market_key("totals", market_type, market_name)
                prices_by_outcome = _best_back_prices(
                    prices_by_market_id.get(market_id),
                    back_side=back_side,
                )
                market_updated_at = _latest_timestamp_value(
                    market.get("modifiedAt"),
                    market.get("createdAt"),
                    market.get("settledAt"),
                    event_live_state.get("updated_at") if isinstance(event_live_state, dict) else None,
                    event.get("modifiedAt"),
                )

                outcomes = []
                for outcome_id in _doc_ref_ids(market.get("marketOutcomes")):
                    outcome = outcomes_by_id.get(outcome_id)
                    price_row = prices_by_outcome.get(outcome_id) if isinstance(prices_by_outcome, dict) else None
                    price = _safe_float(price_row.get("price")) if isinstance(price_row, dict) else None
                    if not isinstance(outcome, dict) or price is None or price <= 1:
                        continue
                    title = _clean_team_name(outcome.get("title"))
                    if not title:
                        continue
                    stake_value = _safe_float(price_row.get("amount")) if isinstance(price_row, dict) else None
                    row = {"title": title, "price": round(float(price), 6)}
                    if stake_value is not None and stake_value > 0:
                        row["stake"] = round(float(stake_value), 6)
                    if market_updated_at:
                        row["last_updated"] = market_updated_at
                    observed_at = price_row.get("observed_at") if isinstance(price_row, dict) else None
                    if observed_at not in (None, ""):
                        row["observed_at"] = observed_at
                    outcomes.append(row)

                if len(outcomes) < 2:
                    continue

                if target_h2h_key in requested_markets and len(outcomes) == 2:
                    if "HANDICAP" not in market_type and "OVER_UNDER" not in market_type:
                        if "MONEYLINE" in market_type or "FULL_TIME_RESULT" in market_type or "MATCH_RESULT" in market_type or "WINNER" in market_type:
                            titles = {_normalize_token(item["title"]) for item in outcomes}
                            if "draw" not in titles:
                                out_by_token = {_normalize_token(item["title"]): item for item in outcomes}
                                home_out = out_by_token.get(_normalize_token(home_team))
                                away_out = out_by_token.get(_normalize_token(away_team))
                                if home_out and away_out:
                                    h2h = {
                                        "key": target_h2h_key,
                                        "outcomes": [
                                            {
                                                "name": home_team,
                                                "price": home_out["price"],
                                                **({"last_updated": home_out["last_updated"]} if home_out.get("last_updated") else {}),
                                                **({"observed_at": home_out["observed_at"]} if home_out.get("observed_at") not in (None, "") else {}),
                                                **({"stake": home_out["stake"]} if _safe_float(home_out.get("stake")) else {}),
                                            },
                                            {
                                                "name": away_team,
                                                "price": away_out["price"],
                                                **({"last_updated": away_out["last_updated"]} if away_out.get("last_updated") else {}),
                                                **({"observed_at": away_out["observed_at"]} if away_out.get("observed_at") not in (None, "") else {}),
                                                **({"stake": away_out["stake"]} if _safe_float(away_out.get("stake")) else {}),
                                            },
                                        ],
                                    }
                                else:
                                    h2h = {
                                        "key": target_h2h_key,
                                        "outcomes": [
                                            {
                                                "name": outcomes[0]["title"],
                                                "price": outcomes[0]["price"],
                                                **({"last_updated": outcomes[0]["last_updated"]} if outcomes[0].get("last_updated") else {}),
                                                **({"observed_at": outcomes[0]["observed_at"]} if outcomes[0].get("observed_at") not in (None, "") else {}),
                                                **({"stake": outcomes[0]["stake"]} if _safe_float(outcomes[0].get("stake")) else {}),
                                            },
                                            {
                                                "name": outcomes[1]["title"],
                                                "price": outcomes[1]["price"],
                                                **({"last_updated": outcomes[1]["last_updated"]} if outcomes[1].get("last_updated") else {}),
                                                **({"observed_at": outcomes[1]["observed_at"]} if outcomes[1].get("observed_at") not in (None, "") else {}),
                                                **({"stake": outcomes[1]["stake"]} if _safe_float(outcomes[1].get("stake")) else {}),
                                            },
                                        ],
                                    }
                                if best_h2h is None or _score_market(h2h) > _score_market(best_h2h):
                                    best_h2h = h2h

                target_h2h_3_way_key = _scoped_market_key("h2h_3_way", market_type, market_name)
                if target_h2h_3_way_key in requested_markets and len(outcomes) == 3:
                    if "FULL_TIME_RESULT" in market_type or "MATCH_RESULT" in market_type:
                        by_token = {_normalize_token(item["title"]): item for item in outcomes}
                        home_out = by_token.get(_normalize_token(home_team))
                        away_out = by_token.get(_normalize_token(away_team))
                        draw_out = next((item for item in outcomes if _normalize_token(item["title"]) == "draw"), None)
                        if home_out and draw_out and away_out:
                            market_3_way = {
                                "key": target_h2h_3_way_key,
                                "outcomes": [
                                    {
                                        "name": home_team,
                                        "price": home_out["price"],
                                        **({"last_updated": home_out["last_updated"]} if home_out.get("last_updated") else {}),
                                        **({"observed_at": home_out["observed_at"]} if home_out.get("observed_at") not in (None, "") else {}),
                                        **({"stake": home_out["stake"]} if _safe_float(home_out.get("stake")) else {}),
                                    },
                                    {
                                        "name": draw_out["title"],
                                        "price": draw_out["price"],
                                        **({"last_updated": draw_out["last_updated"]} if draw_out.get("last_updated") else {}),
                                        **({"observed_at": draw_out["observed_at"]} if draw_out.get("observed_at") not in (None, "") else {}),
                                        **({"stake": draw_out["stake"]} if _safe_float(draw_out.get("stake")) else {}),
                                    },
                                    {
                                        "name": away_team,
                                        "price": away_out["price"],
                                        **({"last_updated": away_out["last_updated"]} if away_out.get("last_updated") else {}),
                                        **({"observed_at": away_out["observed_at"]} if away_out.get("observed_at") not in (None, "") else {}),
                                        **({"stake": away_out["stake"]} if _safe_float(away_out.get("stake")) else {}),
                                    },
                                ],
                            }
                            sig = _market_signature(market_3_way)
                            prev = by_signature.get(sig)
                            if prev is None or _score_market(market_3_way) > _score_market(prev):
                                by_signature[sig] = market_3_way

                if target_spread_key in requested_markets and len(outcomes) == 2 and "HANDICAP" in market_type:
                    value_a, value_b = _parse_market_value_pair(market.get("marketValue"))
                    spread_outcomes = []
                    for idx, outcome in enumerate(outcomes):
                        name, point = _parse_spread_title(outcome["title"])
                        if point is None:
                            point = value_a if idx == 0 else value_b
                        if point is None:
                            continue
                        spread_outcomes.append(
                            {
                                "name": name or (home_team if idx == 0 else away_team),
                                "price": outcome["price"],
                                "point": round(float(point), 6),
                                **({"last_updated": outcome["last_updated"]} if outcome.get("last_updated") else {}),
                                **({"observed_at": outcome["observed_at"]} if outcome.get("observed_at") not in (None, "") else {}),
                                **({"stake": outcome["stake"]} if _safe_float(outcome.get("stake")) else {}),
                            }
                        )
                    if len(spread_outcomes) == 2:
                        spread = {"key": target_spread_key, "outcomes": spread_outcomes}
                        sig = _market_signature(spread)
                        prev = by_signature.get(sig)
                        if prev is None or _score_market(spread) > _score_market(prev):
                            by_signature[sig] = spread

                if target_total_key in requested_markets and len(outcomes) == 2 and "OVER_UNDER" in market_type:
                    market_total = _parse_market_value_single(market.get("marketValue"))
                    totals = {}
                    for outcome in outcomes:
                        side, point = _parse_total_title(outcome["title"])
                        if not side:
                            continue
                        if point is None:
                            point = market_total
                        if point is None:
                            continue
                        totals[side] = {
                            "name": side,
                            "price": outcome["price"],
                            "point": round(float(point), 6),
                            **({"last_updated": outcome["last_updated"]} if outcome.get("last_updated") else {}),
                            **({"observed_at": outcome["observed_at"]} if outcome.get("observed_at") not in (None, "") else {}),
                            **({"stake": outcome["stake"]} if _safe_float(outcome.get("stake")) else {}),
                        }
                    if "Over" in totals and "Under" in totals:
                        if abs(float(totals["Over"]["point"]) - float(totals["Under"]["point"])) <= 1e-6:
                            total_market = {"key": target_total_key, "outcomes": [totals["Over"], totals["Under"]]}
                            sig = _market_signature(total_market)
                            prev = by_signature.get(sig)
                            if prev is None or _score_market(total_market) > _score_market(prev):
                                by_signature[sig] = total_market

                if len(outcomes) == 2:
                    dynamic_key = None
                    for alias in _market_aliases_for_type(market_type, market_name):
                        if (
                            alias == "h2h"
                            or alias.startswith("h2h_")
                            or alias == "spreads"
                            or alias.startswith("spreads_")
                            or alias == "totals"
                            or alias.startswith("totals_")
                        ):
                            continue
                        if alias in requested_markets:
                            dynamic_key = alias
                            break
                    if dynamic_key:
                        value_a, value_b = _parse_market_value_pair(market.get("marketValue"))
                        shared_value = _parse_market_value_single(market.get("marketValue"))
                        dynamic_outcomes = []
                        for idx, outcome in enumerate(outcomes):
                            row = {
                                "name": outcome["title"],
                                "price": outcome["price"],
                            }
                            if outcome.get("last_updated"):
                                row["last_updated"] = outcome["last_updated"]
                            if outcome.get("observed_at") not in (None, ""):
                                row["observed_at"] = outcome["observed_at"]
                            if _safe_float(outcome.get("stake")):
                                row["stake"] = outcome["stake"]
                            _, title_point = _parse_spread_title(outcome["title"])
                            point = title_point
                            if point is None and value_a is not None and value_b is not None:
                                point = value_a if idx == 0 else value_b
                            if point is None:
                                point = shared_value
                            if point is not None:
                                row["point"] = round(float(point), 6)
                            dynamic_outcomes.append(row)
                        dynamic_market = {"key": dynamic_key, "outcomes": dynamic_outcomes}
                        sig = _market_signature(dynamic_market)
                        prev = by_signature.get(sig)
                        if prev is None or _score_market(dynamic_market) > _score_market(prev):
                            by_signature[sig] = dynamic_market

            market_list: List[dict] = []
            if best_h2h is not None:
                market_list.append(best_h2h)
            market_list.extend(by_signature.values())

            if not market_list:
                continue

            first_market_id = _normalize_text(event_markets[0].get("id")) if event_markets else ""

            stats["events_with_market_count"] += 1
            events_out.append(
                {
                    "id": event_id,
                    "sport_key": sport_key,
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": commence,
                    "live_state": event_live_state,
                    "bookmakers": [
                        {
                            "key": PROVIDER_KEY,
                            "title": PROVIDER_TITLE,
                            "event_id": event_id,
                            "event_url": _event_url(event, event_groups_by_id, first_market_id),
                            "live_state": event_live_state,
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
    "source": BETDEX_SOURCE or "api",
    "back_price_side": _back_price_side(),
    "session_auth_mode": _configured_session_auth_mode(),
    "session_cache": "miss",
    "skipped_unsupported_sport": False,
    "events_payload_count": 0,
    "events_sport_filtered_count": 0,
    "markets_payload_count": 0,
    "prices_payload_count": 0,
    "price_rows_for": 0,
    "price_rows_against": 0,
    "price_rows_unknown": 0,
    "events_with_market_count": 0,
    "events_returned_count": 0,
    "pages_fetched": 0,
    "retries_used": 0,
}
fetch_events_async.last_stats = dict(fetch_events.last_stats)

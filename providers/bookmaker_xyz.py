from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import httpx
import requests

from ._async_http import get_shared_client, request_json, request_text

PROVIDER_KEY = "bookmaker_xyz"
PROVIDER_TITLE = "bookmaker.xyz"

BOOKMAKER_XYZ_SOURCE = os.getenv("BOOKMAKER_XYZ_SOURCE", "api").strip().lower()
BOOKMAKER_XYZ_SAMPLE_PATH = os.getenv(
    "BOOKMAKER_XYZ_SAMPLE_PATH",
    os.path.join("data", "bookmaker_xyz_sample.json"),
).strip()
BOOKMAKER_XYZ_PUBLIC_BASE = os.getenv("BOOKMAKER_XYZ_PUBLIC_BASE", "https://bookmaker.xyz").strip()
BOOKMAKER_XYZ_HOME_URL = os.getenv("BOOKMAKER_XYZ_HOME_URL", "https://bookmaker.xyz/").strip()
BOOKMAKER_XYZ_GRAPH_BASE = os.getenv(
    "BOOKMAKER_XYZ_GRAPH_BASE",
    "https://thegraph-1.onchainfeed.org/subgraphs/name/azuro-protocol",
).strip()
BOOKMAKER_XYZ_MARKET_MANAGER_BASE = os.getenv(
    "BOOKMAKER_XYZ_MARKET_MANAGER_BASE",
    "https://api.onchainfeed.org/api/v1/public/market-manager",
).strip()
BOOKMAKER_XYZ_CHAIN_IDS_RAW = os.getenv("BOOKMAKER_XYZ_CHAIN_IDS", "137,100,8453").strip()
BOOKMAKER_XYZ_PAGE_SIZE_RAW = os.getenv("BOOKMAKER_XYZ_PAGE_SIZE", "400").strip()
BOOKMAKER_XYZ_MAX_PAGES_RAW = os.getenv("BOOKMAKER_XYZ_MAX_PAGES", "6").strip()
BOOKMAKER_XYZ_GAMES_PER_PAGE_RAW = os.getenv("BOOKMAKER_XYZ_GAMES_PER_PAGE", "100").strip()
BOOKMAKER_XYZ_MAX_GAME_PAGES_RAW = os.getenv("BOOKMAKER_XYZ_MAX_GAME_PAGES", "10").strip()
BOOKMAKER_XYZ_CONDITION_BATCH_SIZE_RAW = os.getenv("BOOKMAKER_XYZ_CONDITION_BATCH_SIZE", "50").strip()
BOOKMAKER_XYZ_TIMEOUT_RAW = os.getenv("BOOKMAKER_XYZ_TIMEOUT_SECONDS", "25").strip()
BOOKMAKER_XYZ_RETRIES_RAW = os.getenv("BOOKMAKER_XYZ_RETRIES", "2").strip()
BOOKMAKER_XYZ_RETRY_BACKOFF_RAW = os.getenv("BOOKMAKER_XYZ_RETRY_BACKOFF", "0.5").strip()
BOOKMAKER_XYZ_CACHE_TTL_RAW = os.getenv("BOOKMAKER_XYZ_CACHE_TTL", "12").strip()
BOOKMAKER_XYZ_DICT_CACHE_TTL_RAW = os.getenv("BOOKMAKER_XYZ_DICT_CACHE_TTL", "21600").strip()
BOOKMAKER_XYZ_DICT_DISK_CACHE_CLEANUP_SECONDS_RAW = os.getenv(
    "BOOKMAKER_XYZ_DICT_DISK_CACHE_CLEANUP_SECONDS",
    "300",
).strip()
BOOKMAKER_XYZ_LOOKBACK_SECONDS_RAW = os.getenv("BOOKMAKER_XYZ_LOOKBACK_SECONDS", "7200").strip()
BOOKMAKER_XYZ_DICT_DISK_CACHE_DIR = os.getenv(
    "BOOKMAKER_XYZ_DICT_DISK_CACHE_DIR",
    os.path.join("data", "bookmaker_xyz_dict_cache"),
).strip()
BOOKMAKER_XYZ_DICTIONARY_SOURCE = os.getenv(
    "BOOKMAKER_XYZ_DICTIONARY_SOURCE",
    "auto",
).strip().lower()
BOOKMAKER_XYZ_INCLUDE_LIVE = os.getenv("BOOKMAKER_XYZ_INCLUDE_LIVE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
BOOKMAKER_XYZ_USER_AGENT = os.getenv(
    "BOOKMAKER_XYZ_USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
).strip()

DICT_SKIP_TEAM_PLAYER_MARKET_IDS = {"3", "31"}
AZURO_DICTIONARIES_REGISTRY_URL = "https://registry.npmjs.org/@azuro-org/dictionaries"
AZURO_DICTIONARIES_PACKAGE_ENTRY = "package/lib/index.js"

CHAIN_ID_TO_SLUG = {
    100: "gnosis",
    137: "polygon",
    8453: "base",
}
CHAIN_ID_TO_ENVIRONMENT = {
    100: "GnosisXDAI",
    137: "PolygonUSDT",
    8453: "BaseWETH",
}
ENVIRONMENT_TO_SLUG = {
    environment.strip().lower(): CHAIN_ID_TO_SLUG[chain_id]
    for chain_id, environment in CHAIN_ID_TO_ENVIRONMENT.items()
    if chain_id in CHAIN_ID_TO_SLUG and environment.strip()
}
PUBLIC_COUNTRY_SLUG_OVERRIDES = {
    "united-states": "usa",
}

BOOKMAKER_XYZ_SPORT_FILTERS: Dict[str, Dict[str, str]] = {
    "americanfootball_nfl": {
        "sport_slug": "american-football",
        "league_slug": "nfl",
    },
    "americanfootball_ncaaf": {
        "sport_slug": "american-football",
        "league_slug": "ncaa-football",
    },
    "basketball_nba": {
        "sport_slug": "basketball",
        "league_slug": "nba",
        "country_slug": "united-states",
    },
    "basketball_ncaab": {
        "sport_slug": "basketball",
        "league_slug": "ncaa",
    },
    "basketball_euroleague": {
        "sport_slug": "basketball",
        "league_slug": "euroleague",
        "country_slug": "international-tournaments",
    },
    "basketball_germany_bbl": {
        "sport_slug": "basketball",
        "league_slug": "bbl",
        "country_slug": "germany",
    },
    "basketball_spain_liga_acb": {
        "sport_slug": "basketball",
        "league_slug": "liga-acb",
        "country_slug": "spain",
    },
    "basketball_italy_serie_a": {
        "sport_slug": "basketball",
        "league_slug": "serie-a",
        "country_slug": "italy",
    },
    "basketball_france_pro_a": {
        "sport_slug": "basketball",
        "league_slug": "pro-a",
        "country_slug": "france",
    },
    "baseball_mlb": {
        "sport_slug": "baseball",
        "league_slug": "mlb",
    },
    "baseball_mlb_spring_training": {
        "sport_slug": "baseball",
        "league_slug": "mlb-spring-training",
        "country_slug": "united-states",
    },
    "icehockey_nhl": {
        "sport_slug": "ice-hockey",
        "league_slug": "nhl",
        "country_slug": "united-states",
    },
    "icehockey_khl": {
        "sport_slug": "ice-hockey",
        "league_slug": "khl",
        "country_slug": "russia",
    },
    "icehockey_ahl": {
        "sport_slug": "ice-hockey",
        "league_slug": "ahl",
        "country_slug": "united-states",
    },
    "soccer_epl": {
        "sport_slug": "football",
        "league_slug": "premier-league",
        "country_slug": "england",
    },
    "soccer_england_championship": {
        "sport_slug": "football",
        "league_slug": "championship",
        "country_slug": "england",
    },
    "soccer_england_league_one": {
        "sport_slug": "football",
        "league_slug": "league-one",
        "country_slug": "england",
    },
    "soccer_england_league_two": {
        "sport_slug": "football",
        "league_slug": "league-two",
        "country_slug": "england",
    },
    "soccer_spain_la_liga": {
        "sport_slug": "football",
        "league_slug": "laliga",
        "country_slug": "spain",
    },
    "soccer_germany_bundesliga": {
        "sport_slug": "football",
        "league_slug": "bundesliga",
        "country_slug": "germany",
    },
    "soccer_italy_serie_a": {
        "sport_slug": "football",
        "league_slug": "serie-a",
        "country_slug": "italy",
    },
    "soccer_france_ligue_one": {
        "sport_slug": "football",
        "league_slug": "ligue-1",
        "country_slug": "france",
    },
    "soccer_usa_mls": {
        "sport_slug": "football",
        "league_slug": "mls",
        "country_slug": "united-states",
    },
    "soccer_portugal_primeira_liga": {
        "sport_slug": "football",
        "league_slug": "primeira-liga",
        "country_slug": "portugal",
    },
    "soccer_netherlands_eredivisie": {
        "sport_slug": "football",
        "league_slug": "eredivisie",
        "country_slug": "netherlands",
    },
    "soccer_brazil_serie_a": {
        "sport_slug": "football",
        "league_slug": "brasileirao-serie-a",
        "country_slug": "brazil",
    },
    "soccer_argentina_liga_profesional": {
        "sport_slug": "football",
        "league_slug": "liga-profesional",
        "country_slug": "argentina",
    },
    "soccer_mexico_liga_mx": {
        "sport_slug": "football",
        "league_slug": "liga-mx",
        "country_slug": "mexico",
    },
    "soccer_turkey_super_lig": {
        "sport_slug": "football",
        "league_slug": "super-lig",
        "country_slug": "turkey",
    },
    "mma_ufc": {
        "sport_slug": "mma",
        "league_slug": "ufc",
        "country_slug": "international-tournaments",
    },
    "boxing_professional": {
        "sport_slug": "boxing",
        "league_slug": "professional-boxing",
        "country_slug": "international-tournaments",
    },
    "rugby_union_six_nations": {
        "sport_slug": "rugby-union",
        "league_slug": "six-nations",
        "country_slug": "international-tournaments",
    },
    "rugby_league_nrl": {
        "sport_slug": "rugby-league",
        "league_slug": "national-rugby-league",
        "country_slug": "australia",
    },
    "tennis_atp_indian_wells": {
        "sport_slug": "tennis",
        "league_slug": "indian-wells",
        "country_slug": "united-states",
    },
    "tennis_wta_indian_wells": {
        "sport_slug": "tennis",
        "league_slug": "indian-wells-w",
        "country_slug": "united-states",
    },
}
AZURO_DYNAMIC_SPORT_KEY_PREFIX = "azuro__"

SPORT_SLUG_HINTS: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("american-football",),
    "americanfootball_ncaaf": ("american-football",),
    "basketball_nba": ("basketball",),
    "basketball_ncaab": ("basketball",),
    "baseball_mlb": ("baseball",),
    "icehockey_nhl": ("ice-hockey",),
    "soccer_epl": ("football",),
    "soccer_spain_la_liga": ("football",),
    "soccer_germany_bundesliga": ("football",),
    "soccer_italy_serie_a": ("football",),
    "soccer_france_ligue_one": ("football",),
    "soccer_usa_mls": ("football",),
}

SPORT_LEAGUE_HINTS: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("nfl",),
    "americanfootball_ncaaf": ("ncaaf", "ncaa football", "college football"),
    "basketball_nba": ("nba",),
    "basketball_ncaab": ("ncaa", "college"),
    "baseball_mlb": ("mlb", "major league baseball"),
    "icehockey_nhl": ("nhl", "national hockey league"),
    "soccer_epl": ("premier league", "english premier league", "epl"),
    "soccer_spain_la_liga": ("la liga",),
    "soccer_germany_bundesliga": ("bundesliga",),
    "soccer_italy_serie_a": ("serie a",),
    "soccer_france_ligue_one": ("ligue 1", "ligue one"),
    "soccer_usa_mls": ("mls", "major league soccer"),
}

SPORT_LEAGUE_SLUG_HINTS: Dict[str, Sequence[str]] = {
    "soccer_epl": ("premier-league",),
    "soccer_spain_la_liga": ("la-liga",),
    "soccer_germany_bundesliga": ("bundesliga",),
    "soccer_italy_serie_a": ("serie-a",),
    "soccer_france_ligue_one": ("ligue-1",),
    "soccer_usa_mls": ("major-league-soccer", "mls"),
}

SPORT_COUNTRY_HINTS: Dict[str, Sequence[str]] = {
    "soccer_epl": ("england",),
    "soccer_spain_la_liga": ("spain",),
    "soccer_germany_bundesliga": ("germany",),
    "soccer_italy_serie_a": ("italy",),
    "soccer_france_ligue_one": ("france",),
    "soccer_usa_mls": ("united states", "usa"),
}

GRAPHQL_CONDITIONS_QUERY = """
query Conditions(
  $first: Int
  $skip: Int
  $where: Condition_filter!
  $orderBy: Condition_orderBy
  $orderDirection: OrderDirection
) {
  conditions(
    first: $first
    skip: $skip
    where: $where
    orderBy: $orderBy
    orderDirection: $orderDirection
    subgraphError: allow
  ) {
    id
    conditionId
    state
    turnover
    margin
    isExpressForbidden
    outcomes {
      outcomeId
      title
      currentOdds
      sortOrder
    }
    game {
      id
      gameId
      slug
      title
      startsAt
      state
      sport {
        sportId
        slug
        name
      }
      league {
        slug
        name
      }
      country {
        slug
        name
      }
      participants {
        name
      }
    }
  }
}
""".strip()

CONDITIONS_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "conditions": [],
    "meta": {},
}

DICTIONARY_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "data": None,
    "source": "",
}
DICTIONARY_DISK_CACHE_STATE: Dict[str, float] = {
    "last_cleanup_at": 0.0,
}
SCAN_CACHE_LOCK = threading.RLock()
SCAN_CACHE_CONTEXT: Dict[str, object] = {
    "active": False,
    "dictionaries": None,
    "dictionary_meta": {},
    "conditions": None,
    "conditions_meta": {},
}
SCAN_CACHE_ASYNC_LOCK: Optional[asyncio.Lock] = None


def enable_scan_cache() -> None:
    with SCAN_CACHE_LOCK:
        SCAN_CACHE_CONTEXT["active"] = True
        SCAN_CACHE_CONTEXT["dictionaries"] = None
        SCAN_CACHE_CONTEXT["dictionary_meta"] = {}
        SCAN_CACHE_CONTEXT["conditions"] = None
        SCAN_CACHE_CONTEXT["conditions_meta"] = {}


def disable_scan_cache() -> None:
    with SCAN_CACHE_LOCK:
        SCAN_CACHE_CONTEXT["active"] = False
        SCAN_CACHE_CONTEXT["dictionaries"] = None
        SCAN_CACHE_CONTEXT["dictionary_meta"] = {}
        SCAN_CACHE_CONTEXT["conditions"] = None
        SCAN_CACHE_CONTEXT["conditions_meta"] = {}


def _scan_cache_async_lock() -> asyncio.Lock:
    global SCAN_CACHE_ASYNC_LOCK
    if SCAN_CACHE_ASYNC_LOCK is None:
        SCAN_CACHE_ASYNC_LOCK = asyncio.Lock()
    return SCAN_CACHE_ASYNC_LOCK


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


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_market_key(value: object) -> str:
    token = _normalize_text(value).lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


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
    text = _normalize_text(value)
    if not text:
        return None
    if text.isdigit():
        try:
            ts = int(text)
            return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OSError, OverflowError):
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


def _public_base() -> str:
    base = (BOOKMAKER_XYZ_PUBLIC_BASE or "").strip() or "https://bookmaker.xyz"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _home_url() -> str:
    value = (BOOKMAKER_XYZ_HOME_URL or "").strip() or "https://bookmaker.xyz/"
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = f"https://{value}"
    return value


def _graph_base() -> str:
    base = (BOOKMAKER_XYZ_GRAPH_BASE or "").strip() or (
        "https://thegraph-1.onchainfeed.org/subgraphs/name/azuro-protocol"
    )
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _headers() -> Dict[str, str]:
    headers = {}
    if BOOKMAKER_XYZ_USER_AGENT:
        headers["User-Agent"] = BOOKMAKER_XYZ_USER_AGENT
    return headers


def _retrying_get(url: str, retries: int, backoff_seconds: float, timeout: int) -> Tuple[str, int]:
    retriable = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    last_error: Optional[ProviderError] = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=_headers(), timeout=timeout)
        except requests.RequestException as exc:
            last_error = ProviderError(f"bookmaker.xyz network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if response.status_code >= 400:
            if response.status_code in retriable and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise ProviderError(
                f"bookmaker.xyz request failed ({response.status_code})",
                status_code=response.status_code,
            )
        return response.text, attempt
    if last_error:
        raise last_error
    raise ProviderError("bookmaker.xyz request failed")


def _retrying_get_bytes(url: str, retries: int, backoff_seconds: float, timeout: int) -> Tuple[bytes, int]:
    retriable = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    last_error: Optional[ProviderError] = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=_headers(), timeout=timeout)
        except requests.RequestException as exc:
            last_error = ProviderError(f"bookmaker.xyz network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if response.status_code >= 400:
            if response.status_code in retriable and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise ProviderError(
                f"bookmaker.xyz request failed ({response.status_code})",
                status_code=response.status_code,
            )
        return response.content or b"", attempt
    if last_error:
        raise last_error
    raise ProviderError("bookmaker.xyz request failed")


async def _retrying_get_async(
    client: httpx.AsyncClient,
    url: str,
    retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Tuple[str, int]:
    return await request_text(
        client,
        "GET",
        url,
        headers=_headers(),
        timeout=float(timeout),
        retries=retries,
        backoff_seconds=backoff_seconds,
        error_cls=ProviderError,
        network_error_prefix="bookmaker.xyz network error",
        status_error_message=lambda status_code: f"bookmaker.xyz request failed ({status_code})",
    )


def _retrying_graphql(
    url: str,
    query: str,
    variables: Dict[str, object],
    retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Tuple[dict, int]:
    retriable = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    last_error: Optional[ProviderError] = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                url,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json", "Accept": "application/json", **_headers()},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = ProviderError(f"bookmaker.xyz GraphQL network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if response.status_code >= 400:
            if response.status_code in retriable and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise ProviderError(
                f"bookmaker.xyz GraphQL failed ({response.status_code})",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            last_error = ProviderError("Failed to parse bookmaker.xyz GraphQL response")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if isinstance(payload, dict) and isinstance(payload.get("errors"), list) and payload["errors"]:
            first = payload["errors"][0]
            message = _normalize_text(first.get("message") if isinstance(first, dict) else first)
            raise ProviderError(f"bookmaker.xyz GraphQL error: {message or 'unknown error'}")
        if not isinstance(payload, dict):
            raise ProviderError("bookmaker.xyz GraphQL response is invalid")
        return payload, attempt
    if last_error:
        raise last_error
    raise ProviderError("bookmaker.xyz GraphQL failed")


async def _retrying_graphql_async(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    variables: Dict[str, object],
    retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Tuple[dict, int]:
    payload, attempt = await request_json(
        client,
        "POST",
        url,
        json_payload={"query": query, "variables": variables},
        headers={"Content-Type": "application/json", "Accept": "application/json", **_headers()},
        timeout=float(timeout),
        retries=retries,
        backoff_seconds=backoff_seconds,
        error_cls=ProviderError,
        network_error_prefix="bookmaker.xyz GraphQL network error",
        parse_error_message="Failed to parse bookmaker.xyz GraphQL response",
        status_error_message=lambda status_code: f"bookmaker.xyz GraphQL failed ({status_code})",
    )
    if isinstance(payload, dict) and isinstance(payload.get("errors"), list) and payload["errors"]:
        first = payload["errors"][0]
        message = _normalize_text(first.get("message") if isinstance(first, dict) else first)
        raise ProviderError(f"bookmaker.xyz GraphQL error: {message or 'unknown error'}")
    if not isinstance(payload, dict):
        raise ProviderError("bookmaker.xyz GraphQL response is invalid")
    return payload, attempt


async def _request_market_manager_json_async(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    json_payload: Optional[dict] = None,
    retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Tuple[dict, int]:
    payload, attempt = await request_json(
        client,
        method,
        url,
        params=params,
        json_payload=json_payload,
        headers={"Accept": "application/json", **_headers()},
        timeout=float(timeout),
        retries=retries,
        backoff_seconds=backoff_seconds,
        error_cls=ProviderError,
        network_error_prefix="bookmaker.xyz market-manager network error",
        parse_error_message="Failed to parse bookmaker.xyz market-manager response",
        status_error_message=lambda status_code: f"bookmaker.xyz market-manager request failed ({status_code})",
    )
    if not isinstance(payload, dict):
        raise ProviderError("bookmaker.xyz market-manager response is invalid")
    return payload, attempt


def _parse_chain_ids() -> List[int]:
    out: List[int] = []
    for token in re.split(r"[,\s]+", BOOKMAKER_XYZ_CHAIN_IDS_RAW or ""):
        if not token:
            continue
        try:
            chain_id = int(float(token))
        except (TypeError, ValueError):
            continue
        if chain_id in CHAIN_ID_TO_SLUG and chain_id not in out:
            out.append(chain_id)
    if not out:
        out = [137]
    return out


def _market_manager_base() -> str:
    base = (BOOKMAKER_XYZ_MARKET_MANAGER_BASE or "").strip() or (
        "https://api.onchainfeed.org/api/v1/public/market-manager"
    )
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _market_manager_environments() -> List[str]:
    environments: List[str] = []
    for chain_id in _parse_chain_ids():
        environment = CHAIN_ID_TO_ENVIRONMENT.get(chain_id)
        if environment and environment not in environments:
            environments.append(environment)
    if not environments:
        environments.append("PolygonUSDT")
    return environments


def _parse_dynamic_sport_filter(sport_key: str) -> Optional[Dict[str, str]]:
    key = _normalize_text(sport_key).lower()
    if not key.startswith(AZURO_DYNAMIC_SPORT_KEY_PREFIX):
        return None
    parts = [part for part in key.split("__") if part]
    if len(parts) < 4:
        return None
    _, sport_slug, league_slug, country_slug = parts[:4]
    if not sport_slug or not league_slug or not country_slug:
        return None
    return {
        "sport_slug": sport_slug,
        "league_slug": league_slug,
        "country_slug": country_slug,
    }


def _sport_filter_config(sport_key: str) -> Optional[Dict[str, str]]:
    config = BOOKMAKER_XYZ_SPORT_FILTERS.get(_normalize_text(sport_key).lower())
    if isinstance(config, dict):
        return dict(config)
    return _parse_dynamic_sport_filter(sport_key)


def _game_matches_filter(game: dict, config: Optional[Dict[str, str]]) -> bool:
    if not isinstance(game, dict):
        return False
    if not isinstance(config, dict):
        return True
    sport_slug = _normalize_text((game.get("sport") or {}).get("slug")).lower()
    league_slug = _normalize_text((game.get("league") or {}).get("slug")).lower()
    country_slug = _normalize_text((game.get("country") or {}).get("slug")).lower()
    expected_sport = _normalize_text(config.get("sport_slug")).lower()
    expected_league = _normalize_text(config.get("league_slug")).lower()
    expected_country = _normalize_text(config.get("country_slug")).lower()
    if expected_sport and sport_slug != expected_sport:
        return False
    if expected_league and league_slug != expected_league:
        return False
    if expected_country and country_slug != expected_country:
        return False
    return True


def _chunked_strs(values: Sequence[str], size: int) -> List[List[str]]:
    batch_size = max(1, int(size or 1))
    return [list(values[index : index + batch_size]) for index in range(0, len(values), batch_size)]


def _extract_const_asset_path(home_html: str) -> Optional[str]:
    if not home_html:
        return None
    match = re.search(r"/assets/const-[^\"'\s>]+\.js", home_html)
    if not match:
        return None
    return match.group(0)


def _extract_x_object_literal(js_source: str) -> str:
    marker = "var Y=Ch,X="
    start = js_source.find(marker)
    if start < 0:
        raise ProviderError("Failed to locate bookmaker.xyz dictionaries object marker")
    brace_start = js_source.find("{", start)
    if brace_start < 0:
        raise ProviderError("Failed to locate bookmaker.xyz dictionaries object start")

    i = brace_start
    depth = 0
    in_single = False
    in_double = False
    in_back = False
    escaped = False
    while i < len(js_source):
        ch = js_source[i]
        if in_single:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_single = False
        elif in_double:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
        elif in_back:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "`":
                in_back = False
        else:
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == "`":
                in_back = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return js_source[brace_start : i + 1]
        i += 1
    raise ProviderError("Failed to parse bookmaker.xyz dictionaries object")


def _parse_dictionaries_via_node(x_object_literal: str, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="bookmaker_xyz_dict_") as temp_dir:
        module_path = os.path.join(temp_dir, "x_module.js")
        with open(module_path, "w", encoding="utf-8") as handle:
            handle.write("const X=")
            handle.write(x_object_literal)
            handle.write(";\nmodule.exports = X;\n")
        node_script = (
            "const X=require(process.argv[1]);"
            "const payload={"
            "marketNames:X.marketNames||{},"
            "marketDescriptions:X.marketDescriptions||{},"
            "outcomes:X.outcomes||{},"
            "selections:X.selections||{},"
            "teamPlayers:X.teamPlayers||{},"
            "points:X.points||{}"
            "};"
            "process.stdout.write(JSON.stringify(payload));"
        )
        result = subprocess.run(
            ["node", "-e", node_script, module_path],
            capture_output=True,
            text=True,
            timeout=max(5, timeout),
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise ProviderError(f"Failed to parse bookmaker.xyz dictionaries via node: {stderr}")
        try:
            payload = json.loads(result.stdout or "{}")
        except ValueError as exc:
            raise ProviderError("Failed to decode bookmaker.xyz dictionaries JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderError("bookmaker.xyz dictionaries payload is invalid")
        return payload


def _parse_official_dictionaries_via_node(module_source: str, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="azuro_dict_") as temp_dir:
        module_path = os.path.join(temp_dir, "azuro_dictionaries.js")
        with open(module_path, "w", encoding="utf-8") as handle:
            handle.write(module_source)
        node_script = (
            "const m=require(process.argv[1]);"
            "const dictionaries=m.dictionaries||{};"
            "const outcomes={};"
            "for (const key of Object.keys(dictionaries.outcomes||{})) {"
            "outcomes[key]=dictionaries.outcomes[key];"
            "}"
            "const payload={"
            "marketNames:dictionaries.marketNames||{},"
            "marketDescriptions:dictionaries.marketDescriptions||{},"
            "outcomes,"
            "selections:dictionaries.selections||{},"
            "teamPlayers:dictionaries.teamPlayers||{},"
            "points:dictionaries.points||{}"
            "};"
            "process.stdout.write(JSON.stringify(payload));"
        )
        result = subprocess.run(
            ["node", "-e", node_script, module_path],
            capture_output=True,
            text=True,
            timeout=max(5, timeout),
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise ProviderError(f"Failed to parse Azuro official dictionaries via node: {stderr}")
        try:
            payload = json.loads(result.stdout or "{}")
        except ValueError as exc:
            raise ProviderError("Failed to decode Azuro official dictionaries JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderError("Azuro official dictionaries payload is invalid")
        return payload


def _dictionary_disk_cache_path(source_url: object) -> str:
    base_dir = _normalize_text(BOOKMAKER_XYZ_DICT_DISK_CACHE_DIR)
    if not base_dir:
        return ""
    token = hashlib.sha256(_normalize_text(source_url).encode("utf-8")).hexdigest()[:24]
    if not token:
        return ""
    return os.path.join(base_dir, f"{token}.json")


def _dictionary_disk_cache_ttl_seconds() -> int:
    return _int_or_default(BOOKMAKER_XYZ_DICT_CACHE_TTL_RAW, 21600, min_value=0)


def _dictionary_disk_cache_cleanup_seconds() -> int:
    return _int_or_default(BOOKMAKER_XYZ_DICT_DISK_CACHE_CLEANUP_SECONDS_RAW, 300, min_value=0)


def _prune_dictionary_disk_cache(force: bool = False) -> None:
    base_dir = _normalize_text(BOOKMAKER_XYZ_DICT_DISK_CACHE_DIR)
    if not base_dir or not os.path.isdir(base_dir):
        return
    now = time.time()
    with SCAN_CACHE_LOCK:
        last_cleanup_at = float(DICTIONARY_DISK_CACHE_STATE.get("last_cleanup_at", 0.0) or 0.0)
        if not force and (now - last_cleanup_at) < _dictionary_disk_cache_cleanup_seconds():
            return
        DICTIONARY_DISK_CACHE_STATE["last_cleanup_at"] = now
    ttl = _dictionary_disk_cache_ttl_seconds()
    stale_before = now - ttl if ttl > 0 else now
    try:
        entries = os.listdir(base_dir)
    except OSError:
        return
    for name in entries:
        if not name.endswith(".json"):
            continue
        cache_path = os.path.join(base_dir, name)
        try:
            modified_at = os.path.getmtime(cache_path)
        except OSError:
            continue
        if ttl <= 0 or modified_at < stale_before:
            try:
                os.remove(cache_path)
            except OSError:
                continue


def _load_dictionaries_from_disk_cache(source_url: object) -> Optional[dict]:
    cache_path = _dictionary_disk_cache_path(source_url)
    if not cache_path or not os.path.exists(cache_path):
        return None
    ttl = _dictionary_disk_cache_ttl_seconds()
    if ttl <= 0:
        return None
    _prune_dictionary_disk_cache()
    try:
        modified_at = os.path.getmtime(cache_path)
    except OSError:
        return None
    if (time.time() - modified_at) > ttl:
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    dictionaries = payload.get("dictionaries")
    if not isinstance(dictionaries, dict):
        return None
    outcomes = dictionaries.get("outcomes")
    if not isinstance(outcomes, dict) or not outcomes:
        return None
    return dictionaries


def _persist_dictionaries_to_disk_cache(source_url: object, dictionaries: dict) -> None:
    cache_path = _dictionary_disk_cache_path(source_url)
    if not cache_path or not isinstance(dictionaries, dict):
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        temp_path = f"{cache_path}.tmp"
        payload = {
            "saved_at": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "source_url": _normalize_text(source_url),
            "dictionaries": dictionaries,
        }
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temp_path, cache_path)
        _prune_dictionary_disk_cache()
    except OSError:
        return


def _load_dictionaries(
    retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Tuple[Optional[dict], dict]:
    now = time.time()
    ttl = _dictionary_disk_cache_ttl_seconds()
    cache_valid = ttl > 0 and now < float(DICTIONARY_CACHE.get("expires_at", 0.0))
    cached_data = DICTIONARY_CACHE.get("data")
    if cache_valid and isinstance(cached_data, dict):
        return cached_data, {"cache": "hit", "source": DICTIONARY_CACHE.get("source", "")}

    meta = {"cache": "miss", "source": "", "error": ""}
    retries_used = 0
    source_mode = (BOOKMAKER_XYZ_DICTIONARY_SOURCE or "auto").lower()
    if source_mode in {"auto", "official", "npm", "azuro", "package"}:
        dictionaries, official_meta = _load_dictionaries_from_official_package(
            retries=retries,
            backoff_seconds=backoff_seconds,
            timeout=timeout,
        )
        if isinstance(dictionaries, dict):
            DICTIONARY_CACHE["data"] = dictionaries
            DICTIONARY_CACHE["expires_at"] = now + ttl if ttl > 0 else now
            DICTIONARY_CACHE["source"] = _normalize_text(official_meta.get("source"))
            return dictionaries, official_meta
        if source_mode in {"official", "npm", "azuro", "package"}:
            cached = DICTIONARY_CACHE.get("data")
            if isinstance(cached, dict):
                official_meta.update(
                    {
                        "cache": "stale",
                        "source": DICTIONARY_CACHE.get("source", ""),
                    }
                )
                return cached, official_meta
            return None, official_meta
    try:
        home_html, attempt = _retrying_get(_home_url(), retries, backoff_seconds, timeout)
        retries_used += attempt
        asset_path = _extract_const_asset_path(home_html)
        if not asset_path:
            raise ProviderError("Failed to detect bookmaker.xyz const asset path")
        const_url = _public_base().rstrip("/") + asset_path
        cached_disk = _load_dictionaries_from_disk_cache(const_url)
        if isinstance(cached_disk, dict):
            DICTIONARY_CACHE["data"] = cached_disk
            DICTIONARY_CACHE["expires_at"] = now + ttl if ttl > 0 else now
            DICTIONARY_CACHE["source"] = const_url
            meta.update({"cache": "disk_hit", "source": const_url, "retries_used": retries_used})
            return cached_disk, meta
        const_js, attempt = _retrying_get(const_url, retries, backoff_seconds, timeout)
        retries_used += attempt
        x_object_literal = _extract_x_object_literal(const_js)
        dictionaries = _parse_dictionaries_via_node(x_object_literal, timeout=timeout)
        _persist_dictionaries_to_disk_cache(const_url, dictionaries)
        DICTIONARY_CACHE["data"] = dictionaries
        DICTIONARY_CACHE["expires_at"] = now + ttl if ttl > 0 else now
        DICTIONARY_CACHE["source"] = const_url
        meta.update({"source": const_url, "retries_used": retries_used})
        return dictionaries, meta
    except Exception as exc:
        cached = DICTIONARY_CACHE.get("data")
        if isinstance(cached, dict):
            meta.update(
                {
                    "cache": "stale",
                    "source": DICTIONARY_CACHE.get("source", ""),
                    "error": str(exc),
                    "retries_used": retries_used,
                }
            )
            return cached, meta
        meta.update({"error": str(exc), "retries_used": retries_used})
        return None, meta


async def _load_dictionaries_async(
    client: httpx.AsyncClient,
    retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Tuple[Optional[dict], dict]:
    now = time.time()
    ttl = _dictionary_disk_cache_ttl_seconds()
    cache_valid = ttl > 0 and now < float(DICTIONARY_CACHE.get("expires_at", 0.0))
    cached_data = DICTIONARY_CACHE.get("data")
    if cache_valid and isinstance(cached_data, dict):
        return cached_data, {"cache": "hit", "source": DICTIONARY_CACHE.get("source", "")}

    meta = {"cache": "miss", "source": "", "error": ""}
    retries_used = 0
    source_mode = (BOOKMAKER_XYZ_DICTIONARY_SOURCE or "auto").lower()
    if source_mode in {"auto", "official", "npm", "azuro", "package"}:
        dictionaries, official_meta = await asyncio.to_thread(
            _load_dictionaries_from_official_package,
            retries,
            backoff_seconds,
            timeout,
        )
        if isinstance(dictionaries, dict):
            DICTIONARY_CACHE["data"] = dictionaries
            DICTIONARY_CACHE["expires_at"] = now + ttl if ttl > 0 else now
            DICTIONARY_CACHE["source"] = _normalize_text(official_meta.get("source"))
            return dictionaries, official_meta
        if source_mode in {"official", "npm", "azuro", "package"}:
            cached = DICTIONARY_CACHE.get("data")
            if isinstance(cached, dict):
                official_meta.update(
                    {
                        "cache": "stale",
                        "source": DICTIONARY_CACHE.get("source", ""),
                    }
                )
                return cached, official_meta
            return None, official_meta
    try:
        home_html, attempt = await _retrying_get_async(
            client,
            _home_url(),
            retries,
            backoff_seconds,
            timeout,
        )
        retries_used += attempt
        asset_path = _extract_const_asset_path(home_html)
        if not asset_path:
            raise ProviderError("Failed to detect bookmaker.xyz const asset path")
        const_url = _public_base().rstrip("/") + asset_path
        cached_disk = await asyncio.to_thread(_load_dictionaries_from_disk_cache, const_url)
        if isinstance(cached_disk, dict):
            DICTIONARY_CACHE["data"] = cached_disk
            DICTIONARY_CACHE["expires_at"] = now + ttl if ttl > 0 else now
            DICTIONARY_CACHE["source"] = const_url
            meta.update({"cache": "disk_hit", "source": const_url, "retries_used": retries_used})
            return cached_disk, meta
        const_js, attempt = await _retrying_get_async(
            client,
            const_url,
            retries,
            backoff_seconds,
            timeout,
        )
        retries_used += attempt
        x_object_literal = _extract_x_object_literal(const_js)
        dictionaries = await asyncio.to_thread(
            _parse_dictionaries_via_node,
            x_object_literal,
            timeout,
        )
        await asyncio.to_thread(_persist_dictionaries_to_disk_cache, const_url, dictionaries)
        DICTIONARY_CACHE["data"] = dictionaries
        DICTIONARY_CACHE["expires_at"] = now + ttl if ttl > 0 else now
        DICTIONARY_CACHE["source"] = const_url
        meta.update({"source": const_url, "retries_used": retries_used})
        return dictionaries, meta
    except Exception as exc:
        cached = DICTIONARY_CACHE.get("data")
        if isinstance(cached, dict):
            meta.update(
                {
                    "cache": "stale",
                    "source": DICTIONARY_CACHE.get("source", ""),
                    "error": str(exc),
                    "retries_used": retries_used,
                }
            )
            return cached, meta
        meta.update({"error": str(exc), "retries_used": retries_used})
        return None, meta


def _load_dictionaries_from_official_package(
    retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Tuple[Optional[dict], dict]:
    meta = {"cache": "miss", "source": "", "error": "", "source_strategy": "official_package"}
    retries_used = 0
    try:
        metadata_text, attempt = _retrying_get(
            AZURO_DICTIONARIES_REGISTRY_URL,
            retries,
            backoff_seconds,
            timeout,
        )
        retries_used += attempt
        metadata = json.loads(metadata_text or "{}")
        if not isinstance(metadata, dict):
            raise ProviderError("Azuro dictionaries registry response is invalid")
        latest = _normalize_text((metadata.get("dist-tags") or {}).get("latest"))
        version_payload = (metadata.get("versions") or {}).get(latest)
        if not isinstance(version_payload, dict):
            raise ProviderError("Failed to resolve latest Azuro dictionaries package version")
        tarball_url = _normalize_text((version_payload.get("dist") or {}).get("tarball"))
        if not tarball_url:
            raise ProviderError("Failed to resolve Azuro dictionaries tarball URL")
        meta["package_version"] = latest
        cached_disk = _load_dictionaries_from_disk_cache(tarball_url)
        if isinstance(cached_disk, dict):
            meta.update({"cache": "disk_hit", "source": tarball_url, "retries_used": retries_used})
            return cached_disk, meta
        tarball_bytes, attempt = _retrying_get_bytes(
            tarball_url,
            retries,
            backoff_seconds,
            timeout,
        )
        retries_used += attempt
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as archive:
            member = archive.extractfile(AZURO_DICTIONARIES_PACKAGE_ENTRY)
            if member is None:
                raise ProviderError("Failed to locate Azuro dictionaries package entry")
            module_source = member.read().decode("utf-8")
        dictionaries = _parse_official_dictionaries_via_node(module_source, timeout=timeout)
        _persist_dictionaries_to_disk_cache(tarball_url, dictionaries)
        meta.update({"source": tarball_url, "retries_used": retries_used})
        return dictionaries, meta
    except Exception as exc:
        meta.update({"error": str(exc), "retries_used": retries_used})
        return None, meta


def _graph_endpoint_for_chain(chain_id: int) -> str:
    slug = CHAIN_ID_TO_SLUG.get(chain_id)
    if not slug:
        raise ProviderError(f"Unsupported bookmaker.xyz chain id: {chain_id}")
    return f"{_graph_base()}/azuro-data-feed-{slug}"


async def _load_market_manager_snapshot_async(
    client: httpx.AsyncClient,
    sport_key: str,
    retries: int,
    backoff_seconds: float,
    timeout: int,
) -> Tuple[List[dict], dict]:
    filter_config = _sport_filter_config(sport_key)
    if not isinstance(filter_config, dict):
        return [], {
            "cache": "miss",
            "cache_key": _normalize_text(sport_key),
            "source_strategy": "official_market_manager",
            "environments": [],
            "pages_fetched": 0,
            "requests_made": 0,
            "retries_used": 0,
            "games_matched": 0,
            "conditions_fetched": 0,
        }

    ttl = _int_or_default(BOOKMAKER_XYZ_CACHE_TTL_RAW, 12, min_value=0)
    cache_key = json.dumps(
        {
            "sport_key": _normalize_text(sport_key).lower(),
            "include_live": bool(BOOKMAKER_XYZ_INCLUDE_LIVE),
            "filters": filter_config,
            "environments": _market_manager_environments(),
        },
        sort_keys=True,
    )
    now = time.time()
    cache_valid = ttl > 0 and now < float(CONDITIONS_CACHE.get("expires_at", 0.0))
    cached_conditions = CONDITIONS_CACHE.get("conditions")
    cached_key = _normalize_text(CONDITIONS_CACHE.get("key"))
    if cache_valid and cached_key == cache_key and isinstance(cached_conditions, list):
        meta = dict(CONDITIONS_CACHE.get("meta") or {})
        meta["cache"] = "hit"
        return cached_conditions, meta

    per_page = max(10, _int_or_default(BOOKMAKER_XYZ_GAMES_PER_PAGE_RAW, 100, min_value=10))
    max_pages = _int_or_default(BOOKMAKER_XYZ_MAX_GAME_PAGES_RAW, 10, min_value=1)
    batch_size = _int_or_default(BOOKMAKER_XYZ_CONDITION_BATCH_SIZE_RAW, 50, min_value=1)
    game_states = ["Prematch", "Live"] if BOOKMAKER_XYZ_INCLUDE_LIVE else ["Prematch"]
    base_url = _market_manager_base()
    pages_fetched = 0
    requests_made = 0
    retries_used = 0
    environments_used: List[str] = []
    games_by_environment: Dict[str, Dict[str, dict]] = {}

    for environment in _market_manager_environments():
        environment_games: Dict[str, dict] = {}
        for game_state in game_states:
            for page in range(1, max_pages + 1):
                params = {
                    "environment": environment,
                    "gameState": game_state,
                    "sportSlug": filter_config.get("sport_slug"),
                    "leagueSlug": filter_config.get("league_slug"),
                    "conditionState": "Active",
                    "orderBy": "startsAt",
                    "orderDirection": "asc",
                    "page": page,
                    "perPage": per_page,
                }
                payload, attempt = await _request_market_manager_json_async(
                    client,
                    "GET",
                    f"{base_url}/games-by-filters",
                    params=params,
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                    timeout=timeout,
                )
                pages_fetched += 1
                requests_made += 1
                retries_used += attempt
                games = payload.get("games") or payload.get("items") or []
                if not isinstance(games, list):
                    break
                for raw_game in games:
                    if not isinstance(raw_game, dict):
                        continue
                    if not _game_matches_filter(raw_game, filter_config):
                        continue
                    game_id = _normalize_text(raw_game.get("gameId") or raw_game.get("id"))
                    if not game_id or game_id in environment_games:
                        continue
                    environment_games[game_id] = {
                        **raw_game,
                        "__environment": environment,
                    }
                if len(games) < per_page:
                    break
        if environment_games:
            environments_used.append(environment)
            games_by_environment[environment] = environment_games

    all_conditions: List[dict] = []
    for environment, game_lookup in games_by_environment.items():
        game_ids = list(game_lookup.keys())
        for game_id_batch in _chunked_strs(game_ids, batch_size):
            payload, attempt = await _request_market_manager_json_async(
                client,
                "POST",
                f"{base_url}/conditions-by-game-ids",
                json_payload={
                    "environment": environment,
                    "gameIds": game_id_batch,
                },
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout=timeout,
            )
            requests_made += 1
            retries_used += attempt
            conditions = payload.get("conditions") or []
            if not isinstance(conditions, list):
                continue
            for raw_condition in conditions:
                if not isinstance(raw_condition, dict):
                    continue
                if _normalize_text(raw_condition.get("state")).lower() != "active":
                    continue
                raw_game = raw_condition.get("game") if isinstance(raw_condition.get("game"), dict) else {}
                game_id = _normalize_text((raw_game or {}).get("gameId"))
                game = game_lookup.get(game_id)
                if not isinstance(game, dict):
                    continue
                all_conditions.append(
                    {
                        **raw_condition,
                        "game": dict(game),
                    }
                )

    meta = {
        "cache": "miss",
        "cache_key": _normalize_text(sport_key),
        "source_strategy": "official_market_manager",
        "environments": environments_used,
        "pages_fetched": pages_fetched,
        "requests_made": requests_made,
        "retries_used": retries_used,
        "games_matched": sum(len(item) for item in games_by_environment.values()),
        "conditions_fetched": len(all_conditions),
    }
    CONDITIONS_CACHE["key"] = cache_key
    CONDITIONS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    CONDITIONS_CACHE["conditions"] = all_conditions
    CONDITIONS_CACHE["meta"] = meta
    return all_conditions, meta


def _load_conditions_snapshot(
    retries: int,
    backoff_seconds: float,
    timeout: int,
    page_size: int,
    max_pages: int,
) -> Tuple[List[dict], dict]:
    ttl = _int_or_default(BOOKMAKER_XYZ_CACHE_TTL_RAW, 12, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(CONDITIONS_CACHE.get("expires_at", 0.0))
    cached_conditions = CONDITIONS_CACHE.get("conditions")
    if cache_valid and isinstance(cached_conditions, list):
        meta = dict(CONDITIONS_CACHE.get("meta") or {})
        meta["cache"] = "hit"
        return cached_conditions, meta

    lookback_seconds = _int_or_default(BOOKMAKER_XYZ_LOOKBACK_SECONDS_RAW, 7200, min_value=0)
    min_starts_at = int(time.time()) - lookback_seconds
    game_state_filter = ["Prematch", "Live"] if BOOKMAKER_XYZ_INCLUDE_LIVE else ["Prematch"]
    where_filter = {
        "state": "Active",
        "game_": {
            "state_in": game_state_filter,
            "startsAt_gt": str(min_starts_at),
        },
    }

    all_conditions: List[dict] = []
    total_retries_used = 0
    pages_fetched = 0
    chains_queried: List[int] = []

    for chain_id in _parse_chain_ids():
        endpoint = _graph_endpoint_for_chain(chain_id)
        chains_queried.append(chain_id)
        for page in range(max_pages):
            variables = {
                "first": page_size,
                "skip": page * page_size,
                "where": where_filter,
                "orderBy": "conditionId",
                "orderDirection": "desc",
            }
            payload, retries_used = _retrying_graphql(
                endpoint,
                GRAPHQL_CONDITIONS_QUERY,
                variables,
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout=timeout,
            )
            total_retries_used += retries_used
            pages_fetched += 1
            page_items = (
                payload.get("data", {}).get("conditions")
                if isinstance(payload.get("data"), dict)
                else []
            )
            if not isinstance(page_items, list):
                break
            valid_page_items = [item for item in page_items if isinstance(item, dict)]
            for item in valid_page_items:
                item["__chain_id"] = chain_id
                all_conditions.append(item)
            if len(valid_page_items) < page_size:
                break

    meta = {
        "cache": "miss",
        "pages_fetched": pages_fetched,
        "retries_used": total_retries_used,
        "chains_queried": chains_queried,
    }
    CONDITIONS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    CONDITIONS_CACHE["conditions"] = all_conditions
    CONDITIONS_CACHE["meta"] = meta
    return all_conditions, meta


async def _load_conditions_snapshot_async(
    client: httpx.AsyncClient,
    retries: int,
    backoff_seconds: float,
    timeout: int,
    page_size: int,
    max_pages: int,
) -> Tuple[List[dict], dict]:
    ttl = _int_or_default(BOOKMAKER_XYZ_CACHE_TTL_RAW, 12, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(CONDITIONS_CACHE.get("expires_at", 0.0))
    cached_conditions = CONDITIONS_CACHE.get("conditions")
    if cache_valid and isinstance(cached_conditions, list):
        meta = dict(CONDITIONS_CACHE.get("meta") or {})
        meta["cache"] = "hit"
        return cached_conditions, meta

    lookback_seconds = _int_or_default(BOOKMAKER_XYZ_LOOKBACK_SECONDS_RAW, 7200, min_value=0)
    min_starts_at = int(time.time()) - lookback_seconds
    game_state_filter = ["Prematch", "Live"] if BOOKMAKER_XYZ_INCLUDE_LIVE else ["Prematch"]
    where_filter = {
        "state": "Active",
        "game_": {
            "state_in": game_state_filter,
            "startsAt_gt": str(min_starts_at),
        },
    }

    all_conditions: List[dict] = []
    total_retries_used = 0
    pages_fetched = 0
    chains_queried: List[int] = []

    for chain_id in _parse_chain_ids():
        endpoint = _graph_endpoint_for_chain(chain_id)
        chains_queried.append(chain_id)
        for page in range(max_pages):
            variables = {
                "first": page_size,
                "skip": page * page_size,
                "where": where_filter,
                "orderBy": "conditionId",
                "orderDirection": "desc",
            }
            payload, retries_used = await _retrying_graphql_async(
                client,
                endpoint,
                GRAPHQL_CONDITIONS_QUERY,
                variables,
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout=timeout,
            )
            total_retries_used += retries_used
            pages_fetched += 1
            page_items = (
                payload.get("data", {}).get("conditions")
                if isinstance(payload.get("data"), dict)
                else []
            )
            if not isinstance(page_items, list):
                break
            valid_page_items = [item for item in page_items if isinstance(item, dict)]
            for item in valid_page_items:
                item["__chain_id"] = chain_id
                all_conditions.append(item)
            if len(valid_page_items) < page_size:
                break

    meta = {
        "cache": "miss",
        "pages_fetched": pages_fetched,
        "retries_used": total_retries_used,
        "chains_queried": chains_queried,
    }
    CONDITIONS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    CONDITIONS_CACHE["conditions"] = all_conditions
    CONDITIONS_CACHE["meta"] = meta
    return all_conditions, meta


def _event_url(game: dict) -> str:
    base = _public_base()
    environment = _normalize_text(game.get("__environment") or game.get("environment")).lower()
    environment_slug = ENVIRONMENT_TO_SLUG.get(environment, environment if environment in CHAIN_ID_TO_SLUG.values() else "")
    sport_slug = _normalize_text(((game.get("sport") if isinstance(game, dict) else {}) or {}).get("slug"))
    country_slug = _normalize_text(((game.get("country") if isinstance(game, dict) else {}) or {}).get("slug"))
    country_slug = PUBLIC_COUNTRY_SLUG_OVERRIDES.get(country_slug.lower(), country_slug) if country_slug else ""
    league_slug = _normalize_text(((game.get("league") if isinstance(game, dict) else {}) or {}).get("slug"))
    game_id = _normalize_text(game.get("gameId"))
    if environment_slug and sport_slug and league_slug and game_id:
        path_parts = [environment_slug, "sports", sport_slug]
        if country_slug:
            path_parts.append(country_slug)
        path_parts.extend([league_slug, game_id])
        return f"{base}/{'/'.join(quote(part, safe='') for part in path_parts)}"
    slug = _normalize_text(game.get("slug"))
    if slug:
        return f"{base}/sports/{quote(slug, safe='')}"
    if game_id:
        return f"{base}/games/{quote(game_id, safe='')}"
    return base


def _sport_matches_requested(sport_key: str, game: dict) -> bool:
    filter_config = _sport_filter_config(sport_key)
    if isinstance(filter_config, dict):
        return _game_matches_filter(game, filter_config)
    sport = game.get("sport") if isinstance(game, dict) else {}
    league = game.get("league") if isinstance(game, dict) else {}
    country = game.get("country") if isinstance(game, dict) else {}
    sport_slug = _normalize_text((sport or {}).get("slug")).lower()
    sport_name = _normalize_text((sport or {}).get("name")).lower()
    league_slug = _normalize_text((league or {}).get("slug")).lower()
    league_name = _normalize_text((league or {}).get("name")).lower()
    country_slug = _normalize_text((country or {}).get("slug")).lower()
    country_name = _normalize_text((country or {}).get("name")).lower()
    game_title = _normalize_text(game.get("title")).lower()

    slug_hints = SPORT_SLUG_HINTS.get(sport_key) or ()
    if slug_hints and not any(
        hint == sport_slug or hint == sport_name or hint in sport_slug or hint in sport_name
        for hint in slug_hints
    ):
        return False
    league_slug_hints = SPORT_LEAGUE_SLUG_HINTS.get(sport_key) or ()
    if league_slug_hints and not any(
        hint == league_slug or hint in league_slug
        for hint in league_slug_hints
    ):
        return False
    country_hints = SPORT_COUNTRY_HINTS.get(sport_key) or ()
    country_haystack = " ".join(part for part in (country_name, country_slug) if part)
    if country_hints and not any(hint in country_haystack for hint in country_hints):
        return False
    league_hints = SPORT_LEAGUE_HINTS.get(sport_key) or ()
    if not league_hints:
        return True
    haystack = " ".join(part for part in (league_slug, league_name, country_name, country_slug, game_title) if part)
    return any(hint in haystack for hint in league_hints)


def _dict_get(mapping: dict, key: object) -> object:
    if not isinstance(mapping, dict):
        return None
    if key in mapping:
        return mapping[key]
    key_text = _normalize_text(key)
    if key_text in mapping:
        return mapping[key_text]
    try:
        key_int = int(float(key_text))
    except (TypeError, ValueError):
        return None
    if key_int in mapping:
        return mapping[key_int]
    return mapping.get(str(key_int))


def _market_signature(market: dict) -> str:
    key = _normalize_text(market.get("key"))
    outcomes = market.get("outcomes") if isinstance(market.get("outcomes"), list) else []
    parts = [key]
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        parts.append(_normalize_text(outcome.get("name")))
        point = outcome.get("point")
        if point is not None:
            parts.append(str(point))
    return "|".join(parts)


def _market_score(market: dict) -> float:
    outcomes = market.get("outcomes") if isinstance(market.get("outcomes"), list) else []
    score = 0.0
    for outcome in outcomes:
        price = _safe_float(outcome.get("price"))
        if price and price > 1:
            score += price
    return score


def _condition_market_aliases(market_meta: dict) -> List[str]:
    aliases: List[str] = []
    market_id = _normalize_text(market_meta.get("market_id"))
    market_name = _normalize_text(market_meta.get("market_name"))
    market_key = _normalize_text(market_meta.get("market_key"))

    for token in (market_key, market_name, market_id):
        key = _normalize_market_key(token)
        if key:
            aliases.append(key)

    market_name_lc = market_name.lower()
    if market_id in {"1", "19", "20", "21"} or any(
        token in market_name_lc for token in ("winner", "full time result", "match winner")
    ):
        aliases.append("h2h")
    if "handicap" in market_name_lc:
        aliases.append("spreads")
    if any(token in market_name_lc for token in ("total", "over/under", "over under")):
        aliases.append("totals")
    if "both teams to score" in market_name_lc or "btts" in market_name_lc:
        aliases.extend(["btts", "both_teams_to_score"])

    out: List[str] = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            out.append(alias)
            seen.add(alias)
    return out


def _looks_segmented_market_label(value: object) -> bool:
    label = _normalize_text(value).lower()
    if not label:
        return False
    return any(
        token in label
        for token in ("half", "quarter", "period", "inning", "set", "map", "game:")
    )


def _condition_is_segmented(condition: dict) -> bool:
    for key in ("gamePeriodId", "period", "periodId", "gamePeriod"):
        raw = _normalize_text(condition.get(key))
        if raw and raw != "1":
            return True
    for key in ("marketName", "name", "title", "conditionName", "condition"):
        if _looks_segmented_market_label(condition.get(key)):
            return True
    market = condition.get("market")
    if isinstance(market, dict):
        for key in ("name", "marketName", "label", "type"):
            if _looks_segmented_market_label(market.get(key)):
                return True
        raw_period = _normalize_text(
            market.get("gamePeriodId") or market.get("period") or market.get("periodId")
        )
        if raw_period and raw_period != "1":
            return True
    return False


def _selection_alias_tokens(value: object) -> set[str]:
    token = _normalize_text(value).strip().lower()
    if not token:
        return set()
    aliases = {token}
    compact = token.replace(" ", "")
    aliases.add(compact)
    if token in {"1", "team 1", "team1", "h1", "home", "home team"}:
        aliases.update({"1", "team1", "h1", "home"})
    if token in {"2", "team 2", "team2", "h2", "away", "away team"}:
        aliases.update({"2", "team2", "h2", "away"})
    if token in {"x", "draw"}:
        aliases.update({"x", "draw"})
    if token in {"over", "o"}:
        aliases.update({"over", "o"})
    if token in {"under", "u"}:
        aliases.update({"under", "u"})
    return aliases


def _find_selection(parsed_outcomes: Sequence[dict], aliases: set[str]) -> Optional[dict]:
    for item in parsed_outcomes:
        label_aliases = _selection_alias_tokens(item.get("selection_label"))
        if label_aliases.intersection(aliases):
            return item
    return None


def _normalize_condition_market(
    condition: dict,
    home_team: str,
    away_team: str,
    requested_markets: set,
    dictionaries: Optional[dict],
) -> Optional[dict]:
    if not dictionaries:
        return None
    outcomes_dict = dictionaries.get("outcomes") if isinstance(dictionaries, dict) else {}
    market_names = dictionaries.get("marketNames") if isinstance(dictionaries, dict) else {}
    selections = dictionaries.get("selections") if isinstance(dictionaries, dict) else {}
    points = dictionaries.get("points") if isinstance(dictionaries, dict) else {}
    team_players = dictionaries.get("teamPlayers") if isinstance(dictionaries, dict) else {}

    raw_outcomes = condition.get("outcomes")
    if not isinstance(raw_outcomes, list) or len(raw_outcomes) < 2:
        return None

    parsed_outcomes: List[dict] = []
    market_meta: Optional[dict] = None
    for index, raw_outcome in enumerate(raw_outcomes):
        if not isinstance(raw_outcome, dict):
            continue
        outcome_id = _normalize_text(raw_outcome.get("outcomeId"))
        price = _safe_float(raw_outcome.get("currentOdds"))
        if price is None:
            price = _safe_float(raw_outcome.get("odds"))
        if not outcome_id or price is None or price <= 1:
            continue
        outcome_meta = _dict_get(outcomes_dict, outcome_id)
        if not isinstance(outcome_meta, dict):
            continue
        market_id = _normalize_text(outcome_meta.get("marketId"))
        game_period_id = _normalize_text(outcome_meta.get("gamePeriodId"))
        game_type_id = _normalize_text(outcome_meta.get("gameTypeId"))
        team_player_id = _normalize_text(outcome_meta.get("teamPlayerId"))
        selection_id = _normalize_text(outcome_meta.get("selectionId"))
        points_id = _normalize_text(outcome_meta.get("pointsId"))

        market_parts = [market_id, game_period_id, game_type_id]
        if team_player_id and market_id not in DICT_SKIP_TEAM_PLAYER_MARKET_IDS:
            market_parts.append(team_player_id)
        market_key = "-".join(part for part in market_parts if part)
        market_name = _normalize_text(_dict_get(market_names, market_key))

        selection_label = _normalize_text(_dict_get(selections, selection_id))
        if not selection_label:
            selection_label = _normalize_text(raw_outcome.get("title"))
        base_selection_label = selection_label
        team_player_label = _normalize_text(_dict_get(team_players, team_player_id))
        if team_player_label and market_id in DICT_SKIP_TEAM_PLAYER_MARKET_IDS:
            selection_label = team_player_label
        point_label = _normalize_text(_dict_get(points, points_id))
        selection_name = selection_label
        if point_label:
            selection_name = f"{selection_name} ({point_label})" if selection_name else f"({point_label})"

        point_value = _safe_float(point_label)
        sort_order = _safe_float(raw_outcome.get("sortOrder"))
        if sort_order is None:
            sort_order = float(index)
        parsed_outcomes.append(
            {
                "outcome_id": outcome_id,
                "price": round(float(price), 6),
                "selection_label": selection_label,
                "selection_base_label": base_selection_label,
                "selection_name": selection_name,
                "point": point_value,
                "sort_order": sort_order,
            }
        )
        if market_meta is None:
            market_meta = {
                "market_id": market_id,
                "game_period_id": game_period_id,
                "game_type_id": game_type_id,
                "market_key": market_key,
                "market_name": market_name,
            }

    if market_meta is None or len(parsed_outcomes) < 2:
        return None
    game_period_id = _normalize_text(market_meta.get("game_period_id"))
    if game_period_id not in {"1", "76"}:
        return None

    parsed_outcomes.sort(
        key=lambda item: (
            item.get("sort_order") if item.get("sort_order") is not None else 9999.0,
            item.get("selection_label") or "",
        )
    )

    home_selection = _find_selection(parsed_outcomes, {"1", "team1", "h1", "home"})
    away_selection = _find_selection(parsed_outcomes, {"2", "team2", "h2", "away"})
    over_selection = _find_selection(parsed_outcomes, {"over", "o"})
    under_selection = _find_selection(parsed_outcomes, {"under", "u"})
    market_name = _normalize_text(market_meta.get("market_name")).lower()
    market_id = _normalize_text(market_meta.get("market_id"))

    if "h2h" in requested_markets and home_selection and away_selection and len(parsed_outcomes) == 2:
        looks_like_winner = (
            market_id in {"1", "19", "20", "21"}
            or "winner" in market_name
            or "full time result" in market_name
            or "match winner" in market_name
        )
        is_segment = _looks_segmented_market_label(market_name)
        if looks_like_winner and not is_segment:
            return {
                "key": "h2h",
                "outcomes": [
                    {"name": home_team, "price": home_selection["price"]},
                    {"name": away_team, "price": away_selection["price"]},
                ],
            }

    if "spreads" in requested_markets and home_selection and away_selection and len(parsed_outcomes) == 2 and "handicap" in market_name:
        point_1 = home_selection.get("point")
        point_2 = away_selection.get("point")
        if point_1 is not None and point_2 is not None:
            return {
                "key": "spreads",
                "outcomes": [
                    {"name": home_team, "price": home_selection["price"], "point": round(float(point_1), 6)},
                    {"name": away_team, "price": away_selection["price"], "point": round(float(point_2), 6)},
                ],
            }

    if "totals" in requested_markets and over_selection and under_selection and len(parsed_outcomes) == 2:
        if "total" in market_name:
            over_point = over_selection.get("point")
            under_point = under_selection.get("point")
            if over_point is not None and under_point is not None and abs(float(over_point) - float(under_point)) <= 1e-6:
                point = round(float(over_point), 6)
                return {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": over_selection["price"], "point": point},
                        {"name": "Under", "price": under_selection["price"], "point": point},
                    ],
                }

    dynamic_key = None
    for alias in _condition_market_aliases(market_meta):
        if alias in {"h2h", "spreads", "totals"}:
            continue
        if alias in requested_markets:
            dynamic_key = alias
            break
    if dynamic_key and len(parsed_outcomes) == 2:
        if dynamic_key == "btts":
            dynamic_key = (
                "both_teams_to_score" if "both_teams_to_score" in requested_markets else "btts"
            )
        dynamic_outcomes = []
        for index, item in enumerate(parsed_outcomes[:2]):
            name = _normalize_text(
                item.get("selection_base_label")
                or item.get("selection_label")
                or item.get("selection_name")
            )
            if not name:
                name = f"Outcome {index + 1}"
            row = {"name": name, "price": item["price"]}
            point = item.get("point")
            if point is not None:
                row["point"] = round(float(point), 6)
            if market_name:
                row["description"] = _normalize_text(market_meta.get("market_name"))
            dynamic_outcomes.append(row)
        return {"key": dynamic_key, "outcomes": dynamic_outcomes}

    return None


def _fallback_h2h_market(condition: dict, home_team: str, away_team: str) -> Optional[dict]:
    if _condition_is_segmented(condition):
        return None
    outcomes = condition.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return None
    parsed = []
    for index, raw in enumerate(outcomes):
        if not isinstance(raw, dict):
            return None
        price = _safe_float(raw.get("currentOdds"))
        if price is None:
            price = _safe_float(raw.get("odds"))
        if price is None or price <= 1:
            return None
        sort_order = _safe_float(raw.get("sortOrder"))
        if sort_order is None:
            sort_order = float(index)
        parsed.append({"price": round(float(price), 6), "sort_order": sort_order})
    parsed.sort(key=lambda item: item.get("sort_order") if item.get("sort_order") is not None else 9999.0)
    return {
        "key": "h2h",
        "outcomes": [
            {"name": home_team, "price": parsed[0]["price"]},
            {"name": away_team, "price": parsed[1]["price"]},
        ],
    }


def _turnover_value(condition: dict) -> float:
    turnover = _safe_float(condition.get("turnover"))
    if turnover is not None:
        return turnover
    return 0.0


def _normalize_snapshot_to_events(
    conditions: Sequence[dict],
    sport_key: str,
    requested_markets: set,
    dictionaries: Optional[dict],
) -> Tuple[List[dict], dict]:
    stats = {
        "events_sport_filtered_count": 0,
        "conditions_sport_filtered_count": 0,
        "events_with_market_count": 0,
        "fallback_h2h_used_count": 0,
        "dictionary_market_count": 0,
    }
    events_by_id: Dict[str, dict] = {}

    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        game = condition.get("game")
        if not isinstance(game, dict):
            continue
        if not _sport_matches_requested(sport_key, game):
            continue
        stats["conditions_sport_filtered_count"] += 1

        participants = game.get("participants") if isinstance(game.get("participants"), list) else []
        if len(participants) < 2:
            continue
        home_team = _normalize_text(participants[0].get("name") if isinstance(participants[0], dict) else "")
        away_team = _normalize_text(participants[1].get("name") if isinstance(participants[1], dict) else "")
        if not home_team or not away_team:
            continue

        game_id = _normalize_text(game.get("gameId") or game.get("id"))
        if not game_id:
            continue
        chain_id = _normalize_text(condition.get("__chain_id"))
        event_id = f"{chain_id}:{game_id}" if chain_id else game_id
        commence = _normalize_commence_time(game.get("startsAt"))
        if not commence:
            continue

        entry = events_by_id.get(event_id)
        if not entry:
            entry = {
                "id": event_id,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence,
                "event_url": _event_url(game),
                "markets_by_sig": {},
                "fallback_candidate": None,
            }
            events_by_id[event_id] = entry
            stats["events_sport_filtered_count"] += 1

        market = _normalize_condition_market(
            condition,
            home_team=home_team,
            away_team=away_team,
            requested_markets=requested_markets,
            dictionaries=dictionaries,
        )
        if market:
            sig = _market_signature(market)
            previous = entry["markets_by_sig"].get(sig)
            if previous is None or _market_score(market) > _market_score(previous):
                entry["markets_by_sig"][sig] = market
            stats["dictionary_market_count"] += 1
        else:
            fallback_market = _fallback_h2h_market(condition, home_team, away_team)
            if fallback_market and "h2h" in requested_markets:
                current = entry.get("fallback_candidate")
                turnover = _turnover_value(condition)
                if current is None or turnover > float(current.get("turnover") or 0.0):
                    entry["fallback_candidate"] = {
                        "turnover": turnover,
                        "market": fallback_market,
                    }

    events_out: List[dict] = []
    for event in events_by_id.values():
        market_list = list(event.get("markets_by_sig", {}).values())
        if not market_list:
            fallback_candidate = event.get("fallback_candidate")
            if isinstance(fallback_candidate, dict) and isinstance(fallback_candidate.get("market"), dict):
                market_list.append(fallback_candidate["market"])
                stats["fallback_h2h_used_count"] += 1
        if not market_list:
            continue
        stats["events_with_market_count"] += 1
        events_out.append(
            {
                "id": event["id"],
                "sport_key": sport_key,
                "home_team": event["home_team"],
                "away_team": event["away_team"],
                "commence_time": event["commence_time"],
                "bookmakers": [
                    {
                        "key": PROVIDER_KEY,
                        "title": PROVIDER_TITLE,
                        "event_id": event["id"],
                        "event_url": event.get("event_url"),
                        "markets": market_list,
                    }
                ],
            }
        )
    return events_out, stats


def _load_file_events(path: str) -> List[dict]:
    if not path:
        raise ProviderError("BOOKMAKER_XYZ_SAMPLE_PATH is empty")
    if not os.path.exists(path):
        raise ProviderError(f"bookmaker.xyz sample file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ProviderError(f"Failed to read bookmaker.xyz sample file: {exc}") from exc
    if not isinstance(payload, list):
        raise ProviderError("bookmaker.xyz sample payload must be a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def _set_last_stats(stats: dict) -> None:
    fetch_events.last_stats = stats
    fetch_events_async.last_stats = stats


async def fetch_events_async(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    _ = regions  # Reserved for future region-specific routing.
    stats = {
        "provider": PROVIDER_KEY,
        "source": BOOKMAKER_XYZ_SOURCE or "api",
        "events_payload_count": 0,
        "events_sport_filtered_count": 0,
        "conditions_sport_filtered_count": 0,
        "events_with_market_count": 0,
        "dictionary_market_count": 0,
        "fallback_h2h_used_count": 0,
        "events_returned_count": 0,
        "pages_fetched": 0,
        "retries_used": 0,
        "payload_cache": "miss",
        "dictionary_cache": "miss",
        "dictionary_loaded": False,
    }
    _set_last_stats(stats)

    requested_markets = _requested_market_keys(markets)
    if not requested_markets:
        return []

    if bookmakers:
        lowered = {str(book).strip().lower() for book in bookmakers if isinstance(book, str)}
        if PROVIDER_KEY not in lowered and PROVIDER_TITLE.lower() not in lowered:
            return []

    if (BOOKMAKER_XYZ_SOURCE or "api").lower() == "file":
        events = _load_file_events(BOOKMAKER_XYZ_SAMPLE_PATH)
        stats["events_payload_count"] = len(events)
        stats["events_returned_count"] = len(events)
        _set_last_stats(stats)
        return events

    if (BOOKMAKER_XYZ_SOURCE or "api").lower() != "api":
        raise ProviderError("bookmaker.xyz provider supports BOOKMAKER_XYZ_SOURCE=api or file")

    retries = _int_or_default(BOOKMAKER_XYZ_RETRIES_RAW, 2, min_value=0)
    backoff = _float_or_default(BOOKMAKER_XYZ_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    timeout = _int_or_default(BOOKMAKER_XYZ_TIMEOUT_RAW, 25, min_value=1)

    client = await get_shared_client(PROVIDER_KEY, timeout=float(timeout), follow_redirects=True)
    if bool(SCAN_CACHE_CONTEXT.get("active")):
        with SCAN_CACHE_LOCK:
            cached_dictionaries = SCAN_CACHE_CONTEXT.get("dictionaries")
            cached_dictionary_meta = dict(SCAN_CACHE_CONTEXT.get("dictionary_meta") or {})
        if isinstance(cached_dictionaries, dict):
            dictionaries = cached_dictionaries
            dictionary_meta = {
                **cached_dictionary_meta,
                "cache": "scan_hit",
            }
        else:
            async with _scan_cache_async_lock():
                with SCAN_CACHE_LOCK:
                    cached_dictionaries = SCAN_CACHE_CONTEXT.get("dictionaries")
                    cached_dictionary_meta = dict(SCAN_CACHE_CONTEXT.get("dictionary_meta") or {})
                if isinstance(cached_dictionaries, dict):
                    dictionaries = cached_dictionaries
                    dictionary_meta = {
                        **cached_dictionary_meta,
                        "cache": "scan_hit",
                    }
                else:
                    dictionaries, dictionary_meta = await _load_dictionaries_async(
                        client=client,
                        retries=retries,
                        backoff_seconds=backoff,
                        timeout=timeout,
                    )
                    if True:
                        with SCAN_CACHE_LOCK:
                            SCAN_CACHE_CONTEXT["dictionaries"] = dictionaries
                            SCAN_CACHE_CONTEXT["dictionary_meta"] = dict(dictionary_meta or {})
    else:
        dictionaries, dictionary_meta = await _load_dictionaries_async(
            client=client,
            retries=retries,
            backoff_seconds=backoff,
            timeout=timeout,
        )
    if True:
        stats["dictionary_cache"] = _normalize_text(dictionary_meta.get("cache") or "miss")
        stats["dictionary_source"] = _normalize_text(dictionary_meta.get("source"))
        stats["dictionary_error"] = _normalize_text(dictionary_meta.get("error"))
        stats["retries_used"] += int(dictionary_meta.get("retries_used", 0) or 0)
        stats["dictionary_loaded"] = isinstance(dictionaries, dict)

        if bool(SCAN_CACHE_CONTEXT.get("active")):
            with SCAN_CACHE_LOCK:
                cached_conditions = SCAN_CACHE_CONTEXT.get("conditions")
                cached_conditions_meta = dict(SCAN_CACHE_CONTEXT.get("conditions_meta") or {})
            if isinstance(cached_conditions, list) and cached_conditions_meta.get("cache_key") == sport_key:
                conditions = cached_conditions
                payload_meta = {
                    **cached_conditions_meta,
                    "cache": "scan_hit",
                }
            else:
                async with _scan_cache_async_lock():
                    with SCAN_CACHE_LOCK:
                        cached_conditions = SCAN_CACHE_CONTEXT.get("conditions")
                        cached_conditions_meta = dict(SCAN_CACHE_CONTEXT.get("conditions_meta") or {})
                    if isinstance(cached_conditions, list) and cached_conditions_meta.get("cache_key") == sport_key:
                        conditions = cached_conditions
                        payload_meta = {
                            **cached_conditions_meta,
                            "cache": "scan_hit",
                        }
                    else:
                        conditions, payload_meta = await _load_market_manager_snapshot_async(
                            client=client,
                            sport_key=sport_key,
                            retries=retries,
                            backoff_seconds=backoff,
                            timeout=timeout,
                        )
                        with SCAN_CACHE_LOCK:
                            SCAN_CACHE_CONTEXT["conditions"] = conditions
                            SCAN_CACHE_CONTEXT["conditions_meta"] = dict(payload_meta or {})
        else:
            conditions, payload_meta = await _load_market_manager_snapshot_async(
                client=client,
                sport_key=sport_key,
                retries=retries,
                backoff_seconds=backoff,
                timeout=timeout,
            )
    stats["payload_cache"] = _normalize_text(payload_meta.get("cache") or "miss")
    stats["pages_fetched"] = int(payload_meta.get("pages_fetched", 0) or 0)
    stats["requests_made"] = int(payload_meta.get("requests_made", 0) or 0)
    stats["retries_used"] += int(payload_meta.get("retries_used", 0) or 0)
    stats["environments"] = payload_meta.get("environments", [])
    stats["source_strategy"] = _normalize_text(payload_meta.get("source_strategy"))
    stats["events_payload_count"] = len(conditions)

    events_out, normalize_stats = _normalize_snapshot_to_events(
        conditions=conditions,
        sport_key=sport_key,
        requested_markets=requested_markets,
        dictionaries=dictionaries,
    )
    stats.update(normalize_stats)
    stats["events_returned_count"] = len(events_out)
    _set_last_stats(stats)
    return events_out


def fetch_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
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
            )
        )
    raise RuntimeError("fetch_events() cannot be used inside an active event loop; use await fetch_events_async()")


fetch_events.last_stats = {
    "provider": PROVIDER_KEY,
    "source": BOOKMAKER_XYZ_SOURCE or "api",
    "events_returned_count": 0,
}
fetch_events_async.last_stats = dict(fetch_events.last_stats)


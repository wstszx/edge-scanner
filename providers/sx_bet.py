from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
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
    342: "spreads",
}
ORDERS_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "entries": {},
}
LEAGUES_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "entries": {},
}


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


def _as_int(value: object) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fixture_source_mode() -> str:
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
    aliases: List[str] = []
    period_hints = (raw_type, raw_name, type_hint)

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

    odds_one = _moneyline_decimal_from_summary(market.get("bestOddsOutcomeOne"))
    odds_two = _moneyline_decimal_from_summary(market.get("bestOddsOutcomeTwo"))
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
        "description": description,
        "outcome_one_name": outcome_one_label,
        "outcome_two_name": outcome_two_label,
        "outcome_one_point": None,
        "outcome_two_point": None,
    }

    if target_market_base == "h2h":
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


def _event_url(event_id: object) -> str:
    raw = _normalize_text(event_id)
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
                "leagueLabel": row.get("leagueLabel"),
                "leagueId": row.get("leagueId"),
                "markets": [],
            },
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
) -> Tuple[List[dict], dict]:
    source_mode = _fixture_source_mode()
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
) -> Tuple[List[dict], dict]:
    source_mode = _fixture_source_mode()
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
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], int, dict]:
    ttl = _int_or_default(SX_BET_ODDS_CACHE_TTL_RAW, 4, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(ODDS_CACHE.get("expires_at", 0.0))
    cache_entries = ODDS_CACHE.get("entries") if isinstance(ODDS_CACHE.get("entries"), dict) else {}
    odds_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    unresolved: List[str] = []
    for market_hash in market_hashes:
        key = f"{base_token}:{market_hash}"
        if cache_valid and key in cache_entries:
            odds_map[market_hash] = cache_entries[key]
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
            odds_one = _moneyline_decimal_from_outcome_payload(out_one)
            odds_two = _moneyline_decimal_from_outcome_payload(out_two)
            odds_map[market_hash] = (odds_one, odds_two)
            if odds_one is None and odds_two is None:
                lookup_meta["best_odds_null_count"] += 1
            elif odds_one is not None and odds_two is not None:
                lookup_meta["best_odds_with_both_odds"] += 1
                lookup_meta["best_odds_with_any_odds"] += 1
            else:
                lookup_meta["best_odds_with_any_odds"] += 1
            cache_entries[f"{base_token}:{market_hash}"] = (odds_one, odds_two)

    ODDS_CACHE["entries"] = cache_entries
    ODDS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    return odds_map, retries_used_total, lookup_meta


async def _load_best_odds_map_async(
    client: httpx.AsyncClient,
    market_hashes: Sequence[str],
    base_token: str,
    retries: int,
    backoff_seconds: float,
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], int, dict]:
    ttl = _int_or_default(SX_BET_ODDS_CACHE_TTL_RAW, 4, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(ODDS_CACHE.get("expires_at", 0.0))
    cache_entries = ODDS_CACHE.get("entries") if isinstance(ODDS_CACHE.get("entries"), dict) else {}
    odds_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    unresolved: List[str] = []
    for market_hash in market_hashes:
        key = f"{base_token}:{market_hash}"
        if cache_valid and key in cache_entries:
            odds_map[market_hash] = cache_entries[key]
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
            odds_one = _moneyline_decimal_from_outcome_payload(out_one)
            odds_two = _moneyline_decimal_from_outcome_payload(out_two)
            odds_map[market_hash] = (odds_one, odds_two)
            if odds_one is None and odds_two is None:
                lookup_meta["best_odds_null_count"] += 1
            elif odds_one is not None and odds_two is not None:
                lookup_meta["best_odds_with_both_odds"] += 1
                lookup_meta["best_odds_with_any_odds"] += 1
            else:
                lookup_meta["best_odds_with_any_odds"] += 1
            cache_entries[f"{base_token}:{market_hash}"] = (odds_one, odds_two)

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
) -> List[dict]:
    _ = regions
    stats = {
        "provider": PROVIDER_KEY,
        "source": SX_BET_SOURCE or "api",
        "fixture_source_mode": _fixture_source_mode(),
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
    fixtures, meta = await _load_upcoming_fixtures_async(
        client=client,
        sport_id=sport_id,
        base_token=base_token,
        retries=retries,
        backoff_seconds=backoff,
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

    candidates: List[dict] = []
    unresolved_hashes: List[str] = []
    candidate_hashes: List[str] = []
    for fixture in fixtures:
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
                candidates.append(
                    {
                        "id": fixture_id or str(market_hash or ""),
                        "event_id": fixture_event_id or str(market_hash or ""),
                        "home_team": team_one,
                        "away_team": team_two,
                        "commence_time": commence,
                        **normalized_market,
                    }
                )

    if True:
        odds_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        stake_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        odds_task = None
        stake_task = None
        if unresolved_hashes:
            unique_hashes = list(dict.fromkeys(unresolved_hashes))
            stats["odds_lookup_requested"] = len(unique_hashes)
            odds_task = asyncio.create_task(
                _load_best_odds_map_async(
                    client=client,
                    market_hashes=unique_hashes,
                    base_token=base_token,
                    retries=retries,
                    backoff_seconds=backoff,
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
            unresolved_unique_hashes = list(dict.fromkeys(unresolved_hashes))
            missing_hashes = [market_hash for market_hash in unresolved_unique_hashes if market_hash not in odds_map]
            null_hashes = [
                market_hash
                for market_hash in unresolved_unique_hashes
                if market_hash in odds_map
                and odds_map[market_hash][0] is None
                and odds_map[market_hash][1] is None
            ]
            stats["odds_lookup_missing_hash_entries"] = len(missing_hashes)
            stats["odds_lookup_sample_missing_hashes"] = missing_hashes[:5]
            stats["odds_lookup_sample_null_hashes"] = null_hashes[:5]
            stats["odds_lookup_resolved"] = sum(
                1
                for market_hash in unresolved_unique_hashes
                if market_hash in odds_map
                and odds_map[market_hash][0] is not None
                and odds_map[market_hash][1] is not None
            )
            stats["odds_lookup_unresolved_after_lookup"] = len(unresolved_unique_hashes) - int(
                stats.get("odds_lookup_resolved", 0) or 0
            )

        if stake_task is not None:
            stake_map, retries_used, order_meta = await stake_task
            stats["retries_used"] += retries_used
            stats["orders_lookup_response_count"] = int(order_meta.get("orders_rows", 0) or 0)
            stats["orders_lookup_missing_market_hash_rows"] = int(
                order_meta.get("orders_missing_market_hash", 0) or 0
            )
            candidate_unique_hashes = list(dict.fromkeys(candidate_hashes))
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

    events_by_id: Dict[str, dict] = {}
    for candidate in candidates:
        odds_one = candidate.get("odds_one")
        odds_two = candidate.get("odds_two")
        market_hash = candidate.get("market_hash")
        if (odds_one is None or odds_two is None) and market_hash:
            mapped = odds_map.get(market_hash)
            if mapped:
                odds_one = odds_one or mapped[0]
                odds_two = odds_two or mapped[1]
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
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": candidate.get("commence_time"),
                "markets_by_sig": {},
            },
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
                "bookmakers": [
                    {
                        "key": PROVIDER_KEY,
                        "title": PROVIDER_TITLE,
                        "event_id": event["event_id"],
                        "event_url": _event_url(event["event_id"]),
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
    "source": SX_BET_SOURCE or "api",
    "fixture_source_mode": _fixture_source_mode(),
    "fixture_source_used": "",
    "events_returned_count": 0,
}
fetch_events_async.last_stats = dict(fetch_events.last_stats)

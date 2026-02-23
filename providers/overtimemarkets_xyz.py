from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote

import requests

PROVIDER_KEY = "overtimemarkets_xyz"
PROVIDER_TITLE = "overtimemarkets.xyz"

OTM_SOURCE = os.getenv("OVERTIMEMARKETS_SOURCE", "api").strip().lower()
OTM_API_BASE = os.getenv("OVERTIMEMARKETS_API_BASE", "https://api.overtime.io/overtime-v2").strip()
OTM_NETWORK_RAW = os.getenv("OVERTIMEMARKETS_NETWORK", "10").strip()
OTM_API_KEY = os.getenv("OVERTIMEMARKETS_API_KEY", "").strip()
OTM_PUBLIC_BASE = os.getenv("OVERTIMEMARKETS_PUBLIC_BASE", "https://overtimemarkets.xyz").strip()
OTM_TIMEOUT_RAW = os.getenv("OVERTIMEMARKETS_TIMEOUT_SECONDS", "20").strip()
OTM_RETRIES_RAW = os.getenv("OVERTIMEMARKETS_RETRIES", "2").strip()
OTM_RETRY_BACKOFF_RAW = os.getenv("OVERTIMEMARKETS_RETRY_BACKOFF", "0.5").strip()
OTM_CACHE_TTL_RAW = os.getenv("OVERTIMEMARKETS_CACHE_TTL", "45").strip()
OTM_INCLUDE_LIVE = os.getenv("OVERTIMEMARKETS_INCLUDE_LIVE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OTM_ONLY_BASIC_PROPERTIES = os.getenv("OVERTIMEMARKETS_ONLY_BASIC_PROPERTIES", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
OTM_ONLY_MAIN_MARKETS = os.getenv("OVERTIMEMARKETS_ONLY_MAIN_MARKETS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
OTM_UNGROUP = os.getenv("OVERTIMEMARKETS_UNGROUP", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
OTM_STATUS = os.getenv("OVERTIMEMARKETS_STATUS", "open").strip().lower()
OTM_USER_AGENT = os.getenv(
    "OVERTIMEMARKETS_USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
).strip()
OTM_LEAGUE_MAP_RAW = os.getenv("OVERTIMEMARKETS_LEAGUE_MAP", "").strip()

OTM_MARKETS_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "markets": [],
}

SPORT_LABEL_HINTS: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("football", "american football"),
    "americanfootball_ncaaf": ("football", "american football"),
    "basketball_nba": ("basketball",),
    "basketball_ncaab": ("basketball",),
    "baseball_mlb": ("baseball",),
    "icehockey_nhl": ("hockey", "ice hockey"),
    "soccer_epl": ("soccer",),
    "soccer_spain_la_liga": ("soccer",),
    "soccer_germany_bundesliga": ("soccer",),
    "soccer_italy_serie_a": ("soccer",),
    "soccer_france_ligue_one": ("soccer",),
    "soccer_usa_mls": ("soccer",),
}

LEAGUE_HINTS: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("nfl",),
    "americanfootball_ncaaf": ("ncaaf", "ncaa football", "college football"),
    "basketball_nba": ("nba",),
    "basketball_ncaab": ("ncaa", "college"),
    "baseball_mlb": ("mlb",),
    "icehockey_nhl": ("nhl",),
    "soccer_epl": ("premier league", "epl"),
    "soccer_spain_la_liga": ("la liga",),
    "soccer_germany_bundesliga": ("bundesliga",),
    "soccer_italy_serie_a": ("serie a",),
    "soccer_france_ligue_one": ("ligue 1", "ligue one"),
    "soccer_usa_mls": ("mls", "major league soccer"),
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


def _api_base() -> str:
    base = (OTM_API_BASE or "").strip() or "https://api.overtime.io/overtime-v2"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _public_base() -> str:
    base = (OTM_PUBLIC_BASE or "").strip() or "https://overtimemarkets.xyz"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if OTM_USER_AGENT:
        headers["User-Agent"] = OTM_USER_AGENT
    if OTM_API_KEY:
        headers["x-api-key"] = OTM_API_KEY
    return headers


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_market_key(value: object) -> str:
    token = _normalize_text(value).lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


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


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_commence_time(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1e12:
            timestamp /= 1000.0
        try:
            return dt.datetime.utcfromtimestamp(timestamp).replace(microsecond=0).isoformat() + "Z"
        except (OSError, OverflowError, ValueError):
            return None
    text = _normalize_text(value)
    if not text:
        return None
    if text.isdigit():
        return _normalize_commence_time(int(text))
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


def _decimal_odds_from_value(value) -> Optional[float]:
    odd = _safe_float(value)
    if odd is None or odd <= 0:
        return None
    if odd > 1:
        return odd
    if odd >= 1:
        return None
    converted = 1.0 / odd
    return converted if converted > 1 else None


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


def _curl_get_json(
    url: str,
    params: Dict[str, object],
    headers: Dict[str, str],
    timeout: int,
) -> object:
    curl_bin = shutil.which("curl") or shutil.which("curl.exe")
    if not curl_bin:
        raise ProviderError("curl is not available for fallback requests")
    cmd = [curl_bin, "-sS", "-G", url, "--max-time", str(timeout), "-w", "\n%{http_code}"]
    for key, value in params.items():
        cmd.extend(["--data-urlencode", f"{key}={value}"])
    for key, value in headers.items():
        cmd.extend(["-H", f"{key}: {value}"])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "curl request failed"
        raise ProviderError(f"Overtime request failed: {message}")
    output = result.stdout or ""
    body, sep, status_part = output.rpartition("\n")
    if not sep:
        body = output
        status_code = 0
    else:
        try:
            status_code = int(status_part.strip())
        except ValueError:
            status_code = 0
    if status_code >= 400:
        message = body.strip()[:200] if body else f"HTTP {status_code}"
        raise ProviderError(f"Overtime API request failed ({status_code}): {message}", status_code=status_code)
    try:
        return json.loads(body)
    except ValueError as exc:
        raise ProviderError("Failed to parse Overtime API response") from exc


def _request_json(
    path: str,
    params: Dict[str, object],
    retries: int,
    backoff_seconds: float,
) -> Tuple[object, int]:
    url = f"{_api_base()}/{path.lstrip('/')}"
    timeout = _int_or_default(OTM_TIMEOUT_RAW, 20, min_value=1)
    retriable_status = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    last_error: Optional[ProviderError] = None
    last_request_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        except requests.RequestException as exc:
            last_request_exc = exc
            last_error = ProviderError(f"Overtime network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            break
        if response.status_code >= 400:
            if response.status_code in retriable_status and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            message = ""
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    message = _normalize_text(payload.get("message") or payload.get("error"))
            except ValueError:
                message = response.text.strip()[:120]
            detail = f": {message}" if message else ""
            raise ProviderError(
                f"Overtime API request failed ({response.status_code}){detail}",
                status_code=response.status_code,
            )
        try:
            return response.json(), attempt
        except ValueError as exc:
            last_error = ProviderError("Failed to parse Overtime API response")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc

    # Requests-to-Overtime TLS sometimes fails in some Python environments; fallback to curl.
    if last_request_exc and isinstance(last_request_exc, requests.exceptions.SSLError):
        payload = _curl_get_json(url, params=params, headers=_headers(), timeout=timeout)
        return payload, retries
    if last_error:
        raise last_error
    raise ProviderError("Overtime request failed")


def _extract_markets(payload: object) -> List[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("markets"), list):
        return [item for item in payload.get("markets") if isinstance(item, dict)]
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("markets"), list):
        return [item for item in data.get("markets") if isinstance(item, dict)]
    return []


def _parse_manual_league_map() -> Dict[str, Set[str]]:
    if not OTM_LEAGUE_MAP_RAW:
        return {}
    try:
        payload = json.loads(OTM_LEAGUE_MAP_RAW)
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    mapping: Dict[str, Set[str]] = {}
    for sport_key, value in payload.items():
        key = _normalize_text(sport_key)
        if not key:
            continue
        values: Set[str] = set()
        if isinstance(value, list):
            for item in value:
                item_text = _normalize_text(item).lower()
                if item_text:
                    values.add(item_text)
        else:
            item_text = _normalize_text(value).lower()
            if item_text:
                values.add(item_text)
        if values:
            mapping[key] = values
    return mapping


def _market_matches_sport(
    market: dict,
    sport_key: str,
    manual_league_map: Dict[str, Set[str]],
) -> bool:
    if not sport_key:
        return True
    sport_text = _normalize_text(market.get("sport") or market.get("sportName")).lower()
    league_text = _normalize_text(market.get("leagueName")).lower()
    allowed_sports = SPORT_LABEL_HINTS.get(sport_key, ())
    if allowed_sports and not any(item in sport_text for item in allowed_sports):
        return False
    manual = manual_league_map.get(sport_key)
    if manual is not None:
        if not league_text:
            return False
        return any(token in league_text for token in manual)
    hints = LEAGUE_HINTS.get(sport_key, ())
    if not hints:
        return True
    return any(hint in league_text for hint in hints)


def _is_h2h_market(market: dict) -> bool:
    type_id = _safe_float(market.get("typeId"))
    if type_id is not None and int(type_id) == 0:
        return True
    market_type = _normalize_text(market.get("type")).lower()
    return any(token in market_type for token in ("winner", "money", "h2h", "win"))


def _market_aliases(market: dict) -> List[str]:
    aliases: List[str] = []
    market_type = _normalize_text(
        market.get("type") or market.get("marketType") or market.get("typeName")
    )
    market_name = _normalize_text(
        market.get("marketName") or market.get("name") or market.get("label")
    )
    for raw in (market_type, market_name):
        key = _normalize_market_key(raw)
        if key:
            aliases.append(key)

    period_hints = (market_type, market_name)
    market_type_lc = market_type.lower()
    market_name_lc = market_name.lower()
    if _is_h2h_market(market):
        aliases.append(_scoped_market_alias("h2h", *period_hints))
    if any(token in market_type_lc for token in ("spread", "handicap")) or any(
        token in market_name_lc for token in ("spread", "handicap")
    ):
        aliases.append(_scoped_market_alias("spreads", *period_hints))
    if any(token in market_type_lc for token in ("total", "over_under", "overunder")) or any(
        token in market_name_lc for token in ("total", "over/under", "over under")
    ):
        aliases.append(_scoped_market_alias("totals", *period_hints))
    if "btts" in market_type_lc or "both teams to score" in market_name_lc:
        aliases.extend(["btts", "both_teams_to_score"])

    out: List[str] = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            out.append(alias)
            seen.add(alias)
    return out


def _market_point_value(market: dict) -> Optional[float]:
    for key in ("line", "marketValue", "value", "handicap", "spread", "total", "point"):
        if key not in market:
            continue
        raw = market.get(key)
        if isinstance(raw, (int, float)):
            parsed = _safe_float(raw)
            if parsed is not None:
                return parsed
        match = re.search(r"[-+]?\d+(?:\.\d+)?", _normalize_text(raw))
        if match:
            parsed = _safe_float(match.group(0))
            if parsed is not None:
                return parsed
    return None


def _market_option_names(market: dict) -> Optional[Tuple[str, str]]:
    for key in ("options", "positions", "outcomes", "sides"):
        options = market.get(key)
        if not isinstance(options, list) or len(options) < 2:
            continue
        names: List[str] = []
        for item in options[:2]:
            if isinstance(item, dict):
                name = _normalize_text(
                    item.get("name") or item.get("label") or item.get("title") or item.get("outcome")
                )
            else:
                name = _normalize_text(item)
            names.append(name)
        if names[0] and names[1]:
            return names[0], names[1]
    return None


def _build_market(
    row: dict,
    requested_markets: set[str],
    home_team: str,
    away_team: str,
) -> Optional[dict]:
    target_market = None
    for alias in _market_aliases(row):
        if alias in requested_markets:
            target_market = alias
            break
    if not target_market:
        return None
    if target_market == "btts":
        target_market = (
            "both_teams_to_score" if "both_teams_to_score" in requested_markets else "btts"
        )
    target_market_base = _base_market_key(target_market)

    odds_raw = row.get("odds")
    if not isinstance(odds_raw, list) or len(odds_raw) < 2:
        return None
    odd_one = _decimal_odds_from_value(odds_raw[0])
    odd_two = _decimal_odds_from_value(odds_raw[1])
    if odd_one is None or odd_two is None or odd_one <= 1 or odd_two <= 1:
        return None

    if target_market_base == "h2h":
        if len(odds_raw) != 2:
            return None
        return {
            "key": target_market,
            "outcomes": [
                {"name": home_team, "price": round(float(odd_one), 6)},
                {"name": away_team, "price": round(float(odd_two), 6)},
            ],
        }

    point = _market_point_value(row)
    if target_market_base == "spreads":
        if point is None:
            return None
        value = round(float(point), 6)
        return {
            "key": target_market,
            "outcomes": [
                {"name": home_team, "price": round(float(odd_one), 6), "point": value},
                {"name": away_team, "price": round(float(odd_two), 6), "point": round(-value, 6)},
            ],
        }

    if target_market_base == "totals":
        if point is None:
            return None
        value = round(float(point), 6)
        return {
            "key": target_market,
            "outcomes": [
                {"name": "Over", "price": round(float(odd_one), 6), "point": value},
                {"name": "Under", "price": round(float(odd_two), 6), "point": value},
            ],
        }

    option_names = _market_option_names(row)
    outcome_one_name = option_names[0] if option_names else "Outcome 1"
    outcome_two_name = option_names[1] if option_names else "Outcome 2"
    description = _normalize_text(
        row.get("marketName") or row.get("name") or row.get("type") or row.get("typeName")
    )
    dynamic = {
        "key": target_market,
        "outcomes": [
            {"name": outcome_one_name, "price": round(float(odd_one), 6)},
            {"name": outcome_two_name, "price": round(float(odd_two), 6)},
        ],
    }
    if point is not None:
        value = round(float(point), 6)
        dynamic["outcomes"][0]["point"] = value
        dynamic["outcomes"][1]["point"] = value
    if description:
        dynamic["outcomes"][0]["description"] = description
        dynamic["outcomes"][1]["description"] = description
    return dynamic


def _event_url(game_id: object) -> str:
    raw = _normalize_text(game_id)
    if not raw:
        return ""
    return f"{_public_base()}/markets/{quote(raw, safe='')}"


def _load_otm_markets(
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[dict], dict]:
    ttl = _int_or_default(OTM_CACHE_TTL_RAW, 45, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(OTM_MARKETS_CACHE.get("expires_at", 0.0))
    if cache_valid and isinstance(OTM_MARKETS_CACHE.get("markets"), list):
        return OTM_MARKETS_CACHE["markets"], {"cache": "hit", "requests": 0, "retries_used": 0}

    params = {
        "onlyBasicProperties": "true" if OTM_ONLY_BASIC_PROPERTIES else "false",
        "ungroup": "true" if OTM_UNGROUP else "false",
        "onlyMainMarkets": "true" if OTM_ONLY_MAIN_MARKETS else "false",
        "status": OTM_STATUS or "open",
    }
    network = _int_or_default(OTM_NETWORK_RAW, 10, min_value=1)
    payload, retries_used = _request_json(
        f"networks/{network}/markets",
        params=params,
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    markets = _extract_markets(payload)
    requests_count = 1
    if OTM_INCLUDE_LIVE:
        live_payload, live_retries = _request_json(
            f"networks/{network}/live-markets",
            params={
                "onlyBasicProperties": "true" if OTM_ONLY_BASIC_PROPERTIES else "false",
                "ungroup": "true" if OTM_UNGROUP else "false",
                "onlyMainMarkets": "true" if OTM_ONLY_MAIN_MARKETS else "false",
            },
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        markets.extend(_extract_markets(live_payload))
        retries_used += live_retries
        requests_count += 1

    OTM_MARKETS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    OTM_MARKETS_CACHE["markets"] = markets
    return markets, {"cache": "miss", "requests": requests_count, "retries_used": retries_used}


def fetch_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    _ = regions
    stats = {
        "provider": PROVIDER_KEY,
        "source": OTM_SOURCE or "api",
        "auth_configured": bool(OTM_API_KEY),
        "events_payload_count": 0,
        "events_sport_filtered_count": 0,
        "events_h2h_candidates_count": 0,
        "events_with_odds_count": 0,
        "events_returned_count": 0,
        "requests_made": 0,
        "retries_used": 0,
        "payload_cache": "miss",
    }
    fetch_events.last_stats = stats

    requested_markets = _requested_market_keys(markets)
    if not requested_markets:
        return []
    if bookmakers:
        lowered = {str(book).strip().lower() for book in bookmakers if isinstance(book, str)}
        if PROVIDER_KEY not in lowered and PROVIDER_TITLE.lower() not in lowered:
            return []
    if (OTM_SOURCE or "api").lower() != "api":
        raise ProviderError("Overtime provider currently supports OVERTIMEMARKETS_SOURCE=api only")
    if not OTM_API_KEY:
        stats["auth_error"] = "missing_api_key"
        fetch_events.last_stats = stats
        return []

    retries = _int_or_default(OTM_RETRIES_RAW, 2, min_value=0)
    backoff = _float_or_default(OTM_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    manual_league_map = _parse_manual_league_map()

    try:
        market_rows, meta = _load_otm_markets(retries=retries, backoff_seconds=backoff)
    except ProviderError as exc:
        if exc.status_code in {401, 403}:
            stats["auth_error"] = str(exc)
            fetch_events.last_stats = stats
            return []
        raise

    stats["payload_cache"] = meta.get("cache", "miss")
    stats["requests_made"] += int(meta.get("requests", 0) or 0)
    stats["retries_used"] += int(meta.get("retries_used", 0) or 0)
    stats["events_payload_count"] = len(market_rows)

    events_by_id: Dict[str, dict] = {}
    for row in market_rows:
        if not isinstance(row, dict):
            continue
        if not _market_matches_sport(row, sport_key, manual_league_map):
            continue
        stats["events_sport_filtered_count"] += 1
        home_team = _normalize_text(row.get("homeTeam"))
        away_team = _normalize_text(row.get("awayTeam"))
        if not (home_team and away_team):
            continue

        market_payload = _build_market(
            row=row,
            requested_markets=requested_markets,
            home_team=home_team,
            away_team=away_team,
        )
        if not market_payload:
            continue
        if market_payload.get("key") == "h2h":
            stats["events_h2h_candidates_count"] += 1
        stats["events_with_odds_count"] += 1

        event_id = _normalize_text(row.get("gameId") or row.get("id"))
        if not event_id:
            continue
        commence = _normalize_commence_time(row.get("maturityDate") or row.get("maturity"))
        if not commence:
            continue

        current = events_by_id.get(event_id)
        if current is None:
            current = {
                "id": event_id,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence,
                "event_id": event_id,
                "event_url": _event_url(event_id),
                "markets_by_sig": {},
            }
            events_by_id[event_id] = current

        signature = _market_signature(market_payload)
        previous = current["markets_by_sig"].get(signature)
        if previous is None or _market_score(market_payload) > _market_score(previous):
            current["markets_by_sig"][signature] = market_payload

    events_out = []
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
                        "event_url": event["event_url"],
                        "markets": market_list,
                    }
                ],
            }
        )
    stats["events_returned_count"] = len(events_out)
    fetch_events.last_stats = stats
    return events_out


fetch_events.last_stats = {
    "provider": PROVIDER_KEY,
    "source": OTM_SOURCE or "api",
    "events_returned_count": 0,
}

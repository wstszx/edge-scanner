from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import requests

PROVIDER_KEY = "sx_bet"
PROVIDER_TITLE = "SX Bet"

SX_BET_SOURCE = os.getenv("SX_BET_SOURCE", "api").strip().lower()
SX_BET_API_BASE = os.getenv("SX_BET_API_BASE", "https://api.sx.bet").strip()
SX_BET_PUBLIC_BASE = os.getenv("SX_BET_PUBLIC_BASE", "https://sx.bet").strip()
SX_BET_BASE_TOKEN = os.getenv(
    "SX_BET_BASE_TOKEN",
    "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
).strip()
SX_BET_TIMEOUT_RAW = os.getenv("SX_BET_TIMEOUT_SECONDS", "20").strip()
SX_BET_RETRIES_RAW = os.getenv("SX_BET_RETRIES", "2").strip()
SX_BET_RETRY_BACKOFF_RAW = os.getenv("SX_BET_RETRY_BACKOFF", "0.5").strip()
SX_BET_PAGE_TTL_RAW = os.getenv("SX_BET_PAGE_CACHE_TTL", "45").strip()
SX_BET_ODDS_CACHE_TTL_RAW = os.getenv("SX_BET_ODDS_CACHE_TTL", "30").strip()
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
            return dt.datetime.utcfromtimestamp(timestamp).replace(microsecond=0).isoformat() + "Z"
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


def _normalize_token(value: object) -> str:
    text = _normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def _market_type_aliases(market: dict) -> List[str]:
    raw_type = _normalize_text(market.get("marketType") or market.get("type")).upper()
    raw_name = _normalize_text(market.get("marketName") or market.get("name"))
    aliases: List[str] = []

    for raw in (raw_type, raw_name):
        key = _normalize_market_key(raw)
        if key:
            aliases.append(key)

    type_lc = raw_type.lower()
    name_lc = raw_name.lower()

    if any(token in type_lc for token in ("money_line", "moneyline", "match_odds", "h2h")) or any(
        token in name_lc for token in ("moneyline", "match winner", "winner", "to win")
    ):
        aliases.append("h2h")
    if any(token in type_lc for token in ("spread", "handicap")) or any(
        token in name_lc for token in ("spread", "handicap")
    ):
        aliases.append("spreads")
    if any(token in type_lc for token in ("total", "over_under", "overunder")) or any(
        token in name_lc for token in ("total", "over/under", "over under")
    ):
        aliases.append("totals")
    if "btts" in type_lc or "both teams to score" in name_lc:
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

    if target_market_key == "h2h":
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

    if target_market_key == "spreads":
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

    if target_market_key == "totals":
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
    SX summary fields may already be decimal odds or probabilities.
    - > 1: treat as decimal odds
    - (0,1): treat as probability and convert to decimal
    """
    odds = _safe_float(value)
    if odds is None:
        return None
    if odds <= 0:
        return None
    if odds > 1:
        return odds
    if odds >= 1:
        return None
    converted = 1.0 / odds
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


def _load_upcoming_fixtures(
    sport_id: int,
    base_token: str,
    retries: int,
    backoff_seconds: float,
) -> Tuple[List[dict], dict]:
    ttl = _int_or_default(SX_BET_PAGE_TTL_RAW, 45, min_value=0)
    cache_key = f"{base_token}:{sport_id}"
    now = time.time()
    cache_valid = ttl > 0 and now < float(UPCOMING_CACHE.get("expires_at", 0.0))
    cache_entries = UPCOMING_CACHE.get("entries")
    if cache_valid and isinstance(cache_entries, dict):
        cached = cache_entries.get(cache_key)
        if isinstance(cached, list):
            return cached, {"pages_fetched": 0, "retries_used": 0, "cache": "hit"}

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

    entries = cache_entries if isinstance(cache_entries, dict) else {}
    entries[cache_key] = fixtures
    UPCOMING_CACHE["entries"] = entries
    UPCOMING_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    return fixtures, {"pages_fetched": 1, "retries_used": retries_used, "cache": "miss"}


def _load_best_odds_map(
    market_hashes: Sequence[str],
    base_token: str,
    retries: int,
    backoff_seconds: float,
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], int, dict]:
    ttl = _int_or_default(SX_BET_ODDS_CACHE_TTL_RAW, 30, min_value=0)
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


def fetch_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    _ = regions
    stats = {
        "provider": PROVIDER_KEY,
        "source": SX_BET_SOURCE or "api",
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
        "dropped_missing_odds_count": 0,
        "dropped_invalid_odds_count": 0,
        "sample_dropped_market_hashes": [],
        "sample_filtered_leagues": [],
        "pages_fetched": 0,
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
    manual_league_map = _parse_manual_league_map()

    fixtures, meta = _load_upcoming_fixtures(
        sport_id=sport_id,
        base_token=base_token,
        retries=retries,
        backoff_seconds=backoff,
    )
    stats["payload_cache"] = meta.get("cache", "miss")
    stats["pages_fetched"] = int(meta.get("pages_fetched", 0) or 0)
    stats["retries_used"] += int(meta.get("retries_used", 0) or 0)
    stats["fixtures_payload_count"] = len(fixtures)

    candidates: List[dict] = []
    unresolved_hashes: List[str] = []
    for fixture in fixtures:
        if not _fixture_matches_sport(sport_key, fixture, manual_league_map):
            stats["fixtures_filtered_out_by_league_count"] += 1
            league_label = _normalize_text(fixture.get("leagueLabel"))
            if league_label and len(stats["sample_filtered_leagues"]) < 5:
                if league_label not in stats["sample_filtered_leagues"]:
                    stats["sample_filtered_leagues"].append(league_label)
            continue
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

    odds_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if unresolved_hashes:
        unique_hashes = list(dict.fromkeys(unresolved_hashes))
        stats["odds_lookup_requested"] = len(unique_hashes)
        odds_map, retries_used, lookup_meta = _load_best_odds_map(
            market_hashes=unique_hashes,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff,
        )
        stats["retries_used"] += retries_used
        stats["odds_lookup_response_count"] = int(lookup_meta.get("best_odds_items", 0) or 0)
        stats["odds_lookup_null_entries"] = int(lookup_meta.get("best_odds_null_count", 0) or 0)
        resolved_partial = int(lookup_meta.get("best_odds_with_any_odds", 0) or 0) - int(
            lookup_meta.get("best_odds_with_both_odds", 0) or 0
        )
        stats["odds_lookup_resolved_partial"] = max(0, resolved_partial)
        missing_hashes = [market_hash for market_hash in unique_hashes if market_hash not in odds_map]
        null_hashes = [
            market_hash
            for market_hash in unique_hashes
            if market_hash in odds_map
            and odds_map[market_hash][0] is None
            and odds_map[market_hash][1] is None
        ]
        stats["odds_lookup_missing_hash_entries"] = len(missing_hashes)
        stats["odds_lookup_sample_missing_hashes"] = missing_hashes[:5]
        stats["odds_lookup_sample_null_hashes"] = null_hashes[:5]
        stats["odds_lookup_resolved"] = sum(
            1
            for market_hash in unique_hashes
            if market_hash in odds_map
            and odds_map[market_hash][0] is not None
            and odds_map[market_hash][1] is not None
        )
        stats["odds_lookup_unresolved_after_lookup"] = len(unique_hashes) - int(
            stats.get("odds_lookup_resolved", 0) or 0
        )

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

        description = _normalize_text(candidate.get("description"))
        if description and candidate.get("market_key") not in {"h2h", "spreads", "totals"}:
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
    fetch_events.last_stats = stats
    return events_out


fetch_events.last_stats = {
    "provider": PROVIDER_KEY,
    "source": SX_BET_SOURCE or "api",
    "events_returned_count": 0,
}

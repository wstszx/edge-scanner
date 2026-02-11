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
    SX orders endpoint returns percentage odds:
    - 0..1 => probability
    - 1..100 => percent probability
    """
    raw = _safe_float(value)
    if raw is None or raw <= 0:
        return None
    if raw > 1:
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
) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], int]:
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
            market_hash = _normalize_text(item.get("marketHash"))
            if not market_hash:
                continue
            out_one = item.get("outcomeOne") if isinstance(item.get("outcomeOne"), dict) else {}
            out_two = item.get("outcomeTwo") if isinstance(item.get("outcomeTwo"), dict) else {}
            odds_one = _moneyline_decimal_from_percentage(out_one.get("percentageOdds"))
            odds_two = _moneyline_decimal_from_percentage(out_two.get("percentageOdds"))
            odds_map[market_hash] = (odds_one, odds_two)
            cache_entries[f"{base_token}:{market_hash}"] = (odds_one, odds_two)

    ODDS_CACHE["entries"] = cache_entries
    ODDS_CACHE["expires_at"] = now + ttl if ttl > 0 else now
    return odds_map, retries_used_total


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
        "fixtures_market_found_count": 0,
        "events_returned_count": 0,
        "odds_lookup_requested": 0,
        "odds_lookup_resolved": 0,
        "pages_fetched": 0,
        "retries_used": 0,
        "payload_cache": "miss",
    }
    fetch_events.last_stats = stats

    requested_markets = {str(item).strip().lower() for item in (markets or []) if str(item).strip()}
    if "h2h" not in requested_markets:
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
            continue
        stats["fixtures_sport_filtered_count"] += 1
        team_one = _normalize_text(fixture.get("teamOne"))
        team_two = _normalize_text(fixture.get("teamTwo"))
        if not (team_one and team_two):
            continue
        commence = _normalize_commence_time(fixture.get("gameTime"))
        if not commence:
            continue

        markets_list = fixture.get("markets")
        if not isinstance(markets_list, list):
            continue
        moneyline = None
        for market in markets_list:
            if not isinstance(market, dict):
                continue
            if _normalize_text(market.get("marketType")).upper() == "MONEY_LINE":
                moneyline = market
                break
        if not isinstance(moneyline, dict):
            continue
        stats["fixtures_market_found_count"] += 1

        odds_one = _moneyline_decimal_from_summary(moneyline.get("bestOddsOutcomeOne"))
        odds_two = _moneyline_decimal_from_summary(moneyline.get("bestOddsOutcomeTwo"))
        market_hash = _normalize_text(moneyline.get("marketHash"))
        if (odds_one is None or odds_two is None) and market_hash:
            unresolved_hashes.append(market_hash)

        candidates.append(
            {
                "id": _normalize_text(fixture.get("id") or fixture.get("eventId") or market_hash),
                "event_id": _normalize_text(fixture.get("eventId") or fixture.get("id") or market_hash),
                "home_team": team_one,
                "away_team": team_two,
                "commence_time": commence,
                "market_hash": market_hash,
                "odds_one": odds_one,
                "odds_two": odds_two,
                "outcome_one_name": _normalize_text(moneyline.get("outcomeOneName")) or team_one,
                "outcome_two_name": _normalize_text(moneyline.get("outcomeTwoName")) or team_two,
            }
        )

    odds_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if unresolved_hashes:
        unique_hashes = list(dict.fromkeys(unresolved_hashes))
        stats["odds_lookup_requested"] = len(unique_hashes)
        odds_map, retries_used = _load_best_odds_map(
            market_hashes=unique_hashes,
            base_token=base_token,
            retries=retries,
            backoff_seconds=backoff,
        )
        stats["retries_used"] += retries_used

    events_out: List[dict] = []
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
            continue
        if odds_one <= 1 or odds_two <= 1:
            continue
        stats["odds_lookup_resolved"] += 1 if market_hash else 0

        event_id = candidate.get("event_id") or candidate.get("id")
        home_team = candidate.get("home_team")
        away_team = candidate.get("away_team")
        if not (event_id and home_team and away_team):
            continue
        events_out.append(
            {
                "id": candidate.get("id") or event_id,
                "sport_key": sport_key,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": candidate.get("commence_time"),
                "bookmakers": [
                    {
                        "key": PROVIDER_KEY,
                        "title": PROVIDER_TITLE,
                        "event_id": event_id,
                        "event_url": _event_url(event_id),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {
                                        "name": home_team,
                                        "price": round(float(odds_one), 6),
                                    },
                                    {
                                        "name": away_team,
                                        "price": round(float(odds_two), 6),
                                    },
                                ],
                            }
                        ],
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

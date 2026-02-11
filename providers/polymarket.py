from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests

PROVIDER_KEY = "polymarket"
PROVIDER_TITLE = "Polymarket"

POLYMARKET_SOURCE = os.getenv("POLYMARKET_SOURCE", "api").strip().lower()
POLYMARKET_API_BASE = os.getenv("POLYMARKET_API_BASE", "https://gamma-api.polymarket.com").strip()
POLYMARKET_PUBLIC_BASE = os.getenv("POLYMARKET_PUBLIC_BASE", "https://polymarket.com").strip()
POLYMARKET_GAME_TAG_ID = os.getenv("POLYMARKET_GAME_TAG_ID", "100639").strip()
POLYMARKET_PAGE_SIZE_RAW = os.getenv("POLYMARKET_PAGE_SIZE", "200").strip()
POLYMARKET_MAX_PAGES_RAW = os.getenv("POLYMARKET_MAX_PAGES", "8").strip()
POLYMARKET_RETRIES_RAW = os.getenv("POLYMARKET_RETRIES", "2").strip()
POLYMARKET_RETRY_BACKOFF_RAW = os.getenv("POLYMARKET_RETRY_BACKOFF", "0.5").strip()
POLYMARKET_TIMEOUT_RAW = os.getenv("POLYMARKET_TIMEOUT_SECONDS", "20").strip()
POLYMARKET_SPORT_TAG_CACHE_TTL_RAW = os.getenv("POLYMARKET_SPORT_TAG_CACHE_TTL", "600").strip()
POLYMARKET_EVENTS_CACHE_TTL_RAW = os.getenv("POLYMARKET_EVENTS_CACHE_TTL", "60").strip()
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

SPORT_TAG_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "mapping": {},
}
EVENTS_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "events": [],
}

SPORT_ALIASES: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("nfl", "american-football"),
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
    base = (POLYMARKET_API_BASE or "").strip() or "https://gamma-api.polymarket.com"
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


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_token(value: object) -> str:
    text = _normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


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
            return dt.datetime.utcfromtimestamp(timestamp).replace(microsecond=0).isoformat() + "Z"
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


def _price_to_decimal_odds(price) -> Optional[float]:
    probability = _safe_float(price)
    if probability is None or probability <= 0 or probability >= 1:
        return None
    odds = 1.0 / probability
    if odds <= 1:
        return None
    return round(odds, 6)


def _pick_two_outcome_market(
    event: dict,
    home_team: str,
    away_team: str,
) -> Optional[dict]:
    home_token = _team_token(home_team)
    away_token = _team_token(away_team)
    if not home_token or not away_token:
        return None

    direct_candidates: List[dict] = []
    yes_by_team: Dict[str, float] = {}
    has_draw_prompt = False

    for market in (event.get("markets") or []):
        if not isinstance(market, dict):
            continue
        outcomes = [_normalize_text(item) for item in _parse_list(market.get("outcomes"))]
        prices = _parse_list(market.get("outcomePrices"))
        if len(outcomes) != 2 or len(prices) != 2:
            continue
        odds_0 = _price_to_decimal_odds(prices[0])
        odds_1 = _price_to_decimal_odds(prices[1])
        if odds_0 is None or odds_1 is None:
            continue

        outcome_tokens = [_team_token(outcomes[0]), _team_token(outcomes[1])]
        question = _normalize_text(market.get("question")).lower()

        if "draw" in question:
            has_draw_prompt = True
            continue

        if set(outcome_tokens) == {home_token, away_token}:
            direct_candidates.append(
                {
                    "outcomes": [
                        {"name": outcomes[0], "price": odds_0},
                        {"name": outcomes[1], "price": odds_1},
                    ],
                    "volume": _safe_float(market.get("volumeNum") or market.get("volume")) or 0.0,
                }
            )
            continue

        normalized_outcomes = [token.lower() for token in outcome_tokens]
        if normalized_outcomes == ["yes", "no"] or normalized_outcomes == ["no", "yes"]:
            yes_index = 0 if normalized_outcomes[0] == "yes" else 1
            yes_odds = odds_0 if yes_index == 0 else odds_1
            team_match = re.match(
                r"^\s*will\s+(.+?)\s+win(?:\s+on\s+\d{4}-\d{2}-\d{2})?\??\s*$",
                _normalize_text(market.get("question")),
                flags=re.IGNORECASE,
            )
            if team_match:
                team_token = _team_token(team_match.group(1))
                if team_token == home_token:
                    yes_by_team[home_token] = yes_odds
                elif team_token == away_token:
                    yes_by_team[away_token] = yes_odds

    if direct_candidates:
        best = max(direct_candidates, key=lambda item: item.get("volume", 0.0))
        outcomes = best["outcomes"]
        # Normalize outcome order to home/away.
        ordered = { _team_token(item["name"]): item for item in outcomes }
        home_outcome = ordered.get(home_token)
        away_outcome = ordered.get(away_token)
        if home_outcome and away_outcome:
            return {
                "key": "h2h",
                "outcomes": [
                    {"name": home_team, "price": home_outcome["price"]},
                    {"name": away_team, "price": away_outcome["price"]},
                ],
            }
        return {
            "key": "h2h",
            "outcomes": outcomes,
        }

    if has_draw_prompt:
        return None
    if home_token in yes_by_team and away_token in yes_by_team:
        return {
            "key": "h2h",
            "outcomes": [
                {"name": home_team, "price": yes_by_team[home_token]},
                {"name": away_team, "price": yes_by_team[away_token]},
            ],
        }
    return None


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
    ttl = _int_or_default(POLYMARKET_EVENTS_CACHE_TTL_RAW, 60, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(EVENTS_CACHE.get("expires_at", 0.0))
    cached_events = EVENTS_CACHE.get("events")
    if cache_valid and isinstance(cached_events, list):
        return cached_events, {"pages_fetched": 0, "retries_used": 0, "cache": "hit"}

    all_events: List[dict] = []
    total_retries = 0
    pages_fetched = 0
    offset = 0
    for _page in range(max_pages):
        payload, retries_used = _request_json(
            "events",
            {
                "tag_id": POLYMARKET_GAME_TAG_ID or "100639",
                "active": "true",
                "closed": "false",
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
    return all_events, {"pages_fetched": pages_fetched, "retries_used": total_retries, "cache": "miss"}


def fetch_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    _ = regions  # Reserved for future region-specific routing.
    stats = {
        "provider": PROVIDER_KEY,
        "source": POLYMARKET_SOURCE or "api",
        "events_payload_count": 0,
        "events_sports_count": 0,
        "events_sport_filtered_count": 0,
        "events_matchup_count": 0,
        "events_with_market_count": 0,
        "events_returned_count": 0,
        "pages_fetched": 0,
        "retries_used": 0,
        "payload_cache": "miss",
    }
    fetch_events.last_stats = stats

    supported_markets = {str(item).strip().lower() for item in (markets or []) if str(item).strip()}
    if "h2h" not in supported_markets:
        return []

    if bookmakers:
        lowered = {str(book).strip().lower() for book in bookmakers if isinstance(book, str)}
        if PROVIDER_KEY not in lowered and PROVIDER_TITLE.lower() not in lowered:
            return []

    if (POLYMARKET_SOURCE or "api").lower() != "api":
        raise ProviderError("Polymarket provider currently supports POLYMARKET_SOURCE=api only")

    retries = _int_or_default(POLYMARKET_RETRIES_RAW, 2, min_value=0)
    backoff = _float_or_default(POLYMARKET_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    page_size = _int_or_default(POLYMARKET_PAGE_SIZE_RAW, 200, min_value=1)
    max_pages = _int_or_default(POLYMARKET_MAX_PAGES_RAW, 8, min_value=1)

    sport_tag_mapping = _load_sport_tag_mapping(retries=retries, backoff_seconds=backoff)
    payload, payload_meta = _load_active_game_events(
        retries=retries,
        backoff_seconds=backoff,
        page_size=page_size,
        max_pages=max_pages,
    )
    stats["payload_cache"] = payload_meta.get("cache", "miss")
    stats["pages_fetched"] += int(payload_meta.get("pages_fetched", 0) or 0)
    stats["retries_used"] += int(payload_meta.get("retries_used", 0) or 0)

    events_out: List[dict] = []
    stats["events_payload_count"] = len(payload)
    for event in payload:
        if not isinstance(event, dict):
            continue
        if not _event_is_sports(event):
            continue
        stats["events_sports_count"] += 1
        if not _event_matches_sport(event, sport_key, sport_tag_mapping):
            continue
        stats["events_sport_filtered_count"] += 1

        matchup = _extract_matchup(event)
        if not matchup:
            continue
        home_team, away_team = matchup
        stats["events_matchup_count"] += 1

        market_h2h = _pick_two_outcome_market(event, home_team, away_team)
        if not market_h2h:
            continue
        stats["events_with_market_count"] += 1

        commence = _normalize_commence_time(
            event.get("startDate")
            or event.get("creationDate")
            or event.get("createdAt")
        )
        if not commence:
            commence = _normalize_commence_time(
                (event.get("markets") or [{}])[0].get("startDate")
                if isinstance(event.get("markets"), list) and event.get("markets")
                else None
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
                "bookmakers": [
                    {
                        "key": PROVIDER_KEY,
                        "title": PROVIDER_TITLE,
                        "event_id": event_id,
                        "event_url": _event_url(event),
                        "markets": [market_h2h],
                    }
                ],
            }
        )

    stats["events_returned_count"] = len(events_out)
    fetch_events.last_stats = stats
    return events_out


fetch_events.last_stats = {
    "provider": PROVIDER_KEY,
    "source": POLYMARKET_SOURCE or "api",
    "events_returned_count": 0,
}

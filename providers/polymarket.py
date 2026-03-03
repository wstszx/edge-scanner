from __future__ import annotations

import concurrent.futures
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
POLYMARKET_CLOB_BASE = os.getenv("POLYMARKET_CLOB_BASE", "https://clob.polymarket.com").strip()
POLYMARKET_PUBLIC_BASE = os.getenv("POLYMARKET_PUBLIC_BASE", "https://polymarket.com").strip()
POLYMARKET_GAME_TAG_ID = os.getenv("POLYMARKET_GAME_TAG_ID", "100639").strip()
POLYMARKET_PAGE_SIZE_RAW = os.getenv("POLYMARKET_PAGE_SIZE", "200").strip()
POLYMARKET_MAX_PAGES_RAW = os.getenv("POLYMARKET_MAX_PAGES", "8").strip()
POLYMARKET_RETRIES_RAW = os.getenv("POLYMARKET_RETRIES", "2").strip()
POLYMARKET_RETRY_BACKOFF_RAW = os.getenv("POLYMARKET_RETRY_BACKOFF", "0.5").strip()
POLYMARKET_TIMEOUT_RAW = os.getenv("POLYMARKET_TIMEOUT_SECONDS", "20").strip()
POLYMARKET_SPORT_TAG_CACHE_TTL_RAW = os.getenv("POLYMARKET_SPORT_TAG_CACHE_TTL", "600").strip()
POLYMARKET_EVENTS_CACHE_TTL_RAW = os.getenv("POLYMARKET_EVENTS_CACHE_TTL", "60").strip()
POLYMARKET_CLOB_BOOK_CACHE_TTL_RAW = os.getenv("POLYMARKET_CLOB_BOOK_CACHE_TTL", "20").strip()
POLYMARKET_CLOB_MAX_BOOKS_RAW = os.getenv("POLYMARKET_CLOB_MAX_BOOKS", "300").strip()
POLYMARKET_CLOB_BOOK_WORKERS_RAW = os.getenv("POLYMARKET_CLOB_BOOK_WORKERS", "8").strip()
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
CLOB_BOOK_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "entries": {},
}

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
    if end_at is not None and end_at < now_utc:
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
    if end_at is not None and end_at < now_utc:
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


def _parse_clob_token_ids(value: object) -> List[str]:
    token_ids: List[str] = []
    for item in _parse_list(value):
        token_id = _normalize_text(item)
        if token_id and token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids


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

    ttl = _int_or_default(POLYMARKET_CLOB_BOOK_CACHE_TTL_RAW, 20, min_value=0)
    now = time.time()
    cache_valid = ttl > 0 and now < float(CLOB_BOOK_CACHE.get("expires_at", 0.0))
    cache_entries = CLOB_BOOK_CACHE.get("entries") if isinstance(CLOB_BOOK_CACHE.get("entries"), dict) else {}

    depth_by_token: Dict[str, Optional[float]] = {}
    unresolved: List[str] = []
    for token_id in unique_token_ids:
        if cache_valid and token_id in cache_entries:
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
                    book_errors += 1
                    continue
                books_fetched += 1
                depth_by_token[resolved_token_id] = depth
                cache_entries[resolved_token_id] = depth
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_fetch_single_book, token_id) for token_id in unresolved]
                for future in concurrent.futures.as_completed(futures):
                    resolved_token_id, depth, retry_count, had_error = future.result()
                    retries_used += retry_count
                    if had_error:
                        depth_by_token[resolved_token_id] = None
                        cache_entries[resolved_token_id] = None
                        book_errors += 1
                        continue
                    books_fetched += 1
                    depth_by_token[resolved_token_id] = depth
                    cache_entries[resolved_token_id] = depth

    CLOB_BOOK_CACHE["entries"] = cache_entries
    CLOB_BOOK_CACHE["expires_at"] = now + ttl if ttl > 0 else now

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


def _market_outcome_row(name: str, price: float, stake: Optional[float] = None) -> dict:
    row = {"name": name, "price": round(float(price), 6)}
    if stake is not None and stake > 0:
        row["stake"] = round(float(stake), 6)
    return row


def _pick_match_markets(
    event: dict,
    home_team: str,
    away_team: str,
    requested_markets: set[str],
    now_utc: dt.datetime,
    clob_depth_map: Optional[Dict[str, Optional[float]]] = None,
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
        odds_0 = _price_to_decimal_odds(prices[0])
        odds_1 = _price_to_decimal_odds(prices[1])
        if odds_0 is None or odds_1 is None:
            continue

        token_ids = _parse_clob_token_ids(market.get("clobTokenIds"))
        stakes = [None, None]
        if isinstance(clob_depth_map, dict):
            for idx in range(2):
                if idx >= len(token_ids):
                    continue
                depth = _safe_float(clob_depth_map.get(token_ids[idx]))
                if depth is not None and depth > 0:
                    stakes[idx] = round(float(depth), 6)

        outcome_tokens = [_team_token(outcomes[0]), _team_token(outcomes[1])]
        question = _normalize_text(market.get("question")).lower()

        if "draw" in question:
            has_draw_prompt = True

        if set(outcome_tokens) == {home_token, away_token}:
            direct_candidates.append(
                {
                    "outcomes": [
                        _market_outcome_row(outcomes[0], odds_0, stakes[0]),
                        _market_outcome_row(outcomes[1], odds_1, stakes[1]),
                    ],
                    "volume": _safe_float(market.get("volumeNum") or market.get("volume")) or 0.0,
                }
            )
            continue

        normalized_outcomes = [token.lower() for token in outcome_tokens]
        if normalized_outcomes == ["yes", "no"] or normalized_outcomes == ["no", "yes"]:
            yes_index = 0 if normalized_outcomes[0] == "yes" else 1
            yes_odds = odds_0 if yes_index == 0 else odds_1
            no_odds = odds_1 if yes_index == 0 else odds_0
            yes_stake = stakes[yes_index]
            no_stake = stakes[1 - yes_index]
            if "both teams to score" in question or "btts" in question:
                btts_yes = {"odds": yes_odds, "stake": yes_stake}
                btts_no = {"odds": no_odds, "stake": no_stake}
                continue
            if "draw" in question:
                draw_yes = {"odds": yes_odds, "stake": yes_stake}
                continue
            team_match = re.match(
                r"^\s*will\s+(.+?)\s+win(?:\s+on\s+\d{4}-\d{2}-\d{2})?\??\s*$",
                _normalize_text(market.get("question")),
                flags=re.IGNORECASE,
            )
            if team_match:
                team_token = _team_token(team_match.group(1))
                if team_token == home_token:
                    yes_by_team[home_token] = {"odds": yes_odds, "stake": yes_stake}
                elif team_token == away_token:
                    yes_by_team[away_token] = {"odds": yes_odds, "stake": yes_stake}

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
                            ),
                            _market_outcome_row(
                                away_team,
                                away_outcome["price"],
                                _safe_float(away_outcome.get("stake")),
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
                        ),
                        _market_outcome_row(
                            away_team,
                            away_data["odds"],
                            _safe_float(away_data.get("stake")),
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
                        ),
                        _market_outcome_row(
                            "Draw",
                            draw_yes["odds"],
                            _safe_float(draw_yes.get("stake")),
                        ),
                        _market_outcome_row(
                            away_team,
                            away_data["odds"],
                            _safe_float(away_data.get("stake")),
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
                    _market_outcome_row("Yes", btts_yes["odds"], _safe_float(btts_yes.get("stake"))),
                    _market_outcome_row("No", btts_no["odds"], _safe_float(btts_no.get("stake"))),
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
        "events_status_filtered_count": 0,
        "events_matchup_count": 0,
        "events_with_market_count": 0,
        "events_returned_count": 0,
        "clob_tokens_requested": 0,
        "clob_tokens_considered": 0,
        "clob_tokens_truncated": 0,
        "clob_books_fetched": 0,
        "clob_books_with_depth": 0,
        "clob_book_errors": 0,
        "pages_fetched": 0,
        "retries_used": 0,
        "payload_cache": "miss",
    }
    fetch_events.last_stats = stats

    supported_markets = _requested_market_keys(markets)
    if not ({"h2h", "h2h_3_way", "btts", "both_teams_to_score"} & supported_markets):
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
    now_utc = dt.datetime.now(dt.timezone.utc)
    stats["events_payload_count"] = len(payload)
    filtered_events: List[Tuple[dict, str, str]] = []
    clob_token_ids: List[str] = []
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

        matchup = _extract_matchup(event)
        if not matchup:
            continue
        home_team, away_team = matchup
        stats["events_matchup_count"] += 1
        filtered_events.append((event, home_team, away_team))

        for market in (event.get("markets") or []):
            if not isinstance(market, dict):
                continue
            if not _market_is_tradeable(market, now_utc):
                continue
            outcomes = _parse_list(market.get("outcomes"))
            prices = _parse_list(market.get("outcomePrices"))
            if len(outcomes) != 2 or len(prices) != 2:
                continue
            clob_token_ids.extend(_parse_clob_token_ids(market.get("clobTokenIds"))[:2])

    clob_depth_map: Dict[str, Optional[float]] = {}
    if clob_token_ids:
        clob_depth_map, clob_meta = _load_clob_depth_map(
            token_ids=clob_token_ids,
            retries=retries,
            backoff_seconds=backoff,
        )
        stats["retries_used"] += int(clob_meta.get("retries_used", 0) or 0)
        stats["clob_tokens_requested"] = int(clob_meta.get("token_count_requested", 0) or 0)
        stats["clob_tokens_considered"] = int(clob_meta.get("token_count_considered", 0) or 0)
        stats["clob_tokens_truncated"] = int(clob_meta.get("token_count_truncated", 0) or 0)
        stats["clob_books_fetched"] = int(clob_meta.get("books_fetched", 0) or 0)
        stats["clob_books_with_depth"] = int(clob_meta.get("books_with_depth", 0) or 0)
        stats["clob_book_errors"] = int(clob_meta.get("book_errors", 0) or 0)

    for event, home_team, away_team in filtered_events:
        market_list = _pick_match_markets(
            event,
            home_team,
            away_team,
            supported_markets,
            now_utc,
            clob_depth_map=clob_depth_map,
        )
        if not market_list:
            continue
        stats["events_with_market_count"] += 1

        commence = _normalize_commence_time(
            event.get("startTime")
        )
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
                "bookmakers": [
                    {
                        "key": PROVIDER_KEY,
                        "title": PROVIDER_TITLE,
                        "event_id": event_id,
                        "event_url": _event_url(event),
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
    "source": POLYMARKET_SOURCE or "api",
    "events_returned_count": 0,
}

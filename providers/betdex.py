from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests

PROVIDER_KEY = "betdex"
PROVIDER_TITLE = "BetDEX"

BETDEX_SOURCE = os.getenv("BETDEX_SOURCE", "api").strip().lower()
BETDEX_SAMPLE_PATH = os.getenv("BETDEX_SAMPLE_PATH", str(Path("data") / "betdex_sample.json")).strip()
BETDEX_SESSION_URL = os.getenv("BETDEX_SESSION_URL", "https://www.betdex.com/api/session").strip()
BETDEX_MONACO_API_BASE = os.getenv("BETDEX_MONACO_API_BASE", "https://production.api.monacoprotocol.xyz").strip()
BETDEX_PUBLIC_BASE = os.getenv("BETDEX_PUBLIC_BASE", "https://www.betdex.com").strip()
BETDEX_TIMEOUT_RAW = os.getenv("BETDEX_TIMEOUT_SECONDS", "20").strip()
BETDEX_RETRIES_RAW = os.getenv("BETDEX_RETRIES", "2").strip()
BETDEX_RETRY_BACKOFF_RAW = os.getenv("BETDEX_RETRY_BACKOFF", "0.5").strip()
BETDEX_EVENTS_PAGE_SIZE_RAW = os.getenv("BETDEX_EVENTS_PAGE_SIZE", "250").strip()
BETDEX_EVENTS_MAX_PAGES_RAW = os.getenv("BETDEX_EVENTS_MAX_PAGES", "8").strip()
BETDEX_MARKETS_PAGE_SIZE_RAW = os.getenv("BETDEX_MARKETS_PAGE_SIZE", "500").strip()
BETDEX_MARKETS_MAX_PAGES_RAW = os.getenv("BETDEX_MARKETS_MAX_PAGES", "8").strip()
BETDEX_EVENT_BATCH_SIZE_RAW = os.getenv("BETDEX_EVENT_BATCH_SIZE", "60").strip()
BETDEX_PRICE_BATCH_SIZE_RAW = os.getenv("BETDEX_PRICE_BATCH_SIZE", "120").strip()
BETDEX_MARKET_STATUSES_RAW = os.getenv("BETDEX_MARKET_STATUSES", "Open").strip()
BETDEX_USER_AGENT = os.getenv(
    "BETDEX_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
).strip()

SUPPORTED_MARKETS = {"h2h", "spreads", "totals"}

SPORT_SUBCATEGORY_DEFAULTS: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("AMFOOT",),
    "basketball_nba": ("BBALL",),
    "basketball_ncaab": ("BBALL",),
    "baseball_mlb": ("BASEBALL",),
    "icehockey_nhl": ("ICEHKY",),
    "soccer_epl": ("FOOTBALL",),
    "soccer_spain_la_liga": ("FOOTBALL",),
    "soccer_germany_bundesliga": ("FOOTBALL",),
    "soccer_italy_serie_a": ("FOOTBALL",),
    "soccer_france_ligue_one": ("FOOTBALL",),
    "soccer_usa_mls": ("FOOTBALL",),
}

SPORT_LEAGUE_HINTS: Dict[str, Sequence[str]] = {
    "americanfootball_nfl": ("nfl",),
    "basketball_nba": ("nba",),
    "basketball_ncaab": ("ncaab", "ncaa", "college"),
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
) -> Tuple[object, int]:
    retriable = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    last_error: Optional[ProviderError] = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                params=params,
                headers=_headers(access_token),
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


def _normalize_commence_time(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts /= 1000.0
            return dt.datetime.utcfromtimestamp(ts).replace(microsecond=0).isoformat() + "Z"
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


def _fetch_access_token(retries: int, backoff_seconds: float, timeout: int) -> Tuple[str, int]:
    payload, retries_used = _request_json(
        _session_url(),
        params=None,
        access_token=None,
        retries=retries,
        backoff_seconds=backoff_seconds,
        timeout=timeout,
    )
    if not isinstance(payload, dict):
        raise ProviderError("BetDEX session endpoint returned an invalid payload")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or not sessions or not isinstance(sessions[0], dict):
        raise ProviderError("BetDEX session endpoint returned no sessions")
    token = _normalize_text(sessions[0].get("accessToken"))
    if not token:
        raise ProviderError("BetDEX session endpoint returned an empty access token")
    return token, retries_used


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
        pages_fetched += 1
        retries_used += attempt
        if not isinstance(payload, dict):
            continue
        for entry in payload.get("prices") or []:
            if not isinstance(entry, dict):
                continue
            market_id = _normalize_text(entry.get("marketId"))
            if market_id:
                out[market_id] = entry
    return out, {"pages_fetched": pages_fetched, "retries_used": retries_used}

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


def _best_for_prices(entry: Optional[dict]) -> Dict[str, float]:
    best: Dict[str, float] = {}
    if not isinstance(entry, dict):
        return best
    prices = entry.get("prices")
    if not isinstance(prices, list):
        return best
    for item in prices:
        if not isinstance(item, dict):
            continue
        if _normalize_text(item.get("side")).lower() != "for":
            continue
        outcome_id = _normalize_text(item.get("outcomeId"))
        price = _safe_float(item.get("price"))
        if not outcome_id or price is None or price <= 1:
            continue
        if outcome_id not in best or price > best[outcome_id]:
            best[outcome_id] = price
    return best


def _market_type_id(market: dict) -> str:
    return _normalize_text(_doc_ref_first_id(market.get("marketType"))).upper()


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


def _event_url(event: dict, event_groups_by_id: Dict[str, dict]) -> str:
    code = _normalize_text(event.get("code"))
    event_id = _normalize_text(event.get("id"))
    if not code and not event_id:
        return ""
    group_id = _doc_ref_first_id(event.get("eventGroup"))
    event_group = event_groups_by_id.get(group_id or "")
    subcategory_id = _doc_ref_first_id(event_group.get("subcategory")) if isinstance(event_group, dict) else None
    if code and group_id and subcategory_id:
        return (
            f"{_public_base()}/events/{quote(subcategory_id.lower(), safe='')}"
            f"/{quote(group_id.lower(), safe='')}/{quote(code.lower(), safe='')}"
        )
    return f"{_public_base()}/events/{quote((code or event_id).lower(), safe='')}"


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

def fetch_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    _ = regions
    requested_markets = {
        _normalize_text(item).lower()
        for item in (markets or [])
        if _normalize_text(item)
    } & SUPPORTED_MARKETS

    stats = {
        "provider": PROVIDER_KEY,
        "source": BETDEX_SOURCE or "api",
        "events_payload_count": 0,
        "events_sport_filtered_count": 0,
        "markets_payload_count": 0,
        "prices_payload_count": 0,
        "events_with_market_count": 0,
        "events_returned_count": 0,
        "pages_fetched": 0,
        "retries_used": 0,
    }
    fetch_events.last_stats = stats

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
        fetch_events.last_stats = stats
        return normalized
    if source != "api":
        raise ProviderError("BetDEX provider supports BETDEX_SOURCE=api or BETDEX_SOURCE=file")

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

    token, retries_used = _fetch_access_token(retries=retries, backoff_seconds=backoff, timeout=timeout)
    stats["retries_used"] += retries_used

    events_dataset, events_meta = _fetch_events_dataset(
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
        all_events_dataset, all_events_meta = _fetch_events_dataset(
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
        fetch_events.last_stats = stats
        return []

    stats["events_sport_filtered_count"] = len(filtered_events)

    event_ids = list(dict.fromkeys(_normalize_text(event.get("id")) for event in filtered_events if _normalize_text(event.get("id"))))
    markets_dataset, markets_meta = _fetch_markets_dataset(
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
    if market_ids:
        prices_by_market_id, prices_meta = _fetch_prices_by_market(
            token=token,
            market_ids=market_ids,
            retries=retries,
            backoff_seconds=backoff,
            timeout=timeout,
            batch_size=price_batch_size,
        )
        stats["pages_fetched"] += int(prices_meta.get("pages_fetched", 0) or 0)
        stats["retries_used"] += int(prices_meta.get("retries_used", 0) or 0)
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

        best_h2h: Optional[dict] = None
        by_signature: Dict[str, dict] = {}

        for market in event_markets:
            market_id = _normalize_text(market.get("id"))
            market_type = _market_type_id(market)
            prices_by_outcome = _best_for_prices(prices_by_market_id.get(market_id))

            outcomes = []
            for outcome_id in _doc_ref_ids(market.get("marketOutcomes")):
                outcome = outcomes_by_id.get(outcome_id)
                price = prices_by_outcome.get(outcome_id)
                if not isinstance(outcome, dict) or price is None or price <= 1:
                    continue
                title = _clean_team_name(outcome.get("title"))
                if not title:
                    continue
                outcomes.append({"title": title, "price": round(float(price), 6)})

            if len(outcomes) < 2:
                continue

            if "h2h" in requested_markets and len(outcomes) == 2:
                if "HANDICAP" not in market_type and "OVER_UNDER" not in market_type:
                    if "MONEYLINE" in market_type or "FULL_TIME_RESULT" in market_type or "MATCH_RESULT" in market_type:
                        titles = {_normalize_token(item["title"]) for item in outcomes}
                        if "draw" not in titles:
                            out_by_token = {_normalize_token(item["title"]): item for item in outcomes}
                            home_out = out_by_token.get(_normalize_token(home_team))
                            away_out = out_by_token.get(_normalize_token(away_team))
                            if home_out and away_out:
                                h2h = {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": home_team, "price": home_out["price"]},
                                        {"name": away_team, "price": away_out["price"]},
                                    ],
                                }
                            else:
                                h2h = {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": outcomes[0]["title"], "price": outcomes[0]["price"]},
                                        {"name": outcomes[1]["title"], "price": outcomes[1]["price"]},
                                    ],
                                }
                            if best_h2h is None or _score_market(h2h) > _score_market(best_h2h):
                                best_h2h = h2h

            if "spreads" in requested_markets and len(outcomes) == 2 and "HANDICAP" in market_type:
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
                        }
                    )
                if len(spread_outcomes) == 2:
                    spread = {"key": "spreads", "outcomes": spread_outcomes}
                    sig = _market_signature(spread)
                    prev = by_signature.get(sig)
                    if prev is None or _score_market(spread) > _score_market(prev):
                        by_signature[sig] = spread

            if "totals" in requested_markets and len(outcomes) == 2 and "OVER_UNDER" in market_type:
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
                    }
                if "Over" in totals and "Under" in totals:
                    if abs(float(totals["Over"]["point"]) - float(totals["Under"]["point"])) <= 1e-6:
                        total_market = {"key": "totals", "outcomes": [totals["Over"], totals["Under"]]}
                        sig = _market_signature(total_market)
                        prev = by_signature.get(sig)
                        if prev is None or _score_market(total_market) > _score_market(prev):
                            by_signature[sig] = total_market

        market_list: List[dict] = []
        if best_h2h is not None:
            market_list.append(best_h2h)
        market_list.extend(by_signature.values())

        if not market_list:
            continue

        stats["events_with_market_count"] += 1
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
                        "event_url": _event_url(event, event_groups_by_id),
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
    "source": BETDEX_SOURCE or "api",
    "events_payload_count": 0,
    "events_sport_filtered_count": 0,
    "markets_payload_count": 0,
    "prices_payload_count": 0,
    "events_with_market_count": 0,
    "events_returned_count": 0,
    "pages_fetched": 0,
    "retries_used": 0,
}

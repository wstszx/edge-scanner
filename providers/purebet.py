from __future__ import annotations

import concurrent.futures
import datetime as dt
import difflib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests


class ProviderError(Exception):
    """Raised for provider-specific recoverable issues."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


PUREBET_BOOK_KEY = "purebet"

PUREBET_TITLE = "Purebet"

PUREBET_SOURCE = os.getenv("PUREBET_SOURCE", "api").strip().lower()

PUREBET_SAMPLE_PATH = os.getenv(
    "PUREBET_SAMPLE_PATH", str(Path("data") / "purebet_sample.json")
).strip()

PUREBET_API_BASE = os.getenv("PUREBET_API_BASE", "").strip()

PUREBET_DEFAULT_BASE = "https://v3api.purebet.io"

PUREBET_LIVE = os.getenv("PUREBET_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}

PUREBET_USER_AGENT = os.getenv(
    "PUREBET_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
).strip()

PUREBET_ORIGIN = os.getenv("PUREBET_ORIGIN", "https://purebet.io").strip()

PUREBET_REFERER = os.getenv("PUREBET_REFERER", "https://purebet.io/").strip()

PUREBET_MIN_STAKE_RAW = os.getenv("PUREBET_MIN_STAKE", "50").strip()

PUREBET_MAX_AGE_SECONDS_RAW = os.getenv("PUREBET_MAX_AGE_SECONDS", "60").strip()

PUREBET_MARKET_WORKERS_RAW = os.getenv("PUREBET_MARKET_WORKERS", "8").strip()

PUREBET_MARKET_RETRIES_RAW = os.getenv("PUREBET_MARKET_RETRIES", "2").strip()

PUREBET_RETRY_BACKOFF_RAW = os.getenv("PUREBET_RETRY_BACKOFF", "0.4").strip()

PUREBET_MARKETS_ENABLED = os.getenv("PUREBET_MARKETS_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

PUREBET_LEAGUE_SYNC_ENABLED = os.getenv(
    "PUREBET_LEAGUE_SYNC_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}

PUREBET_LEAGUE_SYNC_TTL_RAW = os.getenv("PUREBET_LEAGUE_SYNC_TTL", "600").strip()

PUREBET_DEFAULT_LEAGUE_MAP = {
    487: "basketball_nba",
    493: "basketball_ncaab",
    889: "americanfootball_nfl",
    1980: "soccer_epl",
    2196: "soccer_spain_la_liga",
    1842: "soccer_germany_bundesliga",
    2436: "soccer_italy_serie_a",
    2036: "soccer_france_ligue_one",
    2663: "soccer_usa_mls",
}

PUREBET_LEAGUE_MAP_RAW = os.getenv("PUREBET_LEAGUE_MAP", "").strip()

PUREBET_SUPPORTED_MARKETS = {"h2h", "spreads", "totals"}

PUREBET_ACTIVE_LEAGUES_CACHE: Dict[str, object] = {
    "expires_at": 0.0,
    "mapping": {},
    "meta": {},
}

def _purebet_public_base() -> str:
    base = (PUREBET_ORIGIN or "").strip() or "https://purebet.io"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")

def _purebet_event_url(event_id: object) -> str:
    raw = f"{event_id or ''}".strip()
    if not raw:
        return ""
    return f"{_purebet_public_base()}/event/{quote(raw, safe='')}"

def _purebet_headers() -> Dict[str, str]:
    headers = {}
    if PUREBET_ORIGIN:
        headers["Origin"] = PUREBET_ORIGIN
    if PUREBET_REFERER:
        headers["Referer"] = PUREBET_REFERER
    if PUREBET_USER_AGENT:
        headers["User-Agent"] = PUREBET_USER_AGENT
    return headers

def _purebet_min_stake() -> float:
    try:
        return max(0.0, float(PUREBET_MIN_STAKE_RAW))
    except (TypeError, ValueError):
        return 0.0

def _purebet_max_age_seconds() -> int:
    try:
        return max(0, int(float(PUREBET_MAX_AGE_SECONDS_RAW)))
    except (TypeError, ValueError):
        return 0

def _purebet_market_workers() -> int:
    try:
        return max(1, int(float(PUREBET_MARKET_WORKERS_RAW)))
    except (TypeError, ValueError):
        return 8

def _purebet_market_retries() -> int:
    try:
        return max(0, int(float(PUREBET_MARKET_RETRIES_RAW)))
    except (TypeError, ValueError):
        return 2

def _purebet_retry_backoff() -> float:
    try:
        return max(0.0, float(PUREBET_RETRY_BACKOFF_RAW))
    except (TypeError, ValueError):
        return 0.4

def _purebet_league_sync_ttl() -> int:
    try:
        return max(0, int(float(PUREBET_LEAGUE_SYNC_TTL_RAW)))
    except (TypeError, ValueError):
        return 600

def _purebet_get_json(
    url: str,
    params: Dict[str, object],
    headers: Dict[str, str],
    retries: int,
    backoff_seconds: float,
    timeout: int = 30,
) -> Tuple[object, int]:
    """Return (json_payload, retries_used). Raises ProviderError on final failure."""
    last_error: Optional[ProviderError] = None
    retriable_status = {429, 500, 502, 503, 504}
    attempts = max(0, retries) + 1
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_error = ProviderError(f"Purebet network error: {exc}")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        if response.status_code >= 400:
            if response.status_code in retriable_status and attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise ProviderError(
                f"Purebet API request failed ({response.status_code})",
                status_code=response.status_code,
            )
        try:
            return response.json(), attempt
        except ValueError as exc:
            last_error = ProviderError("Failed to parse Purebet API response")
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
    if last_error:
        raise last_error
    raise ProviderError("Purebet request failed")

def _load_event_list(path: str) -> List[dict]:
    if not path:
        raise ProviderError("Purebet source file path is empty")
    path_obj = Path(path)
    if not path_obj.exists():
        raise ProviderError(f"Purebet source file not found: {path}")
    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ProviderError(f"Failed to read Purebet source file: {exc}") from exc
    if not isinstance(payload, list):
        raise ProviderError("Purebet source file must be a JSON array of events")
    return [item for item in payload if isinstance(item, dict)]

def _normalize_purebet_events(events: List[dict]) -> List[dict]:
    normalized: List[dict] = []
    for event in events:
        event_id = (
            event.get("event")
            or event.get("_id")
            or event.get("eventId")
            or event.get("id")
        )
        event_url = _purebet_event_url(event_id)
        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        for book in bookmakers:
            if not isinstance(book, dict):
                continue
            if not book.get("key"):
                book["key"] = PUREBET_BOOK_KEY
            if not book.get("title"):
                book["title"] = PUREBET_TITLE
            if event_id and not (book.get("event_id") or book.get("eventId") or book.get("id")):
                book["event_id"] = event_id
            if event_url and not (book.get("event_url") or book.get("eventUrl") or book.get("url")):
                book["event_url"] = event_url
        normalized.append(event)
    return normalized

def _base_purebet_league_map() -> Dict[str, str]:
    mapping = {str(key): value for key, value in PUREBET_DEFAULT_LEAGUE_MAP.items()}
    if not PUREBET_LEAGUE_MAP_RAW:
        return mapping
    try:
        payload = json.loads(PUREBET_LEAGUE_MAP_RAW)
    except ValueError:
        return mapping
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str) and value.strip():
                mapping[str(key)] = value.strip()
    return mapping

def _infer_sport_key_from_active_league(league: dict) -> Optional[str]:
    if not isinstance(league, dict):
        return None
    sport_name = _normalize_text(league.get("sportName"))
    sport_id = str(league.get("sport") or "")
    league_name = _normalize_text(league.get("name"))
    abbr = _normalize_text(league.get("abbr"))
    country = _normalize_text(league.get("country"))
    haystack = " ".join(value for value in (league_name, abbr, country) if value)
    if not haystack:
        return None

    if "nfl" in haystack:
        return "americanfootball_nfl"
    if "nba" in haystack:
        return "basketball_nba"
    if "ncaa" in haystack and "basket" in haystack:
        return "basketball_ncaab"

    is_soccer = sport_name == "soccer" or sport_id == "29"
    if is_soccer:
        if "premier league" in haystack and ("eng" in abbr or "england" in haystack):
            return "soccer_epl"
        if "la liga" in haystack or ("spain" in haystack and "liga" in haystack):
            return "soccer_spain_la_liga"
        if "bundesliga" in haystack or ("germany" in haystack and "bundesliga" in haystack):
            return "soccer_germany_bundesliga"
        if "serie a" in haystack or ("italy" in haystack and "serie" in haystack):
            return "soccer_italy_serie_a"
        if "ligue 1" in haystack or ("france" in haystack and "ligue" in haystack):
            return "soccer_france_ligue_one"
        if "mls" in haystack or "major league soccer" in haystack:
            return "soccer_usa_mls"

    is_basketball = sport_name == "basketball" or sport_id == "4"
    if is_basketball:
        if "nba" in haystack:
            return "basketball_nba"
        if "ncaa" in haystack:
            return "basketball_ncaab"
    return None

def _build_dynamic_purebet_league_map(
    leagues: Sequence[dict], base_mapping: Dict[str, str]
) -> Tuple[Dict[str, str], dict]:
    dynamic: Dict[str, str] = {}
    unresolved_samples: List[str] = []
    total = 0
    inferred = 0
    unresolved = 0
    for league in leagues:
        if not isinstance(league, dict):
            continue
        league_id = league.get("_id") or league.get("id")
        if league_id is None:
            continue
        total += 1
        key = str(league_id)
        if key in base_mapping:
            continue
        inferred_sport = _infer_sport_key_from_active_league(league)
        if inferred_sport:
            dynamic[key] = inferred_sport
            inferred += 1
            continue
        unresolved += 1
        if len(unresolved_samples) < 5:
            name = league.get("name") or league.get("abbr") or key
            unresolved_samples.append(f"{key}:{name}")
    return dynamic, {
        "total": total,
        "inferred": inferred,
        "unresolved": unresolved,
        "unresolved_samples": unresolved_samples,
    }

def _load_purebet_league_map(
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    stats: Optional[dict] = None,
) -> Dict[str, str]:
    mapping = _base_purebet_league_map()
    if stats is not None:
        stats["league_sync_enabled"] = bool(PUREBET_LEAGUE_SYNC_ENABLED)
    if not PUREBET_LEAGUE_SYNC_ENABLED:
        if stats is not None:
            stats["league_sync_source"] = "disabled"
        return mapping
    if not base_url:
        if stats is not None:
            stats["league_sync_source"] = "no_base_url"
        return mapping
    now = time.time()
    ttl = _purebet_league_sync_ttl()
    cache_valid = ttl > 0 and now < float(PUREBET_ACTIVE_LEAGUES_CACHE.get("expires_at", 0.0))
    cached_mapping = PUREBET_ACTIVE_LEAGUES_CACHE.get("mapping") or {}
    cached_meta = PUREBET_ACTIVE_LEAGUES_CACHE.get("meta") or {}
    if cache_valid and isinstance(cached_mapping, dict):
        mapping.update(cached_mapping)
        if stats is not None:
            stats["league_sync_source"] = "cache"
            stats["league_sync_total_leagues"] = int(cached_meta.get("total", 0) or 0)
            stats["league_sync_dynamic_added"] = len(cached_mapping)
            stats["league_sync_unresolved"] = int(cached_meta.get("unresolved", 0) or 0)
            stats["league_sync_unresolved_samples"] = list(cached_meta.get("unresolved_samples", []))
        return mapping

    leagues_url = f"{base_url.rstrip('/')}/activeLeagues"
    retries = _purebet_market_retries()
    backoff = _purebet_retry_backoff()
    try:
        payload, retries_used = _purebet_get_json(
            leagues_url,
            {},
            headers or _purebet_headers(),
            retries=retries,
            backoff_seconds=backoff,
            timeout=30,
        )
    except ProviderError as exc:
        if isinstance(cached_mapping, dict):
            mapping.update(cached_mapping)
        if stats is not None:
            stats["league_sync_source"] = "stale_cache" if cached_mapping else "error"
            stats["league_sync_error"] = str(exc)
            stats["league_sync_dynamic_added"] = len(cached_mapping) if isinstance(cached_mapping, dict) else 0
        return mapping

    if not isinstance(payload, list):
        if isinstance(cached_mapping, dict):
            mapping.update(cached_mapping)
        if stats is not None:
            stats["league_sync_source"] = "stale_cache" if cached_mapping else "invalid_payload"
            stats["league_sync_error"] = "Purebet activeLeagues response must be a JSON array"
            stats["league_sync_dynamic_added"] = len(cached_mapping) if isinstance(cached_mapping, dict) else 0
        return mapping

    dynamic_mapping, meta = _build_dynamic_purebet_league_map(payload, mapping)
    expires_at = now + ttl if ttl > 0 else now
    PUREBET_ACTIVE_LEAGUES_CACHE["expires_at"] = expires_at
    PUREBET_ACTIVE_LEAGUES_CACHE["mapping"] = dynamic_mapping
    PUREBET_ACTIVE_LEAGUES_CACHE["meta"] = meta
    mapping.update(dynamic_mapping)
    if stats is not None:
        stats["league_sync_source"] = "live"
        stats["league_sync_retries"] = retries_used
        stats["league_sync_total_leagues"] = int(meta.get("total", 0) or 0)
        stats["league_sync_dynamic_added"] = len(dynamic_mapping)
        stats["league_sync_unresolved"] = int(meta.get("unresolved", 0) or 0)
        stats["league_sync_unresolved_samples"] = list(meta.get("unresolved_samples", []))
    return mapping

def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip().lower()

def _normalize_team_name(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    tokens = [token for token in normalized.split() if token]
    drop_tokens = {
        "fc",
        "cf",
        "sc",
        "ac",
        "afc",
        "u23",
        "u21",
        "u20",
        "u19",
        "u18",
        "u17",
        "women",
        "woman",
        "ladies",
        "reserves",
        "ii",
        "iii",
    }
    cleaned = [token for token in tokens if token not in drop_tokens]
    return " ".join(cleaned)

def _team_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def _epoch_to_iso(value: float) -> Optional[str]:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    if timestamp > 1e12:
        timestamp /= 1000.0
    try:
        return dt.datetime.utcfromtimestamp(timestamp).replace(microsecond=0).isoformat() + "Z"
    except (OSError, OverflowError, ValueError):
        return None

def _normalize_commence_time(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _epoch_to_iso(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return _epoch_to_iso(int(text))
        try:
            if text.endswith("Z"):
                dt.datetime.fromisoformat(text[:-1])
                return text
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return (
            parsed.astimezone(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    return None

def _resolve_purebet_sport_key(event: dict, league_map: Dict[str, str]) -> Optional[str]:
    for key in ("sport_key", "sportKey", "sport"):
        raw = event.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    league_id = event.get("leagueId") or event.get("league") or event.get("league_id")
    if league_id is None:
        return None
    return league_map.get(str(league_id))

def _is_moneyline_market(market_type: str) -> bool:
    if not market_type:
        return False
    for token in ("moneyline", "ml", "h2h", "match_winner", "matchwinner", "winner", "win"):
        if token in market_type:
            return True
    return False

def _normalize_purebet_market_type(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    if text in {"ah", "asian handicap", "asian_handicap", "handicap"}:
        return "AH"
    if text in {"ou", "over/under", "over_under", "totals", "total"}:
        return "OU"
    if text in {"1x2", "match winner", "match_winner", "matchwinner", "winner", "h2h", "ml", "moneyline"}:
        return "H2H"
    if text in {"btts", "both teams to score", "both_teams_to_score"}:
        return "BTTS"
    return text.upper()

def _parse_purebet_market_value(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text:
            return None
        if "/" in text:
            parts = [part for part in text.split("/") if part]
            if parts:
                text = parts[0]
        try:
            return float(text)
        except ValueError:
            return None
    return None

def _purebet_is_recent(last_updated, max_age_seconds: int, now_epoch: int) -> bool:
    if max_age_seconds <= 0:
        return True
    timestamp = _safe_float(last_updated)
    if timestamp is None:
        return True
    if timestamp > 1e12:
        timestamp /= 1000.0
    age = now_epoch - int(timestamp)
    if age < 0:
        return True
    return age <= max_age_seconds

def _select_purebet_side(
    entries: Optional[Sequence[dict]], min_stake: float, max_age_seconds: int, now_epoch: int
) -> Optional[dict]:
    if not isinstance(entries, list):
        return None
    best = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        odds = _safe_float(entry.get("odds") or entry.get("price") or entry.get("decimalOdds"))
        if odds is None or odds <= 1:
            continue
        stake = _safe_float(entry.get("stake") or entry.get("maxStake") or entry.get("liquidity"))
        if min_stake > 0 and (stake is None or stake < min_stake):
            continue
        last_updated = entry.get("lastUpdated") or entry.get("last_update") or entry.get("timestamp")
        if not _purebet_is_recent(last_updated, max_age_seconds, now_epoch):
            continue
        if best is None or odds > best["odds"]:
            best = {"odds": odds, "stake": stake, "last_updated": last_updated}
    return best

def _normalize_purebet_markets_payload(
    payload: Sequence[dict],
    home: str,
    away: str,
    supported_markets: Sequence[str],
) -> List[dict]:
    if not isinstance(payload, list):
        return []
    supported = set(supported_markets or [])
    if not supported:
        return []
    min_stake = _purebet_min_stake()
    max_age_seconds = _purebet_max_age_seconds()
    now_epoch = int(dt.datetime.utcnow().timestamp())
    normalized: List[dict] = []
    for market in payload:
        if not isinstance(market, dict):
            continue
        period = market.get("period")
        if period not in (None, 1, "1"):
            continue
        market_type = _normalize_purebet_market_type(
            market.get("marketType") or market.get("type") or market.get("market_type")
        )
        if not market_type:
            continue
        if market_type == "AH" and "spreads" not in supported:
            continue
        if market_type == "OU" and "totals" not in supported:
            continue
        if market_type == "H2H" and "h2h" not in supported:
            continue
        if market_type == "BTTS":
            continue
        side0 = _select_purebet_side(
            market.get("side0odds"), min_stake, max_age_seconds, now_epoch
        )
        side1 = _select_purebet_side(
            market.get("side1odds"), min_stake, max_age_seconds, now_epoch
        )
        side2 = _select_purebet_side(
            market.get("side2odds"), min_stake, max_age_seconds, now_epoch
        )
        if not side0 or not side1:
            continue
        market_value = _parse_purebet_market_value(
            market.get("marketValue") or market.get("value") or market.get("market_value")
        )
        if market_type == "AH":
            if market_value is None:
                continue
            home_point = float(market_value)
            away_point = -home_point
            normalized.append(
                {
                    "key": "spreads",
                    "outcomes": [
                        {
                            "name": home,
                            "price": side0["odds"],
                            "point": home_point,
                            "stake": side0.get("stake"),
                            "last_updated": side0.get("last_updated"),
                        },
                        {
                            "name": away,
                            "price": side1["odds"],
                            "point": away_point,
                            "stake": side1.get("stake"),
                            "last_updated": side1.get("last_updated"),
                        },
                    ],
                }
            )
        elif market_type == "OU":
            if market_value is None:
                continue
            total = float(market_value)
            normalized.append(
                {
                    "key": "totals",
                    "outcomes": [
                        {
                            "name": "Over",
                            "price": side0["odds"],
                            "point": total,
                            "stake": side0.get("stake"),
                            "last_updated": side0.get("last_updated"),
                        },
                        {
                            "name": "Under",
                            "price": side1["odds"],
                            "point": total,
                            "stake": side1.get("stake"),
                            "last_updated": side1.get("last_updated"),
                        },
                    ],
                }
            )
        elif market_type == "H2H":
            outcomes = [
                {
                    "name": home,
                    "price": side0["odds"],
                    "stake": side0.get("stake"),
                    "last_updated": side0.get("last_updated"),
                },
                {
                    "name": away,
                    "price": side1["odds"],
                    "stake": side1.get("stake"),
                    "last_updated": side1.get("last_updated"),
                },
            ]
            if side2:
                outcomes.append(
                    {
                        "name": "Draw",
                        "price": side2["odds"],
                        "stake": side2.get("stake"),
                        "last_updated": side2.get("last_updated"),
                    }
                )
            normalized.append({"key": "h2h", "outcomes": outcomes})
    return normalized

def _fetch_purebet_event_markets(
    base_url: str,
    event: dict,
    supported_markets: Sequence[str],
    headers: Dict[str, str],
    retries: int,
    backoff_seconds: float,
) -> dict:
    event_id = event.get("id")
    if not event_id:
        return {"event_id": None, "markets": [], "retries_used": 0, "error": "missing_event_id"}
    market_url = f"{base_url.rstrip('/')}/markets"
    try:
        payload, retries_used = _purebet_get_json(
            market_url,
            {"event": event_id},
            headers,
            retries=retries,
            backoff_seconds=backoff_seconds,
            timeout=30,
        )
    except ProviderError as exc:
        return {
            "event_id": event_id,
            "markets": [],
            "retries_used": retries,
            "error": str(exc),
        }
    if not isinstance(payload, list):
        return {
            "event_id": event_id,
            "markets": [],
            "retries_used": retries_used,
            "error": "invalid_markets_payload",
        }
    markets = _normalize_purebet_markets_payload(
        payload,
        event.get("home_team") or "",
        event.get("away_team") or "",
        supported_markets,
    )
    return {
        "event_id": event_id,
        "markets": markets,
        "retries_used": retries_used,
        "error": None,
    }

def _normalize_purebet_h2h_markets(odds: Sequence[dict], home: str, away: str) -> List[dict]:
    groups: Dict[str, dict] = {}
    for item in odds:
        if not isinstance(item, dict):
            continue
        market = item.get("market") if isinstance(item.get("market"), dict) else {}
        market_id = market.get("id") or item.get("marketId") or "default"
        market_type = _normalize_text(
            market.get("type") or market.get("marketType") or item.get("marketType")
        )
        side = market.get("side") if "side" in market else item.get("side")
        try:
            side_val = int(side)
        except (TypeError, ValueError):
            continue
        if side_val not in (0, 1):
            continue
        price = _safe_float(item.get("odds") or item.get("price") or item.get("decimalOdds"))
        if price is None or price <= 1:
            continue
        point = market.get("point") if isinstance(market, dict) else None
        if point is None:
            point = item.get("point")
        if point not in (None, 0, 0.0, "0", "0.0"):
            if not _is_moneyline_market(market_type):
                continue
        group = groups.setdefault(str(market_id), {"type": market_type, "sides": {}})
        if market_type and not group["type"]:
            group["type"] = market_type
        group["sides"][side_val] = {"price": price}

    candidates = [
        group
        for group in groups.values()
        if 0 in group["sides"] and 1 in group["sides"]
    ]
    if not candidates:
        return []
    moneyline = [group for group in candidates if _is_moneyline_market(group["type"])]
    if moneyline:
        candidates = moneyline
    best = max(
        candidates,
        key=lambda group: min(group["sides"][0]["price"], group["sides"][1]["price"]),
    )
    return [
        {
            "key": "h2h",
            "outcomes": [
                {"name": home, "price": best["sides"][0]["price"]},
                {"name": away, "price": best["sides"][1]["price"]},
            ],
        }
    ]

def _normalize_purebet_v3_events(
    payload: Sequence[dict],
    sport_key: str,
    markets: Sequence[str],
    base_url: Optional[str] = None,
    league_map: Optional[Dict[str, str]] = None,
    allow_empty_markets: bool = False,
) -> List[dict]:
    supported_markets = PUREBET_SUPPORTED_MARKETS.intersection(markets or [])
    if not supported_markets:
        return []
    if league_map is None:
        league_map = _load_purebet_league_map()
    normalized: List[dict] = []
    for event in payload:
        if not isinstance(event, dict):
            continue
        event_sport_key = _resolve_purebet_sport_key(event, league_map)
        if not event_sport_key:
            continue
        if sport_key and event_sport_key != sport_key:
            continue
        home = (event.get("homeTeam") or event.get("home_team") or "").strip()
        away = (event.get("awayTeam") or event.get("away_team") or "").strip()
        commence = _normalize_commence_time(
            event.get("startTime") or event.get("start_time") or event.get("start")
        )
        if not (home and away and commence):
            continue
        event_id = (
            event.get("event")
            or event.get("_id")
            or event.get("eventId")
            or event.get("id")
        )
        event_url = _purebet_event_url(event_id)
        markets_out: List[dict] = []
        odds = event.get("odds")
        if isinstance(odds, list) and "h2h" in supported_markets:
            markets_out.extend(_normalize_purebet_h2h_markets(odds, home, away))
        if not markets_out and not allow_empty_markets:
            continue
        normalized.append(
            {
                "id": event_id,
                "sport_key": event_sport_key,
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "league_id": event.get("leagueId") or event.get("league") or event.get("league_id"),
                "bookmakers": [
                    {
                        "key": PUREBET_BOOK_KEY,
                        "title": PUREBET_TITLE,
                        "event_id": event_id,
                        "event_url": event_url,
                        "markets": markets_out,
                    }
                ],
            }
        )
    return normalized

def fetch_events(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
) -> List[dict]:
    source = PUREBET_SOURCE or "api"
    stats = {
        "source": source,
        "events_payload_count": 0,
        "events_normalized_count": 0,
        "events_returned_count": 0,
        "details_enabled": False,
        "details_requested": 0,
        "details_success": 0,
        "details_failed": 0,
        "details_empty": 0,
        "details_retries": 0,
        "details_workers": 0,
        "details_error_samples": [],
        "league_sync_enabled": False,
        "league_sync_source": "not_used",
        "league_sync_total_leagues": 0,
        "league_sync_dynamic_added": 0,
        "league_sync_unresolved": 0,
        "league_sync_unresolved_samples": [],
    }
    fetch_events.last_stats = stats
    if bookmakers:
        lowered = {str(book).strip().lower() for book in bookmakers if isinstance(book, str)}
        if PUREBET_BOOK_KEY not in lowered and PUREBET_TITLE.lower() not in lowered:
            stats["events_returned_count"] = 0
            return []
    if source == "file":
        events = _normalize_purebet_events(_load_event_list(PUREBET_SAMPLE_PATH))
        stats["events_payload_count"] = len(events)
        stats["events_normalized_count"] = len(events)
    else:
        base_url = PUREBET_API_BASE or PUREBET_DEFAULT_BASE
        if not base_url:
            raise ProviderError(
                "Purebet API base URL not configured. Set PUREBET_API_BASE or use PUREBET_SOURCE=file."
            )
        url = f"{base_url.rstrip('/')}/events"
        params = {"live": "true" if PUREBET_LIVE else "false"}
        payload, retries_used = _purebet_get_json(
            url,
            params,
            _purebet_headers(),
            retries=_purebet_market_retries(),
            backoff_seconds=_purebet_retry_backoff(),
            timeout=30,
        )
        stats["details_retries"] += retries_used
        if not isinstance(payload, list):
            raise ProviderError("Purebet API response must be a JSON array of events")
        stats["events_payload_count"] = len(payload)
        supported_markets = PUREBET_SUPPORTED_MARKETS.intersection(markets or [])
        needs_details = (
            PUREBET_MARKETS_ENABLED
            and bool({"spreads", "totals"}.intersection(supported_markets))
        )
        stats["details_enabled"] = bool(needs_details)
        league_map = _load_purebet_league_map(
            base_url=base_url,
            headers=_purebet_headers(),
            stats=stats,
        )
        events = _normalize_purebet_v3_events(
            payload,
            sport_key,
            markets,
            base_url=base_url,
            league_map=league_map,
            allow_empty_markets=needs_details,
        )
        stats["events_normalized_count"] = len(events)
        if needs_details and events:
            workers = min(_purebet_market_workers(), len(events))
            retries = _purebet_market_retries()
            backoff = _purebet_retry_backoff()
            stats["details_workers"] = workers
            stats["details_requested"] = len(events)
            event_map = {str(event.get("id")): event for event in events if event.get("id")}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _fetch_purebet_event_markets,
                        base_url,
                        event,
                        supported_markets,
                        _purebet_headers(),
                        retries,
                        backoff,
                    )
                    for event in events
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        stats["details_failed"] += 1
                        if len(stats["details_error_samples"]) < 5:
                            stats["details_error_samples"].append(str(exc))
                        continue
                    stats["details_retries"] += int(result.get("retries_used") or 0)
                    event_id = result.get("event_id")
                    event_obj = event_map.get(str(event_id))
                    if event_obj is None:
                        continue
                    error = result.get("error")
                    extra_markets = result.get("markets") or []
                    if error:
                        stats["details_failed"] += 1
                        if len(stats["details_error_samples"]) < 5:
                            stats["details_error_samples"].append(str(error))
                        continue
                    if not extra_markets:
                        stats["details_empty"] += 1
                        continue
                    stats["details_success"] += 1
                    bookmakers_list = event_obj.get("bookmakers")
                    if not isinstance(bookmakers_list, list) or not bookmakers_list:
                        fallback_event_id = event_obj.get("id")
                        fallback_event_url = _purebet_event_url(fallback_event_id)
                        event_obj["bookmakers"] = [
                            {
                                "key": PUREBET_BOOK_KEY,
                                "title": PUREBET_TITLE,
                                "event_id": fallback_event_id,
                                "event_url": fallback_event_url,
                                "markets": [],
                            }
                        ]
                        bookmakers_list = event_obj["bookmakers"]
                    book = bookmakers_list[0]
                    markets_list = book.get("markets")
                    if not isinstance(markets_list, list):
                        markets_list = []
                    if any(m.get("key") == "h2h" for m in extra_markets):
                        markets_list = [m for m in markets_list if m.get("key") != "h2h"]
                    markets_list.extend(extra_markets)
                    book["markets"] = markets_list
    if sport_key:
        events = [event for event in events if event.get("sport_key") == sport_key]
    events = [
        event
        for event in events
        if any(
            isinstance(book, dict)
            and isinstance(book.get("markets"), list)
            and book.get("markets")
            for book in (event.get("bookmakers") or [])
        )
    ]
    if bookmakers:
        filtered = []
        for event in events:
            books = event.get("bookmakers") or []
            if not isinstance(books, list):
                continue
            kept = [
                book
                for book in books
                if (book.get("key") or "").strip() in bookmakers
                or (book.get("title") or "").strip() in bookmakers
            ]
            if kept:
                event["bookmakers"] = kept
                filtered.append(event)
        events = filtered
    stats["events_returned_count"] = len(events)
    fetch_events.last_stats = stats
    return events

PROVIDER_KEY = PUREBET_BOOK_KEY
PROVIDER_TITLE = PUREBET_TITLE
fetch_events.last_stats = {}

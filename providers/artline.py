from __future__ import annotations

import asyncio
import datetime as dt
import os
import re
from typing import Dict, List, Optional, Sequence

import httpx

from ._async_http import get_shared_client, request_json

PROVIDER_KEY = "artline"
PROVIDER_TITLE = "Artline"

ARTLINE_SOURCE = os.getenv("ARTLINE_SOURCE", "api").strip().lower()
ARTLINE_API_BASE = os.getenv("ARTLINE_API_BASE", "https://api.artline.bet/api").strip()
ARTLINE_PUBLIC_BASE = os.getenv("ARTLINE_PUBLIC_BASE", "https://artline.bet").strip()
ARTLINE_TIMEOUT_RAW = os.getenv("ARTLINE_TIMEOUT_SECONDS", "20").strip()
ARTLINE_RETRIES_RAW = os.getenv("ARTLINE_RETRIES", "2").strip()
ARTLINE_RETRY_BACKOFF_RAW = os.getenv("ARTLINE_RETRY_BACKOFF", "0.5").strip()
ARTLINE_DETAIL_MAX_CONCURRENCY_RAW = os.getenv("ARTLINE_DETAIL_MAX_CONCURRENCY", "8").strip()
ARTLINE_USER_AGENT = os.getenv(
    "ARTLINE_USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
).strip()

ARTLINE_SPORT_FILTERS: Dict[str, Dict[str, str]] = {
    "basketball_nba": {"sport": "basketball", "tournament_id": "17"},
    "basketball_euroleague": {"sport": "basketball", "tournament_id": "13"},
    "basketball_france_pro_a": {"sport": "basketball", "tournament_id": "652"},
    "icehockey_nhl": {"sport": "hockey", "tournament_id": "136"},
    "icehockey_ahl": {"sport": "hockey", "tournament_id": "567"},
    "soccer_epl": {"sport": "football", "tournament_id": "1"},
    "soccer_england_championship": {"sport": "football", "tournament_id": "49"},
    "soccer_england_league_one": {"sport": "football", "tournament_id": "41"},
    "soccer_england_league_two": {"sport": "football", "tournament_id": "42"},
    "soccer_portugal_primeira_liga": {"sport": "football", "tournament_id": "6"},
    "soccer_netherlands_eredivisie": {"sport": "football", "tournament_id": "7"},
    "soccer_argentina_liga_profesional": {"sport": "football", "tournament_id": "66"},
    "soccer_mexico_liga_mx": {"sport": "football", "tournament_id": "26"},
    "soccer_spain_la_liga": {"sport": "football", "tournament_id": "2"},
    "soccer_italy_serie_a": {"sport": "football", "tournament_id": "3"},
    "soccer_germany_bundesliga": {"sport": "football", "tournament_id": "4"},
    "soccer_france_ligue_one": {"sport": "football", "tournament_id": "5"},
    "soccer_usa_mls": {"sport": "football", "tournament_id": "52"},
    "basketball_germany_bbl": {"sport": "basketball", "tournament_id": "693"},
    "baseball_mlb": {"sport": "baseball", "tournament_id": "102"},
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


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_market_key(value: object) -> str:
    token = _normalize_text(value).lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


def _requested_market_keys(markets: Sequence[str]) -> set[str]:
    return {_normalize_market_key(item) for item in (markets or []) if _normalize_market_key(item)}


def _safe_float(value: object) -> Optional[float]:
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
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def _api_base() -> str:
    base = _normalize_text(ARTLINE_API_BASE) or "https://api.artline.bet/api"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    base = base.rstrip("/")
    if not base.endswith("/api"):
        base = f"{base}/api"
    return base


def _detail_max_concurrency() -> int:
    return _int_or_default(ARTLINE_DETAIL_MAX_CONCURRENCY_RAW, 8, min_value=1)


def _public_base() -> str:
    base = _normalize_text(ARTLINE_PUBLIC_BASE) or "https://artline.bet"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    return base.rstrip("/")


def _headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if ARTLINE_USER_AGENT:
        headers["User-Agent"] = ARTLINE_USER_AGENT
    return headers


def _supports_sport(sport_key: str) -> bool:
    return _normalize_text(sport_key).lower() in ARTLINE_SPORT_FILTERS


def _normalize_commence_time(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1e12:
            timestamp /= 1000.0
        try:
            return (
                dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except (OSError, OverflowError, ValueError):
            return None
    text = _normalize_text(value)
    if not text:
        return None
    if text.isdigit():
        return _normalize_commence_time(int(text))
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


def _payload_data(payload: object) -> object:
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


async def _request_json_async(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    params: Optional[dict] = None,
    json_payload: object = None,
    retries: Optional[int] = None,
    backoff_seconds: Optional[float] = None,
) -> tuple[object, int]:
    if retries is None:
        retries = _int_or_default(ARTLINE_RETRIES_RAW, 2, min_value=0)
    if backoff_seconds is None:
        backoff_seconds = _float_or_default(ARTLINE_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    timeout = _int_or_default(ARTLINE_TIMEOUT_RAW, 20, min_value=1)
    url = f"{_api_base()}/{path.lstrip('/')}"
    return await request_json(
        client,
        method,
        url,
        params=params,
        headers=_headers(),
        json_payload=json_payload,
        timeout=float(timeout),
        retries=retries,
        backoff_seconds=backoff_seconds,
        error_cls=ProviderError,
        network_error_prefix="Artline network error",
        parse_error_message="Failed to parse Artline response as JSON",
        status_error_message=lambda status_code: f"Artline request failed ({status_code})",
    )


def _include_live_games(context: Optional[dict]) -> bool:
    return isinstance(context, dict) and bool(context.get("live"))


def _game_live_state_payload(game: object, games_type: str) -> Optional[dict]:
    if not isinstance(game, dict):
        return None
    explicit_live = _bool_or_none(game.get("is_live"))
    payload: Dict[str, object] = {}
    if explicit_live is not None:
        payload["is_live"] = explicit_live
        payload["status"] = "live" if explicit_live else "scheduled"
    elif _normalize_text(games_type).lower() == "live":
        payload["is_live"] = True
        payload["status"] = "live"
    period = game.get("period")
    if period not in (None, ""):
        payload["period"] = period
    raw_status = game.get("status")
    if raw_status not in (None, ""):
        payload["provider_status"] = raw_status
    return payload or None


def _outcome_row(name: str, price: float) -> dict:
    return {"name": name, "price": round(float(price), 6)}


def _spread_outcome_row(name: str, price: float, point: float) -> dict:
    return {
        "name": name,
        "price": round(float(price), 6),
        "point": round(float(point), 6),
    }


def _total_outcome_row(name: str, price: float, point: float) -> dict:
    return {
        "name": name,
        "price": round(float(price), 6),
        "point": round(float(point), 6),
    }


def _store_best_outcome(store: Dict[str, dict], key: str, candidate: dict) -> None:
    existing = store.get(key)
    existing_price = _safe_float(existing.get("price")) if isinstance(existing, dict) else None
    candidate_price = _safe_float(candidate.get("price")) if isinstance(candidate, dict) else None
    if candidate_price is None:
        return
    if existing_price is None or candidate_price > existing_price:
        store[key] = candidate


def _market_signature(market: dict) -> str:
    key = _normalize_text(market.get("key"))
    outcomes = market.get("outcomes") if isinstance(market.get("outcomes"), list) else []
    parts: List[str] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        name = _normalize_market_key(outcome.get("name"))
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


def _store_best_market(store: Dict[str, dict], market: dict) -> None:
    signature = _market_signature(market)
    previous = store.get(signature)
    if previous is None or _market_score(market) > _market_score(previous):
        store[signature] = market


def _market_needs_detail_fetch(requested_markets: set[str]) -> bool:
    return "team_totals" in requested_markets


def _normalize_game_markets(
    events: object,
    *,
    home_team: str,
    away_team: str,
    requested_markets: set[str],
) -> List[dict]:
    if not isinstance(events, list):
        return []

    two_way: Dict[str, dict] = {}
    three_way: Dict[str, dict] = {}
    totals_by_point: Dict[float, Dict[str, dict]] = {}
    spreads_by_abs_point: Dict[float, Dict[str, dict]] = {}
    team_totals_by_sig: Dict[str, Dict[float, Dict[str, dict]]] = {
        "home": {},
        "away": {},
    }
    for event in events:
        if not isinstance(event, dict):
            continue
        if int(event.get("status", 0) or 0) != 1:
            continue
        price = _safe_float(event.get("value"))
        if price is None or price <= 1:
            continue
        event_name = _normalize_text(event.get("event_name_value"))
        if event_name == "0_ml_1":
            _store_best_outcome(two_way, "home", _outcome_row(home_team, price))
        elif event_name == "0_ml_2":
            _store_best_outcome(two_way, "away", _outcome_row(away_team, price))
        elif event_name == "0_win_0":
            _store_best_outcome(three_way, "draw", _outcome_row("Draw", price))
        elif event_name == "0_win_1":
            _store_best_outcome(three_way, "home", _outcome_row(home_team, price))
        elif event_name == "0_win_2":
            _store_best_outcome(three_way, "away", _outcome_row(away_team, price))
        else:
            totals_match = re.match(
                r"^0_(to|tu)-[^_]+_([0-9]+)_(-?\d+(?:\.\d+)?)$",
                event_name,
                flags=re.IGNORECASE,
            )
            if totals_match:
                side_token, scope_token, point_token = totals_match.groups()
                point_value = _safe_float(point_token)
                if point_value is None:
                    continue
                outcome_name = "Over" if side_token.lower() == "to" else "Under"
                if scope_token == "0":
                    market_bucket = totals_by_point.setdefault(round(float(point_value), 6), {})
                    _store_best_outcome(
                        market_bucket,
                        outcome_name,
                        _total_outcome_row(outcome_name, price, point_value),
                    )
                    continue
                if scope_token in {"1", "2"}:
                    side_key = "home" if scope_token == "1" else "away"
                    team_name = home_team if side_key == "home" else away_team
                    market_bucket = team_totals_by_sig[side_key].setdefault(
                        round(float(point_value), 6),
                        {},
                    )
                    row = _total_outcome_row(outcome_name, price, point_value)
                    row["description"] = team_name
                    _store_best_outcome(market_bucket, outcome_name, row)
                continue

            spread_match = re.match(
                r"^0_f-[^_]+_([12])_(-?\d+(?:\.\d+)?)$",
                event_name,
                flags=re.IGNORECASE,
            )
            if spread_match:
                team_token, point_token = spread_match.groups()
                point_value = _safe_float(point_token)
                if point_value is None or abs(float(point_value)) <= 1e-9:
                    continue
                abs_point = round(abs(float(point_value)), 6)
                market_bucket = spreads_by_abs_point.setdefault(abs_point, {})
                is_positive = float(point_value) > 0
                if team_token == "1":
                    bucket_key = "home_positive" if is_positive else "home_negative"
                    candidate = _spread_outcome_row(home_team, price, point_value)
                else:
                    bucket_key = "away_positive" if is_positive else "away_negative"
                    candidate = _spread_outcome_row(away_team, price, point_value)
                _store_best_outcome(market_bucket, bucket_key, candidate)

    markets_by_sig: Dict[str, dict] = {}
    has_two_way_market = "h2h" in requested_markets and {"home", "away"}.issubset(two_way)
    if has_two_way_market:
        _store_best_market(
            markets_by_sig,
            {
                "key": "h2h",
                "outcomes": [two_way["home"], two_way["away"]],
            },
        )

    want_three_way = "h2h_3_way" in requested_markets or ("h2h" in requested_markets and not has_two_way_market)
    if want_three_way and {"home", "draw", "away"}.issubset(three_way):
        _store_best_market(
            markets_by_sig,
            {
                "key": "h2h_3_way",
                "outcomes": [three_way["home"], three_way["draw"], three_way["away"]],
            },
        )

    if "totals" in requested_markets:
        for point_value in sorted(totals_by_point):
            bucket = totals_by_point.get(point_value) or {}
            if "Over" not in bucket or "Under" not in bucket:
                continue
            _store_best_market(
                markets_by_sig,
                {
                    "key": "totals",
                    "outcomes": [bucket["Over"], bucket["Under"]],
                },
            )

    if "spreads" in requested_markets:
        for abs_point in sorted(spreads_by_abs_point):
            bucket = spreads_by_abs_point.get(abs_point) or {}
            if "home_negative" in bucket and "away_positive" in bucket:
                spread_market = {
                    "key": "spreads",
                    "outcomes": [bucket["home_negative"], bucket["away_positive"]],
                }
            elif "home_positive" in bucket and "away_negative" in bucket:
                spread_market = {
                    "key": "spreads",
                    "outcomes": [bucket["home_positive"], bucket["away_negative"]],
                }
            else:
                continue
            _store_best_market(markets_by_sig, spread_market)

    if "team_totals" in requested_markets:
        for side_key, point_map in team_totals_by_sig.items():
            for point_value in sorted(point_map):
                bucket = point_map.get(point_value) or {}
                if "Over" not in bucket or "Under" not in bucket:
                    continue
                _store_best_market(
                    markets_by_sig,
                    {
                        "key": "team_totals",
                        "outcomes": [bucket["Over"], bucket["Under"]],
                    },
                )

    return list(markets_by_sig.values())


def _event_url(event: object) -> str:
    if not isinstance(event, dict):
        return _public_base()
    event_id = _normalize_text(event.get("event_id") or event.get("id"))
    sport_key = _normalize_text(event.get("sport_key")).lower()
    sport_filter = ARTLINE_SPORT_FILTERS.get(sport_key) or {}
    sport_slug = _normalize_text(event.get("sport") or sport_filter.get("sport"))
    if not (event_id and sport_slug):
        return _public_base()
    live_state = event.get("live_state") if isinstance(event.get("live_state"), dict) else None
    is_live = _bool_or_none(event.get("is_live"))
    if is_live is None and isinstance(live_state, dict):
        is_live = _bool_or_none(live_state.get("is_live"))
    games_type = "live" if is_live else "prematch"
    return f"{_public_base()}/bookmaker/match/{games_type}/{sport_slug}/{event_id}"


def _set_last_stats(stats: dict) -> None:
    fetch_events.last_stats = stats
    fetch_events_async.last_stats = stats


async def _load_game_detail_events_async(
    client: httpx.AsyncClient,
    *,
    games_type: str,
    sport_slug: str,
    event_id: str,
    retries: int,
    backoff_seconds: float,
) -> Optional[List[dict]]:
    payload, _ = await _request_json_async(
        client,
        "GET",
        f"lines/game/{games_type}/{sport_slug}/{event_id}",
        retries=retries,
        backoff_seconds=backoff_seconds,
    )
    data = _payload_data(payload)
    if not isinstance(data, dict):
        return None
    events = data.get("events")
    if not isinstance(events, list):
        return None
    return [item for item in events if isinstance(item, dict)]


async def fetch_events_async(
    sport_key: str,
    markets: Sequence[str],
    regions: Sequence[str],
    bookmakers: Optional[Sequence[str]] = None,
    context: Optional[dict] = None,
) -> List[dict]:
    _ = regions  # Reserved for future region-specific routing.
    requested_markets = _requested_market_keys(markets)
    sport_token = _normalize_text(sport_key).lower()
    games_type = "live" if _include_live_games(context) else "prematch"
    stats = {
        "provider": PROVIDER_KEY,
        "source": ARTLINE_SOURCE or "api",
        "sport_key": sport_token,
        "games_type": games_type,
        "skipped_unsupported_sport": False,
        "payload_shape": "",
        "payload_games_count": 0,
        "live_feed_empty": False,
        "detail_enrichment_requested": 0,
        "detail_enrichment_succeeded": 0,
        "detail_enrichment_failed": 0,
        "events_with_market_count": 0,
        "events_returned_count": 0,
        "retries_used": 0,
    }
    _set_last_stats(stats)

    if not requested_markets:
        return []

    if bookmakers:
        lowered = {str(book).strip().lower() for book in bookmakers if isinstance(book, str)}
        if PROVIDER_KEY not in lowered and PROVIDER_TITLE.lower() not in lowered:
            return []

    if not _supports_sport(sport_token):
        stats["skipped_unsupported_sport"] = True
        _set_last_stats(stats)
        return []

    if (ARTLINE_SOURCE or "api").lower() != "api":
        raise ProviderError("Artline provider supports ARTLINE_SOURCE=api only")

    sport_filter = ARTLINE_SPORT_FILTERS[sport_token]
    payload = {
        "games_type": games_type,
        "sport": sport_filter["sport"],
        "tournament_id": sport_filter["tournament_id"],
    }

    retries = _int_or_default(ARTLINE_RETRIES_RAW, 2, min_value=0)
    backoff = _float_or_default(ARTLINE_RETRY_BACKOFF_RAW, 0.5, min_value=0.0)
    timeout = _int_or_default(ARTLINE_TIMEOUT_RAW, 20, min_value=1)
    client = await get_shared_client(PROVIDER_KEY, timeout=float(timeout), follow_redirects=True)
    response_payload, retries_used = await _request_json_async(
        client,
        "POST",
        "lines",
        json_payload=payload,
        retries=retries,
        backoff_seconds=backoff,
    )
    stats["retries_used"] = retries_used

    data = _payload_data(response_payload)
    if isinstance(data, dict):
        stats["payload_shape"] = "dict"
    elif isinstance(data, list):
        stats["payload_shape"] = "list"
    elif data is None:
        stats["payload_shape"] = "none"
    else:
        stats["payload_shape"] = type(data).__name__.lower()
    sport_block = data.get(sport_filter["sport"]) if isinstance(data, dict) else None
    games = sport_block.get("games") if isinstance(sport_block, dict) else []
    if not isinstance(games, list):
        games = []
    stats["payload_games_count"] = len(games)
    stats["live_feed_empty"] = games_type == "live" and len(games) == 0

    detailed_events_by_id: Dict[str, List[dict]] = {}
    if games and _market_needs_detail_fetch(requested_markets):
        semaphore = asyncio.Semaphore(_detail_max_concurrency())

        async def _detail_job(game: dict) -> tuple[str, Optional[List[dict]], Optional[str]]:
            event_id = _normalize_text(game.get("id"))
            if not event_id:
                return "", None, None
            async with semaphore:
                try:
                    events_payload = await _load_game_detail_events_async(
                        client,
                        games_type=games_type,
                        sport_slug=sport_filter["sport"],
                        event_id=event_id,
                        retries=retries,
                        backoff_seconds=backoff,
                    )
                    return event_id, events_payload, None
                except ProviderError as exc:
                    return event_id, None, str(exc)

        stats["detail_enrichment_requested"] = len(games)
        detail_tasks = [asyncio.create_task(_detail_job(game)) for game in games if isinstance(game, dict)]
        for task in asyncio.as_completed(detail_tasks):
            event_id, detail_events, error = await task
            if not event_id:
                continue
            if error:
                stats["detail_enrichment_failed"] += 1
                continue
            if isinstance(detail_events, list):
                detailed_events_by_id[event_id] = detail_events
                stats["detail_enrichment_succeeded"] += 1
            else:
                stats["detail_enrichment_failed"] += 1

    events_out: List[dict] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        event_id = _normalize_text(game.get("id"))
        if not event_id:
            continue
        team_1 = game.get("team_1") if isinstance(game.get("team_1"), dict) else {}
        team_2 = game.get("team_2") if isinstance(game.get("team_2"), dict) else {}
        home_team = _normalize_text(team_1.get("value"))
        away_team = _normalize_text(team_2.get("value"))
        if not (home_team and away_team):
            continue
        commence_time = _normalize_commence_time(
            game.get("start_at_timestamp") or game.get("start_at")
        )
        if not commence_time:
            continue
        game_events = detailed_events_by_id.get(event_id)
        if not isinstance(game_events, list):
            game_events = game.get("events")
        normalized_markets = _normalize_game_markets(
            game_events,
            home_team=home_team,
            away_team=away_team,
            requested_markets=requested_markets,
        )
        if not normalized_markets:
            continue

        live_state = _game_live_state_payload(game, games_type)
        normalized_event = {
            "id": event_id,
            "sport_key": sport_token,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "live_state": live_state,
            "bookmakers": [
                {
                    "key": PROVIDER_KEY,
                    "title": PROVIDER_TITLE,
                    "event_id": event_id,
                    "event_url": _event_url(
                        {
                            "event_id": event_id,
                            "sport_key": sport_token,
                            "live_state": live_state,
                        }
                    ),
                    "live_state": live_state,
                    "markets": normalized_markets,
                }
            ],
        }
        stats["events_with_market_count"] += 1
        events_out.append(normalized_event)

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
    "source": ARTLINE_SOURCE or "api",
    "events_returned_count": 0,
}
fetch_events_async.last_stats = dict(fetch_events.last_stats)

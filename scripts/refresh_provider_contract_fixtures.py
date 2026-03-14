from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from providers import betdex, bookmaker_xyz, polymarket, sx_bet


DEFAULT_OUTPUT = ROOT_DIR / "tests" / "fixtures" / "provider_contract_replay.json"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _unique_market_keys(markets: Iterable[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen = set()
    for market in markets:
        key = str((market or {}).get("key") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _normalize_pair(values: Iterable[str]) -> tuple[str, str]:
    items = [str(value or "").strip().lower() for value in values if str(value or "").strip()]
    return tuple(sorted(items)[:2])  # type: ignore[return-value]


def _match_home_away(record: dict[str, Any], home_team: str, away_team: str) -> bool:
    return _normalize_pair([record.get("home_team"), record.get("away_team")]) == _normalize_pair(
        [home_team, away_team]
    )


def _trim_polymarket_sports_payload(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    selected: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        sport_token = polymarket._normalize_token(item.get("sport"))
        if sport_token in {"nhl", "hockey"}:
            selected.append(item)
    if selected:
        return selected
    return [item for item in payload[:3] if isinstance(item, dict)]


async def _build_polymarket_fixture() -> dict[str, Any]:
    sport_key = "icehockey_nhl"
    normalized = await polymarket.fetch_events_async(sport_key, ["h2h"], ["us"])
    if not normalized:
        raise RuntimeError("Polymarket returned no NHL h2h events to build a contract fixture")
    target_event = normalized[0]
    target_home = target_event["home_team"]
    target_away = target_event["away_team"]

    client = await polymarket.get_shared_client(polymarket.PROVIDER_KEY, timeout=20.0, follow_redirects=True)
    sports_payload, _ = await polymarket._request_json_async(client, "sports", {}, retries=0, backoff_seconds=0.1)
    events_payload, _ = await polymarket._request_json_async(
        client,
        "events",
        {
            "tag_id": polymarket.POLYMARKET_GAME_TAG_ID or "100639",
            "active": "true",
            "closed": "false",
            "archived": "false",
            "order": "id",
            "ascending": "false",
            "limit": 200,
            "offset": 0,
        },
        retries=0,
        backoff_seconds=0.1,
    )
    if not isinstance(events_payload, list):
        raise RuntimeError("Polymarket events payload was not a list")
    raw_event = next(
        (
            item
            for item in events_payload
            if isinstance(item, dict)
            and _normalize_pair(polymarket._extract_matchup(item) or ()) == _normalize_pair([target_home, target_away])
        ),
        None,
    )
    if not isinstance(raw_event, dict):
        raise RuntimeError(f"Failed to locate raw Polymarket event for {target_home} vs {target_away}")

    return {
        "sport_key": sport_key,
        "requests": {
            "sports": _trim_polymarket_sports_payload(sports_payload),
            "events": [raw_event],
        },
        "expected": {
            "home_team": target_home,
            "away_team": target_away,
            "market_keys": _unique_market_keys(target_event["bookmakers"][0]["markets"]),
            "event_count": 1,
        },
    }


def _trim_betdex_event_payload(
    payload: dict[str, Any],
    event_id: str,
) -> dict[str, Any]:
    events = [item for item in payload.get("events") or [] if isinstance(item, dict) and betdex._normalize_text(item.get("id")) == event_id]
    participants_needed: set[str] = set()
    group_needed: set[str] = set()
    for event in events:
        participants_needed.update(betdex._doc_ref_ids(event.get("participants")))
        group_id = betdex._doc_ref_first_id(event.get("eventGroup"))
        if group_id:
            group_needed.add(group_id)
    return {
        "events": events,
        "eventGroups": [
            item
            for item in payload.get("eventGroups") or []
            if isinstance(item, dict) and betdex._normalize_text(item.get("id")) in group_needed
        ],
        "participants": [
            item
            for item in payload.get("participants") or []
            if isinstance(item, dict) and betdex._normalize_text(item.get("id")) in participants_needed
        ],
        "_meta": {"_page": {"_totalPages": 1}},
    }


def _trim_betdex_market_payload(payload: dict[str, Any], event_id: str) -> tuple[dict[str, Any], list[str]]:
    markets = [
        item
        for item in payload.get("markets") or []
        if isinstance(item, dict) and betdex._doc_ref_first_id(item.get("event")) == event_id
    ]
    market_ids = [betdex._normalize_text(item.get("id")) for item in markets if betdex._normalize_text(item.get("id"))]
    outcome_ids: set[str] = set()
    for market in markets:
        outcome_ids.update(betdex._doc_ref_ids(market.get("marketOutcomes")))
    trimmed = {
        "markets": markets,
        "marketOutcomes": [
            item
            for item in payload.get("marketOutcomes") or []
            if isinstance(item, dict) and betdex._normalize_text(item.get("id")) in outcome_ids
        ],
        "_meta": {"_page": {"_totalPages": 1}},
    }
    return trimmed, market_ids


def _trim_betdex_prices_payload(payload: dict[str, Any], market_ids: Iterable[str]) -> dict[str, Any]:
    market_id_set = {str(item) for item in market_ids if str(item)}
    return {
        "prices": [
            item
            for item in payload.get("prices") or []
            if isinstance(item, dict) and betdex._normalize_text(item.get("marketId")) in market_id_set
        ]
    }


async def _build_betdex_fixture() -> dict[str, Any]:
    sport_key = "icehockey_nhl"
    normalized = await betdex.fetch_events_async(sport_key, ["h2h", "spreads", "totals"], ["us"])
    if not normalized:
        raise RuntimeError("BetDEX returned no NHL events to build a contract fixture")
    target_event = normalized[0]
    target_event_id = str(target_event["id"])

    client = await betdex.get_shared_client(betdex.PROVIDER_KEY, timeout=20.0, follow_redirects=True)
    session_payload, _ = await betdex._request_json_async(
        client,
        betdex._session_url(),
        params=None,
        access_token=None,
        retries=0,
        backoff_seconds=0.1,
        timeout=20,
    )
    token = betdex._extract_access_token(session_payload)
    events_payload, _ = await betdex._request_json_async(
        client,
        f"{betdex._api_base()}/events",
        params={"active": "true", "page": 0, "size": 50, "subcategoryIds": betdex.SPORT_SUBCATEGORY_DEFAULTS[sport_key][0]},
        access_token=token,
        retries=0,
        backoff_seconds=0.1,
        timeout=20,
    )
    if not isinstance(events_payload, dict):
        raise RuntimeError("BetDEX events payload was not a dict")
    trimmed_events = _trim_betdex_event_payload(events_payload, target_event_id)

    markets_payload, _ = await betdex._request_json_async(
        client,
        f"{betdex._api_base()}/markets",
        params=[
            ("eventIds", target_event_id),
            ("published", "true"),
            ("page", 0),
            ("size", 100),
            ("statuses", "Open"),
        ],
        access_token=token,
        retries=0,
        backoff_seconds=0.1,
        timeout=20,
    )
    if not isinstance(markets_payload, dict):
        raise RuntimeError("BetDEX markets payload was not a dict")
    trimmed_markets, market_ids = _trim_betdex_market_payload(markets_payload, target_event_id)

    prices_payload, _ = await betdex._request_json_async(
        client,
        f"{betdex._api_base()}/market-prices",
        params=[("marketIds", market_id) for market_id in market_ids],
        access_token=token,
        retries=0,
        backoff_seconds=0.1,
        timeout=20,
    )
    if not isinstance(prices_payload, dict):
        raise RuntimeError("BetDEX prices payload was not a dict")

    return {
        "sport_key": sport_key,
        "requests": {
            "session": session_payload,
            "events": trimmed_events,
            "markets": trimmed_markets,
            "prices": _trim_betdex_prices_payload(prices_payload, market_ids),
        },
        "expected": {
            "home_team": target_event["home_team"],
            "away_team": target_event["away_team"],
            "market_keys": _unique_market_keys(target_event["bookmakers"][0]["markets"]),
            "event_count": 1,
        },
    }


def _trim_sx_summary_payload(payload: dict[str, Any], target_fixture: dict[str, Any]) -> dict[str, Any]:
    target_fixture_id = sx_bet._normalize_text(target_fixture.get("id") or target_fixture.get("eventId"))
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    out_sports: list[dict[str, Any]] = []
    for sport in data.get("sports") or []:
        if not isinstance(sport, dict):
            continue
        out_leagues: list[dict[str, Any]] = []
        for league in sport.get("leagues") or []:
            if not isinstance(league, dict):
                continue
            fixtures = [
                item
                for item in league.get("fixtures") or []
                if isinstance(item, dict)
                and sx_bet._normalize_text(item.get("id") or item.get("eventId")) == target_fixture_id
            ]
            if fixtures:
                out_leagues.append({**league, "fixtures": fixtures})
        if out_leagues:
            out_sports.append({**sport, "leagues": out_leagues})
    return {"status": payload.get("status"), "data": {"sports": out_sports}}


def _trim_sx_best_odds_payload(payload: dict[str, Any], market_hashes: Iterable[str]) -> dict[str, Any]:
    hashes = {str(item) for item in market_hashes if str(item)}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    best_odds = [
        item
        for item in data.get("bestOdds") or []
        if isinstance(item, dict) and sx_bet._normalize_text(item.get("marketHash")) in hashes
    ]
    return {"status": payload.get("status"), "data": {"bestOdds": best_odds}}


def _trim_sx_orders_payload(payload: object, market_hashes: Iterable[str]) -> object:
    hashes = {str(item) for item in market_hashes if str(item)}
    rows = [
        item
        for item in sx_bet._extract_orders_payload(payload)
        if isinstance(item, dict) and sx_bet._normalize_text(item.get("marketHash")) in hashes
    ]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), dict):
            return {**payload, "data": {**payload["data"], "orders": rows}}
        return {**payload, "data": rows}
    return rows


async def _build_sx_bet_fixture() -> dict[str, Any]:
    sport_key = "icehockey_nhl"
    normalized = await sx_bet.fetch_events_async(sport_key, ["h2h", "spreads", "totals"], ["us"])
    if not normalized:
        raise RuntimeError("SX Bet returned no NHL events to build a contract fixture")
    target_event = normalized[0]
    target_home = target_event["home_team"]
    target_away = target_event["away_team"]

    client = await sx_bet.get_shared_client(sx_bet.PROVIDER_KEY, timeout=20.0, follow_redirects=True)
    summary_payload, _ = await sx_bet._request_json_async(
        client,
        f"summary/upcoming/{sx_bet.SX_BET_BASE_TOKEN}/{sx_bet.SX_SPORT_ID_MAP[sport_key]}",
        retries=0,
        backoff_seconds=0.1,
    )
    if not isinstance(summary_payload, dict):
        raise RuntimeError("SX Bet summary payload was not a dict")
    fixtures = []
    for sport in ((summary_payload.get("data") or {}).get("sports") or []):
        if not isinstance(sport, dict):
            continue
        for league in sport.get("leagues") or []:
            if not isinstance(league, dict):
                continue
            for fixture in league.get("fixtures") or []:
                if not isinstance(fixture, dict):
                    continue
                fixture_home = sx_bet._normalize_text(fixture.get("teamOne"))
                fixture_away = sx_bet._normalize_text(fixture.get("teamTwo"))
                if _normalize_pair([fixture_home, fixture_away]) == _normalize_pair([target_home, target_away]):
                    fixtures.append(fixture)
    if not fixtures:
        raise RuntimeError(f"Failed to locate raw SX Bet fixture for {target_home} vs {target_away}")
    target_fixture = fixtures[0]
    market_hashes = [
        sx_bet._normalize_text(item.get("marketHash"))
        for item in target_fixture.get("markets") or []
        if isinstance(item, dict) and sx_bet._normalize_text(item.get("marketHash"))
    ]
    best_odds_payload, _ = await sx_bet._request_json_async(
        client,
        "orders/odds/best",
        params={"marketHashes": ",".join(market_hashes), "baseToken": sx_bet.SX_BET_BASE_TOKEN},
        retries=0,
        backoff_seconds=0.1,
    )
    orders_payload, _ = await sx_bet._request_json_async(
        client,
        "orders",
        params={"marketHashes": ",".join(market_hashes), "baseToken": sx_bet.SX_BET_BASE_TOKEN},
        retries=0,
        backoff_seconds=0.1,
    )
    return {
        "sport_key": sport_key,
        "requests": {
            "summary": _trim_sx_summary_payload(summary_payload, target_fixture),
            "best_odds": _trim_sx_best_odds_payload(best_odds_payload, market_hashes if isinstance(best_odds_payload, dict) else []),
            "orders": _trim_sx_orders_payload(orders_payload, market_hashes),
        },
        "expected": {
            "home_team": target_home,
            "away_team": target_away,
            "market_keys": _unique_market_keys(target_event["bookmakers"][0]["markets"]),
            "event_count": 1,
        },
    }


def _trim_bookmaker_dictionaries(conditions: list[dict[str, Any]], dictionaries: dict[str, Any]) -> dict[str, Any]:
    subset = {
        "outcomes": {},
        "marketNames": {},
        "selections": {},
        "points": {},
        "teamPlayers": {},
    }
    outcomes_dict = dictionaries.get("outcomes") if isinstance(dictionaries.get("outcomes"), dict) else {}
    market_names = dictionaries.get("marketNames") if isinstance(dictionaries.get("marketNames"), dict) else {}
    selections = dictionaries.get("selections") if isinstance(dictionaries.get("selections"), dict) else {}
    points = dictionaries.get("points") if isinstance(dictionaries.get("points"), dict) else {}
    team_players = dictionaries.get("teamPlayers") if isinstance(dictionaries.get("teamPlayers"), dict) else {}

    for condition in conditions:
        for raw_outcome in condition.get("outcomes") or []:
            if not isinstance(raw_outcome, dict):
                continue
            outcome_id = bookmaker_xyz._normalize_text(raw_outcome.get("outcomeId"))
            outcome_meta = bookmaker_xyz._dict_get(outcomes_dict, outcome_id)
            if not isinstance(outcome_meta, dict):
                continue
            subset["outcomes"][outcome_id] = outcome_meta
            market_id = bookmaker_xyz._normalize_text(outcome_meta.get("marketId"))
            game_period_id = bookmaker_xyz._normalize_text(outcome_meta.get("gamePeriodId"))
            game_type_id = bookmaker_xyz._normalize_text(outcome_meta.get("gameTypeId"))
            team_player_id = bookmaker_xyz._normalize_text(outcome_meta.get("teamPlayerId"))
            selection_id = bookmaker_xyz._normalize_text(outcome_meta.get("selectionId"))
            points_id = bookmaker_xyz._normalize_text(outcome_meta.get("pointsId"))
            market_parts = [market_id, game_period_id, game_type_id]
            if team_player_id and market_id not in bookmaker_xyz.DICT_SKIP_TEAM_PLAYER_MARKET_IDS:
                market_parts.append(team_player_id)
            market_key = "-".join(part for part in market_parts if part)
            market_name = bookmaker_xyz._dict_get(market_names, market_key)
            if market_name is not None:
                subset["marketNames"][market_key] = market_name
            if selection_id:
                selection_label = bookmaker_xyz._dict_get(selections, selection_id)
                if selection_label is not None:
                    subset["selections"][selection_id] = selection_label
            if points_id:
                point_label = bookmaker_xyz._dict_get(points, points_id)
                if point_label is not None:
                    subset["points"][points_id] = point_label
            if team_player_id:
                team_player_label = bookmaker_xyz._dict_get(team_players, team_player_id)
                if team_player_label is not None:
                    subset["teamPlayers"][team_player_id] = team_player_label
    return subset


async def _build_bookmaker_xyz_fixture() -> dict[str, Any]:
    sport_key = "icehockey_nhl"
    normalized = await bookmaker_xyz.fetch_events_async(sport_key, ["h2h", "spreads", "totals"], ["us"])
    target_event = next(
        (
            event
            for event in normalized
            if "h2h" in _unique_market_keys(event["bookmakers"][0]["markets"])
        ),
        None,
    )
    if not isinstance(target_event, dict):
        raise RuntimeError("bookmaker.xyz returned no NHL h2h events to build a contract fixture")

    client = await bookmaker_xyz.get_shared_client(bookmaker_xyz.PROVIDER_KEY, timeout=25.0, follow_redirects=True)
    dictionaries, dict_meta = await bookmaker_xyz._load_dictionaries_async(
        client=client,
        retries=0,
        backoff_seconds=0.1,
        timeout=25,
    )
    conditions, payload_meta = await bookmaker_xyz._load_market_manager_snapshot_async(
        client=client,
        sport_key=sport_key,
        retries=0,
        backoff_seconds=0.1,
        timeout=25,
    )
    if not isinstance(dictionaries, dict):
        raise RuntimeError("bookmaker.xyz dictionaries were not available")
    if not isinstance(conditions, list):
        raise RuntimeError("bookmaker.xyz conditions payload was not a list")

    target_home = target_event["home_team"]
    target_away = target_event["away_team"]
    target_conditions = []
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        game = condition.get("game") if isinstance(condition.get("game"), dict) else {}
        game_id = bookmaker_xyz._normalize_text(game.get("gameId") or game.get("id"))
        event_game_id = str(target_event["id"]).split(":")[-1]
        if game_id and game_id == event_game_id:
            target_conditions.append(condition)
            continue
        participants = game.get("participants") if isinstance(game.get("participants"), list) else []
        participant_names = [
            bookmaker_xyz._normalize_text(item.get("name"))
            for item in participants
            if isinstance(item, dict)
        ]
        if _normalize_pair(participant_names[:2]) == _normalize_pair([target_home, target_away]):
            target_conditions.append(condition)
    if not target_conditions:
        raise RuntimeError(f"Failed to locate raw bookmaker.xyz conditions for {target_home} vs {target_away}")

    return {
        "sport_key": sport_key,
        "raw": {
            "conditions": target_conditions,
            "dictionaries": _trim_bookmaker_dictionaries(target_conditions, dictionaries),
            "dictionary_meta": dict_meta,
            "payload_meta": payload_meta,
        },
        "expected": {
            "home_team": target_home,
            "away_team": target_away,
            "market_keys": _unique_market_keys(target_event["bookmakers"][0]["markets"]),
            "event_count": 1,
        },
    }


async def _build_fixture_payload() -> dict[str, Any]:
    polymarket.disable_scan_cache()
    betdex.ACCESS_TOKEN_CACHE["token"] = ""
    betdex.ACCESS_TOKEN_CACHE["expires_at"] = 0.0
    sx_bet.UPCOMING_CACHE["expires_at"] = 0.0
    sx_bet.UPCOMING_CACHE["entries"] = {}
    sx_bet.ODDS_CACHE["expires_at"] = 0.0
    sx_bet.ODDS_CACHE["entries"] = {}
    sx_bet.ORDERS_CACHE["expires_at"] = 0.0
    sx_bet.ORDERS_CACHE["entries"] = {}
    bookmaker_xyz.disable_scan_cache()
    bookmaker_xyz.CONDITIONS_CACHE["expires_at"] = 0.0
    bookmaker_xyz.CONDITIONS_CACHE["conditions"] = []
    bookmaker_xyz.CONDITIONS_CACHE["meta"] = {}
    bookmaker_xyz.CONDITIONS_CACHE["key"] = ""

    providers = {
        "polymarket": await _build_polymarket_fixture(),
        "betdex": await _build_betdex_fixture(),
        "sx_bet": await _build_sx_bet_fixture(),
        "bookmaker_xyz": await _build_bookmaker_xyz_fixture(),
    }
    return {
        "generated_at": _utc_now(),
        "providers": providers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh recorded provider contract replay fixtures.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the JSON fixture file.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(_build_fixture_payload())
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote provider contract fixtures to {output_path}")


if __name__ == "__main__":
    main()

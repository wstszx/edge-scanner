from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import httpx


ARTLINE_BASE_URL = "https://api.artline.bet"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "edge-scanner artline probe/1.0",
}


class ArtlineApiError(RuntimeError):
    def __init__(self, result: dict[str, Any]) -> None:
        payload = result.get("payload")
        if isinstance(payload, dict):
            message = str(payload.get("message") or result.get("status_code") or "API request failed")
        else:
            message = str(result.get("status_code") or "API request failed")
        super().__init__(message)
        self.result = result


def _normalize_api_path(path: str) -> str:
    clean = "/" + str(path or "").strip().lstrip("/")
    if clean == "/sanctum/csrf-cookie":
        return clean
    if not clean.startswith("/api/"):
        clean = "/api" + clean
    return clean


def _build_url(path: str) -> str:
    return f"{ARTLINE_BASE_URL}{_normalize_api_path(path)}"


def _coerce_csv(values: Iterable[str] | None) -> str:
    items: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text:
            items.append(text)
    return ",".join(items)


def _build_lines_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {"games_type": args.games_type}
    if getattr(args, "page", None) is not None:
        payload["page"] = args.page
    if getattr(args, "search", None):
        payload["search"] = args.search
    if getattr(args, "sport_type", None):
        payload["sport_type"] = args.sport_type
    if getattr(args, "sport", None):
        payload["sport"] = _coerce_csv(args.sport)
    if getattr(args, "tournament_id", None):
        payload["tournament_id"] = _coerce_csv(args.tournament_id)
    if getattr(args, "region_id", None):
        payload["region_id"] = _coerce_csv(args.region_id)
    if getattr(args, "except_games_id", None):
        payload["except_games_id"] = _coerce_csv(args.except_games_id)
    return {key: value for key, value in payload.items() if value not in ("", None)}


def _request_json(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    url = _build_url(path)
    with httpx.Client(timeout=float(timeout), follow_redirects=True, headers=DEFAULT_HEADERS) as client:
        response = client.request(method.upper(), url, params=params, json=json_payload)
    content_type = str(response.headers.get("Content-Type", ""))
    try:
        payload: Any = response.json()
    except ValueError:
        payload = response.text
    result = {
        "url": url,
        "method": method.upper(),
        "status_code": response.status_code,
        "content_type": content_type,
        "ok": response.is_success,
        "params": params or {},
        "json_payload": json_payload or {},
        "payload": payload,
    }
    if response.status_code >= 400:
        raise ArtlineApiError(result)
    return result


def _payload_data(result: dict[str, Any]) -> Any:
    payload = result.get("payload")
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _truncate(value: Any, limit: int = 140) -> str:
    text = _safe_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _collect_lines_games(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    data = _payload_data(result)
    games: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(data, dict):
        return games
    for sport_key, sport_block in data.items():
        if not isinstance(sport_block, dict):
            continue
        for game in sport_block.get("games") or []:
            if isinstance(game, dict):
                games.append((str(sport_key), game))
    return games


def _print_result_json(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _print_settings_summary(result: dict[str, Any], limit: int) -> None:
    data = _payload_data(result)
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    if not isinstance(data, dict):
        print(_truncate(data, 500))
        return
    themes = data.get("themes")
    locales = data.get("locales")
    if isinstance(themes, list):
        print(f"themes: {', '.join(str(item) for item in themes[:limit])}")
    if isinstance(locales, list):
        print(f"locales: {', '.join(str(item) for item in locales[:limit])}")
    if "settings" in data and isinstance(data["settings"], dict):
        print("settings:")
        for key, value in list(data["settings"].items())[:limit]:
            print(f"  {key}: {value}")
        return
    items = list(data.items())[:limit]
    for key, value in items:
        if isinstance(value, list):
            print(f"{key}: {len(value)} items")
        elif isinstance(value, dict):
            print(f"{key}: {len(value)} keys")
        else:
            print(f"{key}: {_truncate(value)}")


def _print_top_summary(result: dict[str, Any], limit: int) -> None:
    data = _payload_data(result)
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    matches = data if isinstance(data, list) else []
    print(f"matches: {len(matches)}")
    for match in matches[:limit]:
        if not isinstance(match, dict):
            continue
        print(
            f"- {match.get('id')} | {match.get('sport')} | live={match.get('is_live')} | "
            f"{match.get('team_1_value')} vs {match.get('team_2_value')} | start={match.get('start_at_timestamp')}"
        )


def _print_lines_summary(result: dict[str, Any], limit: int) -> None:
    data = _payload_data(result)
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    if not isinstance(data, dict):
        print(_truncate(data, 500))
        return
    counts: list[tuple[str, int]] = []
    for sport_key, sport_block in data.items():
        games = []
        if isinstance(sport_block, dict):
            games = sport_block.get("games") or []
        counts.append((str(sport_key), len(games)))
    counts.sort(key=lambda item: (-item[1], item[0]))
    print("sports:")
    for sport_key, count in counts[:limit]:
        print(f"  {sport_key}: {count}")
    preview = _collect_lines_games(result)[:limit]
    if preview:
        print("games:")
    for sport_key, game in preview:
        team_1 = (game.get("team_1") or {}).get("value")
        team_2 = (game.get("team_2") or {}).get("value")
        events = game.get("events") or []
        print(
            f"  - {sport_key} | {game.get('id')} | {team_1} vs {team_2} | "
            f"events={len(events)} | live={game.get('is_live')} | start={game.get('start_at')}"
        )


def _print_match_summary(result: dict[str, Any], events_limit: int) -> None:
    match = _payload_data(result)
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    if not isinstance(match, dict):
        print(_truncate(match, 500))
        return
    team_1 = (match.get("team_1") or {}).get("value")
    team_2 = (match.get("team_2") or {}).get("value")
    tournament = (match.get("tournament") or {}).get("value")
    print(f"match: {match.get('id')} | {team_1} vs {team_2}")
    print(f"tournament: {tournament}")
    print(f"live: {match.get('is_live')} | status: {match.get('status')} | start: {match.get('start_at')}")
    print(f"max_bet: {match.get('max_bet')}")
    events = match.get("events") or []
    print(f"events: {len(events)}")
    for event in events[:events_limit]:
        if not isinstance(event, dict):
            continue
        data_event = event.get("data_event") or {}
        print(
            f"  - {event.get('id')} | value={event.get('value')} | status={event.get('status')} | "
            f"{data_event.get('group_translations')} | {event.get('event_name_value')}"
        )


def _print_fantasy_rooms_summary(result: dict[str, Any], limit: int) -> None:
    rooms = _payload_data(result)
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    rooms = rooms if isinstance(rooms, list) else []
    print(f"rooms: {len(rooms)}")
    for room in rooms[:limit]:
        if not isinstance(room, dict):
            continue
        sport = room.get("sport") or {}
        tickets = room.get("tickets") or []
        print(
            f"- {room.get('id')} | {sport.get('system_name')} | {room.get('tournament_name')} | "
            f"type={room.get('type')} | live={room.get('is_live')} | tickets={len(tickets)}"
        )


def _print_fantasy_room_summary(result: dict[str, Any], teams_limit: int, players_limit: int) -> None:
    room = _payload_data(result)
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    if not isinstance(room, dict):
        print(_truncate(room, 500))
        return
    sport = room.get("sport") or {}
    print(f"room: {room.get('id')} | {room.get('tournament_name')} | sport={sport.get('system_name')}")
    print(f"type: {room.get('type')} | live: {room.get('is_live')} | start_at: {room.get('start_at')}")
    tickets = room.get("tickets") or []
    print(f"tickets: {len(tickets)}")
    for ticket in tickets[:teams_limit]:
        if not isinstance(ticket, dict):
            continue
        print(
            f"  - ticket {ticket.get('id')} | price={ticket.get('price')} {ticket.get('price_type')} | "
            f"budget={ticket.get('budget')} | team_size={ticket.get('team_size')}"
        )
    teams = room.get("teams") or []
    print(f"teams: {len(teams)}")
    for team in teams[:teams_limit]:
        if not isinstance(team, dict):
            continue
        players = team.get("players") or []
        print(f"  - {team.get('value')} | players={len(players)}")
        for player in players[:players_limit]:
            if not isinstance(player, dict):
                continue
            print(
                f"      * {player.get('value')} | cost={player.get('cost')} | "
                f"country={player.get('country_code')} | score={player.get('score')}"
            )


def _print_fantasy_ticket_summary(result: dict[str, Any]) -> None:
    ticket = _payload_data(result)
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    if not isinstance(ticket, dict):
        print(_truncate(ticket, 500))
        return
    for key in ("id", "price", "price_type", "budget", "sets", "sets_max"):
        print(f"{key}: {ticket.get(key)}")
    sets_other = ticket.get("sets_other") or []
    print(f"sets_other: {len(sets_other)}")


def _print_store_categories_summary(result: dict[str, Any], limit: int) -> None:
    data = _payload_data(result)
    items = data if isinstance(data, list) else []
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    print(f"categories: {len(items)}")
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        print(f"- {item.get('id')} | {item.get('name')} | parent={item.get('parent_id')} | sort={item.get('sort')}")


def _print_store_goods_summary(result: dict[str, Any], limit: int) -> None:
    data = _payload_data(result)
    items = data if isinstance(data, list) else []
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    print(f"goods: {len(items)}")
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        price = item.get("price") or {}
        print(
            f"- {item.get('id')} | {item.get('name')} | "
            f"price={price.get('green')}/{price.get('blue')}/{price.get('orange')} | "
            f"sort={item.get('sort_order')}"
        )


def _print_generic_summary(result: dict[str, Any], limit: int) -> None:
    payload = result.get("payload")
    print(f"{result['method']} {result['url']} -> {result['status_code']}")
    if isinstance(payload, dict):
        keys = list(payload.keys())
        print(f"payload keys: {', '.join(keys[:limit])}")
        if "message" in payload:
            print(f"message: {payload['message']}")
    elif isinstance(payload, list):
        print(f"items: {len(payload)}")
        for item in payload[:limit]:
            print(f"- {_truncate(json.dumps(item, ensure_ascii=False), 180)}")
    else:
        print(_truncate(payload, 1000))


def _write_output(path: str | None, result: dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_key_value_pairs(items: Iterable[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or ():
        text = str(item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Expected key=value pair, got: {text}")
        key, value = text.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="Print full wrapped JSON response.")
    common.add_argument("--output", help="Write wrapped JSON response to a file.")
    common.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Request timeout in seconds.")

    parser = argparse.ArgumentParser(description="Probe public Artline API endpoints from the command line.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    settings = subparsers.add_parser("settings", parents=[common], help="Fetch settings endpoints.")
    settings.add_argument(
        "section",
        nargs="?",
        default="root",
        choices=("root", "sports", "fantasy", "achievements"),
        help="Which settings endpoint to request.",
    )
    settings.add_argument("--limit", type=int, default=10)

    top = subparsers.add_parser("top", parents=[common], help="Fetch top matches.")
    top.add_argument("--limit", type=int, default=10)

    lines = subparsers.add_parser("lines", parents=[common], help="Fetch lines list payloads.")
    lines.add_argument("--games-type", required=True, choices=("prematch", "live", "outright", "top"))
    lines.add_argument("--sport", action="append", help="Repeat to request multiple sport slugs.")
    lines.add_argument("--sport-type", choices=("all", "common", "cyber"))
    lines.add_argument("--tournament-id", action="append")
    lines.add_argument("--region-id", action="append")
    lines.add_argument("--except-games-id", action="append")
    lines.add_argument("--search")
    lines.add_argument("--page", type=int)
    lines.add_argument("--limit", type=int, default=10)

    match = subparsers.add_parser("match", parents=[common], help="Fetch one match details endpoint.")
    match.add_argument("--type", required=True, choices=("prematch", "live"))
    match.add_argument("--sport", required=True, help="Sport slug, for example football.")
    match.add_argument("--id", required=True, help="Match id from /api/lines/top or /api/lines.")
    match.add_argument("--events-limit", type=int, default=20)

    fantasy_rooms = subparsers.add_parser("fantasy-rooms", parents=[common], help="Fetch fantasy room list.")
    fantasy_rooms.add_argument("--sport")
    fantasy_rooms.add_argument("--mode")
    fantasy_rooms.add_argument("--page", type=int)
    fantasy_rooms.add_argument("--limit", type=int, default=10)

    fantasy_room = subparsers.add_parser("fantasy-room", parents=[common], help="Fetch one fantasy room.")
    fantasy_room.add_argument("room_id", help="Fantasy room id.")
    fantasy_room.add_argument("--teams-limit", type=int, default=5)
    fantasy_room.add_argument("--players-limit", type=int, default=8)

    fantasy_ticket = subparsers.add_parser("fantasy-ticket", parents=[common], help="Fetch one fantasy ticket.")
    fantasy_ticket.add_argument("ticket_id", help="Fantasy ticket id.")

    store_category = subparsers.add_parser("store-category", parents=[common], help="Fetch store categories.")
    store_category.add_argument("--limit", type=int, default=10)

    store_good = subparsers.add_parser("store-good", parents=[common], help="Fetch store goods.")
    store_good.add_argument("--page", type=int)
    store_good.add_argument("--category")
    store_good.add_argument("--search")
    store_good.add_argument("--limit", type=int, default=10)

    request = subparsers.add_parser("request", parents=[common], help="Fetch an arbitrary Artline endpoint.")
    request.add_argument("path", help="Endpoint path, for example /settings or /api/settings.")
    request.add_argument("--method", default="GET", choices=("GET", "POST"))
    request.add_argument("--param", action="append", help="Repeat key=value query params.")
    request.add_argument("--json-field", action="append", help="Repeat key=value JSON body fields.")
    request.add_argument("--json-string", help="Raw JSON body string.")
    request.add_argument("--limit", type=int, default=10)

    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "settings":
        path = {
            "root": "/settings",
            "sports": "/settings/sports",
            "fantasy": "/settings/fantasy",
            "achievements": "/settings/achievements",
        }[args.section]
        return _request_json("GET", path, timeout=args.timeout)
    if args.command == "top":
        return _request_json("GET", "/lines/top", timeout=args.timeout)
    if args.command == "lines":
        return _request_json(
            "POST",
            "/lines",
            json_payload=_build_lines_payload(args),
            timeout=args.timeout,
        )
    if args.command == "match":
        return _request_json(
            "GET",
            f"/lines/game/{args.type}/{args.sport}/{args.id}",
            timeout=args.timeout,
        )
    if args.command == "fantasy-rooms":
        params = {key: value for key, value in {"sport": args.sport, "mode": args.mode, "page": args.page}.items() if value not in (None, "")}
        return _request_json("GET", "/fantasy/room", params=params, timeout=args.timeout)
    if args.command == "fantasy-room":
        return _request_json("GET", f"/fantasy/room/{args.room_id}", timeout=args.timeout)
    if args.command == "fantasy-ticket":
        return _request_json("GET", f"/fantasy/ticket/{args.ticket_id}", timeout=args.timeout)
    if args.command == "store-category":
        return _request_json("GET", "/store/category", timeout=args.timeout)
    if args.command == "store-good":
        params = {
            key: value
            for key, value in {"page": args.page, "category": args.category, "search": args.search}.items()
            if value not in (None, "")
        }
        return _request_json("GET", "/store/good", params=params, timeout=args.timeout)
    if args.command == "request":
        params = _parse_key_value_pairs(args.param)
        json_payload = _parse_key_value_pairs(args.json_field)
        if args.json_string:
            json_payload = json.loads(args.json_string)
        return _request_json(
            args.method,
            args.path,
            params=params or None,
            json_payload=json_payload or None,
            timeout=args.timeout,
        )
    raise ValueError(f"Unsupported command: {args.command}")


def _print_summary(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if args.command == "settings":
        _print_settings_summary(result, args.limit)
        return
    if args.command == "top":
        _print_top_summary(result, args.limit)
        return
    if args.command == "lines":
        _print_lines_summary(result, args.limit)
        return
    if args.command == "match":
        _print_match_summary(result, args.events_limit)
        return
    if args.command == "fantasy-rooms":
        _print_fantasy_rooms_summary(result, args.limit)
        return
    if args.command == "fantasy-room":
        _print_fantasy_room_summary(result, args.teams_limit, args.players_limit)
        return
    if args.command == "fantasy-ticket":
        _print_fantasy_ticket_summary(result)
        return
    if args.command == "store-category":
        _print_store_categories_summary(result, args.limit)
        return
    if args.command == "store-good":
        _print_store_goods_summary(result, args.limit)
        return
    _print_generic_summary(result, getattr(args, "limit", 10))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except ArtlineApiError as exc:
        result = exc.result
        _write_output(args.output, result)
        if args.json:
            _print_result_json(result)
        else:
            _print_generic_summary(result, getattr(args, "limit", 10))
        return 1
    _write_output(args.output, result)
    if args.json:
        _print_result_json(result)
    else:
        _print_summary(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from settings import apply_config_env

load_dotenv()
apply_config_env()

from flask import Flask, jsonify, render_template, request  # noqa: E402

from config import (  # noqa: E402
    BOOKMAKER_OPTIONS,
    BOOKMAKER_URLS,
    DEFAULT_ALL_SPORTS,
    DEFAULT_ARBITRAGE_SORT,
    DEFAULT_BANKROLL,
    DEFAULT_BOOKMAKER_KEYS,
    DEFAULT_COMMISSION,
    DEFAULT_DENSITY,
    DEFAULT_EXCHANGE_ONLY,
    DEFAULT_LANGUAGE,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIDDLE_SORT,
    DEFAULT_MIN_ROI,
    DEFAULT_NOTIFY_POPUP_ENABLED,
    DEFAULT_NOTIFY_SOUND_ENABLED,
    DEFAULT_ODDS_FORMAT,
    DEFAULT_PLUS_EV_SORT,
    DEFAULT_REGION_KEYS,
    DEFAULT_SHARP_BOOK,
    DEFAULT_SPORT_OPTIONS,
    DEFAULT_STAKE_AMOUNT,
    DEFAULT_AUTO_SCAN_ENABLED,
    DEFAULT_AUTO_SCAN_MINUTES,
    KELLY_OPTIONS,
    MIN_MIDDLE_GAP,
    MIN_EDGE_PERCENT,
    REGION_OPTIONS,
    SHARP_BOOKS,
    SHOW_POSITIVE_EV_ONLY,
)
from scanner import run_scan  # noqa: E402

app = Flask(__name__)


def _env_flag(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _split_api_keys(value: Optional[str]) -> list[str]:
    if not value:
        return []
    parts = [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]
    keys = []
    seen = set()
    for key in parts:
        if key in seen:
            continue
        keys.append(key)
        seen.add(key)
    return keys


def _payload_api_keys(payload: dict) -> list[str]:
    keys = payload.get("apiKeys")
    if isinstance(keys, list):
        normalized = []
        seen = set()
        for key in keys:
            if not isinstance(key, str):
                continue
            cleaned = key.strip()
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        if normalized:
            return normalized
    elif isinstance(keys, str):
        normalized = _split_api_keys(keys)
        if normalized:
            return normalized
    single = payload.get("apiKey")
    if isinstance(single, str):
        return _split_api_keys(single)
    return []


ENV_API_KEYS = _split_api_keys(
    os.getenv("ODDS_API_KEYS") or os.getenv("THEODDSAPI_API_KEYS")
)
if not ENV_API_KEYS:
    ENV_API_KEYS = _split_api_keys(
        os.getenv("ODDS_API_KEY") or os.getenv("THEODDSAPI_API_KEY")
    )

ENV_ALL_MARKETS = _env_flag(os.getenv("ARBITRAGE_ALL_MARKETS"))
ENV_PUREBET_ENABLED = _env_flag(os.getenv("PUREBET_ENABLED"))
ENV_SAVE_SCAN = _env_flag(os.getenv("SCAN_SAVE_ENABLED"))
ENV_SAVE_DIR = os.getenv("SCAN_SAVE_DIR", str(Path("data") / "scans")).strip()


def _should_save_scan(payload: dict) -> bool:
    if "saveScan" in payload:
        return bool(payload.get("saveScan"))
    return ENV_SAVE_SCAN


def _save_scan_payload(payload: dict, result: dict) -> Optional[str]:
    if not ENV_SAVE_DIR:
        return None
    try:
        target_dir = Path(ENV_SAVE_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{timestamp}_{os.urandom(3).hex()}.json"
        path = target_dir / filename
        data = {
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "request": payload,
            "result": result,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return str(path)
    except OSError:
        return None


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        default_sports=DEFAULT_SPORT_OPTIONS,
        region_options=REGION_OPTIONS,
        bookmaker_options=BOOKMAKER_OPTIONS,
        default_bookmaker_keys=DEFAULT_BOOKMAKER_KEYS,
        default_commission_percent=int(DEFAULT_COMMISSION * 100),
        has_env_key=bool(ENV_API_KEYS),
        sharp_books=SHARP_BOOKS,
        default_sharp_book=DEFAULT_SHARP_BOOK,
        default_min_edge_percent=MIN_EDGE_PERCENT,
        default_bankroll=DEFAULT_BANKROLL,
        default_kelly_fraction=DEFAULT_KELLY_FRACTION,
        default_stake_amount=DEFAULT_STAKE_AMOUNT,
        default_min_roi=DEFAULT_MIN_ROI,
        default_exchange_only=DEFAULT_EXCHANGE_ONLY,
        default_arbitrage_sort=DEFAULT_ARBITRAGE_SORT,
        default_min_gap=MIN_MIDDLE_GAP,
        default_positive_ev_only=SHOW_POSITIVE_EV_ONLY,
        default_middle_sort=DEFAULT_MIDDLE_SORT,
        default_plus_ev_sort=DEFAULT_PLUS_EV_SORT,
        default_auto_scan_enabled=DEFAULT_AUTO_SCAN_ENABLED,
        default_auto_scan_minutes=DEFAULT_AUTO_SCAN_MINUTES,
        default_notify_sound_enabled=DEFAULT_NOTIFY_SOUND_ENABLED,
        default_notify_popup_enabled=DEFAULT_NOTIFY_POPUP_ENABLED,
        default_odds_format=DEFAULT_ODDS_FORMAT,
        default_density=DEFAULT_DENSITY,
        default_language=DEFAULT_LANGUAGE,
        default_all_sports=DEFAULT_ALL_SPORTS,
        kelly_options=KELLY_OPTIONS,
        bookmaker_links=BOOKMAKER_URLS,
    )


@app.route("/scan", methods=["POST"])
def scan() -> tuple:
    payload = request.get_json(force=True, silent=True) or {}
    api_keys = ENV_API_KEYS or _payload_api_keys(payload)
    sports = payload.get("sports") or []
    all_sports = bool(payload.get("allSports")) if "allSports" in payload else DEFAULT_ALL_SPORTS
    all_markets = bool(payload.get("allMarkets")) if "allMarkets" in payload else ENV_ALL_MARKETS
    stake = payload.get("stake")
    regions = payload.get("regions")
    bookmakers = payload.get("bookmakers")
    commission = payload.get("commission")
    include_purebet = (
        bool(payload.get("includePurebet")) if "includePurebet" in payload else ENV_PUREBET_ENABLED
    )
    sharp_book = (payload.get("sharpBook") or DEFAULT_SHARP_BOOK).strip().lower()
    try:
        min_edge_percent = (
            float(payload.get("minEdgePercent")) if payload.get("minEdgePercent") is not None else MIN_EDGE_PERCENT
        )
    except (TypeError, ValueError):
        min_edge_percent = MIN_EDGE_PERCENT
    min_edge_percent = max(0.0, min_edge_percent)
    try:
        bankroll_value = float(payload.get("bankroll")) if payload.get("bankroll") is not None else DEFAULT_BANKROLL
    except (TypeError, ValueError):
        bankroll_value = DEFAULT_BANKROLL
    bankroll_value = max(0.0, bankroll_value)
    try:
        kelly_fraction = (
            float(payload.get("kellyFraction"))
            if payload.get("kellyFraction") is not None
            else DEFAULT_KELLY_FRACTION
        )
    except (TypeError, ValueError):
        kelly_fraction = DEFAULT_KELLY_FRACTION
    kelly_fraction = max(0.0, min(kelly_fraction, 1.0))
    try:
        stake_value = float(stake) if stake is not None else DEFAULT_STAKE_AMOUNT
    except (TypeError, ValueError):
        stake_value = DEFAULT_STAKE_AMOUNT
    if isinstance(regions, list):
        regions_value = [str(region) for region in regions if isinstance(region, str)]
    else:
        regions_value = None
    if isinstance(bookmakers, list):
        bookmakers_value = [str(book) for book in bookmakers if isinstance(book, str) and book.strip()]
    else:
        bookmakers_value = None
    try:
        commission_percent = float(commission) if commission is not None else None
    except (TypeError, ValueError):
        commission_percent = None
    commission_rate = (
        commission_percent / 100.0 if commission_percent is not None else DEFAULT_COMMISSION
    )
    result = run_scan(
        api_key=api_keys,
        sports=sports,
        all_sports=all_sports,
        all_markets=all_markets,
        stake_amount=stake_value,
        regions=regions_value or DEFAULT_REGION_KEYS,
        bookmakers=bookmakers_value,
        commission_rate=commission_rate,
        sharp_book=sharp_book,
        min_edge_percent=min_edge_percent,
        bankroll=bankroll_value,
        kelly_fraction=kelly_fraction,
        include_purebet=include_purebet,
    )
    if _should_save_scan(payload):
        saved_path = _save_scan_payload(payload, result)
        if saved_path:
            result["scan_saved_path"] = saved_path
        else:
            result["scan_save_error"] = "Failed to save scan payload"
    status = 200 if result.get("success") else result.get("error_code", 500)
    return jsonify(result), status


def _port_available(port: int) -> bool:
    if port <= 0:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _choose_port(preferred: Optional[int]) -> int:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([5000, 5050, 8000])
    seen = set()
    for port in candidates:
        if port in seen:
            continue
        seen.add(port)
        if _port_available(port):
            return port
    return 0  # fall back to OS-chosen port


def open_browser(port: int) -> None:
    webbrowser.open_new(f"http://localhost:{port}/")


def ensure_data_dir() -> None:
    Path("data").mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sports Arbitrage Scanner server")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the local server on (default 5000, auto-fallback if busy)",
    )
    args = parser.parse_args()

    ensure_data_dir()
    port = _choose_port(args.port)
    if port > 0:
        timer = threading.Timer(1.0, open_browser, args=(port,))
        timer.start()
    else:
        timer = None
    try:
        app.run(port=port or 0, debug=False)
    finally:
        if timer:
            timer.cancel()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from settings import apply_config_env
from werkzeug.exceptions import BadRequest

load_dotenv()
apply_config_env()

from flask import Flask, jsonify, render_template, request, send_from_directory  # noqa: E402

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
    DEFAULT_THEME,
    DEFAULT_AUTO_SCAN_ENABLED,
    DEFAULT_AUTO_SCAN_MINUTES,
    KELLY_OPTIONS,
    MIN_MIDDLE_GAP,
    MIN_EDGE_PERCENT,
    REGION_OPTIONS,
    SHARP_BOOKS,
    SHOW_POSITIVE_EV_ONLY,
)
from providers import PROVIDER_FETCHERS, resolve_provider_key  # noqa: E402
from scanner import run_scan  # noqa: E402
from history import get_history_manager  # noqa: E402
from notifier import get_notifier  # noqa: E402

app = Flask(__name__)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        app.static_folder,
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return default
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _env_flag(value: Optional[str]) -> bool:
    return _coerce_bool(value, default=False)


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
ENV_SAVE_SCAN = _env_flag(os.getenv("SCAN_SAVE_ENABLED"))
ENV_SAVE_DIR = os.getenv("SCAN_SAVE_DIR", str(Path("data") / "scans")).strip()


def _should_save_scan(payload: dict) -> bool:
    if "saveScan" in payload:
        return _coerce_bool(payload.get("saveScan"), default=ENV_SAVE_SCAN)
    return ENV_SAVE_SCAN


def _save_scan_payload(payload: dict, result: dict) -> Optional[str]:
    if not ENV_SAVE_DIR:
        return None
    try:
        target_dir = Path(ENV_SAVE_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)
        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{timestamp}_{os.urandom(3).hex()}.json"
        path = target_dir / filename
        data = {
            "saved_at": now_utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "request": _sanitize_scan_request(payload),
            "result": result,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        _cleanup_old_scan_payloads(target_dir, keep_path=path)
        return str(path)
    except OSError:
        return None


def _cleanup_old_scan_payloads(target_dir: Path, keep_path: Path) -> None:
    for scan_file in target_dir.glob("scan_*.json"):
        if scan_file == keep_path:
            continue
        try:
            scan_file.unlink()
        except OSError:
            continue


def _sanitize_scan_request(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    sanitized = dict(payload)
    if "apiKey" in sanitized:
        value = sanitized.get("apiKey")
        if isinstance(value, str) and value.strip():
            sanitized["apiKey"] = "***redacted***"
    if "apiKeys" in sanitized:
        value = sanitized.get("apiKeys")
        if isinstance(value, list):
            sanitized["apiKeys"] = ["***redacted***" for _ in value]
        elif isinstance(value, str) and value.strip():
            sanitized["apiKeys"] = "***redacted***"
    return sanitized


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
        default_theme=DEFAULT_THEME,
        default_language=DEFAULT_LANGUAGE,
        default_all_sports=DEFAULT_ALL_SPORTS,
        default_all_markets=ENV_ALL_MARKETS,
        kelly_options=KELLY_OPTIONS,
        bookmaker_links=BOOKMAKER_URLS,
        custom_provider_keys=list(PROVIDER_FETCHERS.keys()),
    )


@app.route("/scan", methods=["POST"])
def scan() -> tuple:
    raw_body = request.get_data(cache=True, as_text=False)
    if raw_body:
        try:
            payload = request.get_json(force=True, silent=False)
        except BadRequest:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid JSON payload",
                        "error_code": 400,
                    }
                ),
                400,
            )
    else:
        payload = {}
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Scan payload must be a JSON object",
                    "error_code": 400,
                }
            ),
            400,
        )
    api_keys = ENV_API_KEYS or _payload_api_keys(payload)
    sports = payload.get("sports") or []
    all_sports = _coerce_bool(payload.get("allSports"), default=DEFAULT_ALL_SPORTS)
    all_markets = _coerce_bool(payload.get("allMarkets"), default=ENV_ALL_MARKETS)
    stake = payload.get("stake")
    regions = payload.get("regions")
    bookmakers = payload.get("bookmakers")
    commission = payload.get("commission")
    include_purebet = (
        _coerce_bool(payload.get("includePurebet"), default=False)
        if "includePurebet" in payload
        else None
    )
    include_providers_raw = payload.get("includeProviders")
    sharp_book_raw = payload.get("sharpBook")
    if isinstance(sharp_book_raw, str):
        sharp_book = sharp_book_raw.strip().lower() or DEFAULT_SHARP_BOOK
    else:
        sharp_book = DEFAULT_SHARP_BOOK
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
    if isinstance(include_providers_raw, list):
        include_providers_value = [
            str(provider)
            for provider in include_providers_raw
            if isinstance(provider, str) and provider.strip()
        ]
    elif isinstance(include_providers_raw, str):
        include_providers_value = [
            item.strip()
            for item in re.split(r"[,\s]+", include_providers_raw)
            if item.strip()
        ]
    else:
        include_providers_value = None
    if include_providers_value is None and bookmakers_value:
        derived = []
        seen = set()
        for book in bookmakers_value:
            provider_key = resolve_provider_key(book)
            if provider_key and provider_key not in seen:
                derived.append(provider_key)
                seen.add(provider_key)
        include_providers_value = derived
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
        include_providers=include_providers_value,
    )
    if result.get("success"):
        scan_time = result.get("scan_time", "")
        # Persist history (non-blocking)
        try:
            get_history_manager().save_opportunities(result, scan_time=scan_time)
        except Exception:
            pass
        # Send notifications in background thread (non-blocking)
        notifier = get_notifier()
        if notifier.is_configured:
            def _notify():
                try:
                    notifier.notify_opportunities(
                        arb_list=result.get("opportunities") or [],
                        middle_list=result.get("middles") or [],
                        ev_list=result.get("plus_ev") or [],
                        scan_time=scan_time,
                    )
                except Exception:
                    pass
            threading.Thread(target=_notify, daemon=True).start()
    if _should_save_scan(payload):
        saved_path = _save_scan_payload(payload, result)
        if saved_path:
            result["scan_saved_path"] = saved_path
        else:
            result["scan_save_error"] = "Failed to save scan payload"
    status = 200 if result.get("success") else result.get("error_code", 500)
    return jsonify(result), status


@app.route("/history", methods=["GET"])
def history() -> tuple:
    """Return recent scan history opportunities.

    Query params:
        mode    — ``arbitrage``, ``middles``, ``ev`` (default: all)
        limit   — max records to return (default 200, max 1000)
    """
    mode = request.args.get("mode", "").strip().lower() or None
    try:
        limit = min(1000, max(1, int(request.args.get("limit", "200"))))
    except (TypeError, ValueError):
        limit = 200
    try:
        records = get_history_manager().load_recent(limit=limit, mode=mode)
        return jsonify({"success": True, "records": records, "count": len(records)}), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/history/stats", methods=["GET"])
def history_stats() -> tuple:
    """Return history storage statistics."""
    try:
        stats = get_history_manager().get_stats()
        return jsonify({"success": True, **stats}), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


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

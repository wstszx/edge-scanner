from __future__ import annotations

import atexit
import argparse
import copy
import json
import logging
import os
import re
import socket
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import requests
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
    DEFAULT_SPORT_KEYS,
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
    SPORT_OPTIONS,
)
from providers import PROVIDER_FETCHERS, resolve_provider_key  # noqa: E402
from scanner import (  # noqa: E402
    DEFAULT_LIVE_PROVIDER_KEYS,
    LIVE_SUPPORTED_PROVIDER_KEYS,
    SCAN_MODE_LIVE,
    SCAN_MODE_PREMATCH,
    run_scan,
)
from history import get_history_manager  # noqa: E402
from notifier import get_notifier  # noqa: E402

app = Flask(__name__)
_BACKGROUND_SERVICES_LOCK = threading.Lock()
_BACKGROUND_SERVICES_STARTED = False
_SERVER_AUTO_SCAN_LOCK = threading.Lock()
_SERVER_AUTO_SCAN_THREAD: Optional[threading.Thread] = None
_SERVER_AUTO_SCAN_STOP_EVENT = threading.Event()
_SCAN_EXECUTION_LOCK = threading.Lock()
_SERVER_AUTO_SCAN_CONFIG: Optional[dict] = None
_SERVER_AUTO_SCAN_CONFIG_VERSION = 0
_FX_RATE_CACHE_LOCK = threading.Lock()
_FX_RATE_CACHE: Optional[dict] = None
_FX_RATE_CACHE_EXPIRES_AT = 0.0

ARB_CALC_SUPPORTED_CURRENCIES = (
    "USD",
    "EUR",
    "CNY",
    "HKD",
    "GBP",
    "JPY",
    "KRW",
    "SGD",
    "AUD",
    "CAD",
    "CHF",
    "NZD",
)
DEFAULT_ARB_CALC_CURRENCY = "USD"
FX_RATE_PROVIDER_NAME = "Frankfurter"
FX_RATE_PROVIDER_URL = os.getenv(
    "FX_RATE_PROVIDER_URL",
    "https://api.frankfurter.dev/v1/latest",
).strip()
FX_RATE_REFERENCE_CURRENCY = "EUR"
try:
    FX_RATE_CACHE_TTL_SECONDS = max(
        60.0,
        float(os.getenv("FX_RATE_CACHE_TTL_SECONDS", "1800").strip()),
    )
except (TypeError, ValueError):
    FX_RATE_CACHE_TTL_SECONDS = 1800.0
try:
    FX_RATE_TIMEOUT_SECONDS = max(
        2.0,
        float(os.getenv("FX_RATE_TIMEOUT_SECONDS", "10").strip()),
    )
except (TypeError, ValueError):
    FX_RATE_TIMEOUT_SECONDS = 10.0


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
ENV_PROVIDER_ONLY_MODE = _coerce_bool(
    os.getenv("SCAN_CUSTOM_PROVIDERS_ONLY"),
    default=True,
)
ENV_SERVER_AUTO_SCAN_ENABLED = _coerce_bool(
    os.getenv("SERVER_AUTO_SCAN_ENABLED"),
    default=True,
)
ENV_SERVER_AUTO_SCAN_RUN_ON_START = _coerce_bool(
    os.getenv("SERVER_AUTO_SCAN_RUN_ON_START"),
    default=True,
)
try:
    ENV_SERVER_AUTO_SCAN_INTERVAL_MINUTES = max(
        1,
        int(float(os.getenv("SERVER_AUTO_SCAN_INTERVAL_MINUTES", str(DEFAULT_AUTO_SCAN_MINUTES)).strip())),
    )
except (TypeError, ValueError):
    ENV_SERVER_AUTO_SCAN_INTERVAL_MINUTES = DEFAULT_AUTO_SCAN_MINUTES
ENV_SERVER_AUTO_SCAN_CONFIG_PATH = os.getenv(
    "SERVER_AUTO_SCAN_CONFIG_PATH",
    str(Path("data") / "server_auto_scan_config.json"),
).strip()
ENV_SAVE_SCAN = _env_flag(os.getenv("SCAN_SAVE_ENABLED"))
ENV_SAVE_DIR = os.getenv("SCAN_SAVE_DIR", str(Path("data") / "scans")).strip()
ENV_PROVIDER_SNAPSHOT_DIR = os.getenv(
    "CUSTOM_PROVIDER_SNAPSHOT_DIR", str(Path("data") / "provider_snapshots")
).strip()
ENV_CROSS_PROVIDER_REPORT_FILENAME = os.getenv(
    "CROSS_PROVIDER_MATCH_REPORT_FILENAME", "cross_provider_match_report.json"
).strip()


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


def _provider_snapshot_path(provider_key: str) -> Optional[Path]:
    if not isinstance(provider_key, str):
        return None
    token = re.sub(r"[^a-z0-9._-]+", "_", provider_key.strip().lower())
    if not token:
        return None
    base_dir = Path(ENV_PROVIDER_SNAPSHOT_DIR)
    return base_dir / f"{token}.json"


def _cross_provider_report_path() -> Optional[Path]:
    token = re.sub(
        r"[^a-z0-9._-]+",
        "_",
        str(ENV_CROSS_PROVIDER_REPORT_FILENAME or "").strip().lower(),
    )
    if not token:
        token = "cross_provider_match_report.json"
    if not token.endswith(".json"):
        token = f"{token}.json"
    base_dir = Path(ENV_PROVIDER_SNAPSHOT_DIR)
    return base_dir / token


def _start_background_provider_services(wait_timeout: Optional[float] = None) -> None:
    global _BACKGROUND_SERVICES_STARTED
    with _BACKGROUND_SERVICES_LOCK:
        if _BACKGROUND_SERVICES_STARTED:
            return
        for provider_key in ("polymarket", "sx_bet"):
            try:
                if provider_key == "polymarket":
                    from providers.polymarket import ensure_realtime_started
                else:
                    from providers.sx_bet import ensure_realtime_started
                ensure_realtime_started(wait_timeout=wait_timeout)
            except Exception:
                logging.warning("Failed to prewarm provider background service: %s", provider_key, exc_info=True)
        _BACKGROUND_SERVICES_STARTED = True


def _stop_background_provider_services() -> None:
    global _BACKGROUND_SERVICES_STARTED
    with _BACKGROUND_SERVICES_LOCK:
        for provider_key in ("polymarket", "sx_bet"):
            try:
                if provider_key == "polymarket":
                    from providers.polymarket import stop_realtime
                else:
                    from providers.sx_bet import stop_realtime
                stop_realtime()
            except Exception:
                logging.warning("Failed to stop provider background service: %s", provider_key, exc_info=True)
        _BACKGROUND_SERVICES_STARTED = False


def _provider_runtime_status(provider_key: str) -> Optional[dict]:
    key = resolve_provider_key(provider_key)
    if key == "polymarket":
        from providers.polymarket import realtime_status
        return realtime_status()
    if key == "sx_bet":
        from providers.sx_bet import realtime_status
        return realtime_status()
    return None


atexit.register(_stop_background_provider_services)


def _extract_opportunity_list(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        opportunities = payload.get("opportunities")
        if isinstance(opportunities, list):
            return [item for item in opportunities if isinstance(item, dict)]
    return []


def _normalize_scan_mode(value: object) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {SCAN_MODE_PREMATCH, SCAN_MODE_LIVE}:
            return text
    return SCAN_MODE_PREMATCH


def _server_auto_scan_payload() -> dict:
    provider_keys = list(PROVIDER_FETCHERS.keys())
    payload = {
        "scanMode": SCAN_MODE_PREMATCH,
        "sports": list(DEFAULT_SPORT_KEYS),
        "allSports": DEFAULT_ALL_SPORTS,
        "allMarkets": ENV_ALL_MARKETS,
        "stake": DEFAULT_STAKE_AMOUNT,
        "regions": list(DEFAULT_REGION_KEYS),
        "bookmakers": provider_keys if ENV_PROVIDER_ONLY_MODE else list(DEFAULT_BOOKMAKER_KEYS),
        "commission": float(DEFAULT_COMMISSION * 100.0),
        "sharpBook": DEFAULT_SHARP_BOOK,
        "minEdgePercent": MIN_EDGE_PERCENT,
        "bankroll": DEFAULT_BANKROLL,
        "kellyFraction": DEFAULT_KELLY_FRACTION,
    }
    if ENV_PROVIDER_ONLY_MODE:
        payload["includeProviders"] = provider_keys
    return payload


def _default_server_auto_scan_config() -> dict:
    return {
        "enabled": bool(ENV_SERVER_AUTO_SCAN_ENABLED),
        "interval_minutes": int(ENV_SERVER_AUTO_SCAN_INTERVAL_MINUTES),
        "payload": _server_auto_scan_payload(),
    }


def _server_auto_scan_config_summary(config: Optional[dict]) -> str:
    if not isinstance(config, dict):
        return "enabled=False config=missing"
    payload = config.get("payload") if isinstance(config.get("payload"), dict) else {}
    sports = payload.get("sports") if isinstance(payload.get("sports"), list) else []
    bookmakers = payload.get("bookmakers") if isinstance(payload.get("bookmakers"), list) else []
    include_providers = (
        payload.get("includeProviders")
        if isinstance(payload.get("includeProviders"), list)
        else []
    )
    providers_label = ",".join(str(item) for item in include_providers[:6]) or "-"
    scan_mode = _normalize_scan_mode(payload.get("scanMode"))
    return (
        f"enabled={bool(config.get('enabled'))} "
        f"interval={int(config.get('interval_minutes') or ENV_SERVER_AUTO_SCAN_INTERVAL_MINUTES)}m "
        f"mode={scan_mode} sports={len(sports)} bookmakers={len(bookmakers)} providers={providers_label}"
    )


def _server_auto_scan_config_path() -> Optional[Path]:
    raw = str(ENV_SERVER_AUTO_SCAN_CONFIG_PATH or "").strip()
    if not raw:
        return None
    return Path(raw)


def _normalize_server_auto_scan_config(raw: object) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    scan_payload = raw.get("payload")
    if not isinstance(scan_payload, dict):
        scan_payload = raw

    sports_raw = scan_payload.get("sports")
    if isinstance(sports_raw, list):
        sports = [str(item) for item in sports_raw if isinstance(item, str) and item.strip()]
    else:
        sports = list(DEFAULT_SPORT_KEYS)

    regions_raw = scan_payload.get("regions")
    if isinstance(regions_raw, list):
        regions = [str(item) for item in regions_raw if isinstance(item, str) and item.strip()]
    else:
        regions = list(DEFAULT_REGION_KEYS)

    bookmakers_raw = scan_payload.get("bookmakers")
    if isinstance(bookmakers_raw, list):
        bookmakers = [str(item) for item in bookmakers_raw if isinstance(item, str) and item.strip()]
    else:
        bookmakers = []

    include_providers_raw = scan_payload.get("includeProviders")
    if isinstance(include_providers_raw, list):
        include_providers = [
            str(item)
            for item in include_providers_raw
            if isinstance(item, str) and item.strip()
        ]
    else:
        include_providers = []
    scan_mode = _normalize_scan_mode(scan_payload.get("scanMode"))

    try:
        stake_value = (
            float(scan_payload.get("stake"))
            if scan_payload.get("stake") is not None
            else DEFAULT_STAKE_AMOUNT
        )
    except (TypeError, ValueError):
        stake_value = DEFAULT_STAKE_AMOUNT

    try:
        commission_value = (
            float(scan_payload.get("commission"))
            if scan_payload.get("commission") is not None
            else float(DEFAULT_COMMISSION * 100.0)
        )
    except (TypeError, ValueError):
        commission_value = float(DEFAULT_COMMISSION * 100.0)

    try:
        min_edge_percent = (
            float(scan_payload.get("minEdgePercent"))
            if scan_payload.get("minEdgePercent") is not None
            else MIN_EDGE_PERCENT
        )
    except (TypeError, ValueError):
        min_edge_percent = MIN_EDGE_PERCENT

    try:
        bankroll_value = (
            float(scan_payload.get("bankroll"))
            if scan_payload.get("bankroll") is not None
            else DEFAULT_BANKROLL
        )
    except (TypeError, ValueError):
        bankroll_value = DEFAULT_BANKROLL

    try:
        kelly_fraction = (
            float(scan_payload.get("kellyFraction"))
            if scan_payload.get("kellyFraction") is not None
            else DEFAULT_KELLY_FRACTION
        )
    except (TypeError, ValueError):
        kelly_fraction = DEFAULT_KELLY_FRACTION

    try:
        interval_minutes = (
            int(float(raw.get("intervalMinutes")))
            if raw.get("intervalMinutes") is not None
            else int(float(raw.get("interval_minutes")))
            if raw.get("interval_minutes") is not None
            else ENV_SERVER_AUTO_SCAN_INTERVAL_MINUTES
        )
    except (TypeError, ValueError):
        interval_minutes = ENV_SERVER_AUTO_SCAN_INTERVAL_MINUTES
    interval_minutes = max(1, interval_minutes)

    sharp_book_raw = scan_payload.get("sharpBook")
    if isinstance(sharp_book_raw, str):
        sharp_book = sharp_book_raw.strip().lower() or DEFAULT_SHARP_BOOK
    else:
        sharp_book = DEFAULT_SHARP_BOOK

    return {
        "enabled": _coerce_bool(raw.get("enabled"), default=ENV_SERVER_AUTO_SCAN_ENABLED),
        "interval_minutes": interval_minutes,
        "payload": {
            "scanMode": scan_mode,
            "sports": sports,
            "allSports": _coerce_bool(scan_payload.get("allSports"), default=DEFAULT_ALL_SPORTS),
            "allMarkets": _coerce_bool(scan_payload.get("allMarkets"), default=ENV_ALL_MARKETS),
            "stake": stake_value,
            "regions": regions,
            "bookmakers": bookmakers,
            "includeProviders": include_providers,
            "commission": commission_value,
            "sharpBook": sharp_book,
            "minEdgePercent": max(0.0, min_edge_percent),
            "bankroll": max(0.0, bankroll_value),
            "kellyFraction": max(0.0, min(kelly_fraction, 1.0)),
        },
    }


def _persist_server_auto_scan_config(config: dict) -> None:
    path = _server_auto_scan_config_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def _load_server_auto_scan_config() -> Optional[dict]:
    path = _server_auto_scan_config_path()
    if path is None or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logging.warning("Failed to load server auto scan config", exc_info=True)
        return None
    return _normalize_server_auto_scan_config(payload)


def _set_server_auto_scan_config(config: Optional[dict], *, persist: bool = True) -> None:
    global _SERVER_AUTO_SCAN_CONFIG
    global _SERVER_AUTO_SCAN_CONFIG_VERSION
    normalized = _normalize_server_auto_scan_config(config) if config is not None else None
    with _SERVER_AUTO_SCAN_LOCK:
        _SERVER_AUTO_SCAN_CONFIG = copy.deepcopy(normalized) if normalized is not None else None
        _SERVER_AUTO_SCAN_CONFIG_VERSION += 1
    if persist and normalized is not None:
        try:
            _persist_server_auto_scan_config(normalized)
        except OSError:
            logging.warning("Failed to persist server auto scan config", exc_info=True)
    if normalized is None:
        logging.info("Server auto scan config cleared")
    else:
        logging.info(
            "Server auto scan config updated: %s",
            _server_auto_scan_config_summary(normalized),
        )


def _get_server_auto_scan_config() -> tuple[Optional[dict], int]:
    with _SERVER_AUTO_SCAN_LOCK:
        config = copy.deepcopy(_SERVER_AUTO_SCAN_CONFIG)
        version = int(_SERVER_AUTO_SCAN_CONFIG_VERSION)
    return config, version


def _execute_scan_payload(
    payload: dict,
    *,
    save_scan_override: Optional[bool] = None,
    background: bool = False,
) -> dict:
    if background:
        acquired = _SCAN_EXECUTION_LOCK.acquire(blocking=False)
        if not acquired:
            return {
                "success": False,
                "error": "Scan already in progress",
                "error_code": 409,
            }
    else:
        _SCAN_EXECUTION_LOCK.acquire()
    try:
        _start_background_provider_services(wait_timeout=0.0)
        api_keys = ENV_API_KEYS or _payload_api_keys(payload)
        scan_mode = _normalize_scan_mode(payload.get("scanMode"))
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
                float(payload.get("minEdgePercent"))
                if payload.get("minEdgePercent") is not None
                else MIN_EDGE_PERCENT
            )
        except (TypeError, ValueError):
            min_edge_percent = MIN_EDGE_PERCENT
        min_edge_percent = max(0.0, min_edge_percent)
        try:
            bankroll_value = (
                float(payload.get("bankroll"))
                if payload.get("bankroll") is not None
                else DEFAULT_BANKROLL
            )
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
            bookmakers_value = [
                str(book)
                for book in bookmakers
                if isinstance(book, str) and book.strip()
            ]
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
        if (include_providers_value is None or not include_providers_value) and bookmakers_value:
            derived = []
            seen = set()
            for book in bookmakers_value:
                provider_key = resolve_provider_key(book)
                if provider_key and provider_key not in seen:
                    derived.append(provider_key)
                    seen.add(provider_key)
            include_providers_value = derived
        if ENV_PROVIDER_ONLY_MODE:
            api_keys = []
            provider_bookmakers = []
            seen_provider_books = set()
            for book in bookmakers_value or []:
                provider_key = resolve_provider_key(book)
                if not provider_key or provider_key in seen_provider_books:
                    continue
                provider_bookmakers.append(provider_key)
                seen_provider_books.add(provider_key)
            bookmakers_value = provider_bookmakers or None

            normalized_providers = []
            seen_providers = set()
            for provider in include_providers_value or []:
                provider_key = resolve_provider_key(provider)
                if not provider_key or provider_key in seen_providers:
                    continue
                normalized_providers.append(provider_key)
                seen_providers.add(provider_key)
            include_providers_value = (
                normalized_providers
                or provider_bookmakers
                or list(PROVIDER_FETCHERS.keys())
            )
        try:
            commission_percent = float(commission) if commission is not None else None
        except (TypeError, ValueError):
            commission_percent = None
        commission_rate = (
            commission_percent / 100.0
            if commission_percent is not None
            else DEFAULT_COMMISSION
        )
        result = run_scan(
            api_key=api_keys,
            sports=sports,
            scan_mode=scan_mode,
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
        if isinstance(result, dict) and "scan_mode" not in result:
            result["scan_mode"] = scan_mode
        if result.get("success"):
            scan_time = result.get("scan_time", "")
            arbitrage_items = _extract_opportunity_list(result.get("arbitrage"))
            if not arbitrage_items:
                arbitrage_items = _extract_opportunity_list(result.get("opportunities"))
            middle_items = _extract_opportunity_list(result.get("middles"))
            plus_ev_items = _extract_opportunity_list(result.get("plus_ev"))
            try:
                get_history_manager().save_opportunities(
                    {
                        "opportunities": arbitrage_items,
                        "middles": middle_items,
                        "plus_ev": plus_ev_items,
                    },
                    scan_time=scan_time,
                )
            except Exception:
                logging.warning("Failed to save scan history", exc_info=True)
            notifier = get_notifier()
            if notifier.is_configured:
                def _notify():
                    try:
                        notifier.notify_opportunities(
                            arb_list=arbitrage_items,
                            middle_list=middle_items,
                            ev_list=plus_ev_items,
                            scan_time=scan_time,
                        )
                    except Exception:
                        logging.warning("Failed to send notifications", exc_info=True)
                threading.Thread(target=_notify, daemon=True).start()
        should_save_scan = (
            _should_save_scan(payload)
            if save_scan_override is None
            else bool(save_scan_override)
        )
        if should_save_scan:
            saved_path = _save_scan_payload(payload, result)
            if saved_path:
                result["scan_saved_path"] = saved_path
            else:
                result["scan_save_error"] = "Failed to save scan payload"
        return result
    finally:
        _SCAN_EXECUTION_LOCK.release()


def _server_auto_scan_loop() -> None:
    next_run_at: Optional[float] = None
    last_config_version = -1
    last_enabled = False
    while not _SERVER_AUTO_SCAN_STOP_EVENT.is_set():
        config, config_version = _get_server_auto_scan_config()
        current_enabled = bool(config and config.get("enabled"))
        if config_version != last_config_version:
            last_config_version = config_version
            if current_enabled:
                interval_seconds = max(
                    60.0,
                    float(config.get("interval_minutes") or ENV_SERVER_AUTO_SCAN_INTERVAL_MINUTES) * 60.0,
                )
                if not last_enabled:
                    next_run_at = time.time() if ENV_SERVER_AUTO_SCAN_RUN_ON_START else time.time() + interval_seconds
                elif next_run_at is None:
                    next_run_at = time.time() + interval_seconds
                else:
                    next_run_at = min(next_run_at, time.time() + interval_seconds)
                logging.info(
                    "Server auto scan scheduled: %s",
                    _server_auto_scan_config_summary(config),
                )
            else:
                next_run_at = None
                logging.info("Server auto scan idle: no enabled config")
        last_enabled = current_enabled
        if not current_enabled:
            if _SERVER_AUTO_SCAN_STOP_EVENT.wait(1.0):
                break
            continue
        interval_seconds = max(
            60.0,
            float(config.get("interval_minutes") or ENV_SERVER_AUTO_SCAN_INTERVAL_MINUTES) * 60.0,
        )
        if next_run_at is None:
            next_run_at = time.time() if ENV_SERVER_AUTO_SCAN_RUN_ON_START else time.time() + interval_seconds
        now = time.time()
        if now >= next_run_at:
            try:
                logging.info(
                    "Server auto scan starting: %s",
                    _server_auto_scan_config_summary(config),
                )
                result = _execute_scan_payload(
                    config.get("payload") or {},
                    background=True,
                    save_scan_override=False,
                )
            except Exception:
                logging.exception("Server auto scan crashed")
                next_run_at = time.time() + interval_seconds
                continue
            if result.get("success"):
                logging.info(
                    "Server auto scan completed at %s",
                    result.get("scan_time", ""),
                )
            else:
                logging.warning(
                    "Server auto scan did not complete: %s",
                    result.get("error", "unknown error"),
                )
            next_run_at = time.time() + interval_seconds
            continue
        wait_seconds = min(1.0, max(0.1, next_run_at - now))
        if _SERVER_AUTO_SCAN_STOP_EVENT.wait(wait_seconds):
            break


def _start_server_auto_scan() -> None:
    global _SERVER_AUTO_SCAN_THREAD
    if not ENV_SERVER_AUTO_SCAN_ENABLED:
        return
    with _SERVER_AUTO_SCAN_LOCK:
        if _SERVER_AUTO_SCAN_THREAD and _SERVER_AUTO_SCAN_THREAD.is_alive():
            return
        _SERVER_AUTO_SCAN_STOP_EVENT.clear()
        _SERVER_AUTO_SCAN_THREAD = threading.Thread(
            target=_server_auto_scan_loop,
            name="server-auto-scan",
            daemon=True,
        )
        _SERVER_AUTO_SCAN_THREAD.start()
    logging.info("Server auto scan thread started")


def _stop_server_auto_scan() -> None:
    global _SERVER_AUTO_SCAN_THREAD
    with _SERVER_AUTO_SCAN_LOCK:
        _SERVER_AUTO_SCAN_STOP_EVENT.set()
        thread = _SERVER_AUTO_SCAN_THREAD
        _SERVER_AUTO_SCAN_THREAD = None
    if thread and thread.is_alive():
        thread.join(timeout=2.0)
    logging.info("Server auto scan thread stopped")


atexit.register(_stop_server_auto_scan)


def _fetch_fx_rate_payload(force: bool = False) -> dict:
    global _FX_RATE_CACHE, _FX_RATE_CACHE_EXPIRES_AT
    now = time.time()
    with _FX_RATE_CACHE_LOCK:
        if (
            not force
            and _FX_RATE_CACHE is not None
            and now < _FX_RATE_CACHE_EXPIRES_AT
        ):
            return copy.deepcopy(_FX_RATE_CACHE)

    symbols = ",".join(
        code
        for code in ARB_CALC_SUPPORTED_CURRENCIES
        if code != FX_RATE_REFERENCE_CURRENCY
    )
    response = requests.get(
        FX_RATE_PROVIDER_URL,
        params={
            "base": FX_RATE_REFERENCE_CURRENCY,
            "symbols": symbols,
        },
        timeout=FX_RATE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    rates = {FX_RATE_REFERENCE_CURRENCY: 1.0}
    raw_rates = payload.get("rates") if isinstance(payload, dict) else {}
    if not isinstance(raw_rates, dict):
        raw_rates = {}
    for code in ARB_CALC_SUPPORTED_CURRENCIES:
        if code == FX_RATE_REFERENCE_CURRENCY:
            continue
        value = raw_rates.get(code)
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        rates[code] = float(value)
    missing = [
        code for code in ARB_CALC_SUPPORTED_CURRENCIES if code not in rates
    ]
    if missing:
        raise ValueError(f"missing FX rates for: {', '.join(missing)}")

    cached_payload = {
        "provider": FX_RATE_PROVIDER_NAME,
        "reference_currency": FX_RATE_REFERENCE_CURRENCY,
        "currencies": list(ARB_CALC_SUPPORTED_CURRENCIES),
        "rates": rates,
        "source_date": payload.get("date") if isinstance(payload, dict) else None,
        "fetched_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "stale": False,
    }
    with _FX_RATE_CACHE_LOCK:
        _FX_RATE_CACHE = copy.deepcopy(cached_payload)
        _FX_RATE_CACHE_EXPIRES_AT = now + FX_RATE_CACHE_TTL_SECONDS
    return cached_payload


def _get_fx_rate_payload(force: bool = False) -> dict:
    try:
        return _fetch_fx_rate_payload(force=force)
    except Exception:
        logging.exception("Failed to refresh FX reference rates")
        with _FX_RATE_CACHE_LOCK:
            if _FX_RATE_CACHE is None:
                raise
            stale_payload = copy.deepcopy(_FX_RATE_CACHE)
        stale_payload["stale"] = True
        return stale_payload


@app.route("/fx-rates", methods=["GET"])
def fx_rates() -> tuple:
    force_refresh = _coerce_bool(request.args.get("force"), default=False)
    try:
        payload = _get_fx_rate_payload(force=force_refresh)
    except Exception:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Failed to load FX rates",
                    "error_code": 502,
                }
            ),
            502,
        )
    return jsonify({"success": True, **payload}), 200


@app.route("/")
def index() -> str:
    _start_background_provider_services(wait_timeout=0.0)
    return render_template(
        "index.html",
        sport_options=SPORT_OPTIONS,
        default_sport_keys=DEFAULT_SPORT_KEYS,
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
        arb_calc_currencies=ARB_CALC_SUPPORTED_CURRENCIES,
        default_arb_calc_currency=DEFAULT_ARB_CALC_CURRENCY,
        default_auto_scan_enabled=DEFAULT_AUTO_SCAN_ENABLED,
        default_auto_scan_minutes=DEFAULT_AUTO_SCAN_MINUTES,
        default_scan_mode=SCAN_MODE_PREMATCH,
        default_notify_sound_enabled=DEFAULT_NOTIFY_SOUND_ENABLED,
        default_notify_popup_enabled=DEFAULT_NOTIFY_POPUP_ENABLED,
        default_odds_format=DEFAULT_ODDS_FORMAT,
        default_density=DEFAULT_DENSITY,
        default_theme=DEFAULT_THEME,
        default_language=DEFAULT_LANGUAGE,
        default_all_sports=DEFAULT_ALL_SPORTS,
        default_all_markets=ENV_ALL_MARKETS,
        provider_only_mode=ENV_PROVIDER_ONLY_MODE,
        kelly_options=KELLY_OPTIONS,
        bookmaker_links=BOOKMAKER_URLS,
        custom_provider_keys=list(PROVIDER_FETCHERS.keys()),
        default_live_provider_keys=list(DEFAULT_LIVE_PROVIDER_KEYS),
        live_supported_provider_keys=list(LIVE_SUPPORTED_PROVIDER_KEYS),
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
    result = _execute_scan_payload(payload)
    status = 200 if result.get("success") else result.get("error_code", 500)
    return jsonify(result), status


@app.route("/server-auto-scan-config", methods=["GET", "POST"])
def server_auto_scan_config() -> tuple:
    if request.method == "GET":
        config, _ = _get_server_auto_scan_config()
        return jsonify({"success": True, "config": config}), 200

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
    normalized = _normalize_server_auto_scan_config(payload)
    if normalized is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Auto scan config payload must be a JSON object",
                    "error_code": 400,
                }
            ),
            400,
        )
    _set_server_auto_scan_config(normalized, persist=True)
    config, _ = _get_server_auto_scan_config()
    return jsonify({"success": True, "config": config}), 200


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


@app.route("/provider-snapshots/<provider_key>", methods=["GET"])
def provider_snapshot(provider_key: str) -> tuple:
    snapshot_path = _provider_snapshot_path(provider_key)
    if snapshot_path is None:
        return jsonify({"success": False, "error": "Invalid provider key"}), 400
    if not snapshot_path.is_file():
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"No snapshot found for provider '{provider_key}'",
                }
            ),
            404,
        )
    try:
        with snapshot_path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    return (
        jsonify(
            {
                "success": True,
                "provider_key": provider_key,
                "snapshot": snapshot,
            }
        ),
        200,
    )


@app.route("/provider-runtime/<provider_key>", methods=["GET"])
def provider_runtime(provider_key: str) -> tuple:
    _start_background_provider_services(wait_timeout=0.0)
    status = _provider_runtime_status(provider_key)
    if status is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"No runtime status is available for provider '{provider_key}'",
                }
            ),
            404,
        )
    return jsonify({"success": True, "provider_key": resolve_provider_key(provider_key), **status}), 200


@app.route("/cross-provider-report", methods=["GET"])
def cross_provider_report() -> tuple:
    report_path = _cross_provider_report_path()
    if report_path is None:
        return jsonify({"success": False, "error": "Invalid report path"}), 400
    if not report_path.is_file():
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Cross-provider report file not found",
                }
            ),
            404,
        )
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    return jsonify({"success": True, "report": report}), 200


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
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
    else:
        logging.getLogger().setLevel(logging.INFO)
    loaded_auto_scan_config = _load_server_auto_scan_config()
    persist_startup_config = loaded_auto_scan_config is None
    if loaded_auto_scan_config is None:
        loaded_auto_scan_config = _default_server_auto_scan_config()
        logging.info("No persisted server auto scan config found; using default startup config")
    _set_server_auto_scan_config(
        loaded_auto_scan_config,
        persist=persist_startup_config,
    )
    _start_background_provider_services()
    _start_server_auto_scan()
    port = _choose_port(args.port)
    if port > 0:
        timer = threading.Timer(1.0, open_browser, args=(port,))
        timer.start()
    else:
        timer = None
    try:
        app.run(port=port or 0, debug=False)
    finally:
        _stop_server_auto_scan()
        _stop_background_provider_services()
        if timer:
            timer.cancel()


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import httpx

from ._async_http import get_shared_client, request_json
from .capabilities import ProviderCapability, sorted_unique_tuple

PROVIDER_KEY = "artline"
PROVIDER_TITLE = "Artline"

ARTLINE_SOURCE = os.getenv("ARTLINE_SOURCE", "api").strip().lower()
ARTLINE_API_BASE = os.getenv("ARTLINE_API_BASE", "https://api.artline.bet/api").strip()
ARTLINE_PUBLIC_BASE = os.getenv("ARTLINE_PUBLIC_BASE", "https://artline.bet").strip()
ARTLINE_TIMEOUT_RAW = os.getenv("ARTLINE_TIMEOUT_SECONDS", "20").strip()
ARTLINE_RETRIES_RAW = os.getenv("ARTLINE_RETRIES", "2").strip()
ARTLINE_RETRY_BACKOFF_RAW = os.getenv("ARTLINE_RETRY_BACKOFF", "0.5").strip()
ARTLINE_DETAIL_MAX_CONCURRENCY_RAW = os.getenv("ARTLINE_DETAIL_MAX_CONCURRENCY", "8").strip()
ARTLINE_MIN_BET_RAW = os.getenv("ARTLINE_MIN_BET", "5").strip()
ARTLINE_CURRENCY_TYPE = os.getenv("ARTLINE_CURRENCY_TYPE", "balance").strip() or "balance"
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
    "tennis_atp": {"sport": "tennis"},
    "tennis_wta": {"sport": "tennis"},
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

PROVIDER_CAPABILITY = ProviderCapability(
    key=PROVIDER_KEY,
    title=PROVIDER_TITLE,
    supported_sport_keys=sorted_unique_tuple(ARTLINE_SPORT_FILTERS),
    supported_markets=sorted_unique_tuple(
        ("h2h", "h2h_3_way", "spreads", "totals", "team_totals")
    ),
    live_mode_supported=True,
    liquidity_confidence="estimated",
    notes=(
        "Artline exposes web-event URLs and max_bet diagnostics; execution is paper/manual-web only.",
    ),
)


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


def _artline_min_bet() -> float:
    value = _safe_float(ARTLINE_MIN_BET_RAW)
    if value is None:
        return 5.0
    return max(0.0, value)


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


def _selection_id(event: object) -> str:
    if not isinstance(event, dict):
        return ""
    return _normalize_text(
        event.get("id")
        or event.get("event_id")
        or event.get("eventId")
        or event.get("selection_id")
        or event.get("selectionId")
    )


def _api_base() -> str:
    base = _normalize_text(ARTLINE_API_BASE) or "https://api.artline.bet/api"
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        base = f"https://{base}"
    base = base.rstrip("/")
    if not base.endswith("/api"):
        base = f"{base}/api"
    return base


def _api_origin() -> str:
    base = _api_base()
    return re.sub(r"/api$", "", base)


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


def _web_headers(*, cookie: str, csrf_token: str = "", referer: str = "") -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": _public_base(),
        "Referer": referer or f"{_public_base()}/",
    }
    headers.update(_headers())
    if cookie:
        headers["Cookie"] = cookie
    if csrf_token:
        headers["X-XSRF-TOKEN"] = csrf_token
    return headers


def _truthy(value: object) -> bool:
    token = _normalize_text(value).lower()
    return token in {"1", "true", "yes", "y", "on"}


def _safe_cookie_name(name: str) -> str:
    normalized = _normalize_text(name)
    if normalized in {"XSRF-TOKEN", "apiato", "laravel_session"}:
        return normalized
    if len(normalized) >= 32 and re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
        return "session_cookie"
    return "other_cookie"


def _cookie_names_from_header(cookie_header: str) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for part in _normalize_text(cookie_header).split(";"):
        if "=" not in part:
            continue
        name = _safe_cookie_name(part.split("=", 1)[0].strip())
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _cookie_header_from_mapping(cookies: object) -> str:
    if not isinstance(cookies, dict):
        try:
            cookies = dict(cookies)
        except Exception:
            return ""
    parts: List[str] = []
    for name, value in cookies.items():
        name_text = _normalize_text(name)
        value_text = _normalize_text(value)
        if name_text and value_text:
            parts.append(f"{name_text}={value_text}")
    return "; ".join(parts)


def _csrf_token_from_cookie_header(cookie_header: str) -> str:
    for part in _normalize_text(cookie_header).split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        if name.strip().upper() == "XSRF-TOKEN":
            return urllib.parse.unquote(value.strip())
    return ""


def _bootstrap_artline_csrf(client: httpx.Client) -> dict:
    try:
        response = client.get(
            f"{_api_origin()}/sanctum/csrf-cookie",
            headers=_web_headers(cookie="", referer=f"{_public_base()}/"),
        )
    except httpx.HTTPError as exc:
        return {"cookie_header": "", "source": "csrf_bootstrap", "error": str(exc)}
    cookie_header = _cookie_header_from_mapping(getattr(client, "cookies", {}))
    return {
        "cookie_header": cookie_header,
        "source": "csrf_bootstrap",
        "cookie_names": _cookie_names_from_header(cookie_header),
        "http_status": response.status_code,
    }


def _chrome_epoch_now() -> int:
    return int((time.time() + 11644473600) * 1_000_000)


def _browser_cookie_roots() -> List[tuple[str, Path]]:
    custom = _normalize_text(os.getenv("ARTLINE_BROWSER_PROFILE_DIRS", ""))
    if custom:
        rows: List[tuple[str, Path]] = []
        for idx, raw_path in enumerate(re.split(r"[;\n]", custom), start=1):
            path = Path(raw_path.strip().strip('"'))
            if path:
                rows.append((f"custom{idx}", path))
        return rows

    local = _normalize_text(os.getenv("LOCALAPPDATA", ""))
    if not local:
        return []
    base = Path(local)
    return [
        ("chrome", base / "Google" / "Chrome" / "User Data"),
        ("edge", base / "Microsoft" / "Edge" / "User Data"),
        ("brave", base / "BraveSoftware" / "Brave-Browser" / "User Data"),
    ]


def _candidate_chromium_cookie_dbs() -> List[tuple[str, Path, Path]]:
    candidates: List[tuple[str, Path, Path]] = []
    for browser, root in _browser_cookie_roots():
        if not root.exists():
            continue
        profile_dirs = [root] if (root / "Network" / "Cookies").exists() else []
        profile_dirs.extend(
            path
            for path in root.iterdir()
            if path.is_dir() and ((path / "Network" / "Cookies").exists() or (path / "Cookies").exists())
        )
        seen: set[Path] = set()
        for profile_dir in profile_dirs:
            for cookie_db in (profile_dir / "Network" / "Cookies", profile_dir / "Cookies"):
                if not cookie_db.exists() or cookie_db in seen:
                    continue
                seen.add(cookie_db)
                label = f"{browser}:{profile_dir.name}"
                local_state = root / "Local State"
                candidates.append((label, cookie_db, local_state))
    return candidates


def _chrome_debug_ports() -> List[int]:
    ports: List[int] = []
    custom = _normalize_text(os.getenv("ARTLINE_CHROME_DEBUG_PORTS", ""))
    if custom:
        for token in re.split(r"[,;\s]+", custom):
            if not token:
                continue
            try:
                ports.append(int(token))
            except ValueError:
                continue
        return ports

    for _, root in _browser_cookie_roots():
        active_port = root / "DevToolsActivePort"
        if not active_port.exists():
            continue
        try:
            first_line = active_port.read_text(encoding="utf-8").splitlines()[0].strip()
            ports.append(int(first_line))
        except (OSError, IndexError, ValueError):
            continue
    return ports


def _resolve_artline_cdp_cookie_header() -> dict:
    errors: List[str] = []
    for port in _chrome_debug_ports():
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version",
                timeout=2,
            ) as response:
                version_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            errors.append(f"chrome-cdp:{port}:HTTPError:{exc.code}")
            continue
        except Exception as exc:
            errors.append(f"chrome-cdp:{port}:{type(exc).__name__}")
            continue
        websocket_url = _normalize_text(version_payload.get("webSocketDebuggerUrl"))
        if not websocket_url:
            errors.append(f"chrome-cdp:{port}:missing_websocket_url")
            continue
        try:
            from websockets.sync.client import connect

            with connect(websocket_url, open_timeout=2, close_timeout=1) as websocket:
                request = {
                    "id": 1,
                    "method": "Network.getCookies",
                    "params": {"urls": [_public_base(), "https://api.artline.bet"]},
                }
                websocket.send(json.dumps(request, separators=(",", ":")))
                while True:
                    payload = json.loads(websocket.recv(timeout=2))
                    if payload.get("id") != 1:
                        continue
                    if payload.get("error"):
                        errors.append(f"chrome-cdp:{port}:protocol_error")
                        break
                    cookies = payload.get("result", {}).get("cookies") or []
                    parts: List[str] = []
                    names: List[str] = []
                    seen: set[str] = set()
                    for cookie in cookies:
                        if not isinstance(cookie, dict):
                            continue
                        domain = _normalize_text(cookie.get("domain")).lstrip(".").lower()
                        if not domain.endswith("artline.bet"):
                            continue
                        name = _normalize_text(cookie.get("name"))
                        value = _normalize_text(cookie.get("value"))
                        if not (name and value):
                            continue
                        parts.append(f"{name}={value}")
                        safe_name = _safe_cookie_name(name)
                        if safe_name not in seen:
                            names.append(safe_name)
                            seen.add(safe_name)
                    if parts:
                        return {
                            "cookie_header": "; ".join(parts),
                            "source": f"chrome-cdp:{port}",
                            "cookie_names": names,
                            "errors": errors,
                        }
                    errors.append(f"chrome-cdp:{port}:no_artline_cookies")
                    break
        except Exception as exc:
            errors.append(f"chrome-cdp:{port}:{type(exc).__name__}")
            continue
    return {"cookie_header": "", "source": None, "cookie_names": [], "errors": errors}


def _dpapi_decrypt(ciphertext: bytes) -> Optional[bytes]:
    if not ciphertext:
        return None
    try:
        import win32crypt  # type: ignore

        value = win32crypt.CryptUnprotectData(ciphertext, None, None, None, 0)[1]
        return bytes(value) if value is not None else None
    except Exception:
        return None


def _chromium_master_key(local_state_path: Path) -> Optional[bytes]:
    try:
        local_state = json.loads(local_state_path.read_text(encoding="utf-8"))
        encrypted_key = local_state.get("os_crypt", {}).get("encrypted_key")
    except Exception:
        return None
    if not encrypted_key:
        return None
    try:
        raw = base64.b64decode(encrypted_key)
    except Exception:
        return None
    if raw.startswith(b"DPAPI"):
        raw = raw[5:]
    return _dpapi_decrypt(raw)


def _decrypt_chromium_cookie(encrypted_value: bytes, master_key: Optional[bytes]) -> str:
    if not encrypted_value:
        return ""
    if encrypted_value.startswith((b"v10", b"v11")):
        if not master_key:
            return ""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = encrypted_value[3:15]
            ciphertext = encrypted_value[15:]
            return AESGCM(master_key).decrypt(nonce, ciphertext, None).decode("utf-8")
        except Exception:
            return ""
    decrypted = _dpapi_decrypt(encrypted_value)
    if not decrypted:
        return ""
    try:
        return decrypted.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def _read_artline_cookies_from_db(
    *,
    cookie_db: Path,
    local_state: Path,
) -> tuple[List[tuple[str, str]], List[str]]:
    rows: List[tuple[str, str]] = []
    errors: List[str] = []
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as handle:
            tmp_path = Path(handle.name)
        try:
            shutil.copy2(cookie_db, tmp_path)
        except OSError as exc:
            reason = "cookie_db_locked" if getattr(exc, "winerror", None) == 32 else type(exc).__name__
            return [], [reason]

        master_key = _chromium_master_key(local_state)
        now = _chrome_epoch_now()
        with sqlite3.connect(str(tmp_path)) as connection:
            connection.row_factory = sqlite3.Row
            cookie_rows = connection.execute(
                """
                SELECT host_key, name, value, encrypted_value, expires_utc
                FROM cookies
                WHERE host_key LIKE ?
                ORDER BY host_key, name
                """,
                ("%artline.bet%",),
            ).fetchall()
        for cookie_row in cookie_rows:
            expires_utc = int(cookie_row["expires_utc"] or 0)
            if expires_utc and expires_utc < now:
                continue
            name = _normalize_text(cookie_row["name"])
            if not name:
                continue
            value = _normalize_text(cookie_row["value"])
            if not value:
                value = _decrypt_chromium_cookie(bytes(cookie_row["encrypted_value"] or b""), master_key)
            if value:
                rows.append((name, value))
    except sqlite3.Error as exc:
        errors.append(type(exc).__name__)
    finally:
        if tmp_path:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return rows, errors


def resolve_artline_browser_cookie_header() -> dict:
    errors: List[str] = []
    seen_names: set[str] = set()
    cdp_probe = _resolve_artline_cdp_cookie_header()
    cdp_errors = cdp_probe.get("errors") or []
    errors.extend(str(error) for error in cdp_errors)
    if cdp_probe.get("cookie_header"):
        return cdp_probe
    for label, cookie_db, local_state in _candidate_chromium_cookie_dbs():
        rows, row_errors = _read_artline_cookies_from_db(cookie_db=cookie_db, local_state=local_state)
        errors.extend(f"{label}:{error}" for error in row_errors)
        if not rows:
            continue
        cookie_parts: List[str] = []
        names: List[str] = []
        for name, value in rows:
            safe_name = _safe_cookie_name(name)
            if safe_name not in seen_names:
                names.append(safe_name)
                seen_names.add(safe_name)
            cookie_parts.append(f"{name}={value}")
        if cookie_parts:
            return {
                "cookie_header": "; ".join(cookie_parts),
                "source": label,
                "cookie_names": names,
                "errors": errors,
            }
    if not errors:
        errors.append("no_artline_cookies")
    return {"cookie_header": "", "source": None, "cookie_names": [], "errors": errors}


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


def _execution_diagnostics(game: object) -> dict:
    if not isinstance(game, dict):
        return {}
    max_bet = _safe_float(game.get("max_bet"))
    if max_bet is None:
        return {}
    min_bet = _artline_min_bet()
    diagnostics = {
        "artline_max_bet": round(float(max_bet), 6),
        "artline_min_bet": round(float(min_bet), 6),
        "executable": bool(max_bet >= min_bet),
    }
    if max_bet < min_bet:
        diagnostics["reason"] = "max_bet_below_min_bet"
    return diagnostics


def _decorate_markets_for_execution(markets: List[dict], diagnostics: dict, observed_at: float) -> None:
    max_bet = _safe_float(diagnostics.get("artline_max_bet")) if isinstance(diagnostics, dict) else None
    executable = bool(isinstance(diagnostics, dict) and diagnostics.get("executable") and max_bet is not None)
    for market in markets:
        if not isinstance(market, dict):
            continue
        outcomes = market.get("outcomes") if isinstance(market.get("outcomes"), list) else []
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            outcome.setdefault("quote_source", "rest_snapshot")
            outcome.setdefault("observed_at", observed_at)
            if executable:
                outcome.setdefault("max_stake", round(float(max_bet), 6))


def _with_betslip_fields(
    row: dict,
    *,
    selection_id: object = None,
    provider_event_name: object = None,
) -> dict:
    selection_text = _normalize_text(selection_id)
    event_name_text = _normalize_text(provider_event_name)
    if selection_text:
        row["selection_id"] = selection_text
    if selection_text and event_name_text:
        row["provider_event_name"] = event_name_text
    return row


def _outcome_row(
    name: str,
    price: float,
    *,
    selection_id: object = None,
    provider_event_name: object = None,
) -> dict:
    return _with_betslip_fields(
        {"name": name, "price": round(float(price), 6)},
        selection_id=selection_id,
        provider_event_name=provider_event_name,
    )


def _spread_outcome_row(
    name: str,
    price: float,
    point: float,
    *,
    selection_id: object = None,
    provider_event_name: object = None,
) -> dict:
    return _with_betslip_fields(
        {
            "name": name,
            "price": round(float(price), 6),
            "point": round(float(point), 6),
        },
        selection_id=selection_id,
        provider_event_name=provider_event_name,
    )


def _total_outcome_row(
    name: str,
    price: float,
    point: float,
    *,
    selection_id: object = None,
    provider_event_name: object = None,
) -> dict:
    return _with_betslip_fields(
        {
            "name": name,
            "price": round(float(price), 6),
            "point": round(float(point), 6),
        },
        selection_id=selection_id,
        provider_event_name=provider_event_name,
    )


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


def _sport_uses_detail_h2h(sport_key: str) -> bool:
    return _normalize_text(sport_key).lower() in {"icehockey_nhl", "icehockey_ahl"}


def _market_needs_detail_fetch(requested_markets: set[str], sport_key: str) -> bool:
    if "team_totals" in requested_markets:
        return True
    if "h2h" in requested_markets and _sport_uses_detail_h2h(sport_key):
        return True
    return False


def _h2h_selection_ids_need_detail(events: object) -> bool:
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict):
            continue
        if int(event.get("status", 0) or 0) != 1:
            continue
        event_name = _normalize_text(event.get("event_name_value"))
        if event_name in {"0_ml_1", "0_ml_2", "1_ml_1", "1_ml_2"} and not _selection_id(event):
            return True
    return False


def _needs_detail_fetch_for_games(
    games: Sequence[object],
    *,
    requested_markets: set[str],
    sport_key: str,
) -> bool:
    if _market_needs_detail_fetch(requested_markets, sport_key):
        return True
    if "h2h" not in requested_markets:
        return False
    return any(
        _h2h_selection_ids_need_detail(game.get("events") if isinstance(game, dict) else None)
        for game in games
    )


def _normalize_game_markets(
    events: object,
    *,
    home_team: str,
    away_team: str,
    sport_key: str = "",
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
        betslip_selection_id = _selection_id(event)
        sport_token = _normalize_text(sport_key).lower()
        is_hockey_detail_moneyline = _sport_uses_detail_h2h(sport_token) and event_name == "1_ml_1"
        is_hockey_detail_moneyline_away = _sport_uses_detail_h2h(sport_token) and event_name == "1_ml_2"
        if event_name == "0_ml_1" or is_hockey_detail_moneyline:
            _store_best_outcome(
                two_way,
                "home",
                _outcome_row(
                    home_team,
                    price,
                    selection_id=betslip_selection_id,
                    provider_event_name=event_name,
                ),
            )
        elif event_name == "0_ml_2" or is_hockey_detail_moneyline_away:
            _store_best_outcome(
                two_way,
                "away",
                _outcome_row(
                    away_team,
                    price,
                    selection_id=betslip_selection_id,
                    provider_event_name=event_name,
                ),
            )
        elif event_name == "0_win_0":
            _store_best_outcome(
                three_way,
                "draw",
                _outcome_row(
                    "Draw",
                    price,
                    selection_id=betslip_selection_id,
                    provider_event_name=event_name,
                ),
            )
        elif event_name == "0_win_1":
            _store_best_outcome(
                three_way,
                "home",
                _outcome_row(
                    home_team,
                    price,
                    selection_id=betslip_selection_id,
                    provider_event_name=event_name,
                ),
            )
        elif event_name == "0_win_2":
            _store_best_outcome(
                three_way,
                "away",
                _outcome_row(
                    away_team,
                    price,
                    selection_id=betslip_selection_id,
                    provider_event_name=event_name,
                ),
            )
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
                        _total_outcome_row(
                            outcome_name,
                            price,
                            point_value,
                            selection_id=betslip_selection_id,
                            provider_event_name=event_name,
                        ),
                    )
                    continue
                if scope_token in {"1", "2"}:
                    side_key = "home" if scope_token == "1" else "away"
                    team_name = home_team if side_key == "home" else away_team
                    market_bucket = team_totals_by_sig[side_key].setdefault(
                        round(float(point_value), 6),
                        {},
                    )
                    row = _total_outcome_row(
                        outcome_name,
                        price,
                        point_value,
                        selection_id=betslip_selection_id,
                        provider_event_name=event_name,
                    )
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
                    candidate = _spread_outcome_row(
                        home_team,
                        price,
                        point_value,
                        selection_id=betslip_selection_id,
                        provider_event_name=event_name,
                    )
                else:
                    bucket_key = "away_positive" if is_positive else "away_negative"
                    candidate = _spread_outcome_row(
                        away_team,
                        price,
                        point_value,
                        selection_id=betslip_selection_id,
                        provider_event_name=event_name,
                    )
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


def build_web_max_bet_payload(
    *,
    sport: object,
    game_id: object,
    selection_id: object,
    is_live: object = False,
) -> dict:
    sport_text = _normalize_text(sport)
    game_text = _normalize_text(game_id)
    selection_text = _normalize_text(selection_id)
    event = {
        "is_live": bool(_bool_or_none(is_live)),
        "sport": sport_text,
        "game_id": game_text,
        "event_id": selection_text,
        "sum": 0,
    }
    return {
        "currency_type": ARTLINE_CURRENCY_TYPE,
        "type": "solo",
        "events": json.dumps([event], separators=(",", ":")),
        "sum": 0,
    }


def _extract_web_max_bet(payload: object) -> Optional[float]:
    if isinstance(payload, dict):
        direct = _safe_float(payload.get("max_bet"))
        if direct is not None:
            return direct
        data = payload.get("data")
        if isinstance(data, dict):
            nested = _safe_float(data.get("max_bet"))
            if nested is not None:
                return nested
    return None


def preflight_web_max_bet(
    *,
    sport: object,
    game_id: object,
    selection_id: object,
    is_live: object = False,
    stake: object = None,
    event_url: object = "",
    cookie: Optional[str] = None,
    csrf_token: Optional[str] = None,
) -> dict:
    sport_text = _normalize_text(sport)
    game_text = _normalize_text(game_id)
    selection_text = _normalize_text(selection_id)
    requested_stake = _safe_float(stake)
    if not (sport_text and game_text and selection_text):
        return {
            "status": "not_configured",
            "reason": "missing_betslip_identifiers",
            "sport": sport_text or None,
            "game_id": game_text or None,
            "selection_id": selection_text or None,
            "requested_stake": requested_stake,
        }
    resolved_cookie = _normalize_text(cookie if cookie is not None else os.getenv("ARTLINE_COOKIE", ""))
    cookie_source = "explicit" if cookie is not None else ("env" if resolved_cookie else None)
    browser_cookie_probe: Optional[dict] = None
    if not resolved_cookie and cookie is None and _truthy(os.getenv("ARTLINE_AUTO_BROWSER_COOKIES", "")):
        browser_cookie_probe = resolve_artline_browser_cookie_header()
        resolved_cookie = _normalize_text(browser_cookie_probe.get("cookie_header"))
        cookie_source = _normalize_text(browser_cookie_probe.get("source")) or "browser"
    resolved_csrf = _normalize_text(
        csrf_token
        if csrf_token is not None
        else (os.getenv("ARTLINE_CSRF_TOKEN", "") or os.getenv("ARTLINE_XSRF_TOKEN", ""))
    )
    if resolved_cookie and not resolved_csrf:
        resolved_csrf = _csrf_token_from_cookie_header(resolved_cookie)
    result = {
        "status": "auth_required",
        "reason": "missing_artline_cookie",
        "sport": sport_text,
        "game_id": game_text,
        "selection_id": selection_text,
        "requested_stake": requested_stake,
        "cookie_source": cookie_source,
    }
    if browser_cookie_probe is not None:
        result["cookie_names"] = browser_cookie_probe.get("cookie_names") or []
        result["browser_cookie_errors"] = browser_cookie_probe.get("errors") or []
    payload = build_web_max_bet_payload(
        sport=sport_text,
        game_id=game_text,
        selection_id=selection_text,
        is_live=is_live,
    )
    referer = _normalize_text(event_url) or _event_url(
        {
            "event_id": game_text,
            "sport": sport_text,
            "is_live": bool(_bool_or_none(is_live)),
        }
    )
    timeout = _int_or_default(ARTLINE_TIMEOUT_RAW, 20, min_value=1)
    url = f"{_api_base()}/bets/max-bet"
    bootstrap_probe: Optional[dict] = None
    try:
        with httpx.Client(timeout=float(timeout)) as client:
            if not resolved_cookie:
                bootstrap_probe = _bootstrap_artline_csrf(client)
                resolved_cookie = _normalize_text(bootstrap_probe.get("cookie_header"))
                resolved_csrf = _csrf_token_from_cookie_header(resolved_cookie)
                cookie_source = "csrf_bootstrap"
                result["cookie_source"] = cookie_source
                result["cookie_names"] = bootstrap_probe.get("cookie_names") or []
                result["bootstrap_http_status"] = bootstrap_probe.get("http_status")
                if not resolved_cookie:
                    return result
            response = client.post(
                url,
                json=payload,
                headers=_web_headers(cookie=resolved_cookie, csrf_token=resolved_csrf, referer=referer),
            )
    except httpx.HTTPError as exc:
        return {
            **result,
            "status": "error",
            "reason": "network_error",
            "error": str(exc),
        }

    status_code = response.status_code
    try:
        response_payload: object = response.json()
    except ValueError:
        response_payload = None

    base = {
        "sport": sport_text,
        "game_id": game_text,
        "selection_id": selection_text,
        "requested_stake": requested_stake,
        "http_status": status_code,
        "cookie_source": cookie_source,
        "cookie_names": _cookie_names_from_header(resolved_cookie),
    }
    if browser_cookie_probe is not None:
        base["browser_cookie_errors"] = browser_cookie_probe.get("errors") or []
    if bootstrap_probe is not None:
        base["bootstrap_http_status"] = bootstrap_probe.get("http_status")
    if status_code in {401, 403, 419}:
        reason = "csrf_required" if status_code == 419 else "auth_required"
        return {**base, "status": "auth_required", "reason": reason}
    if status_code >= 400:
        return {**base, "status": "error", "reason": "http_error"}

    max_bet = _extract_web_max_bet(response_payload)
    if max_bet is None:
        return {**base, "status": "error", "reason": "missing_max_bet"}
    min_bet = _artline_min_bet()
    executable = bool(max_bet >= min_bet and (requested_stake is None or max_bet + 1e-9 >= requested_stake))
    return {
        **base,
        "status": "verified" if executable else "limited",
        "reason": None if executable else "max_bet_below_requested_stake",
        "max_bet": round(float(max_bet), 6),
        "min_bet": round(float(min_bet), 6),
        "executable": executable,
    }


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
        "max_bet_below_min_bet_count": 0,
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
    }
    if sport_filter.get("tournament_id"):
        payload["tournament_id"] = sport_filter["tournament_id"]

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
    observed_at = dt.datetime.now(dt.timezone.utc).timestamp()

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

    if stats['live_feed_empty']:
        try:
            live_all_payload, _ = await _request_json_async(
                client,
                'POST',
                'lines',
                json_payload={'games_type': 'live'},
                retries=retries,
                backoff_seconds=backoff,
            )
        except ProviderError:
            live_all_payload = None
        live_all_data = _payload_data(live_all_payload)
        if isinstance(live_all_data, dict):
            live_all_sports_available: Dict[str, int] = {}
            for live_sport_key, live_sport_block in live_all_data.items():
                if not isinstance(live_sport_block, dict):
                    continue
                live_sport_games = live_sport_block.get('games')
                if not isinstance(live_sport_games, list):
                    continue
                live_game_count = len(live_sport_games)
                if live_game_count <= 0:
                    continue
                normalized_live_sport = _normalize_text(live_sport_key)
                if not normalized_live_sport:
                    continue
                live_all_sports_available[normalized_live_sport] = live_game_count
            stats['live_all_sports_available'] = live_all_sports_available
            stats['live_all_total_games'] = sum(live_all_sports_available.values())

    detailed_events_by_id: Dict[str, List[dict]] = {}
    if games and _needs_detail_fetch_for_games(
        games,
        requested_markets=requested_markets,
        sport_key=sport_token,
    ):
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
            sport_key=sport_token,
            requested_markets=requested_markets,
        )
        if not normalized_markets:
            continue

        live_state = _game_live_state_payload(game, games_type)
        execution_diagnostics = _execution_diagnostics(game)
        if execution_diagnostics.get("reason") == "max_bet_below_min_bet":
            stats["max_bet_below_min_bet_count"] += 1
        _decorate_markets_for_execution(normalized_markets, execution_diagnostics, observed_at)
        bookmaker = {
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
            "quote_source": "rest_snapshot",
            "observed_at": observed_at,
            "markets": normalized_markets,
        }
        if execution_diagnostics:
            bookmaker["execution_diagnostics"] = execution_diagnostics
        normalized_event = {
            "id": event_id,
            "sport_key": sport_token,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "live_state": live_state,
            "bookmakers": [bookmaker],
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

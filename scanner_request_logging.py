"""Request logging helpers for scanner network calls."""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import inspect
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

SCAN_REQUEST_LOG_ENABLED = os.getenv("SCAN_REQUEST_LOG_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SCAN_REQUEST_LOG_DIR = os.getenv(
    "SCAN_REQUEST_LOG_DIR",
    str(Path("data") / "request_logs"),
).strip()
SCAN_REQUEST_LOG_MAX_BODY_CHARS_RAW = os.getenv("SCAN_REQUEST_LOG_MAX_BODY_CHARS", "2000").strip()
SCAN_REQUEST_LOG_RETENTION_FILES_RAW = os.getenv(
    "SCAN_REQUEST_LOG_RETENTION_FILES",
    "20",
).strip()
try:
    SCAN_REQUEST_LOG_MAX_BODY_CHARS = max(0, int(float(SCAN_REQUEST_LOG_MAX_BODY_CHARS_RAW)))
except (TypeError, ValueError):
    SCAN_REQUEST_LOG_MAX_BODY_CHARS = 2000
try:
    SCAN_REQUEST_LOG_RETENTION_FILES = max(
        0,
        int(float(SCAN_REQUEST_LOG_RETENTION_FILES_RAW)),
    )
except (TypeError, ValueError):
    SCAN_REQUEST_LOG_RETENTION_FILES = 20

_REQUEST_LOG_SENSITIVE_KEYS = {
    "apikey",
    "api_key",
    "authorization",
    "token",
    "secret",
    "password",
    "cookie",
    "xapikey",
    "x_api_key",
    "session",
}
_REQUEST_TRACE_LOCK = threading.RLock()
_REQUEST_TRACE_ACTIVE: List["_ScanRequestLogger"] = []
_REQUEST_TRACE_PATCHED = False
_REQUEST_TRACE_CONTEXT = contextvars.ContextVar("scan_request_logger", default=None)
_REQUESTS_SESSION_REQUEST_ORIGINAL = requests.sessions.Session.request


def _iso_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _cleanup_old_request_logs(target_dir: Path, keep_path: Optional[Path] = None) -> None:
    if SCAN_REQUEST_LOG_RETENTION_FILES <= 0:
        return
    try:
        log_paths = sorted(
            target_dir.glob("requests_*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return

    kept = 0
    for path in log_paths:
        if keep_path is not None and path == keep_path:
            kept += 1
            continue
        kept += 1
        if kept <= SCAN_REQUEST_LOG_RETENTION_FILES:
            continue
        try:
            path.unlink()
        except OSError:
            continue


def _is_sensitive_log_key(key: object) -> bool:
    token = re.sub(r"[^a-z0-9]+", "", str(key or "").strip().lower())
    if not token:
        return False
    if token in _REQUEST_LOG_SENSITIVE_KEYS:
        return True
    return any(part in token for part in ("apikey", "authorization", "token", "secret", "password"))


def _sanitize_for_request_log(value: object, key_hint: Optional[str] = None) -> object:
    if key_hint and _is_sensitive_log_key(key_hint):
        return "***redacted***"
    if isinstance(value, dict):
        return {str(key): _sanitize_for_request_log(item, key_hint=str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_request_log(item, key_hint=key_hint) for item in value]
    if isinstance(value, bytes):
        return _truncate_request_log_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        return _truncate_request_log_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate_request_log_text(str(value))


def _truncate_request_log_text(text: str, limit: Optional[int] = None) -> str:
    capped = SCAN_REQUEST_LOG_MAX_BODY_CHARS if limit is None else max(0, int(limit))
    value = str(text or "")
    if capped <= 0:
        return ""
    if len(value) <= capped:
        return value
    return f"{value[:capped]}...<truncated {len(value) - capped} chars>"


def _sanitize_request_log_url(url: object) -> str:
    raw_url = str(url or "")
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return raw_url
    if not parsed.query:
        return raw_url
    redacted_params = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if _is_sensitive_log_key(key):
            redacted_params.append((key, "***redacted***"))
        else:
            redacted_params.append((key, value))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(redacted_params, doseq=True),
            parsed.fragment,
        )
    )


def _response_preview_for_request_log(response: requests.Response) -> dict:
    content_type = str(response.headers.get("Content-Type", ""))
    body_preview = ""
    size_bytes: Optional[int] = None
    if SCAN_REQUEST_LOG_MAX_BODY_CHARS > 0:
        try:
            body_bytes = response.content or b""
        except Exception:
            body_bytes = b""
        size_bytes = len(body_bytes)
        lower_ct = content_type.lower()
        if "json" in lower_ct:
            try:
                payload = response.json()
                encoded = json.dumps(
                    _sanitize_for_request_log(payload),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                body_preview = _truncate_request_log_text(encoded)
            except ValueError:
                body_preview = _truncate_request_log_text(response.text)
        elif lower_ct.startswith("text/") or "xml" in lower_ct or "javascript" in lower_ct:
            body_preview = _truncate_request_log_text(response.text)
        else:
            body_preview = f"<binary:{size_bytes or 0} bytes>"
    return {
        "status_code": response.status_code,
        "ok": bool(response.ok),
        "headers": _sanitize_for_request_log(dict(response.headers)),
        "content_type": content_type,
        "size_bytes": size_bytes,
        "body_preview": body_preview,
    }


def _build_request_log_entry(
    method: object,
    url: object,
    kwargs: dict,
    elapsed_ms: float,
    response: Optional[requests.Response] = None,
    error: Optional[str] = None,
) -> dict:
    headers = kwargs.get("headers")
    request_payload = {
        "method": str(method or "").upper() or "GET",
        "url": _sanitize_request_log_url(url),
        "params": _sanitize_for_request_log(kwargs.get("params")),
        "headers": _sanitize_for_request_log(dict(headers)) if isinstance(headers, dict) else _sanitize_for_request_log(headers),
        "json": _sanitize_for_request_log(kwargs.get("json")),
        "data": _sanitize_for_request_log(kwargs.get("data")),
        "timeout": _sanitize_for_request_log(kwargs.get("timeout")),
    }
    entry = {
        "type": "request",
        "time": _iso_now(),
        "elapsed_ms": round(float(elapsed_ms or 0.0), 2),
        "request": request_payload,
    }
    if response is not None:
        entry["response"] = _response_preview_for_request_log(response)
    if error:
        entry["error"] = _truncate_request_log_text(str(error), limit=max(512, SCAN_REQUEST_LOG_MAX_BODY_CHARS))
    return entry


class _ScanRequestLogger:
    def __init__(self, scan_time: str) -> None:
        self.scan_time = scan_time
        self.path = ""
        self.error = ""
        self.requests_logged = 0
        self.owner_thread_id = threading.get_ident()
        self.enabled = SCAN_REQUEST_LOG_ENABLED and bool(SCAN_REQUEST_LOG_DIR)
        self._lock = threading.Lock()
        self._handle = None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            target_dir = Path(SCAN_REQUEST_LOG_DIR)
            target_dir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = target_dir / f"requests_{stamp}_{uuid.uuid4().hex[:6]}.jsonl"
            self._handle = path.open("w", encoding="utf-8")
            self.path = str(path)
            self._write(
                {
                    "type": "meta",
                    "scan_time": self.scan_time,
                    "created_at": _iso_now(),
                    "max_body_chars": SCAN_REQUEST_LOG_MAX_BODY_CHARS,
                }
            )
            _cleanup_old_request_logs(target_dir, keep_path=path)
        except OSError as exc:
            self.enabled = False
            self.error = f"Failed to create request log file: {exc}"
            self._handle = None
            self.path = ""

    def _write(self, payload: dict) -> None:
        if not self._handle:
            return
        with self._lock:
            self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._handle.flush()

    def log_request(self, payload: dict) -> None:
        if not self._handle:
            return
        with self._lock:
            self.requests_logged += 1
            enriched = dict(payload)
            enriched["seq"] = self.requests_logged
            self._handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            self._handle.flush()

    def log_meta(self, payload: dict) -> None:
        if not self._handle:
            return
        self._write(payload)

    def close(self) -> None:
        if not self._handle:
            return
        with self._lock:
            try:
                self._handle.write(
                    json.dumps(
                        {
                            "type": "summary",
                            "closed_at": _iso_now(),
                            "requests_logged": self.requests_logged,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self._handle.flush()
            finally:
                try:
                    self._handle.close()
                finally:
                    self._handle = None


def _active_request_loggers() -> List["_ScanRequestLogger"]:
    with _REQUEST_TRACE_LOCK:
        return list(_REQUEST_TRACE_ACTIVE)


def _set_current_request_logger(logger: Optional["_ScanRequestLogger"]) -> None:
    _REQUEST_TRACE_CONTEXT.set(logger)


def _current_request_logger() -> Optional["_ScanRequestLogger"]:
    logger = _REQUEST_TRACE_CONTEXT.get()
    if isinstance(logger, _ScanRequestLogger):
        return logger
    return None


def _select_request_logger(active: Sequence["_ScanRequestLogger"]) -> Optional["_ScanRequestLogger"]:
    current = _current_request_logger()
    if current is not None:
        for logger in active:
            if logger is current:
                return logger
        return None
    if len(active) == 1:
        return active[0]
    return None


def _instrumented_session_request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
    started_at = time.perf_counter()
    response = None
    error = None
    try:
        response = _REQUESTS_SESSION_REQUEST_ORIGINAL(self, method, url, **kwargs)
        return response
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        active = _active_request_loggers()
        logger = _select_request_logger(active)
        if logger is not None:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            try:
                entry = _build_request_log_entry(
                    method=method,
                    url=url,
                    kwargs=kwargs,
                    elapsed_ms=elapsed_ms,
                    response=response,
                    error=error,
                )
                logger.log_request(entry)
            except Exception:
                pass


def _ensure_request_logging_patch() -> None:
    global _REQUEST_TRACE_PATCHED
    with _REQUEST_TRACE_LOCK:
        if _REQUEST_TRACE_PATCHED:
            return
        requests.sessions.Session.request = _instrumented_session_request
        _REQUEST_TRACE_PATCHED = True


def _activate_request_logger(logger: _ScanRequestLogger) -> None:
    _set_current_request_logger(logger)
    if not logger.enabled:
        return
    _ensure_request_logging_patch()
    with _REQUEST_TRACE_LOCK:
        for idx in range(len(_REQUEST_TRACE_ACTIVE) - 1, -1, -1):
            stale = _REQUEST_TRACE_ACTIVE[idx]
            if stale is logger:
                _REQUEST_TRACE_ACTIVE.pop(idx)
        _REQUEST_TRACE_ACTIVE.append(logger)


def _deactivate_request_logger(logger: _ScanRequestLogger) -> None:
    with _REQUEST_TRACE_LOCK:
        for idx in range(len(_REQUEST_TRACE_ACTIVE) - 1, -1, -1):
            if _REQUEST_TRACE_ACTIVE[idx] is logger:
                _REQUEST_TRACE_ACTIVE.pop(idx)
                break
    if _current_request_logger() is logger:
        _set_current_request_logger(None)


def _run_with_request_logger(
    logger: Optional["_ScanRequestLogger"],
    func,
    *args,
    **kwargs,
):
    previous = _current_request_logger()
    _set_current_request_logger(logger)
    try:
        return func(*args, **kwargs)
    finally:
        _set_current_request_logger(previous)


def _submit_with_request_logger(executor, func, *args, **kwargs):
    logger = _current_request_logger()
    return executor.submit(_run_with_request_logger, logger, func, *args, **kwargs)


async def _run_async_with_request_logger(
    logger: Optional["_ScanRequestLogger"],
    func,
    *args,
    **kwargs,
):
    previous = _current_request_logger()
    _set_current_request_logger(logger)
    try:
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
    finally:
        _set_current_request_logger(previous)


async def _call_with_request_logger_async(func, *args, **kwargs):
    logger = _current_request_logger()
    if inspect.iscoroutinefunction(func):
        return await _run_async_with_request_logger(logger, func, *args, **kwargs)
    return await asyncio.to_thread(_run_with_request_logger, logger, func, *args, **kwargs)


def _attach_request_log_info(result: dict, logger: _ScanRequestLogger) -> dict:
    if not isinstance(result, dict):
        return result
    if logger.path:
        result["request_log"] = {
            "enabled": True,
            "path": logger.path,
            "requests_logged": logger.requests_logged,
        }
        if logger.error:
            result["request_log"]["error"] = logger.error
    elif SCAN_REQUEST_LOG_ENABLED:
        result["request_log"] = {"enabled": False}
        if logger.error:
            result["request_log"]["error"] = logger.error
    return result

from __future__ import annotations

import atexit
import asyncio
import datetime as dt
import json
import threading
import time
from typing import Callable, Dict, Optional, Sequence, Tuple, Type

import httpx


_SHARED_CLIENTS_LOCK = threading.Lock()
_SHARED_CLIENTS: Dict[Tuple[str, int, float, bool], dict] = {}


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scanner_logging_module():
    try:
        import scanner
    except Exception:
        return None
    required = (
        "_active_request_loggers",
        "_select_request_logger",
        "_sanitize_for_request_log",
        "_sanitize_request_log_url",
        "_truncate_request_log_text",
    )
    if not all(hasattr(scanner, name) for name in required):
        return None
    return scanner


async def get_shared_client(
    namespace: str,
    *,
    timeout: float,
    follow_redirects: bool = True,
) -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    key = (
        str(namespace or "default"),
        id(loop),
        float(timeout),
        bool(follow_redirects),
    )
    with _SHARED_CLIENTS_LOCK:
        cached = _SHARED_CLIENTS.get(key)
        client = cached.get("client") if isinstance(cached, dict) else None
        if isinstance(client, httpx.AsyncClient) and not client.is_closed:
            return client
    client = httpx.AsyncClient(timeout=float(timeout), follow_redirects=follow_redirects)
    with _SHARED_CLIENTS_LOCK:
        _SHARED_CLIENTS[key] = {
            "client": client,
            "loop": loop,
        }
    return client


def shutdown_shared_clients(timeout_seconds: float = 2.0) -> None:
    with _SHARED_CLIENTS_LOCK:
        entries = list(_SHARED_CLIENTS.values())
        _SHARED_CLIENTS.clear()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        client = entry.get("client")
        loop = entry.get("loop")
        if not isinstance(client, httpx.AsyncClient) or client.is_closed:
            continue
        try:
            if isinstance(loop, asyncio.AbstractEventLoop) and loop.is_running() and not loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(client.aclose(), loop)
                future.result(timeout=max(0.1, float(timeout_seconds)))
            else:
                asyncio.run(client.aclose())
        except Exception:
            continue


atexit.register(shutdown_shared_clients)


def _httpx_response_preview(scanner_module, response: httpx.Response) -> dict:
    content_type = str(response.headers.get("Content-Type", ""))
    body_preview = ""
    size_bytes: Optional[int] = None
    max_chars = int(getattr(scanner_module, "SCAN_REQUEST_LOG_MAX_BODY_CHARS", 0) or 0)
    if max_chars > 0:
        body_bytes = response.content or b""
        size_bytes = len(body_bytes)
        lower_ct = content_type.lower()
        if "json" in lower_ct:
            try:
                payload = response.json()
                encoded = json.dumps(
                    scanner_module._sanitize_for_request_log(payload),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                body_preview = scanner_module._truncate_request_log_text(encoded)
            except ValueError:
                body_preview = scanner_module._truncate_request_log_text(response.text)
        elif lower_ct.startswith("text/") or "xml" in lower_ct or "javascript" in lower_ct:
            body_preview = scanner_module._truncate_request_log_text(response.text)
        else:
            body_preview = f"<binary:{size_bytes or 0} bytes>"
    return {
        "status_code": response.status_code,
        "ok": response.is_success,
        "headers": scanner_module._sanitize_for_request_log(dict(response.headers)),
        "content_type": content_type,
        "size_bytes": size_bytes,
        "body_preview": body_preview,
    }


def log_httpx_request(
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[Dict[str, str]] = None,
    json_payload: object = None,
    data: object = None,
    timeout: object = None,
    elapsed_ms: float,
    response: Optional[httpx.Response] = None,
    error: Optional[str] = None,
) -> None:
    scanner_module = _scanner_logging_module()
    if scanner_module is None:
        return
    try:
        active = scanner_module._active_request_loggers()
        logger = scanner_module._select_request_logger(active)
        if logger is None:
            return
        entry = {
            "type": "request",
            "time": _iso_now(),
            "elapsed_ms": round(float(elapsed_ms or 0.0), 2),
            "request": {
                "method": str(method or "").upper() or "GET",
                "url": scanner_module._sanitize_request_log_url(url),
                "params": scanner_module._sanitize_for_request_log(params),
                "headers": scanner_module._sanitize_for_request_log(dict(headers or {})),
                "json": scanner_module._sanitize_for_request_log(json_payload),
                "data": scanner_module._sanitize_for_request_log(data),
                "timeout": scanner_module._sanitize_for_request_log(timeout),
            },
        }
        if response is not None:
            entry["response"] = _httpx_response_preview(scanner_module, response)
        if error:
            entry["error"] = scanner_module._truncate_request_log_text(str(error), limit=max(512, int(getattr(scanner_module, "SCAN_REQUEST_LOG_MAX_BODY_CHARS", 0) or 0)))
        logger.log_request(entry)
    except Exception:
        pass


async def request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[Dict[str, str]] = None,
    json_payload: object = None,
    data: object = None,
    timeout: Optional[float] = None,
    retries: int = 0,
    backoff_seconds: float = 0.0,
    retriable_status: Optional[Sequence[int]] = None,
    error_cls: Type[Exception],
    network_error_prefix: str,
    parse_error_message: str,
    status_error_message: Callable[[int], str],
) -> Tuple[object, int]:
    retriable = set(retriable_status or (429, 500, 502, 503, 504))
    attempts = max(0, int(retries)) + 1
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        started_at = time.perf_counter()
        response: Optional[httpx.Response] = None
        error: Optional[str] = None
        retry_after_response = False
        try:
            response = await client.request(
                method,
                url,
                params=params,
                headers=headers,
                json=json_payload,
                data=data,
                timeout=timeout,
            )
            if response.status_code >= 400:
                error = status_error_message(response.status_code)
                if response.status_code in retriable and attempt < attempts - 1:
                    retry_after_response = True
                else:
                    raise error_cls(error, status_code=response.status_code)
            else:
                try:
                    return response.json(), attempt
                except ValueError as exc:
                    error = parse_error_message
                    last_error = error_cls(parse_error_message)
                    if attempt < attempts - 1:
                        await asyncio.sleep(backoff_seconds * (2**attempt))
                        continue
                    raise last_error from exc
        except httpx.RequestError as exc:
            error = f"{network_error_prefix}: {exc}"
            last_error = error_cls(error)
            if attempt < attempts - 1:
                await asyncio.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        finally:
            log_httpx_request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json_payload=json_payload,
                data=data,
                timeout=timeout,
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                response=response,
                error=error,
            )
        if retry_after_response:
            await asyncio.sleep(backoff_seconds * (2**attempt))
            continue
    if last_error is not None:
        raise last_error
    raise error_cls(f"Request failed for {url}")


async def request_text(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[Dict[str, str]] = None,
    json_payload: object = None,
    data: object = None,
    timeout: Optional[float] = None,
    retries: int = 0,
    backoff_seconds: float = 0.0,
    retriable_status: Optional[Sequence[int]] = None,
    error_cls: Type[Exception],
    network_error_prefix: str,
    status_error_message: Callable[[int], str],
) -> Tuple[str, int]:
    retriable = set(retriable_status or (429, 500, 502, 503, 504))
    attempts = max(0, int(retries)) + 1
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        started_at = time.perf_counter()
        response: Optional[httpx.Response] = None
        error: Optional[str] = None
        retry_after_response = False
        try:
            response = await client.request(
                method,
                url,
                params=params,
                headers=headers,
                json=json_payload,
                data=data,
                timeout=timeout,
            )
            if response.status_code >= 400:
                error = status_error_message(response.status_code)
                if response.status_code in retriable and attempt < attempts - 1:
                    retry_after_response = True
                else:
                    raise error_cls(error, status_code=response.status_code)
            else:
                return response.text, attempt
        except httpx.RequestError as exc:
            error = f"{network_error_prefix}: {exc}"
            last_error = error_cls(error)
            if attempt < attempts - 1:
                await asyncio.sleep(backoff_seconds * (2**attempt))
                continue
            raise last_error from exc
        finally:
            log_httpx_request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json_payload=json_payload,
                data=data,
                timeout=timeout,
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                response=response,
                error=error,
            )
        if retry_after_response:
            await asyncio.sleep(backoff_seconds * (2**attempt))
            continue
    if last_error is not None:
        raise last_error
    raise error_cls(f"Request failed for {url}")

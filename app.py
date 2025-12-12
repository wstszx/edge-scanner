from __future__ import annotations

import argparse
import socket
import threading
import webbrowser
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request

from config import (
    DEFAULT_COMMISSION,
    DEFAULT_REGION_KEYS,
    DEFAULT_SPORT_OPTIONS,
    REGION_OPTIONS,
)
from scanner import run_scan

app = Flask(__name__)


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        default_sports=DEFAULT_SPORT_OPTIONS,
        region_options=REGION_OPTIONS,
        default_commission_percent=int(DEFAULT_COMMISSION * 100),
    )


@app.route("/scan", methods=["POST"])
def scan() -> tuple:
    payload = request.get_json(force=True, silent=True) or {}
    api_key = (payload.get("apiKey") or "").strip()
    sports = payload.get("sports") or []
    all_sports = bool(payload.get("allSports"))
    stake = payload.get("stake")
    regions = payload.get("regions")
    commission = payload.get("commission")
    try:
        stake_value = float(stake) if stake is not None else 100.0
    except (TypeError, ValueError):
        stake_value = 100.0
    if isinstance(regions, list):
        regions_value = [str(region) for region in regions if isinstance(region, str)]
    else:
        regions_value = None
    try:
        commission_percent = float(commission) if commission is not None else None
    except (TypeError, ValueError):
        commission_percent = None
    commission_rate = (
        commission_percent / 100.0 if commission_percent is not None else DEFAULT_COMMISSION
    )
    result = run_scan(
        api_key,
        sports,
        all_sports,
        stake_value,
        regions_value or DEFAULT_REGION_KEYS,
        commission_rate,
    )
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

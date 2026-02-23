"""Historical scan opportunity tracking.

Saves arbitrage / middles / +EV opportunities to JSONL files so users can
review past finds without re-running scans.

Usage
-----
    from history import HistoryManager
    hm = HistoryManager()
    hm.save_opportunities(scan_result, scan_time="2024-01-01T12:00:00Z")
    recent = hm.load_recent(limit=100, mode="arbitrage")
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ---------------------------------------------------------------------------
# Configuration (reads same env/settings chain as the rest of the app)
# ---------------------------------------------------------------------------

HISTORY_ENABLED: bool = os.getenv("HISTORY_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
HISTORY_DIR: str = os.getenv(
    "HISTORY_DIR", str(Path("data") / "history")
).strip()
HISTORY_MAX_RECORDS: int = max(
    100,
    int(float(os.getenv("HISTORY_MAX_RECORDS", "10000").strip() or "10000")),
)

_MODE_FILES = {
    "arbitrage": "arbitrage_history.jsonl",
    "middles": "middles_history.jsonl",
    "ev": "ev_history.jsonl",
}


def _extract_mode_items(
    scan_result: dict,
    mode_key: str,
    legacy_key: Optional[str] = None,
) -> List[dict]:
    candidates = []
    if legacy_key:
        candidates.append(scan_result.get(legacy_key))
    candidates.append(scan_result.get(mode_key))

    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
        if isinstance(candidate, dict):
            nested = candidate.get("opportunities")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    return []


class HistoryManager:
    """Thread-safe append-only JSONL history store."""

    def __init__(
        self,
        history_dir: Optional[str] = None,
        max_records: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._dir = Path(history_dir or HISTORY_DIR)
        self._max = max_records if max_records is not None else HISTORY_MAX_RECORDS
        self._enabled = enabled if enabled is not None else HISTORY_ENABLED
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_opportunities(self, scan_result: dict, scan_time: Optional[str] = None) -> int:
        """Persist all opportunities from a ``run_scan`` result dict.

        Returns the total number of records written.
        """
        if not self._enabled:
            return 0
        if not isinstance(scan_result, dict):
            return 0

        ts = scan_time or _utc_now()
        written = 0
        for mode, mode_key, legacy_key in [
            ("arbitrage", "arbitrage", "opportunities"),
            ("middles", "middles", None),
            ("ev", "plus_ev", None),
        ]:
            items = _extract_mode_items(scan_result, mode_key=mode_key, legacy_key=legacy_key)
            if not items:
                continue
            records = [_flatten_record(item, ts, mode) for item in items]
            written += self._append_records(mode, records)
        return written

    def load_recent(
        self,
        limit: int = 200,
        mode: Optional[str] = None,
    ) -> List[dict]:
        """Return the most-recent ``limit`` opportunity records.

        If *mode* is given (``"arbitrage"``, ``"middles"``, or ``"ev"``), only
        records from that mode are returned.  Otherwise all modes are merged
        and sorted by scan_time descending.
        """
        if mode:
            modes = [mode] if mode in _MODE_FILES else []
        else:
            modes = list(_MODE_FILES.keys())

        all_records: List[dict] = []
        for m in modes:
            all_records.extend(self._read_tail(m, limit))

        all_records.sort(key=lambda r: r.get("scan_time", ""), reverse=True)
        return all_records[:limit]

    def get_stats(self) -> dict:
        """Return record counts per mode and disk usage."""
        stats: dict = {"enabled": self._enabled, "dir": str(self._dir), "modes": {}}
        for mode, filename in _MODE_FILES.items():
            path = self._dir / filename
            count = 0
            size_bytes = 0
            if path.exists():
                size_bytes = path.stat().st_size
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                count += 1
                except OSError:
                    pass
            stats["modes"][mode] = {"count": count, "size_bytes": size_bytes}
        return stats

    def clear(self, mode: Optional[str] = None) -> None:
        """Delete history files (all or a specific mode)."""
        modes = [mode] if mode and mode in _MODE_FILES else list(_MODE_FILES.keys())
        for m in modes:
            path = self._dir / _MODE_FILES[m]
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> bool:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False

    def _append_records(self, mode: str, records: List[dict]) -> int:
        if not records:
            return 0
        if not self._ensure_dir():
            return 0
        path = self._dir / _MODE_FILES[mode]
        written = 0
        with self._lock:
            try:
                with path.open("a", encoding="utf-8") as fh:
                    for rec in records:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
            except OSError:
                return written
            # Trim to max_records
            self._trim(path)
        return written

    def _trim(self, path: Path) -> None:
        """Keep only the last ``_max`` lines in the file (in-place truncation)."""
        try:
            lines = path.read_bytes().splitlines(keepends=True)
        except OSError:
            return
        if len(lines) <= self._max:
            return
        trimmed = lines[-self._max:]
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_bytes(b"".join(trimmed))
            tmp.replace(path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _read_tail(self, mode: str, limit: int) -> List[dict]:
        path = self._dir / _MODE_FILES[mode]
        if not path.exists():
            return []
        records: List[dict] = []
        try:
            lines = path.read_bytes().splitlines()
        except OSError:
            return []
        for raw in reversed(lines):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(rec, dict):
                rec.setdefault("mode", mode)
                records.append(rec)
            if len(records) >= limit:
                break
        return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _flatten_record(item: dict, scan_time: str, mode: str) -> dict:
    """Extract a compact, storable representation of an opportunity."""
    base: dict = {
        "scan_time": scan_time,
        "mode": mode,
        "sport": item.get("sport"),
        "sport_display": item.get("sport_display"),
        "event": item.get("event"),
        "commence_time": item.get("commence_time"),
        "market": item.get("market"),
    }
    if mode == "arbitrage":
        base["roi_percent"] = item.get("roi_percent")
        base["books"] = [
            {
                "outcome": o.get("outcome"),
                "bookmaker": o.get("bookmaker"),
                "price": o.get("price"),
            }
            for o in (item.get("best_odds") or [])
        ]
    elif mode == "middles":
        ev = item.get("ev")
        if ev is None:
            ev = item.get("ev_percent")
        base["ev"] = ev
        base["ev_percent"] = item.get("ev_percent")
        gap_points = item.get("gap_points")
        gap = item.get("gap")
        if gap_points is None and isinstance(gap, dict):
            gap_points = gap.get("points")
        base["gap_points"] = gap_points
        probability = item.get("probability")
        if probability is None:
            probability = item.get("middle_probability")
        if probability is None:
            probability_percent = item.get("probability_percent")
            try:
                probability = float(probability_percent) / 100.0
            except (TypeError, ValueError):
                probability = None
        base["probability"] = probability
        side_a = item.get("side_a") if isinstance(item.get("side_a"), dict) else {}
        side_b = item.get("side_b") if isinstance(item.get("side_b"), dict) else {}
        book_a = item.get("book_a") or side_a.get("bookmaker")
        line_a = item.get("line_a") if item.get("line_a") is not None else side_a.get("line")
        book_b = item.get("book_b") or side_b.get("bookmaker")
        line_b = item.get("line_b") if item.get("line_b") is not None else side_b.get("line")
        base["books"] = [
            {"bookmaker": book_a, "line": line_a},
            {"bookmaker": book_b, "line": line_b},
        ]
    elif mode == "ev":
        base["edge_percent"] = item.get("edge_percent") or item.get("net_edge_percent")
        base["ev_per_100"] = item.get("ev_per_100")
        bet = item.get("bet") or {}
        sharp = item.get("sharp") or {}
        base["soft_book"] = bet.get("soft_book")
        base["soft_odds"] = bet.get("soft_odds")
        base["sharp_book"] = sharp.get("book")
        base["fair_odds"] = sharp.get("fair_odds")
    return base


# ---------------------------------------------------------------------------
# Module-level singleton (for use in app.py)
# ---------------------------------------------------------------------------

_default_manager: Optional[HistoryManager] = None


def get_history_manager() -> HistoryManager:
    """Return the module-level singleton HistoryManager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = HistoryManager()
    return _default_manager

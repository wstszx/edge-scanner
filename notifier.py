"""External notification system for arbitrage / +EV alerts.

Supports two channels:
- **Webhook**: HTTP POST with JSON payload (configurable URL + optional HMAC signing)
- **Telegram**: Bot API message to a chat

Configuration (via .env or environment variables)
-------------------------------------------------
    NOTIFY_WEBHOOK_URL=https://your-endpoint/scan-alert
    NOTIFY_WEBHOOK_SECRET=optional_hmac_secret
    NOTIFY_TELEGRAM_TOKEN=123456:ABCdef...
    NOTIFY_TELEGRAM_CHAT_ID=-1001234567890
    NOTIFY_MIN_ROI=2.0        # Only notify for arb ROI >= this percent
    NOTIFY_MIN_EDGE=5.0       # Only notify for +EV edge >= this percent
    NOTIFY_MIN_EV=0.0         # Only notify for middles EV >= this value
    NOTIFY_TIMEOUT_SECONDS=10 # HTTP request timeout
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from datetime import datetime, timezone
from typing import List, Optional

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOTIFY_WEBHOOK_URL: str = os.getenv("NOTIFY_WEBHOOK_URL", "").strip()
NOTIFY_WEBHOOK_SECRET: str = os.getenv("NOTIFY_WEBHOOK_SECRET", "").strip()
NOTIFY_TELEGRAM_TOKEN: str = os.getenv("NOTIFY_TELEGRAM_TOKEN", "").strip()
NOTIFY_TELEGRAM_CHAT_ID: str = os.getenv("NOTIFY_TELEGRAM_CHAT_ID", "").strip()

_min_roi_raw = os.getenv("NOTIFY_MIN_ROI", "0.0").strip()
_min_edge_raw = os.getenv("NOTIFY_MIN_EDGE", "0.0").strip()
_min_ev_raw = os.getenv("NOTIFY_MIN_EV", "0.0").strip()
_timeout_raw = os.getenv("NOTIFY_TIMEOUT_SECONDS", "10").strip()

try:
    NOTIFY_MIN_ROI: float = float(_min_roi_raw)
except ValueError:
    NOTIFY_MIN_ROI = 0.0
try:
    NOTIFY_MIN_EDGE: float = float(_min_edge_raw)
except ValueError:
    NOTIFY_MIN_EDGE = 0.0
try:
    NOTIFY_MIN_EV: float = float(_min_ev_raw)
except ValueError:
    NOTIFY_MIN_EV = 0.0
try:
    NOTIFY_TIMEOUT_SECONDS: int = max(1, int(float(_timeout_raw)))
except ValueError:
    NOTIFY_TIMEOUT_SECONDS = 10


class Notifier:
    """Send opportunity alerts via Webhook and/or Telegram.

    All network calls happen synchronously inside the caller's thread so they
    should be dispatched via a daemon thread in ``app.py`` to avoid blocking
    the scan response.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        min_roi: float = NOTIFY_MIN_ROI,
        min_edge: float = NOTIFY_MIN_EDGE,
        min_ev: float = NOTIFY_MIN_EV,
        timeout: int = NOTIFY_TIMEOUT_SECONDS,
    ) -> None:
        self.webhook_url = (webhook_url or NOTIFY_WEBHOOK_URL).strip()
        self.webhook_secret = (webhook_secret or NOTIFY_WEBHOOK_SECRET).strip()
        self.telegram_token = (telegram_token or NOTIFY_TELEGRAM_TOKEN).strip()
        self.telegram_chat_id = (telegram_chat_id or NOTIFY_TELEGRAM_CHAT_ID).strip()
        self.min_roi = min_roi
        self.min_edge = min_edge
        self.min_ev = min_ev
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Return True if at least one notification channel is enabled."""
        return bool(self.webhook_url) or bool(
            self.telegram_token and self.telegram_chat_id
        )

    @staticmethod
    def _as_float(value: object) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _middle_ev_value(cls, item: dict) -> float:
        for key in ("ev", "ev_percent", "ev_dollars"):
            value = cls._as_float(item.get(key))
            if value is not None:
                return value
        return 0.0

    @classmethod
    def _middle_gap_points(cls, item: dict) -> Optional[float]:
        gap_points = cls._as_float(item.get("gap_points"))
        if gap_points is not None:
            return gap_points
        gap = item.get("gap")
        if isinstance(gap, dict):
            return cls._as_float(gap.get("points"))
        return None

    @classmethod
    def _middle_probability(cls, item: dict) -> Optional[float]:
        for key in ("probability", "middle_probability"):
            value = cls._as_float(item.get(key))
            if value is not None:
                return value
        percent = cls._as_float(item.get("probability_percent"))
        if percent is not None:
            return percent / 100.0
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def notify_opportunities(
        self,
        arb_list: List[dict],
        middle_list: List[dict],
        ev_list: List[dict],
        scan_time: Optional[str] = None,
    ) -> dict:
        """Filter by thresholds and send alerts.  Returns a status summary dict."""
        if not self.is_configured:
            return {"sent": False, "reason": "not_configured"}

        ts = scan_time or _utc_now()

        filtered_arb = [
            o for o in (arb_list or [])
            if isinstance(o, dict) and (o.get("roi_percent") or 0) >= self.min_roi
        ]
        filtered_ev = [
            o for o in (ev_list or [])
            if isinstance(o, dict)
            and ((o.get("edge_percent") or o.get("net_edge_percent") or 0) >= self.min_edge)
        ]
        filtered_mid = [
            o for o in (middle_list or [])
            if isinstance(o, dict) and self._middle_ev_value(o) >= self.min_ev
        ]

        if not filtered_arb and not filtered_ev and not filtered_mid:
            return {"sent": False, "reason": "nothing_above_threshold"}

        summary = {
            "scan_time": ts,
            "arbitrage_count": len(filtered_arb),
            "middles_count": len(filtered_mid),
            "ev_count": len(filtered_ev),
            "top_arbitrage": self._top_arb(filtered_arb),
            "top_middles": self._top_middles(filtered_mid),
            "top_ev": self._top_ev(filtered_ev),
        }

        results: dict = {"sent": False, "channels": {}}
        if self.webhook_url:
            results["channels"]["webhook"] = self._send_webhook(summary)
            results["sent"] = True
        if self.telegram_token and self.telegram_chat_id:
            results["channels"]["telegram"] = self._send_telegram(
                self._format_telegram_message(summary)
            )
            results["sent"] = True

        return results

    # ------------------------------------------------------------------
    # Channel senders
    # ------------------------------------------------------------------

    def _send_webhook(self, payload: dict) -> dict:
        if _requests is None:
            return {"ok": False, "error": "requests library not available"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.webhook_secret:
            sig = hmac.new(
                self.webhook_secret.encode("utf-8"), body, hashlib.sha256
            ).hexdigest()
            headers["X-Signature-SHA256"] = f"sha256={sig}"
        try:
            resp = _requests.post(
                self.webhook_url,
                data=body,
                headers=headers,
                timeout=self.timeout,
            )
            return {"ok": resp.ok, "status_code": resp.status_code}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _send_telegram(self, message: str) -> dict:
        if _requests is None:
            return {"ok": False, "error": "requests library not available"}
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            resp = _requests.post(
                url,
                json={
                    "chat_id": self.telegram_chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=self.timeout,
            )
            return {"ok": resp.ok, "status_code": resp.status_code}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _top_arb(items: List[dict], n: int = 3) -> List[dict]:
        sorted_items = sorted(items, key=lambda x: x.get("roi_percent") or 0, reverse=True)
        out = []
        for o in sorted_items[:n]:
            out.append(
                {
                    "event": o.get("event"),
                    "sport": o.get("sport_display") or o.get("sport"),
                    "market": o.get("market"),
                    "roi_percent": o.get("roi_percent"),
                    "books": [b.get("bookmaker") for b in (o.get("best_odds") or [])],
                }
            )
        return out

    @staticmethod
    def _top_middles(items: List[dict], n: int = 3) -> List[dict]:
        sorted_items = sorted(items, key=Notifier._middle_ev_value, reverse=True)
        out = []
        for o in sorted_items[:n]:
            out.append(
                {
                    "event": o.get("event"),
                    "market": o.get("market"),
                    "gap_points": Notifier._middle_gap_points(o),
                    "ev": Notifier._middle_ev_value(o),
                    "probability": Notifier._middle_probability(o),
                }
            )
        return out

    @staticmethod
    def _top_ev(items: List[dict], n: int = 3) -> List[dict]:
        sorted_items = sorted(
            items,
            key=lambda x: x.get("edge_percent") or x.get("net_edge_percent") or 0,
            reverse=True,
        )
        out = []
        for o in sorted_items[:n]:
            bet = o.get("bet") or {}
            out.append(
                {
                    "event": o.get("event"),
                    "market": o.get("market"),
                    "soft_book": bet.get("soft_book"),
                    "edge_percent": o.get("edge_percent") or o.get("net_edge_percent"),
                    "ev_per_100": o.get("ev_per_100"),
                }
            )
        return out

    def _format_telegram_message(self, summary: dict) -> str:
        lines = [
            f"🔍 <b>Edge Scanner Alert</b>  {summary['scan_time']}",
            "",
        ]
        if summary["arbitrage_count"]:
            lines.append(f"⚡ <b>Arbitrage</b> ({summary['arbitrage_count']} found)")
            for o in summary["top_arbitrage"]:
                roi = o.get("roi_percent", 0)
                lines.append(
                    f"  • {o.get('event')} [{o.get('market')}] "
                    f"+{roi:.2f}% ROI — {', '.join(filter(None, o.get('books') or []))}"
                )
        if summary["middles_count"]:
            lines.append(f"\n🎯 <b>Middles</b> ({summary['middles_count']} found)")
            for o in summary["top_middles"]:
                lines.append(
                    f"  • {o.get('event')} [{o.get('market')}] "
                    f"gap={o.get('gap_points')} EV={o.get('ev')}"
                )
        if summary["ev_count"]:
            lines.append(f"\n📈 <b>+EV</b> ({summary['ev_count']} found)")
            for o in summary["top_ev"]:
                lines.append(
                    f"  • {o.get('event')} [{o.get('market')}] "
                    f"{o.get('soft_book')} edge={o.get('edge_percent'):.1f}%"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_notifier: Optional[Notifier] = None


def get_notifier() -> Notifier:
    """Return the module-level singleton Notifier."""
    global _default_notifier
    if _default_notifier is None:
        _default_notifier = Notifier()
    return _default_notifier

"""Project configuration loader."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = "settings.json"


def _config_path() -> Path:
    raw = os.getenv("APP_CONFIG_PATH", "").strip() or DEFAULT_CONFIG_PATH
    return Path(raw)


def load_config() -> Dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def apply_config_env() -> None:
    config = load_config()
    if not config:
        return
    for key, value in config.items():
        if not isinstance(key, str) or not key:
            continue
        if key in os.environ:
            continue
        os.environ[key] = _stringify(value)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


def sorted_unique_tuple(values: Iterable[object]) -> Tuple[str, ...]:
    normalized = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ProviderCapability:
    key: str
    title: str
    supported_sport_keys: Tuple[str, ...]
    supported_markets: Tuple[str, ...]
    live_mode_supported: bool

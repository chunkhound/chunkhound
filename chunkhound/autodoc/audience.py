from __future__ import annotations

from chunkhound.core.audience import normalize_audience


def _normalize_audience(audience: str | None) -> str:
    return normalize_audience(audience)


from __future__ import annotations


def _normalize_taint(taint: str | None) -> str:
    if not taint:
        return "balanced"
    cleaned = taint.strip().lower()
    if cleaned in {"technical", "balanced", "end-user"}:
        return cleaned
    return "balanced"


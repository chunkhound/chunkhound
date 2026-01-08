from __future__ import annotations

from typing import Literal

Audience = Literal["technical", "balanced", "end-user"]


_AUDIENCE_ALIASES: dict[str, Audience] = {
    "1": "technical",
    "technical": "technical",
    "2": "balanced",
    "balanced": "balanced",
    "": "balanced",
    "3": "end-user",
    "end-user": "end-user",
    "end_user": "end-user",
    "enduser": "end-user",
}


def normalize_audience(value: str | None) -> Audience:
    """Best-effort normalization for internal use; unknown values become 'balanced'."""
    normalized = (value or "").strip().lower()
    return _AUDIENCE_ALIASES.get(normalized, "balanced")


def parse_audience(value: str) -> Audience:
    """Strict parsing for user input; unknown values raise ValueError."""
    normalized = value.strip().lower()
    resolved = _AUDIENCE_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            "Invalid audience value. Use 1|2|3 or technical|balanced|end-user."
        )
    return resolved


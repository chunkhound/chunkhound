"""Per-response websearch verdicts: evidence about one response, not health.

A verdict classifies a single engine response (served, empty, refused,
transient, drifted). Engine health — strikes, cooldowns, probes — is a
separate, accumulated judgement owned by the orchestrator's registry: one
response must never by itself put an engine into cooldown.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    SERVED = "served"
    EMPTY = "empty"
    REFUSED = "refused"
    TRANSIENT = "transient"
    DRIFT = "drift"


@dataclass(frozen=True, slots=True)
class EngineResponse:
    """One engine's classified response for one query."""

    rows: tuple[tuple[str, str, str], ...]
    verdict: Verdict
    status: int
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class EngineRun:
    """One engine's aggregated verdict over a full multi-query scan."""

    rows: list[tuple[str, str, str]]
    verdict: Verdict
    evidence: str = ""
    strike: bool = False


def verdict_from_status(status: int) -> Verdict:
    """Map a non-200 HTTP status to a verdict.

    200 is not a verdict source: SERVED must come from parsed rows, never
    from the status alone (the arm below exists only for table completeness
    and callers must not rely on it).
    """
    if status == 200:
        return Verdict.SERVED
    if status in (202, 403, 429):
        return Verdict.REFUSED
    if status == 408 or 300 <= status <= 399 or 500 <= status <= 599:
        return Verdict.TRANSIENT
    return Verdict.DRIFT


def status_evidence(status: int) -> str:
    """Human-readable evidence phrase for a status-derived verdict."""
    return f"{verdict_from_status(status).value} status {status}"

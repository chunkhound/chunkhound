"""Persistent learned capabilities for LLM request features."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Literal, TypeGuard

from loguru import logger

from chunkhound.utils.cache_paths import platform_cache_root

_CACHE_ENV = "CHUNKHOUND_LLM_CAPABILITY_CACHE"
_CACHE_FILENAME = "llm-capabilities.json"
_FORMAT_VERSION = 1
_ACCEPTED_TTL_SECONDS = 30 * 24 * 60 * 60
_REJECTED_TTL_SECONDS = 24 * 60 * 60
_MAX_CACHE_ENTRIES = 256
CapabilityState = Literal["unknown", "accepted", "rejected"]
UNKNOWN: CapabilityState = "unknown"
ACCEPTED: CapabilityState = "accepted"
REJECTED: CapabilityState = "rejected"


def capability_key(provider: str, model: str) -> str:
    """Return the stable capability cache key for a provider/model pair."""
    return f"{provider}:{model}"


def state_ttl_seconds(state: CapabilityState) -> float:
    """Return the lease duration for a persisted capability decision."""
    if state == ACCEPTED:
        return _ACCEPTED_TTL_SECONDS
    if state == REJECTED:
        return _REJECTED_TTL_SECONDS
    raise ValueError("Unknown capabilities do not have a lease")


def deadline_for(state: CapabilityState, timestamp: float) -> float:
    """Return the absolute lease deadline for a decision made at ``timestamp``."""
    return timestamp + state_ttl_seconds(state)


def _read_decision(entry: object) -> tuple[CapabilityState, float] | None:
    """Return a live capability decision and its persisted deadline."""
    if not isinstance(entry, dict):
        return None

    accepted = entry.get("accepted")
    timestamp = entry.get("ts")
    format_version = entry.get("format_version")
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version != _FORMAT_VERSION
        or not isinstance(accepted, bool)
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
    ):
        return None

    state: CapabilityState = ACCEPTED if accepted else REJECTED
    try:
        if not math.isfinite(timestamp):
            return None
        if time.time() - timestamp >= state_ttl_seconds(state):
            return None
        return (state, deadline_for(state, timestamp))
    except OverflowError:
        # Timestamps too extreme for float arithmetic cannot be trusted.
        return None


def _is_live_entry(entry: object) -> TypeGuard[dict[str, Any]]:
    """Return whether a persisted entry is well-formed, current, and unexpired."""
    return _read_decision(entry) is not None


def _is_newer_format_entry(entry: object) -> bool:
    """Return whether an entry was persisted by a newer cache format."""
    if not isinstance(entry, dict):
        return False
    format_version = entry.get("format_version")
    return (
        isinstance(format_version, int)
        and not isinstance(format_version, bool)
        and format_version > _FORMAT_VERSION
    )


def _entry_timestamp(entry: object) -> float:
    """Return an entry's timestamp, or -inf when it cannot be trusted."""
    if isinstance(entry, dict):
        timestamp = entry.get("ts")
        if (
            isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and math.isfinite(timestamp)
        ):
            return float(timestamp)
    return float("-inf")


def _prune_to_capacity(payload: dict[str, Any]) -> None:
    """Evict the oldest entries when the cache exceeds its entry cap.

    A newer-format entry is kept by the live-entry filter even though this
    version cannot read its lease, so a v1 writer never deletes a newer
    writer's data. The cap may still evict the oldest such row, and a lost row
    only costs one re-probe.
    """
    overflow = len(payload) - _MAX_CACHE_ENTRIES
    if overflow <= 0:
        return
    oldest = sorted(payload, key=lambda key: _entry_timestamp(payload[key]))[:overflow]
    for key in oldest:
        del payload[key]


def _default_cache_path() -> Path:
    """Return the user-scoped capability cache file path."""
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override).expanduser()

    return platform_cache_root() / _CACHE_FILENAME


class LLMCapabilityStore:
    """Store provider/model capabilities with state-dependent leases.

    Persistence is best-effort and uncoordinated: concurrent writers race on
    a whole-file rewrite, so the last writer wins and a concurrently learned
    sibling entry may be lost. Writes replace the file atomically, so readers
    never observe a torn payload and a lost entry merely costs one re-probe.
    """

    def __init__(self) -> None:
        self._path = _default_cache_path()

    def get_with_expiry(
        self, provider: str, model: str
    ) -> tuple[CapabilityState, float | None]:
        """Return a live decision and its persisted deadline without writing."""
        decision = _read_decision(self._read().get(capability_key(provider, model)))
        return decision if decision is not None else (UNKNOWN, None)

    def get(self, provider: str, model: str) -> CapabilityState:
        """Return a live decision, or unknown when no live lease exists."""
        return self.get_with_expiry(provider, model)[0]

    def set(self, provider: str, model: str, state: CapabilityState) -> float:
        """Persist a capability decision and return its deadline."""
        if state not in {ACCEPTED, REJECTED}:
            raise ValueError("Capability state must be 'accepted' or 'rejected'")

        timestamp = time.time()
        deadline = deadline_for(state, timestamp)
        try:
            payload = {
                key: entry
                for key, entry in self._read().items()
                if _is_live_entry(entry) or _is_newer_format_entry(entry)
            }
            payload[capability_key(provider, model)] = {
                "accepted": state == ACCEPTED,
                "ts": timestamp,
                "format_version": _FORMAT_VERSION,
            }
            _prune_to_capacity(payload)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self._path.with_name(
                f"{self._path.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            try:
                temporary_path.write_text(
                    json.dumps(payload) + "\n", encoding="utf-8"
                )
                os.replace(temporary_path, self._path)
            finally:
                temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.opt(exception=True).warning(
                "Failed to persist LLM capability cache at {}", self._path
            )
        return deadline

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            # ValueError covers json.JSONDecodeError and UnicodeDecodeError;
            # RecursionError covers pathologically nested JSON payloads.
            return {}
        return data if isinstance(data, dict) else {}

"""Shared daemon-status derivation helpers for MCP surfaces."""

from __future__ import annotations

import copy
from typing import Any

from chunkhound.version import __version__


def normalize_scan_progress(scan_progress: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized scan_progress payload with a realtime section."""
    progress = copy.deepcopy(scan_progress or {})
    realtime = progress.get("realtime")
    if not isinstance(realtime, dict):
        realtime = {}
        progress["realtime"] = realtime
    return progress


def derive_daemon_status(scan_progress: dict[str, Any] | None) -> dict[str, Any]:
    """Derive the public daemon_status payload from scan and realtime state."""
    progress = normalize_scan_progress(scan_progress)
    realtime = progress["realtime"]
    resync = realtime.get("resync")
    if not isinstance(resync, dict):
        resync = {}

    scan_error = progress.get("scan_error")
    realtime_error = realtime.get("last_error") or resync.get("last_error")
    realtime_state = realtime.get("service_state")
    live_indexing_state = realtime.get("live_indexing_state")
    realtime_needs_resync = bool(resync.get("needs_resync"))
    is_scanning = bool(progress.get("is_scanning"))
    # query_ready answers whether at least one successful index already exists.
    # Later live-indexing degradation must not erase that searchability signal.
    query_ready = bool(progress.get("scan_completed_at")) and not is_scanning

    degraded = (
        bool(scan_error)
        or bool(realtime_error)
        or realtime_state == "degraded"
        or live_indexing_state == "stalled"
        or realtime_needs_resync
    )
    if degraded:
        status = "degraded"
    elif query_ready:
        status = "ready"
    else:
        status = "initializing"

    return {
        "status": status,
        "server_version": __version__,
        "query_ready": query_ready,
        "scan_progress": progress,
    }

"""Exact filenames of ChunkHound-owned service temp files.

Service temp files live inside a watched root (e.g. the watchdog delivery
probe) but carry no indexable content. They are matched by exact filename,
never by the broad ".chunkhound-" prefix, so user files like
".chunkhound-notes.py" stay admissible.
"""

from __future__ import annotations

SERVICE_TEMP_PROBE_FILENAME = ".chunkhound-probe.tmp"
SERVICE_TEMP_HYDRATION_MARKER_FILENAME = ".chunkhound-hydrated.json"

SERVICE_TEMP_FILE_NAMES: frozenset[str] = frozenset(
    {
        SERVICE_TEMP_PROBE_FILENAME,
        SERVICE_TEMP_HYDRATION_MARKER_FILENAME,
    }
)


def is_service_temp_name(name: str) -> bool:
    return name in SERVICE_TEMP_FILE_NAMES

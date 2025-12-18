from __future__ import annotations

from typing import Any


def compute_db_scope_stats(
    services: Any, scope_label: str
) -> tuple[int, int, set[str]]:
    """Compute indexed file/chunk totals and scoped file set for the folder."""
    scope_total_files = 0
    scope_total_chunks = 0
    scoped_files: set[str] = set()
    try:
        provider = getattr(services, "provider", None)
        if provider is None:
            return 0, 0, scoped_files
        prefix = None if scope_label == "/" else scope_label.rstrip("/") + "/"

        # Preferred: use provider-level aggregation to avoid loading full chunk code.
        get_scope_stats = getattr(provider, "get_scope_stats", None)
        if callable(get_scope_stats):
            total_files, total_chunks = get_scope_stats(prefix)
            return int(total_files), int(total_chunks), set()

        # Fallback: scan all chunk metadata (legacy providers/stubs).
        chunks_meta = provider.get_all_chunks_with_metadata()
        for chunk in chunks_meta:
            path = (chunk.get("file_path") or "").replace("\\", "/")
            if not path:
                continue
            if prefix and not path.startswith(prefix):
                continue
            scoped_files.add(path)
            scope_total_chunks += 1
        scope_total_files = len(scoped_files)
    except Exception:
        return 0, 0, set()

    return scope_total_files, scope_total_chunks, scoped_files

from __future__ import annotations

from collections.abc import Iterable

from loguru import logger

from chunkhound.code_mapper.utils import compute_scope_prefix
from chunkhound.database_factory import DatabaseServices


def compute_db_scope_stats(
    services: DatabaseServices, scope_label: str
) -> tuple[int, int, set[str]]:
    """Compute indexed file/chunk totals and scoped file set for the folder."""
    scope_total_files = 0
    scope_total_chunks = 0
    scoped_files: set[str] = set()
    try:
        provider = getattr(services, "provider", None)
        if provider is None:
            return 0, 0, scoped_files
        prefix = compute_scope_prefix(scope_label)

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
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(f"Code Mapper: failed to compute scope stats: {exc}")
        return 0, 0, set()

    return scope_total_files, scope_total_chunks, scoped_files


def compute_unreferenced_scope_files(
    services: DatabaseServices,
    scope_label: str,
    referenced_files: Iterable[str],
) -> list[str] | None:
    """Return list of unreferenced files for the scope, or None if unavailable."""
    try:
        provider = getattr(services, "provider", None)
        get_scope_file_paths = getattr(provider, "get_scope_file_paths", None)
        if not callable(get_scope_file_paths):
            return None

        prefix = compute_scope_prefix(scope_label)
        all_files = get_scope_file_paths(prefix)
        referenced_set = {
            str(p).replace("\\", "/")
            for p in referenced_files
            if p and (not prefix or str(p).replace("\\", "/").startswith(prefix))
        }
        unreferenced = [
            str(p).replace("\\", "/")
            for p in all_files
            if p and str(p).replace("\\", "/") not in referenced_set
        ]
        return unreferenced
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning(f"Code Mapper: failed to compute unreferenced files: {exc}")
        return None

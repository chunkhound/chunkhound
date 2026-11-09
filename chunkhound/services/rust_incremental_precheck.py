"""Decides, on Python's own connection, whether an incremental Rust run has
anything to write -- before the database is handed to Rust.

Handing the database to Rust (see rust_pipeline_runner.run_rust_indexing_phase)
closes Python's connection and refuses every reader with
DatabaseRustPipelineInProgressError until Rust returns. An MCP server's
start-up catch-up does that on every start, and the common start changes
nothing: a fresh worktree seeded from another checkout's index has new mtimes
and identical bytes. Measured on a 1174-file index, such a start refused
searches for 39-46 s, most of it spent on work that wrote no file.

This module mirrors the classification of the Rust differ
(src/pipeline/differ.rs::compute_diff) so the caller can skip the handoff when
the answer is "nothing", and hand Rust only the changed files otherwise:

- no DB row, NULL stored mtime, an unreadable file, or a file whose relative
  key can't be derived: changed;
- mtime differs beyond the epsilon: changed unless the file's xxh3_64 digest
  equals the stored content_hash (the same digest Rust computes; see
  differ.rs's test_hash_matches_python_xxh3_64);
- mtime equal but a known stored size differs: changed;
- otherwise unchanged.

A file is called unchanged only when its bytes are unchanged: either the hash
matches, or mtime and size both match (the same trust Rust's mtime fast path
places in them). A disagreement with Rust over the mtime read (the two DuckDB
builds' session time zones) can only make this check hash more files.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunkhound.core.utils.path_utils import get_relative_path_safe
from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.utils.hashing import compute_file_hash

# The same columns and the same mtime expression as the Rust differ's
# snapshot (src/db/duckdb_backend/read.rs FILE_STATE_SELECT): casting the
# naive stored TIMESTAMP back through TIMESTAMPTZ reverses the local-time
# shift applied when it was written.
_FILE_STATE_SELECT = (
    "SELECT path, EXTRACT(EPOCH FROM modified_time::TIMESTAMPTZ) AS mtime, "
    "size, content_hash FROM files"
)


@dataclass(frozen=True)
class RustIncrementalWork:
    """What an incremental Rust run would do, decided without handing it the DB."""

    changed: list[Path]
    removed: list[str]
    skipped_by_hash: int

    def has_work(self, *, do_cleanup: bool) -> bool:
        return bool(self.changed) or (do_cleanup and bool(self.removed))


def classify_files(
    rows: list[dict[str, Any]],
    files: list[Path],
    project_root: Path,
    mtime_epsilon: float,
) -> RustIncrementalWork:
    """Classify discovered files against the DB's file rows, as the Rust differ does."""
    db_state = {str(r["path"]): r for r in rows if r.get("path")}
    changed: list[Path] = []
    seen: set[str] = set()
    skipped_by_hash = 0
    for path in files:
        try:
            rel = get_relative_path_safe(path, project_root).as_posix()
        except ValueError:
            # Rust reports these as "not under project root"; let it.
            changed.append(path)
            continue
        seen.add(rel)
        row = db_state.get(rel)
        if row is None or row.get("mtime") is None:
            changed.append(path)
            continue
        try:
            st = path.stat()
        except OSError:
            changed.append(path)
            continue
        if abs(st.st_mtime - float(row["mtime"])) > mtime_epsilon:
            stored_hash = row.get("content_hash")
            try:
                same_bytes = (
                    bool(stored_hash) and compute_file_hash(path) == stored_hash
                )
            except (OSError, ValueError):
                # ValueError: compute_file_hash refuses a path that stopped
                # being a regular file after stat() (deleted or replaced).
                same_bytes = False
            if same_bytes:
                skipped_by_hash += 1
            else:
                changed.append(path)
        elif row.get("size") is not None and int(row["size"]) != st.st_size:
            changed.append(path)
    removed = sorted(p for p in db_state if p not in seen)
    return RustIncrementalWork(
        changed=changed, removed=removed, skipped_by_hash=skipped_by_hash
    )


async def find_rust_incremental_work(
    db: DatabaseProvider,
    files: list[Path],
    project_root: Path,
    mtime_epsilon: float,
) -> RustIncrementalWork:
    """Read the file rows through the provider's ordinary (shared) connection
    and classify every discovered file. Runs off the event loop: hashing a
    fresh worktree reads every file, and the MCP server must keep answering
    meanwhile."""

    def _run() -> RustIncrementalWork:
        rows = db.execute_query(_FILE_STATE_SELECT, [])
        return classify_files(rows, files, project_root, mtime_epsilon)

    return await asyncio.to_thread(_run)

"""Contract test: orphan cleanup parity between the Rust and Python pipelines.

Covers a gap found comparing docs/rust-pipeline-architecture.html against the
implementation: IndexingCoordinator.process_directory() unconditionally skipped
Python's _cleanup_orphaned_files() whenever CHUNKHOUND_USE_RUST=1.

Note: a plain incremental Rust run already deletes DB rows for any file absent
from the discovered-file list handed to it (whatever the cause -- deleted from
disk, or excluded by a pattern change) via its own diff logic, so that case was
never actually broken. The gap only surfaces with force_reindex=True: Rust
skips its own diff entirely on a forced full reindex (incremental=False means
no delete_paths are computed at all), and -- before this fix -- Python's
cleanup was *also* skipped whenever the Rust pipeline was active, regardless of
force_reindex. So force_reindex=True + Rust used to mean zero orphan cleanup of
any kind: a file deleted from disk kept its stale row forever.

This test drives that scenario through the real production entry point,
IndexingCoordinator.process_directory() via DirectoryIndexingService, with the
Rust pipeline forced on, and asserts the deleted file's row is still cleaned up.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import duckdb
import pytest

from chunkhound.core.config.config import Config
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from tests.contracts.pipeline_harness import disconnect_registry_db

try:
    import chunkhound_native

    _NATIVE_AVAILABLE = hasattr(chunkhound_native, "IndexingPipeline")
except ImportError:
    _NATIVE_AVAILABLE = False


def _build_config(
    root: Path, db_dir: Path, *, exclude: list[str], force_reindex: bool = False
) -> Config:
    return Config(
        target_dir=root,
        database={"provider": "duckdb", "path": str(db_dir)},
        indexing={
            "include": ["**/*.py"],
            "exclude": exclude,
            "force_reindex": force_reindex,
        },
        embeddings_disabled=True,
    )


def _indexed_paths(db_dir: Path) -> set[str]:
    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        rows = conn.execute("SELECT path FROM files").fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


@pytest.mark.skipif(
    not _NATIVE_AVAILABLE, reason="chunkhound_native.IndexingPipeline not built"
)
def test_rust_pipeline_force_reindex_still_cleans_up_deleted_files(
    tmp_path, monkeypatch
):
    """A file deleted from disk must be cleaned up even on a Rust-driven,
    force_reindex=True run.

    force_reindex=True makes Rust skip its own diff entirely (incremental=False
    means no delete_paths are computed at all), so it never notices the file is
    gone. Before this fix, Python's _cleanup_orphaned_files() -- which would
    have caught it independently -- was *also* skipped whenever the Rust
    pipeline was active, regardless of force_reindex. Net effect pre-fix:
    force_reindex=True + Rust performed zero orphan cleanup of any kind, and
    the deleted file's row lingered forever.
    """
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    root = tmp_path / "repo"
    root.mkdir()
    keep_file = root / "keep.py"
    keep_file.write_text("def keep():\n    return 1\n")
    deleted_file = root / "deleted.py"
    deleted_file.write_text("def deleted():\n    return 1\n")

    db_dir = tmp_path / "db"

    # Pass 1: normal incremental index of both files.
    config = _build_config(root, db_dir, exclude=[])
    configure_registry(config)
    coordinator = create_indexing_coordinator()
    service = DirectoryIndexingService(indexing_coordinator=coordinator, config=config)
    asyncio.run(service.process_directory(root, no_embeddings=True))

    before = _indexed_paths(db_dir)
    assert "keep.py" in before
    assert "deleted.py" in before

    # Release the DuckDB connection before a second run touches the same file
    # (Rust's own run already leaves it disconnected, but this also covers
    # the first pass running through the Python path, e.g. if
    # CHUNKHOUND_USE_RUST gets toggled off upstream in the future).
    disconnect_registry_db()

    deleted_file.unlink()

    config2 = _build_config(root, db_dir, exclude=[], force_reindex=True)
    configure_registry(config2)
    coordinator2 = create_indexing_coordinator()
    service2 = DirectoryIndexingService(
        indexing_coordinator=coordinator2, config=config2
    )
    asyncio.run(service2.process_directory(root, no_embeddings=True))

    after = _indexed_paths(db_dir)
    assert "keep.py" in after
    assert "deleted.py" not in after, (
        "file deleted from disk should still be cleaned up on a "
        "force_reindex=True run under the Rust pipeline"
    )

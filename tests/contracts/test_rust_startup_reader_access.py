"""Contract test: an incremental Rust run with nothing to write keeps the
database available to readers.

Each MCP server start runs an incremental catch-up through the Rust pipeline.
Handing the database to Rust publishes ``set_rust_pipeline_in_progress(True)``
and closes Python's connection, so every search in that window is refused with
DatabaseRustPipelineInProgressError. Measured on a 1174-file index, a start
with no content changes refused searches for 39-46 s, most of it a compaction
Rust ran on a run that wrote nothing.

A fresh git worktree seeded from another checkout's index is the common case:
every mtime differs from the DB, every file's bytes match its stored hash.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from chunkhound.core.config.config import Config
from chunkhound.providers.database.serial_database_provider import (
    SerialDatabaseProvider,
)
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.rust_incremental_precheck import find_rust_incremental_work
from tests.contracts.pipeline_harness import disconnect_registry_db, files_table_paths


def _config(root: Path, db_dir: Path) -> Config:
    return Config(
        target_dir=root,
        database={"provider": "duckdb", "path": str(db_dir)},
        indexing={"include": ["**/*.py"]},
        embeddings_disabled=True,
    )


def test_unchanged_reindex_never_takes_the_database_from_readers(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    root = tmp_path / "repo"
    root.mkdir()
    for name in ("alpha.py", "beta.py"):
        (root / name).write_text(f"def {name[:-3]}_marker():\n    return 1\n")
    db_dir = tmp_path / "db"

    config = _config(root, db_dir)
    configure_registry(config)
    asyncio.run(
        DirectoryIndexingService(
            indexing_coordinator=create_indexing_coordinator(), config=config
        ).process_directory(root, no_embeddings=True)
    )
    disconnect_registry_db()

    # Same bytes, new mtimes: what a fresh worktree checkout looks like to
    # the index it was seeded from.
    for path in root.iterdir():
        st = path.stat()
        os.utime(path, (st.st_atime + 1000, st.st_mtime + 1000))

    published: list[bool] = []
    original = SerialDatabaseProvider.set_rust_pipeline_in_progress

    def _record(self: SerialDatabaseProvider, active: bool) -> None:
        published.append(active)
        original(self, active)

    monkeypatch.setattr(
        SerialDatabaseProvider, "set_rust_pipeline_in_progress", _record
    )

    configure_registry(config)
    coordinator = create_indexing_coordinator()
    stats = asyncio.run(
        DirectoryIndexingService(
            indexing_coordinator=coordinator, config=config
        ).process_directory(root, no_embeddings=True)
    )

    assert True not in published, (
        "an incremental run with no content changes handed the database to "
        "Rust, refusing every reader for the length of the run"
    )
    assert stats.files_processed == 0
    results, _ = coordinator._db.search_regex("alpha_marker")
    assert results, "a search after the unchanged run must return results"
    disconnect_registry_db()
    assert files_table_paths(db_dir) == {"alpha.py", "beta.py"}


def test_editing_one_file_reindexes_it_and_keeps_every_other_file(
    tmp_path, monkeypatch
):
    """With nothing removed, Rust is handed only the changed files. It must
    run with cleanup off: with cleanup on, Rust treats every row missing from
    the list it was given as a deleted file."""
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    root = tmp_path / "repo"
    root.mkdir()
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (root / name).write_text(f"def {name[:-3]}_marker():\n    return 1\n")
    db_dir = tmp_path / "db"

    config = _config(root, db_dir)
    configure_registry(config)
    asyncio.run(
        DirectoryIndexingService(
            indexing_coordinator=create_indexing_coordinator(), config=config
        ).process_directory(root, no_embeddings=True)
    )
    disconnect_registry_db()

    (root / "alpha.py").write_text("def alpha_edited_marker():\n    return 2\n")

    configure_registry(config)
    coordinator = create_indexing_coordinator()
    stats = asyncio.run(
        DirectoryIndexingService(
            indexing_coordinator=coordinator, config=config
        ).process_directory(root, no_embeddings=True)
    )

    assert stats.files_processed == 1
    assert coordinator._db.search_regex("alpha_edited_marker")[0]
    assert coordinator._db.search_regex("gamma_marker")[0]
    disconnect_registry_db()
    assert files_table_paths(db_dir) == {"alpha.py", "beta.py", "gamma.py"}


def _index_then_touch(root: Path, config: Config) -> None:
    """Index root, then give every file a new mtime with the same bytes."""
    configure_registry(config)
    asyncio.run(
        DirectoryIndexingService(
            indexing_coordinator=create_indexing_coordinator(), config=config
        ).process_directory(root, no_embeddings=True)
    )
    disconnect_registry_db()
    for path in root.iterdir():
        st = path.stat()
        os.utime(path, (st.st_atime + 1000, st.st_mtime + 1000))


def test_precheck_refreshes_the_mtime_of_a_hash_confirmed_file(
    tmp_path, monkeypatch
):
    """A file the precheck confirms unchanged by hash gets its stored mtime
    refreshed, so the next start does not hash it again."""
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    root = tmp_path / "repo"
    root.mkdir()
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (root / name).write_text(f"def {name[:-3]}_marker():\n    return 1\n")
    config = _config(root, tmp_path / "db")
    _index_then_touch(root, config)

    configure_registry(config)
    db = create_indexing_coordinator()._db
    files = sorted(root.iterdir())
    first = asyncio.run(find_rust_incremental_work(db, files, root, 0.01))
    second = asyncio.run(find_rust_incremental_work(db, files, root, 0.01))
    disconnect_registry_db()

    assert first.skipped_by_hash == len(files)
    assert second.changed == []
    assert second.skipped_by_hash == 0


def test_python_change_scan_refreshes_the_mtime_of_a_hash_confirmed_file(
    tmp_path, monkeypatch
):
    """Phase 2.5 of the Python pipeline: a file confirmed unchanged by hash
    gets its stored mtime refreshed, so the next run does not hash it again."""
    # Unset means Rust ON, which bypasses Phase 2.5.
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")

    root = tmp_path / "repo"
    root.mkdir()
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (root / name).write_text(f"def {name[:-3]}_marker():\n    return 1\n")
    config = _config(root, tmp_path / "db")
    _index_then_touch(root, config)

    hashed: list[Path] = []
    original = IndexingCoordinator._compute_hash_with_fallback

    def _spy(self: IndexingCoordinator, file_path: Path) -> str | None:
        hashed.append(file_path)
        return original(self, file_path)

    monkeypatch.setattr(IndexingCoordinator, "_compute_hash_with_fallback", _spy)

    hashes_per_run: list[int] = []
    for _ in range(2):
        hashed.clear()
        configure_registry(config)
        asyncio.run(
            DirectoryIndexingService(
                indexing_coordinator=create_indexing_coordinator(), config=config
            ).process_directory(root, no_embeddings=True)
        )
        disconnect_registry_db()
        hashes_per_run.append(len(hashed))

    assert hashes_per_run == [3, 0]

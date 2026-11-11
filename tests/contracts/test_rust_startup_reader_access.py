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

import duckdb
import pytest

from chunkhound.core.config.config import Config
from chunkhound.pipeline_bridge import rust_mtime_epsilon
from chunkhound.providers.database.serial_database_provider import (
    SerialDatabaseProvider,
)
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services import indexing_coordinator as coordinator_module
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.rust_incremental_precheck import (
    RustIncrementalWork,
    RustRunPlan,
    find_rust_incremental_work,
)
from tests.contracts.pipeline_harness import (
    disconnect_registry_db,
    files_table_paths,
    index_with_rust,
)


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


def _execute(db_dir: Path, sql: str, params: list[object]) -> None:
    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        conn.execute(sql, params)
    finally:
        conn.close()


def _null_stored_mtime(root: Path, db_dir: Path) -> None:
    _execute(
        db_dir, "UPDATE files SET modified_time = NULL WHERE path = ?", ["alpha.py"]
    )


def _resize_keeping_mtime(root: Path, db_dir: Path) -> None:
    path = root / "alpha.py"
    st = path.stat()
    path.write_text("def alpha_resized_marker():\n    return 22\n")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def _touch(path: Path) -> None:
    st = path.stat()
    os.utime(path, (st.st_atime + 1000, st.st_mtime + 1000))


def _touch_without_stored_hash(root: Path, db_dir: Path) -> None:
    _execute(
        db_dir, "UPDATE files SET content_hash = NULL WHERE path = ?", ["alpha.py"]
    )
    _touch(root / "alpha.py")


def _touch_same_bytes(root: Path, db_dir: Path) -> None:
    _touch(root / "alpha.py")


def _leave_untouched(root: Path, db_dir: Path) -> None:
    pass


@pytest.mark.parametrize(
    ("mutate", "rust_reindexes"),
    [
        pytest.param(_null_stored_mtime, 1, id="null-stored-mtime"),
        pytest.param(_resize_keeping_mtime, 1, id="same-mtime-new-size"),
        pytest.param(_touch_without_stored_hash, 1, id="new-mtime-no-stored-hash"),
        pytest.param(_touch_same_bytes, 0, id="new-mtime-same-bytes"),
        pytest.param(_leave_untouched, 0, id="untouched"),
    ],
)
def test_precheck_agrees_with_the_rust_differ(
    tmp_path, monkeypatch, mutate, rust_reindexes
):
    """The precheck re-implements src/pipeline/differ.rs::compute_diff: for the
    same files and rows it must call a file changed exactly when the Rust run,
    handed the full file list, re-indexes it."""
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

    mutate(root, db_dir)

    configure_registry(config)
    coordinator = create_indexing_coordinator()
    # The precheck's verdict, without its mtime refresh: Rust's differ below
    # must see the same rows, or a hash-confirmed file would reach it with
    # the new mtime and never exercise its hash path.
    with monkeypatch.context() as m:
        m.setattr(coordinator._db, "refresh_file_mtimes", lambda entries: 0)
        precheck = asyncio.run(
            find_rust_incremental_work(
                coordinator._db,
                sorted(root.glob("*.py")),
                root,
                rust_mtime_epsilon(config),
            )
        )

    # Rust's verdict: the precheck bypassed, Rust gets the full list and
    # runs its own diff.
    async def _hand_everything_to_rust(
        db, files, project_root, mtime_epsilon, *, do_cleanup, files_to_process
    ) -> RustRunPlan:
        return RustRunPlan(
            files_to_process=files_to_process,
            do_cleanup=do_cleanup,
            skip=False,
            work=RustIncrementalWork(changed=[], removed=[], skipped_by_hash=0),
        )

    monkeypatch.setattr(coordinator_module, "plan_rust_run", _hand_everything_to_rust)
    stats = asyncio.run(
        DirectoryIndexingService(
            indexing_coordinator=coordinator, config=config
        ).process_directory(root, no_embeddings=True)
    )
    disconnect_registry_db()

    assert len(precheck.changed) == stats.files_processed, (
        f"precheck called {[p.name for p in precheck.changed]} changed; "
        f"the Rust differ re-indexed {stats.files_processed} file(s)"
    )
    assert stats.files_processed == rust_reindexes


def _hnsw_index_names(db_dir: Path) -> set[str]:
    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        rows = conn.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
    finally:
        conn.close()
    return {name for (name,) in rows if "hnsw" in name.lower()}


def test_unchanged_reindex_restores_an_hnsw_index_a_crash_left_missing(
    tmp_path, monkeypatch
):
    """A run that never hands the database to Rust must still recreate the
    HNSW index a crashed earlier run left missing, as Rust's open() would."""
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    root = tmp_path / "repo"
    root.mkdir()
    for name in ("alpha.py", "beta.py"):
        (root / name).write_text(f"def {name[:-3]}_marker():\n    return 1\n")
    db_dir = tmp_path / "db"

    first = index_with_rust(root, db_dir, incremental=True)
    assert not first.errors, f"Unexpected errors: {first.errors}"
    names = _hnsw_index_names(db_dir)
    assert names, "expected an HNSW index after the first run"

    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        conn.execute("LOAD vss")
        for name in names:
            conn.execute(f'DROP INDEX "{name}"')
        conn.execute("CHECKPOINT")
    finally:
        conn.close()
    assert not _hnsw_index_names(db_dir), "the simulated crash should leave no index"

    published: list[bool] = []
    original = SerialDatabaseProvider.set_rust_pipeline_in_progress

    def _record(self: SerialDatabaseProvider, active: bool) -> None:
        published.append(active)
        original(self, active)

    monkeypatch.setattr(
        SerialDatabaseProvider, "set_rust_pipeline_in_progress", _record
    )

    config = _config(root, db_dir)
    configure_registry(config)
    stats = asyncio.run(
        DirectoryIndexingService(
            indexing_coordinator=create_indexing_coordinator(), config=config
        ).process_directory(root, no_embeddings=True)
    )
    disconnect_registry_db()

    assert True not in published, "the unchanged run handed the database to Rust"
    assert stats.files_processed == 0
    assert _hnsw_index_names(db_dir), (
        "an unchanged run must still recreate an HNSW index that is missing"
    )


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


def test_precheck_refreshes_the_mtime_of_a_hash_confirmed_file(tmp_path, monkeypatch):
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

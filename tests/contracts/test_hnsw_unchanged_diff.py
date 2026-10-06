"""Contract: an empty incremental diff leaves the HNSW index in place.

A second Rust-pipeline index whose diff has no changed files and no orphan
deletes used to drop every HNSW index and build it again, then checkpoint
the database. ``write-index`` ``(1, 1)`` means the existing index was left
alone. ``(0, 1)`` means a rebuild is starting. A real write, an orphan
delete, a full reindex, and a compaction run still take the rebuild path.
``open()`` restores an index dropped outside the pipeline before that
decision, and the skip then leaves the restored index in place.
"""

import shutil
from pathlib import Path

import duckdb

from tests.contracts.pipeline_harness import (
    collect_table_counts,
    files_table_paths,
    index_with_rust,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _copy_fixture(dest: Path) -> None:
    shutil.copytree(FIXTURE_DIR, dest)


def _index(fixture: Path, db_dir: Path, **kwargs):
    calls: list[tuple[str, int, int]] = []

    def progress_callback(phase: str, current: int, total: int) -> None:
        calls.append((phase, int(current), int(total)))

    result = index_with_rust(
        fixture,
        db_dir,
        skip_embeddings=False,
        progress_callback=progress_callback,
        **kwargs,
    )
    return result, calls


def _hnsw_index_names(db_dir: Path) -> list[str]:
    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        rows = conn.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows if "hnsw" in row[0].lower()]


def _write_index_call(calls: list[tuple[str, int, int]]) -> tuple[str, int, int]:
    hits = [call for call in calls if call[0] == "write-index"]
    assert len(hits) == 1, f"expected one write-index call, got {calls}"
    return hits[0]


def _phase_names(calls: list[tuple[str, int, int]]) -> list[str]:
    return [call[0] for call in calls]


class TestHnswUnchangedDiff:
    """Empty diffs must not rebuild HNSW; writes, deletes, and compaction must."""

    def test_unchanged_incremental_run_leaves_hnsw_in_place(self, tmp_path: Path):
        fixture = tmp_path / "src"
        db_dir = tmp_path / "db"
        _copy_fixture(fixture)

        first, _first_calls = _index(fixture, db_dir, incremental=False)
        assert not first.errors, f"Unexpected errors: {first.errors}"
        assert first.chunks_written > 0
        assert _hnsw_index_names(db_dir), "first index did not create an HNSW index"
        before = collect_table_counts(db_dir)

        second, calls = _index(fixture, db_dir, incremental=True)
        assert not second.errors, f"Unexpected errors: {second.errors}"
        assert second.files_processed == 0
        assert second.chunks_written == 0
        assert second.embeddings_generated == 0
        assert _write_index_call(calls) == ("write-index", 1, 1)
        assert "write-compact" not in _phase_names(calls)
        assert collect_table_counts(db_dir) == before
        assert _hnsw_index_names(db_dir), "empty diff removed the HNSW index"

    def test_modified_file_still_rebuilds_hnsw(self, tmp_path: Path):
        fixture = tmp_path / "src"
        db_dir = tmp_path / "db"
        _copy_fixture(fixture)
        _index(fixture, db_dir, incremental=False)

        main_py = fixture / "main.py"
        main_py.write_text(
            main_py.read_text(encoding="utf-8") + "\ndef added():\n    return 1\n"
        )

        second, calls = _index(fixture, db_dir, incremental=True)
        assert not second.errors, f"Unexpected errors: {second.errors}"
        assert second.chunks_written > 0
        assert _write_index_call(calls) == ("write-index", 0, 1)
        assert _hnsw_index_names(db_dir)

    def test_orphan_delete_still_rebuilds_hnsw(self, tmp_path: Path):
        fixture = tmp_path / "src"
        db_dir = tmp_path / "db"
        _copy_fixture(fixture)
        _index(fixture, db_dir, incremental=False)

        before = collect_table_counts(db_dir)
        victim = sorted(files_table_paths(db_dir))[0]
        (fixture / victim).unlink()

        second, calls = _index(fixture, db_dir, incremental=True)
        assert not second.errors, f"Unexpected errors: {second.errors}"
        assert _write_index_call(calls) == ("write-index", 0, 1)
        assert collect_table_counts(db_dir)["files"] == before["files"] - 1
        assert _hnsw_index_names(db_dir)

    def test_full_reindex_still_rebuilds_hnsw(self, tmp_path: Path):
        fixture = tmp_path / "src"
        db_dir = tmp_path / "db"
        _copy_fixture(fixture)
        _index(fixture, db_dir, incremental=False)

        second, calls = _index(fixture, db_dir, incremental=False)
        assert not second.errors, f"Unexpected errors: {second.errors}"
        assert _write_index_call(calls) == ("write-index", 0, 1)
        assert _hnsw_index_names(db_dir)

    def test_empty_diff_still_compacts(self, tmp_path: Path):
        fixture = tmp_path / "src"
        db_dir = tmp_path / "db"
        _copy_fixture(fixture)
        first, _first_calls = _index(fixture, db_dir, incremental=False)
        assert not first.errors, f"Unexpected errors: {first.errors}"
        before = collect_table_counts(db_dir)

        second, calls = _index(
            fixture,
            db_dir,
            incremental=True,
            compaction_threshold=0.0,
            compaction_min_size_mb=0,
        )
        phases = _phase_names(calls)
        assert not second.errors, f"Unexpected errors: {second.errors}"
        assert second.chunks_written == 0
        assert "write-compact" in phases
        assert "write-index" not in phases
        assert collect_table_counts(db_dir) == before
        assert _hnsw_index_names(db_dir), (
            "compaction left the embeddings table unindexed"
        )

    def test_missing_index_is_restored_then_left_in_place(self, tmp_path: Path):
        fixture = tmp_path / "src"
        db_dir = tmp_path / "db"
        _copy_fixture(fixture)
        _index(fixture, db_dir, incremental=False)

        names = _hnsw_index_names(db_dir)
        assert names
        conn = duckdb.connect(str(db_dir / "chunks.db"))
        try:
            for name in names:
                safe = name.replace('"', '""')
                conn.execute(f'DROP INDEX IF EXISTS "{safe}"')
            conn.execute("CHECKPOINT")
        finally:
            conn.close()
        assert _hnsw_index_names(db_dir) == []

        second, calls = _index(fixture, db_dir, incremental=True)
        assert not second.errors, f"Unexpected errors: {second.errors}"
        assert second.chunks_written == 0
        assert _write_index_call(calls) == ("write-index", 1, 1)
        assert _hnsw_index_names(db_dir), (
            "open() did not restore the dropped HNSW index"
        )

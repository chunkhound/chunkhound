"""Contract test: an incremental run with nothing to write leaves the HNSW index alone.

The store thread brackets its writes with `drop_all_hnsw_indexes()` and
`ensure_all_hnsw_indexes()` so a bulk load builds the index once, at the end.
A run whose diff is empty — no changed files, no orphan deletes, no
compaction due — has nothing to bracket: dropping the index and building it
again over the same rows is a full HNSW build for no change (issue #431).

DuckDB exposes nothing that tells a rebuilt index from the one it replaced
(`duckdb_indexes().index_oid` survives a drop and recreate), so the probe is
the pipeline's own progress contract: the `write-index` phase arrives as
`(0, 1)` when the indexes are about to be rebuilt and as `(1, 1)` when they
were left in place.
"""

import shutil
from pathlib import Path

import duckdb
import pytest

from tests.contracts.pipeline_harness import collect_table_counts, index_with_rust

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"
REBUILT = ("write-index", 0, 1)
LEFT_IN_PLACE = ("write-index", 1, 1)


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


def _drop_hnsw_indexes(db_dir: Path) -> None:
    """Drop every HNSW index, as a run killed mid-bracket would leave them."""
    names = _hnsw_index_names(db_dir)
    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        conn.execute("LOAD vss")
        for name in names:
            conn.execute(f'DROP INDEX "{name}"')
        conn.execute("CHECKPOINT")
    finally:
        conn.close()


def _run(work_dir: Path, db_dir: Path, *, incremental: bool = True, **kwargs):
    """Rust run, incremental by default; returns the result and every progress event."""
    events: list[tuple[str, int, int]] = []
    result = index_with_rust(
        work_dir,
        db_dir,
        incremental=incremental,
        progress_callback=lambda phase, current, total: events.append(
            (phase, current, total)
        ),
        **kwargs,
    )
    assert not result.errors, f"Unexpected errors: {result.errors}"
    return result, events


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """A private copy of the fixtures, so a test can edit or remove a file."""
    target = tmp_path / "fixtures"
    shutil.copytree(FIXTURE_DIR, target)
    return target


@pytest.fixture
def indexed_db(work_dir: Path, tmp_path: Path) -> Path:
    """A database holding one full index of `work_dir`, with its HNSW index."""
    db_dir = tmp_path / "db"
    first = index_with_rust(work_dir, db_dir, incremental=True)
    assert not first.errors, f"Unexpected errors: {first.errors}"
    assert first.chunks_written > 0, "the first run should index the fixtures"
    assert _hnsw_index_names(db_dir), "expected an HNSW index after the first run"
    return db_dir


class TestHnswKeptOnEmptyDiff:
    """Only a run that writes, deletes or compacts may rebuild the HNSW index."""

    def test_empty_incremental_run_keeps_the_index(
        self, work_dir: Path, indexed_db: Path
    ):
        names_before = _hnsw_index_names(indexed_db)
        counts_before = collect_table_counts(indexed_db)

        second, events = _run(work_dir, indexed_db)

        assert second.files_processed == 0, "nothing changed, so nothing is processed"
        assert second.chunks_written == 0
        assert LEFT_IN_PLACE in events, (
            "an incremental run with an empty diff rebuilt the HNSW index"
        )
        assert REBUILT not in events
        assert "write-compact" not in {phase for phase, _, _ in events}
        assert _hnsw_index_names(indexed_db) == names_before
        assert collect_table_counts(indexed_db) == counts_before

    def test_run_that_writes_keeps_the_bracket(self, work_dir: Path, indexed_db: Path):
        main_py = work_dir / "main.py"
        main_py.write_text(
            main_py.read_text() + "\n\ndef added_for_hnsw_test():\n    return 431\n"
        )

        second, events = _run(work_dir, indexed_db)

        assert second.files_processed == 1
        assert REBUILT in events
        assert LEFT_IN_PLACE not in events
        assert _hnsw_index_names(indexed_db), "the index must exist after a write"

    def test_non_incremental_run_keeps_the_bracket(
        self, work_dir: Path, indexed_db: Path
    ):
        """Unchanged files, but a full run writes every one of them again."""
        second, events = _run(work_dir, indexed_db, incremental=False)

        assert second.files_processed > 0
        assert REBUILT in events
        assert LEFT_IN_PLACE not in events
        assert _hnsw_index_names(indexed_db)

    def test_run_that_only_deletes_keeps_the_bracket(
        self, work_dir: Path, indexed_db: Path
    ):
        """No changed files, but an orphan to delete: rows change, so bracket."""
        (work_dir / "notes.md").unlink()

        second, events = _run(work_dir, indexed_db)

        assert second.files_processed == 0
        assert REBUILT in events
        assert LEFT_IN_PLACE not in events
        assert _hnsw_index_names(indexed_db)

    def test_empty_run_that_compacts_keeps_the_bracket(
        self, work_dir: Path, indexed_db: Path
    ):
        """Compaction rebuilds the index with the metrics the drop captures."""
        second, events = _run(
            work_dir, indexed_db, compaction_threshold=0.0, compaction_min_size_mb=0
        )

        phases = {phase for phase, _, _ in events}
        assert second.files_processed == 0
        assert "write-compact" in phases
        assert "write-index" not in phases
        assert _hnsw_index_names(indexed_db)

    def test_empty_run_restores_an_index_a_crash_left_missing(
        self, work_dir: Path, indexed_db: Path
    ):
        """Skipping the bracket must not skip crash recovery."""
        names_before = _hnsw_index_names(indexed_db)
        _drop_hnsw_indexes(indexed_db)
        assert not _hnsw_index_names(indexed_db), (
            "the simulated crash should leave no index"
        )

        second, events = _run(work_dir, indexed_db)

        assert second.files_processed == 0
        assert LEFT_IN_PLACE in events
        assert _hnsw_index_names(indexed_db) == names_before, (
            "an empty run must still recreate an index that is missing"
        )

"""Contract test: the Rust pipeline enforces disk_usage_limit_mb mid-run.

Mirrors `tests/unit/test_disk_usage_limit.py`'s Python-path contract
(`test_store_parsed_results_disk_limit_exceeded`) for the Rust pipeline:
`disk_usage_limit_mb=0.0` deterministically trips on the very first batch
(any non-empty DB file already has size >= 0.0), so no rows get written and
`PipelineReport.disk_limit_exceeded` is set — as data, never a raised
exception.
"""

import tempfile
from pathlib import Path

import duckdb
import pytest

from tests.contracts.mock_embed import MOCK_DIMS
from tests.contracts.pipeline_harness import (
    IndexResult,
    files_table_paths,
    hnsw_index_names,
    index_with_rust,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _write_many_fixture_files(project_dir: Path, count: int) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (project_dir / f"mod_{i}.py").write_text(f"def fn_{i}():\n    return {i}\n")


def _disk_limit_trip(
    db_dir: Path, project_dir: Path = FIXTURE_DIR, *, skip_embeddings: bool = True
) -> tuple[IndexResult, list[str]]:
    """Force compaction eligibility but stop writes at the disk limit."""
    phases: list[str] = []
    result = index_with_rust(
        project_dir,
        db_dir,
        skip_embeddings=skip_embeddings,
        incremental=True,
        disk_usage_limit_mb=0.0,
        compaction_threshold=0.0,
        compaction_min_size_mb=0,
        progress_callback=lambda phase, current, total: phases.append(phase),
    )
    assert result.disk_limit_exceeded is True
    assert not any(phase.startswith("write-compact") for phase in phases)
    return result, phases


def _seed_indexed_project(tmp_path: Path) -> tuple[Path, Path, IndexResult]:
    project_dir, db_dir = tmp_path / "project", tmp_path / "db"
    _write_many_fixture_files(project_dir, count=5)
    before = index_with_rust(project_dir, db_dir)
    assert before.chunks_written > 0
    assert before.embedding_tuples
    assert hnsw_index_names(db_dir)
    return project_dir, db_dir, before


def _assert_hnsw_search_usable(db_dir: Path) -> None:
    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        conn.execute("LOAD vss")
        chunk_id, vector = conn.execute(
            f"SELECT chunk_id, embedding FROM embeddings_{MOCK_DIMS} LIMIT 1"
        ).fetchone()
        query = (
            "SELECT chunk_id, "
            f"array_cosine_distance(embedding, ?::FLOAT[{MOCK_DIMS}]) AS distance "
            f"FROM embeddings_{MOCK_DIMS} ORDER BY distance LIMIT 1"
        )
        plan = conn.execute("EXPLAIN " + query, [vector]).fetchall()
        assert "HNSW_INDEX_SCAN" in str(plan)
        found_id, distance = conn.execute(query, [vector]).fetchone()
        assert found_id == chunk_id
        assert distance == pytest.approx(0.0, abs=1e-6)
    finally:
        conn.close()


class TestDiskUsageLimitRust:
    @pytest.mark.asyncio
    async def test_zero_limit_stops_all_writes_without_raising(self):
        """disk_usage_limit_mb=0.0 must stop writes and report the trip as data."""
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "db"

            # No exception should propagate out of index_with_rust()/pipeline.run() —
            # Python's own DiskUsageLimitExceededError contract is a returned
            # status, never a raised exception, and the Rust mirror must match.
            result = index_with_rust(
                FIXTURE_DIR, db_dir, skip_embeddings=True, disk_usage_limit_mb=0.0
            )

            assert result.disk_limit_exceeded is True
            assert result.disk_limit_current_mb is not None
            assert result.disk_limit_max_mb == 0.0
            assert result.chunks_written == 0, (
                "no chunks should be written once the very first pre-write check trips"
            )

    @pytest.mark.asyncio
    async def test_no_limit_configured_writes_normally(self):
        """disk_usage_limit_mb=None (the default) must not affect a normal run."""
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "db"

            result = index_with_rust(FIXTURE_DIR, db_dir, skip_embeddings=True)

            assert result.disk_limit_exceeded is False
            assert result.disk_limit_current_mb is None
            assert result.disk_limit_max_mb is None
            assert result.chunks_written > 0

    @pytest.mark.asyncio
    async def test_disk_limit_trip_skips_compaction(self):
        """A disk-limit trip must never trigger compaction.

        Compaction's EXPORT/IMPORT rewrite transiently roughly doubles disk
        usage — the opposite of what a disk-limit trip should do. Forces
        compaction eligibility (compaction_threshold=0.0,
        compaction_min_size_mb=0 -- the same technique as
        test_compaction_before_index.py) plus disk_usage_limit_mb=0.0, so
        without the fix a `write-compact*` phase would fire.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _disk_limit_trip(Path(tmp) / "db")

    @pytest.mark.hnsw
    @pytest.mark.asyncio
    async def test_disk_limit_trip_preserves_data_and_searchable_index(self, tmp_path):
        """A rejected update keeps persisted data and a usable HNSW index."""
        project_dir, db_dir, before = _seed_indexed_project(tmp_path)
        (project_dir / "mod_0.py").write_text("def changed():\n    return 999\n")
        result, phases = _disk_limit_trip(db_dir, project_dir, skip_embeddings=False)
        assert result.chunks_written == 0
        assert result.chunk_tuples == before.chunk_tuples
        assert result.embedding_tuples == before.embedding_tuples
        assert files_table_paths(db_dir) == {row[0] for row in before.chunk_tuples}
        assert "write-index" in phases
        assert "write-index-unavailable" not in phases
        assert hnsw_index_names(db_dir)
        _assert_hnsw_search_usable(db_dir)

    @pytest.mark.asyncio
    async def test_disk_limit_trip_drains_in_flight_batches_without_hanging(self):
        """A trip must cleanly cancel every batch already queued upstream,
        not just the trivial single-batch case the other tests above cover.

        The standard 5-file fixture with the default parse_batch_size=200
        always collapses into exactly one batch, so store's `drop(store_rx)`
        on trip has nothing queued to cancel -- the cooperative-shutdown
        cascade (embed's `store_tx.send()` failing, which drops `parse_rx`,
        which fails parse's `parse_tx.send()`) never actually exercises
        multiple in-flight batches contending on the bounded channels. Here,
        many files with parse_batch_size=1 force dozens of concurrent
        batches; disk_usage_limit_mb=0.0 trips before the store thread's very
        first receive (backend.open() already leaves a non-empty chunks.db).
        If the shutdown cascade were broken -- e.g. the drop happening too
        late, or not at all -- parse/embed would block forever on a full
        store_tx (capacity 8) with nothing draining it, and this test would
        hang instead of completing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            _write_many_fixture_files(project_dir, count=50)
            db_dir = Path(tmp) / "db"

            result = index_with_rust(
                project_dir,
                db_dir,
                skip_embeddings=True,
                parse_batch_size=1,
                disk_usage_limit_mb=0.0,
            )

            assert result.disk_limit_exceeded is True
            assert result.chunks_written == 0, (
                "the trip fires before the store thread's first receive, so "
                "none of the ~50 in-flight batches should land"
            )

    @pytest.mark.asyncio
    async def test_disk_limit_trip_still_deletes_orphans(self):
        """A pre-recv disk-limit trip must still apply orphan deletes.

        The limit exists to stop inserts (growth). Deleting rows for files
        removed from disk frees space and must not be discarded when store
        aborts the insert loop and drops store_rx.
        """
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir()
            keep = project_dir / "keep.py"
            orphan = project_dir / "orphan.py"
            keep.write_text("def keep():\n    return 1\n")
            orphan.write_text("def orphan():\n    return 1\n")
            db_dir = Path(tmp) / "db"

            first = index_with_rust(project_dir, db_dir, skip_embeddings=True)
            assert first.disk_limit_exceeded is False
            assert first.chunks_written > 0
            assert files_table_paths(db_dir) == {"keep.py", "orphan.py"}

            orphan.unlink()

            result = index_with_rust(
                project_dir,
                db_dir,
                skip_embeddings=True,
                incremental=True,
                disk_usage_limit_mb=0.0,
            )

            assert result.disk_limit_exceeded is True
            assert result.chunks_written == 0, (
                "the trip fires before insert recv, so no replacement rows "
                "should be written"
            )
            paths = files_table_paths(db_dir)
            assert "orphan.py" not in paths, (
                "orphan deletes must run even when the DB is already over "
                "the disk limit"
            )
            assert "keep.py" in paths

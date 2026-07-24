"""Phase 1: Identical chunks — Python vs Rust pipeline.

Indexes the fixture directory with both the Python and Rust pipelines
(skip_embeddings=True) and asserts byte-identical chunk tuples.

This test MUST FAIL until the Rust IndexingPipeline is implemented.
"""

import duckdb
import pytest
import tempfile
from pathlib import Path

from tests.contracts.pipeline_harness import (
    IndexResult,
    assert_identical,
    index_with_python,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


async def _index_with_rust(
    fixture_dir: Path,
    db_dir: Path,
    *,
    skip_embeddings: bool = True,
) -> IndexResult:
    """Index *fixture_dir* using the Rust pipeline.

    Returns an IndexResult suitable for comparison.

    Raises NotImplementedError until the Rust pipeline is available.
    """
    try:
        from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
    except ImportError:
        raise NotImplementedError(
            "Rust IndexingPipeline is not yet available in chunkhound_native. "
            "Implement Phase 1 scaffolding first."
        ) from None

    # Build PipelineConfig.
    config_dict = {
        "project_root": str(fixture_dir.resolve()),
        "db_path": str(db_dir.resolve()),
        "db_batch_size": 100,
        "compaction_threshold": 0.60,
        "compaction_batch_threshold": 10,
        "compaction_min_size_mb": 10,
        "parse_batch_size": 200,
        "parse_thread_pool_size": 4,
        "embed_batch_size": 200,
        "force_reindex": False,
        "mtime_epsilon_seconds": 0.01,
        "skip_cleanup": False,
        "skip_embeddings": skip_embeddings,
        "per_file_timeout_secs": 3.0,
        "per_file_timeout_min_size_kb": 128,
        "detect_embedded_sql": True,
        "config_file_size_threshold_kb": 20,
        "embedding_provider": "",
        "embedding_model": "",
    }

    pipeline = IndexingPipeline(config_dict)

    # Discover files via glob (matches what the Python pipeline finds).
    files = sorted(fixture_dir.resolve().glob("*"))
    file_paths = [str(f) for f in files if f.is_file()]

    from chunkhound.pipeline_bridge import parse_batch_callback

    report = pipeline.run(
        files=file_paths,
        parse_batch_callback=parse_batch_callback,
        embed_batch_callback=None,
        progress_callback=None,
    )

    # Collect chunk tuples directly from the Rust-backed DB.
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    rows = conn.execute(
        """
        SELECT
            f.path AS file_path,
            c.chunk_type,
            c.symbol,
            c.code,
            c.start_line,
            c.end_line
        FROM chunks c
        JOIN files f ON f.id = c.file_id
        ORDER BY f.path, c.start_line, c.symbol
        """
    ).fetchall()
    conn.close()

    chunk_tuples = []
    for row in rows:
        chunk_tuples.append(
            (
                str(row[0]),
                str(row[1]),
                str(row[2] or ""),
                str(row[3] or ""),
                int(row[4] or 0),
                int(row[5] or 0),
            )
        )

    return IndexResult(
        files_processed=report.files_processed,
        chunks_written=report.chunks_written,
        embeddings_generated=report.embeddings_generated,
        chunk_tuples=chunk_tuples,
        errors=list(report.errors) if report.errors else [],
    )


class TestIdenticalChunks:
    """Python and Rust pipelines must produce identical chunk output."""

    @pytest.mark.asyncio
    async def test_identical_chunks_no_embeddings(self):
        """Index the fixture with both pipelines → identical chunk tuples."""
        with tempfile.TemporaryDirectory() as tmp_py, tempfile.TemporaryDirectory() as tmp_rs:
            db_py = Path(tmp_py) / "db"
            db_py.mkdir(parents=True, exist_ok=True)
            db_rs = Path(tmp_rs) / "db"
            db_rs.mkdir(parents=True, exist_ok=True)

            result_py: IndexResult = await index_with_python(
                FIXTURE_DIR, db_py, skip_embeddings=True
            )

            try:
                result_rs: IndexResult = await _index_with_rust(
                    FIXTURE_DIR, db_rs, skip_embeddings=True
                )
            except NotImplementedError as e:
                pytest.fail(
                    f"Rust pipeline not implemented yet: {e}\n"
                    "This is expected in Phase 1 — implement IndexingPipeline to fix."
                )

            assert_identical(result_py, result_rs)
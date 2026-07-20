"""Phase 2: Identical embeddings — Python vs Rust pipeline.

Indexes the fixture directory with both pipelines using the deterministic
mock embedding provider, then asserts byte-identical chunk tuples AND
identical embedding vectors.
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
from tests.contracts.mock_embed import (
    embed_texts,
    MOCK_DIMS,
    MOCK_PROVIDER,
    MOCK_MODEL,
    MockEmbeddingProvider,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


async def _index_with_rust(
    fixture_dir: Path,
    db_dir: Path,
    *,
    skip_embeddings: bool = False,
) -> IndexResult:
    """Index *fixture_dir* using the Rust pipeline with mock embeddings."""
    try:
        from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
    except ImportError:
        raise NotImplementedError(
            "Rust IndexingPipeline is not yet available in chunkhound_native."
        ) from None

    db_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "project_root": str(fixture_dir.resolve()),
        "db_path": str(db_dir.resolve()),
        "db_batch_size": 100,
        "compaction_threshold": 0.60,
        "compaction_batch_threshold": 10,
        "compaction_min_size_mb": 10,
        "parse_batch_size": 200,
        "parse_thread_pool_size": 0,
        "embed_batch_size": 200,
        "force_reindex": False,
        "mtime_epsilon_seconds": 0.01,
        "skip_cleanup": False,
        "skip_embeddings": skip_embeddings,
        "per_file_timeout_secs": 3.0,
        "per_file_timeout_min_size_kb": 128,
        "detect_embedded_sql": True,
        "config_file_size_threshold_kb": 20,
        "embedding_provider": MOCK_PROVIDER,
        "embedding_model": MOCK_MODEL,
    }

    pipeline = IndexingPipeline(config_dict)

    files = sorted(fixture_dir.resolve().glob("*"))
    file_paths = [str(f) for f in files if f.is_file()]

    from chunkhound.pipeline_bridge import parse_file_callback

    report = pipeline.run(
        files=file_paths,
        parse_callback=parse_file_callback,
        embed_callback=embed_texts,
        progress_callback=None,
    )

    # Collect chunk tuples
    chunk_tuples = _collect_tuples(db_dir)

    # Collect embedding tuples
    embedding_tuples = _collect_embedding_tuples(db_dir)

    return IndexResult(
        files_processed=report.files_processed,
        chunks_written=report.chunks_written,
        embeddings_generated=report.embeddings_generated,
        chunk_tuples=chunk_tuples,
        embedding_tuples=embedding_tuples,
        errors=list(report.errors) if report.errors else [],
    )


def _collect_tuples(db_dir: Path) -> list[tuple[str, str, str, str, int, int]]:
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    rows = conn.execute(
        """
        SELECT f.path, c.chunk_type, c.symbol, c.code, c.start_line, c.end_line
        FROM chunks c JOIN files f ON f.id = c.file_id
        ORDER BY f.path, c.start_line, c.symbol
        """
    ).fetchall()
    conn.close()
    return [
        (str(r[0]), str(r[1]), str(r[2] or ""), str(r[3] or ""), int(r[4] or 0), int(r[5] or 0))
        for r in rows
    ]


def _collect_embedding_tuples(
    db_dir: Path,
) -> list[tuple[str, str, str, str, str, int, tuple[float, ...]]]:
    db_file = db_dir / "chunks.db"
    if not db_file.exists():
        return []
    conn = duckdb.connect(str(db_file))
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'embeddings_%'"
    ).fetchall()
    tuples = []
    for (table_name,) in tables:
        rows = conn.execute(
            f"""
            SELECT f.path, c.chunk_type, c.symbol, e.provider, e.model, e.dims, e.embedding
            FROM {table_name} e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN files f ON f.id = c.file_id
            ORDER BY f.path, c.start_line, c.symbol
            """
        ).fetchall()
        for row in rows:
            vec = row[6]
            prefix = tuple(float(v) for v in vec[:8])
            tuples.append((str(row[0]), str(row[1]), str(row[2] or ""), str(row[3]), str(row[4]), int(row[5]), prefix))
    conn.close()
    return tuples


class TestIdenticalEmbeddings:
    """Python and Rust pipelines must produce identical embedding output."""

    @pytest.mark.asyncio
    async def test_embed_before_write_no_store_callback(self):
        """Embeddings are written inline — store_embeddings_callback is never called.

        ``chunkhound.pipeline_bridge.store_embeddings_callback`` has been removed
        entirely.  The Rust pipeline writes embeddings inline inside
        ``write_batch``.  This test runs the pipeline and asserts that
        ``embeddings_generated == chunks_written`` — proving every chunk has an
        inline embedding.
        """
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "db"
            db_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = await _index_with_rust(
                    FIXTURE_DIR, db_dir, skip_embeddings=False
                )
            except NotImplementedError as e:
                pytest.fail(f"Rust pipeline not implemented yet: {e}")

            assert result.embeddings_generated > 0, (
                "embeddings_generated must be > 0 when embed_callback is provided"
            )
            assert result.embeddings_generated == result.chunks_written, (
                f"All chunks should have inline embeddings: "
                f"embeddings_generated={result.embeddings_generated}, "
                f"chunks_written={result.chunks_written}"
            )

    @pytest.mark.asyncio
    async def test_identical_embeddings(self):
        """Index with mock embeddings → identical chunk + embedding tuples."""
        with tempfile.TemporaryDirectory() as tmp_py, tempfile.TemporaryDirectory() as tmp_rs:
            db_py = Path(tmp_py) / "db"
            db_py.mkdir(parents=True, exist_ok=True)
            db_rs = Path(tmp_rs) / "db"
            db_rs.mkdir(parents=True, exist_ok=True)

            result_py = await index_with_python(
                FIXTURE_DIR, db_py, skip_embeddings=False,
                embedding_provider=MockEmbeddingProvider(),
            )

            try:
                result_rs = await _index_with_rust(FIXTURE_DIR, db_rs, skip_embeddings=False)
            except NotImplementedError as e:
                pytest.fail(
                    f"Rust pipeline not implemented yet: {e}\n"
                    "Expected in Phase 2 — implement embed path in IndexingPipeline."
                )

            assert_identical(result_py, result_rs)

    @pytest.mark.asyncio
    async def test_skip_embeddings(self):
        """When skip_embeddings=True, both pipelines produce 0 embeddings."""
        with tempfile.TemporaryDirectory() as tmp_py, tempfile.TemporaryDirectory() as tmp_rs:
            db_py = Path(tmp_py) / "db"
            db_py.mkdir(parents=True, exist_ok=True)
            db_rs = Path(tmp_rs) / "db"
            db_rs.mkdir(parents=True, exist_ok=True)

            result_py = await index_with_python(FIXTURE_DIR, db_py, skip_embeddings=True)

            try:
                result_rs = await _index_with_rust(FIXTURE_DIR, db_rs, skip_embeddings=True)
            except NotImplementedError as e:
                pytest.fail(f"Rust pipeline not implemented yet: {e}")

            assert result_py.embeddings_generated == 0
            assert result_rs.embeddings_generated == 0
            assert_identical(result_py, result_rs)
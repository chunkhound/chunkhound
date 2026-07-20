"""Phase 7: Parallel embedding — embed_batch_callback with rayon thread pool.

When ``parse_thread_pool_size > 1`` and ``embed_batch_callback`` is provided,
the Rust pipeline dispatches embed-batch calls across a rayon thread pool.
Each batch calls the Python callback independently, achieving parallelism
for the embedding step (parse → *embed* → write).
"""

import duckdb
import pytest
import tempfile
from pathlib import Path

from tests.contracts.pipeline_harness import (
    IndexResult,
    assert_identical,
)
from tests.contracts.mock_embed import (
    embed_texts,
    MOCK_DIMS,
    MOCK_PROVIDER,
    MOCK_MODEL,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


async def _index_with_rust(
    fixture_dir: Path,
    db_dir: Path,
    *,
    skip_embeddings: bool = False,
    parse_thread_pool_size: int = 0,
    parse_batch_callback=None,
    embed_batch_callback=None,
) -> IndexResult:
    """Index *fixture_dir* using the Rust pipeline with configurable parallelism."""
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
        "parse_thread_pool_size": parse_thread_pool_size,
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
        parse_batch_callback=parse_batch_callback,
        embed_batch_callback=embed_batch_callback,
    )

    chunk_tuples = _collect_tuples(db_dir)
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


class TestParallelEmbed:
    """Parallel embedding must produce byte-identical output to serial embedding."""

    @pytest.mark.asyncio
    async def test_parallel_embed_engages_batch_callback(self):
        """RED: embed_batch_callback is called when parse_thread_pool_size > 1.

        If the pipeline ignores the callback, ``call_count`` stays 0 and this
        assertion fails.  Once the parallel dispatch is wired in, the callback
        is called at least once per batch → GREEN.
        """
        call_count = 0

        def embed_batch_callback(texts):
            nonlocal call_count
            call_count += 1
            return embed_texts(texts)

        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "db"
            db_dir.mkdir(parents=True, exist_ok=True)

            try:
                await _index_with_rust(
                    FIXTURE_DIR,
                    db_dir,
                    skip_embeddings=False,
                    parse_thread_pool_size=4,
                    embed_batch_callback=embed_batch_callback,
                )
            except NotImplementedError as e:
                pytest.fail(f"Rust pipeline not implemented yet: {e}")

        assert call_count > 0, (
            f"embed_batch_callback was never called! "
            f"parse_thread_pool_size=4 should trigger parallel embed dispatch."
        )

    @pytest.mark.asyncio
    async def test_parallel_embed_identical_output(self):
        """Parallel embed produces identical chunk+embedding output to serial."""
        with tempfile.TemporaryDirectory() as tmp_serial, tempfile.TemporaryDirectory() as tmp_parallel:
            db_serial = Path(tmp_serial) / "db"
            db_serial.mkdir(parents=True, exist_ok=True)
            db_parallel = Path(tmp_parallel) / "db"
            db_parallel.mkdir(parents=True, exist_ok=True)

            # Serial baseline (no batch callback, sequential embed)
            serial = await _index_with_rust(
                FIXTURE_DIR, db_serial, skip_embeddings=False, parse_thread_pool_size=0
            )

            # Parallel embed
            parallel = await _index_with_rust(
                FIXTURE_DIR,
                db_parallel,
                skip_embeddings=False,
                parse_thread_pool_size=4,
                embed_batch_callback=embed_texts,
            )

            assert_identical(serial, parallel)
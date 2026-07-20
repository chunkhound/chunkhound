"""Phase 8: Pipeline parallelism — overlapping parse, embed, and write stages.

When ``pipeline_parallel=True`` and ``parse_thread_pool_size > 1``, the
Rust pipeline runs parse + embed + write in a producer-consumer chain
connected by channels.  Parse threads produce parsed chunks, embed threads
consume and embed them, and the main thread writes embedded batches to
DuckDB using per-batch transactions.

The contract: output must be byte-identical to the sequential pipeline
(pipeline_parallel=False).
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
    pipeline_parallel: bool = False,
    incremental: bool = False,
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
        "pipeline_parallel": pipeline_parallel,
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
        incremental=incremental,
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


class TestPipelineParallel:
    """Pipeline-parallel mode must produce byte-identical output to sequential."""

    @pytest.mark.asyncio
    async def test_pipeline_parallel_identical_output(self):
        """RED: pipeline_parallel=True produces identical chunks + embeddings.

        When the pipeline runs in parallel mode (parse → channel → embed →
        channel → write), the final DB must contain the exact same chunks and
        embeddings as the sequential pipeline.
        """
        with tempfile.TemporaryDirectory() as tmp_seq, tempfile.TemporaryDirectory() as tmp_par:
            db_seq = Path(tmp_seq) / "db"
            db_seq.mkdir(parents=True, exist_ok=True)
            db_par = Path(tmp_par) / "db"
            db_par.mkdir(parents=True, exist_ok=True)

            # Sequential baseline
            sequential = await _index_with_rust(
                FIXTURE_DIR,
                db_seq,
                skip_embeddings=False,
                parse_thread_pool_size=0,
                pipeline_parallel=False,
            )

            # Parallel pipeline (parse+embed+write overlapped)
            from chunkhound.pipeline_bridge import parse_batch_callback

            parallel = await _index_with_rust(
                FIXTURE_DIR,
                db_par,
                skip_embeddings=False,
                parse_thread_pool_size=4,
                parse_batch_callback=parse_batch_callback,
                embed_batch_callback=embed_texts,
                pipeline_parallel=True,
            )

            assert_identical(sequential, parallel)

    @pytest.mark.asyncio
    async def test_pipeline_parallel_incremental(self):
        """RED: pipeline_parallel=True works with incremental updates.

        After an initial index, modifying a file and re-indexing with
        pipeline_parallel=True must detect the change and produce the
        correct chunk count (26 for the fixture after one file edit).
        """
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            shutil.copytree(FIXTURE_DIR, work_dir, dirs_exist_ok=True)

            db_dir = Path(tmp) / "db"
            db_dir.mkdir(parents=True, exist_ok=True)

            # Initial index (sequential, to establish baseline).
            initial = await _index_with_rust(
                work_dir,
                db_dir,
                skip_embeddings=False,
                parse_thread_pool_size=0,
                pipeline_parallel=False,
            )

            # Sanity: 5 files, 25 chunks for the standard fixture.
            assert initial.chunks_written == 25, (
                f"Expected 25 chunks from initial index, got {initial.chunks_written}"
            )

            # Modify main.py: append a function.
            main_py = work_dir / "main.py"
            assert main_py.exists(), "main.py missing from fixture copy"
            original = main_py.read_text()
            main_py.write_text(
                original
                + "\n\ndef incremental_test_func():\n    return 'added in phase 8'\n"
            )

            # Incremental re-index with pipeline_parallel=True.
            from chunkhound.pipeline_bridge import parse_batch_callback

            incremental = await _index_with_rust(
                work_dir,
                db_dir,
                skip_embeddings=False,
                parse_thread_pool_size=4,
                parse_batch_callback=parse_batch_callback,
                embed_batch_callback=embed_texts,
                pipeline_parallel=True,
                incremental=True,
            )

            # After editing one file, the incremental DB must contain all chunks
            # from the full re-index.  The chunks_written field counts only newly
            # written chunks (the changed file).  Verify via DB content.
            #
            # Do a full sequential re-index on a fresh DB for comparison.
            db_full = Path(tmp) / "db_full"
            db_full.mkdir(parents=True, exist_ok=True)
            full_result = await _index_with_rust(
                work_dir,
                db_full,
                skip_embeddings=False,
                parse_thread_pool_size=0,
                pipeline_parallel=False,
            )

            full_chunks = set(full_result.chunk_tuples)
            inc_chunks = set(incremental.chunk_tuples)
            missing = full_chunks - inc_chunks
            extra = inc_chunks - full_chunks
            assert not missing, f"Chunks missing from incremental ({len(missing)}): {sorted(missing)[:3]}"
            assert not extra, f"Extra chunks in incremental ({len(extra)}): {sorted(extra)[:3]}"

            # Verify incremental mode was actually used (fewer files processed).
            assert incremental.files_processed < full_result.files_processed, (
                f"Incremental should process fewer files: "
                f"inc={incremental.files_processed} < full={full_result.files_processed}"
            )
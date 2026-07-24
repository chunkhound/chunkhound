"""Contract test: embed callback raises for every batch.

Design's `test_embed_api_failure` contract: an embed callback that raises is
caught per-batch inside the Rust pipeline (`embed_batch_parallel`); chunks are
still stored with `embedding=NULL`, and the pipeline continues rather than
aborting. This already matches the design with no production changes needed —
this test just exercises it.

Note: `PipelineReport.errors` is not yet wired for embed failures (a known,
separately-tracked gap — the embed thread only `log::warn!`s on failure), so
this test does not assert on `report.errors`.
"""

import duckdb
import pytest
import tempfile
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def failing_embed(texts: list[str]) -> list[list[float]]:
    """Always raises — simulates a total embed API outage (HTTP 500/timeout)."""
    raise RuntimeError("simulated embed API failure")


def _rust_config(project_root: Path, db_dir: Path) -> dict:
    return {
        "project_root": str(project_root.resolve()),
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
        "skip_embeddings": False,
        "per_file_timeout_secs": 3.0,
        "per_file_timeout_min_size_kb": 128,
        "detect_embedded_sql": True,
        "config_file_size_threshold_kb": 20,
        "embedding_provider": "mock-fail",
        "embedding_model": "mock-fail-v1",
    }


def _db_counts(db_dir: Path) -> dict:
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    try:
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
        embeddings = 0
        for (table_name,) in tables:
            embeddings += conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return {"chunks": chunks, "embeddings": embeddings}
    finally:
        conn.close()


class TestEmbedApiFailure:
    """A total embed API failure must not crash the pipeline."""

    @pytest.mark.asyncio
    async def test_embed_failure_stores_chunks_without_embeddings(self):
        try:
            from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
        except ImportError:
            pytest.fail(
                "Rust IndexingPipeline is not yet available in chunkhound_native."
            )
        from chunkhound.pipeline_bridge import parse_batch_callback

        with tempfile.TemporaryDirectory() as tmp_db:
            db_dir = Path(tmp_db) / "db"
            db_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(FIXTURE_DIR.resolve().glob("*"))
            file_paths = [str(f) for f in files if f.is_file()]

            pipeline = IndexingPipeline(_rust_config(FIXTURE_DIR, db_dir))

            # Must not raise — the failure is caught per-batch inside
            # embed_batch_parallel, not propagated to the caller.
            report = pipeline.run(
                files=file_paths,
                parse_batch_callback=parse_batch_callback,
                embed_batch_callback=failing_embed,
                progress_callback=None,
                incremental=False,
            )

            assert report.chunks_written > 0, "Chunks should still be stored"
            assert report.embeddings_generated == 0, (
                "No vectors should be attached when every embed call fails"
            )

            counts = _db_counts(db_dir)
            assert counts["chunks"] > 0
            assert counts["embeddings"] == 0, (
                "No embedding rows should exist when every embed call fails"
            )

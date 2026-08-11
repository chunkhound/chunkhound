"""Contract test: embed callback raises for every batch, or returns too few vectors.

Design's `test_embed_api_failure` contract: an embed callback that raises is
caught per-batch inside the Rust pipeline (`embed_batch_parallel`); chunks are
still stored with `embedding=NULL`, and the pipeline continues rather than
aborting. The failure is also surfaced in `report.errors` (one entry per
affected file) instead of only reaching a log line, so callers can detect
that some chunks were written without embeddings.

The same contract applies to a well-formed but *short* response (a provider
that returns fewer vectors than requested texts without raising) — that
failure mode doesn't go through the exception path at all, so it's covered
by a separate test below.
"""

import tempfile
from pathlib import Path

import pytest

from tests.contracts.pipeline_harness import collect_table_counts, default_rust_config

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def failing_embed(texts: list[str]) -> list[list[float]]:
    """Always raises — simulates a total embed API outage (HTTP 500/timeout)."""
    raise RuntimeError("simulated embed API failure")


class TestEmbedApiFailure:
    """A total embed API failure must not crash the pipeline."""

    @pytest.mark.asyncio
    async def test_embed_failure_stores_chunks_without_embeddings(self):
        try:
            from chunkhound_native import (
                IndexingPipeline,  # type: ignore[import-untyped]
            )
        except ImportError:
            pytest.fail(
                "Rust IndexingPipeline is not yet available in chunkhound_native."
            )
        from chunkhound.pipeline_bridge import parse_batch_callback

        with tempfile.TemporaryDirectory() as tmp_db:
            db_dir = Path(tmp_db) / "db"
            db_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(FIXTURE_DIR.resolve().glob("*"))
            file_entries = [(str(f), f.name) for f in files if f.is_file()]

            pipeline = IndexingPipeline(
                default_rust_config(
                    FIXTURE_DIR,
                    db_dir,
                    skip_embeddings=False,
                    embedding_provider="mock-fail",
                    embedding_model="mock-fail-v1",
                )
            )

            # Must not raise — the failure is caught per-batch inside
            # embed_batch_parallel, not propagated to the caller.
            report = pipeline.run(
                files=file_entries,
                parse_batch_callback=parse_batch_callback,
                embed_batch_callback=failing_embed,
                progress_callback=None,
                incremental=False,
            )

            assert report.chunks_written > 0, "Chunks should still be stored"
            assert report.embeddings_generated == 0, (
                "No vectors should be attached when every embed call fails"
            )
            assert report.errors, (
                "Embed failures must be surfaced in report.errors, not just logged"
            )
            assert all("simulated embed API failure" in err for err in report.errors), (
                f"Every reported error should trace back to the injected failure: {report.errors}"
            )

            counts = collect_table_counts(db_dir)
            assert counts["chunks"] > 0
            assert counts["embeddings"] == 0, (
                "No embedding rows should exist when every embed call fails"
            )


def short_embed(texts: list[str]) -> list[list[float]]:
    """Returns one fewer vector than requested — a truncated response with
    no exception, simulating a provider that silently drops a chunk."""
    return [[0.1, 0.2, 0.3] for _ in range(max(0, len(texts) - 1))]


class TestEmbedShortResponse:
    """A well-formed-but-short embed response must not go unreported."""

    @pytest.mark.asyncio
    async def test_short_response_reports_missing_embedding(self):
        try:
            from chunkhound_native import (
                IndexingPipeline,  # type: ignore[import-untyped]
            )
        except ImportError:
            pytest.fail(
                "Rust IndexingPipeline is not yet available in chunkhound_native."
            )
        from chunkhound.pipeline_bridge import parse_batch_callback

        with tempfile.TemporaryDirectory() as tmp_db:
            db_dir = Path(tmp_db) / "db"
            db_dir.mkdir(parents=True, exist_ok=True)

            files = sorted(FIXTURE_DIR.resolve().glob("*"))
            file_entries = [(str(f), f.name) for f in files if f.is_file()]

            pipeline = IndexingPipeline(
                default_rust_config(
                    FIXTURE_DIR,
                    db_dir,
                    skip_embeddings=False,
                    embedding_provider="mock-short",
                    embedding_model="mock-short-v1",
                )
            )

            report = pipeline.run(
                files=file_entries,
                parse_batch_callback=parse_batch_callback,
                embed_batch_callback=short_embed,
                progress_callback=None,
                incremental=False,
            )

            assert report.chunks_written > 0
            assert report.embeddings_generated == report.chunks_written - 1, (
                "Exactly one chunk should be missing its embedding"
            )
            assert report.errors, (
                "A short (non-exception) embed response must still be "
                "surfaced in report.errors, not just logged"
            )
            assert any("vector(s)" in err for err in report.errors), (
                f"Error should describe the vector/chunk-count mismatch: {report.errors}"
            )

            counts = collect_table_counts(db_dir)
            assert counts["chunks"] == report.chunks_written
            assert counts["embeddings"] == report.embeddings_generated

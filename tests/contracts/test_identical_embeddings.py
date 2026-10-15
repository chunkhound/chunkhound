"""Phase 2: Identical embeddings — Python vs Rust pipeline.

Cross-pipeline identical-output coverage (with and without embeddings, and
under forced multi-batch parsing) lives in
``test_identical_chunks.py::TestIdenticalChunks.test_identical_output`` —
this file covers the one embedding behavior that test doesn't: that the
Rust pipeline never uses a separate store-embeddings callback.
"""

import tempfile
from pathlib import Path

import pytest

from tests.contracts.pipeline_harness import index_with_rust

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


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

            result = index_with_rust(FIXTURE_DIR, db_dir, skip_embeddings=False)

            assert result.embeddings_generated > 0, (
                "embeddings_generated must be > 0 when embed_callback is provided"
            )
            assert result.embeddings_generated == result.chunks_written, (
                f"All chunks should have inline embeddings: "
                f"embeddings_generated={result.embeddings_generated}, "
                f"chunks_written={result.chunks_written}"
            )

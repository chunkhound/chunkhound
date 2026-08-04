"""Phase 2: Identical embeddings — Python vs Rust pipeline.

Indexes the fixture directory with both pipelines using the deterministic
mock embedding provider, then asserts byte-identical chunk tuples AND
identical embedding vectors.
"""

import tempfile
from pathlib import Path

import pytest

from tests.contracts.mock_embed import MockEmbeddingProvider
from tests.contracts.pipeline_harness import (
    assert_identical,
    index_with_python,
    index_with_rust,
)

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

            result_rs = index_with_rust(FIXTURE_DIR, db_rs, skip_embeddings=False)

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

            result_rs = index_with_rust(FIXTURE_DIR, db_rs, skip_embeddings=True)

            assert result_py.embeddings_generated == 0
            assert result_rs.embeddings_generated == 0
            assert_identical(result_py, result_rs)

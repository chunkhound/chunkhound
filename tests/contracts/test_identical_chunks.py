"""Identical output — Python vs Rust pipeline.

Indexes the fixture directory with both the Python and Rust pipelines and
asserts byte-identical chunk (and, where embeddings are on, embedding)
tuples across the scenarios that matter: no embeddings, embeddings via the
deterministic mock provider, and embeddings with parse_batch_size=1 (forces
the Rust pipeline's parse/embed/store stages to overlap — batch N+1 parsing
while batch N is still being embedded/stored — instead of the 5-file
fixture fitting in a single default 200-file batch with nothing to
overlap).
"""

import tempfile
from pathlib import Path

import pytest

from tests.contracts.mock_embed import MockEmbeddingProvider
from tests.contracts.pipeline_harness import (
    IndexResult,
    assert_identical,
    index_with_python,
    index_with_rust,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


class TestIdenticalChunks:
    """Python and Rust pipelines must produce identical output."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "skip_embeddings, parse_batch_size",
        [
            pytest.param(True, None, id="no-embeddings"),
            pytest.param(False, None, id="with-embeddings"),
            pytest.param(False, 1, id="with-embeddings-parse-batch-1"),
        ],
    )
    async def test_identical_output(self, skip_embeddings, parse_batch_size):
        """Index the fixture with both pipelines → identical output."""
        with (
            tempfile.TemporaryDirectory() as tmp_py,
            tempfile.TemporaryDirectory() as tmp_rs,
        ):
            db_py = Path(tmp_py) / "db"
            db_py.mkdir(parents=True, exist_ok=True)
            db_rs = Path(tmp_rs) / "db"
            db_rs.mkdir(parents=True, exist_ok=True)

            python_kwargs = {"skip_embeddings": skip_embeddings}
            if not skip_embeddings:
                python_kwargs["embedding_provider"] = MockEmbeddingProvider()

            rust_kwargs = {"skip_embeddings": skip_embeddings}
            if parse_batch_size is not None:
                rust_kwargs["parse_batch_size"] = parse_batch_size

            result_py: IndexResult = await index_with_python(
                FIXTURE_DIR, db_py, **python_kwargs
            )
            result_rs: IndexResult = index_with_rust(FIXTURE_DIR, db_rs, **rust_kwargs)

            if skip_embeddings:
                assert result_py.embeddings_generated == 0
                assert result_rs.embeddings_generated == 0

            assert_identical(result_py, result_rs)

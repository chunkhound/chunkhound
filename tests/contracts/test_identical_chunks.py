"""Identical chunks — Python vs Rust pipeline.

Indexes the fixture directory with both the Python and Rust pipelines
(skip_embeddings=True) and asserts byte-identical chunk tuples.
"""

import tempfile
from pathlib import Path

import pytest

from tests.contracts.pipeline_harness import (
    IndexResult,
    assert_identical,
    index_with_python,
    index_with_rust,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


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

            result_rs: IndexResult = index_with_rust(
                FIXTURE_DIR, db_rs, skip_embeddings=True
            )

            assert_identical(result_py, result_rs)

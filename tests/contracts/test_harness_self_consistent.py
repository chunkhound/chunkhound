"""Phase 0: Test harness self-consistency.

Indexes the fixture directory twice with the Python pipeline and asserts
byte-identical output. Proves the comparison harness is correct before
any Rust code exists.
"""

import pytest
import tempfile
from pathlib import Path

from tests.contracts.pipeline_harness import (
    IndexResult,
    assert_identical,
    index_with_python,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


class TestHarnessSelfConsistent:
    """The harness must produce identical output on repeated runs."""

    @pytest.mark.asyncio
    async def test_harness_python_self_consistent(self):
        """Index the fixture twice → identical results."""
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            db1 = Path(tmp1) / "db"
            db2 = Path(tmp2) / "db"

            result_a: IndexResult = await index_with_python(
                FIXTURE_DIR, db1, skip_embeddings=True
            )
            result_b: IndexResult = await index_with_python(
                FIXTURE_DIR, db2, skip_embeddings=True
            )

            assert_identical(result_a, result_b)

    @pytest.mark.asyncio
    async def test_fixture_produces_expected_files(self):
        """Verify the fixture directory is indexed as expected."""
        with tempfile.TemporaryDirectory() as tmp:
            result = await index_with_python(
                FIXTURE_DIR, Path(tmp) / "db", skip_embeddings=True
            )

        # The fixture should produce some files and chunks
        assert result.files_processed > 0, "Expected some files to be indexed"
        assert result.chunks_written > 0, "Expected some chunks to be created"
        assert result.embeddings_generated == 0, "Embeddings should be skipped"

        # All fixture files should appear somewhere in the chunk tuples
        expected_files = {"main.py", "lib.rs", "settings.json", "notes.md", "empty.py"}
        found_files = {t[0] for t in result.chunk_tuples}
        # empty.py may produce no chunks — that's expected.
        # At minimum, main.py and settings.json must produce chunks.
        assert "main.py" in found_files, "main.py must produce chunks"
        # lib.rs may or may not appear depending on include patterns (default
        # patterns cover common languages; Rust (.rs) is included).
        assert "lib.rs" in found_files, (
            "lib.rs must produce chunks (default include patterns cover .rs)"
        )
        assert "settings.json" in found_files, (
            "settings.json must produce chunks"
        )
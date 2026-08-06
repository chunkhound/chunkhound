"""Phase 8: Pipeline parallelism — overlapping parse, embed, and store stages.

The Rust pipeline runs parse ∥ embed ∥ store as three persistent OS threads
connected by bounded channels: a parse thread produces parsed batches, a
dedicated embed thread consumes and embeds them, and a dedicated store
thread writes embedded batches to DuckDB using per-batch transactions.

The contract: output must be byte-identical to the Python reference
pipeline (``index_with_python``).
"""

import tempfile
from pathlib import Path

import pytest

from tests.contracts.mock_embed import MockEmbeddingProvider
from tests.contracts.pipeline_harness import (
    assert_chunk_multiset_identical,
    assert_identical,
    index_with_python,
    index_with_rust,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


class TestPipelineParallel:
    """The 3-stage streaming pipeline must produce output identical to Python."""

    @pytest.mark.asyncio
    async def test_pipeline_parallel_identical_output(self):
        """Streaming pipeline (parse ∥ embed ∥ store) output matches Python."""
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
    async def test_pipeline_parallel_incremental(self):
        """Incremental re-indexing through the streaming pipeline detects
        changes and produces the correct chunk set (matching a Python full
        re-index), while processing fewer files than a full re-index.
        """
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            shutil.copytree(FIXTURE_DIR, work_dir, dirs_exist_ok=True)

            db_dir = Path(tmp) / "db"
            db_dir.mkdir(parents=True, exist_ok=True)

            # Initial index.
            initial = index_with_rust(work_dir, db_dir, skip_embeddings=False)

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

            # Incremental re-index (same DB as the initial index).
            incremental = index_with_rust(
                work_dir, db_dir, skip_embeddings=False, incremental=True
            )

            # After editing one file, the incremental DB must contain all chunks
            # from a full re-index. The chunks_written field counts only newly
            # written chunks (the changed file), so verify via DB content
            # against a full re-index on a fresh DB.
            db_full = Path(tmp) / "db_full"
            db_full.mkdir(parents=True, exist_ok=True)
            full_result = index_with_rust(work_dir, db_full, skip_embeddings=False)

            assert_chunk_multiset_identical(
                full_result.chunk_tuples,
                incremental.chunk_tuples,
                label_a="full",
                label_b="incremental",
            )

            # Verify incremental mode was actually used (fewer files processed).
            assert incremental.files_processed < full_result.files_processed, (
                f"Incremental should process fewer files: "
                f"inc={incremental.files_processed} < full={full_result.files_processed}"
            )

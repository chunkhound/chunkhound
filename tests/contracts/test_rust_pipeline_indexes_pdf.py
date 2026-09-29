"""Rust pipeline must store chunks for a PDF, the same way it stores chunks for code.

Existing PDF tests call IndexingCoordinator.process_file with a PDF parser
directly, so they never reach the Rust pipeline. Contract tests such as
test_identical_chunks and test_binary_file_not_reprocessed index a directory
through index_with_rust and require real source files to land as chunks with
no skip reason. A PDF is a supported file type and must be covered the same way.
"""

import shutil
from pathlib import Path

import pytest

from tests.contracts.pipeline_harness import index_with_rust

FIXTURE_PDF = (
    Path(__file__).resolve().parent.parent / "fixtures" / "cast_research_paper.pdf"
)

pytest.importorskip("pymupdf")


class TestRustPipelineIndexesPdf:
    """Indexing a PDF through the Rust pipeline stores its chunks."""

    def test_pdf_file_is_stored_as_chunks(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "project"
        work_dir.mkdir()
        shutil.copy2(FIXTURE_PDF, work_dir / FIXTURE_PDF.name)
        db_dir = tmp_path / "db"

        result = index_with_rust(work_dir, db_dir, skip_embeddings=True)

        pdf_chunks = [
            chunk for chunk in result.chunk_tuples if chunk[0].endswith(".pdf")
        ]
        assert pdf_chunks, (
            f"zero pdf chunks; indexed {FIXTURE_PDF.name} through the Rust "
            f"pipeline and stored {result.chunks_written} chunks"
        )

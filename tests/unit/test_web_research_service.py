"""Contracts for composing in-process research over fetched pages."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import (
    ChunkType,
    FileId,
    Language,
    LineNumber,
)
from chunkhound.embeddings import LocalEmbeddingResult
from chunkhound.services.research.shared.file_reader import resolve_source_text
from chunkhound.services.transient_search_service import TransientSearchService
from chunkhound.services.web_research_service import (
    pages_to_chunks,
    research_web_pages,
)


def test_pages_to_chunks_keeps_source_url_without_writing_files() -> None:
    chunks = pages_to_chunks(
        [("https://example.invalid/docs", ".md", "# Docs\nUseful content")]
    )

    assert chunks
    assert chunks[0].file_path == "https://example.invalid/docs"
    assert chunks[0].code == "# Docs\nUseful content"


@pytest.mark.asyncio
async def test_research_factory_receives_transient_search_service() -> None:
    manager = MagicMock()

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="test",
            provider="test",
            dims=2,
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    research = MagicMock()
    research.deep_research = AsyncMock(return_value={"answer": "ANSWER"})
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return research

    with patch(
        "chunkhound.services.web_research_service.ResearchServiceFactory.create",
        side_effect=create,
    ):
        result = await research_web_pages(
            "query",
            [("https://example.invalid/docs", ".md", "# Docs")],
            MagicMock(),
            manager,
            MagicMock(),
        )

    assert result == {"answer": "ANSWER"}
    assert isinstance(captured["db_services"].search_service, TransientSearchService)


def test_pages_to_chunks_skips_unusable_pdf_error_chunks() -> None:
    error = Chunk(
        symbol="pdf_unavailable",
        start_line=LineNumber(1),
        end_line=LineNumber(1),
        code="PDF parsing not available (PyMuPDF not installed)",
        chunk_type=ChunkType.UNKNOWN,
        file_id=FileId(0),
        language=Language.PDF,
    )

    with patch(
        "chunkhound.services.web_research_service.PDFMapping.parse_pdf_content",
        return_value=[error],
    ):
        assert (
            pages_to_chunks([("https://example.invalid/a.pdf", ".pdf", b"%PDF")]) == []
        )


def test_resolve_source_text_uses_transient_provider() -> None:
    class Provider:
        def get_transient_file_content(self, file_path: str) -> str | None:
            return "full page markdown"

        def get_base_directory(self):
            raise AssertionError("disk should not be consulted")

    assert (
        resolve_source_text(Provider(), "https://example.invalid/docs")
        == "full page markdown"
    )


@pytest.mark.asyncio
async def test_research_consumes_pages_incrementally_and_stores_full_text() -> None:
    live: list[str] = []
    full = "FULL PAGE BODY USED FOR SYNTHESIS"

    async def pages():
        live.append(full)
        yield "https://example.invalid/docs", ".md", full
        live.pop()

    manager = MagicMock()

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="test",
            provider="test",
            dims=2,
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    captured: dict = {}
    research = MagicMock()

    async def deep_research(query: str):
        await captured["db_services"].search_service.search_semantic(query)
        provider = captured["db_services"].provider
        assert (
            provider.get_transient_file_content("https://example.invalid/docs") == full
        )
        return {"answer": "ANSWER"}

    research.deep_research = deep_research

    def create(**kwargs):
        captured.update(kwargs)
        return research

    with patch(
        "chunkhound.services.web_research_service.ResearchServiceFactory.create",
        side_effect=create,
    ):
        result = await research_web_pages(
            "query",
            pages(),
            MagicMock(),
            manager,
            MagicMock(),
        )

    assert result == {"answer": "ANSWER"}
    assert live == []

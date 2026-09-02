"""Contracts for composing in-process research over fetched pages."""

import asyncio
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
    captured: dict = {}

    async def deep_research(query: str):
        await captured["db_services"].search_service.search_semantic(query)
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


def test_pages_to_chunks_skips_pdf_parse_errors() -> None:
    error = Chunk(
        symbol="pdf_parse_error",
        start_line=LineNumber(1),
        end_line=LineNumber(1),
        code="Error parsing PDF: malformed file",
        chunk_type=ChunkType.UNKNOWN,
        file_id=FileId(0),
        language=Language.PDF,
    )

    with patch(
        "chunkhound.services.web_research_service.PDFMapping.parse_pdf_content",
        return_value=[error],
    ):
        assert (
            pages_to_chunks([("https://example.invalid/bad.pdf", ".pdf", b"%PDF")])
            == []
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
        search_service = captured["db_services"].search_service
        first, second = await asyncio.gather(
            search_service.search_semantic(query),
            search_service.search_semantic(f"{query} follow-up"),
        )
        third = await search_service.search_semantic(f"{query} later")
        assert first[0][0]["content"] == full
        assert second[0][0]["content"] == full
        assert third[0][0]["content"] == full
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


@pytest.mark.asyncio
async def test_research_rejects_pages_without_usable_content() -> None:
    error = Chunk(
        symbol="pdf_parse_error",
        start_line=LineNumber(1),
        end_line=LineNumber(1),
        code="Error parsing PDF: malformed file",
        chunk_type=ChunkType.UNKNOWN,
        file_id=FileId(0),
        language=Language.PDF,
    )
    warnings: list[str] = []
    manager = MagicMock()
    manager.embed_texts = AsyncMock(
        return_value=LocalEmbeddingResult(
            embeddings=[[1.0, 0.0]],
            model="test",
            provider="test",
            dims=2,
        )
    )
    captured: dict = {}
    research = MagicMock()

    async def deep_research(query: str):
        await captured["db_services"].search_service.search_semantic(query)
        return {"answer": "must not escape"}

    research.deep_research = deep_research

    def create(**kwargs):
        captured.update(kwargs)
        return research

    with (
        patch(
            "chunkhound.services.web_research_service.ResearchServiceFactory.create",
            side_effect=create,
        ),
        patch(
            "chunkhound.services.web_research_service.PDFMapping.parse_pdf_content",
            return_value=[error],
        ),
        pytest.raises(ValueError, match="No usable page content"),
    ):
        await research_web_pages(
            "query",
            [("https://example.invalid/bad.pdf", ".pdf", b"%PDF")],
            MagicMock(),
            manager,
            MagicMock(),
            warning_callback=warnings.append,
        )

    assert warnings == ["No usable content parsed from https://example.invalid/bad.pdf"]


@pytest.mark.asyncio
async def test_research_propagates_error_after_partial_page_stream() -> None:
    async def pages():
        for index in range(100):
            yield f"https://example.invalid/{index}", ".md", f"page {index}"
        raise RuntimeError("page stream failed")

    manager = MagicMock()
    manager.embed_texts = AsyncMock(
        return_value=LocalEmbeddingResult(
            embeddings=[[1.0, 0.0]],
            model="test",
            provider="test",
            dims=2,
        )
    )
    captured: dict = {}
    observed: list[BaseException] = []
    research = MagicMock()

    async def deep_research(query: str):
        search_service = captured["db_services"].search_service
        concurrent = await asyncio.gather(
            search_service.search_semantic(query),
            search_service.search_semantic(f"{query} concurrent"),
            return_exceptions=True,
        )
        late = await asyncio.gather(
            search_service.search_semantic(f"{query} late"),
            return_exceptions=True,
        )
        observed.extend(
            item for item in [*concurrent, *late] if isinstance(item, BaseException)
        )
        return {"answer": "must not escape"}

    research.deep_research = deep_research

    def create(**kwargs):
        captured.update(kwargs)
        return research

    with (
        patch(
            "chunkhound.services.web_research_service.ResearchServiceFactory.create",
            side_effect=create,
        ),
        pytest.raises(RuntimeError, match="page stream failed"),
    ):
        await research_web_pages(
            "query",
            pages(),
            MagicMock(),
            manager,
            MagicMock(),
        )

    assert len(observed) == 3
    assert all(
        isinstance(error, RuntimeError) and str(error) == "page stream failed"
        for error in observed
    )


@pytest.mark.asyncio
async def test_research_cancellation_finalizes_page_producer() -> None:
    started = asyncio.Event()
    finalized = asyncio.Event()

    async def pages():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()
        yield "https://example.invalid/never", ".md", "never"

    manager = MagicMock()
    manager.embed_texts = AsyncMock(
        return_value=LocalEmbeddingResult(
            embeddings=[[1.0, 0.0]],
            model="test",
            provider="test",
            dims=2,
        )
    )
    captured: dict = {}
    research = MagicMock()

    async def deep_research(query: str):
        await captured["db_services"].search_service.search_semantic(query)
        return {"answer": "must not escape"}

    research.deep_research = deep_research

    def create(**kwargs):
        captured.update(kwargs)
        return research

    with patch(
        "chunkhound.services.web_research_service.ResearchServiceFactory.create",
        side_effect=create,
    ):
        task = asyncio.create_task(
            research_web_pages(
                "query",
                pages(),
                MagicMock(),
                manager,
                MagicMock(),
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert finalized.is_set()

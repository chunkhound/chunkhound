"""Contracts for composing in-process research over fetched pages."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tiktoken

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
from chunkhound.services.vector_cache import VectorCache
from chunkhound.services.web_research_service import (
    pages_to_chunks,
    research_web_pages,
)


@pytest.mark.parametrize("extension", [".md", ".pdf"])
@pytest.mark.parametrize(
    "content",
    [
        "漢字龍" * 3333,
        "漢字龍🧑🏽‍💻🚀abc" * 1000,
        "🧑🏽‍💻" + " \t" * 6000 + "漢字龍",
    ],
    ids=["cjk-single-line", "mixed-emoji-single-line", "whitespace-heavy"],
)
def test_web_chunks_fit_openai_embedding_limit_without_losing_text(
    extension: str, content: str
) -> None:
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    url = f"https://example.invalid/source{extension}"
    parsed = Chunk(
        symbol="page_7_paragraph_2",
        start_line=LineNumber(70),
        end_line=LineNumber(71),
        code=content,
        chunk_type=ChunkType.BLOCK,
        file_id=FileId(0),
        language=Language.PDF,
        parent_header="Chapter 7",
        metadata={"page": 7},
    )
    # The third-party PDF parser boundary may return large semantic chunks.
    with patch(
        "chunkhound.services.web_research_service.PDFMapping.parse_pdf_content",
        return_value=[parsed],
    ):
        chunks = pages_to_chunks(
            [(url, extension, b"%PDF" if extension == ".pdf" else content)]
        )

    assert len(chunks) > 1
    assert "".join(chunk.code for chunk in chunks) == content
    assert len({chunk.symbol for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.file_path == url
        assert len(chunk.code.encode("utf-8")) <= 4800
        assert len(encoding.encode(chunk.code, disallowed_special=())) <= 8191
        if extension == ".pdf":
            assert chunk.symbol.startswith(parsed.symbol)
            assert (chunk.start_line, chunk.end_line) == (70, 71)
            assert chunk.parent_header == parsed.parent_header
            assert chunk.metadata == parsed.metadata
        else:
            assert (chunk.start_line, chunk.end_line) == (1, 1)


@pytest.mark.asyncio
async def test_web_embeddings_keep_full_batches_with_safe_text_and_pdf_chunks(
    monkeypatch,
) -> None:
    from chunkhound.services import web_research_service as web

    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    content = "漢字龍🧑🏽‍💻🚀abc" * 6000
    parsed = Chunk(
        symbol="page_1",
        start_line=LineNumber(1),
        end_line=LineNumber(1),
        code=content,
        chunk_type=ChunkType.BLOCK,
        file_id=FileId(0),
        language=Language.PDF,
    )
    monkeypatch.setattr(web, "_WEB_VECTOR_CACHE", VectorCache(max_entries=0))
    monkeypatch.setattr(web.PDFMapping, "parse_pdf_content", lambda *args: [parsed])
    batches: list[list[str]] = []
    query = "embedding safety"

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        # A real tokenizer checks the provider's per-input constraint at the
        # external embedding boundary; no network or heuristic token counts.
        for text in texts:
            assert len(encoding.encode(text, disallowed_special=())) <= 8191
        if texts != [query]:
            batches.append(texts)
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="text-embedding-3-small",
            provider="openai",
            dims=2,
            embedding_space_identity="test-space",
        )

    def create(**kwargs):
        async def deep_research(query, previous_query=None):
            results, _ = await kwargs["db_services"].search_service.search_semantic(
                query
            )
            assert results
            return {"answer": "done"}

        return MagicMock(deep_research=deep_research)

    monkeypatch.setattr(web.ResearchServiceFactory, "create", create)
    await research_web_pages(
        query,
        [
            ("https://example.invalid/docs", ".md", content),
            ("https://example.invalid/docs.pdf", ".pdf", b"%PDF"),
        ],
        MagicMock(),
        MagicMock(embed_texts=AsyncMock(side_effect=embed)),
        MagicMock(),
    )
    assert len(batches) >= 2
    assert all(len(batch) == 100 for batch in batches[:-1])
    assert 1 <= len(batches[-1]) <= 100
    assert "".join(text for batch in batches for text in batch) == content * 2


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
            embedding_space_identity="test-space",
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    research = MagicMock()
    captured: dict = {}

    async def deep_research(query: str, previous_query: str | None = None):
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
            embedding_space_identity="test-space",
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    captured: dict = {}
    research = MagicMock()

    async def deep_research(query: str, previous_query: str | None = None):
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
            embedding_space_identity="test-space",
        )
    )
    captured: dict = {}
    research = MagicMock()

    async def deep_research(query: str, previous_query: str | None = None):
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
            embedding_space_identity="test-space",
        )
    )
    captured: dict = {}
    observed: list[BaseException] = []
    research = MagicMock()

    async def deep_research(query: str, previous_query: str | None = None):
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
            embedding_space_identity="test-space",
        )
    )
    captured: dict = {}
    research = MagicMock()

    async def deep_research(query: str, previous_query: str | None = None):
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

"""Contracts for composing in-process research over fetched pages."""

import asyncio
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
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
from chunkhound.services.research.shared.file_reader import (
    FileReader,
    resolve_source_text,
)
from chunkhound.services.transient_search_service import TransientSearchService
from chunkhound.services.vector_cache import VectorCache
from chunkhound.services.web_research_service import (
    _bounded_pdf_chunks,
    _parse_pdf_isolated,
    research_web_pages,
)
from tests.helpers.pdf_worker_fixture import (
    blocking_pdf_worker,
    invalid_chunk_type_worker,
    malformed_pdf_worker,
)


async def _embed(texts: list[str]) -> LocalEmbeddingResult:
    return LocalEmbeddingResult(
        embeddings=[[1.0, 0.0] for _ in texts],
        model="test",
        provider="test",
        dims=2,
        embedding_space_identity="test-space",
    )


def _fake_pdf_parser(parsed: list[Chunk]):
    """Stand in for the PDF worker process; keep production bounding."""

    async def parse_pdf(content: bytes, source_url: str) -> list[Chunk]:
        bounded, _ = _bounded_pdf_chunks(parsed)
        return [replace(chunk, file_path=source_url) for chunk in bounded]

    return parse_pdf


async def _research_results(pages, warnings: list[str] | None = None) -> list[dict]:
    """Run public web research and return every chunk semantic search exposes."""
    captured: dict = {}
    found: list[dict] = []

    async def deep_research(query: str, previous_query: str | None = None):
        results, _ = await captured["db_services"].search_service.search_semantic(
            query, page_size=1000
        )
        found.extend(results)
        return {"answer": "done"}

    def create(**kwargs):
        captured.update(kwargs)
        return MagicMock(deep_research=deep_research)

    with patch(
        "chunkhound.services.web_research_service.ResearchServiceFactory.create",
        side_effect=create,
    ):
        await research_web_pages(
            "query",
            pages,
            MagicMock(),
            MagicMock(
                embed_texts=AsyncMock(side_effect=_embed),
                embed_query_texts=AsyncMock(side_effect=_embed),
            ),
            MagicMock(),
            warning_callback=None if warnings is None else warnings.append,
        )
    return found


def _part(result: dict) -> int:
    match = re.search(r"(\d+)$", str(result["symbol"]))
    return int(match.group(1)) if match else 0


@pytest.mark.asyncio
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
async def test_web_chunks_fit_openai_embedding_limit_without_losing_text(
    extension: str, content: str, monkeypatch
) -> None:
    from chunkhound.services import web_research_service as web

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
    )
    monkeypatch.setattr(web, "_WEB_VECTOR_CACHE", VectorCache(max_entries=0))
    # The third-party PDF parser boundary may return large semantic chunks.
    monkeypatch.setattr(web, "_parse_pdf_isolated", _fake_pdf_parser([parsed]))

    results = sorted(
        await _research_results(
            [(url, extension, b"%PDF" if extension == ".pdf" else content)]
        ),
        key=_part,
    )

    assert len(results) > 1
    assert "".join(result["content"] for result in results) == content
    assert len({result["symbol"] for result in results}) == len(results)
    for result in results:
        assert result["file_path"] == url
        assert len(result["content"].encode("utf-8")) <= 4800
        assert len(encoding.encode(result["content"], disallowed_special=())) <= 8191
        if extension == ".pdf":
            assert result["symbol"].startswith(parsed.symbol)
            assert (result["start_line"], result["end_line"]) == (70, 71)
        else:
            assert (result["start_line"], result["end_line"]) == (1, 1)


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

    async def parse_pdf(content: bytes, source_url: str) -> list[Chunk]:
        bounded, _ = web._bounded_pdf_chunks([parsed])
        return [
            web.replace(chunk, file_path=source_url) for chunk in bounded
        ]

    monkeypatch.setattr(web, "_parse_pdf_isolated", parse_pdf)
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
        MagicMock(
                embed_texts=AsyncMock(side_effect=embed),
                embed_query_texts=AsyncMock(side_effect=embed),
            ),
        MagicMock(),
    )
    assert len(batches) >= 2
    assert all(len(batch) == 100 for batch in batches[:-1])
    assert 1 <= len(batches[-1]) <= 100
    assert "".join(text for batch in batches for text in batch) == content * 2


@pytest.mark.asyncio
async def test_isolated_pdf_parser_enforces_hard_timeout(monkeypatch) -> None:
    from chunkhound.services import web_research_service as web

    monkeypatch.setattr(web, "_PDF_PARSE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(web, "_pdf_parse_worker", blocking_pdf_worker)

    with pytest.raises(TimeoutError, match="PDF parsing timed out"):
        await web._parse_pdf_isolated(b"%PDF", "https://example.invalid/slow.pdf")


@pytest.mark.asyncio
async def test_isolated_pdf_parser_rejects_malformed_payload(monkeypatch) -> None:
    from chunkhound.services import web_research_service as web

    monkeypatch.setattr(web, "_pdf_parse_worker", malformed_pdf_worker)

    with pytest.raises(RuntimeError, match="invalid chunk"):
        await web._parse_pdf_isolated(
            b"%PDF", "https://example.invalid/malformed.pdf"
        )


@pytest.mark.asyncio
async def test_isolated_pdf_parser_rejects_invalid_chunk_type(monkeypatch) -> None:
    from chunkhound.services import web_research_service as web

    monkeypatch.setattr(web, "_pdf_parse_worker", invalid_chunk_type_worker)

    with pytest.raises(RuntimeError, match="invalid chunk"):
        await web._parse_pdf_isolated(
            b"%PDF", "https://example.invalid/invalid-type.pdf"
        )


@pytest.mark.asyncio
async def test_isolated_pdf_parser_cancellation_reaps_worker(monkeypatch) -> None:
    from chunkhound.services import web_research_service as web

    stopped = asyncio.Event()
    real_stop = web._stop_pdf_process

    def observed_stop(process) -> None:
        real_stop(process)
        stopped.set()

    monkeypatch.setattr(web, "_pdf_parse_worker", blocking_pdf_worker)
    monkeypatch.setattr(web, "_stop_pdf_process", observed_stop)
    task = asyncio.create_task(
        web._parse_pdf_isolated(b"%PDF", "https://example.invalid/slow.pdf")
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_isolated_pdf_parser_extracts_fixture_without_disk_writes() -> None:
    content = Path("tests/fixtures/test_simple.pdf").read_bytes()

    chunks = await _parse_pdf_isolated(
        content, "https://example.invalid/test_simple.pdf"
    )

    assert chunks
    assert all(
        chunk.file_path == "https://example.invalid/test_simple.pdf"
        for chunk in chunks
    )
    assert all(len(chunk.code) <= 1_200 for chunk in chunks)


@pytest.mark.asyncio
async def test_web_results_keep_source_url_without_writing_files(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    results = await _research_results(
        [("https://example.invalid/docs", ".md", "# Docs\nUseful content")]
    )

    assert [(r["file_path"], r["content"]) for r in results] == [
        ("https://example.invalid/docs", "# Docs\nUseful content")
    ]
    assert list(tmp_path.iterdir()) == []


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
    manager.embed_query_texts = manager.embed_texts
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("symbol", "code"),
    [
        ("pdf_unavailable", "PDF parsing not available (PyMuPDF not installed)"),
        ("pdf_parse_error", "Error parsing PDF: malformed file"),
    ],
)
async def test_research_skips_unusable_pdf_error_chunks(
    symbol: str, code: str, monkeypatch
) -> None:
    from chunkhound.services import web_research_service as web

    error = Chunk(
        symbol=symbol,
        start_line=LineNumber(1),
        end_line=LineNumber(1),
        code=code,
        chunk_type=ChunkType.UNKNOWN,
        file_id=FileId(0),
        language=Language.PDF,
    )
    monkeypatch.setattr(web, "_parse_pdf_isolated", _fake_pdf_parser([error]))
    warnings: list[str] = []

    results = await _research_results(
        [
            ("https://example.invalid/bad.pdf", ".pdf", b"%PDF"),
            ("https://example.invalid/docs", ".md", "Useful content"),
        ],
        warnings,
    )

    assert [r["file_path"] for r in results] == ["https://example.invalid/docs"]
    assert warnings == ["No usable content parsed from https://example.invalid/bad.pdf"]


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
    manager.embed_query_texts = manager.embed_texts
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
    manager.embed_query_texts = manager.embed_texts
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
            "chunkhound.services.web_research_service._parse_pdf_isolated",
            new=AsyncMock(return_value=[]),
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

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="test",
            provider="test",
            dims=2,
            embedding_space_identity="test-space",
        )

    manager = MagicMock()
    manager.embed_texts = AsyncMock(side_effect=embed)
    manager.embed_query_texts = manager.embed_texts
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
    manager.embed_query_texts = manager.embed_texts
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


_MALFORMED_PDF = (
    b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>\nendobj\n"
    b"4 0 obj\n<< /Length 999 >>\nstream\nBT foo bar baz ET\nendstream\nendobj\n"
    b"xref\n0 1\nbroken\ntrailer\n<< /Root 1 0 R >>\nstartxref\n12345\n%%EOF"
)


def test_isolated_pdf_parser_keeps_stdout_clean_for_mcp_stdio() -> None:
    """Malformed-PDF diagnostics must never reach stdout (MCP JSON-RPC channel)."""
    script = (
        "import asyncio, sys\n"
        "from chunkhound.services.web_research_service import _parse_pdf_isolated\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        asyncio.run(_parse_pdf_isolated("
        "sys.stdin.buffer.read(), 'https://example.invalid/bad.pdf'))\n"
        "    except Exception:\n"
        "        pass\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=_MALFORMED_PDF,
        capture_output=True,
        timeout=60,
        check=True,
    )

    assert result.stdout == b""
    assert b"MuPDF" in result.stderr


@pytest.mark.asyncio
async def test_large_pdf_research_evidence_is_the_matched_chunk_text() -> None:
    """Synthesis must receive the exact PDF text each search hit matched.

    PDF chunk line numbers do not map onto the normalized page text, so a
    large PDF must not be re-sliced by line range when evidence is selected.
    """
    pdf_url = "https://example.invalid/cast_research_paper.pdf"
    content = Path("tests/fixtures/cast_research_paper.pdf").read_bytes()

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="test",
            provider="test",
            dims=2,
            embedding_space_identity="test-space",
        )

    embedding_manager = MagicMock()
    embedding_manager.embed_texts = AsyncMock(side_effect=embed)
    embedding_manager.embed_query_texts = embedding_manager.embed_texts
    llm_manager = MagicMock()
    llm_manager.get_utility_provider.return_value.estimate_tokens = len
    captured: dict = {}
    research = MagicMock()

    async def deep_research(query: str, previous_query: str | None = None):
        services = captured["db_services"]
        results, _ = await services.search_service.search_semantic(query, page_size=20)
        source = await services.provider.get_transient_file_content_async(pdf_url)
        assert source and len(source) * 4 > 64_000  # The large-source path.
        files = await FileReader(services).read_files_with_budget(results, llm_manager)
        return {"results": results, "evidence": files[pdf_url]}

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
            [(pdf_url, ".pdf", content)],
            MagicMock(),
            embedding_manager,
            llm_manager,
        )

    assert result["results"]
    for hit in result["results"]:
        assert hit["content"] in result["evidence"]


def test_large_markdown_page_is_cached_once_counted() -> None:
    from chunkhound.services.web_research_service import (
        _text_chunks,
        _TransientProvider,
    )

    url = "https://example.com/big"
    text = "".join(f"line {i} " + "x" * 80 + "\n" for i in range(130_000))  # ~11 MiB
    provider = _TransientProvider()
    provider._store((url, ".md", text), text, _text_chunks(url, text))
    assert provider.get_transient_file_content(url) == text

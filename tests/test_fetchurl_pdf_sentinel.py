"""Contract tests for fetchurl's PDF-sentinel and empty-content handling.

Covers user-visible short-circuits that prevent the LLM from being called with
useless input: the ``pdf_unavailable`` environment error, the
``pdf_parse_error`` diagnostic, and PDFs that parse to zero chunks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.utils.fetchurl import FetchUrlError, extract, run_fetchurl


def _sentinel_chunk(symbol: str, code: str) -> Chunk:
    return Chunk(
        symbol=symbol,
        start_line=LineNumber(1),
        end_line=LineNumber(1),
        code=code,
        chunk_type=ChunkType.UNKNOWN,
        file_id=FileId(0),
        language=Language.PDF,
    )


def test_extract_raises_actionable_error_for_pdf_unavailable():
    sentinel = _sentinel_chunk(
        "pdf_unavailable",
        "PDF parsing not available (PyMuPDF not installed)",
    )
    with patch(
        "chunkhound.utils.fetchurl.PDFMapping.parse_pdf_content",
        return_value=[sentinel],
    ):
        with pytest.raises(FetchUrlError) as excinfo:
            extract(".pdf", b"%PDF-1.4 fake", "http://example.com/x.pdf", {})
    message = str(excinfo.value)
    assert "PyMuPDF" in message
    assert "uv add pymupdf" in message


@pytest.mark.asyncio
async def test_run_fetchurl_skips_llm_when_pdf_parse_fails(tmp_path):
    url = "http://example.com/broken.pdf"
    detail = "Error parsing PDF: cannot open document (encrypted)"
    sentinel = _sentinel_chunk("pdf_parse_error", detail)

    llm_manager = MagicMock()
    provider = MagicMock()
    provider.complete = AsyncMock(
        side_effect=AssertionError(
            "LLM must not be called on PDF parse-error short-circuit"
        )
    )
    llm_manager.get_utility_provider.return_value = provider

    embedding_provider = MagicMock()
    warning_callback = MagicMock()

    async def fake_fetch(u, cfg, warning_callback=None):
        return (".pdf", b"%PDF-1.4 fake", {"title": None})

    with patch("chunkhound.utils.fetchurl._fetch_with_retry", side_effect=fake_fetch), \
         patch(
             "chunkhound.utils.fetchurl.PDFMapping.parse_pdf_content",
             return_value=[sentinel],
         ):
        answer = await run_fetchurl(
            url,
            "what is this doc about",
            Config(target_dir=tmp_path),
            embedding_provider,
            llm_manager,
            warning_callback=warning_callback,
        )

    assert url in answer
    assert detail in answer
    assert "could not extract" in answer.lower()
    provider.complete.assert_not_called()
    # The wrapped answer body already carries `url` + `detail`; a warning
    # would duplicate the same string in the caller's UI (MCP footer, CLI
    # `[WARN]` line). Callers rely on the body as the sole diagnostic.
    warning_callback.assert_not_called()


@pytest.mark.asyncio
async def test_run_fetchurl_skips_llm_when_pdf_has_no_content(tmp_path):
    url = "http://example.com/empty.pdf"

    llm_manager = MagicMock()
    provider = MagicMock()
    provider.complete = AsyncMock(
        side_effect=AssertionError(
            "LLM must not be called on zero-chunk PDF short-circuit"
        )
    )
    llm_manager.get_utility_provider.return_value = provider

    embedding_provider = MagicMock()
    warning_callback = MagicMock()

    async def fake_fetch(u, cfg, warning_callback=None):
        return (".pdf", b"%PDF-1.4 fake", {"title": None})

    with patch("chunkhound.utils.fetchurl._fetch_with_retry", side_effect=fake_fetch), \
         patch(
             "chunkhound.utils.fetchurl.PDFMapping.parse_pdf_content",
             return_value=[],
         ):
        answer = await run_fetchurl(
            url,
            "what is this doc about",
            Config(target_dir=tmp_path),
            embedding_provider,
            llm_manager,
            warning_callback=warning_callback,
        )

    assert url in answer
    assert "no usable content" in answer.lower()
    assert "PDF extracted zero chunks" in answer
    provider.complete.assert_not_called()
    # `_wrap_no_content` already puts `url` + reason in the body; skipping
    # the warning avoids duplicating the same string for callers.
    warning_callback.assert_not_called()

"""Contract tests for fetchurl's MCP surface.

Bullets 1-2 of spec §16.2 (tools/list filtering + schema shape) are covered
in ``tests/test_mcp_tool_consistency.py`` — see
``test_fetchurl_hidden_without_capabilities`` and ``test_fetchurl_schema``.
Bullet 4 (dispatcher reranker guard in ``common.py``) is derivable from
tested primitives: ``TOOL_REGISTRY["fetchurl"].requires_reranker`` is
asserted in ``test_tool_capability_requirements``, and
``has_reranker_support`` is exercised in ``test_embeddings.py``. No
per-tool runtime assertion.

This module covers bullet 3: ``fetchurl_impl`` wraps LLM output in the
shared ``Source:`` box, plus the user-visible ``MCPError`` translation
contract for every failure class ``fetchurl_impl`` handles.
"""

from __future__ import annotations

import asyncio
import ssl
import urllib.error
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.config import Config
from chunkhound.mcp_server.common import MCPError
from chunkhound.mcp_server.tools import fetchurl_impl
from chunkhound.utils.fetchurl import FetchUrlError


@pytest.mark.asyncio
async def test_fetchurl_mcp_call_returns_source_boxed_markdown(tmp_path):
    url = "http://example.com/doc.md"
    canned_md = "# Hello\n\nBody paragraph."

    # query="" routes through option_truncate — the reranker path is not
    # exercised here, so the returned mock is unused on this path.
    embedding_manager = MagicMock()

    llm_manager = MagicMock()
    llm_manager.get_utility_provider.return_value.complete = AsyncMock(
        return_value=SimpleNamespace(content=canned_md)
    )

    async def fake_fetch(u, cfg, warning_callback=None):
        return (".md", "# Hello\n\nBody paragraph.", {"title": "Hello"})

    with patch(
        "chunkhound.utils.fetchurl._fetch_with_retry", side_effect=fake_fetch
    ):
        result = await fetchurl_impl(
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
            config=Config(target_dir=tmp_path),
            url=url,
            query="",
        )

    assert result == f"Source: {url}\n{'=' * 60}\n{canned_md}\n{'=' * 60}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised, expected_fragment",
    [
        pytest.param(
            asyncio.TimeoutError(), "fetchurl timed out", id="asyncio_timeout"
        ),
        pytest.param(
            FetchUrlError("blocked host"),
            "fetchurl failed: blocked host",
            id="blocked_host",
        ),
        # HTTPError must match before URLError (subclass); asserting the
        # "fetchurl failed: HTTP {code} {reason}" shape here also catches an
        # except-clause reorder that would strip the status code.
        pytest.param(
            urllib.error.HTTPError("u", 503, "Bad Gateway", {}, None),
            "fetchurl failed: HTTP 503 Bad Gateway",
            id="http_error",
        ),
        pytest.param(
            urllib.error.URLError("dns fail"),
            "fetchurl failed: dns fail",
            id="url_error",
        ),
        # SSLError single-arg construction stringifies as the args tuple
        # ("('x',)"), so use (errno, strerror) — SSLError strips OSError's
        # "[Errno N]" prefix and yields just strerror.
        pytest.param(
            ssl.SSLError(1, "cert expired"),
            "fetchurl failed: cert expired",
            id="ssl_error",
        ),
        pytest.param(
            ValueError("Unsupported content-type: 'image/png'"),
            "fetchurl failed: Unsupported content-type: 'image/png'",
            id="unsupported_content_type",
        ),
    ],
)
async def test_fetchurl_translates_failures_to_mcperror(
    raised, expected_fragment, tmp_path
):
    # Patch on the source module: ``run_fetchurl`` is lazy-imported inside
    # ``fetchurl_impl``, so ``chunkhound.mcp_server.tools.run_fetchurl`` has
    # no attribute to bind before the first call.
    with patch(
        "chunkhound.utils.fetchurl.run_fetchurl",
        new=AsyncMock(side_effect=raised),
    ):
        with pytest.raises(MCPError) as exc:
            await fetchurl_impl(
                embedding_manager=MagicMock(),
                llm_manager=MagicMock(),
                config=Config(target_dir=tmp_path),
                url="https://example.com",
                query="",
            )

    assert expected_fragment in str(exc.value)

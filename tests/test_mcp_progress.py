"""Tests for MCP progress notifications (issue #309).

Verifies that tools emit progress notifications when a progressToken is
present, and stay silent when it is absent.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.mcp_server.common import MCPError, handle_tool_call
from chunkhound.mcp_server.tools import (
    _emit,
    deep_research_impl,
    execute_tool,
    websearch_impl,
)

# ---------------------------------------------------------------------------
# _emit helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_calls_reporter() -> None:
    """_emit invokes the reporter with the supplied arguments."""
    calls: list[tuple[int, int | None, str]] = []

    async def reporter(progress: int, total: int | None, message: str) -> None:
        calls.append((progress, total, message))

    await _emit(reporter, 1, 3, "Hello")
    assert calls == [(1, 3, "Hello")]


@pytest.mark.asyncio
async def test_emit_no_op_when_none() -> None:
    """_emit is a no-op and does not raise when reporter is None."""
    await _emit(None, 1, 3, "Hello")


# ---------------------------------------------------------------------------
# search_impl progress emissions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_regex_emits_progress() -> None:
    """Regex search emits 2 monotonically-increasing progress steps."""
    calls: list[tuple[int, int | None, str]] = []

    async def reporter(progress: int, total: int | None, message: str) -> None:
        calls.append((progress, total, message))

    mock_services = MagicMock()
    mock_services.search_service.search_regex_async = AsyncMock(
        return_value=([], {"total": 0, "offset": 0, "has_more": False})
    )

    await execute_tool(
        tool_name="search",
        services=mock_services,
        embedding_manager=None,
        arguments={"type": "regex", "query": "def foo"},
        progress_reporter=reporter,
    )

    assert len(calls) == 2
    progresses = [c[0] for c in calls]
    assert progresses == sorted(progresses), (
        "progress values must be monotonically increasing"
    )
    assert all(c[1] == 2 for c in calls)
    assert calls[0][2] == "Searching index…"
    assert calls[1][2] == "Formatting results…"


@pytest.mark.asyncio
async def test_search_emits_no_progress_without_token() -> None:
    """No notifications emitted when progress_reporter is None."""
    mock_services = MagicMock()
    mock_services.search_service.search_regex_async = AsyncMock(
        return_value=([], {"total": 0, "offset": 0, "has_more": False})
    )

    result = await execute_tool(
        tool_name="search",
        services=mock_services,
        embedding_manager=None,
        arguments={"type": "regex", "query": "def foo"},
        progress_reporter=None,
    )
    assert result is not None


@pytest.mark.asyncio
async def test_search_semantic_emits_progress() -> None:
    """Semantic search emits 2 progress steps including 'Embedding query…'."""
    calls: list[tuple[int, int | None, str]] = []

    async def reporter(progress: int, total: int | None, message: str) -> None:
        calls.append((progress, total, message))

    mock_provider = MagicMock()
    mock_provider.name = "openai"
    mock_provider.model = "text-embedding-3-small"

    mock_embedding_manager = MagicMock()
    mock_embedding_manager.list_providers.return_value = ["openai"]
    mock_embedding_manager.get_provider.return_value = mock_provider

    mock_services = MagicMock()
    mock_services.search_service.search_semantic = AsyncMock(
        return_value=([], {"total": 0, "offset": 0, "has_more": False})
    )

    await execute_tool(
        tool_name="search",
        services=mock_services,
        embedding_manager=mock_embedding_manager,
        arguments={"type": "semantic", "query": "authentication logic"},
        progress_reporter=reporter,
    )

    progresses = [c[0] for c in calls]
    assert progresses == sorted(progresses)
    assert all(c[1] == 2 for c in calls)
    assert calls[0][2] == "Embedding query…"
    assert calls[-1][2] == "Formatting results…"


# ---------------------------------------------------------------------------
# websearch_impl progress emissions
# ---------------------------------------------------------------------------


def _make_mock_proc(returncode: int = 0, stdout: bytes = b"ANSWER: test") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


@pytest.mark.asyncio
async def test_websearch_emits_3_progress_steps() -> None:
    """websearch emits (1,3), (2,3), (3,3) progress steps in order."""
    calls: list[tuple[int, int | None, str]] = []

    async def reporter(progress: int, total: int | None, message: str) -> None:
        calls.append((progress, total, message))

    mock_proc = _make_mock_proc()

    with (
        patch(
            "chunkhound.utils.websearch_core.search",
            return_value=[("title", "http://x.com", "snippet")],
        ),
        patch("chunkhound.utils.websearch_core.fetch_and_save", new=AsyncMock()),
        patch(
            "chunkhound.utils.websearch_core.clamp_limit", side_effect=lambda x: x
        ),
        patch(
            "chunkhound.utils.websearch_core.build_quickresearch_argv_core",
            return_value=["echo", "OK"],
        ),
        patch(
            "chunkhound.utils.websearch_core.websearch_timeout", return_value=30.0
        ),
        patch(
            "chunkhound.utils.websearch_postprocess.replace_paths_with_urls",
            side_effect=lambda s, m: s,
        ),
        patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)
        ),
    ):
        await websearch_impl(
            embedding_manager=MagicMock(),
            llm_manager=None,
            config=MagicMock(),
            query="test query",
            limit=3,
            progress_reporter=reporter,
        )

    assert len(calls) == 3
    assert [c[0] for c in calls] == [1, 2, 3]
    assert all(c[1] == 3 for c in calls)
    assert calls[0][2] == "Searching web…"
    assert calls[1][2] == "Fetching pages…"
    assert calls[2][2] == "Researching results…"


@pytest.mark.asyncio
async def test_websearch_emits_no_progress_without_reporter() -> None:
    """websearch completes without raising when progress_reporter is None."""
    mock_proc = _make_mock_proc()

    with (
        patch(
            "chunkhound.utils.websearch_core.search",
            return_value=[("title", "http://x.com", "snippet")],
        ),
        patch("chunkhound.utils.websearch_core.fetch_and_save", new=AsyncMock()),
        patch(
            "chunkhound.utils.websearch_core.clamp_limit", side_effect=lambda x: x
        ),
        patch(
            "chunkhound.utils.websearch_core.build_quickresearch_argv_core",
            return_value=["echo", "OK"],
        ),
        patch(
            "chunkhound.utils.websearch_core.websearch_timeout", return_value=30.0
        ),
        patch(
            "chunkhound.utils.websearch_postprocess.replace_paths_with_urls",
            side_effect=lambda s, m: s,
        ),
        patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)
        ),
    ):
        result = await websearch_impl(
            embedding_manager=MagicMock(),
            llm_manager=None,
            config=MagicMock(),
            query="test query",
            limit=3,
            progress_reporter=None,
        )

    assert result is not None


@pytest.mark.asyncio
async def test_websearch_emits_step1_before_empty_results_error() -> None:
    """Step 1 fires before the early-exit on empty search results."""
    calls: list[tuple[int, int | None, str]] = []

    async def reporter(progress: int, total: int | None, message: str) -> None:
        calls.append((progress, total, message))

    with (
        patch("chunkhound.utils.websearch_core.search", return_value=[]),
        patch(
            "chunkhound.utils.websearch_core.clamp_limit", side_effect=lambda x: x
        ),
    ):
        with pytest.raises(MCPError):
            await websearch_impl(
                embedding_manager=MagicMock(),
                llm_manager=None,
                config=MagicMock(),
                query="no results query",
                limit=3,
                progress_reporter=reporter,
            )

    assert len(calls) == 1
    assert calls[0] == (1, 3, "Searching web…")


# ---------------------------------------------------------------------------
# deep_research_impl progress emissions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_research_emits_progress() -> None:
    """code_research emits 2 progress steps in order."""
    calls: list[tuple[int, int | None, str]] = []

    async def reporter(progress: int, total: int | None, message: str) -> None:
        calls.append((progress, total, message))

    mock_provider = MagicMock()
    mock_provider.supports_reranking.return_value = True

    mock_embedding_manager = MagicMock()
    mock_embedding_manager.get_provider.return_value = mock_provider

    mock_research_service = MagicMock()
    mock_research_service.deep_research = AsyncMock(
        return_value={"answer": "test", "sources": []}
    )

    with patch("chunkhound.mcp_server.tools.ResearchServiceFactory") as mock_factory:
        mock_factory.create.return_value = mock_research_service
        await deep_research_impl(
            services=MagicMock(),
            embedding_manager=mock_embedding_manager,
            llm_manager=MagicMock(),
            config=MagicMock(),
            query="how does auth work",
            progress_reporter=reporter,
        )

    assert len(calls) == 2
    assert [c[0] for c in calls] == [1, 2]
    assert all(c[1] == 2 for c in calls)
    assert calls[0][2] == "Setting up research service…"
    assert calls[1][2] == "Running analysis (retrieval + LLM reranking)…"


@pytest.mark.asyncio
async def test_code_research_emits_no_progress_without_reporter() -> None:
    """code_research completes without raising when progress_reporter is None."""
    mock_provider = MagicMock()
    mock_provider.supports_reranking.return_value = True

    mock_embedding_manager = MagicMock()
    mock_embedding_manager.get_provider.return_value = mock_provider

    mock_research_service = MagicMock()
    mock_research_service.deep_research = AsyncMock(
        return_value={"answer": "test", "sources": []}
    )

    with patch("chunkhound.mcp_server.tools.ResearchServiceFactory") as mock_factory:
        mock_factory.create.return_value = mock_research_service
        result = await deep_research_impl(
            services=MagicMock(),
            embedding_manager=mock_embedding_manager,
            llm_manager=MagicMock(),
            config=MagicMock(),
            query="how does auth work",
            progress_reporter=None,
        )

    assert result is not None


# ---------------------------------------------------------------------------
# handle_tool_call wires progress_reporter through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_tool_call_passes_reporter() -> None:
    """handle_tool_call forwards progress_reporter to execute_tool."""
    calls: list[tuple[int, int | None, str]] = []

    async def reporter(progress: int, total: int | None, message: str) -> None:
        calls.append((progress, total, message))

    init_event = asyncio.Event()
    init_event.set()

    mock_services = MagicMock()
    mock_services.search_service.search_regex_async = AsyncMock(
        return_value=([], {"total": 0, "offset": 0, "has_more": False})
    )

    with patch("mcp.types") as mock_types:
        mock_types.TextContent = MagicMock(side_effect=lambda **kw: kw)
        await handle_tool_call(
            tool_name="search",
            arguments={"type": "regex", "query": "hello"},
            services=mock_services,
            embedding_manager=None,
            initialization_complete=init_event,
            progress_reporter=reporter,
        )

    assert len(calls) >= 2, "Expected at least 2 progress notifications"
    progresses = [c[0] for c in calls]
    assert progresses == sorted(progresses)

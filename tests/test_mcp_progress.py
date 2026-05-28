"""Tests for MCP progress notifications (issue #309).

Verifies that tools emit progress notifications when a progressToken is
present, and stay silent when it is absent.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.mcp_server.tools import _emit, execute_tool


# ---------------------------------------------------------------------------
# _emit helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_calls_reporter() -> None:
    calls: list[tuple] = []

    async def reporter(progress: int, total: int | None, message: str) -> None:
        calls.append((progress, total, message))

    await _emit(reporter, 1, 3, "Hello")
    assert calls == [(1, 3, "Hello")]


@pytest.mark.asyncio
async def test_emit_no_op_when_none() -> None:
    # Must not raise even with None reporter
    await _emit(None, 1, 3, "Hello")


# ---------------------------------------------------------------------------
# search_impl progress emissions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_regex_emits_progress() -> None:
    """Regex search emits 2 monotonically-increasing progress steps."""
    calls: list[tuple] = []

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
    assert progresses == sorted(progresses), "progress values must be monotonically increasing"
    assert calls[0][2] == "Searching index…"
    assert calls[1][2] == "Formatting results…"


@pytest.mark.asyncio
async def test_search_emits_no_progress_without_token() -> None:
    """No notifications emitted when progress_reporter is None."""
    mock_services = MagicMock()
    mock_services.search_service.search_regex_async = AsyncMock(
        return_value=([], {"total": 0, "offset": 0, "has_more": False})
    )

    # Should complete without error and without calling any reporter
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
    calls: list[tuple] = []

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
    assert calls[0][2] == "Embedding query…"
    assert calls[-1][2] == "Formatting results…"


# ---------------------------------------------------------------------------
# handle_tool_call wires progress_reporter through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_tool_call_passes_reporter() -> None:
    """handle_tool_call forwards progress_reporter to execute_tool."""
    from chunkhound.mcp_server.common import handle_tool_call

    calls: list[tuple] = []

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

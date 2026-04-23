"""Test that CompactionError produces a structured retry-hint response."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.exceptions import CompactionError
from chunkhound.mcp_server.common import compaction_error_response
from chunkhound.services.compaction_service import CompactionService


def _capture_stdio_handler(server) -> object:
    """Capture the raw stdio call handler registered with the MCP SDK."""
    captured = {}
    original_call_tool = server.server.call_tool

    def capturing_call_tool(**kwargs):
        decorator = original_call_tool(**kwargs)

        def wrapper(fn):
            captured["handler"] = fn
            return decorator(fn)

        return wrapper

    server.server.call_tool = capturing_call_tool
    server._register_tools()
    return captured["handler"]


def _make_indexing_config(force_reindex: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        include=["*.py"],
        exclude=[],
        config_file_size_threshold_kb=100,
        force_reindex=force_reindex,
        model_copy=lambda update=None, **_: _make_indexing_config(
            force_reindex=(update or {}).get("force_reindex", force_reindex)
        ),
    )


def _configure_post_reindex_recovery_server(server, tmp_path: Path) -> None:
    indexing_config = _make_indexing_config()
    server.config.database.path = str(tmp_path / "test.db")
    server.config.indexing = indexing_config
    server.config.model_copy.side_effect = lambda update=None, **_: SimpleNamespace(
        indexing=(update or {}).get("indexing", indexing_config)
    )
    server._initialization_complete.set()
    server.services = MagicMock()
    server.services.provider.is_connected = True
    server.services.indexing_coordinator.generate_missing_embeddings = AsyncMock(
        return_value={"status": "success", "generated": 0}
    )
    server.realtime_indexing = MagicMock()
    server.realtime_indexing.drain_compaction_deferred_files = AsyncMock(
        return_value=set()
    )
    server.realtime_indexing.restore_compaction_deferred_files = AsyncMock()
    server._target_path = tmp_path
    server._compaction_service = CompactionService(tmp_path / "test.db", server.config)
    server._compaction_service._last_error = RuntimeError("initial failure")
    server._scan_progress["background_compaction"].update(
        {
            "phase": "failed",
            "pending_recovery": True,
            "retry_attempted": False,
            "last_error": "initial failure",
        }
    )


def test_transient_compaction_error_has_retry_hint():
    """Transient compaction errors advise the client to retry."""
    exc = CompactionError("Compaction in progress", operation="connection")
    resp = compaction_error_response(exc)
    assert resp["error"]["type"] == "CompactionError"
    assert resp["error"]["retry_hint"] == (
        "Database compaction in progress. Retry in a few seconds."
    )


def test_unrecoverable_compaction_error_no_retry():
    """Unrecoverable errors advise restore/re-index, not retry."""
    exc = CompactionError(
        "no valid database or backup found",
        operation="recovery",
        recoverable=False,
    )
    resp = compaction_error_response(exc)
    assert resp["error"]["type"] == "CompactionError"
    assert resp["error"]["retry_hint"] == (
        "Database recovery failed after interrupted compaction. "
        "Restore from backup or re-index."
    )


@pytest.mark.asyncio
async def test_stdio_handler_returns_error_on_compaction_during_ensure_services():
    """When ensure_services() raises CompactionError, the handler returns JSON.

    Verifies the catch at stdio.py's handle_all_tools, not just the helper.
    """
    from chunkhound.mcp_server.stdio import StdioMCPServer

    config = MagicMock()
    config.debug = False

    server = StdioMCPServer(config=config)

    # Extract the inner handler captured by the MCP SDK decorator.
    # call_tool() returns a decorator that wraps our function; the SDK
    # stores the wrapper in request_handlers, but the original function
    # is accessible via __wrapped__ or we can intercept registration.
    handler = _capture_stdio_handler(server)

    exc = CompactionError("Compaction in progress", operation="connection")
    with patch.object(
        server,
        "ensure_services",
        new_callable=AsyncMock,
        side_effect=exc,
    ):
        result = await handler("search", {"query": "test"})

    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["error"]["type"] == "CompactionError"
    assert payload["error"]["retry_hint"] == (
        "Database compaction in progress. Retry in a few seconds."
    )


@pytest.mark.asyncio
async def test_stdio_handler_returns_restore_hint_for_unrecoverable_recovery():
    """Stdio handler must preserve the exact recovery contract for clients."""
    from chunkhound.mcp_server.stdio import StdioMCPServer

    config = MagicMock()
    config.debug = False

    server = StdioMCPServer(config=config)
    handler = _capture_stdio_handler(server)

    exc = CompactionError(
        "no valid database or backup found",
        operation="recovery",
        recoverable=False,
    )
    with patch.object(
        server,
        "ensure_services",
        new_callable=AsyncMock,
        side_effect=exc,
    ):
        result = await handler("search", {"query": "test"})

    payload = json.loads(result[0].text)
    assert payload["error"]["type"] == "CompactionError"
    assert payload["error"]["retry_hint"] == (
        "Database recovery failed after interrupted compaction. "
        "Restore from backup or re-index."
    )


@pytest.mark.asyncio
async def test_stdio_handler_returns_post_reindex_hint_from_real_recovery_path(
    tmp_path: Path,
):
    """Stdio handler must surface failed post-compaction catch-up to clients."""
    from chunkhound.mcp_server.stdio import StdioMCPServer

    config = MagicMock()
    config.debug = False

    server = StdioMCPServer(config=config)
    handler = _capture_stdio_handler(server)
    _configure_post_reindex_recovery_server(server, tmp_path)
    server.services.indexing_coordinator.process_directory = AsyncMock(
        side_effect=RuntimeError("retry failed")
    )

    with patch("chunkhound.mcp_server.stdio.handle_tool_call", new_callable=AsyncMock):
        first_result = await handler("search", {"query": "test"})
        status_after_first_call = server.get_background_compaction_status()
        second_result = await handler("search", {"query": "test"})

    first_payload = json.loads(first_result[0].text)
    second_payload = json.loads(second_result[0].text)
    assert first_payload["error"]["type"] == "CompactionError"
    assert first_payload["error"]["message"] == (
        "Compaction error (operation=post_reindex): retry failed"
    )
    assert first_payload["error"]["retry_hint"] == (
        "Post-compaction catch-up reindex failed. Retry after recovery or "
        "restart the MCP server."
    )
    assert second_payload == first_payload
    server.services.indexing_coordinator.process_directory.assert_awaited_once()
    assert status_after_first_call["pending_recovery"] is True
    assert status_after_first_call["retry_attempted"] is True


@pytest.mark.asyncio
async def test_stdio_handler_allows_normal_tool_execution_after_successful_recovery(
    tmp_path: Path,
) -> None:
    """Successful recovery clears the compaction error path for stdio clients."""
    from chunkhound.mcp_server.stdio import StdioMCPServer

    config = MagicMock()
    config.debug = False

    server = StdioMCPServer(config=config)
    handler = _capture_stdio_handler(server)
    _configure_post_reindex_recovery_server(server, tmp_path)
    server.services.indexing_coordinator.process_directory = AsyncMock(
        return_value={
            "status": "success",
            "files_processed": 1,
            "total_chunks": 2,
        }
    )

    success_result = [SimpleNamespace(type="text", text=json.dumps({"ok": True}))]
    with patch(
        "chunkhound.mcp_server.stdio.handle_tool_call",
        AsyncMock(return_value=success_result),
    ) as mocked_handle_tool_call:
        result = await handler("search", {"query": "test"})

    assert result == success_result
    mocked_handle_tool_call.assert_awaited_once()
    server.services.indexing_coordinator.process_directory.assert_awaited_once()
    server.services.indexing_coordinator.generate_missing_embeddings.assert_awaited_once()
    status = server.get_background_compaction_status()
    assert status["phase"] == "idle"
    assert status["pending_recovery"] is False
    assert status["retry_attempted"] is False

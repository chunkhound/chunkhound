"""Test that CompactionError produces a structured retry-hint response."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.exceptions import CompactionError
from chunkhound.mcp_server.common import compaction_error_response


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
    """When ensure_services() raises CompactionError, the handler returns structured JSON.

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
    with patch.object(server, "ensure_services", new_callable=AsyncMock, side_effect=exc):
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
    with patch.object(server, "ensure_services", new_callable=AsyncMock, side_effect=exc):
        result = await handler("search", {"query": "test"})

    payload = json.loads(result[0].text)
    assert payload["error"]["type"] == "CompactionError"
    assert payload["error"]["retry_hint"] == (
        "Database recovery failed after interrupted compaction. "
        "Restore from backup or re-index."
    )

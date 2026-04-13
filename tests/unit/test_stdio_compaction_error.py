"""Test that CompactionError produces a structured retry-hint response."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.exceptions import CompactionError
from chunkhound.mcp_server.common import compaction_error_response


def test_transient_compaction_error_has_retry_hint():
    """Transient compaction errors advise the client to retry."""
    exc = CompactionError("Compaction in progress", operation="connection")
    resp = compaction_error_response(exc)
    assert resp["error"]["type"] == "CompactionError"
    assert "retry" in resp["error"]["retry_hint"].lower()


def test_unrecoverable_compaction_error_no_retry():
    """Unrecoverable errors advise restore/re-index, not retry."""
    exc = CompactionError(
        "no valid database or backup found",
        operation="recovery",
        recoverable=False,
    )
    resp = compaction_error_response(exc)
    assert resp["error"]["type"] == "CompactionError"
    assert "recovery failed" in resp["error"]["retry_hint"].lower()
    assert "retry in" not in resp["error"]["retry_hint"].lower()


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
    captured = {}
    original_call_tool = server.server.call_tool

    def capturing_call_tool(**kwargs):
        """Intercept registration to capture the raw handler."""
        decorator = original_call_tool(**kwargs)

        def wrapper(fn):
            captured["handler"] = fn
            return decorator(fn)

        return wrapper

    server.server.call_tool = capturing_call_tool
    server._register_tools()  # re-register to capture
    handler = captured["handler"]

    exc = CompactionError("Compaction in progress", operation="connection")
    with patch.object(server, "ensure_services", new_callable=AsyncMock, side_effect=exc):
        result = await handler("search", {"query": "test"})

    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["error"]["type"] == "CompactionError"
    assert "retry" in payload["error"]["retry_hint"].lower()

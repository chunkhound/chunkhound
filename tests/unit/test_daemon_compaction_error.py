"""Tests for daemon transport handling of compaction errors."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.exceptions import CompactionError
from chunkhound.daemon.server import ChunkHoundDaemon


@pytest.mark.asyncio
async def test_daemon_tools_call_returns_post_reindex_hint(tmp_path: Path) -> None:
    """Daemon tool calls should return structured compaction payloads."""
    config = MagicMock()
    config.debug = False

    with patch("chunkhound.mcp_server.base.create_services"), patch(
        "chunkhound.mcp_server.base.EmbeddingManager"
    ):
        daemon = ChunkHoundDaemon(
            config=config,
            args=MagicMock(),
            socket_path="tcp://127.0.0.1:0",
            project_dir=tmp_path,
        )

    daemon._initialization_complete.set()
    exc = CompactionError("retry failed", operation="post_reindex")
    with patch.object(daemon, "ensure_services", new_callable=AsyncMock, side_effect=exc):
        result = await daemon._handle_tools_call(
            {
                "id": 7,
                "params": {
                    "name": "search",
                    "arguments": {"type": "regex", "query": "test"},
                },
            }
        )

    assert result["result"]["isError"] is True
    payload = json.loads(result["result"]["content"][0]["text"])
    assert payload["error"]["type"] == "CompactionError"
    assert payload["error"]["retry_hint"] == (
        "Post-compaction catch-up reindex failed. Retry after recovery or "
        "restart the MCP server."
    )

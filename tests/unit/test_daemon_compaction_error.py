"""Tests for daemon transport handling of compaction errors."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.daemon.server import ChunkHoundDaemon
from chunkhound.services.compaction_service import CompactionService


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


def _configure_post_reindex_recovery_daemon(
    daemon: ChunkHoundDaemon,
    tmp_path: Path,
) -> None:
    indexing_config = _make_indexing_config()
    daemon.config.database.path = str(tmp_path / "test.db")
    daemon.config.indexing = indexing_config
    daemon.config.model_copy.side_effect = lambda update=None, **_: SimpleNamespace(
        indexing=(update or {}).get("indexing", indexing_config)
    )
    daemon._initialization_complete.set()
    daemon.services = MagicMock()
    daemon.services.provider.is_connected = True
    daemon.services.indexing_coordinator.generate_missing_embeddings = AsyncMock(
        return_value={"status": "success", "generated": 0}
    )
    daemon.realtime_indexing = MagicMock()
    daemon.realtime_indexing.drain_compaction_deferred_directories = AsyncMock(
        return_value=set()
    )
    daemon.realtime_indexing.restore_compaction_deferred_directories = AsyncMock()
    daemon.realtime_indexing.replay_compaction_deferred_directory = AsyncMock()
    daemon.realtime_indexing.drain_compaction_deferred_files = AsyncMock(
        return_value=set()
    )
    daemon.realtime_indexing.drain_compaction_deferred_removals = AsyncMock(
        return_value=set()
    )
    daemon.realtime_indexing.drain_compaction_deferred_file_work = AsyncMock(
        return_value=(set(), set())
    )
    daemon.realtime_indexing.restore_compaction_deferred_files = AsyncMock()
    daemon.realtime_indexing.restore_compaction_deferred_removals = AsyncMock()
    daemon._target_path = tmp_path
    daemon._compaction_service = CompactionService(tmp_path / "test.db", daemon.config)
    daemon._compaction_service._last_error = RuntimeError("initial failure")
    daemon._scan_progress["background_compaction"].update(
        {
            "phase": "failed",
            "pending_recovery": True,
            "retry_attempted": False,
            "last_error": "initial failure",
        }
    )


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

    _configure_post_reindex_recovery_daemon(daemon, tmp_path)
    daemon.services.indexing_coordinator.process_directory = AsyncMock(
        side_effect=RuntimeError("retry failed")
    )

    with patch("chunkhound.daemon.server.handle_tool_call", new_callable=AsyncMock):
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
    assert payload["error"]["message"] == (
        "Compaction error (operation=post_reindex): retry failed"
    )
    assert payload["error"]["retry_hint"] == (
        "Post-compaction catch-up reindex failed. Retry after recovery or "
        "restart the MCP server."
    )
    status_after_first_call = daemon.get_background_compaction_status()
    assert status_after_first_call["pending_recovery"] is True
    assert status_after_first_call["retry_attempted"] is True

    with patch("chunkhound.daemon.server.handle_tool_call", new_callable=AsyncMock):
        second_result = await daemon._handle_tools_call(
            {
                "id": 8,
                "params": {
                    "name": "search",
                    "arguments": {"type": "regex", "query": "test"},
                },
            }
        )

    second_payload = json.loads(second_result["result"]["content"][0]["text"])
    assert second_payload == payload


@pytest.mark.asyncio
async def test_daemon_tools_call_allows_normal_tool_execution_after_successful_recovery(
    tmp_path: Path,
) -> None:
    """Successful recovery clears the compaction error path for daemon clients."""
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

    _configure_post_reindex_recovery_daemon(daemon, tmp_path)
    daemon.services.indexing_coordinator.process_directory = AsyncMock(
        return_value={
            "status": "success",
            "files_processed": 1,
            "total_chunks": 2,
        }
    )

    success_content = [SimpleNamespace(type="text", text=json.dumps({"ok": True}))]
    with patch(
        "chunkhound.daemon.server.handle_tool_call",
        AsyncMock(return_value=success_content),
    ) as mocked_handle_tool_call:
        result = await daemon._handle_tools_call(
            {
                "id": 9,
                "params": {
                    "name": "search",
                    "arguments": {"type": "regex", "query": "test"},
                },
            }
        )

    assert result["result"]["isError"] is False
    assert result["result"]["content"] == [{"type": "text", "text": json.dumps({"ok": True})}]
    status = daemon.get_background_compaction_status()
    assert status["phase"] == "idle"
    assert status["pending_recovery"] is False
    assert status["retry_attempted"] is False

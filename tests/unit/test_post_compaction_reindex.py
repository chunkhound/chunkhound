"""Unit tests for MCPServerBase._post_compaction_reindex."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.mcp_server.base import MCPServerBase


class ConcreteMCPServer(MCPServerBase):
    """Minimal concrete implementation for testing base class behavior."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


@pytest.fixture
def server(tmp_path: Path) -> ConcreteMCPServer:
    config = MagicMock()
    config.database.path = str(tmp_path / "test.db")
    config.target_dir = tmp_path

    with patch("chunkhound.mcp_server.base.create_services"), \
         patch("chunkhound.mcp_server.base.EmbeddingManager"):
        srv = ConcreteMCPServer(config=config)

    srv.services = MagicMock()
    srv._target_path = tmp_path
    return srv


async def test_clears_compaction_deferrals_before_reindex(
    server: ConcreteMCPServer,
) -> None:
    """Reindex sees deferred-state cleanup already applied."""

    class RealtimeStub:
        def __init__(self) -> None:
            self.deferred_files = {"stale.py"}

        async def clear_compaction_deferred_files(self) -> None:
            self.deferred_files.clear()

    realtime = RealtimeStub()
    server.realtime_indexing = realtime

    mock_stats = MagicMock(files_processed=5, chunks_created=42)

    async def _record_reindex(*args, **kwargs):
        assert realtime.deferred_files == set(), (
            "Post-compaction reindex must run only after stale deferred state is cleared"
        )
        return mock_stats

    with patch("chunkhound.mcp_server.base.DirectoryIndexingService") as MockDIS:
        MockDIS.return_value.process_directory = AsyncMock(side_effect=_record_reindex)
        await server._post_compaction_reindex()


async def test_uses_force_reindex(server: ConcreteMCPServer) -> None:
    """DirectoryIndexingService receives a config with force_reindex=True."""
    server.realtime_indexing = None
    server.config.model_copy.return_value.indexing.force_reindex = True

    mock_stats = MagicMock(files_processed=3, chunks_created=10)
    with patch("chunkhound.mcp_server.base.DirectoryIndexingService") as MockDIS:
        MockDIS.return_value.process_directory = AsyncMock(return_value=mock_stats)
        await server._post_compaction_reindex()

    called_config = MockDIS.call_args.kwargs["config"]
    assert called_config.indexing.force_reindex is True


async def test_skips_when_no_services(server: ConcreteMCPServer) -> None:
    """_post_compaction_reindex exits early when services is None."""
    server.services = None

    with patch("chunkhound.mcp_server.base.DirectoryIndexingService") as MockDIS:
        await server._post_compaction_reindex()
        MockDIS.assert_not_called()


async def test_skips_when_no_target_path(server: ConcreteMCPServer) -> None:
    """_post_compaction_reindex exits early when _target_path is None."""
    server._target_path = None

    with patch("chunkhound.mcp_server.base.DirectoryIndexingService") as MockDIS:
        await server._post_compaction_reindex()
        MockDIS.assert_not_called()


async def test_reindex_error_is_swallowed(server: ConcreteMCPServer) -> None:
    """_post_compaction_reindex must not propagate exceptions — MCP server must stay alive."""
    server.realtime_indexing = MagicMock()
    server.realtime_indexing.clear_compaction_deferred_files = AsyncMock(
        side_effect=RuntimeError("boom")
    )
    # Should not raise
    await server._post_compaction_reindex()

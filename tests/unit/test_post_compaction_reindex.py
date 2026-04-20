"""Unit tests for MCPServerBase._post_compaction_reindex."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.mcp_server.base import MCPServerBase

from tests.unit.conftest import CaptureCoordinator


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

    class _OrderingCoordinator(CaptureCoordinator):
        async def process_directory(self, directory, **kwargs):
            assert realtime.deferred_files == set(), (
                "Post-compaction reindex must run only after stale deferred state is cleared"
            )
            return await super().process_directory(directory, **kwargs)

    coordinator = _OrderingCoordinator()
    server.services.indexing_coordinator = coordinator

    await server._post_compaction_reindex()

    assert coordinator.call_count == 1, "reindex coordinator was never called"


async def test_uses_force_reindex(server: ConcreteMCPServer) -> None:
    """Post-compaction reindex always forces a full reindex regardless of the current config value."""
    # Start with force_reindex=False to prove the implementation overrides it unconditionally.
    # Wire model_copy so the Pydantic update chain produces a real IndexingConfig(force_reindex=True).
    server.config.indexing = IndexingConfig(force_reindex=False)
    server.config.model_copy.side_effect = lambda update=None, **kw: types.SimpleNamespace(
        indexing=(update or {}).get("indexing", server.config.indexing)
    )
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator

    await server._post_compaction_reindex()

    assert coordinator.last_force_reindex is True


async def test_skips_when_no_services(server: ConcreteMCPServer) -> None:
    """_post_compaction_reindex exits early when services is None."""
    server.services = None
    # If the guard fails, accessing None.indexing_coordinator raises AttributeError → test fails.
    await server._post_compaction_reindex()


async def test_skips_when_no_target_path(server: ConcreteMCPServer) -> None:
    """_post_compaction_reindex exits early when _target_path is None."""
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator
    server._target_path = None

    await server._post_compaction_reindex()

    assert coordinator.call_count == 0


async def test_reindex_error_is_swallowed(server: ConcreteMCPServer) -> None:
    """_post_compaction_reindex must not propagate exceptions — MCP server must stay alive."""
    server.realtime_indexing = MagicMock()
    server.realtime_indexing.clear_compaction_deferred_files = AsyncMock(
        side_effect=RuntimeError("boom")
    )
    # Should not raise
    await server._post_compaction_reindex()


async def test_reindex_skipped_when_clear_raises(server: ConcreteMCPServer) -> None:
    """If clear_compaction_deferred_files raises, reindex is not attempted."""
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator
    server.realtime_indexing = MagicMock()
    server.realtime_indexing.clear_compaction_deferred_files = AsyncMock(
        side_effect=RuntimeError("fail")
    )

    await server._post_compaction_reindex()  # must not raise

    # A partial-clear state means deferred file records are stale; proceeding
    # with reindex on inconsistent state could silently leave entries un-marked.
    assert coordinator.call_count == 0, "reindex must not be attempted when clear raises"

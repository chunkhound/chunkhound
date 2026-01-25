"""Test that MCP server initialization is non-blocking (scan runs in background)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chunkhound.mcp_server.base import MCPServerBase


class ConcreteMCPServer(MCPServerBase):
    """Minimal concrete implementation for testing base class behavior."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


class TestNonBlockingInitialization:
    """Verify initialization returns before scan completes."""

    @pytest.mark.asyncio
    async def test_initialization_returns_before_scan_completes(self, tmp_path: Path):
        """Verify _scan_progress shows incomplete when initialize() returns."""
        # Create minimal config mock
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        # Mock create_services to avoid real DB operations
        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            # Mock EmbeddingManager
            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)

                # Initialize and immediately check state
                await server.initialize()

                # Key invariant: scan has NOT completed at this point
                progress = server._scan_progress

                # Verify we haven't completed scanning yet
                # (either still scanning, or scan hasn't started because it's deferred)
                assert progress.get("scan_completed_at") is None, (
                    "Initialization should return before scan completes"
                )

                # Cleanup
                await server.cleanup()

    @pytest.mark.asyncio
    async def test_scan_progress_fields_exist(self, tmp_path: Path):
        """Verify scan_progress dict has expected structure."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)
                await server.initialize()

                progress = server._scan_progress

                # All expected fields should exist
                assert "is_scanning" in progress
                assert "files_processed" in progress
                assert "chunks_created" in progress
                assert "scan_started_at" in progress
                assert "scan_completed_at" in progress

                await server.cleanup()

    @pytest.mark.asyncio
    async def test_initialized_flag_set_before_scan_starts(self, tmp_path: Path):
        """Verify _initialized is True before background scan begins."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path

        with patch("chunkhound.mcp_server.base.create_services") as mock_create:
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            with patch("chunkhound.mcp_server.base.EmbeddingManager"):
                server = ConcreteMCPServer(config=config)

                # Before initialize
                assert not server._initialized

                await server.initialize()

                # After initialize - should be True immediately
                assert server._initialized

                # But scan should not be complete yet
                assert server._scan_progress["scan_completed_at"] is None

                await server.cleanup()

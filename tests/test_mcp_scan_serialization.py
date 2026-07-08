"""Test that directory scans are serialized by _scan_lock and update progress correctly.

Validates the core guarantees of :meth:`MCPServerBase._run_directory_scan`:

1. A normal scan sets ``is_scanning``, ``files_processed``, ``chunks_created``
   and clears ``scan_error`` on success.
2. Concurrent calls are serialised — the second blocks on ``_scan_lock`` until
   the first releases it.
3. A scan failure resets ``is_scanning`` and records the error string.
4. The *trigger* label is threaded through to the debug log so operators can
   distinguish ``initial`` scans from ``realtime_resync`` scans.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.mcp_server.base import MCPServerBase


class _TestableMCPServer(MCPServerBase):
    """Concrete subclass that exercises base-class scan machinery directly."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


class TestScanLockSerialization:
    """Verify _run_directory_scan lock, progress, and error behavior."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def config(self, tmp_path: Path) -> MagicMock:
        """Return a minimal Config mock suitable for unit tests."""
        config = MagicMock()
        config.database.path = str(tmp_path / "test.db")
        config.embedding = None
        config.llm = None
        config.target_dir = tmp_path
        # index.realtime_backend is never accessed in _run_directory_scan
        # so we don't bother mocking indexing here.
        return config

    @pytest.fixture
    def server(self, config: MagicMock, tmp_path: Path) -> _TestableMCPServer:
        """Return a test server with minimal mocking of external dependencies.

        **Patched imports in this fixture:**

        - ``create_services`` — avoids real DB/embedding setup.
        - ``EmbeddingManager`` — avoids real provider registration.
        """
        with (
            patch("chunkhound.mcp_server.base.create_services") as mock_create,
            patch("chunkhound.mcp_server.base.EmbeddingManager"),
        ):
            mock_services = MagicMock()
            mock_services.provider.is_connected = False
            mock_create.return_value = mock_services

            srv = _TestableMCPServer(config=config)
            # Manually wire the minimal state that _run_directory_scan needs.
            srv._scan_target_path = tmp_path
            srv.services = MagicMock()
            srv.services.indexing_coordinator = MagicMock()
            return srv

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_indexing_service(**kwargs: object) -> MagicMock:
        """Build a ``DirectoryIndexingService`` mock whose ``process_directory``
        returns an object with the given keyword attributes."""
        mock_stats = MagicMock()
        for k, v in kwargs.items():
            setattr(mock_stats, k, v)
        return MagicMock(process_directory=AsyncMock(return_value=mock_stats))

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_normal_scan_updates_progress(
        self, server: _TestableMCPServer
    ) -> None:
        """A successful scan should set final stats and clear scanning state."""
        with patch(
            "chunkhound.mcp_server.base.DirectoryIndexingService",
            return_value=self._mock_indexing_service(
                files_processed=42,
                chunks_created=7,
            ),
        ):
            await server._run_directory_scan(
                server._scan_target_path, trigger="initial"
            )

        progress = server._scan_progress
        assert progress["files_processed"] == 42
        assert progress["chunks_created"] == 7
        assert progress["is_scanning"] is False
        assert progress["scan_error"] is None
        assert progress["query_ready_at"] is not None

    @pytest.mark.asyncio
    async def test_concurrent_scans_are_serialized(
        self, server: _TestableMCPServer,
    ) -> None:
        """A second call to _run_directory_scan must wait for the first to finish.

        The test uses an ``asyncio.Event`` to make the first scan block inside
        ``process_directory`` so we can observe that the second scan stalls on
        ``_scan_lock`` and only runs after the first completes.
        """
        release_event = asyncio.Event()
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        process_calls: list[str] = []  # tracks which triggers were executed

        async def monitored_process(
            target_path: Path, *, no_embeddings: bool = False  # noqa: ARG001
        ) -> MagicMock:
            trigger = _current_trigger  # captured from outer scope
            process_calls.append(trigger)
            if trigger == "first":
                first_entered.set()
                await release_event.wait()
            else:
                second_entered.set()
            return MagicMock(files_processed=10, chunks_created=3)

        _current_trigger = "first"
        mock_svc = MagicMock(
            process_directory=AsyncMock(side_effect=monitored_process),
        )

        with patch(
            "chunkhound.mcp_server.base.DirectoryIndexingService",
            return_value=mock_svc,
        ):
            # Start first scan — it will block inside process_directory.
            task1 = asyncio.create_task(
                server._run_directory_scan(
                    server._scan_target_path, trigger="first",
                ),
            )
            # Wait until task1 has acquired the lock and entered process_directory.
            await asyncio.wait_for(first_entered.wait(), timeout=5.0)

            assert process_calls == ["first"], (
                "First scan should have entered process_directory"
            )

            # Start second scan — it should block at the lock.
            _current_trigger = "second"
            task2 = asyncio.create_task(
                server._run_directory_scan(
                    server._scan_target_path, trigger="second",
                ),
            )
            await asyncio.sleep(0)

            # The second scan must NOT have entered process_directory yet.
            assert not second_entered.is_set()
            assert process_calls == ["first"], (
                "Second scan should be blocked by _scan_lock"
            )

            # Release the first scan.
            release_event.set()
            await asyncio.wait_for(second_entered.wait(), timeout=5.0)
            await asyncio.wait_for(asyncio.gather(task1, task2), timeout=5.0)

        assert process_calls == ["first", "second"], (
            "Both scans should have executed sequentially"
        )

    @pytest.mark.asyncio
    async def test_failure_resets_scan_state(
        self, server: _TestableMCPServer,
    ) -> None:
        """A scan that raises must clear ``is_scanning`` and record the error."""
        async def failing_process(
            target_path: Path, *, no_embeddings: bool = False  # noqa: ARG001
        ) -> MagicMock:
            raise RuntimeError("Simulated scan failure")

        with patch(
            "chunkhound.mcp_server.base.DirectoryIndexingService",
            return_value=MagicMock(
                process_directory=AsyncMock(side_effect=failing_process),
            ),
        ):
            with pytest.raises(RuntimeError, match="Simulated scan failure"):
                await server._run_directory_scan(
                    server._scan_target_path, trigger="initial",
                )

        assert server._scan_progress["is_scanning"] is False
        assert "Simulated scan failure" in server._scan_progress["scan_error"]
        assert server._scan_progress["query_ready_at"] is None

    @pytest.mark.asyncio
    async def test_trigger_label_appears_in_logs(
        self, server: _TestableMCPServer,
    ) -> None:
        """The *trigger* string should propagate to debug log output."""
        with patch(
            "chunkhound.mcp_server.base.DirectoryIndexingService",
            return_value=self._mock_indexing_service(
                files_processed=3,
                chunks_created=1,
            ),
        ):
            with patch.object(server, "debug_log") as mock_log:
                await server._run_directory_scan(
                    server._scan_target_path, trigger="realtime_resync",
                )

        logged_text = " ".join(str(c) for c in mock_log.call_args_list)
        assert "realtime_resync" in logged_text, (
            f"Trigger 'realtime_resync' should appear in debug log, got: {logged_text}"
        )

    @pytest.mark.asyncio
    async def test_second_scan_preserves_query_ready_timestamp(
        self, server: _TestableMCPServer,
    ) -> None:
        """Running a second scan should preserve query-ready semantics.

        ``query_ready_at`` means at least one successful index exists, so a
        follow-up scan should refresh that timestamp rather than clear it.
        """
        with patch(
            "chunkhound.mcp_server.base.DirectoryIndexingService",
            return_value=self._mock_indexing_service(
                files_processed=5,
                chunks_created=2,
            ),
        ):
            await server._run_directory_scan(
                server._scan_target_path, trigger="initial",
            )
            first_query_ready_at = server._scan_progress["query_ready_at"]
            assert first_query_ready_at is not None

            await server._run_directory_scan(
                server._scan_target_path, trigger="realtime_resync",
            )
            second_query_ready_at = server._scan_progress["query_ready_at"]
            assert second_query_ready_at is not None

    @pytest.mark.asyncio
    async def test_run_directory_scan_requires_services(
        self, config: MagicMock, tmp_path: Path,
    ) -> None:
        """Calling _run_directory_scan with services=None must raise immediately."""
        with (
            patch("chunkhound.mcp_server.base.create_services"),
            patch("chunkhound.mcp_server.base.EmbeddingManager"),
        ):
            srv = _TestableMCPServer(config=config)
            srv._scan_target_path = tmp_path
            srv.services = None  # simulate uninitialised state

            with pytest.raises(RuntimeError, match="Services were not initialized"):
                await srv._run_directory_scan(tmp_path, trigger="initial")

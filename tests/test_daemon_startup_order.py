"""Test that the daemon publishes IPC socket+lock BEFORE the startup barrier blocks.

Verifies the ``ChunkHoundDaemon.run()`` call-order invariant:

    daemon_publish.start → initialization_complete.set → startup_barrier.start
    → startup.complete

This guarantees that ``tools/list`` (which waits on ``_initialization_complete``)
is available before the daemon blocks on Watchman readiness, so clients can
immediately discover tools while the sidecar is still starting.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.config import Config
from chunkhound.daemon.server import ChunkHoundDaemon


class TestDaemonStartupOrder:
    """Verify the publish-then-barrier ordering invariant."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> Config:
        """Create minimal config with a valid database path."""
        db_path = tmp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return Config(
            target_dir=tmp_path,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={},
        )

    @pytest.mark.asyncio
    async def test_publish_before_barrier(
        self,
        config: Config,
        tmp_path: Path,
    ) -> None:
        """Verify call order: daemon_publish.start → set → startup_barrier.start.

        Uses mock injection at the method level so the daemon's real ``run()``
        loop executes but skips heavy I/O (IPC bind, discovery filesystem ops,
        DB connect, realtime start).  Only the call-order of three critical
        operations is checked.
        """
        call_order: list[str] = []
        socket_path = str(tmp_path / "test.sock")
        project_dir = tmp_path

        daemon = ChunkHoundDaemon(
            config,
            args=MagicMock(path=tmp_path),
            socket_path=socket_path,
            project_dir=project_dir,
        )

        # ---- Record method invocations ----
        orig_start_phase = daemon._start_startup_phase

        def record_start_phase(phase_name: str) -> None:
            call_order.append(f"{phase_name}.start")
            orig_start_phase(phase_name)

        orig_set = daemon._initialization_complete.set

        def record_set() -> None:
            call_order.append("initialization_complete.set")
            orig_set()

        orig_complete_startup = daemon._complete_startup

        def record_complete_startup() -> None:
            call_order.append("startup.complete")
            orig_complete_startup()

        # ---- Mock heavy I/O paths ----
        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=None)
        mock_server.__aexit__ = AsyncMock(return_value=None)

        # Inject recorders *before* wiring mocks so the recorder chain is in place.
        daemon._start_startup_phase = record_start_phase  # type: ignore[assignment]
        daemon._initialization_complete.set = record_set  # type: ignore[assignment]
        daemon._complete_startup = record_complete_startup  # type: ignore[assignment]

        # Mock initialize() — fast path: no DB, no realtime, no embeddings.
        async def mock_initialize() -> None:
            daemon._start_startup_phase("initialize")
            daemon._initialized = True
            daemon._scan_target_path = daemon._normalize_requested_target_path(
                tmp_path
            )
            daemon._startup_publish_complete.clear()
            daemon._deferred_start_task = asyncio.create_task(asyncio.sleep(0))
            daemon._startup_failure_message = None

        daemon.initialize = mock_initialize  # type: ignore[assignment]

        # Mock await_startup_barrier — record the phase, don't block.
        async def mock_await_barrier() -> None:
            daemon._start_startup_phase("startup_barrier")
            # Don't block on Watchman/sidecar readiness.

        daemon.await_startup_barrier = mock_await_barrier  # type: ignore[assignment]

        # Mock discovery I/O — no real lock files or sockets.
        daemon._discovery.write_lock = MagicMock()  # type: ignore[assignment]
        daemon._discovery.read_lock = MagicMock(  # type: ignore[assignment]
            return_value={"pid": os.getpid()}
        )
        daemon._discovery.write_registry_entry = MagicMock()  # type: ignore[assignment]
        daemon._discovery.remove_lock = MagicMock()  # type: ignore[assignment]

        # Set shutdown event so the server loop exits immediately after
        # await_startup_barrier completes (avoids blocking on
        # _shutdown_event.wait() inside ``async with server:``).
        daemon._shutdown_event.set()

        # Mock IPC server creation — avoid binding to a real socket.
        with patch(
            "chunkhound.daemon.server.ipc.create_server",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = (mock_server, socket_path)
            await daemon.run()

        # ---- Assert ordering ----
        _assert_key_markers(call_order)

    @pytest.mark.asyncio
    async def test_tools_list_available_before_barrier(
        self,
        config: Config,
        tmp_path: Path,
    ) -> None:
        """Verify that _initialization_complete is set before the barrier blocks.

        This is a concrete consequence of the ordering invariant checked by
        ``test_publish_before_barrier``: if the event is set first, then
        any concurrent call to ``tools/list`` (which awaits the event with a
        timeout) will return tools immediately instead of timing out.
        """
        call_order: list[str] = []
        socket_path = str(tmp_path / "test.sock")
        project_dir = tmp_path

        daemon = ChunkHoundDaemon(
            config,
            args=MagicMock(path=tmp_path),
            socket_path=socket_path,
            project_dir=project_dir,
        )

        # ---- Recorders ----
        orig_set = daemon._initialization_complete.set

        def record_set() -> None:
            call_order.append("initialization_complete.set")
            orig_set()

        daemon._initialization_complete.set = record_set  # type: ignore[assignment]

        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=None)
        mock_server.__aexit__ = AsyncMock(return_value=None)

        # Track when await_startup_barrier *enters* by wrapping _start_startup_phase
        orig_start_phase = daemon._start_startup_phase

        def record_start_phase(phase_name: str) -> None:
            call_order.append(f"{phase_name}.start")
            orig_start_phase(phase_name)

        daemon._start_startup_phase = record_start_phase  # type: ignore[assignment]

        async def mock_initialize() -> None:
            daemon._start_startup_phase("initialize")
            daemon._initialized = True
            daemon._scan_target_path = daemon._normalize_requested_target_path(
                tmp_path
            )
            daemon._startup_publish_complete.clear()
            daemon._deferred_start_task = asyncio.create_task(asyncio.sleep(0))
            daemon._startup_failure_message = None

        daemon.initialize = mock_initialize  # type: ignore[assignment]

        async def mock_await_barrier() -> None:
            daemon._start_startup_phase("startup_barrier")

        daemon.await_startup_barrier = mock_await_barrier  # type: ignore[assignment]

        daemon._discovery.write_lock = MagicMock()  # type: ignore[assignment]
        daemon._discovery.read_lock = MagicMock(  # type: ignore[assignment]
            return_value={"pid": os.getpid()}
        )
        daemon._discovery.write_registry_entry = MagicMock()  # type: ignore[assignment]
        daemon._discovery.remove_lock = MagicMock()  # type: ignore[assignment]

        daemon._shutdown_event.set()

        with patch(
            "chunkhound.daemon.server.ipc.create_server",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = (mock_server, socket_path)
            await daemon.run()

        # The event must be set before the barrier phase starts.
        relevant = [c for c in call_order if c in {
            "initialization_complete.set",
            "startup_barrier.start",
        }]
        assert relevant == [
            "initialization_complete.set",
            "startup_barrier.start",
        ], (
            "Expected initialization_complete.set → startup_barrier.start, "
            f"got: {relevant}"
        )

    @pytest.mark.asyncio
    async def test_graceful_shutdown_removes_artifacts_before_cleanup(
        self,
        config: Config,
        tmp_path: Path,
    ) -> None:
        """Verify artifact removal happens BEFORE cleanup() in _graceful_shutdown."""
        call_order: list[str] = []
        socket_path = str(tmp_path / "test.sock")
        project_dir = tmp_path

        daemon = ChunkHoundDaemon(
            config,
            args=MagicMock(path=tmp_path),
            socket_path=socket_path,
            project_dir=project_dir,
        )

        # --- Inject recorders onto helpers ---
        orig_remove_lock = daemon._discovery.remove_lock

        def record_remove_lock() -> None:
            call_order.append("remove_lock")
            orig_remove_lock()

        orig_unlink = os.unlink

        def record_unlink(path: str) -> None:
            call_order.append("unlink_socket")
            orig_unlink(path)

        daemon._discovery.remove_lock = record_remove_lock  # type: ignore[assignment]

        # Mock cleanup() so it doesn't actually run DB teardown
        orig_cleanup = daemon.cleanup

        async def record_cleanup() -> None:
            call_order.append("cleanup.start")
            await orig_cleanup()

        daemon.cleanup = record_cleanup  # type: ignore[assignment]

        # Write a lock file with the current PID so _graceful_shutdown()'s
        # live PID check finds a match and runs shared-artifact removal.
        lock_path = daemon._discovery.get_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps({
                "pid": os.getpid(),
                "socket_path": socket_path,
                "started_at": 0.0,
                "project_dir": str(daemon._project_dir),
                "auth_token": "test-token",
            })
        )
        daemon._socket_path = socket_path

        with patch("chunkhound.daemon.server.os.unlink", new=record_unlink):
            await daemon._graceful_shutdown()

        assert len(call_order) >= 3, (
            "Expected remove_lock + unlink_socket + cleanup.start, "
            f"got: {call_order}"
        )
        cleanup_idx = call_order.index("cleanup.start")
        for artifact_step in ["remove_lock", "unlink_socket"]:
            assert call_order.index(artifact_step) < cleanup_idx, (
                f"Expected {artifact_step} before cleanup.start, got: {call_order}"
            )

    @pytest.mark.asyncio
    async def test_run_barrier_failure_triggers_artifact_cleanup_before_cleanup(
        self,
        config: Config,
        tmp_path: Path,
    ) -> None:
        """Barrier failure after eager publication must clean artifacts immediately."""
        call_order: list[str] = []
        socket_path = str(tmp_path / "test.sock")

        daemon = ChunkHoundDaemon(
            config,
            args=MagicMock(path=tmp_path),
            socket_path=socket_path,
            project_dir=tmp_path,
        )

        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=None)
        mock_server.__aexit__ = AsyncMock(return_value=None)

        async def mock_initialize() -> None:
            daemon._initialized = True
            daemon._scan_target_path = daemon._normalize_requested_target_path(
                tmp_path
            )
            daemon._startup_publish_complete.clear()
            daemon._deferred_start_task = asyncio.create_task(asyncio.sleep(0))
            daemon._startup_failure_message = None

        async def failing_barrier() -> None:
            raise RuntimeError("startup barrier failed")

        def record_remove_lock() -> None:
            call_order.append("remove_lock")

        def record_unlink(path: str) -> None:
            call_order.append("unlink_socket")

        async def record_cleanup() -> None:
            call_order.append("cleanup.start")

        daemon.initialize = mock_initialize  # type: ignore[assignment]
        daemon.await_startup_barrier = failing_barrier  # type: ignore[assignment]
        daemon.cleanup = record_cleanup  # type: ignore[assignment]
        daemon._discovery.write_lock = MagicMock()  # type: ignore[assignment]
        daemon._discovery.read_lock = MagicMock(  # type: ignore[assignment]
            return_value={"pid": os.getpid()}
        )
        daemon._discovery.write_registry_entry = MagicMock()  # type: ignore[assignment]
        daemon._discovery.remove_lock = record_remove_lock  # type: ignore[assignment]

        with patch(
            "chunkhound.daemon.server.ipc.create_server",
            new_callable=AsyncMock,
        ) as mock_create, patch(
            "chunkhound.daemon.server.os.unlink",
            new=record_unlink,
        ):
            mock_create.return_value = (mock_server, socket_path)
            with pytest.raises(RuntimeError, match="startup barrier failed"):
                await daemon.run()

        cleanup_idx = call_order.index("cleanup.start")
        for artifact_step in ["remove_lock", "unlink_socket"]:
            assert call_order.index(artifact_step) < cleanup_idx, (
                f"Expected {artifact_step} before cleanup.start, got: {call_order}"
            )

    @pytest.mark.asyncio
    async def test_graceful_shutdown_leaves_replacement_registry_entry_during_interleaving(
        self,
        config: Config,
        tmp_path: Path,
    ) -> None:
        """Old daemon shutdown must not touch the shared registry path."""
        socket_path = str(tmp_path / "test.sock")

        daemon = ChunkHoundDaemon(
            config,
            args=MagicMock(path=tmp_path),
            socket_path=socket_path,
            project_dir=tmp_path,
        )

        discovery = daemon._discovery
        registry_path = discovery.get_registry_entry_path()
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(
                {
                    "project_dir": str(tmp_path),
                    "pid": os.getpid(),
                    "socket_path": socket_path,
                    "lock_path": str(discovery.get_lock_path()),
                    "started_at": 0.0,
                }
            )
        )

        replacement_entry = {
            "project_dir": str(tmp_path),
            "pid": os.getpid() + 1,
            "socket_path": "replacement.sock",
            "lock_path": str(discovery.get_lock_path()),
            "started_at": 1.0,
        }

        def replace_registry() -> None:
            registry_path.write_text(json.dumps(replacement_entry))

        discovery.read_lock = MagicMock(return_value={"pid": os.getpid()})  # type: ignore[assignment]
        discovery.remove_lock = MagicMock(side_effect=replace_registry)  # type: ignore[assignment]
        daemon.cleanup = AsyncMock()  # type: ignore[assignment]

        with patch("chunkhound.daemon.server.os.unlink", new=lambda _path: None):
            await daemon._graceful_shutdown()

        assert json.loads(registry_path.read_text()) == replacement_entry, (
            "Shutdown should leave a replacement daemon's registry entry intact"
        )

    @pytest.mark.asyncio
    async def test_graceful_shutdown_skips_shared_artifact_removal_without_live_lock(
        self,
        config: Config,
        tmp_path: Path,
    ) -> None:
        """Without a live lock file, shared artifact removal must not run."""
        socket_path = str(tmp_path / "test.sock")
        project_dir = tmp_path

        daemon = ChunkHoundDaemon(
            config,
            args=MagicMock(path=tmp_path),
            socket_path=socket_path,
            project_dir=project_dir,
        )

        removed_lock = False
        unlinked_socket = False

        def record_remove_lock() -> None:
            nonlocal removed_lock
            removed_lock = True

        def record_unlink(path: str) -> None:
            nonlocal unlinked_socket
            unlinked_socket = True

        daemon._discovery.remove_lock = record_remove_lock  # type: ignore[assignment]

        # No lock file on disk — _graceful_shutdown should detect this
        # via the live PID check and skip artifact removal.
        assert not daemon._discovery.get_lock_path().exists()

        with patch("chunkhound.daemon.server.os.unlink", new=record_unlink):
            await daemon._graceful_shutdown()

        assert not removed_lock, (
            "remove_lock should not be called when no lock file is present"
        )
        assert unlinked_socket, (
            "socket unlink should always run even without lock ownership"
        )

    @pytest.mark.asyncio
    async def test_run_race_loss_cleans_up_socket(
        self,
        config: Config,
        tmp_path: Path,
    ) -> None:
        """PID mismatch in run() must clean up the bound socket before returning."""
        socket_path = str(tmp_path / "test.sock")

        daemon = ChunkHoundDaemon(
            config,
            args=MagicMock(path=tmp_path),
            socket_path=socket_path,
            project_dir=tmp_path,
        )

        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=None)
        mock_server.__aexit__ = AsyncMock(return_value=None)

        async def mock_initialize() -> None:
            daemon._initialized = True
            daemon._scan_target_path = daemon._normalize_requested_target_path(
                tmp_path
            )
            daemon._startup_publish_complete.clear()
            daemon._deferred_start_task = asyncio.create_task(asyncio.sleep(0))
            daemon._startup_failure_message = None

        daemon.initialize = mock_initialize  # type: ignore[assignment]

        # read_lock returns a different PID to simulate race loss
        daemon._discovery.write_lock = MagicMock()  # type: ignore[assignment]
        daemon._discovery.read_lock = MagicMock(  # type: ignore[assignment]
            return_value={"pid": 999999}
        )

        socket_unlinked = False

        def record_unlink(path: str) -> None:
            nonlocal socket_unlinked
            socket_unlinked = True

        daemon._shutdown_event.set()

        with patch(
            "chunkhound.daemon.server.ipc.create_server",
            new_callable=AsyncMock,
        ) as mock_create, patch(
            "chunkhound.daemon.server.os.unlink",
            new=record_unlink,
        ):
            mock_create.return_value = (mock_server, socket_path)
            await daemon.run()

        assert socket_unlinked, (
            "Socket should be unlinked even when lock is owned by another daemon"
        )


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------

_KEY_MARKERS = frozenset({
    "daemon_publish.start",
    "initialization_complete.set",
    "startup_barrier.start",
    "startup.complete",
})


def _assert_key_markers(call_order: list[str]) -> None:
    """Assert that the three critical markers appear in the expected order.

    The full call order includes phases like ``initialize.start`` and
    ``daemon_publish.complete``; we ignore those and only check the
    relative position of the three invariants.
    """
    assert len(call_order) >= 3, (
        f"Not enough recorded calls to verify ordering: {call_order}"
    )

    relevant = [c for c in call_order if c in _KEY_MARKERS]
    assert relevant == [
        "daemon_publish.start",
        "initialization_complete.set",
        "startup_barrier.start",
        "startup.complete",
    ], (
        "Expected daemon_publish.start → initialization_complete.set → "
        f"startup_barrier.start → startup.complete, got: {relevant}"
    )



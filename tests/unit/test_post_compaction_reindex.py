"""Unit tests for MCPServerBase._post_compaction_reindex."""

from __future__ import annotations

import asyncio
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.exceptions import CompactionError
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.services.compaction_service import CompactionService
from tests.unit.conftest import CaptureCoordinator


class ConcreteMCPServer(MCPServerBase):
    """Minimal concrete implementation for testing base class behavior."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


def _realtime_mock() -> MagicMock:
    realtime = MagicMock()
    realtime.drain_compaction_deferred_directories = AsyncMock(return_value=set())
    realtime.restore_compaction_deferred_directories = AsyncMock()
    realtime.replay_compaction_deferred_directory = AsyncMock()
    realtime.drain_compaction_deferred_file_work = AsyncMock(
        return_value=(set(), set())
    )
    realtime.restore_compaction_deferred_files = AsyncMock()
    realtime.restore_compaction_deferred_removals = AsyncMock()
    return realtime


class RemovalReplayState:
    """Stateful realtime fake for deferred removal replay behavior."""

    def __init__(
        self,
        *,
        fail_path: str | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.deferred_files = {"a.py", "b.py", "c.py", "stale.py"}
        self.deferred_removals = {"a.py", "b.py", "c.py"}
        self.deferred_directories: set[str] = set()
        self.replay_attempts: list[str] = []
        self.deleted: set[str] = set()
        self.fail_path = fail_path
        self.fail_with = fail_with
        self._failed_once = False

    async def drain_compaction_deferred_directories(self) -> set[str]:
        drained = set(self.deferred_directories)
        self.deferred_directories.clear()
        return drained

    async def restore_compaction_deferred_directories(
        self, paths: set[str]
    ) -> None:
        self.deferred_directories.update(paths)

    async def replay_compaction_deferred_directory(self, path: str) -> None:
        pass

    async def drain_compaction_deferred_file_work(self) -> tuple[set[str], set[str]]:
        drained = set(self.deferred_files)
        removals = set(self.deferred_removals)
        self.deferred_files.clear()
        self.deferred_removals.clear()
        return drained, removals

    async def restore_compaction_deferred_files(self, paths: set[str]) -> None:
        self.deferred_files.update(paths)

    async def restore_compaction_deferred_removals(self, paths: set[str]) -> None:
        self.deferred_removals.update(paths)
        self.deferred_files.update(paths)

    async def replay_compaction_deferred_removal(self, path: str) -> None:
        self.replay_attempts.append(path)
        if path == self.fail_path and not self._failed_once:
            self._failed_once = True
            if self.fail_with is None:
                raise AssertionError("fail_with must be set when fail_path is set")
            if isinstance(self.fail_with, CompactionError):
                self.deferred_files.add(path)
                self.deferred_removals.add(path)
            raise self.fail_with

        self.deleted.add(path)
        self.deferred_files.discard(path)
        self.deferred_removals.discard(path)


@pytest.fixture
def server(tmp_path: Path) -> ConcreteMCPServer:
    config = MagicMock()
    config.database.path = str(tmp_path / "test.db")
    config.target_dir = tmp_path

    with (
        patch("chunkhound.mcp_server.base.create_services"),
        patch("chunkhound.mcp_server.base.EmbeddingManager"),
    ):
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

        async def drain_compaction_deferred_directories(self) -> set[str]:
            return set()

        async def restore_compaction_deferred_directories(
            self, paths: set[str]
        ) -> None:
            pass

        async def replay_compaction_deferred_directory(self, path: str) -> None:
            pass

        async def drain_compaction_deferred_file_work(
            self,
        ) -> tuple[set[str], set[str]]:
            drained = set(self.deferred_files)
            self.deferred_files.clear()
            return drained, set()

        async def restore_compaction_deferred_files(self, paths: set[str]) -> None:
            self.deferred_files.update(paths)

        async def restore_compaction_deferred_removals(self, paths: set[str]) -> None:
            pass

    realtime = RealtimeStub()
    server.realtime_indexing = realtime

    class _OrderingCoordinator(CaptureCoordinator):
        async def process_directory(self, directory, **kwargs):
            assert realtime.deferred_files == set(), (
                "Post-compaction reindex must run only after stale deferred "
                "state is cleared"
            )
            return await super().process_directory(directory, **kwargs)

    coordinator = _OrderingCoordinator()
    server.services.indexing_coordinator = coordinator

    await server._post_compaction_reindex()

    assert coordinator.call_count == 1, "reindex coordinator was never called"


async def test_uses_force_reindex(server: ConcreteMCPServer) -> None:
    """Post-compaction reindex always forces a full reindex."""
    # Start with force_reindex=False to prove the implementation overrides
    # it unconditionally. Wire model_copy so the Pydantic update chain
    # produces a real IndexingConfig(force_reindex=True).
    server.config.indexing = IndexingConfig(force_reindex=False)
    server.config.model_copy.side_effect = (
        lambda update=None, **kw: types.SimpleNamespace(
            indexing=(update or {}).get("indexing", server.config.indexing)
        )
    )
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator
    server.realtime_indexing = _realtime_mock()

    await server._post_compaction_reindex()

    assert coordinator.last_force_reindex is True


async def test_skips_when_no_services(server: ConcreteMCPServer) -> None:
    """_post_compaction_reindex exits early when services is None."""
    server.services = None
    # If the guard fails, accessing None.indexing_coordinator raises
    # AttributeError and the test fails.
    await server._post_compaction_reindex()


async def test_skips_when_no_target_path(server: ConcreteMCPServer) -> None:
    """_post_compaction_reindex exits early when _target_path is None."""
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator
    server._target_path = None

    await server._post_compaction_reindex()

    assert coordinator.call_count == 0


async def test_reindex_error_is_swallowed(server: ConcreteMCPServer) -> None:
    """_post_compaction_reindex must re-raise so CompactionService records it."""
    server.realtime_indexing = _realtime_mock()
    server.realtime_indexing.drain_compaction_deferred_file_work = AsyncMock(
        return_value=(set(), set())
    )
    server.services.indexing_coordinator = MagicMock()
    server.services.indexing_coordinator.process_directory = AsyncMock(
        side_effect=RuntimeError("boom")
    )

    with pytest.raises(RuntimeError, match="boom"):
        await server._post_compaction_reindex()

    status = server.get_background_compaction_status()
    assert status["phase"] == "failed"
    assert status["pending_recovery"] is True
    assert status["retry_attempted"] is False
    assert status["last_error"] == "boom"


async def test_reindex_skipped_when_clear_raises(server: ConcreteMCPServer) -> None:
    """If deferred-state drain raises, reindex is not attempted."""
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator
    server.realtime_indexing = _realtime_mock()
    server.realtime_indexing.drain_compaction_deferred_file_work = AsyncMock(
        side_effect=RuntimeError("fail")
    )

    with pytest.raises(RuntimeError, match="fail"):
        await server._post_compaction_reindex()

    # A partial-clear state means deferred file records are stale; proceeding
    # with reindex on inconsistent state could silently leave entries un-marked.
    assert coordinator.call_count == 0, (
        "reindex must not be attempted when clear raises"
    )


async def test_atomic_file_work_drain_failure_preserves_state(
    server: ConcreteMCPServer,
) -> None:
    """If the atomic file-work drain fails, reindex is not attempted."""
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator
    server.realtime_indexing = _realtime_mock()
    server.realtime_indexing.drain_compaction_deferred_file_work = AsyncMock(
        side_effect=RuntimeError("file work drain failed")
    )
    server.realtime_indexing.restore_compaction_deferred_files = AsyncMock()
    server.realtime_indexing.restore_compaction_deferred_removals = AsyncMock()

    with pytest.raises(RuntimeError, match="file work drain failed"):
        await server._post_compaction_reindex()

    assert coordinator.call_count == 0
    status = server.get_background_compaction_status()
    assert status["phase"] == "failed"
    assert status["pending_recovery"] is True


async def test_deferred_directories_replayed_before_removal_drain(
    server: ConcreteMCPServer,
) -> None:
    """Directory cleanup replay must run before deferred removals are drained."""
    realtime = _realtime_mock()
    realtime.drain_compaction_deferred_directories = AsyncMock(
        return_value={"/tmp/deleted"}
    )
    calls: list[str] = []

    async def replay_directory(path: str) -> None:
        calls.append(f"dir:{path}")

    async def drain_file_work() -> tuple[set[str], set[str]]:
        calls.append("drain_removals")
        return set(), set()

    realtime.replay_compaction_deferred_directory = AsyncMock(
        side_effect=replay_directory
    )
    realtime.drain_compaction_deferred_file_work = AsyncMock(
        side_effect=drain_file_work
    )
    server.realtime_indexing = realtime
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator

    await server._post_compaction_reindex()

    assert calls == ["dir:/tmp/deleted", "drain_removals"]
    assert coordinator.call_count == 1


async def test_post_compaction_reindex_uses_atomic_file_work_drain(
    server: ConcreteMCPServer,
) -> None:
    """File/removal deferred state must be drained in one atomic snapshot."""
    realtime = _realtime_mock()
    realtime.drain_compaction_deferred_files = AsyncMock(
        side_effect=AssertionError("separate file drain should not be used")
    )
    realtime.drain_compaction_deferred_removals = AsyncMock(
        side_effect=AssertionError("separate removal drain should not be used")
    )
    server.realtime_indexing = realtime
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator

    await server._post_compaction_reindex()

    assert coordinator.call_count == 1


async def test_deferred_directory_replay_failure_restores_directory_only(
    server: ConcreteMCPServer,
) -> None:
    """Directory replay failures preserve only unresolved directory work."""
    realtime = _realtime_mock()
    realtime.drain_compaction_deferred_directories = AsyncMock(
        return_value={"a/old", "b/old"}
    )
    attempts: list[str] = []

    async def replay_directory(path: str) -> None:
        attempts.append(path)
        if path == "a/old":
            raise CompactionError("lookup busy", operation="search")

    realtime.replay_compaction_deferred_directory = AsyncMock(
        side_effect=replay_directory
    )
    server.realtime_indexing = realtime
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator

    with pytest.raises(CompactionError, match="lookup busy"):
        await server._post_compaction_reindex()

    assert attempts == ["a/old"]
    assert coordinator.call_count == 0
    status = server.get_background_compaction_status()
    assert status["pending_recovery"] is True
    last_error = status["last_error"]
    assert "deferred directory cleanup blocked by compaction for a/old" in last_error


async def test_deferred_directory_generic_failure_restores_unresolved_directories(
    server: ConcreteMCPServer,
) -> None:
    """Generic directory replay failures are retryable and skip reindex."""
    realtime = _realtime_mock()
    realtime.drain_compaction_deferred_directories = AsyncMock(
        return_value={"a/old", "b/old"}
    )
    attempts: list[str] = []

    async def replay_directory(path: str) -> None:
        attempts.append(path)
        if path == "a/old":
            raise RuntimeError("lookup failed")

    realtime.replay_compaction_deferred_directory = AsyncMock(
        side_effect=replay_directory
    )
    server.realtime_indexing = realtime
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator

    with pytest.raises(RuntimeError, match="lookup failed"):
        await server._post_compaction_reindex()

    assert attempts == ["a/old"]
    assert coordinator.call_count == 0
    status = server.get_background_compaction_status()
    assert status["pending_recovery"] is True
    assert "deferred directory cleanup failed for a/old" in status["last_error"]


async def test_reindex_failure_restores_deferred_state(
    server: ConcreteMCPServer,
) -> None:
    """Failed catch-up must restore deferred paths for later recovery."""
    drained = {"stale.py"}
    server.realtime_indexing = _realtime_mock()
    server.realtime_indexing.drain_compaction_deferred_file_work = AsyncMock(
        return_value=(drained, set())
    )
    server.realtime_indexing.restore_compaction_deferred_files = AsyncMock()
    server.services.indexing_coordinator = MagicMock()
    server.services.indexing_coordinator.process_directory = AsyncMock(
        side_effect=RuntimeError("reindex failed")
    )

    with pytest.raises(RuntimeError, match="reindex failed"):
        await server._post_compaction_reindex()

    status = server.get_background_compaction_status()
    assert status["phase"] == "failed"
    assert status["pending_recovery"] is True


async def test_replay_failure_restores_only_unresolved_removals(
    server: ConcreteMCPServer,
) -> None:
    """Generic partial replay failure keeps failed and unattempted work retryable."""
    realtime = RemovalReplayState(
        fail_path="b.py",
        fail_with=RuntimeError("delete replay failed"),
    )
    server.realtime_indexing = realtime
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator

    with pytest.raises(RuntimeError, match="delete replay failed"):
        await server._post_compaction_reindex()

    assert realtime.replay_attempts == ["a.py", "b.py"]
    assert realtime.deleted == {"a.py"}
    assert realtime.deferred_files == {"b.py", "c.py", "stale.py"}
    assert realtime.deferred_removals == {"b.py", "c.py"}
    assert coordinator.call_count == 0
    status = server.get_background_compaction_status()
    assert status["pending_recovery"] is True
    assert "delete replay failed" in status["last_error"]


async def test_retryable_replay_failure_restores_only_retryable_subset(
    server: ConcreteMCPServer,
) -> None:
    """CompactionError keeps failed and unattempted removals retryable."""
    realtime = RemovalReplayState(
        fail_path="b.py",
        fail_with=CompactionError("retry this removal", operation="compaction"),
    )
    server.realtime_indexing = realtime
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator

    with pytest.raises(CompactionError, match="retry this removal"):
        await server._post_compaction_reindex()

    assert realtime.replay_attempts == ["a.py", "b.py"]
    assert realtime.deleted == {"a.py"}
    assert realtime.deferred_files == {"b.py", "c.py", "stale.py"}
    assert realtime.deferred_removals == {"b.py", "c.py"}
    assert coordinator.call_count == 0
    status = server.get_background_compaction_status()
    assert status["pending_recovery"] is True
    assert "retry this removal" in status["last_error"]


async def test_ensure_services_retries_partial_deferred_removal_replay(
    server: ConcreteMCPServer,
) -> None:
    """Retry replays only the unresolved removal subset and clears recovery."""

    class RealtimeState:
        def __init__(self) -> None:
            self.deferred_files = {"a.py", "b.py", "c.py"}
            self.deferred_removals = {"a.py", "b.py", "c.py"}
            self.deferred_directories: set[str] = set()
            self.replay_attempts: list[str] = []
            self.deleted: set[str] = set()
            self.fail_b_once = True

        async def drain_compaction_deferred_directories(self) -> set[str]:
            drained = set(self.deferred_directories)
            self.deferred_directories.clear()
            return drained

        async def restore_compaction_deferred_directories(
            self, paths: set[str]
        ) -> None:
            self.deferred_directories.update(paths)

        async def replay_compaction_deferred_directory(self, path: str) -> None:
            pass

        async def drain_compaction_deferred_file_work(
            self,
        ) -> tuple[set[str], set[str]]:
            drained = set(self.deferred_files)
            removals = set(self.deferred_removals)
            self.deferred_files.clear()
            self.deferred_removals.clear()
            return drained, removals

        async def restore_compaction_deferred_files(self, paths: set[str]) -> None:
            self.deferred_files.update(paths)

        async def restore_compaction_deferred_removals(self, paths: set[str]) -> None:
            self.deferred_removals.update(paths)
            self.deferred_files.update(paths)

        async def replay_compaction_deferred_removal(self, path: str) -> None:
            self.replay_attempts.append(path)
            if path == "b.py" and self.fail_b_once:
                self.fail_b_once = False
                self.deferred_files.add(path)
                self.deferred_removals.add(path)
                raise CompactionError("busy on b.py", operation="delete")
            self.deleted.add(path)
            self.deferred_files.discard(path)
            self.deferred_removals.discard(path)

    realtime = RealtimeState()
    server.realtime_indexing = realtime
    server.services.provider.is_connected = True
    server._compaction_service = MagicMock()
    server._compaction_service.is_compacting = False
    server._compaction_service.last_error = None
    server._compaction_service.clear_last_error = MagicMock()
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator

    with pytest.raises(CompactionError, match="busy on b.py"):
        await server._post_compaction_reindex()

    assert realtime.replay_attempts == ["a.py", "b.py"]
    assert realtime.deleted == {"a.py"}
    assert realtime.deferred_removals == {"b.py", "c.py"}
    assert "a.py" not in realtime.deferred_files

    await server.ensure_services()

    assert realtime.replay_attempts == ["a.py", "b.py", "b.py", "c.py"]
    assert realtime.deleted == {"a.py", "b.py", "c.py"}
    assert realtime.deferred_files == set()
    assert realtime.deferred_removals == set()
    status = server.get_background_compaction_status()
    assert status["pending_recovery"] is False
    assert status["last_error"] is None


async def test_successful_reindex_does_not_restore_deferred_state(
    server: ConcreteMCPServer,
) -> None:
    """Successful catch-up keeps drained deferred paths cleared."""
    realtime = RemovalReplayState()
    server.realtime_indexing = realtime
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator

    await server._post_compaction_reindex()

    assert realtime.replay_attempts == ["a.py", "b.py", "c.py"]
    assert realtime.deleted == {"a.py", "b.py", "c.py"}
    assert realtime.deferred_files == set()
    assert realtime.deferred_removals == set()
    assert coordinator.call_count == 1
    status = server.get_background_compaction_status()
    assert status["phase"] == "idle"
    assert status["pending_recovery"] is False
    assert status["last_error"] is None


async def test_successful_reindex_clears_stale_compaction_last_error(
    server: ConcreteMCPServer,
) -> None:
    """Successful catch-up clears any stale callback error on the service."""
    coordinator = CaptureCoordinator()
    server.services.indexing_coordinator = coordinator
    server.realtime_indexing = _realtime_mock()
    server._compaction_service = MagicMock()
    server._compaction_service.is_compacting = False
    server._compaction_service.last_error = RuntimeError("stale callback failure")
    server._compaction_service.clear_last_error.side_effect = lambda: setattr(
        server._compaction_service, "last_error", None
    )

    await server._post_compaction_reindex()

    assert coordinator.call_count == 1
    assert server._compaction_service.last_error is None
    status = server.get_background_compaction_status()
    assert status["phase"] == "idle"
    assert status["last_error"] is None
    assert status["pending_recovery"] is False


def test_refresh_background_compaction_status_uses_real_service_property(
    server: ConcreteMCPServer,
) -> None:
    """Status refresh must follow the live CompactionService state."""
    server._compaction_service = CompactionService(
        Path(server.config.database.path),
        server.config,
    )

    status = server.get_background_compaction_status()
    assert status["phase"] == "idle"
    assert status["in_progress"] is False

    with patch.object(
        CompactionService,
        "is_compacting",
        new_callable=PropertyMock,
        return_value=True,
    ):
        active_status = server.get_background_compaction_status()

    assert active_status["phase"] == "compacting"
    assert active_status["in_progress"] is True


async def test_trigger_background_compaction_records_startup_failure(
    server: ConcreteMCPServer,
) -> None:
    """Startup failures must be reflected in public compaction status."""

    class DummyDuckDBProvider:
        is_connected = True

    provider = DummyDuckDBProvider()
    server.services.provider = provider
    server._compaction_service = MagicMock()
    server._compaction_service.is_compacting = False
    server._compaction_service.last_error = RuntimeError("launch failed")
    server._compaction_service.compact_background = AsyncMock(
        side_effect=RuntimeError("launch failed")
    )

    with patch(
        "chunkhound.providers.database.duckdb_provider.DuckDBProvider",
        DummyDuckDBProvider,
    ):
        await server._trigger_background_compaction()

    status = server.get_background_compaction_status()
    assert status["phase"] == "failed"
    assert status["pending_recovery"] is False
    assert status["retry_attempted"] is False
    assert status["last_error"] == "launch failed"


async def test_ensure_services_retries_failed_post_compaction_reindex_once(
    server: ConcreteMCPServer,
) -> None:
    """ensure_services performs one automatic retry for failed catch-up."""
    server.services.provider.is_connected = True
    server._compaction_service = MagicMock()
    server._compaction_service.is_compacting = False
    server._compaction_service.last_error = RuntimeError("needs retry")
    server._compaction_service.clear_last_error.side_effect = lambda: setattr(
        server._compaction_service, "last_error", None
    )
    server._scan_progress["background_compaction"].update(
        {
            "phase": "failed",
            "pending_recovery": True,
            "retry_attempted": False,
            "last_error": "needs retry",
        }
    )

    async def succeed_retry(*, is_recovery_retry: bool = False) -> None:
        server._mark_background_compaction_success()

    server._post_compaction_reindex = AsyncMock(side_effect=succeed_retry)

    services = await server.ensure_services()

    assert services is server.services
    status = server._scan_progress["background_compaction"]
    assert server._compaction_service.last_error is None
    assert status["phase"] == "idle"
    assert status["pending_recovery"] is False
    assert status["retry_attempted"] is False


async def test_ensure_services_raises_after_failed_retry(
    server: ConcreteMCPServer,
) -> None:
    """ensure_services retries once, then preserves real failed recovery state."""
    server.services.provider.is_connected = True
    server.realtime_indexing = _realtime_mock()
    server.realtime_indexing.drain_compaction_deferred_file_work = AsyncMock(
        return_value=(set(), set())
    )
    server.realtime_indexing.restore_compaction_deferred_files = AsyncMock()
    server.services.indexing_coordinator = MagicMock()
    server.services.indexing_coordinator.process_directory = AsyncMock(
        side_effect=[
            RuntimeError("initial failure"),
            RuntimeError("retry failed"),
            AssertionError("post-compaction recovery retried more than once"),
        ]
    )

    with pytest.raises(RuntimeError, match="initial failure"):
        await server._post_compaction_reindex()

    initial_status = server.get_background_compaction_status()
    assert initial_status["phase"] == "failed"
    assert initial_status["pending_recovery"] is True
    assert initial_status["retry_attempted"] is False
    assert initial_status["last_error"] == "initial failure"

    with pytest.raises(CompactionError, match="retry failed") as first_exc_info:
        await server.ensure_services()

    assert first_exc_info.value.operation == "post_reindex"
    assert first_exc_info.value.reason == "retry failed"
    assert "operation=post_reindex" in str(first_exc_info.value)
    status_after_first_call = server.get_background_compaction_status()
    assert status_after_first_call["phase"] == "failed"
    assert status_after_first_call["pending_recovery"] is True
    assert status_after_first_call["retry_attempted"] is True
    assert status_after_first_call["last_error"] == "retry failed"

    with pytest.raises(CompactionError, match="retry failed") as second_exc_info:
        await server.ensure_services()

    assert second_exc_info.value.operation == "post_reindex"
    assert second_exc_info.value.reason == "retry failed"
    assert "operation=post_reindex" in str(second_exc_info.value)
    status = server.get_background_compaction_status()
    assert status["phase"] == "failed"
    assert status["pending_recovery"] is True
    assert status["retry_attempted"] is True
    assert status["last_error"] == "retry failed"


async def test_concurrent_ensure_services_retries_only_once(
    server: ConcreteMCPServer,
) -> None:
    """Concurrent callers share one successful recovery attempt."""
    server.services.provider.is_connected = True
    server._compaction_service = MagicMock()
    server._compaction_service.is_compacting = False
    server._compaction_service.last_error = RuntimeError("needs retry")
    server._compaction_service.clear_last_error.side_effect = lambda: setattr(
        server._compaction_service, "last_error", None
    )
    server._scan_progress["background_compaction"].update(
        {
            "phase": "failed",
            "pending_recovery": True,
            "retry_attempted": False,
            "last_error": "needs retry",
        }
    )
    retry_started = asyncio.Event()
    release_retry = asyncio.Event()

    async def succeed_retry(*, is_recovery_retry: bool = False) -> None:
        retry_started.set()
        await release_retry.wait()
        server._mark_background_compaction_success()

    server._post_compaction_reindex = AsyncMock(side_effect=succeed_retry)

    first_task = asyncio.create_task(server.ensure_services())
    await retry_started.wait()
    second_task = asyncio.create_task(server.ensure_services())
    await asyncio.sleep(0)
    release_retry.set()

    first_services, second_services = await asyncio.gather(first_task, second_task)

    assert first_services is server.services
    assert second_services is server.services
    status = server.get_background_compaction_status()
    assert server._compaction_service.last_error is None
    assert status["phase"] == "idle"
    assert status["pending_recovery"] is False
    assert status["retry_attempted"] is False

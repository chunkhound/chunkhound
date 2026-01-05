"""Unit tests for WatcherManager service.

Tests file watcher lifecycle and event processing.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.services.project_registry import ProjectInfo
from chunkhound.services.watcher_manager import (
    ProjectEventHandler,
    WatcherManager,
)


@pytest.fixture
def mock_coordinator():
    """Create a mock indexing coordinator."""
    coordinator = MagicMock()
    coordinator.process_file = AsyncMock()
    coordinator.delete_file = AsyncMock()
    coordinator._db = MagicMock()
    coordinator._db.delete_file_completely = MagicMock()
    return coordinator


@pytest.fixture
def manager(mock_coordinator):
    """Create a WatcherManager with mock coordinator."""
    return WatcherManager(mock_coordinator)


@pytest.fixture
def sample_project():
    """Create a sample ProjectInfo."""
    return ProjectInfo(
        base_directory=Path("/home/user/myproject"),
        project_name="myproject",
    )


@pytest.fixture
def temp_project(tmp_path):
    """Create a real temporary project directory."""
    project_dir = tmp_path / "test-project"
    project_dir.mkdir()
    (project_dir / "src").mkdir()
    (project_dir / "src" / "main.py").touch()
    return ProjectInfo(
        base_directory=project_dir,
        project_name="test-project",
    )


class TestProjectEventHandler:
    """Tests for ProjectEventHandler class."""

    def test_on_created_file(self):
        """Test handling file creation event."""
        callback = MagicMock()
        handler = ProjectEventHandler(
            project_path=Path("/home/user/project"),
            callback=callback,
            should_index=lambda p: True,
        )
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/home/user/project/src/new.py"

        handler.on_created(event)

        callback.assert_called_once()
        args = callback.call_args[0]
        # Compare resolved paths for cross-platform consistency
        # (macOS resolves /home → /System/Volumes/Data/home)
        assert args[0] == Path("/home/user/project/src/new.py").resolve()
        assert args[1] == "created"

    def test_on_created_directory_ignored(self):
        """Test that directory creation events are ignored."""
        callback = MagicMock()
        handler = ProjectEventHandler(
            project_path=Path("/home/user/project"),
            callback=callback,
            should_index=lambda p: True,
        )
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/home/user/project/src/newdir"

        handler.on_created(event)

        callback.assert_not_called()

    def test_on_created_not_indexable(self):
        """Test that non-indexable files are ignored."""
        callback = MagicMock()
        handler = ProjectEventHandler(
            project_path=Path("/home/user/project"),
            callback=callback,
            should_index=lambda p: False,
        )
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/home/user/project/node_modules/test.js"

        handler.on_created(event)

        callback.assert_not_called()

    def test_on_modified(self):
        """Test handling file modification event."""
        callback = MagicMock()
        handler = ProjectEventHandler(
            project_path=Path("/home/user/project"),
            callback=callback,
            should_index=lambda p: True,
        )
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/home/user/project/src/main.py"

        handler.on_modified(event)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[1] == "modified"

    def test_on_deleted(self):
        """Test handling file deletion event."""
        callback = MagicMock()
        handler = ProjectEventHandler(
            project_path=Path("/home/user/project"),
            callback=callback,
            should_index=lambda p: True,
        )
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/home/user/project/src/old.py"

        handler.on_deleted(event)

        # Deletions always callback (file may have been indexed)
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[1] == "deleted"

    def test_on_moved_both_indexable(self):
        """Test move where both src and dest are indexable."""
        callback = MagicMock()
        handler = ProjectEventHandler(
            project_path=Path("/home/user/project"),
            callback=callback,
            should_index=lambda p: True,
        )
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/home/user/project/src/old.py"
        event.dest_path = "/home/user/project/src/new.py"

        handler.on_moved(event)

        # Should get delete for src and create for dest
        assert callback.call_count == 2
        calls = callback.call_args_list
        assert calls[0][0][1] == "deleted"
        assert calls[1][0][1] == "created"

    def test_on_moved_only_src_indexable(self):
        """Test move where only source is indexable."""
        callback = MagicMock()
        handler = ProjectEventHandler(
            project_path=Path("/home/user/project"),
            callback=callback,
            should_index=lambda p: "node_modules" not in str(p),
        )
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/home/user/project/src/file.py"
        event.dest_path = "/home/user/project/node_modules/file.py"

        handler.on_moved(event)

        # Should only get delete
        callback.assert_called_once()
        assert callback.call_args[0][1] == "deleted"

    def test_on_moved_only_dest_indexable(self):
        """Test move where only destination is indexable (atomic write)."""
        callback = MagicMock()
        handler = ProjectEventHandler(
            project_path=Path("/home/user/project"),
            callback=callback,
            should_index=lambda p: ".tmp" not in str(p),
        )
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/home/user/project/src/file.py.tmp"
        event.dest_path = "/home/user/project/src/file.py"

        handler.on_moved(event)

        # Should only get create (atomic write pattern)
        callback.assert_called_once()
        assert callback.call_args[0][1] == "created"


class TestShouldIndexFile:
    """Tests for file indexability detection."""

    def test_should_index_python_file(self, manager):
        """Test that Python files are indexable."""
        result = manager._should_index_file(Path("/project/src/main.py"))
        assert result is True

    def test_should_index_typescript_file(self, manager):
        """Test that TypeScript files are indexable."""
        result = manager._should_index_file(Path("/project/src/app.tsx"))
        assert result is True

    def test_should_not_index_hidden_file(self, manager):
        """Test that hidden files are not indexable."""
        result = manager._should_index_file(Path("/project/.secret"))
        assert result is False

    def test_should_not_index_node_modules(self, manager):
        """Test that node_modules files are not indexable."""
        result = manager._should_index_file(
            Path("/project/node_modules/lodash/index.js")
        )
        assert result is False

    def test_should_not_index_pycache(self, manager):
        """Test that __pycache__ files are not indexable."""
        result = manager._should_index_file(
            Path("/project/__pycache__/main.cpython-311.pyc")
        )
        assert result is False

    def test_should_not_index_git(self, manager):
        """Test that .git files are not indexable."""
        result = manager._should_index_file(Path("/project/.git/objects/abc"))
        assert result is False

    def test_should_not_index_dotfile(self, manager):
        """Test that dotfiles are not indexed (even allowed ones need proper extension)."""
        # .gitignore is in allowed_dotfiles list but still needs a recognized extension
        # The allowed_dotfiles list only exempts from the hidden file check,
        # but the file still needs to pass extension checks
        result = manager._should_index_file(Path("/project/.gitignore"))
        assert result is False

    def test_should_not_index_hidden_env(self, manager):
        """Test that .env files are not indexed."""
        result = manager._should_index_file(Path("/project/.env"))
        assert result is False

    def test_should_not_index_unknown_extension(self, manager):
        """Test that unknown extensions are not indexable."""
        result = manager._should_index_file(Path("/project/data.xyz"))
        assert result is False


class TestOnFileEvent:
    """Tests for event queuing."""

    def test_on_file_changed_queues_event(self, manager):
        """Test that file events are queued."""
        file_path = Path("/project/src/main.py")

        manager._on_file_event(file_path, "modified")

        assert str(file_path) in manager._pending_events
        event_type, first_seen = manager._pending_events[str(file_path)]
        assert event_type == "modified"

    def test_on_file_created_adds_to_pending(self, manager):
        """Test that created events are added to pending."""
        file_path = Path("/project/src/new.py")

        manager._on_file_event(file_path, "created")

        assert str(file_path) in manager._pending_events
        assert manager._pending_events[str(file_path)][0] == "created"

    def test_on_file_deleted_adds_to_pending(self, manager):
        """Test that deleted events are added to pending."""
        file_path = Path("/project/src/old.py")

        manager._on_file_event(file_path, "deleted")

        assert str(file_path) in manager._pending_events
        assert manager._pending_events[str(file_path)][0] == "deleted"

    def test_debounce_keeps_first_timestamp(self, manager):
        """Test that debouncing preserves first seen timestamp."""
        file_path = Path("/project/src/main.py")
        first_time = time.time()

        manager._on_file_event(file_path, "created")
        _, first_seen = manager._pending_events[str(file_path)]

        time.sleep(0.1)
        manager._on_file_event(file_path, "modified")

        _, second_seen = manager._pending_events[str(file_path)]
        # First seen should not change
        assert second_seen == first_seen

    def test_event_type_updates(self, manager):
        """Test that event type updates on subsequent events."""
        file_path = Path("/project/src/main.py")

        manager._on_file_event(file_path, "created")
        manager._on_file_event(file_path, "modified")

        event_type, _ = manager._pending_events[str(file_path)]
        assert event_type == "modified"

    def test_max_pending_events_limit(self, manager):
        """Test that max_pending_events limit is enforced."""
        manager._loop = asyncio.new_event_loop()

        # Fill up to the limit (use instance variable)
        max_events = manager._max_pending_events
        for i in range(max_events):
            manager._pending_events[f"/project/file{i}.py"] = ("created", time.time())

        # Adding more should trigger force-processing
        with patch.object(manager, "_process_file_event", new_callable=AsyncMock):
            manager._on_file_event(Path("/project/overflow.py"), "created")

        # Should have made room
        assert len(manager._pending_events) <= max_events

        manager._loop.close()


class TestWatcherLifecycle:
    """Tests for watcher start/stop."""

    def test_start_watcher_creates_observer(self, manager, temp_project):
        """Test that starting watcher creates an observer."""
        result = manager.start_watcher(temp_project)

        assert result is True
        assert str(temp_project.base_directory) in manager._watchers
        watcher = manager._watchers[str(temp_project.base_directory)]
        assert watcher.observer.is_alive()

        # Cleanup
        manager.stop_watcher(temp_project.base_directory)

    def test_start_watcher_already_watching(self, manager, temp_project):
        """Test starting watcher when already watching."""
        manager.start_watcher(temp_project)
        result = manager.start_watcher(temp_project)

        assert result is True
        # Cleanup
        manager.stop_watcher(temp_project.base_directory)

    def test_start_watcher_updates_registry(self, manager, temp_project):
        """Test that starting watcher updates registry status."""
        mock_registry = MagicMock()
        manager.set_project_registry(mock_registry)

        manager.start_watcher(temp_project)

        mock_registry.set_watcher_status.assert_called_with(
            str(temp_project.base_directory), active=True
        )

        # Cleanup
        manager.stop_watcher(temp_project.base_directory)

    def test_stop_watcher_removes_observer(self, manager, temp_project):
        """Test that stopping watcher removes the observer."""
        manager.start_watcher(temp_project)
        assert str(temp_project.base_directory) in manager._watchers

        result = manager.stop_watcher(temp_project.base_directory)

        assert result is True
        assert str(temp_project.base_directory) not in manager._watchers

    def test_stop_watcher_not_watching(self, manager, sample_project):
        """Test stopping watcher when not watching."""
        result = manager.stop_watcher(sample_project.base_directory)

        assert result is False

    def test_stop_watcher_updates_registry(self, manager, temp_project):
        """Test that stopping watcher updates registry status."""
        mock_registry = MagicMock()
        manager.set_project_registry(mock_registry)
        manager.start_watcher(temp_project)

        manager.stop_watcher(temp_project.base_directory)

        mock_registry.set_watcher_status.assert_called_with(
            str(temp_project.base_directory), active=False
        )

    def test_stop_all_watchers(self, manager, temp_project):
        """Test stopping all watchers."""
        manager.start_watcher(temp_project)
        assert len(manager._watchers) == 1

        manager.stop_all()

        assert len(manager._watchers) == 0

    def test_stop_all_clears_pending(self, manager, temp_project):
        """Test that stop_all clears pending events."""
        manager._pending_events["/project/file.py"] = ("modified", time.time())
        manager.start_watcher(temp_project)

        manager.stop_all()

        assert len(manager._pending_events) == 0


class TestWatcherStats:
    """Tests for watcher statistics."""

    def test_get_watcher_stats(self, manager, temp_project):
        """Test getting stats for a watcher."""
        manager.start_watcher(temp_project)

        status = manager.get_watcher_status(temp_project.base_directory)

        assert status is not None
        assert status["project_path"] == str(temp_project.base_directory)
        assert status["events_processed"] == 0
        assert status["observer_alive"] is True

        # Cleanup
        manager.stop_watcher(temp_project.base_directory)

    def test_get_watcher_stats_not_found(self, manager, sample_project):
        """Test getting stats for non-existent watcher."""
        status = manager.get_watcher_status(sample_project.base_directory)

        assert status is None

    def test_get_all_status(self, manager, temp_project):
        """Test getting status for all watchers."""
        manager.start_watcher(temp_project)

        all_status = manager.get_all_status()

        assert len(all_status) == 1
        assert str(temp_project.base_directory) in all_status

        # Cleanup
        manager.stop_watcher(temp_project.base_directory)

    def test_get_pending_count(self, manager):
        """Test getting count of pending events."""
        manager._pending_events = {
            "/file1.py": ("created", time.time()),
            "/file2.py": ("modified", time.time()),
        }

        count = manager.get_pending_count()

        assert count == 2


class TestProcessFileEvent:
    """Tests for file event processing."""

    @pytest.mark.asyncio
    async def test_process_deleted_event(self, manager, mock_coordinator):
        """Test processing a deleted file event."""
        file_path = Path("/project/src/deleted.py")

        await manager._process_file_event(file_path, "deleted")

        mock_coordinator.delete_file.assert_called_once_with(file_path)

    @pytest.mark.asyncio
    async def test_process_deleted_fallback(self, manager, mock_coordinator):
        """Test processing delete with fallback to DB method."""
        del mock_coordinator.delete_file
        file_path = Path("/project/src/deleted.py")

        await manager._process_file_event(file_path, "deleted")

        mock_coordinator._db.delete_file_completely.assert_called_once_with(
            str(file_path)
        )

    @pytest.mark.asyncio
    async def test_process_created_event(self, manager, mock_coordinator, tmp_path):
        """Test processing a created file event."""
        file_path = tmp_path / "new.py"
        file_path.touch()

        await manager._process_file_event(file_path, "created")

        mock_coordinator.process_file.assert_called_once_with(
            file_path, skip_embeddings=False, base_directory=None
        )

    @pytest.mark.asyncio
    async def test_process_modified_event(self, manager, mock_coordinator, tmp_path):
        """Test processing a modified file event."""
        file_path = tmp_path / "main.py"
        file_path.touch()

        await manager._process_file_event(file_path, "modified")

        mock_coordinator.process_file.assert_called_once_with(
            file_path, skip_embeddings=False, base_directory=None
        )

    @pytest.mark.asyncio
    async def test_process_event_file_not_exists(self, manager, mock_coordinator):
        """Test that non-existent files are skipped."""
        file_path = Path("/nonexistent/file.py")

        await manager._process_file_event(file_path, "modified")

        mock_coordinator.process_file.assert_not_called()


class TestAsyncLifecycle:
    """Tests for async start/stop."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, manager):
        """Test async start and stop."""
        await manager.start()

        assert manager._running is True
        assert manager._processor_task is not None

        await manager.stop()

        assert manager._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_processor(self, manager):
        """Test that stop cancels the processor task."""
        await manager.start()
        task = manager._processor_task

        await manager.stop()

        assert task.cancelled() or task.done()


class TestFlushPending:
    """Tests for flush_pending."""

    def test_flush_pending_clears_events(self, manager):
        """Test that flush_pending clears the pending events dict."""
        manager._pending_events = {
            "/file1.py": ("created", time.time() - 10),
            "/file2.py": ("modified", time.time() - 10),
        }

        # Without a loop, flush_pending will just clear events and return 0
        # This tests the clearing behavior
        manager._loop = None
        count = manager.flush_pending()

        # Events cleared even if processing fails
        assert len(manager._pending_events) == 0
        # Count is 0 because no loop to process
        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_pending_with_loop(self, manager, mock_coordinator):
        """Test flushing pending events with async processing."""
        # Start the manager to set up loop
        await manager.start()

        manager._pending_events = {
            "/file1.py": ("deleted", time.time() - 10),
        }

        # This should process events via the loop
        # Note: The implementation catches exceptions per-event
        count = manager.flush_pending()

        # Should have attempted to process (count may vary based on timing)
        assert len(manager._pending_events) == 0

        await manager.stop()


class TestDebugSink:
    """Tests for debug logging."""

    def test_debug_sink_called(self, mock_coordinator):
        """Test that debug sink is called."""
        debug_messages = []
        manager = WatcherManager(
            mock_coordinator,
            debug_sink=lambda msg: debug_messages.append(msg),
        )

        manager._debug("Test message")

        assert len(debug_messages) == 1
        assert "Test message" in debug_messages[0]

    def test_debug_no_sink(self, manager):
        """Test debug with no sink doesn't raise."""
        manager._debug_sink = None
        manager._debug("Test message")  # Should not raise


class TestFindProjectForPath:
    """Tests for finding project by path."""

    def test_find_project_for_path_with_registry(self, manager, sample_project):
        """Test finding project when registry is set."""
        mock_registry = MagicMock()
        mock_registry.find_project_for_path.return_value = sample_project
        manager.set_project_registry(mock_registry)

        result = manager._find_project_for_path(
            Path("/home/user/myproject/src/main.py")
        )

        assert result == sample_project
        mock_registry.find_project_for_path.assert_called_once()

    def test_find_project_for_path_no_registry(self, manager):
        """Test finding project when no registry is set."""
        result = manager._find_project_for_path(Path("/project/src/main.py"))

        assert result is None

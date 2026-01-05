"""Unit tests for UpdateQueue service.

Tests update queue for concurrent MCP session support, including:
- Atomic enqueue/dequeue operations with file locking
- Deduplication of duplicate paths
- Batch dequeue operations
- Queue persistence across process boundaries
- Thread and process safety
"""

import json
import tempfile
from pathlib import Path

import pytest

from chunkhound.services.update_queue import UpdateQueue

# Module-level worker functions for multiprocessing tests
# (multiprocessing requires pickle-able functions)


def _enqueue_files_worker(base_dir, process_id, num_files):
    """Worker function for multi-process enqueue test."""
    from pathlib import Path

    from chunkhound.services.update_queue import UpdateQueue

    queue = UpdateQueue(base_dir=Path(base_dir))
    for i in range(num_files):
        file_path = Path(base_dir) / f"process{process_id}_file{i}.py"
        queue.enqueue(file_path)


def _dequeue_batch_worker(base_dir, result_queue):
    """Worker function for multi-process dequeue test."""
    from pathlib import Path

    from chunkhound.services.update_queue import UpdateQueue

    queue = UpdateQueue(base_dir=Path(base_dir))
    dequeued = []

    # Each process tries to dequeue multiple batches
    for _ in range(5):
        batch = queue.dequeue_batch(limit=5)
        dequeued.extend([str(p) for p in batch])

    result_queue.put(dequeued)


class TestUpdateQueue:
    """Test UpdateQueue class for multi-session file reindexing."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def queue(self, temp_project_dir):
        """Create UpdateQueue instance in temp directory."""
        return UpdateQueue(base_dir=temp_project_dir)

    def test_queue_initialization_creates_directory(self, temp_project_dir):
        """Test that queue initialization creates .chunkhound directory."""
        queue = UpdateQueue(base_dir=temp_project_dir)

        assert (temp_project_dir / ".chunkhound").exists()
        assert (temp_project_dir / ".chunkhound" / "update_queue.json").exists()

    def test_enqueue_adds_file_to_queue(self, queue, temp_project_dir):
        """Test enqueuing a file path."""
        file_path = temp_project_dir / "test.py"

        queue.enqueue(file_path)

        assert queue.size() == 1

    def test_enqueue_deduplicates_same_path(self, queue, temp_project_dir):
        """Test that enqueueing same path twice doesn't create duplicates."""
        file_path = temp_project_dir / "test.py"

        queue.enqueue(file_path)
        queue.enqueue(file_path)

        assert queue.size() == 1

    def test_enqueue_multiple_different_paths(self, queue, temp_project_dir):
        """Test enqueueing multiple different paths."""
        file1 = temp_project_dir / "test1.py"
        file2 = temp_project_dir / "test2.py"
        file3 = temp_project_dir / "test3.py"

        queue.enqueue(file1)
        queue.enqueue(file2)
        queue.enqueue(file3)

        assert queue.size() == 3

    def test_dequeue_batch_returns_correct_count(self, queue, temp_project_dir):
        """Test dequeue_batch returns requested number of items."""
        for i in range(10):
            queue.enqueue(temp_project_dir / f"test{i}.py")

        batch = queue.dequeue_batch(limit=5)

        assert len(batch) == 5
        assert queue.size() == 5  # 5 remaining

    def test_dequeue_batch_returns_empty_when_queue_empty(self, queue):
        """Test dequeue_batch returns empty list when queue is empty."""
        batch = queue.dequeue_batch(limit=10)

        assert batch == []
        assert queue.size() == 0

    def test_dequeue_batch_returns_all_when_less_than_limit(
        self, queue, temp_project_dir
    ):
        """Test dequeue_batch returns all items when count < limit."""
        queue.enqueue(temp_project_dir / "test1.py")
        queue.enqueue(temp_project_dir / "test2.py")

        batch = queue.dequeue_batch(limit=10)

        assert len(batch) == 2
        assert queue.size() == 0

    def test_dequeue_batch_fifo_order(self, queue, temp_project_dir):
        """Test dequeue_batch returns items in FIFO order."""
        file1 = temp_project_dir / "test1.py"
        file2 = temp_project_dir / "test2.py"
        file3 = temp_project_dir / "test3.py"

        queue.enqueue(file1)
        queue.enqueue(file2)
        queue.enqueue(file3)

        batch = queue.dequeue_batch(limit=2)

        # Compare resolved paths (enqueue resolves symlinks for consistency)
        # On macOS, /var → /private/var
        assert batch[0] == file1.resolve()
        assert batch[1] == file2.resolve()

    def test_peek_does_not_remove_items(self, queue, temp_project_dir):
        """Test peek() views items without removing them."""
        file1 = temp_project_dir / "test1.py"
        file2 = temp_project_dir / "test2.py"

        queue.enqueue(file1)
        queue.enqueue(file2)

        peeked = queue.peek(limit=2)

        assert len(peeked) == 2
        assert queue.size() == 2  # Still 2 items

    def test_clear_removes_all_items(self, queue, temp_project_dir):
        """Test clear() removes all items from queue."""
        for i in range(5):
            queue.enqueue(temp_project_dir / f"test{i}.py")

        count = queue.clear()

        assert count == 5
        assert queue.size() == 0

    def test_queue_persists_across_instances(self, temp_project_dir):
        """Test queue data persists across UpdateQueue instances."""
        file1 = temp_project_dir / "test1.py"
        file2 = temp_project_dir / "test2.py"

        # First instance
        queue1 = UpdateQueue(base_dir=temp_project_dir)
        queue1.enqueue(file1)
        queue1.enqueue(file2)

        # Second instance (should load persisted data)
        queue2 = UpdateQueue(base_dir=temp_project_dir)

        assert queue2.size() == 2

    def test_concurrent_enqueues_are_atomic_multiprocess(self, temp_project_dir):
        """Test that concurrent enqueues from multiple PROCESSES are atomic.

        This tests the PRIMARY use case: multiple MCP server processes accessing
        the same update queue via InterProcessLock.
        """
        import multiprocessing

        num_processes = 5
        files_per_process = 10

        processes = []
        for i in range(num_processes):
            p = multiprocessing.Process(
                target=_enqueue_files_worker,
                args=(str(temp_project_dir), i, files_per_process),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=10)
            assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"

        # Verify all files were enqueued (no lost updates)
        queue = UpdateQueue(base_dir=temp_project_dir)
        assert queue.size() == num_processes * files_per_process

    def test_concurrent_dequeues_are_atomic_multiprocess(self, temp_project_dir):
        """Test that concurrent dequeues from multiple PROCESSES are atomic.

        This tests the PRIMARY use case: multiple processes dequeuing without
        losing items or getting duplicates.
        """
        import multiprocessing

        # Pre-populate queue
        queue = UpdateQueue(base_dir=temp_project_dir)
        total_files = 100
        for i in range(total_files):
            queue.enqueue(temp_project_dir / f"test{i}.py")

        num_processes = 5
        result_queue = multiprocessing.Queue()
        processes = []

        for i in range(num_processes):
            p = multiprocessing.Process(
                target=_dequeue_batch_worker, args=(str(temp_project_dir), result_queue)
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join(timeout=10)
            assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"

        # Collect all dequeued files
        all_dequeued = []
        while not result_queue.empty():
            batch = result_queue.get()
            all_dequeued.extend(batch)

        # Verify:
        # 1. All files were dequeued (no lost items)
        assert len(all_dequeued) == total_files
        # 2. No duplicates (atomic dequeue)
        assert len(set(all_dequeued)) == total_files
        # 3. Queue is empty
        queue_after = UpdateQueue(base_dir=temp_project_dir)
        assert queue_after.size() == 0

    def test_enqueue_resolves_path_to_absolute(self, queue, temp_project_dir):
        """Test that enqueue() resolves paths to absolute."""
        # Create relative path
        relative_path = Path("test.py")

        # Enqueue (should resolve to absolute)
        queue.enqueue(temp_project_dir / relative_path)

        # Peek to check stored path
        peeked = queue.peek(limit=1)

        # Should be stored as absolute path
        assert peeked[0].is_absolute()

    def test_queue_handles_nonexistent_base_dir(self):
        """Test queue creation with nonexistent base directory."""
        nonexistent = Path("/tmp/nonexistent_chunkhound_test_dir_12345")

        # Should create directory
        queue = UpdateQueue(base_dir=nonexistent)

        assert nonexistent.exists()
        assert (nonexistent / ".chunkhound").exists()

        # Cleanup
        import shutil

        shutil.rmtree(nonexistent)

    def test_queue_file_is_valid_json(self, queue, temp_project_dir):
        """Test that queue file is valid JSON."""
        queue.enqueue(temp_project_dir / "test.py")

        queue_file = temp_project_dir / ".chunkhound" / "update_queue.json"
        content = json.loads(queue_file.read_text())

        assert isinstance(content, list)
        assert len(content) == 1

    def test_size_is_accurate(self, queue, temp_project_dir):
        """Test that size() returns accurate count."""
        assert queue.size() == 0

        queue.enqueue(temp_project_dir / "test1.py")
        assert queue.size() == 1

        queue.enqueue(temp_project_dir / "test2.py")
        assert queue.size() == 2

        queue.dequeue_batch(limit=1)
        assert queue.size() == 1

        queue.clear()
        assert queue.size() == 0

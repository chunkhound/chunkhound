"""Update queue for managing file reindexing across concurrent MCP sessions.

PROBLEM: DuckDB file locking prevents concurrent write access
SOLUTION: Primary session has exclusive write access, secondary sessions queue updates

ARCHITECTURE:
- UpdateQueue: Thread-safe queue for file paths awaiting reindexing
- Inter-process locking: fasteners.InterProcessLock for atomic file operations
- Queue persistence: JSON file at .chunkhound/update_queue.json
- Primary session: Dequeues and processes updates
- Secondary sessions: Enqueue file changes detected by file watcher

CRITICAL CONSTRAINTS:
- Atomic operations: All queue modifications must be atomic via file lock
- No data loss: Failed dequeues must not remove items from queue
- Thread safety: Multiple threads in same process may access queue
- Process safety: Multiple processes (MCP servers) may access queue
- Atomic writes: Use write-then-rename to prevent corruption on crash
"""

import json
import os
import tempfile
import threading
import weakref
from pathlib import Path

import fasteners
from loguru import logger


class UpdateQueue:
    """Thread-safe and process-safe queue for file update operations.

    Uses inter-process file locking to ensure atomic queue operations across
    multiple concurrent MCP server processes.

    PATTERN: Double-locking for safety
    - Inter-process lock: fasteners.InterProcessLock (across processes)
    - File-based persistence: JSON file with atomic read-modify-write

    THREAD SAFETY LIMITATION:
    - InterProcessLock is process-safe but NOT thread-safe
    - Multiple UpdateQueue instances in the same process may have race conditions
    - Use separate processes for concurrent MCP servers, not threads

    USAGE:
        queue = UpdateQueue(base_dir=Path("/path/to/project"))
        queue.enqueue(Path("/path/to/file.py"))
        batch = queue.dequeue_batch(limit=10)
    """

    # Class variable to track active instances (for warning about thread safety)
    _active_instances: weakref.WeakSet["UpdateQueue"] = weakref.WeakSet()
    _instance_lock = threading.Lock()  # Protect instance tracking
    _warned_about_multiple_instances = False  # Only warn once per process

    def __init__(self, base_dir: Path, max_queue_size: int = 10000):
        """Initialize update queue.

        Args:
            base_dir: Base directory for the project (contains .chunkhound/)
            max_queue_size: Maximum number of files allowed in queue (default: 10000)
        """
        # Track active instances and warn about thread safety (thread-safe)
        with UpdateQueue._instance_lock:
            UpdateQueue._active_instances.add(self)
            if (
                len(UpdateQueue._active_instances) > 1
                and not UpdateQueue._warned_about_multiple_instances
            ):
                UpdateQueue._warned_about_multiple_instances = True
                logger.warning(
                    f"Multiple UpdateQueue instances active in same process ({len(UpdateQueue._active_instances)}). "
                    "InterProcessLock is NOT thread-safe. Use separate processes for concurrent MCP servers."
                )

        self.base_dir = base_dir.resolve()
        self.queue_dir = self.base_dir / ".chunkhound"
        self.queue_file = self.queue_dir / "update_queue.json"
        self.lock_file = self.queue_dir / "update_queue.lock"
        self.max_queue_size = max_queue_size

        # Ensure directory exists
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty queue file if doesn't exist
        if not self.queue_file.exists():
            self._write_queue([])

        # Inter-process lock for atomic operations across PROCESSES
        self._lock = fasteners.InterProcessLock(str(self.lock_file))

        # Thread lock for atomic operations across THREADS (same process)
        # Necessary because InterProcessLock is not thread-safe
        self._thread_lock = threading.Lock()

    def _read_queue(self) -> list[str]:
        """Read queue from file (must hold lock).

        Returns:
            List of file paths as strings
        """
        try:
            content = self.queue_file.read_text()
            data = json.loads(content)
            if isinstance(data, list):
                return data
            logger.warning(f"Queue file contains non-list data: {type(data)}")
            return []
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to read queue file: {e}")
            return []

    def _write_queue(self, queue: list[str]) -> None:
        """Write queue to file with atomic write-then-rename (must hold lock).

        Uses tempfile + os.replace() for atomicity. Prevents corruption if process
        crashes mid-write.

        Args:
            queue: List of file paths to write

        Raises:
            Exception: If write fails (queue state unchanged)
        """
        try:
            content = json.dumps(queue, indent=2)

            # Atomic write pattern: temp file + rename
            fd, temp_path = tempfile.mkstemp(
                dir=self.queue_dir,
                prefix=".update_queue.tmp.",
                suffix=".json",
                text=True,
            )
            try:
                # Write to temp file
                os.write(fd, content.encode("utf-8"))
                os.fsync(fd)  # Ensure written to disk
                os.close(fd)

                # Atomic rename (POSIX guarantees atomicity)
                os.replace(temp_path, self.queue_file)

            except Exception as write_err:
                # Clean up temp file on error
                os.close(fd)
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass  # Best effort cleanup
                raise write_err

        except Exception as e:
            logger.error(f"Failed to write queue file atomically: {e}")
            raise

    def enqueue(self, file_path: Path) -> bool:
        """Add a file to the update queue.

        Thread-safe and process-safe. Duplicate paths are automatically deduplicated.

        Args:
            file_path: Absolute path to file needing reindexing

        Returns:
            True if file was enqueued, False if queue is full or file already in queue
        """
        file_str = str(file_path.resolve())

        # Acquire thread lock FIRST, then process lock
        with self._thread_lock:
            with self._lock:
                queue = self._read_queue()

                # Check if queue is full
                if len(queue) >= self.max_queue_size:
                    logger.warning(
                        f"Queue full ({len(queue)} items, max {self.max_queue_size}), dropping: {file_str}"
                    )
                    return False

                # Deduplicate: don't add if already in queue
                if file_str not in queue:
                    queue.append(file_str)
                    self._write_queue(queue)
                    logger.debug(f"Enqueued file for update: {file_str}")
                    return True
                else:
                    logger.debug(f"File already in queue: {file_str}")
                    return False

    def dequeue_batch(self, limit: int = 10) -> list[Path]:
        """Remove and return up to `limit` files from the queue.

        Thread-safe and process-safe. Returns empty list if queue is empty.

        Args:
            limit: Maximum number of files to dequeue

        Returns:
            List of Path objects to process (may be empty)
        """
        # Acquire thread lock FIRST, then process lock
        with self._thread_lock:
            with self._lock:
                queue = self._read_queue()

                if not queue:
                    return []

                # Take up to `limit` items from front of queue
                batch = queue[:limit]
                remaining = queue[limit:]

                # Write back remaining items
                self._write_queue(remaining)

                # Convert to Path objects
                result = [Path(p) for p in batch]
                logger.debug(
                    f"Dequeued {len(result)} files (queue had {len(queue)}, now {len(remaining)})"
                )
                return result

    def size(self) -> int:
        """Get current queue size.

        Thread-safe and process-safe.

        Returns:
            Number of files in queue
        """
        # Acquire thread lock FIRST, then process lock
        with self._thread_lock:
            with self._lock:
                queue = self._read_queue()
                return len(queue)

    def clear(self) -> int:
        """Clear all items from the queue.

        Thread-safe and process-safe.

        Returns:
            Number of items that were cleared
        """
        # Acquire thread lock FIRST, then process lock
        with self._thread_lock:
            with self._lock:
                queue = self._read_queue()
                count = len(queue)
                self._write_queue([])
                logger.info(f"Cleared {count} items from update queue")
                return count

    def peek(self, limit: int = 10) -> list[Path]:
        """View up to `limit` files from queue without removing them.

        Thread-safe and process-safe.

        Args:
            limit: Maximum number of files to peek

        Returns:
            List of Path objects (may be empty)
        """
        # Acquire thread lock FIRST, then process lock
        with self._thread_lock:
            with self._lock:
                queue = self._read_queue()
                batch = queue[:limit]
                return [Path(p) for p in batch]

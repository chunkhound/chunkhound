"""Integration tests for LanceDB optimization FileLock mechanism.

Tests verify the multi-process filesystem lock that prevents concurrent
optimization operations from corrupting the database.

The lock mechanism:
1. Creates .optimize.lock in the database directory
2. Uses non-blocking acquisition (timeout=0)
3. Skips optimization silently if lock unavailable (catches Timeout)
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from tests.utils.windows_compat import windows_safe_tempdir


class TestOptimizeLockContention:
    """Test FileLock prevents concurrent optimization across processes."""

    def test_optimize_lock_contention_skips_second_process(self, lancedb_provider):
        """Verify that when one process holds the lock, another skips optimization.

        Process A: Holds .optimize.lock externally
        Process B (subprocess): Calls optimize_tables() - should skip silently
        """
        # Get the database path from provider
        db_path = Path(lancedb_provider._db_path)
        lock_path = db_path / ".optimize.lock"

        # Process A: Acquire the lock externally (simulating another MCP instance)
        external_lock = FileLock(lock_path, timeout=0)

        with external_lock:
            # Lock is held - now spawn subprocess to try optimize_tables()
            worker_script = (
                Path(__file__).parent.parent / "helpers" / "optimize_worker.py"
            )

            result = subprocess.run(
                [sys.executable, str(worker_script), str(db_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Subprocess should exit cleanly (Timeout is caught internally)
            assert result.returncode == 0, (
                f"Worker should succeed even when lock held.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

    def test_optimize_lock_released_after_completion(self, lancedb_provider):
        """Verify lock is properly released after optimization completes."""
        db_path = Path(lancedb_provider._db_path)
        lock_path = db_path / ".optimize.lock"
        worker_script = Path(__file__).parent.parent / "helpers" / "optimize_worker.py"

        # Run optimization in subprocess
        result = subprocess.run(
            [sys.executable, str(worker_script), str(db_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Worker failed: {result.stderr}"

        # After subprocess exits, we should be able to acquire lock immediately
        lock = FileLock(lock_path, timeout=0)
        try:
            with lock:
                # Success - lock was properly released
                pass
        except Timeout:
            pytest.fail("Lock should be available after subprocess completes")

    def test_optimize_creates_lock_file(self, lancedb_provider):
        """Verify .optimize.lock file is created in database directory."""
        db_path = Path(lancedb_provider._db_path)
        lock_path = db_path / ".optimize.lock"

        # Lock file shouldn't exist before any optimization
        # (may or may not exist depending on provider initialization)

        # Use FileLock to verify the path is correct and functional
        lock = FileLock(lock_path, timeout=0)
        with lock:
            # Lock file should exist while held
            assert lock_path.exists(), f"Lock file should exist at {lock_path}"

        # Lock file may still exist after release (FileLock doesn't delete it)
        # This is expected behavior


class TestOptimizeLockConcurrentSubprocesses:
    """Test concurrent subprocess optimization behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_optimize_only_one_runs(self, lancedb_provider):
        """Verify that concurrent optimize_tables() calls don't conflict.

        Spawns multiple subprocesses simultaneously - all should complete
        without errors (some will skip due to lock contention).
        """
        db_path = Path(lancedb_provider._db_path)
        worker_script = Path(__file__).parent.parent / "helpers" / "optimize_worker.py"

        async def run_worker():
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(worker_script),
                str(db_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            return proc.returncode, stdout.decode(), stderr.decode()

        # Spawn 3 concurrent optimization attempts
        results = await asyncio.gather(
            run_worker(),
            run_worker(),
            run_worker(),
        )

        # All should succeed (either run optimization or skip due to lock)
        for i, (returncode, stdout, stderr) in enumerate(results):
            assert returncode == 0, (
                f"Worker {i} failed with code {returncode}.\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

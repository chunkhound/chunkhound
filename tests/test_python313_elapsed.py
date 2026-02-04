"""Test for Python 3.13 Rich Progress task.elapsed compatibility.

This test reproduces issue where accessing task.elapsed on a Rich Progress task
fails with "'_Task' object has no attribute 'elapsed'" on Python 3.13.

The actual bug was in ChunkHound's _NoRichProgressManager shim class which
provides a minimal Progress-like API for non-TTY environments. The shim's
_Task class was missing the `elapsed` property that the embedding service
expects when calculating speed metrics.
"""

import asyncio
import threading
import time

import pytest
from rich.progress import Progress, TaskID


class TestNoRichProgressManagerShim:
    """Test the _NoRichProgressManager shim that's used in non-TTY environments."""

    def test_shim_task_has_elapsed_attribute(self):
        """Test that the shim _Task class has elapsed attribute."""
        from chunkhound.api.cli.utils.rich_output import _NoRichProgressManager

        manager = _NoRichProgressManager()
        progress = manager.get_progress_instance()

        task_id = progress.add_task("test", total=100)
        task_obj = progress.tasks[task_id]

        # This is the fix - elapsed should exist
        assert hasattr(task_obj, "elapsed"), (
            f"Shim _Task missing 'elapsed' attribute. "
            f"Type: {type(task_obj)}, attrs: {dir(task_obj)}"
        )

    def test_shim_elapsed_returns_float(self):
        """Test that elapsed returns a valid float (time in seconds)."""
        from chunkhound.api.cli.utils.rich_output import _NoRichProgressManager

        manager = _NoRichProgressManager()
        progress = manager.get_progress_instance()

        task_id = progress.add_task("test", total=100)
        task_obj = progress.tasks[task_id]

        elapsed = task_obj.elapsed
        assert isinstance(elapsed, float), f"elapsed should be float, got {type(elapsed)}"
        assert elapsed >= 0, f"elapsed should be non-negative, got {elapsed}"

    def test_shim_elapsed_increases_over_time(self):
        """Test that elapsed actually increases over time."""
        from chunkhound.api.cli.utils.rich_output import _NoRichProgressManager

        manager = _NoRichProgressManager()
        progress = manager.get_progress_instance()

        task_id = progress.add_task("test", total=100)
        task_obj = progress.tasks[task_id]

        elapsed1 = task_obj.elapsed
        time.sleep(0.1)
        elapsed2 = task_obj.elapsed

        assert elapsed2 > elapsed1, f"elapsed should increase: {elapsed1} -> {elapsed2}"

    def test_shim_embedding_service_pattern(self):
        """Test the exact pattern used in embedding_service.py lines 606-611."""
        from chunkhound.api.cli.utils.rich_output import _NoRichProgressManager

        manager = _NoRichProgressManager()
        progress = manager.get_progress_instance()

        embed_task = progress.add_task("Generating embeddings", total=100)
        processed_count = 0

        # Simulate batch processing
        for _ in range(5):
            processed_count += 10
            progress.advance(embed_task, 10)

            # This is the exact pattern from embedding_service.py:606-611
            task_obj = progress.tasks[embed_task]
            if task_obj.elapsed and task_obj.elapsed > 0:
                speed = processed_count / task_obj.elapsed
                progress.update(embed_task, speed=f"{speed:.1f} chunks/s")

        assert task_obj.completed == 50
        assert task_obj.elapsed > 0


class TestRichProgressElapsed:
    """Test Rich Progress task.elapsed access patterns used in embedding_service.py."""

    def test_elapsed_attribute_exists(self):
        """Test that Rich Task has elapsed attribute."""
        with Progress() as progress:
            task_id = progress.add_task("test", total=100)
            task_obj = progress.tasks[task_id]

            # This should work - Rich Task has elapsed property
            assert hasattr(task_obj, "elapsed"), (
                f"Rich Task missing 'elapsed' attribute. "
                f"Type: {type(task_obj)}, attrs: {dir(task_obj)}"
            )

    def test_elapsed_in_sync_context(self):
        """Test accessing elapsed in synchronous context (like embedding_service)."""
        with Progress() as progress:
            task_id = progress.add_task("test", total=100)
            progress.advance(task_id, 10)

            task_obj = progress.tasks[task_id]
            elapsed = task_obj.elapsed

            # elapsed should be a float (time in seconds)
            assert elapsed is not None or elapsed == 0.0
            assert isinstance(elapsed, (int, float)) or elapsed is None

    def test_elapsed_with_threading_lock(self):
        """Test accessing elapsed with threading lock (matches embedding_service pattern)."""
        update_lock = threading.Lock()
        processed_count = 0

        with Progress() as progress:
            task_id = progress.add_task("test", total=100)

            def update_progress():
                nonlocal processed_count
                with update_lock:
                    processed_count += 10
                    progress.advance(task_id, 10)

                    task_obj = progress.tasks[task_id]
                    # This is the exact pattern from embedding_service.py:606-608
                    if task_obj.elapsed and task_obj.elapsed > 0:
                        speed = processed_count / task_obj.elapsed
                        progress.update(task_id, speed=f"{speed:.1f} items/s")

            # Run multiple updates
            for _ in range(5):
                update_progress()

    @pytest.mark.asyncio
    async def test_elapsed_in_async_context(self):
        """Test accessing elapsed from async context (the actual failure mode)."""
        update_lock = threading.Lock()
        processed_count = 0

        progress = Progress()
        progress.start()

        try:
            task_id = progress.add_task("test", total=100)

            async def async_update():
                nonlocal processed_count
                await asyncio.sleep(0.01)  # Simulate async work

                with update_lock:
                    processed_count += 10
                    progress.advance(task_id, 10)

                    # This is the exact pattern that fails on Python 3.13
                    task_obj = progress.tasks[task_id]
                    if task_obj.elapsed and task_obj.elapsed > 0:
                        speed = processed_count / task_obj.elapsed
                        progress.update(task_id, speed=f"{speed:.1f} items/s")

                return 10

            # Run concurrent async tasks (like asyncio.gather in embedding_service)
            tasks = [async_update() for _ in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(f"Task {i} failed: {result}")

        finally:
            progress.stop()

    @pytest.mark.asyncio
    async def test_elapsed_concurrent_gather(self):
        """Test elapsed access during asyncio.gather (exact embedding_service pattern)."""
        progress = Progress()
        progress.start()

        try:
            embed_task = progress.add_task("Generating embeddings", total=50)
            update_lock = threading.Lock()
            processed_count = 0

            async def process_batch_with_optional_progress(batch_num: int) -> int:
                nonlocal processed_count

                # Simulate embedding work
                await asyncio.sleep(0.01)

                # Thread-safe progress update (from embedding_service.py:598-611)
                with update_lock:
                    processed_count += 10
                    progress.advance(embed_task, 10)

                    # Calculate and display speed - THIS IS WHERE IT FAILS
                    task_obj = progress.tasks[embed_task]
                    if task_obj.elapsed and task_obj.elapsed > 0:
                        speed = processed_count / task_obj.elapsed
                        progress.update(embed_task, speed=f"{speed:.1f} chunks/s")

                return 10

            # Create tasks exactly like embedding_service.py:616-620
            tasks = [
                process_batch_with_optional_progress(i)
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify no exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(
                        f"Batch {i} failed with {type(result).__name__}: {result}"
                    )

        finally:
            progress.stop()

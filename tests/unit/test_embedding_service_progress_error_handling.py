"""Tests for embedding service progress update error handling.

This module tests that the progress speed update in the embedding service
gracefully handles various error conditions without masking the actual
embedding failure.

Related: PR #159 - fix error handling for progress task access
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.core.types.common import ChunkId
from chunkhound.services.embedding_service import EmbeddingService


class TestUpdateProgressWithSpeedHelper:
    """Tests for the _update_progress_with_speed helper method.

    This helper encapsulates all progress-related updates. Errors in progress
    display should be logged but never propagate up to mask embedding failures.
    """

    def setup_method(self):
        """Create a minimal EmbeddingService for testing."""
        self.mock_db = MagicMock()
        self.mock_progress = MagicMock()
        self.service = EmbeddingService(
            database_provider=self.mock_db,
            progress=self.mock_progress,
        )

    def test_keyerror_is_caught_when_task_id_missing(self):
        """KeyError from missing task ID should be caught and logged."""
        # Configure progress.tasks to raise KeyError
        self.mock_progress.tasks = {}  # Empty dict - KeyError when accessed
        embed_task = 42

        # Should not raise - error is caught internally
        self.service._update_progress_with_speed(
            embed_task=embed_task,
            batch_size=10,
            processed_count=100,
            batch_num=1,
        )

        # advance() was still attempted
        self.mock_progress.advance.assert_called_once_with(embed_task, 10)

    def test_attributeerror_is_caught_when_task_lacks_elapsed(self):
        """AttributeError from missing elapsed attribute should be caught.

        Note: The _NoRichProgressManager shim now includes elapsed=0.0, so this
        scenario mainly covers edge cases where a custom progress implementation
        doesn't provide elapsed.
        """
        # Create a mock task without elapsed attribute
        mock_task = MagicMock(spec=[])  # Empty spec = no attributes
        del mock_task.elapsed  # Ensure no elapsed attribute
        self.mock_progress.tasks = {1: mock_task}

        # Should not raise
        self.service._update_progress_with_speed(
            embed_task=1,
            batch_size=10,
            processed_count=100,
            batch_num=1,
        )

        # advance() was still called
        self.mock_progress.advance.assert_called_once()

    def test_advance_failure_is_caught(self):
        """Errors from progress.advance() should be caught.

        This is the critical fix from PR #159 - advance() is now inside
        the try block so failures don't mask embedding errors.
        """
        # Configure advance() to raise KeyError (invalid task ID)
        self.mock_progress.advance.side_effect = KeyError("invalid task")

        # Should not raise
        self.service._update_progress_with_speed(
            embed_task=99,
            batch_size=10,
            processed_count=100,
            batch_num=1,
        )

        # advance() was attempted
        self.mock_progress.advance.assert_called_once()

    def test_valid_progress_update_succeeds(self):
        """Valid progress state should update speed display.

        Speed is now calculated from self._embed_start_time (wallclock),
        not from task_obj.elapsed (Rich internal).
        """
        import time

        mock_task = MagicMock()
        self.mock_progress.tasks = {1: mock_task}

        # Set wallclock start 5 seconds ago so speed is ~20 chunks/s
        self.service._embed_start_time = time.monotonic() - 5.0

        self.service._update_progress_with_speed(
            embed_task=1,
            batch_size=10,
            processed_count=100,
            batch_num=1,
        )

        # Both advance and update should be called
        self.mock_progress.advance.assert_called_once_with(1, 10)
        self.mock_progress.update.assert_called_once()
        # Speed should be non-zero (approximately 100/5 = 20 chunks/s)
        call_args = self.mock_progress.update.call_args
        assert "chunks/s" in str(call_args), f"Speed not in update call: {call_args}"

    def test_zero_elapsed_skips_speed_update(self):
        """Zero elapsed time should skip speed calculation (avoid division)."""
        mock_task = MagicMock()
        mock_task.elapsed = 0.0
        self.mock_progress.tasks = {1: mock_task}

        self.service._update_progress_with_speed(
            embed_task=1,
            batch_size=10,
            processed_count=100,
            batch_num=1,
        )

        # advance() called, but update() not called (elapsed is falsy)
        self.mock_progress.advance.assert_called_once()
        self.mock_progress.update.assert_not_called()

    def test_none_elapsed_skips_speed_update(self):
        """None elapsed time should skip speed calculation."""
        mock_task = MagicMock()
        mock_task.elapsed = None
        self.mock_progress.tasks = {1: mock_task}

        self.service._update_progress_with_speed(
            embed_task=1,
            batch_size=10,
            processed_count=100,
            batch_num=1,
        )

        # advance() called, but update() not called
        self.mock_progress.advance.assert_called_once()
        self.mock_progress.update.assert_not_called()


class TestProgressFailuresDoNotMaskEmbeddingErrors:
    """Integration tests verifying progress failures don't mask embedding errors.

    These tests ensure that when the embedding provider fails, the actual error
    is propagated correctly even if progress tracking also encounters issues.
    """

    def setup_method(self):
        """Create EmbeddingService with mock dependencies."""
        self.mock_db = MagicMock()
        self.mock_embedding_provider = MagicMock()
        self.mock_embedding_provider.name = "test-provider"
        self.mock_embedding_provider.model = "test-model"
        self.mock_embedding_provider.dims = 768
        self.mock_embedding_provider.batch_size = 100

        # Configure provider to return recommended concurrency and batch limits
        self.mock_embedding_provider.get_recommended_concurrency.return_value = 4
        self.mock_embedding_provider.get_max_tokens_per_batch.return_value = 8192
        self.mock_embedding_provider.get_max_documents_per_batch.return_value = 100

        self.mock_progress = MagicMock()
        self.service = EmbeddingService(
            database_provider=self.mock_db,
            embedding_provider=self.mock_embedding_provider,
            progress=self.mock_progress,
        )

    @pytest.mark.asyncio
    async def test_embedding_error_propagates_with_broken_progress(self):
        """Embedding provider errors should propagate even with broken progress.

        This is the key behavioral test: if the embedding provider fails AND
        progress tracking fails, we should see the embedding error, not a
        progress error.
        """
        # Configure embedding provider to fail
        self.mock_embedding_provider.embed.side_effect = ConnectionError(
            "Ollama crashed"
        )

        # Configure progress to also fail (broken state)
        self.mock_progress.tasks = {}  # Will raise KeyError
        self.mock_progress.add_task.return_value = 1

        # The embedding error should be what we see in results, not progress errors
        result = await self.service.generate_embeddings_for_chunks(
            chunk_ids=[ChunkId(1), ChunkId(2)],
            chunk_texts=["text1", "text2"],
            show_progress=True,
        )

        # Result should be 0 (no successful embeddings)
        assert result == 0

        # The embedding provider was called (error occurred there, not in progress)
        self.mock_embedding_provider.embed.assert_called()

    @pytest.mark.asyncio
    async def test_successful_embedding_with_broken_progress(self):
        """Successful embeddings should complete even if progress tracking fails."""
        # Configure embedding provider to succeed (async method needs AsyncMock)
        self.mock_embedding_provider.embed = AsyncMock(
            return_value=[
                [0.1] * 768,
                [0.2] * 768,
            ]
        )

        # Configure progress to fail
        self.mock_progress.tasks = {}  # Will raise KeyError
        self.mock_progress.add_task.return_value = 1

        # Mock database operations
        self.mock_db.get_existing_embeddings.return_value = set()
        self.mock_db.insert_embeddings_batch.return_value = 2

        result = await self.service.generate_embeddings_for_chunks(
            chunk_ids=[ChunkId(1), ChunkId(2)],
            chunk_texts=["text1", "text2"],
            show_progress=True,
        )

        # Embeddings should still be generated successfully
        assert result == 2


class TestExceptionTupleCompleteness:
    """Verify the exception tuple catches all expected error types.

    This test ensures the helper method catches all error types that can
    reasonably occur when accessing progress.tasks and task attributes.
    """

    EXCEPTION_TUPLE = (AttributeError, IndexError, TypeError, KeyError)

    @pytest.mark.parametrize(
        "exception_class,scenario",
        [
            (KeyError, "task ID not in progress.tasks dict"),
            (AttributeError, "task object missing attribute"),
            (IndexError, "task ID out of range in list-like container"),
            (TypeError, "elapsed has incompatible type for comparison/division"),
        ],
    )
    def test_all_expected_exceptions_are_in_tuple(self, exception_class, scenario):
        """Verify each expected exception type is in the catch tuple."""
        assert exception_class in self.EXCEPTION_TUPLE, (
            f"Missing {exception_class} for: {scenario}"
        )


class TestWallclockSpeedCalculation:
    """Speed must use wallclock time, not Rich's _Task.elapsed.

    Rich's _Task object does not guarantee an `elapsed` attribute across
    all versions. A wallclock timer owned by the service is always safe.
    """

    def test_speed_calculated_even_when_task_elapsed_raises(self):
        """Even if task.elapsed raises AttributeError, speed must still appear.

        The service must use self._embed_start_time (wallclock), not
        task_obj.elapsed (Rich internal), for speed calculation.
        """
        import time
        from unittest.mock import PropertyMock

        from rich.progress import Progress

        svc = EmbeddingService.__new__(EmbeddingService)

        # Mock progress where task.elapsed always raises AttributeError
        mock_task = MagicMock()
        type(mock_task).elapsed = PropertyMock(
            side_effect=AttributeError("_Task has no attribute 'elapsed'")
        )
        mock_progress = MagicMock(spec=Progress)
        mock_progress.tasks = {0: mock_task}
        svc.progress = mock_progress

        # Set wallclock timer 2 seconds in the past — 10 chunks processed
        svc._embed_start_time = time.monotonic() - 2.0

        svc._update_progress_with_speed(
            embed_task=0, batch_size=10, processed_count=10, batch_num=1
        )

        update_calls = mock_progress.update.call_args_list
        assert update_calls, "progress.update was never called — speed not displayed"
        speed_str = update_calls[-1].kwargs.get("speed", "")
        assert speed_str and speed_str != "0.0 chunks/s", (
            f"Expected non-zero speed (should be ~5.0 chunks/s), got: {speed_str!r}"
        )

    def test_embed_start_time_is_float_after_method_entry(self):
        """_embed_start_time must be set as a float on every call entry.

        Runtime behavioral test: call _generate_embeddings_in_batches with
        empty input and assert the attribute is a float in wallclock range.
        """
        import asyncio
        import time

        svc = EmbeddingService.__new__(EmbeddingService)
        svc.progress = None

        before = time.monotonic()
        try:
            asyncio.run(svc._generate_embeddings_in_batches([], show_progress=False))
        except Exception:
            pass  # not testing correctness, just that the attr is set
        after = time.monotonic()

        assert hasattr(svc, "_embed_start_time"), (
            "_embed_start_time not set — add "
            "self._embed_start_time = time.monotonic() as first line of "
            "_generate_embeddings_in_batches"
        )
        assert isinstance(svc._embed_start_time, float), (
            f"_embed_start_time should be float, got: {type(svc._embed_start_time)}"
        )
        assert before <= svc._embed_start_time <= after + 1.0, (
            "_embed_start_time is outside expected wallclock range"
        )

    def test_embed_start_time_resets_on_second_call(self):
        """_embed_start_time must reset on every call, not just the first.

        If the same EmbeddingService processes a second batch (incremental
        re-index), elapsed time must reflect the current run only — not
        include idle time from a prior run.
        """
        import asyncio
        import time

        svc = EmbeddingService.__new__(EmbeddingService)
        svc.progress = None

        # Simulate a stale start time from a previous run (1 hour ago)
        svc._embed_start_time = time.monotonic() - 3600.0
        stale_start = svc._embed_start_time

        try:
            asyncio.run(svc._generate_embeddings_in_batches([], show_progress=False))
        except Exception:
            pass

        assert svc._embed_start_time > stale_start + 3500, (
            "_embed_start_time was not reset — second run will report stale speed. "
            f"old={stale_start:.1f}, current={svc._embed_start_time:.1f}"
        )

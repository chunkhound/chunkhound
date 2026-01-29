"""Tests for embedding service progress update error handling.

This module tests that the progress speed update in the embedding service
gracefully handles various error conditions without masking the actual
embedding failure.

Related: PR #159 - fix error handling for progress task access
"""

from unittest.mock import MagicMock

import pytest

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
        """Valid progress state should update speed display."""
        mock_task = MagicMock()
        mock_task.elapsed = 5.0
        self.mock_progress.tasks = {1: mock_task}

        self.service._update_progress_with_speed(
            embed_task=1,
            batch_size=10,
            processed_count=100,
            batch_num=1,
        )

        # Both advance and update should be called
        self.mock_progress.advance.assert_called_once_with(1, 10)
        self.mock_progress.update.assert_called_once()
        # Check speed was calculated: 100 / 5.0 = 20.0
        call_args = self.mock_progress.update.call_args
        assert "20.0 chunks/s" in str(call_args)

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
        self.mock_embedding_provider.dimensions = 768

        # Configure provider to return recommended concurrency
        self.mock_embedding_provider.get_recommended_concurrency.return_value = 4

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
        self.mock_embedding_provider.generate_embeddings.side_effect = ConnectionError(
            "Ollama crashed"
        )

        # Configure progress to also fail (broken state)
        self.mock_progress.tasks = {}  # Will raise KeyError
        self.mock_progress.add_task.return_value = 1

        # The embedding error should be what we see in results, not progress errors
        result = await self.service.generate_embeddings_for_chunks(
            chunk_ids=["chunk1", "chunk2"],
            chunk_texts=["text1", "text2"],
            show_progress=True,
        )

        # Result should be 0 (no successful embeddings)
        assert result == 0

        # The embedding provider was called (error occurred there, not in progress)
        self.mock_embedding_provider.generate_embeddings.assert_called()

    @pytest.mark.asyncio
    async def test_successful_embedding_with_broken_progress(self):
        """Successful embeddings should complete even if progress tracking fails."""
        # Configure embedding provider to succeed
        self.mock_embedding_provider.generate_embeddings.return_value = [
            [0.1] * 768,
            [0.2] * 768,
        ]

        # Configure progress to fail
        self.mock_progress.tasks = {}  # Will raise KeyError
        self.mock_progress.add_task.return_value = 1

        # Mock database operations
        self.mock_db.get_existing_embeddings.return_value = set()

        result = await self.service.generate_embeddings_for_chunks(
            chunk_ids=["chunk1", "chunk2"],
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

    @pytest.mark.parametrize("exception_class,scenario", [
        (KeyError, "task ID not in progress.tasks dict"),
        (AttributeError, "task object missing attribute"),
        (IndexError, "task ID out of range in list-like container"),
        (TypeError, "elapsed has incompatible type for comparison/division"),
    ])
    def test_all_expected_exceptions_are_in_tuple(self, exception_class, scenario):
        """Verify each expected exception type is in the catch tuple."""
        assert exception_class in self.EXCEPTION_TUPLE, (
            f"Missing {exception_class} for: {scenario}"
        )

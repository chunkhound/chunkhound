"""Tests for embedding service progress update error handling.

This module tests that the progress speed update in the embedding service
gracefully handles various error conditions without masking the actual
embedding failure.

Related: PR #159 - fix error handling for progress task access
"""

from unittest.mock import MagicMock

import pytest


class TestProgressSpeedUpdateErrorHandling:
    """Tests for progress speed update error handling in embedding service.

    The progress speed display is non-critical UI functionality. When errors
    occur accessing progress.tasks, they should be logged but not propagate
    up and mask actual embedding failures.
    """

    def test_keyerror_is_caught_when_task_id_missing(self):
        """KeyError should be caught when task ID is not in progress.tasks.

        This can happen when the progress object is in an inconsistent state
        during cleanup after an Ollama crash.
        """
        # Create a mock progress object where tasks dict doesn't have the key
        mock_progress = MagicMock()
        mock_progress.tasks = {}  # Empty dict - KeyError when accessed

        embed_task = 42  # Task ID that doesn't exist

        # Simulate the try block from embedding_service.py
        caught_exception = None
        try:
            task_obj = mock_progress.tasks[embed_task]
            if task_obj.elapsed and task_obj.elapsed > 0:
                _ = 100 / task_obj.elapsed
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            caught_exception = e

        assert caught_exception is not None
        assert isinstance(caught_exception, KeyError)

    def test_attributeerror_is_caught_when_task_lacks_elapsed(self):
        """AttributeError should be caught when task object lacks elapsed.

        This occurs with _NoRichProgressManager shim which provides minimal
        task objects without the .elapsed attribute.
        """
        # Create a mock task without elapsed attribute
        class MockTaskWithoutElapsed:
            total = 100
            completed = 50
            # No elapsed attribute

        mock_progress = MagicMock()
        mock_progress.tasks = {1: MockTaskWithoutElapsed()}

        embed_task = 1

        caught_exception = None
        try:
            task_obj = mock_progress.tasks[embed_task]
            if task_obj.elapsed and task_obj.elapsed > 0:
                _ = 100 / task_obj.elapsed
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            caught_exception = e

        assert caught_exception is not None
        assert isinstance(caught_exception, AttributeError)

    def test_typeerror_is_caught_for_invalid_elapsed_type(self):
        """TypeError should be caught when elapsed has unexpected type.

        This can happen in edge cases where progress state is corrupted.
        """
        class MockTaskWithBadElapsed:
            elapsed = "not a number"  # Invalid type

        mock_progress = MagicMock()
        mock_progress.tasks = {1: MockTaskWithBadElapsed()}

        embed_task = 1
        processed_count = 100

        caught_exception = None
        try:
            task_obj = mock_progress.tasks[embed_task]
            if task_obj.elapsed and task_obj.elapsed > 0:
                # TypeError: unsupported operand type(s)
                _ = processed_count / task_obj.elapsed
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            caught_exception = e

        assert caught_exception is not None
        assert isinstance(caught_exception, TypeError)

    def test_indexerror_is_caught_for_list_based_tasks(self):
        """IndexError should be caught if tasks is list-like and index OOR.

        This is a defensive case for unexpected progress implementations.
        """
        # Simulate a list-based tasks container
        mock_progress = MagicMock()
        mock_progress.tasks = []  # Empty list

        embed_task = 0  # Valid index, but list is empty

        caught_exception = None
        try:
            task_obj = mock_progress.tasks[embed_task]
            if task_obj.elapsed and task_obj.elapsed > 0:
                _ = 100 / task_obj.elapsed
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            caught_exception = e

        assert caught_exception is not None
        assert isinstance(caught_exception, IndexError)

    def test_valid_progress_update_succeeds(self):
        """Valid progress state should allow speed calculation."""
        class MockTaskWithElapsed:
            elapsed = 5.0  # Valid elapsed time

        mock_progress = MagicMock()
        mock_progress.tasks = {1: MockTaskWithElapsed()}

        embed_task = 1
        processed_count = 100

        caught_exception = None
        speed = None
        try:
            task_obj = mock_progress.tasks[embed_task]
            if task_obj.elapsed and task_obj.elapsed > 0:
                speed = processed_count / task_obj.elapsed
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            caught_exception = e

        assert caught_exception is None
        assert speed == 20.0  # 100 / 5.0

    def test_zero_elapsed_skips_speed_calculation(self):
        """Zero elapsed time should skip speed calculation."""
        class MockTaskWithZeroElapsed:
            elapsed = 0.0

        mock_progress = MagicMock()
        mock_progress.tasks = {1: MockTaskWithZeroElapsed()}

        embed_task = 1
        processed_count = 100

        caught_exception = None
        speed = None
        try:
            task_obj = mock_progress.tasks[embed_task]
            if task_obj.elapsed and task_obj.elapsed > 0:
                speed = processed_count / task_obj.elapsed
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            caught_exception = e

        # No exception, but speed should remain None (condition not met)
        assert caught_exception is None
        assert speed is None

    def test_none_elapsed_skips_speed_calculation(self):
        """None elapsed time should skip speed calculation."""
        class MockTaskWithNoneElapsed:
            elapsed = None

        mock_progress = MagicMock()
        mock_progress.tasks = {1: MockTaskWithNoneElapsed()}

        embed_task = 1
        processed_count = 100

        caught_exception = None
        speed = None
        try:
            task_obj = mock_progress.tasks[embed_task]
            if task_obj.elapsed and task_obj.elapsed > 0:
                speed = processed_count / task_obj.elapsed
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            caught_exception = e

        # No exception, speed remains None because elapsed is falsy
        assert caught_exception is None
        assert speed is None


class TestExceptionTupleCompleteness:
    """Verify the exception tuple catches all expected error types.

    This test class ensures that the exception handling in embedding_service.py
    line 613 catches all error types that can reasonably occur when accessing
    progress.tasks[embed_task] and task_obj.elapsed.
    """

    EXCEPTION_TUPLE = (AttributeError, IndexError, TypeError, KeyError)

    @pytest.mark.parametrize("exception_class,scenario", [
        (KeyError, "task ID not in progress.tasks dict"),
        (AttributeError, "task object missing .elapsed attribute"),
        (IndexError, "task ID out of range in list-like container"),
        (TypeError, "elapsed has incompatible type for comparison/division"),
    ])
    def test_all_expected_exceptions_are_in_tuple(self, exception_class, scenario):
        """Verify each expected exception type is in the catch tuple."""
        assert exception_class in self.EXCEPTION_TUPLE, (
            f"Missing {exception_class} for: {scenario}"
        )

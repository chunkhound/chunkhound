"""Tests for embedding error classification system."""

import pytest
import time
from unittest.mock import Mock

from chunkhound.core.exceptions import (
    EmbeddingErrorClassification,
    EmbeddingErrorClassifier,
    ErrorCounters,
    ErrorSample,
)


class TestEmbeddingErrorClassification:
    """Test the EmbeddingErrorClassification enum."""

    def test_enum_values(self):
        """Test that the enum has the expected values."""
        assert EmbeddingErrorClassification.PERMANENT.value == "permanent"
        assert EmbeddingErrorClassification.TRANSIENT.value == "transient"
        assert EmbeddingErrorClassification.BATCH_RECOVERABLE.value == "batch_recoverable"

    def test_enum_members(self):
        """Test that all expected members exist."""
        members = list(EmbeddingErrorClassification)
        assert len(members) == 3
        assert EmbeddingErrorClassification.PERMANENT in members
        assert EmbeddingErrorClassification.TRANSIENT in members
        assert EmbeddingErrorClassification.BATCH_RECOVERABLE in members


class TestErrorSample:
    """Test the ErrorSample dataclass."""

    def test_creation(self):
        """Test creating an ErrorSample."""
        sample = ErrorSample(
            timestamp=1234567890.0,
            exception_type="ValueError",
            message="Test error",
            provider="openai",
            model="text-embedding-3-small",
            context={"batch_size": 10},
        )

        assert sample.timestamp == 1234567890.0
        assert sample.exception_type == "ValueError"
        assert sample.message == "Test error"
        assert sample.provider == "openai"
        assert sample.model == "text-embedding-3-small"
        assert sample.context == {"batch_size": 10}

    def test_optional_fields(self):
        """Test that optional fields can be None."""
        sample = ErrorSample(
            timestamp=time.time(),
            exception_type="Exception",
            message="Test",
        )

        assert sample.provider is None
        assert sample.model is None
        assert sample.context is None


class TestErrorCounters:
    """Test the ErrorCounters class."""

    def test_initialization(self):
        """Test ErrorCounters initialization."""
        counters = ErrorCounters()
        assert counters.permanent == 0
        assert counters.transient == 0
        assert counters.batch_recoverable == 0
        assert counters.permanent_samples == []
        assert counters.transient_samples == []
        assert counters.batch_recoverable_samples == []
        assert counters.max_samples_per_type == 5

    def test_increment(self):
        """Test incrementing counters."""
        counters = ErrorCounters()

        counters.increment(EmbeddingErrorClassification.PERMANENT)
        assert counters.permanent == 1

        counters.increment(EmbeddingErrorClassification.TRANSIENT)
        assert counters.transient == 1

        counters.increment(EmbeddingErrorClassification.BATCH_RECOVERABLE)
        assert counters.batch_recoverable == 1

    def test_add_sample(self):
        """Test adding error samples."""
        counters = ErrorCounters()
        exception = ValueError("Test error")

        counters.add_sample(
            EmbeddingErrorClassification.PERMANENT,
            exception,
            provider="openai",
            model="test-model",
            context={"test": "data"},
        )

        assert len(counters.permanent_samples) == 1
        sample = counters.permanent_samples[0]
        assert sample.exception_type == "ValueError"
        assert sample.message == "Test error"
        assert sample.provider == "openai"
        assert sample.model == "test-model"
        assert sample.context == {"test": "data"}

    def test_sample_limit(self):
        """Test that samples are limited to max_samples_per_type."""
        counters = ErrorCounters(max_samples_per_type=2)
        exception = RuntimeError("Test error")

        # Add 3 samples
        for i in range(3):
            counters.add_sample(
                EmbeddingErrorClassification.TRANSIENT,
                exception,
                context={"index": i},
            )

        # Should only keep the last 2
        assert len(counters.transient_samples) == 2
        assert counters.transient_samples[0].context == {"index": 1}
        assert counters.transient_samples[1].context == {"index": 2}

    def test_get_stats(self):
        """Test getting statistics."""
        counters = ErrorCounters()

        # Add some data
        counters.increment(EmbeddingErrorClassification.PERMANENT)
        counters.increment(EmbeddingErrorClassification.TRANSIENT)
        counters.increment(EmbeddingErrorClassification.TRANSIENT)
        counters.add_sample(EmbeddingErrorClassification.PERMANENT, ValueError("test"))

        stats = counters.get_stats()
        assert stats["counts"]["permanent"] == 1
        assert stats["counts"]["transient"] == 2
        assert stats["counts"]["batch_recoverable"] == 0
        assert stats["counts"]["total"] == 3
        assert stats["samples"]["permanent"] == 1
        assert stats["samples"]["transient"] == 0
        assert stats["samples"]["batch_recoverable"] == 0

    def test_reset(self):
        """Test resetting counters."""
        counters = ErrorCounters()

        # Add some data
        counters.increment(EmbeddingErrorClassification.PERMANENT)
        counters.add_sample(EmbeddingErrorClassification.PERMANENT, ValueError("test"))

        # Reset
        counters.reset()

        assert counters.permanent == 0
        assert counters.permanent_samples == []
        stats = counters.get_stats()
        assert stats["counts"]["total"] == 0


class TestEmbeddingErrorClassifier:
    """Test the EmbeddingErrorClassifier class."""

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = EmbeddingErrorClassifier()
        assert isinstance(classifier.counters, ErrorCounters)

        # Test with custom counters
        custom_counters = ErrorCounters(max_samples_per_type=10)
        classifier = EmbeddingErrorClassifier(counters=custom_counters)
        assert classifier.counters is custom_counters

    def test_classify_permanent_errors(self):
        """Test classification of permanent errors."""
        classifier = EmbeddingErrorClassifier()

        # Test various permanent error patterns
        permanent_errors = [
            ValueError("Invalid content encoding"),
            RuntimeError("Unsupported encoding detected"),
            Exception("Content too large for processing"),
            ValueError("Invalid API key provided"),
            Exception("Model not found: invalid-model"),
        ]

        for error in permanent_errors:
            classification = classifier.classify_exception(error)
            assert classification == EmbeddingErrorClassification.PERMANENT

    def test_classify_transient_errors(self):
        """Test classification of transient errors."""
        classifier = EmbeddingErrorClassifier()

        # Test various transient error patterns
        transient_errors = [
            TimeoutError("Request timed out"),
            ConnectionError("Connection failed"),
            Exception("Rate limit exceeded"),
            RuntimeError("Service temporarily unavailable"),
            Exception("Too many requests, try again later"),
        ]

        for error in transient_errors:
            classification = classifier.classify_exception(error)
            assert classification == EmbeddingErrorClassification.TRANSIENT

    def test_classify_batch_recoverable_errors(self):
        """Test classification of batch recoverable errors."""
        classifier = EmbeddingErrorClassifier()

        # Test various batch recoverable error patterns
        batch_errors = [
            Exception("Maximum context length exceeded"),
            ValueError("Token limit exceeded for batch"),
            RuntimeError("Input too long for model"),
            Exception("Batch size too large"),
        ]

        for error in batch_errors:
            classification = classifier.classify_exception(error)
            assert classification == EmbeddingErrorClassification.BATCH_RECOVERABLE

    def test_classify_exception_types(self):
        """Test classification based on exception types."""
        classifier = EmbeddingErrorClassifier()

        # Mock OpenAI-style exceptions
        rate_limit_error = Mock()
        rate_limit_error.__class__.__name__ = "RateLimitError"
        rate_limit_error.__str__ = Mock(return_value="Rate limit exceeded")

        timeout_error = Mock()
        timeout_error.__class__.__name__ = "APITimeoutError"
        timeout_error.__str__ = Mock(return_value="Request timed out")

        bad_request_error = Mock()
        bad_request_error.__class__.__name__ = "BadRequestError"
        bad_request_error.__str__ = Mock(return_value="Maximum context length exceeded")

        # Test classifications
        assert classifier.classify_exception(rate_limit_error) == EmbeddingErrorClassification.TRANSIENT
        assert classifier.classify_exception(timeout_error) == EmbeddingErrorClassification.TRANSIENT
        assert classifier.classify_exception(bad_request_error) == EmbeddingErrorClassification.BATCH_RECOVERABLE

    def test_classify_with_context(self):
        """Test classification with provider and model context."""
        classifier = EmbeddingErrorClassifier()

        error = ValueError("Invalid content encoding")
        classification = classifier.classify_exception(
            error,
            provider="openai",
            model="text-embedding-3-small",
            context={"batch_size": 100},
        )

        # Should be classified as permanent due to "invalid content" pattern
        assert classification == EmbeddingErrorClassification.PERMANENT

        # Check that sample was recorded with context
        assert len(classifier.counters.permanent_samples) == 1
        sample = classifier.counters.permanent_samples[0]
        assert sample.provider == "openai"
        assert sample.model == "text-embedding-3-small"
        assert sample.context == {"batch_size": 100}

    def test_default_classification(self):
        """Test default classification for unknown errors."""
        classifier = EmbeddingErrorClassifier()

        # Unknown error should default to transient (safer)
        unknown_error = Exception("Some unknown error occurred")
        classification = classifier.classify_exception(unknown_error)
        assert classification == EmbeddingErrorClassification.TRANSIENT

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        classifier = EmbeddingErrorClassifier()

        # Classify multiple errors
        classifier.classify_exception(ValueError("Invalid content"))  # Permanent
        classifier.classify_exception(TimeoutError("Request timed out"))  # Transient
        classifier.classify_exception(Exception("Token limit exceeded"))  # Batch recoverable
        classifier.classify_exception(TimeoutError("Connection failed"))  # Transient

        stats = classifier.get_error_stats()
        assert stats["counts"]["permanent"] == 1
        assert stats["counts"]["transient"] == 2
        assert stats["counts"]["batch_recoverable"] == 1
        assert stats["counts"]["total"] == 4

    def test_reset_stats(self):
        """Test resetting statistics."""
        classifier = EmbeddingErrorClassifier()

        # Add some data
        classifier.classify_exception(ValueError("test"))
        assert classifier.get_error_stats()["counts"]["total"] == 1

        # Reset
        classifier.reset_stats()
        assert classifier.get_error_stats()["counts"]["total"] == 0
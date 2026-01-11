"""Embedding Error Classification System for ChunkHound.

This module provides enhanced error classification for embedding failures,
enabling proper retry logic based on error recoverability.

The classification system categorizes errors as:
- PERMANENT: Errors that cannot be recovered from (oversized chunks, invalid content)
- TRANSIENT: Errors that may succeed on retry (network timeouts, rate limits)
- BATCH_RECOVERABLE: Errors that can be recovered by splitting batches (token limits)
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time


class EmbeddingErrorClassification(Enum):
    """Classification categories for embedding errors."""

    PERMANENT = "permanent"
    """Permanent errors that cannot be recovered from.
    Examples: oversized chunks, invalid content, unsupported encoding."""

    TRANSIENT = "transient"
    """Transient errors that may succeed on retry.
    Examples: network timeouts, rate limits, temporary service unavailability."""

    BATCH_RECOVERABLE = "batch_recoverable"
    """Errors that can be recovered by splitting or reducing batch size.
    Examples: token limits exceeded, partial batch failures."""


@dataclass
class ErrorSample:
    """Represents a single error sample for logging and analysis."""

    timestamp: float
    exception_type: str
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ErrorCounters:
    """Tracks error counts and samples for each error classification."""

    permanent: int = 0
    transient: int = 0
    batch_recoverable: int = 0

    # Sample collections with configurable limits
    permanent_samples: List[ErrorSample] = field(default_factory=list)
    transient_samples: List[ErrorSample] = field(default_factory=list)
    batch_recoverable_samples: List[ErrorSample] = field(default_factory=list)

    # Configuration
    max_samples_per_type: int = 5

    def increment(self, classification: EmbeddingErrorClassification) -> None:
        """Increment counter for the given classification."""
        if classification == EmbeddingErrorClassification.PERMANENT:
            self.permanent += 1
        elif classification == EmbeddingErrorClassification.TRANSIENT:
            self.transient += 1
        elif classification == EmbeddingErrorClassification.BATCH_RECOVERABLE:
            self.batch_recoverable += 1

    def add_sample(
        self,
        classification: EmbeddingErrorClassification,
        exception: Exception,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an error sample to the appropriate collection."""
        sample = ErrorSample(
            timestamp=time.time(),
            exception_type=type(exception).__name__,
            message=str(exception),
            provider=provider,
            model=model,
            context=context,
        )

        # Get the appropriate sample list
        if classification == EmbeddingErrorClassification.PERMANENT:
            samples = self.permanent_samples
        elif classification == EmbeddingErrorClassification.TRANSIENT:
            samples = self.transient_samples
        elif classification == EmbeddingErrorClassification.BATCH_RECOVERABLE:
            samples = self.batch_recoverable_samples
        else:
            return

        # Add sample, maintaining max_samples_per_type limit
        samples.append(sample)
        if len(samples) > self.max_samples_per_type:
            samples.pop(0)  # Remove oldest sample

    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "counts": {
                "permanent": self.permanent,
                "transient": self.transient,
                "batch_recoverable": self.batch_recoverable,
                "total": self.permanent + self.transient + self.batch_recoverable,
            },
            "samples": {
                "permanent": len(self.permanent_samples),
                "transient": len(self.transient_samples),
                "batch_recoverable": len(self.batch_recoverable_samples),
            },
        }

    def reset(self) -> None:
        """Reset all counters and samples."""
        self.permanent = 0
        self.transient = 0
        self.batch_recoverable = 0
        self.permanent_samples.clear()
        self.transient_samples.clear()
        self.batch_recoverable_samples.clear()


class EmbeddingErrorClassifier:
    """Classifies embedding exceptions into recoverable categories."""

    def __init__(self, counters: Optional[ErrorCounters] = None):
        """Initialize the classifier.

        Args:
            counters: Optional ErrorCounters instance for tracking statistics.
                     If None, no statistics will be tracked.
        """
        self.counters = counters or ErrorCounters()

    def classify_exception(
        self,
        exception: Exception,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingErrorClassification:
        """Classify an exception based on its type and message.

        Args:
            exception: The exception to classify
            provider: Optional provider name for context
            model: Optional model name for context
            context: Optional additional context

        Returns:
            Classification category for the error
        """
        exception_type = type(exception).__name__
        message = str(exception).lower()

        # Classify based on exception type and message patterns
        classification = self._classify_by_patterns(exception_type, message)

        # Track statistics if counters are enabled
        self.counters.increment(classification)
        self.counters.add_sample(classification, exception, provider, model, context)

        return classification

    def _classify_by_patterns(
        self, exception_type: str, message: str
    ) -> EmbeddingErrorClassification:
        """Classify exception based on type and message patterns.

        Args:
            exception_type: The exception class name
            message: The exception message (lowercased)

        Returns:
            Classification category
        """
        # PERMANENT errors - cannot be recovered from
        permanent_patterns = [
            # Content validation errors
            "invalid content",
            "unsupported encoding",
            "invalid character",
            "malformed content",
            # Size limits that cannot be worked around
            "content too large",
            "maximum file size exceeded",
            "chunk size exceeds maximum",
            "text too large",
            "oversized chunks found",
            # Authentication/configuration errors
            "invalid api key",
            "authentication failed",
            "unauthorized",
            "forbidden",
            "invalid credentials",
            # Model/configuration errors
            "model not found",
            "model not available",
            "unsupported model",
            "invalid model",
            # Request validation errors that aren't token-related
            "bad request" if "token" not in message else None,
            "invalid request" if "token" not in message else None,
        ]

        # TRANSIENT errors - may succeed on retry
        transient_patterns = [
            # Network/connectivity issues
            "timeout",
            "connection error",
            "connection failed",
            "connection reset",
            "connection refused",
            "network error",
            "dns resolution",
            "ssl error",
            "certificate error",
            # Service availability
            "service unavailable",
            "server error",
            "internal server error",
            "temporarily unavailable",
            "maintenance",
            # Rate limiting
            "rate limit exceeded",
            "too many requests",
            "quota exceeded",
            # Temporary service issues
            "temporary failure",
            "try again later",
            "back off",
        ]

        # BATCH_RECOVERABLE errors - can be fixed by splitting batches
        batch_recoverable_patterns = [
            # Token limit errors
            "maximum context length",
            "token limit exceeded",
            "tokens per request",
            "input too long",
            "text too long",
            "batch size too large",
            "too many tokens",
            # Partial batch failures
            "partial failure",
            "batch partially failed",
            "some embeddings failed",
        ]

        # Check patterns in order of specificity (most specific first)
        # Start with batch_recoverable since token errors are most actionable
        for pattern in batch_recoverable_patterns:
            if pattern and pattern in message:
                return EmbeddingErrorClassification.BATCH_RECOVERABLE

        # Check transient patterns
        for pattern in transient_patterns:
            if pattern in message:
                return EmbeddingErrorClassification.TRANSIENT

        # Check permanent patterns
        for pattern in permanent_patterns:
            if pattern and pattern in message:
                return EmbeddingErrorClassification.PERMANENT

        # Special handling for specific exception types
        if exception_type in ("RateLimitError", "TooManyRequests"):
            return EmbeddingErrorClassification.TRANSIENT

        if exception_type in ("APITimeoutError", "TimeoutError", "ConnectTimeout"):
            return EmbeddingErrorClassification.TRANSIENT

        if exception_type in ("APIConnectionError", "ConnectionError"):
            return EmbeddingErrorClassification.TRANSIENT

        if exception_type == "BadRequestError":
            # BadRequest could be token limits (recoverable) or invalid content (permanent)
            if any(token_pattern in message for token_pattern in [
                "maximum context length", "token limit", "tokens per request"
            ]):
                return EmbeddingErrorClassification.BATCH_RECOVERABLE
            else:
                return EmbeddingErrorClassification.PERMANENT

        # Default to transient for unknown errors (safer to retry)
        return EmbeddingErrorClassification.TRANSIENT

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics from the counters."""
        return self.counters.get_stats()

    def reset_stats(self) -> None:
        """Reset error statistics."""
        self.counters.reset()
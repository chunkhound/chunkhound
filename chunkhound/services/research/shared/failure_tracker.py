"""Failure tracking utilities for research operations.

Provides structured failure metrics collection and reporting.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from chunkhound.services.research.shared.error_categorization import categorize_error


@dataclass
class FailureInfo:
    """Information about a single failure."""

    item_description: str
    error_message: str
    error_type: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with item, error, and type fields
        """
        return {
            "item": self.item_description,
            "error": self.error_message,
            "type": self.error_type,
        }


@dataclass
class FailureMetrics:
    """Tracks failures across a batch of operations.

    Collects failure information and provides structured metrics output.
    """

    total_operations: int
    failures: list[FailureInfo] = field(default_factory=list)

    def add_failure(self, item_description: str, exception: Exception) -> None:
        """Record a failure with automatic error categorization.

        Args:
            item_description: Description of the failed item (e.g., file path, query text)
            exception: The exception that occurred
        """
        error_type = categorize_error(exception)
        error_message = f"{type(exception).__name__}: {str(exception)}"

        self.failures.append(
            FailureInfo(
                item_description=item_description,
                error_message=error_message,
                error_type=error_type,
            )
        )

    @property
    def success_count(self) -> int:
        """Number of successful operations."""
        return self.total_operations - len(self.failures)

    @property
    def failure_count(self) -> int:
        """Number of failed operations."""
        return len(self.failures)

    @property
    def failure_rate(self) -> float:
        """Failure rate as a fraction (0.0 to 1.0)."""
        if self.total_operations == 0:
            return 0.0
        return len(self.failures) / self.total_operations

    def to_dict(self, max_items: int = 5) -> dict[str, Any]:
        """Convert to structured dictionary representation.

        Args:
            max_items: Maximum number of failure items to include

        Returns:
            Dictionary with count, total, rate, by_type, and items fields
        """
        if not self.failures:
            # Skip verbose output when no failures
            return {
                "count": 0,
                "total": self.total_operations,
                "rate": 0.0,
            }

        # Count failures by type
        type_counts = Counter(f.error_type for f in self.failures)

        return {
            "count": len(self.failures),
            "total": self.total_operations,
            "rate": self.failure_rate,
            "by_type": dict(type_counts),
            "items": [f.to_dict() for f in self.failures[:max_items]],
        }

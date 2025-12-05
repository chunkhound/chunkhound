"""ChunkHound Core Exceptions - Core exception classes for error handling.

This module contains the exception hierarchy for the ChunkHound system. These
exceptions provide clear error categorization and enable proper error handling
throughout the application.
"""

from typing import Any


class ChunkHoundError(Exception):
    """Base exception for all ChunkHound-specific errors.

    This is the root exception class that all other ChunkHound exceptions
    inherit from. It provides common functionality for error handling,
    context tracking, and debugging.
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """Initialize ChunkHound error.

        Args:
            message: Human-readable error description
            context: Optional dictionary with error context (e.g., file paths, chunk IDs)
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause

    def __str__(self) -> str:
        """Return formatted error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (context: {context_str})"
        return self.message

    def add_context(self, key: str, value: Any) -> "ChunkHoundError":
        """Add context information to the error."""
        self.context[key] = value
        return self


class ValidationError(ChunkHoundError):
    """Raised when data validation fails.

    This exception is used when input data doesn't meet expected format,
    type, or business rule requirements.
    """

    def __init__(
        self, field: str, value: Any, reason: str, context: dict[str, Any] | None = None
    ):
        """Initialize validation error.

        Args:
            field: Name of the field that failed validation
            value: The invalid value
            reason: Description of why validation failed
            context: Optional additional context
        """
        message = f"Validation failed for field '{field}': {reason}"
        super().__init__(message, context)
        self.field = field
        self.value = value
        self.reason = reason


class ModelError(ChunkHoundError):
    """Raised when domain model operations fail.

    This exception is used for errors related to domain model creation,
    manipulation, or state transitions.
    """

    def __init__(
        self,
        model_type: str,
        operation: str,
        reason: str,
        context: dict[str, Any] | None = None,
    ):
        """Initialize model error.

        Args:
            model_type: Type of model that caused the error (e.g., "File", "Chunk")
            operation: Operation that failed (e.g., "create", "update", "validate")
            reason: Description of what went wrong
            context: Optional additional context
        """
        message = f"{model_type} {operation} failed: {reason}"
        super().__init__(message, context)
        self.model_type = model_type
        self.operation = operation
        self.reason = reason


class EmbeddingError(ChunkHoundError):
    """Raised when embedding operations fail.

    This exception is used for errors related to embedding generation,
    storage, or retrieval operations.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
        reason: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize embedding error.

        Args:
            provider: Embedding provider name (e.g., "openai")
            model: Model name (e.g., "text-embedding-3-small")
            operation: Operation that failed (e.g., "generate", "store", "retrieve")
            reason: Description of what went wrong
            context: Optional additional context
        """
        parts = []
        if provider:
            parts.append(f"provider={provider}")
        if model:
            parts.append(f"model={model}")
        if operation:
            parts.append(f"operation={operation}")

        prefix = f"Embedding error ({', '.join(parts)})" if parts else "Embedding error"
        message = f"{prefix}: {reason}" if reason else prefix

        super().__init__(message, context)
        self.provider = provider
        self.model = model
        self.operation = operation
        self.reason = reason


class ParsingError(ChunkHoundError):
    """Raised when file parsing operations fail.

    This exception is used for errors related to parsing source code files,
    AST generation, or chunk extraction.
    """

    def __init__(
        self,
        file_path: str | None = None,
        language: str | None = None,
        operation: str | None = None,
        reason: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize parsing error.

        Args:
            file_path: Path to file that failed to parse
            language: Programming language being parsed
            operation: Parsing operation that failed (e.g., "parse", "extract_chunks")
            reason: Description of what went wrong
            context: Optional additional context
        """
        parts = []
        if file_path:
            parts.append(f"file={file_path}")
        if language:
            parts.append(f"language={language}")
        if operation:
            parts.append(f"operation={operation}")

        prefix = f"Parsing error ({', '.join(parts)})" if parts else "Parsing error"
        message = f"{prefix}: {reason}" if reason else prefix

        super().__init__(message, context)
        self.file_path = file_path
        self.language = language
        self.operation = operation
        self.reason = reason


class DatabaseError(ChunkHoundError):
    """Raised when database operations fail.

    This exception is used for errors related to database connections,
    queries, transactions, or data integrity issues.
    """

    def __init__(
        self,
        operation: str | None = None,
        table: str | None = None,
        reason: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize database error.

        Args:
            operation: Database operation that failed (e.g., "insert", "query", "update")
            table: Database table involved in the operation
            reason: Description of what went wrong
            context: Optional additional context
        """
        parts = []
        if operation:
            parts.append(f"operation={operation}")
        if table:
            parts.append(f"table={table}")

        prefix = f"Database error ({', '.join(parts)})" if parts else "Database error"
        message = f"{prefix}: {reason}" if reason else prefix

        super().__init__(message, context)
        self.operation = operation
        self.table = table
        self.reason = reason


class ConfigurationError(ChunkHoundError):
    """Raised when configuration is invalid or missing.

    This exception is used for errors related to configuration files,
    environment variables, or system setup issues.
    """

    def __init__(
        self,
        config_key: str | None = None,
        config_value: Any | None = None,
        reason: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize configuration error.

        Args:
            config_key: Configuration key that caused the error
            config_value: Invalid configuration value
            reason: Description of what went wrong
            context: Optional additional context
        """
        if config_key:
            message = f"Configuration error for '{config_key}': {reason}"
        else:
            message = (
                f"Configuration error: {reason}" if reason else "Configuration error"
            )

        super().__init__(message, context)
        self.config_key = config_key
        self.config_value = config_value
        self.reason = reason


class ProviderError(ChunkHoundError):
    """Raised when external provider operations fail.

    This exception is used for errors related to external service providers
    like OpenAI API or other third-party integrations.
    """

    def __init__(
        self,
        provider: str | None = None,
        service: str | None = None,
        status_code: int | None = None,
        reason: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize provider error.

        Args:
            provider: Provider name (e.g., "openai")
            service: Service or endpoint that failed
            status_code: HTTP status code if applicable
            reason: Description of what went wrong
            context: Optional additional context
        """
        parts = []
        if provider:
            parts.append(f"provider={provider}")
        if service:
            parts.append(f"service={service}")
        if status_code:
            parts.append(f"status={status_code}")

        prefix = f"Provider error ({', '.join(parts)})" if parts else "Provider error"
        message = f"{prefix}: {reason}" if reason else prefix

        super().__init__(message, context)
        self.provider = provider
        self.service = service
        self.status_code = status_code
        self.reason = reason


class DiskUsageLimitExceededError(ChunkHoundError):
    """Raised when database disk usage exceeds configured limits.

    This exception is used when the database file size grows beyond the
    configured maximum disk usage limit, preventing further indexing operations.
    """

    def __init__(
        self,
        current_size_mb: float,
        limit_mb: float,
        context: dict[str, Any] | None = None,
    ):
        """Initialize disk usage limit exceeded error.

        Args:
            current_size_mb: Current database size in MB
            limit_mb: Configured limit in MB
            context: Optional additional context
        """
        message = f"Database disk usage limit exceeded: {current_size_mb:.1f} MB >= {limit_mb:.1f} MB"
        super().__init__(message, context)
        self.current_size_mb = current_size_mb
        self.limit_mb = limit_mb

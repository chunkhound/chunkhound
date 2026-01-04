"""Exception classes for MCP server.

This module defines exception types used throughout the MCP server.
It is intentionally kept separate from other modules to avoid circular imports.
"""


class MCPError(Exception):
    """Base exception for MCP operations."""

    pass


class ServiceNotInitializedError(MCPError):
    """Raised when services are accessed before initialization."""

    pass


class EmbeddingTimeoutError(MCPError):
    """Raised when embedding operations timeout."""

    pass


class EmbeddingProviderError(MCPError):
    """Raised when embedding provider is not available."""

    pass


class ToolNotFoundError(MCPError):
    """Raised when a requested tool does not exist."""

    pass


class ToolExecutionError(MCPError):
    """Raised when tool execution fails."""

    pass


class InvalidArgumentError(MCPError):
    """Raised when tool arguments are invalid."""

    pass

"""Common utilities and error handling for MCP server.

This module provides shared utilities for the stdio MCP server,
including error handling, response formatting, and validation helpers.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import mcp.types as types  # noqa: F401

    from chunkhound.database_factory import DatabaseServices
    from chunkhound.embeddings import EmbeddingManager
    from chunkhound.llm_manager import LLMManager

# Re-export exceptions for backward compatibility
from .exceptions import (
    MCPError,
)

T = TypeVar("T")

# Shared protocol constants
SUPPORTED_PROTOCOL_VERSIONS = {"2024-11-05", "2025-11-25"}
CURRENT_PROTOCOL_VERSION = "2025-11-25"

# Request size limits
MAX_REQUEST_BODY_SIZE = 1024 * 1024  # 1MB


def is_safe_relative_path(path: str) -> bool:
    """Validate that a relative path doesn't escape the base directory.

    Checks for path traversal attacks like '../../../etc/passwd'.

    Args:
        path: The path string to validate

    Returns:
        True if the path is safe, False if it attempts directory traversal
    """
    # Normalize the path to resolve . and ..
    normalized = os.path.normpath(path)

    # Reject if it starts with parent references or is absolute
    if normalized.startswith("..") or normalized.startswith(os.sep):
        return False

    # On Windows, also check for drive letters
    if os.name == "nt" and len(normalized) >= 2 and normalized[1] == ":":
        return False

    return True


def validate_and_join_path(base: str | Path, relative: str) -> str | None:
    """Safely join a base path with a relative path.

    Validates that the relative path doesn't escape the base directory
    before joining.

    Args:
        base: The base directory path
        relative: The relative path to join

    Returns:
        The joined path string if safe, None if path traversal detected.
        Returns the path using the original base (not symlink-resolved)
        to preserve caller's path format.
    """
    if not is_safe_relative_path(relative):
        return None

    # Resolve for security check (follows symlinks to detect escapes)
    base_path = Path(base).resolve()
    joined_resolved = (base_path / relative).resolve()

    # Double-check: ensure the resolved path is still under base
    try:
        # Check that the joined path starts with the base path
        joined_resolved.relative_to(base_path)
        # Return using original base to preserve caller's path format
        # (without symlink resolution)
        original_base = Path(base)
        return str(original_base / relative)
    except ValueError:
        # relative_to raises ValueError if not a subpath
        return None


# Lazy import to avoid circular dependency
_TOOL_REGISTRY = None
_execute_tool = None


def _get_tool_registry():
    """Lazy load TOOL_REGISTRY to avoid circular import."""
    global _TOOL_REGISTRY
    if _TOOL_REGISTRY is None:
        from .tools import TOOL_REGISTRY

        _TOOL_REGISTRY = TOOL_REGISTRY
    return _TOOL_REGISTRY


def _get_execute_tool():
    """Lazy load execute_tool to avoid circular import."""
    global _execute_tool
    if _execute_tool is None:
        from .tools import execute_tool

        _execute_tool = execute_tool
    return _execute_tool


async def with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float,
    error_message: str = "Operation timed out",
) -> T:
    """Execute coroutine with timeout and custom error message.

    Args:
        coro: Coroutine to execute
        timeout_seconds: Timeout in seconds
        error_message: Custom error message for timeout

    Returns:
        Result of the coroutine

    Raises:
        MCPError: If operation times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise MCPError(error_message)


def format_error_response(
    error: Exception, include_traceback: bool = False
) -> dict[str, Any]:
    """Format exception as standardized error response.

    Args:
        error: Exception to format
        include_traceback: Whether to include full traceback (debug mode)

    Returns:
        Formatted error dict with type and message
    """
    error_type = type(error).__name__
    error_message = str(error)

    response = {
        "error": {
            "type": error_type,
            "message": error_message,
        }
    }

    if include_traceback:
        import traceback

        response["error"]["traceback"] = traceback.format_exc()

    return response


def validate_search_parameters(
    page_size: int | None = None,
    offset: int | None = None,
    max_tokens: int | None = None,
) -> tuple[int, int, int]:
    """Validate and constrain search parameters to acceptable ranges.

    Args:
        page_size: Requested page size (1-100)
        offset: Requested offset (>= 0)
        max_tokens: Requested max response tokens (1000-25000)

    Returns:
        Tuple of (page_size, offset, max_tokens) with validated values
    """
    # Apply constraints with defaults
    validated_page_size = max(1, min(page_size or 10, 100))
    validated_offset = max(0, offset or 0)
    validated_tokens = max(1000, min(max_tokens or 20000, 25000))

    return validated_page_size, validated_offset, validated_tokens


async def handle_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    services: DatabaseServices,
    embedding_manager: EmbeddingManager | None,
    initialization_complete: asyncio.Event,
    debug_mode: bool = False,
    scan_progress: dict | None = None,
    llm_manager: LLMManager | None = None,
    client_context: dict[str, Any] | None = None,
    project_registry: Any = None,
) -> list[types.TextContent] | dict[str, Any]:
    """Unified tool call handler for all MCP servers (MCP 2025-11-25 compliant).

    Single entry point for all tool executions across transports.
    Handles initialization, validation, execution, and formatting.

    Return types per MCP 2025-11-25:
    - Tools with outputSchema: Returns dict (SDK sets structuredContent + content)
    - Tools without outputSchema: Returns list[TextContent] (unstructured)

    Args:
        tool_name: Name of the tool to execute from TOOL_REGISTRY
        arguments: Tool arguments as key-value pairs
        services: Database services bundle for tool execution
        embedding_manager: Optional embedding manager for semantic search
        initialization_complete: Event to wait for server initialization
        debug_mode: Whether to include stack traces in error responses
        scan_progress: Optional scan progress from MCPServerBase
        llm_manager: Optional LLM manager for code_research
        client_context: Optional client context (project path from HTTP header)
        project_registry: Optional ProjectRegistry for multi-project search

    Returns:
        dict for structured tools (SDK auto-sets structuredContent),
        list[TextContent] for unstructured tools

    Raises:
        MCPError: On tool execution failure (caught and formatted as error response)
    """
    try:
        # Lazy import at runtime to construct MCP content objects without
        # forcing hard dependency during module import/collection.
        import mcp.types as types  # noqa: WPS433

        # Wait for init (reduced timeout since server is immediately available)
        await asyncio.wait_for(initialization_complete.wait(), timeout=5.0)

        # Lazy load to avoid circular import
        tool_registry = _get_tool_registry()
        execute_tool_fn = _get_execute_tool()

        # Validate tool exists
        if tool_name not in tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Check embedding requirements
        tool = tool_registry[tool_name]
        if tool.requires_embeddings and not embedding_manager:
            raise ValueError(f"Tool {tool_name} requires embedding provider")

        # Parse arguments (handles both string and typed values)
        parsed_args = parse_mcp_arguments(arguments)

        # Execute tool
        result = await execute_tool_fn(
            tool_name=tool_name,
            services=services,
            embedding_manager=embedding_manager,
            arguments=parsed_args,
            scan_progress=scan_progress,
            llm_manager=llm_manager,
            client_context=client_context,
            project_registry=project_registry,
        )

        # Format response based on tool type (MCP 2025-11-25 structured content)
        # - Tools with outputSchema: Return dict directly (SDK sets structuredContent)
        # - Tools without outputSchema: Return list[TextContent] (unstructured)
        if isinstance(result, str):
            # Raw string response (e.g., code_research markdown) - unstructured
            return [types.TextContent(type="text", text=result)]
        elif tool.output_schema is not None:
            # Tool has outputSchema - return dict for structured content
            # SDK will set structuredContent and serialize to content automatically
            return result
        else:
            # Dict result but no outputSchema - serialize as unstructured content
            response_text = format_tool_response(result, format_type="json")
            return [types.TextContent(type="text", text=response_text)]

    except Exception as e:
        error_response = format_error_response(e, include_traceback=debug_mode)
        return [types.TextContent(type="text", text=json.dumps(error_response))]


def format_json_response(data: Any) -> str:
    """Format data as JSON string for stdio protocol.

    Args:
        data: Data to format

    Returns:
        JSON string representation
    """
    return json.dumps(data, default=str, ensure_ascii=False)


def format_tool_response(result: Any, format_type: str = "dict") -> Any:
    """Format tool result for MCP protocol.

    Args:
        result: Tool execution result
        format_type: "json" for stdio protocol, "dict" for direct dict response

    Returns:
        Formatted result
    """
    if format_type == "json":
        return format_json_response(result)
    elif format_type == "dict":
        # Ensure it's a proper dict (not a TypedDict)
        return dict(result) if hasattr(result, "__dict__") else result
    else:
        return result


def parse_mcp_arguments(args: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate MCP tool arguments.

    Handles common argument patterns and provides defaults.

    Args:
        args: Raw arguments from MCP request

    Returns:
        Parsed and validated arguments
    """
    # Create a copy to avoid modifying original
    parsed = args.copy()

    # Handle common search parameters
    if "page_size" in parsed:
        if not isinstance(parsed["page_size"], int):
            parsed["page_size"] = int(parsed["page_size"])
    if "offset" in parsed:
        if not isinstance(parsed["offset"], int):
            parsed["offset"] = int(parsed["offset"])
    if "max_response_tokens" in parsed:
        if not isinstance(parsed["max_response_tokens"], int):
            parsed["max_response_tokens"] = int(parsed["max_response_tokens"])
    if "threshold" in parsed and parsed["threshold"] is not None:
        if not isinstance(parsed["threshold"], float):
            parsed["threshold"] = float(parsed["threshold"])

    return parsed


def add_common_mcp_arguments(parser: Any) -> None:
    """Add common MCP server arguments to a parser.

    This function adds all the configuration arguments that the
    stdio MCP server supports.

    Args:
        parser: ArgumentParser to add arguments to
    """
    # Positional path argument
    from pathlib import Path

    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory path to index (default: current directory)",
    )

    # Config file argument
    parser.add_argument("--config", type=str, help="Path to configuration file")

    # Database arguments
    parser.add_argument("--db", type=str, help="Database path")
    parser.add_argument(
        "--database-provider", choices=["duckdb", "lancedb"], help="Database provider"
    )

    # Embedding arguments
    parser.add_argument(
        "--provider",
        choices=["openai"],
        help="Embedding provider",
    )
    parser.add_argument("--model", type=str, help="Embedding model")
    parser.add_argument("--api-key", type=str, help="API key for embedding provider")
    parser.add_argument("--base-url", type=str, help="Base URL for embedding provider")

    # Debug flag
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

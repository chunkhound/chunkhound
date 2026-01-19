"""Common utilities and error handling for MCP server.

This module provides shared utilities for the stdio MCP server,
including error handling, response formatting, and validation helpers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:  # type-checkers only; avoid runtime hard dep
    import mcp.types as types  # noqa: F401

from .tools import TOOL_REGISTRY, execute_tool

if TYPE_CHECKING:
    from chunkhound.database_factory import DatabaseServices
    from chunkhound.embeddings import EmbeddingManager
    from chunkhound.llm_manager import LLMManager

T = TypeVar("T")


def has_reranker_support(embedding_manager: EmbeddingManager | None) -> bool:
    """Check if embedding manager has reranker support available.

    Args:
        embedding_manager: Embedding manager to check

    Returns:
        True if reranker support is available, False otherwise
    """
    if not embedding_manager:
        return False
    try:
        provider = embedding_manager.get_provider()
        return hasattr(provider, "supports_reranking") and provider.supports_reranking()
    except Exception:
        return False


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
    config: Any = None,
) -> list[types.TextContent]:
    """Unified tool call handler for all MCP servers.

    Single entry point for all tool executions across transports.
    Handles initialization, validation, execution, and formatting.

    Args:
        tool_name: Name of the tool to execute from TOOL_REGISTRY
        arguments: Tool arguments as key-value pairs
        services: Database services bundle for tool execution
        embedding_manager: Optional embedding manager for semantic search
        initialization_complete: Event to wait for server initialization
        debug_mode: Whether to include stack traces in error responses
        scan_progress: Optional scan progress from MCPServerBase
        llm_manager: Optional LLM manager for code_research
        config: Optional Config instance for research service factory

    Returns:
        List containing a single TextContent with JSON-formatted response

    Raises:
        MCPError: On tool execution failure (caught and formatted as error response)
    """
    try:
        # Lazy import at runtime to construct MCP content objects without
        # forcing hard dependency during module import/collection.
        import mcp.types as types  # noqa: WPS433
        # Wait for initialization (reduced timeout, server is ready)
        await asyncio.wait_for(initialization_complete.wait(), timeout=5.0)

        # Validate tool exists
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Check capability requirements
        tool = TOOL_REGISTRY[tool_name]
        if tool.requires_embeddings and (
            not embedding_manager or not embedding_manager.list_providers()
        ):
            raise ValueError(f"Tool {tool_name} requires embedding provider")
        if tool.requires_llm and not llm_manager:
            raise ValueError(f"Tool {tool_name} requires LLM provider")
        if tool.requires_reranker and not has_reranker_support(embedding_manager):
            raise ValueError(f"Tool {tool_name} requires reranker support")

        # Parse arguments (handles both string and typed values)
        parsed_args = parse_mcp_arguments(arguments)

        # Execute tool
        result = await execute_tool(
            tool_name=tool_name,
            services=services,
            embedding_manager=embedding_manager,
            arguments=parsed_args,
            scan_progress=scan_progress,
            llm_manager=llm_manager,
            config=config,
        )

        # Format response based on result type
        # - code_research tool returns raw markdown string (for rich formatting)
        # - All other tools return dicts (formatted as JSON)
        if isinstance(result, str):
            # Raw string response - pass through directly
            response_text = result
        else:
            # Dict response - format as JSON for MCP protocol
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

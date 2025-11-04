"""Declarative tool registry for MCP servers.

This module defines all MCP tools in a single location, allowing both
stdio and HTTP servers to use the same tool implementations with their
protocol-specific wrappers.

The registry pattern eliminates duplication and ensures consistent behavior
across server types.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict, cast

try:
    from typing import NotRequired  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    from typing_extensions import NotRequired

from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.deep_research_service import DeepResearchService
from chunkhound.version import __version__

# Response size limits (tokens)
MAX_RESPONSE_TOKENS = 20000
MIN_RESPONSE_TOKENS = 1000
MAX_ALLOWED_TOKENS = 25000


def _convert_paths_to_native(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert file paths in search results to native platform format."""
    from pathlib import Path

    for result in results:
        if "file_path" in result and result["file_path"]:
            # Use Path for proper native conversion
            result["file_path"] = str(Path(result["file_path"]))
    return results


# Type definitions for return values
class PaginationInfo(TypedDict):
    """Pagination metadata for search results."""

    offset: int
    page_size: int
    has_more: bool
    total: NotRequired[int | None]
    next_offset: NotRequired[int | None]


class SearchResponse(TypedDict):
    """Response structure for search operations."""

    results: list[dict[str, Any]]
    pagination: PaginationInfo


class HealthStatus(TypedDict):
    """Health check response structure."""

    status: str
    version: str
    database_connected: bool
    embedding_providers: list[str]


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple heuristic (3 chars ≈ 1 token for safety)."""
    return len(text) // 3


def limit_response_size(
    response_data: SearchResponse, max_tokens: int = MAX_RESPONSE_TOKENS
) -> SearchResponse:
    """Limit response size to fit within token limits by reducing results."""
    if not response_data.get("results"):
        return response_data

    # Start with full response and iteratively reduce until under limit
    limited_results = response_data["results"][:]

    while limited_results:
        # Create test response with current results
        test_response = {
            "results": limited_results,
            "pagination": response_data["pagination"],
        }

        # Estimate token count
        response_text = json.dumps(test_response, default=str)
        token_count = estimate_tokens(response_text)

        if token_count <= max_tokens:
            # Update pagination to reflect actual returned results
            actual_count = len(limited_results)
            updated_pagination = response_data["pagination"].copy()
            updated_pagination["page_size"] = actual_count
            updated_pagination["has_more"] = updated_pagination.get(
                "has_more", False
            ) or actual_count < len(response_data["results"])
            if actual_count < len(response_data["results"]):
                updated_pagination["next_offset"] = (
                    updated_pagination.get("offset", 0) + actual_count
                )

            return {"results": limited_results, "pagination": updated_pagination}

        # Remove results from the end to reduce size
        # Remove in chunks for efficiency
        reduction_size = max(1, len(limited_results) // 4)
        limited_results = limited_results[:-reduction_size]

    # If even empty results exceed token limit, return minimal response
    return {
        "results": [],
        "pagination": {
            "offset": response_data["pagination"].get("offset", 0),
            "page_size": 0,
            "has_more": len(response_data["results"]) > 0,
            "total": response_data["pagination"].get("total", 0),
            "next_offset": None,
        },
    }


async def search_regex_impl(
    services: DatabaseServices,
    pattern: str,
    page_size: int = 10,
    offset: int = 0,
    path_filter: str | None = None,
) -> SearchResponse:
    """Core regex search implementation.

    Args:
        services: Database services bundle
        pattern: Regex pattern to search for
        page_size: Number of results per page (1-100)
        offset: Starting offset for pagination
        path_filter: Optional path filter

    Returns:
        Dict with 'results' and 'pagination' keys
    """
    # Validate and constrain parameters
    page_size = max(1, min(page_size, 100))
    offset = max(0, offset)

    # Check database connection
    if services and not services.provider.is_connected:
        services.provider.connect()

    # Perform search using SearchService
    results, pagination = services.search_service.search_regex(
        pattern=pattern,
        page_size=page_size,
        offset=offset,
        path_filter=path_filter,
    )

    # Convert file paths to native platform format
    native_results = _convert_paths_to_native(results)

    return cast(SearchResponse, {"results": native_results, "pagination": pagination})


async def search_semantic_impl(
    services: DatabaseServices,
    embedding_manager: EmbeddingManager,
    query: str,
    page_size: int = 10,
    offset: int = 0,
    provider: str | None = None,
    model: str | None = None,
    threshold: float | None = None,
    path_filter: str | None = None,
) -> SearchResponse:
    """Core semantic search implementation.

    Args:
        services: Database services bundle
        embedding_manager: Embedding manager instance
        query: Search query text
        page_size: Number of results per page (1-100)
        offset: Starting offset for pagination
        provider: Embedding provider name (optional)
        model: Embedding model name (optional)
        threshold: Distance threshold for filtering (optional)
        path_filter: Optional path filter

    Returns:
        Dict with 'results' and 'pagination' keys

    Raises:
        Exception: If no embedding providers available or configured
        asyncio.TimeoutError: If embedding request times out
    """
    # Validate embedding manager and providers
    if not embedding_manager or not embedding_manager.list_providers():
        raise Exception(
            "No embedding providers available. Configure an embedding provider via:\n"
            "1. Create .chunkhound.json with embedding configuration, OR\n"
            "2. Set CHUNKHOUND_EMBEDDING__API_KEY environment variable"
        )

    # Use explicit provider/model from arguments, otherwise get from configured provider
    if not provider or not model:
        try:
            default_provider_obj = embedding_manager.get_provider()
            if not provider:
                provider = default_provider_obj.name
            if not model:
                model = default_provider_obj.model
        except ValueError:
            raise Exception(
                "No default embedding provider configured. "
                "Either specify provider and model explicitly, or configure a default provider."
            )

    # Validate and constrain parameters
    page_size = max(1, min(page_size, 100))
    offset = max(0, offset)

    # Check database connection
    if services and not services.provider.is_connected:
        services.provider.connect()

    # Perform search using SearchService
    results, pagination = await services.search_service.search_semantic(
        query=query,
        page_size=page_size,
        offset=offset,
        threshold=threshold,
        provider=provider,
        model=model,
        path_filter=path_filter,
    )

    # Convert file paths to native platform format
    native_results = _convert_paths_to_native(results)

    return cast(SearchResponse, {"results": native_results, "pagination": pagination})


async def get_stats_impl(
    services: DatabaseServices, scan_progress: dict | None = None
) -> dict[str, Any]:
    """Core stats implementation with scan progress.

    Args:
        services: Database services bundle
        scan_progress: Optional scan progress from MCPServerBase

    Returns:
        Dict with database statistics and scan progress
    """
    stats: dict[str, Any] = services.provider.get_stats()

    # Map provider field names to MCP API field names
    result = {
        "total_files": stats.get("files", 0),
        "total_chunks": stats.get("chunks", 0),
        "total_embeddings": stats.get("embeddings", 0),
        "database_size_mb": stats.get("size_mb", 0),
        "total_providers": stats.get("providers", 0),
    }

    # Add scan progress if available
    if scan_progress:
        result["initial_scan"] = {
            "is_scanning": scan_progress.get("is_scanning", False),
            "files_processed": scan_progress.get("files_processed", 0),
            "chunks_created": scan_progress.get("chunks_created", 0),
            "started_at": scan_progress.get("scan_started_at"),
            "completed_at": scan_progress.get("scan_completed_at"),
            "error": scan_progress.get("scan_error"),
        }

    return result


async def health_check_impl(
    services: DatabaseServices, embedding_manager: EmbeddingManager
) -> HealthStatus:
    """Core health check implementation.

    Args:
        services: Database services bundle
        embedding_manager: Embedding manager instance

    Returns:
        Dict with health status information
    """
    health_status = {
        "status": "healthy",
        "version": __version__,
        "database_connected": services is not None and services.provider.is_connected,
        "embedding_providers": embedding_manager.list_providers()
        if embedding_manager
        else [],
    }

    return cast(HealthStatus, health_status)


async def deep_research_impl(
    services: DatabaseServices,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager,
    query: str,
    depth: str | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Core deep research implementation.

    Args:
        services: Database services bundle
        embedding_manager: Embedding manager instance
        llm_manager: LLM manager instance
        query: Research query
        progress: Optional Rich Progress instance for terminal UI (None for MCP)

    Returns:
        Dict with answer and metadata

    Raises:
        Exception: If LLM or reranker not configured
    """
    # Validate LLM is configured
    if not llm_manager or not llm_manager.is_configured():
        raise Exception(
            "LLM not configured. Configure an LLM provider via:\n"
            "1. Create .chunkhound.json with llm configuration, OR\n"
            "2. Set CHUNKHOUND_LLM_API_KEY environment variable"
        )

    # Validate reranker is configured
    if not embedding_manager or not embedding_manager.list_providers():
        raise Exception(
            "No embedding providers available. Code research requires reranking support."
        )

    embedding_provider = embedding_manager.get_provider()
    if not (
        hasattr(embedding_provider, "supports_reranking")
        and embedding_provider.supports_reranking()
    ):
        raise Exception(
            "Code research requires a provider with reranking support. "
            "Configure a rerank_model in your embedding configuration."
        )

    # Create code research service with dynamic tool name
    # This ensures followup suggestions automatically update if tool is renamed
    research_service = DeepResearchService(
        database_services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        tool_name="code_research",  # Matches tool registration below
        progress=progress,  # Pass progress for terminal UI (None in MCP mode)
    )

    # Perform code research with fixed depth and dynamic budgets
    result = await research_service.deep_research(query)

    return result


@dataclass
class Tool:
    """Tool definition with metadata and implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    implementation: Callable
    requires_embeddings: bool = False


# Define all tools declaratively
TOOL_DEFINITIONS = [
    Tool(
        name="get_stats",
        description="Get database statistics including file, chunk, and embedding counts",
        parameters={
            "properties": {},
            "type": "object",
        },
        implementation=get_stats_impl,
        requires_embeddings=False,
    ),
    Tool(
        name="health_check",
        description="Check server health status",
        parameters={
            "properties": {},
            "type": "object",
        },
        implementation=health_check_impl,
        requires_embeddings=False,
    ),
    Tool(
        name="search_regex",
        description="Find exact code patterns using regular expressions. Use when searching for specific syntax (function definitions, variable names, import statements), exact text matches, or code structure patterns. Best for precise searches where you know the exact pattern.",
        parameters={
            "properties": {
                "pattern": {
                    "description": "Regular expression pattern to search for",
                    "type": "string",
                },
                "page_size": {
                    "default": 10,
                    "description": "Number of results per page (1-100)",
                    "type": "integer",
                },
                "offset": {
                    "default": 0,
                    "description": "Starting position for pagination",
                    "type": "integer",
                },
                "max_response_tokens": {
                    "default": 20000,
                    "description": "Maximum response size in tokens (1000-25000)",
                    "type": "integer",
                },
                "path": {
                    "description": "Optional relative path to limit search scope (e.g., 'src/', 'tests/')",
                    "type": "string",
                },
            },
            "required": ["pattern"],
            "type": "object",
        },
        implementation=search_regex_impl,
        requires_embeddings=False,
    ),
    Tool(
        name="search_semantic",
        description="Find code by meaning and concept rather than exact syntax. Use when searching by description (e.g., 'authentication logic', 'error handling'), looking for similar functionality, or when you're unsure of exact keywords. Understands intent and context beyond literal text matching.",
        parameters={
            "properties": {
                "query": {
                    "description": "Natural language search query",
                    "type": "string",
                },
                "page_size": {
                    "default": 10,
                    "description": "Number of results per page (1-100)",
                    "type": "integer",
                },
                "offset": {
                    "default": 0,
                    "description": "Starting position for pagination",
                    "type": "integer",
                },
                "max_response_tokens": {
                    "default": 20000,
                    "description": "Maximum response size in tokens (1000-25000)",
                    "type": "integer",
                },
                "path": {
                    "description": "Optional relative path to limit search scope (e.g., 'src/', 'tests/')",
                    "type": "string",
                },
                "provider": {
                    "default": "openai",
                    "description": "Embedding provider to use",
                    "type": "string",
                },
                "model": {
                    "default": "text-embedding-3-small",
                    "description": "Embedding model to use",
                    "type": "string",
                },
                "threshold": {
                    "description": "Distance threshold for filtering results (optional)",
                    "type": "number",
                },
            },
            "required": ["query"],
            "type": "object",
        },
        implementation=search_semantic_impl,
        requires_embeddings=True,
    ),
    Tool(
        name="code_research",
        description="Perform deep code research to answer complex questions about your codebase. Use this tool when you need to understand architecture, discover existing implementations, trace relationships between components, or find patterns across multiple files. Returns comprehensive markdown analysis. Synthesis budgets scale automatically based on repository size.",
        parameters={
            "properties": {
                "query": {
                    "description": "Research query to investigate",
                    "type": "string",
                },
                # Optional traversal depth hint for tests and future UI controls
                "depth": {
                    "description": "Traversal depth hint (e.g., 'shallow' or 'deep'). Optional.",
                    "type": "string",
                },
            },
            "required": ["query"],
            "type": "object",
        },
        implementation=deep_research_impl,
        requires_embeddings=True,
    ),
]

# Create registry as a dict for easy lookup
TOOL_REGISTRY: dict[str, Tool] = {tool.name: tool for tool in TOOL_DEFINITIONS}


async def execute_tool(
    tool_name: str,
    services: Any,
    embedding_manager: Any,
    arguments: dict[str, Any],
    scan_progress: dict | None = None,
    llm_manager: Any = None,
) -> dict[str, Any]:
    """Execute a tool from the registry with proper argument handling.

    Args:
        tool_name: Name of the tool to execute
        services: DatabaseServices instance
        embedding_manager: EmbeddingManager instance
        arguments: Tool arguments from the request
        scan_progress: Optional scan progress from MCPServerBase
        llm_manager: Optional LLMManager instance for deep_research

    Returns:
        Tool execution result

    Raises:
        ValueError: If tool not found in registry
        Exception: If tool execution fails
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool = TOOL_REGISTRY[tool_name]

    # Extract implementation-specific arguments
    if tool_name == "get_stats":
        result = await tool.implementation(services, scan_progress)
        return dict(result)

    elif tool_name == "health_check":
        result = await tool.implementation(services, embedding_manager)
        return dict(result)

    elif tool_name == "search_regex":
        # Apply response size limiting
        result = await tool.implementation(
            services=services,
            pattern=arguments["pattern"],
            page_size=arguments.get("page_size", 10),
            offset=arguments.get("offset", 0),
            path_filter=arguments.get("path"),
        )
        max_tokens = arguments.get("max_response_tokens", 20000)
        return dict(limit_response_size(result, max_tokens))

    elif tool_name == "search_semantic":
        # Apply response size limiting
        result = await tool.implementation(
            services=services,
            embedding_manager=embedding_manager,
            query=arguments["query"],
            page_size=arguments.get("page_size", 10),
            offset=arguments.get("offset", 0),
            provider=arguments.get("provider"),
            model=arguments.get("model"),
            threshold=arguments.get("threshold"),
            path_filter=arguments.get("path"),
        )
        max_tokens = arguments.get("max_response_tokens", 20000)
        return dict(limit_response_size(result, max_tokens))

    elif tool_name == "code_research":
        # Code research - return raw markdown directly (not wrapped in JSON)
        # Forward optional depth hint if provided
        depth = arguments.get("depth")
        if depth is not None:
            result = await tool.implementation(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=arguments["query"],
                depth=depth,
            )
        else:
            result = await tool.implementation(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=arguments["query"],
            )
        # Return raw markdown string
        return result.get("answer", f"Research incomplete: Unable to analyze '{arguments['query']}'. Try a more specific query or check that relevant code exists.")

    else:
        raise ValueError(f"Tool {tool_name} not implemented in execute_tool")

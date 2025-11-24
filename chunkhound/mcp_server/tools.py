"""Declarative tool registry for MCP servers.

This module defines all MCP tools in a single location, allowing both
stdio and HTTP servers to use the same tool implementations with their
protocol-specific wrappers.

The registry pattern eliminates duplication and ensures consistent behavior
across server types.
"""

import inspect
import json
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict, Union, cast, get_args, get_origin

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


# =============================================================================
# Schema Generation Infrastructure
# =============================================================================
# These utilities generate JSON Schema from Python function signatures,
# enabling a single source of truth for tool definitions.


@dataclass
class Tool:
    """Tool definition with metadata and implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    implementation: Callable
    requires_embeddings: bool = False


# Tool registry - populated by @register_tool decorator
TOOL_REGISTRY: dict[str, Tool] = {}


def _python_type_to_json_schema_type(type_hint: Any) -> dict[str, Any]:
    """Convert Python type hint to JSON Schema type definition.

    Args:
        type_hint: Python type annotation

    Returns:
        JSON Schema type definition dict
    """
    # Handle None / NoneType
    if type_hint is None or type_hint is type(None):
        return {"type": "null"}

    # Get origin for generic types (list, dict, Union, etc.)
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle Union types (including Optional which is Union[T, None])
    # Note: Python 3.10+ uses types.UnionType for X | Y syntax
    if origin is Union or isinstance(type_hint, types.UnionType):
        # Filter out NoneType to find the actual type
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            # Optional[T] case - just return the T's schema
            return _python_type_to_json_schema_type(non_none_types[0])
        else:
            # Multiple non-None types - use anyOf
            return {"anyOf": [_python_type_to_json_schema_type(t) for t in non_none_types]}

    # Handle basic types
    if type_hint == str or type_hint is str:
        return {"type": "string"}
    elif type_hint == int or type_hint is int:
        return {"type": "integer"}
    elif type_hint == float or type_hint is float:
        return {"type": "number"}
    elif type_hint == bool or type_hint is bool:
        return {"type": "boolean"}
    elif origin is list:
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _python_type_to_json_schema_type(item_type)
        }
    elif origin is dict:
        return {"type": "object"}
    else:
        # Default to object for complex types
        return {"type": "object"}


def _extract_param_descriptions_from_docstring(func: Callable) -> dict[str, str]:
    """Extract parameter descriptions from function docstring.

    Parses Google-style docstring Args section.

    Args:
        func: Function with docstring

    Returns:
        Dict mapping parameter names to their descriptions
    """
    if not func.__doc__:
        return {}

    descriptions: dict[str, str] = {}
    lines = func.__doc__.split('\n')
    in_args_section = False

    for line in lines:
        stripped = line.strip()

        # Detect Args section
        if stripped == "Args:":
            in_args_section = True
            continue

        # Exit Args section when we hit another section or empty line after args
        if in_args_section and (stripped.endswith(':') or (not stripped and descriptions)):
            in_args_section = False

        # Parse parameter descriptions
        if in_args_section and ':' in stripped:
            # Format: "param_name: description"
            parts = stripped.split(':', 1)
            if len(parts) == 2:
                param_name = parts[0].strip()
                description = parts[1].strip()
                descriptions[param_name] = description

    return descriptions


def _generate_json_schema_from_signature(func: Callable) -> dict[str, Any]:
    """Generate JSON Schema from function signature.

    Args:
        func: Function to analyze

    Returns:
        JSON Schema parameters dict compatible with MCP tool schema
    """
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    # Extract parameter descriptions from docstring
    param_descriptions = _extract_param_descriptions_from_docstring(func)

    for param_name, param in sig.parameters.items():
        # Skip service/infrastructure parameters that aren't part of the tool API
        if param_name in ('services', 'embedding_manager', 'llm_manager', 'scan_progress', 'progress'):
            continue

        # Get type hint
        type_hint = param.annotation if param.annotation != inspect.Parameter.empty else Any

        # Convert to JSON Schema type
        schema = _python_type_to_json_schema_type(type_hint)

        # Add description if available from docstring
        if param_name in param_descriptions:
            schema["description"] = param_descriptions[param_name]

        # Add default value if present
        if param.default != inspect.Parameter.empty and param.default is not None:
            schema["default"] = param.default

        properties[param_name] = schema

        # Mark as required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required if required else [],
    }


def register_tool(
    description: str,
    requires_embeddings: bool = False,
    name: str | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator to register a function as an MCP tool.

    Extracts JSON Schema from function signature and registers in TOOL_REGISTRY.

    Args:
        description: Comprehensive tool description for LLM users
        requires_embeddings: Whether tool requires embedding providers
        name: Optional tool name (defaults to function name)

    Returns:
        Decorator function

    Example:
        @register_tool(
            description="Search using regex patterns",
            requires_embeddings=False
        )
        async def search_regex(pattern: str, page_size: int = 10) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__

        # Generate schema from function signature
        parameters = _generate_json_schema_from_signature(func)

        # Register tool in global registry
        TOOL_REGISTRY[tool_name] = Tool(
            name=tool_name,
            description=description,
            parameters=parameters,
            implementation=func,
            requires_embeddings=requires_embeddings,
        )

        return func

    return decorator


# =============================================================================
# Helper Functions
# =============================================================================


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


@register_tool(
    description="Find exact code patterns using regular expressions. Use when searching for specific syntax (function definitions, variable names, import statements), exact text matches, or code structure patterns. Best for precise searches where you know the exact pattern.",
    requires_embeddings=False,
    name="search_regex",
)
async def search_regex_impl(
    services: DatabaseServices,
    pattern: str,
    page_size: int = 10,
    offset: int = 0,
    max_response_tokens: int = 20000,
    path: str | None = None,
) -> SearchResponse:
    """Core regex search implementation.

    Args:
        services: Database services bundle
        pattern: Regex pattern to search for
        page_size: Number of results per page (1-100)
        offset: Starting offset for pagination
        max_response_tokens: Maximum response size in tokens (1000-25000)
        path: Optional path to limit search scope

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
        path_filter=path,
    )

    # Convert file paths to native platform format
    native_results = _convert_paths_to_native(results)

    # Apply response size limiting
    response = cast(SearchResponse, {"results": native_results, "pagination": pagination})
    return limit_response_size(response, max_response_tokens)


@register_tool(
    description="Find code by meaning and concept rather than exact syntax. Use when searching by description (e.g., 'authentication logic', 'error handling'), looking for similar functionality, or when you're unsure of exact keywords. Understands intent and context beyond literal text matching.",
    requires_embeddings=True,
    name="search_semantic",
)
async def search_semantic_impl(
    services: DatabaseServices,
    embedding_manager: EmbeddingManager,
    query: str,
    page_size: int = 10,
    offset: int = 0,
    max_response_tokens: int = 20000,
    path: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    threshold: float | None = None,
) -> SearchResponse:
    """Core semantic search implementation.

    Args:
        services: Database services bundle
        embedding_manager: Embedding manager instance
        query: Search query text
        page_size: Number of results per page (1-100)
        offset: Starting offset for pagination
        max_response_tokens: Maximum response size in tokens (1000-25000)
        path: Optional path to limit search scope
        provider: Embedding provider name (optional, uses configured provider if not specified)
        model: Embedding model name (optional, uses configured model if not specified)
        threshold: Distance threshold for filtering (optional)

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
        path_filter=path,
    )

    # Convert file paths to native platform format
    native_results = _convert_paths_to_native(results)

    # Apply response size limiting
    response = cast(SearchResponse, {"results": native_results, "pagination": pagination})
    return limit_response_size(response, max_response_tokens)


@register_tool(
    description="Get database statistics including file, chunk, and embedding counts",
    requires_embeddings=False,
    name="get_stats",
)
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
    # Ensure DB connection for stats in lazy-connect scenarios
    try:
        if services and not services.provider.is_connected:
            services.provider.connect()
    except Exception:
        # Best-effort: if connect fails, get_stats may still work for providers that lazy-init internally
        pass
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


@register_tool(
    description="Check server health status",
    requires_embeddings=False,
    name="health_check",
)
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


@register_tool(
    description="Perform deep code research to answer complex questions about your codebase. Use this tool when you need to understand architecture, discover existing implementations, trace relationships between components, or find patterns across multiple files. Returns comprehensive markdown analysis. Synthesis budgets scale automatically based on repository size.",
    requires_embeddings=True,
    name="code_research",
)
async def deep_research_impl(
    services: DatabaseServices,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager,
    query: str,
    progress: Any = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Core deep research implementation.

    Args:
        services: Database services bundle
        embedding_manager: Embedding manager instance
        llm_manager: LLM manager instance
        query: Research query
        progress: Optional Rich Progress instance for terminal UI (None for MCP)
        path: Optional relative path to limit research scope
            (e.g., 'tree-sitter-haskell', 'src/')

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
        path_filter=path,
    )

    # Perform code research with fixed depth and dynamic budgets
    result = await research_service.deep_research(query)

    return result


# =============================================================================
# Tool Execution
# =============================================================================


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
        llm_manager: Optional LLMManager instance for code_research

    Returns:
        Tool execution result

    Raises:
        ValueError: If tool not found in registry
        Exception: If tool execution fails
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool = TOOL_REGISTRY[tool_name]

    # Build kwargs by inspecting function signature and mapping available arguments
    sig = inspect.signature(tool.implementation)
    kwargs: dict[str, Any] = {}

    for param_name in sig.parameters.keys():
        # Map infrastructure parameters
        if param_name == "services":
            kwargs["services"] = services
        elif param_name == "embedding_manager":
            kwargs["embedding_manager"] = embedding_manager
        elif param_name == "llm_manager":
            kwargs["llm_manager"] = llm_manager
        elif param_name == "scan_progress":
            kwargs["scan_progress"] = scan_progress
        elif param_name == "progress":
            # Progress parameter for terminal UI (None for MCP mode)
            kwargs["progress"] = None
        elif param_name in arguments:
            # Tool-specific parameter from request
            kwargs[param_name] = arguments[param_name]
        # If parameter not found and has default, it will use the default

    # Execute the tool
    result = await tool.implementation(**kwargs)

    # Handle special return types
    if tool_name == "code_research":
        # Code research returns dict with 'answer' key - return raw markdown string
        if isinstance(result, dict):
            return cast(dict[str, Any], result.get("answer", f"Research incomplete: Unable to analyze '{arguments.get('query', 'unknown')}'. Try a more specific query or check that relevant code exists."))

    # Convert result to dict if it's not already
    if hasattr(result, "__dict__"):
        return dict(result)
    elif isinstance(result, dict):
        return result
    else:
        return {"result": result}

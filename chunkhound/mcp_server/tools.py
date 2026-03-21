"""Declarative tool registry for MCP server.

This module defines all MCP tools in a single location, providing a unified
registry that the stdio server uses for tool definitions.

The registry pattern ensures consistent tool metadata and behavior.
"""

import inspect
import json
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, Union, cast, get_args, get_origin

try:
    from typing import NotRequired  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    from typing_extensions import NotRequired

from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.factory import ResearchServiceFactory

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
    requires_llm: bool = False
    requires_reranker: bool = False


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
            return {
                "anyOf": [_python_type_to_json_schema_type(t) for t in non_none_types]
            }

    # Handle Literal types (e.g., Literal["a", "b"])
    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    # Handle basic types
    if type_hint is str:
        return {"type": "string"}
    elif type_hint is int:
        return {"type": "integer"}
    elif type_hint is float:
        return {"type": "number"}
    elif type_hint is bool:
        return {"type": "boolean"}
    elif origin is list:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_json_schema_type(item_type)}
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
    lines = func.__doc__.split("\n")
    in_args_section = False

    for line in lines:
        stripped = line.strip()

        # Detect Args section
        if stripped == "Args:":
            in_args_section = True
            continue

        # Exit Args section when we hit another section or empty line after args
        if in_args_section and (
            stripped.endswith(":") or (not stripped and descriptions)
        ):
            in_args_section = False

        # Parse parameter descriptions
        if in_args_section and ":" in stripped:
            # Format: "param_name: description"
            parts = stripped.split(":", 1)
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
        if param_name in (
            "services",
            "embedding_manager",
            "llm_manager",
            "scan_progress",
            "progress",
            "config",
        ):
            continue

        # Get type hint
        type_hint = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any
        )

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
    requires_llm: bool = False,
    requires_reranker: bool = False,
    name: str | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator to register a function as an MCP tool.

    Extracts JSON Schema from function signature and registers in TOOL_REGISTRY.

    Args:
        description: Comprehensive tool description for LLM users
        requires_embeddings: Whether tool requires embedding providers
        requires_llm: Whether tool requires LLM provider
        requires_reranker: Whether tool requires reranking support
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
            requires_llm=requires_llm,
            requires_reranker=requires_reranker,
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


# =============================================================================
# Tool Descriptions (optimized for LLM consumption)
# =============================================================================

SEARCH_DESCRIPTION = """Search code by exact pattern (regex) or meaning (semantic).

TYPE SELECTION:
- regex: Exact pattern matching. Use for function names, variable names,
  import statements, or known string patterns.
  Example queries: "def authenticate", "import.*pandas", "TODO:.*fix"

- semantic: Meaning-based search. Use when describing functionality
  conceptually or unsure of exact keywords.
  Example queries: "authentication logic", "error handling for database"

WHEN TO USE: Quick lookup, finding references, exploring unfamiliar code.
DO NOT USE: Multi-file architecture questions (use code_research instead).

OUTPUT: {results: [{file_path, content, start_line, end_line}], pagination}
COST: Fast, cheap - use liberally."""

CODE_RESEARCH_DESCRIPTION = """Deep analysis for architecture and cross-file code.

USE FOR:
- Understanding how systems/features are implemented across files
- Discovering component relationships and dependencies
- Getting architectural explanations with code citations

DO NOT USE:
- Looking for specific code locations (use search instead)
- Simple pattern matching (use search with type="regex")
- You already know where the code is (read files directly)

OUTPUT: Comprehensive markdown with architecture overview, key locations, relationships.
COST: Expensive (LLM synthesis). 10-60s latency. One call often replaces 5-10 searches.

ERROR RECOVERY: If incomplete, try narrower query or use path parameter to scope."""


# =============================================================================
# Tool Implementations
# =============================================================================


@register_tool(
    description=SEARCH_DESCRIPTION,
    requires_embeddings=False,
    name="search",
)
async def search_impl(
    services: DatabaseServices,
    embedding_manager: EmbeddingManager | None,
    type: Literal["regex", "semantic"],
    query: str,
    path: str | None = None,
    page_size: int = 10,
    offset: int = 0,
) -> SearchResponse:
    """Unified search dispatching to regex or semantic based on type.

    Args:
        services: Database services bundle
        embedding_manager: Embedding manager (required for semantic type)
        type: "semantic" for meaning-based, "regex" for exact pattern
        query: Search query (natural language for semantic, regex pattern for regex)
        path: Optional path to limit search scope (e.g., "src/auth/")
        page_size: Number of results per page (1-100)
        offset: Starting offset for pagination

    Returns:
        Dict with 'results' and 'pagination' keys

    Raises:
        ValueError: If type is invalid or semantic search lacks embedding provider
    """
    # Validate type parameter
    if type not in ("semantic", "regex"):
        raise ValueError(
            f"Invalid search type: '{type}'. Must be 'semantic' or 'regex'."
        )

    # Validate and constrain parameters
    page_size = max(1, min(page_size, 100))
    offset = max(0, offset)

    # Check database connection
    if services and not services.provider.is_connected:
        services.provider.connect()

    if type == "semantic":
        # Validate embedding manager for semantic search
        if not embedding_manager or not embedding_manager.list_providers():
            raise ValueError(
                "Semantic search requires embedding provider. "
                "Configure via .chunkhound.json or CHUNKHOUND_EMBEDDING__API_KEY. "
                "Use type='regex' for pattern-based search without embeddings."
            )

        # Get default provider/model
        try:
            provider_obj = embedding_manager.get_provider()
            provider_name = provider_obj.name
            model_name = provider_obj.model
        except ValueError:
            raise ValueError("No default embedding provider configured.")

        # Perform semantic search
        results, pagination = await services.search_service.search_semantic(
            query=query,
            page_size=page_size,
            offset=offset,
            provider=provider_name,
            model=model_name,
            path_filter=path,
        )
    else:  # regex
        # Perform regex search
        results, pagination = await services.search_service.search_regex_async(
            pattern=query,
            page_size=page_size,
            offset=offset,
            path_filter=path,
        )

    # Convert file paths to native platform format
    native_results = _convert_paths_to_native(results)

    # Apply response size limiting
    response = cast(
        SearchResponse, {"results": native_results, "pagination": pagination}
    )
    return limit_response_size(response)


@register_tool(
    description=CODE_RESEARCH_DESCRIPTION,
    requires_embeddings=True,
    requires_llm=True,
    requires_reranker=True,
    name="code_research",
)
async def deep_research_impl(
    services: DatabaseServices,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager,
    query: str,
    progress: Any = None,
    path: str | None = None,
    config: Config | None = None,
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
        config: Application configuration (optional, defaults to environment config)

    Returns:
        Dict with answer and metadata

    Raises:
        Exception: If LLM or reranker not configured
    """
    # Validate reranker is configured
    if not embedding_manager or not embedding_manager.list_providers():
        raise Exception(
            "No embedding providers available. Code research requires reranking "
            "support."
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

    # Create default config from environment if not provided
    if config is None:
        config = Config.from_environment()

    # Create code research service using factory (v1 or v2 based on config)
    # This ensures followup suggestions automatically update if tool is renamed
    research_service = ResearchServiceFactory.create(
        config=config,
        db_services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        tool_name="code_research",
        progress=progress,
        path_filter=path,
    )

    return await research_service.deep_research(query)


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
    config: Config | None = None,
) -> dict[str, Any] | str:
    """Execute a tool from the registry with proper argument handling.

    Args:
        tool_name: Name of the tool to execute
        services: DatabaseServices instance
        embedding_manager: EmbeddingManager instance
        arguments: Tool arguments from the request
        scan_progress: Optional scan progress from MCPServerBase
        llm_manager: Optional LLMManager instance for code_research
        config: Optional Config instance for research service factory

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
        elif param_name == "config":
            kwargs["config"] = config
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
            query_arg = arguments.get("query", "unknown")
            fallback = (
                "Research incomplete: Unable to analyze "
                f"'{query_arg}'. "
                "Try a more specific query or check that relevant code exists."
            )
            answer = result.get("answer", fallback)
            return str(answer)

    # Convert result to dict if it's not already
    if hasattr(result, "__dict__"):
        return dict(result)
    elif isinstance(result, dict):
        return result
    else:
        return {"result": result}

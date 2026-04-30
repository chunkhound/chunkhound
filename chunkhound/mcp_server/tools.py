"""Declarative tool registry for MCP server.

This module defines all MCP tools in a single location, providing a unified
registry that the stdio server uses for tool definitions.

The registry pattern ensures consistent tool metadata and behavior.
"""

import inspect
import json
import re
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
from chunkhound.mcp_server.status import derive_daemon_status
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


_BACKTICK_RUN_RE = re.compile(r"`+")


def format_search_results_markdown(
    results: list[dict[str, Any]],
    pagination: dict[str, Any],
    search_type: str,
) -> str:
    """Render search results as lean markdown for MCP responses.

    Drops chunk_id, chunk_type, language, metadata, file_extension,
    line_count, code_preview, is_truncated, similarity_percentage.
    Retains file_path, line range, symbol/name, content, and (for semantic
    search) similarity percentage. Appends a pagination footer.
    """
    if not results:
        return "No results found."

    blocks: list[str] = []
    for result in results:
        file_path: str = result.get("file_path") or "unknown"
        start_line: int | None = result.get("start_line")
        end_line: int | None = result.get("end_line")
        content: str = result.get("content") or ""
        symbol: str | None = result.get("symbol") or result.get("name")
        similarity: float | None = result.get("similarity")

        lang = result.get("language") or ""
        lang_hint = "" if lang == "unknown" else lang

        # Heading: ## `path` L10–L20 — Symbol (92%)
        parts: list[str] = [f"## `{file_path}`"]
        if start_line is not None:
            line_range = (
                f"L{start_line}–L{end_line}"
                if end_line is not None and end_line != start_line
                else f"L{start_line}"
            )
            parts.append(line_range)
        if symbol:
            parts.append(f"— {symbol}")
        if search_type == "semantic" and similarity is not None:
            pct = int(round(similarity * 100))
            parts.append(f"({pct}%)")

        heading = " ".join(parts)
        # Use a fence longer than any backtick run in the content (CommonMark §6.1).
        max_run = max((len(m.group()) for m in _BACKTICK_RUN_RE.finditer(content)), default=0)
        fence = "`" * max(3, max_run + 1)
        block = f"{heading}\n\n{fence}{lang_hint}\n{content}\n{fence}"
        blocks.append(block)

    body = "\n\n---\n\n".join(blocks)

    # Pagination footer
    offset: int = pagination.get("offset", 0)
    total: int | None = pagination.get("total")
    has_more: bool = pagination.get("has_more", False)
    next_offset: int | None = pagination.get("next_offset")
    page_size: int = pagination.get("page_size", len(results))

    start_num = offset + 1
    end_num = offset + len(results)

    if total is not None and page_size:
        total_pages = max(1, -(-total // page_size))
        current_page = (offset // page_size) + 1
        footer = f"Page {current_page} of {total_pages} (results {start_num}–{end_num} of {total})"
    else:
        footer = f"Results {start_num}–{end_num}"

    if has_more and next_offset is not None:
        footer += f" | next_offset={next_offset}"

    return f"{body}\n\n---\n{footer}"


# =============================================================================
# Tool Descriptions (optimized for LLM consumption)
# =============================================================================

SEARCH_DESCRIPTION = """Pinpoint specific code locations after building understanding with code_research. Returns structurally-parsed code chunks (functions, classes) — large definitions may span multiple results.

TYPE — choose one:
- **regex**: Match exact patterns against code content. Use for known identifiers, imports, or string literals.
  Examples: "def authenticate", "class.*Handler", "import.*pandas", "TODO:.*refactor"
- **semantic**: Find code by meaning via embedding similarity. Use for concepts or when exact identifiers are unknown.
  Examples: "authentication logic", "retry with exponential backoff", "database connection pooling"

DECISION GUIDE:
- Known symbol or pattern → regex
- Concept or behavior → semantic
- Cross-file architecture question → call code_research first

OUTPUT: Markdown blocks — file path, line range, symbol name, code block, pagination footer."""

SEARCH_DESCRIPTION_NO_RESEARCH = """Pinpoint specific code locations — find exact symbols, patterns, or concepts in the indexed codebase. Returns structurally-parsed code chunks (functions, classes) — large definitions may span multiple results.

TYPE — choose one:
- **regex**: Match exact patterns against code content. Use for known identifiers, imports, or string literals.
  Examples: "def authenticate", "class.*Handler", "import.*pandas", "TODO:.*refactor"
- **semantic**: Find code by meaning via embedding similarity. Use for concepts or when exact identifiers are unknown.
  Examples: "authentication logic", "retry with exponential backoff", "database connection pooling"

DECISION GUIDE:
- Known symbol or pattern → regex
- Concept or behavior → semantic

OUTPUT: Markdown blocks — file path, line range, symbol name, code block, pagination footer."""

CODE_RESEARCH_DESCRIPTION = """Start here for any coding task. Call code_research first to understand the relevant code area before writing or modifying code.

WORKFLOW:
1. **Understand** — call code_research to map architecture, components, and data flow
2. **Deepen** — call again with focused queries on specific subsystems discovered in step 1
3. **Pinpoint** — switch to search (regex/semantic) for exact file locations and symbol references
4. **Inspect** — use Explore/grep/read for granular line-level follow-up

WHAT IT RETURNS: Cited markdown report covering architecture overview, key code locations, component relationships, and cross-file data flows.

EXAMPLES:
- "How does authentication work?" — traces the full auth flow across files
- "What happens when a request hits /api/users?" — maps the request lifecycle
- "Explain error handling patterns" — identifies cross-cutting concerns

SCOPE: Use the path parameter to restrict analysis to a subdirectory for faster, focused results.

One call replaces 5-10 manual searches. Call it liberally — understanding first, coding second."""

DAEMON_STATUS_DESCRIPTION = """Report daemon startup, scan, and realtime
indexing health.

USE FOR:
- Checking whether initial indexing has completed
- Inspecting backend-neutral realtime health and resync state
- Debugging degraded daemon behavior without opening log files

OUTPUT: {status, query_ready, scan_progress}
NOTE: Query readiness is derived from scan state on this branch."""


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
        type: Search mode — "regex" for exact pattern matching, "semantic" for meaning-based similarity
        query: For regex: a regex pattern like "def authenticate" or "class.*Handler". For semantic: a natural language concept like "retry logic" or "database connection pooling"
        path: Optional relative subdirectory to restrict search scope, e.g. "src/auth" or "lib/payments" (no leading slash)
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

    return cast(
        SearchResponse, {"results": native_results, "pagination": pagination}
    )


@register_tool(
    description=DAEMON_STATUS_DESCRIPTION,
    requires_embeddings=False,
    name="daemon_status",
)
async def daemon_status_impl(
    scan_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return backend-neutral daemon and realtime status."""
    return derive_daemon_status(scan_progress)


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
    llm_manager: LLMManager | None,
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
        query: Natural language question about codebase architecture or behavior, e.g. "how does authentication work end-to-end?" or "explain the request lifecycle"
        progress: Optional Rich Progress instance for terminal UI (None for MCP)
        path: Optional relative subdirectory to restrict analysis scope, e.g. "src/auth" or "lib/payments" (no leading slash)
        config: Application configuration (optional, defaults to environment config)

    Returns:
        Dict with answer and metadata

    Raises:
        Exception: If LLM or reranker not configured
    """
    # Validate LLM is configured
    if not llm_manager:
        raise Exception(
            "No LLM provider configured. Code research requires an LLM. "
            "Configure an llm section in your chunkhound configuration."
        )

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

    # search tool renders dict → lean markdown for MCP, with markdown-based token limiting
    if tool_name == "search":
        if isinstance(result, dict):
            search_type = arguments.get("type", "regex")
            results_list = list(result.get("results", []))
            pagination = dict(result.get("pagination", {}))
            md = format_search_results_markdown(results_list, pagination, search_type)
            # Keep at least 1 result; preserve original page_size so the footer's
            # total-page count stays calibrated to the requested page size.
            while len(results_list) > 1 and estimate_tokens(md) > MAX_RESPONSE_TOKENS:
                trim = max(1, len(results_list) // 4)
                results_list = results_list[:-trim]
                pagination = {
                    **pagination,
                    "has_more": True,
                    "next_offset": pagination.get("offset", 0) + len(results_list),
                }
                md = format_search_results_markdown(results_list, pagination, search_type)
            return md

    # Convert result to dict if it's not already
    if hasattr(result, "__dict__"):
        return dict(result)
    elif isinstance(result, dict):
        return result
    else:
        return {"result": result}

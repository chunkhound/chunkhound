# MCP Server Module Context

## MODULE_PURPOSE
The MCP server module implements Model Context Protocol server for ChunkHound using stdio transport. This module exposes ChunkHound's search and research capabilities to AI assistants via standardized MCP tools.

## UNIFIED_TOOL_REGISTRY_ARCHITECTURE

### Core Principle: Single Source of Truth
**CRITICAL**: All MCP tool definitions live in `tools.py` via the `@register_tool` decorator.
The stdio server references `TOOL_REGISTRY` for all tool definitions.

**Why This Matters:**
- Centralized tool definitions ensure consistency
- Schema changes automatically applied to stdio server
- Tests validate tool registry (`test_mcp_tool_consistency.py`)

### Schema Auto-Generation Pattern

```python
# PATTERN: Define tool once with decorator
@register_tool(
    description="Find exact code patterns using regular expressions...",
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
    # Implementation...
```

**What Happens:**
1. `@register_tool` decorator extracts JSON Schema from function signature
2. Parses docstring for parameter descriptions
3. Registers in global `TOOL_REGISTRY` dict
4. Stdio server uses this registry for all tool definitions

**Schema Generation Details:**
- `_generate_json_schema_from_signature()` inspects function signature
- Converts Python type hints to JSON Schema types (`str` → `{"type": "string"}`, etc.)
- Extracts descriptions from docstring Args section (Google style)
- Handles Optional types, Union types, defaults automatically
- Filters out infrastructure params (`services`, `embedding_manager`, `llm_manager`, `scan_progress`)

## SERVER_ARCHITECTURE

### Base Class Pattern (`base.py`)

```
MCPServerBase (abstract)
├── __init__: Common initialization (config, services, embedding manager)
├── initialize(): Lazy service creation (databases, embeddings, LLMs)
├── ensure_services(): Thread-safe service initialization
├── cleanup(): Resource cleanup
└── debug_log(): Stderr logging (stdio-safe)

StdioMCPServer (stdio.py)
├── Inherits MCPServerBase
├── Uses MCP SDK
├── Global state (stdio constraint)
└── _register_tools()
```

### Stdio Implementation Details

**Stdio Constraints:**
- Must use global state (connection is singleton)
- NO STDOUT LOGS (breaks JSON-RPC protocol)
- Initialization happens once at server startup

**Base Class Provides:**
- Service initialization logic
- Configuration validation
- Error handling patterns
- Debug logging (stderr-only, stdio-safe)

## TOOL_IMPLEMENTATION_PATTERN

### Core Implementation vs. Protocol Wrapper

```python
# tools.py - Core implementation (protocol-agnostic)
@register_tool(description="...", requires_embeddings=False)
async def search_regex_impl(
    services: DatabaseServices,
    pattern: str,
    page_size: int = 10,
    # ... infrastructure params filtered from schema
) -> SearchResponse:
    # Pure business logic
    results = await services.search_service.search_regex(...)
    return {"results": results, "pagination": pagination}

# stdio.py - Stdio wrapper
async def handle_tool_call(tool_name: str, arguments: dict, ...) -> list[types.TextContent]:
    tool_def = TOOL_REGISTRY[tool_name]
    result = await tool_def.implementation(
        services=services,
        embedding_manager=embedding_manager,
        **arguments,  # User params only
    )
    return [types.TextContent(type="text", text=json.dumps(result))]
```

**Separation of Concerns:**
- `tools.py`: Pure business logic, protocol-agnostic
- `stdio.py`: Stdio protocol wrapper
- `common.py`: Shared utilities (argument parsing, error handling)

## ADDING_NEW_TOOLS

### Step-by-Step Process

1. **Implement core logic in `tools.py`:**
   ```python
   @register_tool(
       description="Comprehensive description for LLM users",
       requires_embeddings=True,  # or False
       name="my_new_tool",  # Optional, defaults to function name
   )
   async def my_tool_impl(
       services: DatabaseServices,
       embedding_manager: EmbeddingManager,  # if requires_embeddings=True
       llm_manager: LLMManager,  # if needed
       query: str,  # User params with type hints
       count: int = 10,  # Defaults work automatically
   ) -> dict[str, Any]:
       """Tool description.

       Args:
           services: Database services (filtered from schema)
           embedding_manager: Embedding manager (filtered from schema)
           llm_manager: LLM manager (filtered from schema)
           query: Search query text
           count: Number of results to return

       Returns:
           Dict with results
       """
       # Implementation...
       return {"results": [...]}
   ```

2. **Stdio mode works automatically** (handles all tools in TOOL_REGISTRY)

3. **Add tests in `test_mcp_tool_consistency.py`:**
   ```python
   def test_my_new_tool_schema():
       """Verify my_new_tool has correct schema."""
       tool = TOOL_REGISTRY["my_new_tool"]
       assert "query" in tool.parameters["properties"]
       assert "query" in tool.parameters["required"]
   ```

**NEVER:**
- Duplicate tool definitions in server files
- Hardcode descriptions (use `TOOL_REGISTRY[name].description`)
- Manually write JSON Schema (derive from function signatures)

**ALWAYS:**
- Use `@register_tool` decorator for all tools
- Extract parameter descriptions from docstrings
- Add consistency tests

## COMMON_MODIFICATIONS

### Changing Tool Parameter

```python
# GOOD: Change in tools.py, propagates automatically
@register_tool(...)
async def search_regex_impl(
    services: DatabaseServices,
    pattern: str,
    limit: int = 10,  # Renamed parameter from page_size
    # ...
```

**Why This Works:**
- Schema auto-generated from signature (picks up `limit`)
- Stdio mode uses `execute_tool()` which calls implementation
- Tests validate consistency

### Adding Parameter Description

```python
# Descriptions come from docstring, NOT decorator
@register_tool(description="Search using regex...")
async def search_regex_impl(
    services: DatabaseServices,
    pattern: str,
    new_param: bool = False,
) -> dict:
    """Search implementation.

    Args:
        services: Database services
        pattern: Regex pattern to search for
        new_param: Enable new experimental feature  # ← Add here
    """
```

Schema will automatically include: `{"new_param": {"type": "boolean", "description": "Enable new experimental feature", "default": false}}`

### Changing Tool Description

```python
# Change only in decorator
@register_tool(
    description="NEW DESCRIPTION HERE",  # ← Change once
    requires_embeddings=False,
)
async def search_regex_impl(...):
    ...

# Stdio server automatically picks this up from TOOL_REGISTRY
```

## TESTING_STRATEGY

### Consistency Tests (`test_mcp_tool_consistency.py`)

**Purpose**: Validate TOOL_REGISTRY structure and decorator behavior

```python
def test_search_regex_schema():
    """Verify search_regex has correct schema from decorator."""
    tool = TOOL_REGISTRY["search_regex"]

    # Check description
    assert "regular expressions" in tool.description.lower()

    # Check parameters auto-generated from signature
    props = tool.parameters["properties"]
    assert "pattern" in props
    assert "page_size" in props

    # Check required fields
    assert "pattern" in tool.parameters["required"]
```

**What This Validates:**
- Tool registration via @register_tool decorator
- Schema auto-generation from function signatures
- Parameter descriptions from docstrings
- Required vs optional parameter detection

### Smoke Tests (`test_smoke.py`)

**Purpose**: Verify server can start without crashes

```python
def test_import_mcp_server():
    """Verify MCP server can be imported."""
    from chunkhound.mcp_server import StdioMCPServer
    assert StdioMCPServer is not None
```

## DEBUGGING_TIPS

### Tool Not Available in MCP Client

**Symptom**: Tool doesn't appear in stdio server

**Check:**
1. Is tool in `TOOL_REGISTRY`? (Check `@register_tool` decorator applied)
2. Does tool require embeddings but none configured? (Check `requires_embeddings` flag)
3. Check server logs for initialization errors

### Parameter Description Missing

**Symptom**: LLM doesn't understand parameter purpose

**Fix**: Add to docstring Args section (not decorator):
```python
@register_tool(description="...")
async def tool_impl(services, param: str):
    """Tool description.

    Args:
        services: Filtered from schema
        param: ADD DESCRIPTION HERE  # ← Must be in docstring
    """
```

## MODIFICATION_RULES

**NEVER:**
- Remove `@register_tool` decorator from tool implementations
- Duplicate tool definitions in server files
- Hardcode JSON Schema (always generate from signatures)
- Use `print()` in any MCP server code (breaks stdio protocol)
- Create tools without adding to `TOOL_REGISTRY`

**ALWAYS:**
- Use `@register_tool` decorator for all new tools
- Extract descriptions from docstrings (Google style Args section)
- Run consistency tests before committing (`uv run pytest tests/test_mcp_*`)
- Keep infrastructure params (`services`, `embedding_manager`, etc.) separate from user params
- Log to stderr only (use `debug_log()` method)

**CRITICAL_CONSTRAINTS:**
- MCP stdio: NO STDOUT LOGS (breaks JSON-RPC protocol)
- Schema generation: Must filter infrastructure params from JSON Schema
- Consistency tests: Validate tool registry correctness

## KEY_FILES

```
mcp_server/
├── base.py           # Abstract base class (common initialization)
├── stdio.py          # Stdio transport server (MCP SDK)
├── tools.py          # Tool registry and implementations (SINGLE SOURCE OF TRUTH)
├── common.py         # Shared utilities (argument parsing, error handling)
└── __init__.py       # Lazy imports (avoid hard dependencies)
```

## PERFORMANCE_NOTES

- **Global state**: Stdio server must initialize once at startup (protocol constraint)
- **Schema caching**: TOOL_REGISTRY populated once at import time
- **Eager initialization**: All services initialized at server startup

## VERSION_HISTORY

- **v5.x.x**: HTTP transport removed - stdio only
  - HTTP MCP server removed (FastMCP dependency removed)
  - Simplified to stdio-only transport
  - Reduced complexity and dependencies
- **v2.x.x**: Unified tool registry architecture introduced
  - Breaking change: `path_filter` → `path` parameter rename
  - Added schema auto-generation from function signatures
  - Added comprehensive consistency tests

# MCP Server Module Context

## MODULE_PURPOSE
The MCP server module implements Model Context Protocol servers for ChunkHound, providing both stdio and HTTP transports. This module exposes ChunkHound's search and research capabilities to AI assistants via standardized MCP tools.

## UNIFIED_TOOL_REGISTRY_ARCHITECTURE

### Core Principle: Single Source of Truth
**CRITICAL**: All MCP tool definitions live in `tools.py` via the `@register_tool` decorator.
Both stdio and HTTP servers reference `TOOL_REGISTRY` - they never duplicate tool definitions.

**Why This Matters:**
- Prevents HTTP/stdio divergence (tools had different descriptions/parameters before unification)
- Schema changes automatically propagate to both transports
- Tests can validate consistency (`test_mcp_tool_consistency.py`, `test_mcp_http_schema_consistency.py`)

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
4. Both stdio and HTTP servers reference this registry

**Schema Generation Details:**
- `_generate_json_schema_from_signature()` inspects function signature
- Converts Python type hints to JSON Schema types (`str` → `{"type": "string"}`, etc.)
- Extracts descriptions from docstring Args section (Google style)
- Handles Optional types, Union types, defaults automatically
- Filters out infrastructure params (`services`, `embedding_manager`, `llm_manager`, `scan_progress`)

### HTTP Schema Patching Mechanism

**Problem**: FastMCP auto-generates schemas from function signatures, but uses verbose formats
(e.g., `anyOf: [{type: "string"}, {type: "null"}]` for optional params).

**Solution**: HTTP server patches FastMCP's schemas after registration:

```python
# Pattern from http_server.py
@self.app.tool(description=TOOL_REGISTRY["search_regex"].description)
async def search_regex(pattern: str, page_size: int = 10, ...) -> dict[str, Any]:
    await self.initialize()
    return await execute_tool(...)

# After all @app.tool decorators execute:
self._patch_tool_schema("search_regex")

def _patch_tool_schema(self, tool_name: str) -> None:
    """Replace FastMCP's auto-generated schema with TOOL_REGISTRY schema."""
    tool = self.app._tool_manager._tools[tool_name]
    tool.parameters = TOOL_REGISTRY[tool_name].parameters  # Overwrite
```

**Why Patch?**
- FastMCP generates schemas, but we need exact parity with stdio mode
- TOOL_REGISTRY schemas are cleaner (no `anyOf` for optionals)
- Ensures HTTP tools match stdio tools exactly (tested in `test_mcp_http_schema_consistency.py`)

## SERVER_ARCHITECTURE

### Base Class Pattern (`base.py`)

```
MCPServerBase (abstract)
├── __init__: Common initialization (config, services, embedding manager)
├── initialize(): Lazy service creation (databases, embeddings, LLMs)
├── ensure_services(): Thread-safe service initialization
├── cleanup(): Resource cleanup
└── debug_log(): Stderr logging (stdio-safe)

StdioMCPServer (stdio.py)           HttpMCPServer (http_server.py)
├── Inherits MCPServerBase          ├── Inherits MCPServerBase
├── Uses MCP SDK                    ├── Uses FastMCP
├── Global state (stdio constraint) ├── Lazy initialization (HTTP pattern)
└── _register_tools()               └── _register_tools() + schema patching
```

### Why Two Implementations?

**Stdio Constraints:**
- Must use global state (connection is singleton)
- NO STDOUT LOGS (breaks JSON-RPC protocol)
- Initialization happens once at server startup

**HTTP Flexibility:**
- Can use lazy initialization (per-request)
- Can log to stdout if needed
- Multiple concurrent connections supported

**Shared via Base Class:**
- Service initialization logic
- Configuration validation
- Error handling patterns
- Debug logging (stderr-only, safe for both modes)

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

# http_server.py - HTTP wrapper
async def search_regex(pattern: str, page_size: int = 10, ...) -> dict[str, Any]:
    return await execute_tool(
        tool_name="search_regex",
        services=self.ensure_services(),
        arguments={"pattern": pattern, "page_size": page_size, ...},
    )
```

**Separation of Concerns:**
- `tools.py`: Pure business logic, protocol-agnostic
- `stdio.py` / `http_server.py`: Protocol-specific wrappers
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

2. **Add HTTP wrapper in `http_server.py`:**
   ```python
   @self.app.tool(description=TOOL_REGISTRY["my_new_tool"].description)
   async def my_new_tool(query: str, count: int = 10) -> dict[str, Any]:
       await self.initialize()
       return await execute_tool(
           tool_name="my_new_tool",
           services=self.ensure_services(),
           embedding_manager=self.embedding_manager,
           arguments={"query": query, "count": count},
           llm_manager=self.llm_manager,
       )

   # In _register_tools(), add to schema patching:
   self._patch_tool_schema("my_new_tool")
   ```

3. **Stdio mode works automatically** (handles all tools in TOOL_REGISTRY)

4. **Add tests in `test_mcp_tool_consistency.py`:**
   ```python
   def test_my_new_tool_schema():
       """Verify my_new_tool has correct schema."""
       tool = TOOL_REGISTRY["my_new_tool"]
       assert "query" in tool.parameters["properties"]
       assert "query" in tool.parameters["required"]
   ```

**NEVER:**
- Duplicate tool definitions between stdio and HTTP
- Hardcode descriptions in server files (use `TOOL_REGISTRY[name].description`)
- Manually write JSON Schema (derive from function signatures)

**ALWAYS:**
- Use `@register_tool` decorator for all tools
- Extract parameter descriptions from docstrings
- Add schema patching call in HTTP server
- Add consistency tests

## COMMON_MODIFICATIONS

### Changing Tool Parameter

```python
# BAD: Changing only in one place
async def search_regex(pattern: str, limit: int = 10):  # Changed page_size → limit
    # This breaks stdio/HTTP consistency!

# GOOD: Change in tools.py, propagates everywhere
@register_tool(...)
async def search_regex_impl(
    services: DatabaseServices,
    pattern: str,
    limit: int = 10,  # Renamed parameter
    # ...
```

**Why This Works:**
- Schema auto-generated from signature (picks up `limit`)
- HTTP wrapper references implementation signature
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
# Change only in decorator, not in server files
@register_tool(
    description="NEW DESCRIPTION HERE",  # ← Change once
    requires_embeddings=False,
)
async def search_regex_impl(...):
    ...

# HTTP server automatically picks this up via:
# @self.app.tool(description=TOOL_REGISTRY["search_regex"].description)
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

### HTTP Schema Consistency Tests (`test_mcp_http_schema_consistency.py`)

**Purpose**: Ensure HTTP server schemas exactly match TOOL_REGISTRY

```python
async def test_http_tools_match_tool_registry(http_server):
    """Verify HTTP tool schemas match TOOL_REGISTRY exactly."""
    for tool_name in TOOL_REGISTRY.keys():
        http_tool = await http_server.app._tool_manager.get_tool(tool_name)
        expected_schema = TOOL_REGISTRY[tool_name].parameters

        # HTTP schema must match TOOL_REGISTRY (schema patching test)
        assert http_tool.parameters == expected_schema
```

**What This Catches:**
- Missing schema patching calls
- FastMCP auto-generation divergence
- Optional parameter format differences (`anyOf` vs simple types)

### Smoke Tests (`test_smoke.py`)

**Purpose**: Verify servers can start without crashes

```python
def test_import_mcp_servers():
    """Verify MCP servers can be imported."""
    from chunkhound.mcp_server import HttpMCPServer, StdioMCPServer
    assert HttpMCPServer is not None
    assert StdioMCPServer is not None
```

## DEBUGGING_TIPS

### Schema Mismatch Between HTTP and Stdio

**Symptom**: Tests fail in `test_mcp_http_schema_consistency.py`

**Common Causes:**
1. Forgot to call `self._patch_tool_schema("tool_name")` in HTTP server
2. HTTP wrapper signature doesn't match implementation signature
3. Tool not in TOOL_REGISTRY (decorator not applied)

**Fix:**
```python
# In http_server.py _register_tools():
@self.app.tool(description=TOOL_REGISTRY["new_tool"].description)
async def new_tool(param: str) -> dict:
    # ...

# ADD THIS:
self._patch_tool_schema("new_tool")
```

### Tool Not Available in MCP Client

**Symptom**: Tool appears in stdio but not HTTP (or vice versa)

**Check:**
1. Is tool in `TOOL_REGISTRY`? (Check `@register_tool` decorator applied)
2. Is HTTP wrapper registered? (Check `@self.app.tool()` decorator)
3. Is schema patching called? (Check `_patch_tool_schema()` call)
4. Does tool require embeddings but none configured? (Check `requires_embeddings` flag)

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
- Duplicate tool definitions between `tools.py` and server files
- Hardcode JSON Schema (always generate from signatures)
- Use `print()` in any MCP server code (breaks stdio protocol)
- Create tools without adding to `TOOL_REGISTRY`

**ALWAYS:**
- Use `@register_tool` decorator for all new tools
- Extract descriptions from docstrings (Google style Args section)
- Call `_patch_tool_schema()` for all HTTP tools
- Run consistency tests before committing (`uv run pytest tests/test_mcp_*`)
- Keep infrastructure params (`services`, `embedding_manager`, etc.) separate from user params
- Log to stderr only (use `debug_log()` method)

**CRITICAL_CONSTRAINTS:**
- MCP stdio: NO STDOUT LOGS (breaks JSON-RPC protocol)
- Schema generation: Must filter infrastructure params from JSON Schema
- HTTP patching: Required to override FastMCP's auto-generated schemas
- Consistency tests: MANDATORY to prevent HTTP/stdio divergence

## KEY_FILES

```
mcp_server/
├── base.py           # Abstract base class (common initialization)
├── stdio.py          # Stdio transport server (MCP SDK)
├── http_server.py    # HTTP transport server (FastMCP)
├── tools.py          # Tool registry and implementations (SINGLE SOURCE OF TRUTH)
├── common.py         # Shared utilities (argument parsing, error handling)
└── __init__.py       # Lazy imports (avoid hard transport dependencies)
```

## PERFORMANCE_NOTES

- **Lazy initialization**: HTTP server initializes services on first request
- **Global state**: Stdio server must initialize once at startup (protocol constraint)
- **Schema caching**: TOOL_REGISTRY populated once at import time
- **No overhead**: Schema patching happens once during server initialization

## VERSION_HISTORY

- **v2.x.x**: Unified tool registry architecture introduced
  - Breaking change: `path_filter` → `path` parameter rename
  - Eliminated HTTP/stdio tool definition duplication
  - Added schema auto-generation from function signatures
  - Added comprehensive consistency tests

# Smoke Tests for ChunkHound

## PURPOSE
Catch critical failures BEFORE they reach users. These tests run in ~10 seconds and prevent:
- Import-time crashes (type annotation errors, syntax errors)
- CLI command failures
- Server startup crashes

## WHEN_TO_RUN
- ALWAYS: Before any commit
- ALWAYS: After modifying type annotations
- ALWAYS: After adding new modules or CLI commands
- AUTOMATED: Should run in CI on every push

## HOW_TO_RUN
```bash
# Quick smoke test (10 seconds)
uv run pytest tests/test_smoke.py -v

# Just import tests (2 seconds)
uv run pytest tests/test_smoke.py::TestModuleImports -v

# Just CLI tests (3 seconds)
uv run pytest tests/test_smoke.py::TestCLICommands -v
```

## WHAT_THEY_TEST

### 1. Module Import Tests
- **Purpose**: Catch syntax/type errors at import time
- **Example Bug Caught**: `_server_config: "Config" | None = None`
- **Coverage**: All chunkhound modules via pkgutil.walk_packages

### 2. CLI Command Tests
- **Purpose**: Ensure CLI doesn't crash on basic operations
- **Coverage**: All major commands with --help flag

### 3. Server Startup Tests
- **Purpose**: Verify MCP stdio server can start without immediate crashes
- **Coverage**: MCP stdio server startup and protocol handshake

### 4. Type Annotation Pattern Tests
- **Purpose**: Detect problematic patterns before they cause crashes
- **Coverage**: Scans for forward reference union patterns

## REAL_WORLD_EXAMPLE
The type annotation bug that previously crashed the MCP server:
```python
# BAD - Crashes at import time
_server_config: "Config" | None = None

# GOOD - Config already imported, no quotes needed
_server_config: Config | None = None
```

Smoke tests catch these issues immediately at import time, preventing runtime failures.

## DESIGN_PRINCIPLES
1. **FAST**: Total runtime < 10 seconds
2. **FOCUSED**: Only test critical paths
3. **RELIABLE**: No flaky tests, no external dependencies
4. **ACTIONABLE**: Clear error messages pointing to the issue

## FUTURE_ADDITIONS
When adding new functionality:
1. Add module to `critical_modules` list if it's user-facing
2. Add CLI command to parametrized test if it has --help
3. Consider adding specific import test for complex modules

## INTEGRATION_WITH_AI_DEVELOPMENT
Since ChunkHound is 100% AI-developed:
- Smoke tests act as guardrails for AI agents
- Prevent syntax/type errors that AI might introduce
- Provide immediate feedback loop during development
- CRITICAL: AI agents should ALWAYS run smoke tests before marking tasks complete
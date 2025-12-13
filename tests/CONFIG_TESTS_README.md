# Config System Unification Tests

This directory contains TDD (Test-Driven Development) tests for the config system unification effort documented in ticket `2025-07-16-feature-config-system-unification-ai-guardrails.md`.

## Test Files

### 1. `test_config_unification.py`
**Purpose**: Core tests for unified configuration patterns across CLI and MCP paths.

**Key Tests**:
- `TestCLIConfigHandling`: Tests CLI commands use `args_to_config()` with validation
- `TestMCPServerConfigHandling`: Tests MCP stdio server uses environment-based config with validation
- `TestConfigConsistencyIntegration`: Tests all paths use same validation and registry setup

**Expected Failures**: Most tests will fail until:
- CLI commands use proper validation
- MCP servers add validation calls
- ApplicationContext is implemented
- Registry configuration is unified

### 2. `test_mcp_server_config_patterns.py`
**Purpose**: Specific tests for MCP server configuration patterns and code analysis.

**Key Tests**:
- `TestMCPServerStdioConfigPattern`: Tests stdio server configuration lifecycle
- `TestMCPServerCodeAnalysis`: Static analysis tests for required imports and patterns
- `TestMCPServerErrorHandling`: Tests error handling for validation failures

**Expected Failures**: Most tests will fail until:
- MCP servers import `validate_config_for_command`
- MCP servers use `create_database_with_dependencies`
- File change processing uses consistent config patterns
- Validation error handling is added

### 3. `test_config_integration.py`
**Purpose**: Integration tests for config consistency across all code paths.

**Key Tests**:
- `TestCLIToMCPConsistency`: Tests CLI and MCP create compatible configurations
- `TestRegistryIntegration`: Tests registry configuration consistency
- `TestErrorHandlingIntegration`: Tests error handling across all paths
- `TestFilePatternIntegration`: Tests file pattern handling consistency
- `TestEndToEndIntegration`: End-to-end tests for complete workflows

**Expected Failures**: Most tests will fail until:
- All paths use same validation logic
- Registry accepts both Config types consistently
- File pattern handling is unified
- Database factory is used consistently

## Running Tests

### Individual Test Run
```bash
uv run python -m pytest tests/test_config_unification.py -v
```

### All Config Tests
```bash
uv run python -m pytest tests/test_config_unification.py tests/test_mcp_server_config_patterns.py tests/test_config_integration.py -v
```

### Test Runner Script
```bash
python tests/run_config_tests.py
```

## Expected Test States

### Before Refactor (Current State)
- ❌ Most tests should **FAIL** - this is expected and correct
- ✅ A few baseline tests might pass (good foundation)
- 🔍 Tests define the target behavior we want to achieve

### After Refactor (Target State)
- ✅ All tests should **PASS** 
- 🎯 Tests verify the unified configuration system works correctly
- 🛡️ Tests act as regression protection for future changes

## Test Design Principles

### 1. **TDD Approach**
- Tests written before implementation
- Tests define desired behavior
- Red → Green → Refactor cycle

### 2. **Comprehensive Coverage**
- CLI command configuration
- MCP stdio server configuration
- Registry configuration
- Database factory usage
- Error handling
- File pattern handling

### 3. **Realistic Scenarios**
- Temporary directories for isolation
- Real config file creation
- Environment variable simulation
- Mock objects for external dependencies

### 4. **Clear Failure Messages**
- Tests explain what pattern is expected
- Failure messages guide implementation
- Comments explain the reasoning

## Key Patterns Tested

### CLI Pattern
```python
# ✅ CORRECT: CLI command configuration
project_dir = Path(args.path) if hasattr(args, "path") else Path.cwd()
unified_config = args_to_config(args, project_dir)

validation_errors = validate_config_for_command(unified_config, "index")
if validation_errors:
    raise ValueError("Invalid configuration")

configure_registry(unified_config._config)
```

### MCP Pattern
```python
# ✅ CORRECT: MCP server configuration  
config = Config()  # No target_dir - uses environment variables

validation_errors = validate_config_for_command(config, "mcp")
if validation_errors:
    raise ValueError(f"Config validation failed: {validation_errors}")

database = create_database_with_dependencies(
    db_path=Path(config.database.path),
    config=config,
    embedding_manager=embedding_manager,
)
```

## Integration with CI/CD

These tests should be run:
- Before any config-related changes
- After implementing ApplicationContext
- After unifying MCP server patterns
- As part of regular regression testing

## Maintenance Notes

- Tests are self-documenting through comments
- Mock fixtures in `conftest.py` provide reusable test setup
- Test patterns can be copied for new config-related features
- Static analysis tests ensure code follows required patterns

## Next Steps

1. **Run tests** to establish baseline failure state
2. **Implement ApplicationContext** (will enable skipped tests)
3. **Add validation to MCP servers** (will fix many failures)
4. **Unify database creation** (will fix factory tests)
5. **Add missing imports** (will fix code analysis tests)
6. **Verify all tests pass** after refactor completion

The tests serve as both specification and verification for the config system unification effort.
# Configuration System Testing Strategy

## Overview
This document outlines the comprehensive testing strategy for the unified configuration system to ensure it works correctly and prevent regressions.

## Testing Approaches

### 1. Unit Tests (test_unified_config_system.py)
Created comprehensive unit tests covering:
- ✅ Precedence order verification (CLI > env > file > defaults)
- ✅ All initialization contexts (CLI, MCP, direct)
- ✅ Deep merging behavior
- ✅ Error handling and validation
- ✅ Project detection and local config
- ✅ Backward compatibility

### 2. Integration Tests
Need to create tests that verify:
- Config works correctly with actual CLI commands
- MCP servers load config properly in real scenarios
- File watchers and periodic indexers use consistent config
- Database initialization respects config settings

### 3. Property-Based Testing
Use hypothesis to test:
- Any valid combination of config sources produces valid config
- Merging is associative (order of operations doesn't matter)
- No data loss during merging
- Type safety is maintained

### 4. End-to-End Tests
Verify complete workflows:
- `chunkhound index` with various config combinations
- `chunkhound mcp stdio` with environment setup
- `chunkhound search` with local config files
- Config file migration scenarios

### 5. Performance Tests
Ensure config loading doesn't regress:
- Measure config initialization time
- Test with large config files
- Verify no repeated file I/O
- Check memory usage

### 6. Security Tests
Verify security aspects:
- API keys are properly masked in logs
- Sensitive data isn't exposed in error messages
- Config files with bad permissions are handled
- Path traversal attacks are prevented

## Critical Test Scenarios

### Scenario 1: MCP Server Launch
```bash
# Test that MCP server gets config from environment
export CHUNKHOUND_DATABASE__PATH=/tmp/test.db
export CHUNKHOUND_EMBEDDING__PROVIDER=openai
export CHUNKHOUND_EMBEDDING__API_KEY=test-key
chunkhound mcp stdio
```

### Scenario 2: CLI with Local Config
```bash
# Test that CLI respects local .chunkhound.json
cd /project/with/config
cat > .chunkhound.json << EOF
{
  "database": {"provider": "lancedb"},
  "embedding": {"provider": "openai", "base_url": "http://localhost:11434/v1"}
}
EOF
chunkhound index .
```

### Scenario 3: Override Chain
```bash
# Test complete override chain
export CHUNKHOUND_DEBUG=false
cat > config.json << EOF
{"debug": true}
EOF
chunkhound --config config.json --debug false index .
# Should result in debug=false (CLI wins)
```

### Scenario 4: Nested Config Merging
```bash
# Test deep merging of nested configs
export CHUNKHOUND_DATABASE__PATH=/env/path
cat > .chunkhound.json << EOF
{
  "database": {
    "provider": "lancedb",
    "lancedb_index_type": "IVF_PQ"
  }
}
EOF
chunkhound --db /cli/path index .
# Should have provider=lancedb, path=/cli/path, index_type=IVF_PQ
```

## Regression Prevention

### 1. Smoke Tests
Add to test_smoke.py:
```python
def test_config_import_no_side_effects():
    """Ensure config import has no side effects."""
    import chunkhound.core.config.config
    # Should not raise or modify environment

def test_all_cli_commands_config_compatible():
    """Ensure all CLI commands can initialize config."""
    commands = ["index", "search", "run", "mcp"]
    for cmd in commands:
        # Should be able to create config for each
```

### 2. Contract Tests
Define contracts that must always be true:
- Config() with no args must always work
- Config(args=x) must respect all args fields
- Environment variables must map correctly
- Local config must be found if in project root

### 3. Mutation Testing
Use mutmut to ensure tests catch changes:
```bash
mutmut run --paths-to-mutate chunkhound/core/config/config.py
```

### 4. Coverage Requirements
- Line coverage: 100% for config.py
- Branch coverage: 100% for precedence logic
- Path coverage: All config source combinations

## Continuous Monitoring

### 1. Performance Benchmarks
```python
# Add to benchmarks/
def benchmark_config_initialization(benchmark):
    benchmark(Config)

def benchmark_config_with_all_sources(benchmark, tmp_path):
    # Set up env, files, args
    benchmark(Config, args=args)
```

### 2. Memory Profiling
```python
# Check for memory leaks in config loading
@profile
def test_config_memory_usage():
    for _ in range(1000):
        Config()
        reset_config()
```

### 3. Real-World Testing
- Test with actual user config files
- Test with production environment variables
- Test with complex project structures

## Test Data Management

### 1. Config File Fixtures
Create fixtures for common scenarios:
```python
@pytest.fixture
def minimal_config():
    return {"database": {"path": "/tmp/test.db"}}

@pytest.fixture
def full_config():
    return {
        "database": {...},
        "embedding": {...},
        "mcp": {...},
        "indexing": {...}
    }

@pytest.fixture
def invalid_configs():
    return [
        "not json",
        {"database": "not an object"},
        {"invalid_key": "value"}
    ]
```

### 2. Environment Fixtures
```python
@pytest.fixture
def clean_env(monkeypatch):
    """Remove all CHUNKHOUND_ env vars."""
    for key in list(os.environ.keys()):
        if key.startswith("CHUNKHOUND_"):
            monkeypatch.delenv(key)

@pytest.fixture
def full_env(monkeypatch):
    """Set all possible env vars."""
    vars = {
        "CHUNKHOUND_DEBUG": "true",
        "CHUNKHOUND_DATABASE__PATH": "/env/db",
        # ... all variables
    }
    for k, v in vars.items():
        monkeypatch.setenv(k, v)
```

## Error Injection Testing

Test resilience to:
1. Corrupted config files
2. Missing directories
3. Permission errors
4. Network timeouts (for embeddings)
5. Invalid type conversions
6. Circular config references

## Documentation Tests

Ensure examples in documentation work:
```python
def test_readme_examples():
    """Test that README config examples work."""
    # Extract code blocks from README
    # Execute and verify they work

def test_docstring_examples():
    """Test that docstring examples work."""
    import doctest
    doctest.testmod(chunkhound.core.config.config)
```

## Migration Testing

Test upgrade paths:
1. Old env vars → new env vars
2. Old config format → new format
3. Missing config → defaults
4. v2.7 configs → v2.8 configs

## Observability

Add logging to track:
- Which config sources were used
- What values were overridden
- Performance metrics
- Validation failures

```python
# Add debug logging
logger.debug(f"Config sources used: {sources}")
logger.debug(f"Final precedence: {precedence_chain}")
logger.debug(f"Load time: {load_time}ms")
```

## Success Criteria

The configuration system is considered properly tested when:
1. ✅ 100% code coverage on config modules
2. ✅ All precedence combinations tested
3. ✅ All error paths tested
4. ✅ Performance benchmarks established
5. ✅ Real-world scenarios validated
6. ✅ No regressions in 10 releases
7. ✅ Migration from old patterns complete
8. ✅ Documentation examples verified
9. ✅ Security review passed
10. ✅ Mutation testing score > 90%
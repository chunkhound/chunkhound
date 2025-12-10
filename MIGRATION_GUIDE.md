# ChunkHound Configuration Migration Guide

This guide helps you migrate from the old configuration system to the new centralized configuration system introduced in v2.2.0.

## Key Changes

### 1. Configuration File Loading

**Old behavior:**
- Automatically loaded `.chunkhound.json` from project root
- Automatically loaded `~/.chunkhound/config.json` from home directory

**New behavior:**
- Config files are only loaded when explicitly specified with `--config` flag
- Example: `chunkhound index . --config .chunkhound.json`

### 2. Configuration Precedence

The new system has a clear hierarchy (highest to lowest priority):
1. CLI arguments
2. Config file (via `--config` path)
3. Environment variables
4. Default values

### 3. Environment Variables

**New variables:**
All new environment variables use the `CHUNKHOUND_` prefix with `__` delimiter for nested values:

- `CHUNKHOUND_DEBUG` - Enable debug mode
- `CHUNKHOUND_DB_PATH` - Database file path
- `CHUNKHOUND_DATABASE__PROVIDER` - Database provider (sqlite/lancedb)
- `CHUNKHOUND_DATABASE__LANCEDB_INDEX_TYPE` - LanceDB index type
- `CHUNKHOUND_EMBEDDING__PROVIDER` - Embedding provider
- `CHUNKHOUND_EMBEDDING__MODEL` - Embedding model name
- `CHUNKHOUND_EMBEDDING__API_KEY` - API key for embeddings
- `CHUNKHOUND_EMBEDDING__BASE_URL` - Base URL for API
- `CHUNKHOUND_EMBEDDING__BATCH_SIZE` - Batch size for embeddings
- `CHUNKHOUND_EMBEDDING__MAX_CONCURRENT` - Max concurrent embedding batches
- `CHUNKHOUND_MCP__TRANSPORT` - MCP transport type (stdio/http)
- `CHUNKHOUND_MCP__PORT` - Port for HTTP transport
- `CHUNKHOUND_MCP__HOST` - Host for HTTP transport
- `CHUNKHOUND_MCP__CORS` - Enable CORS for HTTP
- `CHUNKHOUND_INDEXING__BATCH_SIZE` - Indexing batch size
- `CHUNKHOUND_INDEXING__DB_BATCH_SIZE` - Database batch size
- `CHUNKHOUND_INDEXING__MAX_CONCURRENT` - Max concurrent operations
- `CHUNKHOUND_INDEXING__FORCE_REINDEX` - Force reindexing
- `CHUNKHOUND_INDEXING__CLEANUP` - Enable cleanup
- `CHUNKHOUND_INDEXING__IGNORE_GITIGNORE` - Ignore gitignore files
- `CHUNKHOUND_INDEXING__INCLUDE` - Include patterns (comma-separated)
- `CHUNKHOUND_INDEXING__EXCLUDE` - Exclude patterns (comma-separated)

### 4. CLI Arguments

All configuration options are now available as CLI arguments:

```bash
# Database configuration
chunkhound index . --database-path /path/to/db --database-provider lancedb

# Embedding configuration
chunkhound index . --embedding-provider openai --embedding-model text-embedding-3-small

# MCP configuration (stdio only)
chunkhound mcp

# Indexing configuration
chunkhound index . --indexing-batch-size 1000
```

### 5. Configuration File Format

The configuration file format remains the same JSON structure:

```json
{
  "database": {
    "provider": "lancedb",
    "path": ".chunkhound/db"
  },
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "batch_size": 1000,
    "max_concurrent": 8
  },
  "mcp": {
    "transport": "stdio"
  },
  "indexing": {
    "batch_size": 100,
    "db_batch_size": 5000,
    "include": ["**/*.py", "**/*.js"],
    "exclude": ["**/node_modules/**", "**/venv/**"]
  },
  "debug": false
}
```

## Migration Steps

### Step 1: Update Command Lines

If you were relying on automatic config file loading:

**Old:**
```bash
chunkhound index .
```

**New:**
```bash
chunkhound index . --config .chunkhound.json
```

### Step 2: Update Environment Variables

If using environment variables, update to the new naming:

**Old:**
```bash
export OPENAI_API_KEY=sk-...
export CHUNKHOUND_DB_PATH=/path/to/db
```

**New:**
```bash
export CHUNKHOUND_EMBEDDING__API_KEY=sk-...
export CHUNKHOUND_DATABASE__PATH=/path/to/db
```

### Step 3: Review Configuration Precedence

Remember that CLI arguments now override config file values, which override environment variables:

```bash
# This will use text-embedding-3-large even if config file specifies different model
chunkhound index . --config .chunkhound.json --embedding-model text-embedding-3-large
```

### Step 4: Update Scripts and Automation

Update any scripts or CI/CD pipelines to:
1. Explicitly specify `--config` if using config files
2. Use new environment variable names
3. Take advantage of new CLI arguments for dynamic configuration

## Troubleshooting

### Config File Not Found

If you see an error about config file not found:
- Ensure you're using `--config` with the correct path
- The config file is no longer auto-detected

### Environment Variables Not Working

- Check you're using the correct `CHUNKHOUND_` prefix
- Use `__` (double underscore) for nested values
- Ensure no typos in variable names

### Unexpected Configuration Values

Use `--debug` flag to see which configuration source is being used:
```bash
chunkhound index . --config .chunkhound.json --debug
```

## Benefits of the New System

1. **Explicit Control**: No surprise config files being loaded
2. **Clear Precedence**: Always know which setting wins
3. **Full CLI Support**: Configure everything from command line
4. **Better Debugging**: Clear configuration hierarchy
5. **Consistent Naming**: All ChunkHound vars use same prefix
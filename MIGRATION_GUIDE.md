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
chunkhound index . --embedding-provider openai --embedding-model text-embedding-3-large

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
    "model": "text-embedding-3-large",
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

## Default Embedding Model Change

Starting with this release, the default OpenAI embedding model has changed:

- **Old default:** `text-embedding-3-small` (1536 dimensions)
- **New default:** `text-embedding-3-large` (3072 dimensions)

### Impact

- **Existing indexes are incompatible** — embeddings with different dimensions cannot be mixed. You must reindex after upgrading.
- **Higher quality** — `text-embedding-3-large` produces more accurate embeddings for code search.
- **Higher cost** — `text-embedding-3-large` costs more per token than `text-embedding-3-small`. See [OpenAI pricing](https://openai.com/api/pricing/) for details.

### If you want to keep the old default

Explicitly set the model in your configuration:

```bash
chunkhound index . --embedding-model text-embedding-3-small
```

Or in `.chunkhound.json`:

```json
{
  "embedding": {
    "model": "text-embedding-3-small"
  }
}
```

### If you want to use the new default

Reindex your codebase:

```bash
chunkhound index . --force-reindex
```

## Benefits of the New System

1. **Explicit Control**: No surprise config files being loaded
2. **Clear Precedence**: Always know which setting wins
3. **Full CLI Support**: Configure everything from command line
4. **Better Debugging**: Clear configuration hierarchy
5. **Consistent Naming**: All ChunkHound vars use same prefix

## Matryoshka Embeddings (output_dims)

Some models (e.g., `text-embedding-3-large`, `text-embedding-3-small`) support
matryoshka representation learning: embeddings can be truncated to smaller
dimensions without retraining, trading accuracy for speed/storage.

### Configuration

```bash
# Via CLI
chunkhound index . --output-dims 256

# Via environment variable
export CHUNKHOUND_EMBEDDING__OUTPUT_DIMS=256

# Client-side truncation (for APIs that don't support the dimensions parameter)
chunkhound index . --output-dims 256 --client-side-truncation
export CHUNKHOUND_EMBEDDING__CLIENT_SIDE_TRUNCATION=true
```

### Supported dimensions by model

| Model | Native dims | Min dims | Recommended min |
|-------|-------------|----------|-----------------|
| `text-embedding-3-large` | 3072 | 1 | 256 |
| `text-embedding-3-small` | 1536 | 1 | 512 |

### Impact

- **Existing indexes are incompatible** — changing `output_dims` changes the
  embedding dimension, which requires a full reindex.
- **Smaller dimensions** = faster search, less storage, slightly lower accuracy.
- **Use `--force-reindex`** after changing `output_dims`.

## TLS Verification Default Change

SSL certificate verification is now **enabled by default** for all custom
endpoints. Previously, `verify=False` was used unconditionally for any
non-OpenAI endpoint (Ollama, vLLM, etc.), creating a MITM risk when API keys
are transmitted.

### If you use self-signed certificates

```bash
# Via CLI
chunkhound index . --no-verify-ssl

# Via environment variable
export CHUNKHOUND_EMBEDDING__VERIFY_SSL=false
```

**Note:** `setup_wizard.py` still uses `verify=False` during interactive endpoint
probing, where certificate validation would prevent discovery of local servers.

## OpenAI-Compatible Provider

Custom OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, etc.) now have a
dedicated `openai_compatible` provider type, giving clearer semantics:

```bash
# Via CLI
chunkhound index . \
  --provider openai_compatible \
  --base-url http://localhost:11434/v1 \
  --model nomic-embed-text

# Via environment variables
export CHUNKHOUND_EMBEDDING__PROVIDER=openai_compatible
export CHUNKHOUND_EMBEDDING__BASE_URL=http://localhost:11434/v1
export CHUNKHOUND_EMBEDDING__MODEL=nomic-embed-text
```

`base_url` is **required** for `openai_compatible`. An API key is optional
(local servers typically don't require one).

**Note on DB key:** `openai_compatible` stores embeddings under the same DB
key as `openai` (`provider=openai, model=<model_name>`). If you switch between
official OpenAI and a custom endpoint using the **same model name**, run
`--force-reindex` — the vectors are incompatible despite sharing a key.

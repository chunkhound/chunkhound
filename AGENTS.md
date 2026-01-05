# ChunkHound LLM Context

## PROJECT_IDENTITY
ChunkHound: Semantic and regex search tool for codebases with MCP integration
Built: 100% by AI agents - NO human-written code
Purpose: Transform codebases into searchable knowledge bases for AI assistants

## MODIFICATION_RULES
**NEVER:**
- NEVER Use print() in MCP server (stdio.py, http_server.py, tools.py)
- NEVER Make single-row DB inserts in loops
- NEVER Use forward references (quotes) in type annotations unless needed

**ALWAYS:**
- ALWAYS Run smoke tests before committing: `uv run pytest tests/test_smoke.py`
- ALWAYS Batch embeddings (min: 100, max: provider_limit)
- ALWAYS Drop HNSW indexes for bulk inserts > 50 rows
- ALWAYS Use uv for all Python operations
- ALWAYS Update version via: `uv run scripts/update_version.py`

## KEY_COMMANDS
```bash
# Development
lint:      uv run ruff check chunkhound
typecheck: uv run mypy chunkhound
test:      uv run pytest
smoke:     uv run pytest tests/test_smoke.py -v -n auto  # MANDATORY before commits
format:    uv run ruff format chunkhound

# Running
index:     uv run chunkhound index [directory]
mcp_stdio: uv run chunkhound mcp
mcp_http:  uv run chunkhound mcp http --port 5173
```

## VERSION_MANAGEMENT
Dynamic versioning via hatch-vcs - version derived from git tags.

```bash
# Create release
uv run scripts/update_version.py 4.1.0

# Create pre-release
uv run scripts/update_version.py 4.1.0b1
uv run scripts/update_version.py 4.1.0rc1

# Bump version
uv run scripts/update_version.py --bump minor      # v4.0.1 → v4.1.0
uv run scripts/update_version.py --bump minor b1   # v4.0.1 → v4.1.0b1
```

NEVER manually edit version strings - ALWAYS create git tags instead.

## PUBLISHING_PROCESS
```bash
# 1. Create version tag
uv run scripts/update_version.py X.Y.Z

# 2. Run smoke tests (MANDATORY)
uv run pytest tests/test_smoke.py -v

# 3. Prepare release
./scripts/prepare_release.sh

# 4. Test local install
pip install dist/chunkhound-X.Y.Z-py3-none-any.whl

# 5. Push tag
git push origin vX.Y.Z

# 6. Publish
uv publish
```

## KNOWN_DEPRECATION_WARNINGS
**HDBSCAN + scikit-learn**: `force_all_finite` parameter warning
- Non-breaking, safe to ignore
- Waiting for upstream HDBSCAN fix
- Will break in sklearn 1.8 if not fixed upstream

## PROJECT_MAINTENANCE
- Smoke tests are mandatory guardrails
- Run `uv run mypy chunkhound` during reviews to catch Optional/type boundary issues
- All code patterns should be self-documenting

## MULTI_REPOSITORY_SUPPORT

### Overview
ChunkHound supports two database modes:
- **Per-repo mode** (default): Each repository has its own `.chunkhound/db`
- **Global mode**: All repositories indexed to `~/.chunkhound/global/db`

Global mode enables cross-repository searches and concurrent MCP sessions.

### Configuration
```bash
# Enable global mode (required)
export CHUNKHOUND_DATABASE__MULTI_REPO__ENABLED=true
export CHUNKHOUND_DATABASE__MULTI_REPO__MODE=global
export CHUNKHOUND_DATABASE__MULTI_REPO__GLOBAL_DB_PATH=~/.chunkhound/global/db

# Daemon settings (optional)
export CHUNKHOUND_DATABASE__MULTI_REPO__DAEMON_PORT=5173
export CHUNKHOUND_DATABASE__MULTI_REPO__DAEMON_HOST=127.0.0.1

# File watcher settings (optional)
export CHUNKHOUND_DATABASE__MULTI_REPO__WATCHER_DEBOUNCE_SECONDS=2.0
export CHUNKHOUND_DATABASE__MULTI_REPO__WATCHER_MAX_PENDING_EVENTS=10000
export CHUNKHOUND_DATABASE__MULTI_REPO__WATCHER_MAX_PENDING_PER_PROJECT=500

# Proxy client settings (optional)
export CHUNKHOUND_DATABASE__MULTI_REPO__PROXY_TIMEOUT_SECONDS=120.0
```

### Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CHUNKHOUND_DATABASE__MULTI_REPO__ENABLED` | `false` | Enable multi-repository support |
| `CHUNKHOUND_DATABASE__MULTI_REPO__MODE` | `per-repo` | Database mode: `per-repo` or `global` |
| `CHUNKHOUND_DATABASE__MULTI_REPO__GLOBAL_DB_PATH` | `~/.chunkhound/global/db` | Path to global database |
| `CHUNKHOUND_DATABASE__MULTI_REPO__DAEMON_PORT` | `5173` | HTTP daemon port (1-65535) |
| `CHUNKHOUND_DATABASE__MULTI_REPO__DAEMON_HOST` | `127.0.0.1` | HTTP daemon host (use `0.0.0.0` for all interfaces) |
| `CHUNKHOUND_DATABASE__MULTI_REPO__WATCHER_DEBOUNCE_SECONDS` | `2.0` | Debounce delay for file events (0.1-60.0) |
| `CHUNKHOUND_DATABASE__MULTI_REPO__WATCHER_MAX_PENDING_EVENTS` | `10000` | Max total pending events before force-flush (100-1000000) |
| `CHUNKHOUND_DATABASE__MULTI_REPO__WATCHER_MAX_PENDING_PER_PROJECT` | `500` | Max pending events per project (10-10000) |
| `CHUNKHOUND_DATABASE__MULTI_REPO__PROXY_TIMEOUT_SECONDS` | `120.0` | Proxy client HTTP timeout (5.0-3600.0) |

### Daemon Commands
```bash
# Start HTTP daemon (global mode)
chunkhound daemon start --background

# Check status
chunkhound daemon status

# Stop daemon
chunkhound daemon stop

# View logs
chunkhound daemon logs --follow
```

### MCP Tool Parameters for Cross-Project Search

The `path` parameter accepts a list of paths (absolute or relative):

```python
# Search multiple projects by absolute path
search_semantic(query="auth", path=["/home/user/project-a", "/home/user/project-b"])

# Search subdirectories (relative paths resolved against client context)
search_semantic(query="auth", path=["src/", "tests/"])

# Search by tags (AND logic - must have ALL tags)
search_semantic(query="auth", tags=["backend", "python"])

# Tags + relative paths: apply paths to each tagged project
search_semantic(query="auth", tags=["work"], path=["src/"])
# → searches src/ in each work-tagged project

# Tags + absolute paths: union (tagged projects + absolute path)
search_semantic(query="auth", tags=["work"], path=["/external/lib/"])
# → searches all work projects + /external/lib/

# Tags + mixed paths: relative applied to tagged, absolute added directly
search_semantic(query="auth", tags=["work"], path=["src/", "/external/lib/"])
# → searches src/ in each work project + /external/lib/

# No path = current project (per-repo) or all projects (global mode)
search_semantic(query="auth")

# List all indexed projects (for discoverability)
list_projects()
```

### Tag Management
Tags provide flexible, multi-dimensional categorization of indexed projects.

```bash
# Add tags when indexing
chunkhound index /path/to/project --tags backend,python,work

# Manage tags on existing projects
chunkhound tags add myproject backend python
chunkhound tags remove myproject old-tag
chunkhound tags set myproject tag1 tag2 tag3
chunkhound tags list                    # Show all tags
chunkhound tags list --project myproject  # Show project's tags
```

**Path resolution:**
1. `tags=["x"]` → select projects with ALL specified tags
2. `path=["relative/"]` + tags → apply relative paths to each tagged project
3. `path=["/absolute/"]` + tags → tagged project roots + absolute paths (union)
4. `path=["relative/"]` (no tags) → resolve against client context project
5. `path=["/absolute/"]` (no tags) → use as-is
6. `path=None` → current project (per-repo) or all (global mode)

### Architecture
- **HTTP Daemon**: Single server handles all file watching and indexing
- **ProjectRegistry**: Tracks all indexed base directories with metadata
- **WatcherManager**: Centralized file watching with debounced reindexing
- **SearchContext**: Resolves multi-project queries to concrete paths
- **Multi-Path Search**: OR-based path filtering for `projects=[...]` searches
- **stdio proxy**: Forwards requests to HTTP daemon in global mode

### Multi-Path Search Implementation
When searching with multiple paths via `path=["/a", "/b"]`:
- `_resolve_paths()` in `tools.py` resolves relative paths and tags
- Single path → uses `path_filter` for simple SQL LIKE
- Multiple paths → uses `path_prefixes` for OR-based filtering
- Database generates SQL with OR-based WHERE clause:
  ```sql
  WHERE (f.path LIKE '/path/a/%' OR f.path LIKE '/path/b/%')
  ```
- Results are properly filtered to specified paths

### Critical Constraints
- HTTP daemon owns the database and all writes
- stdio MCP sessions proxy to daemon in global mode
- Per-repo mode: Single session has exclusive access
- All DB operations: Single-threaded via `SerialDatabaseProvider`

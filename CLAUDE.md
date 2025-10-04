# ChunkHound LLM Context

## PROJECT_IDENTITY
ChunkHound: Semantic and regex search tool for codebases with MCP (Model Context Protocol) integration
Built: 100% by AI agents - NO human-written code
Purpose: Transform codebases into searchable knowledge bases for AI assistants

## CRITICAL_CONSTRAINTS
- DuckDB/LanceDB: SINGLE_THREADED_ONLY (concurrent access = segfault/corruption)
- Embedding batching: MANDATORY (100x performance difference)
- Vector index optimization: DROP_BEFORE_BULK_INSERT (20x speedup for >50 embeddings)
- MCP server: NO_STDOUT_LOGS (breaks JSON-RPC protocol)
- File parsing: PARALLEL_BATCHES (CPU-bound parsing across cores, storage remains single-threaded)

## ARCHITECTURE_RATIONALE
- SerialDatabaseProvider: NOT_OPTIONAL (wraps all DB access in single thread)
- Service layers: REQUIRED_FOR_BATCHING (provider-specific optimizations)
- Global state in MCP: STDIO_CONSTRAINT (stateless would break connection)
- Database wrapper: LEGACY_COMPATIBILITY (provides migration path)
- Transaction backup tables: ATOMIC_FILE_UPDATES (ensures consistency)

## MODIFICATION_RULES
- NEVER: Remove SerialDatabaseProvider wrapper
- NEVER: Add concurrent database operations (parsing is parallelized, storage is single-threaded)
- NEVER: Use print() in MCP server
- NEVER: Make single-row DB inserts in loops
- NEVER: Use forward references (quotes) in type annotations unless needed
- ALWAYS: Run smoke tests before committing (uv run pytest tests/test_smoke.py)
- ALWAYS: Batch embeddings (min: 100, max: provider_limit)
- ALWAYS: Drop HNSW indexes for bulk inserts > 50 rows
- ALWAYS: Use uv for all Python operations
- ALWAYS: Update version via scripts/update_version.py

## PERFORMANCE_CRITICAL_NUMBERS
| Operation | Unbatched | Batched | Constraint |
|-----------|-----------|---------|------------|
| Embeddings (1000 texts) | 100s | 1s | API rate limits |
| DB inserts (5000 chunks) | 250s | 1s | Index overhead |
| File update (1000 chunks) | 60s | 5s | Drop/recreate indexes |
| File parsing | Sequential | Parallel (CPU cores) | ProcessPoolExecutor |
| DB operations | - | - | Single-threaded only |

## KEY_COMMANDS
```bash
# Development
lint: uv run ruff check chunkhound
typecheck: uv run mypy chunkhound
test: uv run pytest tests/
smoke: uv run pytest tests/test_smoke.py -v  # ALWAYS run before commits
format: uv run ruff format chunkhound

# Version management
update_version: uv run scripts/update_version.py X.Y.Z
sync_version: uv run scripts/sync_version.py

# Running
index: uv run chunkhound index [directory]
mcp_stdio: uv run chunkhound mcp stdio
mcp_http: uv run chunkhound mcp http --port 5173
```

## COMMON_ERRORS_AND_SOLUTIONS
- "database is locked": SerialDatabaseProvider not wrapping call
- "segmentation fault": Concurrent DB access attempted
- "Rate limit exceeded": Reduce embedding_batch_size or max_concurrent_batches
- "Out of memory": Reduce chunk_batch_size or file_batch_size
- JSON-RPC errors: Check for print() statements in mcp_server.py
- "unsupported operand type(s) for |: 'str' and 'NoneType'": Forward reference with | operator (remove quotes)

## DIRECTORY_STRUCTURE
```
chunkhound/
├── providers/         # Database and embedding implementations
├── services/          # Orchestration and batching logic
├── core/             # Data models and configuration
├── interfaces/       # Protocol definitions (contracts)
├── api/              # CLI and HTTP interfaces
├── mcp_server.py     # MCP stdio server
├── mcp_http_server.py # MCP HTTP server
├── database.py       # Legacy compatibility wrapper
└── CLAUDE.md files   # Directory-specific LLM context
```

## TECHNOLOGY_STACK
- Python 3.10+ (async/await patterns)
- uv (package manager - ALWAYS use this)
- DuckDB (primary) / LanceDB (alternative) 
- Tree-sitter (20+ language parsers)
- OpenAI/Ollama embeddings
- MCP protocol (stdio and HTTP)
- Pydantic (configuration validation)

## TESTING_APPROACH
- Smoke tests: MANDATORY before any commit (tests/test_smoke.py)
  - Module imports: Catches syntax/type annotation errors at import time
  - CLI commands: Ensures all commands at least show help
  - Server startup: Verifies servers can start without crashes
- Unit tests: Core logic (chunking, parsing)
- Integration tests: Provider implementations
- System tests: End-to-end workflows
- Performance tests: Batching optimizations
- Concurrency tests: Thread safety verification

## VERSION_MANAGEMENT
Single source of truth: chunkhound/version.py
Auto-synchronized to all components via imports
NEVER manually edit version strings - use update_version.py

## PUBLISHING_PROCESS
### Pre-release Checklist
1. Update version: `uv run scripts/update_version.py X.Y.Z`
2. Run smoke tests: `uv run pytest tests/test_smoke.py -v` (MANDATORY)
3. Prepare release: `./scripts/prepare_release.sh`
4. Test local install: `pip install dist/chunkhound-X.Y.Z-py3-none-any.whl`

### Dependency Locking Strategy
- `pyproject.toml`: Flexible constraints (>=) for library compatibility
- `uv.lock`: Exact versions for development reproducibility
- `requirements-lock.txt`: Exact versions for production deployment
- `prepare_release.sh` regenerates lock file with: `uv pip compile pyproject.toml --all-extras -o requirements-lock.txt`

### Publishing Commands
```bash
# Prepare release (includes lock file regeneration)
./scripts/prepare_release.sh

# Publish to PyPI (requires PYPI_TOKEN)
uv publish

# Verify published package
pip install chunkhound==X.Y.Z
chunkhound --version
```

### Release Artifacts
- `dist/*.whl`: Python wheel for pip install
- `dist/*.tar.gz`: Source distribution
- `dist/SHA256SUMS`: Checksums for verification
- `requirements-lock.txt`: Exact dependency versions

## PROJECT_MAINTENANCE
- Tickets: /tickets/ directory (active) and /tickets/closed/ (completed)
- No human editing expected - optimize for LLM modification
- All code patterns should be self-documenting with rationale
- Performance numbers justify architectural decisions
- Smoke tests: MANDATORY guardrails preventing import/startup failures
- Testing philosophy: Fast feedback loops for AI development cycles
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
test: uv run pytest tests/ -n auto
smoke: uv run pytest tests/test_smoke.py -v -n auto # ALWAYS run before commits
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
- Tree-sitter (24 language parsers: Python, JavaScript, TypeScript, JSX, TSX, Java, Kotlin, Groovy, C, C++, C#, Go, Rust, Bash, MATLAB, Makefile, Objective-C, PHP, Vue, Zig, JSON, YAML, TOML, Markdown)
- OpenAI/Ollama embeddings
- MCP protocol (stdio and HTTP)
- Pydantic (configuration validation)

## RERANKING_SUPPORT

### Overview
ChunkHound supports reranking for multi-hop semantic search, enabling more accurate search results by combining vector similarity with learned relevance scoring.

### Supported Formats
Two reranking API formats are supported with automatic format detection:

**1. Cohere Format** (Traditional format, widely supported)
- Request: `{"model": "...", "query": "...", "documents": [...], "top_n": 10}`
- Response: `{"results": [{"index": 0, "relevance_score": 0.95}, ...]}`
- Model: REQUIRED in request payload
- Use case: vLLM, Cohere API, OpenAI-compatible rerankers

**2. TEI Format** (Hugging Face Text Embeddings Inference)
- Request: `{"query": "...", "texts": [...]}`
- Response: `{"results": [{"index": 0, "score": 0.95}, ...]}`
- Model: Set at deployment time with `--model-id` flag (optional in request)
- Authorization: Supports `Authorization: Bearer <token>` header when TEI uses `--api-key`
- Use case: TEI servers (BAAI/bge-reranker-base, Alibaba-NLP/gte-reranker, etc.)

**Default:** `auto` mode - automatically detects format from first response (Cohere if `rerank_model` set, TEI otherwise)

### Configuration

**Minimal TEI Configuration (no model field):**
```json
{
  "embedding": {
    "provider": "openai",
    "base_url": "http://localhost:11434/v1",
    "model": "nomic-embed-text",
    "rerank_url": "http://localhost:8080/rerank",
    "rerank_format": "tei"
  }
}
```

**TEI with Optional Model (for documentation):**
```json
{
  "embedding": {
    "rerank_model": "BAAI/bge-reranker-base",
    "rerank_url": "http://localhost:8080/rerank",
    "rerank_format": "tei",
    "api_key": "tei-secret-key"
  }
}
```

**Cohere Format (requires model):**
```json
{
  "embedding": {
    "rerank_model": "rerank-english-v3.0",
    "rerank_url": "http://localhost:8001/rerank",
    "rerank_format": "cohere"
  }
}
```

**Auto-Detection (default):**
```json
{
  "embedding": {
    "rerank_format": "auto"
  }
}
```

### Format Selection Logic
1. **Explicit format** (`rerank_format: "tei"` or `"cohere"`): Use specified format
2. **Auto mode** (`rerank_format: "auto"`):
   - **First request**: Sends format based on config (Cohere if `rerank_model` set, TEI otherwise)
   - **Response parsing**: Detects actual format from response field names (`relevance_score` = Cohere, `score` = TEI)
   - **Subsequent requests**: Uses cached detected format for all future requests
   - **Important**: First request may fail if config-based guess doesn't match server format
   - **Recommendation**: Use explicit format for production to avoid first-request failures
3. **Format detection**: Based on response field names (`relevance_score` vs `score`)
4. **Thread safety**: Format caching is protected by async lock to prevent race conditions

### Validation Rules
- TEI format: `rerank_model` is OPTIONAL (model set at deployment)
- Cohere format: `rerank_model` is REQUIRED
- Both formats: `rerank_url` must be set (absolute or relative to `base_url`)
- Authorization: `api_key` added to `Authorization` header when set

### Common Patterns

**Dual-Endpoint (Ollama + TEI):**
```bash
# Embeddings from Ollama (port 11434)
# Reranking from TEI server (port 8080)
export CHUNKHOUND_EMBEDDING__BASE_URL=http://localhost:11434/v1
export CHUNKHOUND_EMBEDDING__MODEL=nomic-embed-text
export CHUNKHOUND_EMBEDDING__RERANK_URL=http://localhost:8080/rerank
export CHUNKHOUND_EMBEDDING__RERANK_FORMAT=tei
```

**TEI Deployment Example:**
```bash
docker run --gpus all -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-embeddings-inference:1.8 \
  --model-id BAAI/bge-reranker-base \
  --api-key my-secret-key
```

### Verifying Configuration

**Test TEI Endpoint:**
```bash
# Health check
curl http://localhost:8080/health

# Test rerank (TEI format)
curl -X POST http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"query": "python programming", "texts": ["def main():", "function test() {}"]}'

# Expected response:
# {"results": [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.12}]}
```

**Test Cohere Format (vLLM/Cohere):**
```bash
curl -X POST http://localhost:8001/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "rerank-model", "query": "test", "documents": ["doc1", "doc2"]}'

# Expected response:
# {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
```

**Verify ChunkHound Connection:**
```python
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider(
    base_url="http://localhost:11434/v1",
    model="nomic-embed-text",
    rerank_url="http://localhost:8080/rerank",
    rerank_format="tei"
)

# Test reranking
results = await provider.rerank("test query", ["doc1", "doc2", "doc3"])
print(f"Success! Got {len(results)} results")
```

### Implementation Details
- No batching for reranking (context-dependent, small document sets)
- Multi-hop search: 1 initial rerank + N expansion reranks (typically 0-3)
- Time limits: 5 second cap to prevent excessive API calls
- Graceful degradation: Falls back to similarity scores if reranking fails
- Format caching: Auto-detected format cached to avoid repeated detection
- Bounds checking: Returned indices validated against original document count
- Thread safety: Async lock protects format detection cache

### Error Scenarios and Recovery

**Server Not Available**
- **Error**: `httpx.ConnectError: Failed to connect to rerank service`
- **Cause**: Rerank server not running or incorrect URL
- **Recovery**: Start server or verify `rerank_url` configuration
- **Check**: `curl http://localhost:8080/rerank` should respond

**Format Mismatch**
- **Error**: Missing required fields in response (e.g., `rerank_model is required`)
- **Cause**: Server expects different format than configured
- **Recovery**: Set `rerank_format` to match server (TEI for TEI servers, Cohere for vLLM/Cohere)
- **Check**: Test endpoint with both formats to determine which works

**Malformed Response**
- **Error**: `Invalid rerank response: missing 'results' field`
- **Cause**: Server returned unexpected response structure
- **Recovery**: Check server logs, verify server is rerank-compatible
- **Validation**: ChunkHound validates all responses and skips invalid entries

**Out-of-Bounds Indices**
- **Behavior**: Invalid indices silently skipped, logged as warnings
- **Cause**: Buggy rerank server returning indices >= num_documents
- **Recovery**: Check server implementation, update if necessary
- **Protection**: Bounds checking prevents downstream crashes

**Timeout**
- **Error**: `Rerank request timed out after 30s`
- **Cause**: Rerank server slow or unresponsive
- **Recovery**: Increase timeout via `CHUNKHOUND_EMBEDDING__TIMEOUT`
- **Default**: 30 seconds for rerank requests

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
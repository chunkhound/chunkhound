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
- JSON-RPC errors: Check for print() statements in mcp_server/ (stdio.py, http_server.py, tools.py)
- "unsupported operand type(s) for |: 'str' and 'NoneType'": Forward reference with | operator (remove quotes)

## KNOWN_DEPRECATION_WARNINGS

### HDBSCAN + scikit-learn: force_all_finite Parameter (Non-Breaking)
**Warning Message:**
```
FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
  warnings.warn(
```

**Root Cause:**
- HDBSCAN 0.8.40 uses deprecated `force_all_finite` parameter in sklearn's `check_array()` function
- Appears in clustering service during Code Research tool execution
- Source: `/Users/ofri/Documents/GitHub/chunkhound/chunkhound/.venv/lib/python3.11/site-packages/hdbscan/hdbscan_.py:753,1211`

**Current Impact:**
- ⚠️ Warning only (does not break functionality)
- sklearn 1.7.2: Compatible but shows deprecation warning
- sklearn 1.8: Will become an error if HDBSCAN not updated

**Resolution Status:**
- Upstream issue: https://github.com/scikit-learn-contrib/hdbscan/issues/689 (Open)
- ChunkHound code: No changes required (correct API usage)
- Waiting for HDBSCAN upstream fix

**Action Required:**
- Monitor HDBSCAN releases for sklearn 1.8 compatibility
- Warning is non-breaking; safe to ignore until HDBSCAN upstream fix

## MIGRATION_NOTES

### Clustering Algorithm: K-means → HDBSCAN (Two-Phase Approach)

**What Changed:**
- **Old**: K-means clustering with predetermined k (calculated from token budget)
- **New**: Two-phase HDBSCAN clustering:
  - Phase 1: HDBSCAN discovers natural semantic clusters
  - Phase 2: Greedy grouping merges clusters to approach token budget

**Impact on Users:**
1. **Cluster Assignments**: Files may be grouped differently due to semantic clustering
2. **Cluster Counts**: Number of clusters may vary (HDBSCAN finds natural groups vs. fixed k)
3. **Outlier Detection**: HDBSCAN identifies semantically distant files as outliers, merging them into nearest clusters when possible
4. **Metadata Schema**: New fields added to clustering metadata:
   - `num_native_clusters`: Natural clusters discovered by HDBSCAN (Phase 1)
   - `num_outliers`: Files detected as outliers (noise points)
   - `num_clusters`: Final cluster count after merging (Phase 2) - **existing field**

**Benefits:**
- More semantic clustering (files grouped by meaning, not arbitrary k)
- Better handling of diverse codebases (natural cluster discovery)
- Outlier detection identifies files that don't fit semantically

**Performance Considerations:**
- HDBSCAN has O(n²) complexity for baseline implementation (vs. K-means O(n·k·d·iterations))
- For typical workloads (n<1000 files for synthesis), O(n²) is acceptable but untested
- Greedy merge phase is O(k³) typical case, O(k⁴) worst-case where k = num_native_clusters (typically <100)
- Performance testing recommended before production use at scale

**Backward Compatibility:**
- **API**: `ClusteringService.cluster_files()` signature unchanged
- **Metadata**: Existing fields (`num_clusters`, `total_files`, etc.) remain unchanged
- **Integration**: No changes required to synthesis_engine.py or other consumers

**Migration Path:**
- No action required for most users
- If you have custom code that inspects clustering metadata keys, update to handle new fields
- If synthesis quality changes, file a bug report with comparison results

## DIRECTORY_STRUCTURE
```
chunkhound/
├── providers/         # Database and embedding implementations
├── services/          # Orchestration and batching logic
├── core/              # Data models and configuration
├── interfaces/        # Protocol definitions (contracts)
├── api/               # CLI and HTTP interfaces
├── mcp_server/        # MCP server implementations
│   ├── stdio.py       # Stdio transport server
│   ├── http_server.py # HTTP transport server
│   ├── tools.py       # Unified tool registry (single source of truth)
│   ├── base.py        # Common server base class
│   └── common.py      # Shared utilities
├── database.py        # Legacy compatibility wrapper
└── CLAUDE.md files    # Directory-specific LLM context
```

## TECHNOLOGY_STACK
- Python 3.10+ (async/await patterns)
- uv (package manager - ALWAYS use this)
- DuckDB (primary) / LanceDB (alternative)
- Tree-sitter (27 language parsers: Python, JavaScript, TypeScript, JSX, TSX, Java, Kotlin, Groovy, C, C++, C#, Go, Rust, Haskell, Swift, Bash, MATLAB, Makefile, Objective-C, PHP, Vue, Zig, JSON, YAML, TOML, HCL, Markdown)
- Custom parsers (2 formats: TEXT, PDF)
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
- Response (two variants):
  - **Real TEI servers**: `[{"index": 0, "score": 0.95}, ...]` (bare array)
  - **Some proxies/mocks**: `{"results": [{"index": 0, "score": 0.95}, ...]}` (wrapped)
  - **Note**: ChunkHound automatically normalizes both variants to wrapped format internally
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
export CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE=32  # Match TEI server --max-batch-size limit
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

# Expected response (real TEI servers return bare array):
# [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.12}]
# Note: Some proxies may wrap as {"results": [...]} - ChunkHound handles both
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
- Automatic batch splitting for reranking (user-configurable via `rerank_batch_size`)
- Model-specific batch limits: Qwen3-8B (64), Qwen3-4B (96), Qwen3-0.6B (128), others (128 default)
- Bounded override pattern: User can set `rerank_batch_size`, but clamped to model caps for safety
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

**Batch Size Mismatch**
- **Error**: `Client error '413 Payload Too Large'` or `batch size X > maximum allowed batch size Y`
- **Cause**: ChunkHound model-default batch size exceeds server limit (e.g., Qwen3-8B sends 64, TEI limit is 32)
- **Recovery**: Set `CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE` to match server limit
- **Example**: `export CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE=32` for TEI servers
- **Note**: User override is bounded by model caps for safety (prevents OOM)

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
**Dynamic versioning via hatch-vcs** - version automatically derived from git tags
- Source of truth: Git tags (format: v4.0.1, v4.1.0b1, v4.1.0rc2)
- Generated file: chunkhound/_version.py (auto-created during build, gitignored)
- Import location: chunkhound/version.py (imports from _version.py or metadata)
- **PEP 440 compliant**: Supports pre-release versions (alpha, beta, rc)
- Create new version:
  - Release: `uv run scripts/update_version.py 4.1.0`
  - Beta: `uv run scripts/update_version.py 4.1.0b1`
  - Release candidate: `uv run scripts/update_version.py 4.1.0rc1`
  - Bump: `--bump major|minor|patch [prerelease]`
    - Example: `uv run scripts/update_version.py --bump minor b1` → creates v4.1.0b1
- NEVER manually edit version strings - create git tags instead

## PUBLISHING_PROCESS
### Pre-release Checklist
1. Create version tag: `uv run scripts/update_version.py X.Y.Z[{a|b|rc}N]` or `--bump major|minor|patch [prerelease]`
   - Release: `uv run scripts/update_version.py 4.1.0` → creates v4.1.0
   - Beta: `uv run scripts/update_version.py 4.1.0b1` → creates v4.1.0b1
   - RC: `uv run scripts/update_version.py 4.1.0rc1` → creates v4.1.0rc1
   - Bump to beta: `uv run scripts/update_version.py --bump minor b1` → creates v4.1.0b1
   - Version automatically derived from tag during build via hatch-vcs
2. Run smoke tests: `uv run pytest tests/test_smoke.py -v` (MANDATORY)
3. Prepare release: `./scripts/prepare_release.sh`
4. Test local install: `pip install dist/chunkhound-X.Y.Z-py3-none-any.whl`
5. Push tag: `git push origin vX.Y.Z[{a|b|rc}N]`

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
# Services Module Context

## MODULE_PURPOSE
Services layer orchestrates complex workflows by coordinating between providers, parsers, and storage layers.
This is where multi-step operations are broken into phases with proper error handling and batching.

## ARCHITECTURE_PATTERN

### Service Hierarchy
```python
BaseService               # Abstract base with config access
├── IndexingCoordinator  # Orchestrates full indexing workflows
├── EmbeddingService     # Manages embedding generation and batching
├── SearchService        # Coordinates search operations
├── ChunkCacheService    # Manages chunk diffing and caching
└── RealtimeIndexingService  # Handles incremental updates
```

### Design Principles
1. **Single Responsibility**: Each service has one clear workflow to coordinate
2. **Provider Agnostic**: Services work with any provider implementation
3. **Batch Optimization**: Services handle batching for performance
4. **Error Resilience**: Graceful degradation and fallback strategies

## INDEXING_COORDINATOR

### Workflow Phases
```python
# Phase 1: Discovery (NEW: Parallel or Sequential)
files = await _discover_files(directory, patterns, exclude_patterns)

# Phase 2: Parsing (Parallel via ProcessPoolExecutor)
results = await _parse_files_in_batches(files)

# Phase 3: Chunk Diffing (Smart Updates)
new_chunks, updated_chunks = _diff_chunks(existing, new)

# Phase 4: Embedding Generation (Batched)
embeddings = await _generate_embeddings_for_chunks(chunks)

# Phase 5: Storage (Single-Threaded via SerialDatabaseProvider)
await _store_chunks_and_embeddings(chunks, embeddings)
```

### PARALLEL_DISCOVERY (NEW)

#### When to Use
- **Enabled by default** (`parallel_discovery: true` in IndexingConfig)
- **Auto-activates** when >= `min_dirs_for_parallel` top-level directories (default: 4)
- **Auto-falls back** to sequential on errors or small directory structures

#### Performance Characteristics
| Repository Size | Directories | Expected Speedup | Best For |
|-----------------|-------------|------------------|----------|
| Small (<100 files) | <4 dirs | None (uses sequential) | Scripts, small projects |
| Medium (100-1K files) | 4-20 dirs | 1.5-3x | Most applications |
| Large (>1K files) | 20+ dirs | 2-5x | Enterprise monorepos |

#### Configuration
```python
# In IndexingConfig
parallel_discovery: bool = True  # Enable parallel mode
min_dirs_for_parallel: int = 4   # Threshold for activation
max_discovery_workers: int = 16  # Worker process limit
```

#### Architecture
```python
# Parallel Strategy:
1. Partition directory tree at top level
2. Spawn ProcessPoolExecutor workers for each subtree
3. Each worker traverses its subtree independently
4. Pre-load root .gitignore before spawning (inheritance)
5. Aggregate results from all workers
6. Scan root directory files in main thread
7. Sort and return combined file list

# Worker Isolation:
- Each worker has its own pattern cache
- No shared state between workers (avoid GIL contention)
- Workers must be module-level functions (picklable)
```

#### Fallback Strategy
```python
# Automatic fallback to sequential when:
1. parallel_discovery = False (explicit)
2. Directory has < min_dirs_for_parallel top-level dirs
3. Permission errors accessing directory
4. Any exception during parallel execution

# Error Handling:
- Worker errors logged with context (which directory failed)
- Aggregated error reporting (first 5 errors + count)
- Full traceback preserved for debugging (last 3 frames in warning, full in debug)
- Always falls back gracefully to sequential
```

#### Symbolic Link Handling
```python
# SECURITY: Symlinks are intentionally NOT followed (followlinks=False)
#
# Rationale:
- Prevents infinite loops from circular symlinks
- Avoids indexing same content multiple times
- Prevents traversal outside intended directory tree
- Consistent behavior in both parallel and sequential modes
#
# Implementation:
- os.walk() uses followlinks=False (default)
- Symlinks to directories are not traversed
- Symlinks to files are ignored (not indexed)
- Test coverage: test_symlink_loop_protection verifies safety
```

#### Race Condition Safety
```python
# ROBUSTNESS: Handles filesystem changes during traversal
#
# Protected scenarios:
1. Directory deleted between iterdir() and worker spawn
2. Directory changed to file during iteration
3. Subtree deleted while worker is processing
4. Permission changes during traversal
#
# Implementation:
- FileNotFoundError/NotADirectoryError caught in workers
- Errors logged but don't fail entire discovery
- Each worker returns (files, errors) tuple
- Main thread aggregates and logs all worker errors
```

## CRITICAL_CONSTRAINTS

### Thread Safety
- **Database operations**: ALWAYS single-threaded (wrapped in SerialDatabaseProvider)
- **File parsing**: Parallel safe (ProcessPoolExecutor, CPU-bound)
- **Directory discovery**: Parallel safe (NEW: read-only I/O operations)
- **Embedding API calls**: Parallel safe (async batched requests)

### Batching Requirements
- **Embeddings**: MANDATORY batching (100x performance difference)
- **Database inserts**: MANDATORY batching (prevents index thrashing)
- **File parsing**: Parallel batching (utilizes CPU cores)
- **Directory discovery**: Parallel mode (utilizes I/O parallelism)

### Performance Numbers
```python
# Directory Discovery (2000 files, 20 dirs):
Sequential: 2.5s
Parallel:   0.8s
Speedup:    3.1x (on 8-core system)

# Memory Optimization (heapq.merge vs full sort):
# For N files across K workers:
- Full sort: O(N log N) comparisons, O(N) memory
- Heap merge: O(N log K) comparisons, O(K) memory for merge
- 1M files, 16 workers: ~4x fewer comparisons, same memory footprint

# File Parsing (1000 Python files):
Sequential: 45s
Parallel (8 workers): 8s
Speedup:    5.6x

# Embedding Generation (1000 chunks):
Unbatched (1 at a time): 100s
Batched (100 per request): 1s
Speedup:    100x
```

## EMBEDDING_SERVICE

### Provider-Aware Concurrency
```python
# Auto-detects optimal concurrency from provider:
OpenAI: 8 concurrent batches (rate limit: 500 req/min)
VoyageAI: 40 concurrent batches (higher limits)
Ollama: 4 concurrent batches (local resource limit)

# Configuration:
max_concurrent_batches: Optional[int]  # Auto-detected if None
embedding_batch_size: int = 100        # Chunks per API call
```

### Batching Strategy
```python
# Smart chunk batching:
1. Split chunks into batches of `embedding_batch_size`
2. Create semaphore with `max_concurrent_batches` slots
3. Process batches concurrently (respects rate limits)
4. Aggregate results preserving chunk order
5. Return all embeddings together for atomic DB insert
```

## CHUNK_CACHE_SERVICE

### Smart Diffing
```python
# Avoids re-embedding unchanged code:
1. Hash each chunk content
2. Compare with existing chunk hashes
3. Categories:
   - Unchanged: Skip (reuse existing embeddings)
   - Modified: Re-embed (content changed)
   - New: Embed (doesn't exist)
   - Deleted: Remove (no longer in file)

# Performance Impact:
File with 100 chunks, 2 changed:
- Without diffing: Generate 100 embeddings (~1s)
- With diffing: Generate 2 embeddings (~0.02s)
- Speedup: 50x for incremental updates
```

## SEARCH_SERVICE

### Multi-Stage Search
```python
# Stage 1: Vector similarity (DuckDB HNSW or LanceDB)
top_k_chunks = vector_search(query_embedding, k=100)

# Stage 2: Optional re-ranking (if configured)
reranked = rerank_provider.rerank(query, top_k_chunks, top_n=10)

# Stage 3: Context formatting
formatted_results = format_for_display(reranked or top_k_chunks)
```

## COMMON_PATTERNS

### Service Initialization
```python
class MyService(BaseService):
    def __init__(self, db_provider, config, additional_deps):
        super().__init__(config)  # Stores config
        self.db = db_provider     # Provider dependency
        self.deps = additional_deps
```

### Error Handling Pattern
```python
try:
    result = await primary_strategy()
except ProviderError as e:
    logger.warning(f"Primary failed: {e}, trying fallback")
    result = await fallback_strategy()
except Exception as e:
    logger.error(f"Critical failure: {e}")
    raise ServiceError(f"Operation failed: {e}") from e
```

### Batching Pattern
```python
async def process_items(items: list[T]) -> list[R]:
    batch_size = self.config.batch_size
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

    results = []
    for batch in batches:
        batch_results = await process_batch(batch)
        results.extend(batch_results)

    return results
```

## TESTING_STRATEGY

### Service Tests
- **Unit tests**: Mock provider interfaces, test service logic
- **Integration tests**: Real providers, test full workflows
- **Performance tests**: Measure batching speedup, parallel gains
- **Regression tests**: Ensure optimizations don't break correctness

### Performance Benchmarks
```bash
# Run discovery performance tests
uv run pytest tests/test_performance_discovery.py -v

# Run with timing output
uv run pytest tests/test_performance_discovery.py -v -s
```

## COMMON_PITFALLS

- **DONT**: Call database directly from service (use provider methods)
- **DONT**: Make unbatched API calls in loops (batch them)
- **DONT**: Share mutable state between parallel workers
- **DONT**: Assume parallel is always faster (measure and configure)
- **DO**: Use SerialDatabaseProvider wrapper for all DB access
- **DO**: Pre-load shared data before spawning workers
- **DO**: Log performance metrics for optimization
- **DO**: Provide fallback strategies for robustness

## MAINTENANCE_NOTES

### Adding New Service
1. Inherit from `BaseService`
2. Accept providers and config in `__init__`
3. Implement workflow methods (async preferred)
4. Add comprehensive error handling
5. Document performance characteristics
6. Write integration tests

### Optimizing Workflows
1. Identify bottlenecks with profiling
2. Apply batching where possible
3. Consider parallelism for CPU/I/O bound work
4. Measure before/after performance
5. Document optimization rationale
6. Add regression tests

### Configuration Guidelines
- **User-facing**: Add to config with CLI argument
- **Internal tuning**: Add to config as internal setting
- **Performance**: Document impact and recommended values
- **Defaults**: Choose safe values, allow override

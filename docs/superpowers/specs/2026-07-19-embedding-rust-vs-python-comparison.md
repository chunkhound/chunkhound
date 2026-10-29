# Embedding Pipeline: Python vs Rust Design Comparison

**Date:** 2026-07-19
**Purpose:** Gap analysis between the current Python embedding pipeline and the planned Rust embed stage.

---

## 1. Summary Table

| # | Aspect | Python Behavior | Rust Design | Same/Different | User-Visible Change? |
|---|---|---|---|---|---|
| 1 | **API client** | Provider SDKs (`openai.AsyncOpenAI`, `voyageai.Client`) | Raw HTTP via `reqwest::blocking::Client` | Different | No — request bodies are equivalent |
| 2 | **Concurrency** | `asyncio` with `Semaphore(max_concurrent_batches)`; 8 (OpenAI) or 40 (VoyageAI) concurrent | Single `std::thread`, sequential | **Different** | Yes — 8–40× slower throughput |
| 3 | **Dedup mechanism** | DB query per batch: `_filter_existing_embeddings` → SQL `get_existing_embeddings` | In-memory bloom filter, persistent, 1% FPR | **Different** | Yes — faster dedup, 1% false-skip rate |
| 4 | **Dedup key** | `(chunk_id INTEGER, provider, model)` — keyed by chunk identity | `(content_hash, provider, model, output_dims)` — keyed by content | **Different** | Yes — same text under different chunk_ids gets deduped in Rust |
| 5 | **Token estimation** | tiktoken for official OpenAI + Azure; `text.len() // 3` for others | `text.len() / 3` only | Different | Minimal — most chunks won't hit the limit |
| 6 | **Batch accumulation** | Creates all batches upfront, then dispatches concurrently | Accumulates one batch at a time, flushes when full | Different | No — output embeddings are identical |
| 7 | **MAX_CHUNKS_PER_BATCH** | 300 | 300 | Same | No |
| 8 | **OpenAI max_batch_size** | 2048 (implicit from SDK) | 2048 (const) | Same | No |
| 9 | **VoyageAI max_batch_size** | 1000 (from model config `max_texts_per_batch`) | 1000 (const) | Same | No |
| 10 | **Retry attempts** | 3 (configurable) | 3 (configurable) | Same | No |
| 11 | **Retry backoff** | Provider-specific: OpenAI uses `Retry-After` header + jitter; VoyageAI uses category backoffs (30s rate limit, 10s upstream timeout, exp for network) | Uniform exponential: base 1s, max 60s, jitter. `Retry-After` from 429 header honored, no compounding | **Different** | Minimal — retries succeed in both cases |
| 12 | **Error classification** | Granular JSON body parsing (OpenAI SDK exceptions, VoyageAI `_classify_voyageai_error`) | HTTP status code only (with context-length body check) | **Different** | No — all retryable errors trigger retry |
| 13 | **Empty text handling** | Replaced with `"[EMPTY]"` placeholder | Replaced with `"[EMPTY]"` placeholder | Same | No |
| 14 | **Empty/whitespace text** | Silently skipped by `validate_text_input` (VoyageAI) or replaced with `[EMPTY]` (OpenAI) | Skipped when `text.trim().is_empty()` | **Subtly different** | Yes — empty texts are skipped vs replaced with placeholder |
| 15 | **Server-side dimensions param** | `build_dimension_request_param()` → OpenAI: `"dimensions"`, VoyageAI: `"output_dimension"`. Validated against model whitelist | Same params, validated only by `dims <= native_dims` | **Different** | Minimal — invalid dims fail at API vs Rust init |
| 16 | **Client-side truncation** | `apply_client_side_truncation()` (slice + L2-normalize). Validates `output_dims <= raw_dims` | `l2_normalize(&v[..out])` | Same | No |
| 17 | **Output dims validation** | Model whitelist (OpenAI matryoshka range 1..native, VoyageAI supported list). Runtime discovery for unknown models | Bounds only: `dims <= native_dims` | **Different** | Minimal — whitelist violations fail at API |
| 18 | **Azure detection** | `is_azure_openai_endpoint()` on `azure_endpoint` or `base_url` containing `openai.azure.com` | Same: `azure_endpoint.is_some() || base_url.contains("openai.azure.com")` | Same | No |
| 19 | **Azure auth header** | `api-key` header via `AsyncAzureOpenAI` client | `api-key` header on reqwest request | Same | No |
| 20 | **Azure api-version** | Query param via `AsyncAzureOpenAI` client | Query param appended to URL | Same | No |
| 21 | **VoyageAI truncation** | `truncation: True` always in request body | `truncation: true` always in request body | Same | No |
| 22 | **VoyageAI input_type** | `"document"` in request body | `"document"` in request body | Same | No |
| 23 | **Qwen model support** | Full QWEN_MODEL_CONFIG with batch sizes, token limits, rerank limits | Not supported (not in model config table) | **Missing** | Yes — Qwen users lose embeddings |
| 24 | **Dimension discovery** | Runtime `_discovered_native_dims` for unknown/custom models | None — unknown models get fallback `native_dims=1536` or `1024` | **Missing** | Yes — custom-endpoint dims must match fallback |
| 25 | **Response index sorting** | Manual sort by `data.index` before processing | Manual sort by `data.index` before processing | Same | No |
| 26 | **Vector count validation** | Checks `len(embedding_results) == len(chunk_ids)` | Checks `vectors.len() == texts.len()` | Same | No |
| 27 | **Embed store** | DB insert (`insert_embeddings_batch`) per batch. Transaction recovery on abort | Passed via channel to DB Writer stage | Different | No — storage result is equivalent |
| 28 | **Progress bar** | Rich progress bar with speed calculation | None | **Missing** | Yes — no visual progress during embedding |
| 29 | **Logging** | loguru (Python logging) | `log` crate → `pyo3-log` → Python logging | Different impl, same output | No |
| 30 | **Pipeline shutdown** | Exception propagation through asyncio tasks | `Arc<AtomicBool>` cancellation flag, safety-net drain | Different | No — both clean up properly |
| 31 | **Batch token limit check** | Pre-flight: `safe_limit = max_tokens * 0.80`. Splits oversized single texts | Pre-flight: per-chunk `estimate_tokens() > max_tokens_per_chunk()` → skip. No batch token budget | **Different** | Minor — large batches hitting provider token budget would hit context-length error and split in Rust |
| 32 | **VoyageAI batch token budget** | `max_tokens_per_batch` (120k–1M per model) enforced in `embed_batch()` | Not enforced (listed as future work §13) | **Missing** | Minor — VoyageAI batches could exceed token budget, hitting API error then retry |
| 33 | **Health check** | `health_check()` with embed_single("test") | None | **Missing** | No — health check CLI would fail |
| 34 | **Reranking** | Full rerank support (HTTP + SDK) | Out of scope (embed stage only) | N/A | N/A |

---

## 2. Detailed Comparison

### 2.1 Embedding Provider Interface

**Python** (`chunkhound/interfaces/embedding_provider.py`):
- Protocol-based (`class EmbeddingProvider(Protocol)`) with ~30+ methods
- Async throughout (`async def embed()` etc.)
- Sub-protocols: `APIEmbeddingProvider`, `LocalEmbeddingProvider`
- Provider-specific methods: `get_max_tokens_per_batch()`, `get_max_documents_per_batch()`, `get_recommended_concurrency()`, `supports_reranking()`, `rerank()`, `health_check()`, `validate_api_key()`
- Configuration methods: `update_config()`, `get_supported_distances()`, `get_optimal_batch_size()`

**Rust** (§2):
- Trait-based (`trait EmbeddingProvider`) with 6 required methods + 1 default
- Synchronous (`fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>`)
- No sub-traits
- Methods: `embed_batch`, `max_batch_size`, `name`, `model`, `dimensions`, `max_tokens_per_chunk`, `recommended_concurrency`
- No config update, no health check, no reranking

**Impact:** The trait is narrower but covers the embed stage requirements exactly. Missing methods are not needed for the embed pipeline stage. The sync vs async difference is structural — Python wraps provider calls in `asyncio.to_thread()` for blocking SDKs, while Rust runs in its own `std::thread`. No user-visible behavior change for the core embed operation.

### 2.2 Provider Implementations

#### OpenAI

**Python** (`chunkhound/providers/embeddings/openai_provider.py`):
- Uses `openai.AsyncOpenAI` SDK (or `AsyncAzureOpenAI` for Azure)
- Lazy client initialization (`_ensure_client()` called on first `embed()`)
- Model config at `OPENAI_MODEL_CONFIG: dict[str, OpenAIModelConfig]` (L:168)
  - `text-embedding-3-small`: dims=1536, native_dims=1536, max_tokens=8191, matryoshka=True, min_dims=1
  - `text-embedding-3-large`: dims=3072, native_dims=3072, max_tokens=8191, matryoshka=True, min_dims=1
  - `text-embedding-ada-002`: dims=1536, native_dims=1536, max_tokens=8191, matryoshka=False
- Qwen models via `QWEN_MODEL_CONFIG` (L:47–131) — full config with per-model batch sizes, token limits, rerank limits
- `_build_embedding_request_kwargs()` (L:391) — adds `"dimensions"` only for matryoshka models at official endpoints
- `_trust_runtime_output_dims()` (L:356) — custom endpoints skip model whitelist
- Dimension validation: `validate_runtime_output_dims_config`, `validate_positive_output_dims`, `validate_embedding_dims`
- Client-side truncation: `_validate_client_side_truncation_runtime_config` (L:431)
- Response order: extracts by `data.index`, validates no `None` slots

**Rust** (§3.1):
- Uses `reqwest::blocking::Client`
- Direct construction (no lazy init needed — single thread owns the client)
- Model config at `OPENAI_MODEL_CONFIG` (phf map):
  - Same three models with same dims/tokens/matryoshka
  - No Qwen models
- Request building: `json!({ "model": ..., "input": texts })`, adds `"dimensions"` for non-client_side_truncation and matryoshka models
- Azure: `api-key` header, `api-version` query param
- Response order: `data.sort_by_key(|d| d.index)` then collect
- Client-side truncation: `l2_normalize(&v[..out])`

**Differences:**
- **Qwen models:** Python supports them; Rust does not. This is a clear gap — Qwen users would not be able to use the Rust pipeline.
- **Dimension validation scope:** Python validates against model-specific whitelists; Rust validates only `dims <= native_dims`. This means invalid dims would be caught at API request time (400 BadRequest) instead of at init time.
- **Official endpoint dimension gating:** Python checks `is_official_openai_endpoint` and `matryoshka` flags before sending the `dimensions` param; Rust checks `!self.client_side_truncation && self.model_info.matryoshka`. The Rust logic is actually simpler and correct — it sends `dimensions` only when both matryoshka is supported and truncation is server-side.
- **Runtime dimension discovery:** Python caches `_discovered_native_dims` from untruncated API responses for custom models; Rust has no such mechanism. Unknown models always fall back to 1536 dims in Rust.

#### VoyageAI

**Python** (`chunkhound/providers/embeddings/voyageai_provider.py`):
- Uses `voyageai.Client` SDK
- Model config at `VOYAGE_MODEL_CONFIG` (L:60–125):
  - 9 models (voyage-3-large, voyage-code-3, voyage-finance-2, voyage-law-2, voyage-multilingual-2, voyage-large-2-instruct, voyage-3.5, voyage-2, voyage-3.5-lite)
  - Each with: `max_tokens_per_batch`, `max_texts_per_batch`, `context_length`, `dimensions`, `default_dimension`
- `DEFAULT_UNKNOWN_MODEL_CONFIG` for custom/unknown models (L:129–135)
- `max_texts_per_batch=1000` for all models
- `max_tokens_per_batch` varies: 120k, 320k, or 1M
- `context_length` varies: 4k, 16k, or 32k
- `_embed_single_batch_locked()` (L:365): builds request with `truncation=True`, `input_type="document"`, optional `output_dimension`
- Dimension validation: model whitelist for known models, runtime discovery for custom
- Concurrency: 40 for official API, 1 for custom endpoints (due to HTTP 424)

**Rust** (§3.2):
- Uses `reqwest::blocking::Client`
- Model config at `VOYAGE_MODEL_CONFIG` (phf map):
  - 9 models — same list, slightly different field structure
  - Each with: `native_dims`, `max_tokens_per_chunk`, `max_tokens_per_batch`, `supported_dimensions`
  - `native_dims` always 1024
  - `max_tokens_per_chunk` matches Python's `context_length`
- `MAX_BATCH = 1000` (matches Python)
- Request: `truncation: true`, `input_type: "document"`, optional `output_dimension`
- Dimension validation: bounds only, no model whitelist
- No runtime dimension discovery
- No `max_tokens_per_batch` enforcement (future work §13)

**Differences:**
- **Context length vs max_tokens_per_chunk:** Python's `context_length` is per-text; Rust's `max_tokens_per_chunk` is the same concept but named differently. Values match.
- **Per-batch token budget enforcement:** Python enforces `max_tokens_per_batch` in `embed_batch()` by tracking current_tokens; Rust pre-flight only checks per-chunk token limits. A batch of many medium-length texts could exceed the VoyageAI 120k/320k/1M token budget in Rust, triggering a context-length API error → retry → batch split. This is less efficient but ultimately produces the same result.
- **supported_dimensions:** Python exposes the exact list (e.g., `[256, 512, 1024, 2048]`); Rust uses `&'static [usize]` but the trait doesn't expose supported_dimensions — only `dimensions()` returns the current effective dims. This means Rust cannot programmatically answer "which dims are valid?" but the API rejects invalid ones.

### 2.3 Embedding Service / Batching

**Python** (`chunkhound/services/embedding_service.py`):
- `_create_token_aware_batches()` (L:641): Creates ALL batches upfront
  - `MAX_CHUNKS_PER_BATCH = 300` (L:660)
  - `safe_limit = max_tokens * 0.80`
  - Iterates through ALL chunk_data, splitting into batches
  - Returns `list[list[tuple[ChunkId, str]]]`
- `_generate_embeddings_in_batches()` (L:476): Processes batches with concurrent dispatch
  - `asyncio.Semaphore(max_concurrent_batches)` controls concurrency
  - All batch tasks created via `asyncio.gather(*tasks)`
  - Each batch: `embed()` → `insert_embeddings_batch()`
  - Batch splitting on token-limit errors, depth-limited to 3
  - Progress bar updates with speed calculation
- `_filter_existing_embeddings()` (L:415): DB query to find which chunks already have embeddings
  - Calls `self._db.get_existing_embeddings(chunk_ids, provider, model)`
  - Filters out empty/whitespace-text chunks after normalization

**Rust** (§5):
- Accumulates one batch at a time in `EmbedBuffer.batch`
- `MAX_CHUNKS_PER_BATCH = 300`
- `effective_batch_size = min(config.batch_size, provider.max_batch_size(), MAX_CHUNKS_PER_BATCH)`
- Flush triggers:
  - `batch.len() >= effective_batch_size`
  - Channel drain (end of input)
  - Cancellation
- `flush_batch()`: extracts texts → `embed_with_retry()` → scatter vectors back to `pending_files` → emit completed files
- Context-length split: halve batch, retry each half
- Bloom filter for dedup (no DB query)

**Differences:**
- **Batch creation strategy:** Python creates all batches upfront (knowing total count for progress); Rust accumulates and flushes incrementally. Rust cannot show "X of Y batches complete" progress.
- **Concurrency:** Python dispatches multiple embed calls concurrently; Rust is sequential. This is the main throughput difference.
- **Dedup integration:** Python queries DB for existing embeddings (I/O per batch set); Rust uses bloom filter (pure memory, O(1) per chunk). The bloom filter has a 1% false positive rate — ~1 in 100 chunks that SHOULD be embedded will be skipped.
- **Progress display:** Python has Rich progress bars; Rust has none.
- **Error handling scope:** Python catches per-batch errors and continues (`return 0`); Rust's `flush_batch()` propagates errors, which triggers pipeline shutdown. This is a behavioral difference — in Python, a single failed batch doesn't halt the entire embedding run; in Rust, a fatal embed error halts the pipeline.

### 2.4 Dedup (Bloom Filter vs DB Query)

This is the most significant architectural difference.

**Python** (`embedding_service.py:415` `_filter_existing_embeddings`):
```python
existing_chunk_ids = self._db.get_existing_embeddings(
    chunk_ids=[int(cid) for cid in chunk_ids],
    provider=provider_name,
    model=model_name,
)
```
- Query: "which of these chunk_ids already have embeddings in the DB?"
- Exact dedup: no false positives, no false negatives
- Cost: SQL query per batch (indexed on chunk_id, so fast)
- Key: `(chunk_id, provider, model)` — if a chunk is re-parsed and gets a NEW chunk_id but same content, it will be re-embedded
- Dimension-aware: the DB query can filter by dims if needed

**Rust** (§4):
```
bloom_key = format!("{content_hash}:{provider}:{model}:{dims}")
bloom.contains(&key)
```
- In-memory bloom filter, O(1) per chunk
- 1% false positive rate — 1 in 100 eligible chunks skipped per run
- Key: `(content_hash, provider, model, output_dims)` — content-based, so re-parsed identical text is deduped
- Persisted to `.chunkhound/db/embeddings.bloom`
- Populated from DB on startup via `db.populate_bloom()`

**Impact:**
- **Performance:** Bloom filter is MUCH faster for large DBs (no SQL round-trip per batch).
- **Accuracy:** 1% false-positive rate means some chunks won't get embedded on first run. These will be caught on the next indexing run since the bloom filter checks content_hash.
- **Content-based dedup:** Python's chunk_id-based dedup means if the parser produces a different chunk_id for the same text (e.g., after a re-parse), the text gets re-embedded. Rust's content-hash-based key prevents this.
- **Dimension-aware:** Rust includes `output_dims` in the bloom key, so switching dims triggers re-embedding. Python's DB query would also detect this (different dims = different table or row).

### 2.5 Retry & Error Handling

**Python OpenAI** (`openai_provider.py:_embed_batch_internal` at L:1268):
- Catches `openai.RateLimitError`:
  - Extracts `Retry-After` from response headers
  - Falls back to `x-ratelimit-reset-requests` header
  - Falls back to regex parsing of error message
  - Falls back to exponential backoff
  - Caps at 120s
  - Jitter: `random.uniform(0, min(retry_after * 0.1, 5.0))`
- Catches `openai.BadRequestError`:
  - Checks for "maximum context length" + "tokens" → token limit handling
  - Checks for "tokens" + "max" + "per request" → token limit handling
  - Both trigger `handle_token_limit_error()` which splits batch and retries
- Catches `openai.APITimeoutError` / `openai.APIConnectionError`:
  - Fixed `retry_delay` (1s)
  - Logs connection details
- Non-retryable: `EmbeddingProviderError` (domain errors), unknown exceptions

**Python VoyageAI** (`voyageai_provider.py:_classify_voyageai_error` at L:140):
- Category classification: `"rate_limit"`, `"upstream_timeout"`, `"network"`
- Category-specific base backoffs:
  - rate_limit: 30s
  - upstream_timeout: 10s
  - network: `self._retry_delay` (1s)
- Exponential multiplier: `base_delay * (2**attempt)`

**Rust** (§6):
- `classify_response()` based on HTTP status:
  - 401/403 → `Auth` (fatal)
  - 429 → `RateLimited { retry_after }` (retryable, extracts Retry-After header)
  - 400–499 → checks body for "context length" → `ContextLengthExceeded` (retryable with split) or `BadRequest` (fatal)
  - 500–599 → `Provider` (retryable)
  - Other → `Http` (retryable)
- `RetryPolicy`:
  - `max_attempts: 3`, `base_delay: 1s`, `max_delay: 60s`, `jitter: true`
- `embed_with_retry()`:
  - Exponential backoff: `delay = (delay * 2).min(policy.max_delay)`
  - 429 with `Retry-After`: uses that value, no compounding
  - Jitter: `rand::thread_rng().gen_range(0.0..0.25)` → 0–25% of delay
- `ContextLengthExceeded` → propagate for `flush_split_batch()`

**Differences:**
- **Error classification granularity:** Python extracts `Retry-After` from headers, error messages, and regex; Rust only reads the `retry-after` header (no body/message parsing). The Rust approach covers the standard case; Python covers edge cases where headers are missing.
- **VoyageAI category backoffs:** Python uses 30s base for rate limits; Rust uses uniform 1s base (exponential from there). So VoyageAI rate limits recover faster in Python (30s → 60s → 120s) vs Rust (1s → 2s → 4s). Actually wait — the Rust exponential goes 1s → 2s → 4s, which is much faster than 30s → 60s. But Python's 30s is actually MORE conservative because VoyageAI's rate-limit window is 60s, so Python waits for the window to drain before retrying. Rust's aggressive retry would keep hitting 429s. This is a potential **issue** — Rust may waste retry attempts by retrying too quickly after VoyageAI rate limits.
- **Retry-after not compounding:** Both are correct — 429s set `Retry-After`, non-429s use exponential backoff. But the spec says "does not compound" which is correct for 429s but the current Rust code actually does exponential from 1s for non-429 retries.
- **Token limit handling:** Python's `handle_token_limit_error` can split into multiple chunks based on token ratio; Rust always splits in half. Python's approach is more efficient for large batches.

⚠️ **Issue found:** The Rust design spec's `embed_with_retry` shows that for `RateLimited { retry_after }`, it uses `ra.to_owned()` as the wait and does NOT advance the exponential delay. But for HTTP 429 with NO `retry_after` header, `RateLimited { retry_after: None }` just uses `delay` (the exponential value). The Python code would regex-parse the error body for a delay hint in this case. Rust does not do this body regex parsing. If a provider returns 429 without a `Retry-After` header, Rust would retry with 1s → 2s → 4s while Python would parse the body and wait longer.

### 2.6 Token Estimation

**Python** (`chunkhound/core/utils/token_utils.py`):
- `estimate_tokens()` dispatches based on provider:
  - OpenAI official/Azure → `tiktoken.encoding_for_model(model)` → exact count (`enc.encode(text, disallowed_special=())`)
  - OpenAI official fallback → `tiktoken.get_encoding("cl100k_base")`
  - OpenAI custom/compatible → `text.len() // EMBEDDING_CHARS_PER_TOKEN` (3)
  - VoyageAI → `text.len() // EMBEDDING_CHARS_PER_TOKEN` (3)
  - Unknown → `text.len() // DEFAULT_CHARS_PER_TOKEN` (3.5)
- tiktoken failures cached with `functools.cache` to avoid repeated download attempts
- `EMBEDDING_CHARS_PER_TOKEN = 3` (measured empirically)
- `LLM_CHARS_PER_TOKEN = 4`
- `DEFAULT_CHARS_PER_TOKEN = 3.5`

**Rust** (§6.5):
- `estimate_tokens(text) = text.len() / EMBEDDING_CHARS_PER_TOKEN` where `EMBEDDING_CHARS_PER_TOKEN = 3`
- Single formula, no provider-specific logic
- Spec acknowledges this: "Actual token counting (tiktoken) is deferred to provider impl if needed" and "if estimate_tokens produces too many false skips" → future work

**Impact:**
- `text.len() / 3` is slightly conservative (underestimates tokens → smaller batches) compared to tiktoken which is exact
- For code/text, the 3:1 ratio is empirically accurate enough that false skips are rare
- The only time this matters is for chunks approaching `max_tokens_per_chunk` (8191 for OpenAI, 32000 for VoyageAI) — tiktoken might say a chunk is 7800 tokens (safe) while `len()/3` says 9000 (skip). In practice, very few chunks are >24,000 chars.
- **Not a significant behavior change** for typical codebases.

### 2.7 Dimension / Matryoshka Handling

Both implementations follow the same fundamental pattern:

```
output_dims set?
├─ client_side_truncation?
│   ├─ Yes → Request FULL native dims, truncate + L2-normalize client-side
│   └─ No  → Send "dimensions"/"output_dimension" param to API
└─ Not set → Request native dims
```

**Python additions that Rust lacks:**
1. **Model whitelist validation:** Python checks `output_dims in supported_dimensions` for known models at init time. Rust checks `dims <= native_dims` at init time. Invalid dims for a model that doesn't support truncation (e.g., `voyage-finance-2` with `output_dims=256`) would be rejected by Python at init time but by Rust's API request (400) at runtime.
2. **Runtime dimension discovery:** Python's `_discovered_native_dims` cache learns the actual dimension from the first untruncated API response. This is essential for custom/unknown models where the static config has no dimensions info. Rust sets `native_dims=1024` or `1536` as fallbacks.
3. **`supported_dimensions` property:** Python exposes `range(1, native+1)` for client_side_truncation, model whitelist otherwise. Rust's trait has no `supported_dimensions` — only `dimensions()`.

**Impact:**
- For **known models** (all OpenAI, all VoyageAI official): behavior is identical. The Rust bounds check is sufficient because API will reject invalid dims, and the error message will be clear.
- For **custom models**: Rust's lack of dimension discovery means custom models must use the fallback dims (or be explicitly configured with the right `output_dims`). If a custom model has 4096 native dims, Rust won't discover this — it will use 1536 (or 1024 for VoyageAI) and vectors won't match.
- **Client-side truncation** behavior is identical: same slice+L2-normalize logic.

### 2.8 Configuration

**Python** (`chunkhound/core/config/embedding_config.py`):
- `EmbeddingConfig(BaseSettings)` with Pydantic Settings
- `env_prefix="CHUNKHOUND_EMBEDDING_"`, `env_nested_delimiter="__"`
- 30+ fields with validators
- `field_validator` for: `rerank_batch_size`, `output_dims`, `model` (typo fixes), `base_url` (normalization)
- `model_validator` for: `client_side_truncation requires output_dims`, `rerank_config`, `azure_config`
- `load_from_env()` with multiple env var patterns
- `extract_cli_overrides()` for CLI argument integration
- `get_provider_config()` to produce provider-ready dict

**Rust** (§2.4):
- `EmbedConfig` struct with 13 fields
- `from_pydict()` to construct from Python dict
- No validation in Rust — trusts Python Pydantic validation

**Impact:** None. Rust uses Python as the validation layer. The same Pydantic validation runs regardless of whether the pipeline is Python or Rust. The `from_pydict` pattern reuses the Python-validated config.

### 2.9 Logging

**Python:**
- `loguru.logger` throughout
- Levels: `logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()`
- Context-rich: `logger.error(f"[OpenAI-Provider] OVERSIZED CHUNKS FOUND:\n" + ...)`
- Progress: Rich progress bars with speed display
- Debug file: `CHUNKHOUND_DEBUG_FILE` env var writes to `/tmp/chunkhound_debug.log`

**Rust** (§12):
- `log` crate macros → `pyo3-log` bridge → Python `logging` module
- Levels: `log::debug!()`, `log::info!()`, `log::warn!()`, `log::error!()`
- Inherits Python log configuration: `CHUNKHOUND_DEBUG`, `CHUNKHOUND_DEBUG_FILE`, `--debug`, `--verbose`, MCP mode
- Thread safety: `pyo3-log` acquires GIL per log call; embed stage runs in `py.allow_threads()` so brief GIL re-acquisition
- Log messages: "Flushing batch" (DEBUG), "Embed API retry" (WARN), "Embed API failed after 3 attempts" (ERROR), "Skipping oversized chunk" (WARN), "Embed stage complete" (INFO)

**Impact:** Minimal. Log output routes through the same Python logging infrastructure. The message content differs slightly (Python is more verbose) but all important events are covered. No progress bar in Rust.

### 2.10 Pipeline Integration

**Python** (IndexingCoordinator → EmbeddingService):
```
IndexingCoordinator._generate_embeddings(chunk_ids, chunks)
  └─ EmbeddingService.generate_embeddings_for_chunks(chunk_ids, chunk_texts)
       ├─ _filter_existing_embeddings() → DB query
       ├─ _create_token_aware_batches() → create all batches
       └─ _generate_embeddings_in_batches()
            └─ asyncio.gather(process_batch() * N) with Semaphore(N)
                 └─ _embedding_provider.embed(texts) → API call
                 └─ _db.insert_embeddings_batch(embeddings_data) → DB insert
```
- Async concurrent dispatch with Semaphore
- Per-batch: embed API call → DB insert
- DB inserts happen interleaved with API calls (pipeline-style)
- Errors: per-batch catch → `return 0`, other batches continue

**Rust** (§1):
```
Parser (rayon) → crossbeam channel → Embed (1 thread) → crossbeam channel → DB Writer (1 thread)
```
- Sequential in the embed stage
- Chunk arrives → bloom check → accumulate in batch → flush when full → embed API → scatter vectors → emit completed files
- DB Writer is a separate stage that inserts embeddings asynchronously
- Errors: `flush_batch()` returns `Err` → pipeline shutdown via `cancelled` flag

**Key architectural differences:**
1. **Concurrency model:** Python embeds concurrently, DB inserts inline. Rust embeds sequentially but DB inserts are decoupled into a separate thread.
2. **Error isolation:** Python catches per-batch errors and continues. Rust treats any batch failure after retries as fatal to the entire pipeline.
3. **Pipeline composition:** Python's monolithic `IndexingCoordinator` owns everything. Rust's staged pipeline has clear separation: parsing (parallel), embedding (single), DB writing (single).
4. **Backpressure:** Python uses Semaphore to limit in-flight API calls. Rust uses bounded channels (100) between embed and DB Writer.

---

## 3. Gaps Identified

### 3.1 Things Python does that Rust doesn't

| # | Gap | Severity | Notes |
|---|---|---|---|
| 1 | **Concurrent API requests** | **High** | Single-threaded embed stage is 8–40× slower. Python dispatches 8 (OpenAI) or 40 (VoyageAI) concurrent embed calls. |
| 2 | **Qwen model support** | **Medium** | Python supports Qwen3 embedding models via OpenAI-compatible endpoints with dedicated batch configs. Rust has no Qwen entries. |
| 3 | **Runtime dimension discovery** | **Medium** | Custom endpoints with non-standard dimensions won't be auto-detected. Users must configure `output_dims` explicitly. |
| 4 | **VoyageAI per-batch token budget** | **Low** | Rust doesn't enforce `max_tokens_per_batch`, relying on API error + batch split. Listed as future work. |
| 5 | **tiktoken integration** | **Low** | Rust uses `len()/3` only. Exact tiktoken would reduce false skips for borderline chunks. |
| 6 | **Health check** | **Low** | No `health_check()` equivalent. CLI health-check command needs Python path. |
| 7 | **Progress display** | **Medium** | No Rich progress bar with speed calculation. Users see no visual feedback during embedding. |
| 8 | **Per-batch error recovery** | **Medium** | Python's `return 0` on batch failure lets indexing continue. Rust's pipeline shutdown on embed error means a single batch failure halts everything. |
| 9 | **VoyageAI rate-limit backoff** | **Low** | Rust uses 1s base for rate limits (exponential to 2s, 4s); Python uses 30s base. Rust may waste retries on 60s rate-limit windows. |
| 10 | **Empty text handling: VoyageAI** | **Low** | Python's `validate_text_input` silently skips empty texts; Rust replaces with `[EMPTY]`. Different behavior for the same provider. |
| 11 | **Dimension whitelist validation** | **Low** | Python rejects invalid dims at init; Rust defers to API error. Clearer error messages in Python. |

### 3.2 Things Rust does that Python doesn't

| # | Gap | Severity | Notes |
|---|---|---|---|
| 1 | **Bloom filter dedup** | **High** | Much faster dedup for large DBs. 1% false-positive rate means rare re-index needed for skipped chunks. |
| 2 | **Content-hash-based dedup** | **Medium** | Re-parsed identical text avoids re-embedding (Python re-embeds if chunk_id changes). |
| 3 | **Persisted bloom filter** | **Medium** | Fast startup — loads bloom from disk instead of querying DB for all existing embeddings. |
| 4 | **Staged pipeline architecture** | **High** | Decoupled parsing → embedding → DB writing with bounded channels. Better backpressure, cleaner separation. |
| 5 | **GIL-free execution** | **Medium** | Embed runs in `py.allow_threads()`, releasing the GIL. Python's async code holds GIL during non-I/O work. |

---

## 4. Migration Impact

### What users will notice when switching from Python to Rust pipeline:

#### Positive
1. **Faster dedup**: Blooms filter is O(1) vs DB query. Noticeable on large repos with many existing embeddings.
2. **Faster startup**: Bloom loads from disk, no DB query for existing embeddings.
3. **Content-aware dedup**: Re-indexing the same text (after refactor/git operations) won't re-embed.
4. **GIL release**: Embed stage doesn't hold the GIL, allowing Python threads to run concurrently.
5. **Cleaner pipeline model**: Staged architecture with bounded channels gives predictable resource usage.

#### Negative
1. **Slower throughput**: Single-threaded embed is 8–40× slower than Python's concurrent dispatch. With 40 concurrent VoyageAI batches at ~50ms each, that's ~800 embeddings/sec vs ~20/sec.
2. **No progress bar**: No visual feedback during embedding. Users see nothing until completion.
3. **No Qwen support**: Users with Qwen3 embedding models (via Ollama) will need to wait for model config additions.
4. **Batch failure = pipeline halt**: A single permanent batch failure (after retries) stops the entire index. Python continues with partial results.
5. **Custom model dimension discovery**: Users with custom endpoints need to know and configure their `output_dims` explicitly.
6. **Empty text behavior difference**: VoyageAI empty texts are replaced with `[EMPTY]` instead of skipped. This means `[EMPTY]` gets embedded and stored, which could affect search results (empty texts matching search queries).

#### Neutral
1. **Embedding vectors**: Identical for the same inputs (same API calls, same request formats).
2. **Configuration**: Same Pydantic validation, same config sources.
3. **Logging**: Same infrastructure (both route to Python logging).
4. **Error messages**: Different text but same coverage of important events.
5. **VoyageAI batch token budget**: May hit more API errors but eventually produces the same embeddings.

### Recommended pre-migration checks
1. Ensure bloom filter capacity is adequate for the target DB size.
2. Verify `output_dims` is explicitly configured for custom endpoints.
3. Check that Qwen models have been added to the Rust model config table.
4. Consider adding a concurrency toggle to the Rust embed stage for power users.

---

## 5. Overall Assessment

The Rust embed stage design faithfully reproduces the core embedding behavior. The request format, dimension handling, truncation, and retry logic are all correct matches. The two biggest behavioral differences are:

1. **Concurrency** — the Rust stage is deliberately single-threaded. This is acknowledged in the design and deferred to future work ("If single-thread throughput is bottleneck"). For repos with many new chunks, this will be noticeably slower.

2. **Bloom filter dedup** — this is actually an improvement over the Python DB-query approach. The 1% false-positive rate is acceptable (chunks will be embedded on the next run), and the content-hash-based key is more robust than chunk-id-based dedup.

The identified gaps (Qwen support, dimension discovery, progress display) are all addressable in follow-up PRs and none are blocking for the initial Rust embed stage integration.
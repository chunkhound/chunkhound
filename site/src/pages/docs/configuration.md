---
layout: ../../layouts/DocsLayout.astro
title: "Configuration"
description: "Configure embedding providers, database backends, and indexing behavior."
order: 2
section: "manual"
---

# Configuration

ChunkHound is configured through a JSON file, environment variables, and CLI flags.

## Configuration File

Create `.chunkhound.json` in your project root. Here is a full example showing all sections:

```json
{
  "database": {
    "provider": "duckdb",
    "path": ".chunkhound/db"
  },
  "embedding": {
    "provider": "voyageai",
    "model": "voyage-3.5",
    "batch_size": 100
  },
  "indexing": {
    "exclude": ["**/node_modules/**", "**/dist/**"],
    "exclude_mode": "combined",
    "per_file_timeout_seconds": 3.0,
    "batch_size": 50,
    "db_batch_size": 100,
    "detect_embedded_sql": true
  },
  "llm": {
    "provider": "anthropic",
    "utility_model": "claude-haiku-4-5-20251001",
    "synthesis_model": "claude-sonnet-4-5-20250929"
  }
}
```

## Configuration Precedence

Settings are resolved in this order (highest priority first):

1. **CLI arguments** -- flags passed directly on the command line
2. **Config file** -- loaded via `--config` or `CHUNKHOUND_CONFIG_FILE`
3. **Local `.chunkhound.json`** -- auto-detected in the target directory
4. **Environment variables** -- `CHUNKHOUND_*` prefixed variables
5. **Defaults** -- built-in fallback values

## Embedding Providers

| Provider | Config Value | Env Var | Default Model | Notes |
|---|---|---|---|---|
| VoyageAI | `voyageai` | `CHUNKHOUND_EMBEDDING__API_KEY` | `voyage-3.5` | Recommended for code search |
| OpenAI | `openai` | `CHUNKHOUND_EMBEDDING__API_KEY` | `text-embedding-3-small` | Widely available |

### Embedding Options

| Option | Type | Default | Description |
|---|---|---|---|
| `base_url` | `string` | `null` | Custom embedding endpoint. Required for self-hosted OpenAI-compatible embeddings. |
| `ssl_verify` | `boolean` | `true` | Verify TLS certificates for requests sent to `base_url`. Ignored when `base_url` is unset. |
| `rerank_model` | `string` | `null` | Reranking model name (enables multi-hop reranking) |
| `rerank_url` | `string` | `null` | Separate rerank endpoint URL (optional when reranking is served from `base_url`) |
| `rerank_ssl_verify` | `boolean` | `null` | Verify TLS certificates for rerank requests. Inherits `ssl_verify` when unset. |
| `rerank_format` | `string` | `"auto"` | Reranking API format: `cohere`, `tei`, or `auto` |
| `rerank_batch_size` | `number` | `null` | Max documents per rerank request |
| `timeout` | `number` | `30` | Request timeout in seconds |
| `max_retries` | `number` | `3` | Max retry attempts on failure |
| `api_version` | `string` | `null` | Azure OpenAI API version (`YYYY-MM-DD`) |
| `azure_endpoint` | `string` | `null` | Azure OpenAI endpoint (mutually exclusive with `base_url`) |
| `azure_deployment` | `string` | `null` | Azure OpenAI deployment name |

## Database Backends

| Backend | Status | Recommended |
|---|---|---|
| `duckdb` | Stable | Yes — use this |
| `lancedb` | Experimental | No — for evaluation only |

### DuckDB (default)

> **Stable** — recommended for all use cases.

Fast analytical queries and efficient storage.

```json
{
  "database": {
    "provider": "duckdb",
    "path": ".chunkhound/db"
  }
}
```

### LanceDB

> **Experimental** — not recommended for production use. The LanceDB integration is actively developed but may have rough edges around index rebuilding, migration, and edge-case query correctness. Use DuckDB unless you are evaluating LanceDB specifically.

```json
{
  "database": {
    "provider": "lancedb",
    "path": ".chunkhound/db"
  }
}
```

### Database Options

| Option | Type | Default | Description |
|---|---|---|---|
| `max_disk_usage_mb` | `number` | `null` | Max DB size in MB before indexing stops (CLI flag uses GB) |
| `lancedb_index_type` | `string` | `null` | LanceDB vector index type: `auto`, `ivf_hnsw_sq`, or `ivf_rq` |
| `lancedb_optimize_fragment_threshold` | `number` | `100` | Fragment count to trigger LanceDB compaction |

## Indexing Options

| Option | Type | Default | Description |
|---|---|---|---|
| `exclude` | `string[]` | *built-in list* | Glob patterns to exclude from indexing |
| `include` | `string[]` | all supported file types | Glob patterns limiting which files are indexed; files not matching any pattern are skipped |
| `exclude_mode` | `string` | `null` | `combined`, `config_only`, or `gitignore_only`. When an explicit `exclude` list is provided, defaults to `"combined"`; otherwise defaults to `"gitignore_only"` |
| `force_reindex` | `boolean` | `false` | Force re-indexing of all files |
| `max_concurrent` | `number` | `5` | Max concurrent parser workers |
| `cleanup` | `boolean` | `true` | Remove orphaned DB records after indexing |
| `max_file_size_mb` | `number` | `10` | Skip files larger than this (MB) |
| `config_file_size_threshold_kb` | `number` | `20` | Skip structured config files (JSON/YAML/TOML) larger than this (KB); 0 to disable |
| `per_file_timeout_seconds` | `number` | `3.0` | Max parse time per file (0 to disable) |
| `batch_size` | `number` | `50` | Files per parsing batch |
| `db_batch_size` | `number` | `100` | Chunks per database write batch |
| `detect_embedded_sql` | `boolean` | `true` | Index SQL in string literals |
| `per_file_timeout_min_size_kb` | `number` | `128` | Only apply per-file timeout to files at least this large (KB) |

By default, ChunkHound excludes common noise directories (`node_modules`, `dist`, `__pycache__`, `.git`, lock files, build artifacts). Set `exclude_mode: "config_only"` and `exclude: []` to start with a clean slate.

### Exclude Modes

- **`combined`** (default when custom `exclude` patterns are provided) -- merges `.gitignore` rules with your `indexing.exclude` patterns
- **`config_only`** -- only uses patterns from `indexing.exclude`, ignores `.gitignore`
- **`gitignore_only`** (default when no custom `exclude` patterns are provided) -- only uses `.gitignore` rules, ignores config excludes

## LLM Configuration

The LLM provider is used for deep code research (`chunkhound research` and the `code_research` MCP tool).

| Provider | Config Value | Env Var | Utility Default | Synthesis Default | Notes |
|---|---|---|---|---|---|
| Claude Code CLI | `claude-code-cli` | -- | `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` | Uses local Claude Code installation |
| Codex CLI | `codex-cli` | -- | `codex` | `codex` | Uses local Codex CLI installation |
| OpenCode CLI | `opencode-cli` | -- | `opencode/grok-code` | `opencode/grok-code` | Uses local OpenCode CLI installation |
| Anthropic | `anthropic` | `CHUNKHOUND_LLM_API_KEY` | `claude-haiku-4-5-20251001` | `claude-sonnet-4-5-20250929` | Direct API access |
| OpenAI | `openai` | `CHUNKHOUND_LLM_API_KEY` | `gpt-5-nano` | `gpt-5` | Direct API access |
| Gemini | `gemini` | `CHUNKHOUND_LLM_API_KEY` | `gemini-3-pro-preview` | `gemini-3-pro-preview` | Google Gemini API |
| Grok | `grok` | `CHUNKHOUND_LLM_API_KEY` | `grok-4-1-fast-reasoning` | `grok-4-1-fast-reasoning` | xAI API |

`"model"` is a convenience shorthand that sets both `utility_model` and `synthesis_model` to the same value. To use different models per role, set `utility_model` and `synthesis_model` explicitly.

When an OpenAI-compatible LLM provider points at a custom `base_url`, ChunkHound treats it as a generic custom backend. In that mode you must set an explicit model name; ChunkHound does not guess a local default. This applies to `provider: "openai"`, to Grok when routed through a non-official endpoint, and to per-role overrides that resolve to those providers.

### LLM Options

| Option | Type | Default | Description |
|---|---|---|---|
| `utility_provider` | `string` | `null` | Override provider for utility operations |
| `synthesis_provider` | `string` | `null` | Override provider for synthesis operations |
| `timeout` | `number` | `120` | LLM request timeout in seconds |
| `max_retries` | `number` | `3` | Max retry attempts |
| `codex_reasoning_effort` | `string` | `null` | Default reasoning effort for Codex/OpenAI: `minimal`, `low`, `medium`, `high`, `xhigh` |
| `codex_reasoning_effort_utility` | `string` | `null` | Reasoning effort override for utility stage |
| `codex_reasoning_effort_synthesis` | `string` | `null` | Reasoning effort override for synthesis stage |

### Anthropic-specific Options

| Option | Type | Default | Description |
|---|---|---|---|
| `anthropic_thinking_enabled` | `boolean` | `false` | Enable extended thinking |
| `anthropic_thinking_budget_tokens` | `number` | `10000` | Token budget for thinking (min 1024) |
| `anthropic_interleaved_thinking` | `boolean` | `false` | Interleaved thinking for tool use (Claude 4+) |
| `anthropic_effort` | `string` | `null` | Effort parameter: `low`, `medium`, `high` |

## Research Configuration

Controls the `code_research` MCP tool and `chunkhound research` command.

| Option | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `algorithm` | `"v1"\|"v2"\|"v3"` | `"v3"` | `CHUNKHOUND_RESEARCH_ALGORITHM` | Research algorithm version |
| `exhaustive_mode` | `bool` | `false` | `CHUNKHOUND_RESEARCH_EXHAUSTIVE_MODE` | Retrieve everything (no time/count limit) |
| `multi_hop_time_limit` | `number` | `5.0` | `CHUNKHOUND_RESEARCH_MULTI_HOP_TIME_LIMIT` | Max seconds for evidence expansion |
| `multi_hop_result_limit` | `number` | `500` | `CHUNKHOUND_RESEARCH_MULTI_HOP_RESULT_LIMIT` | Max accumulated chunks |
| `target_tokens` | `number` | `20000` | `CHUNKHOUND_RESEARCH_TARGET_TOKENS` | Output token budget for synthesis |
| `query_expansion_enabled` | `bool` | `true` | `CHUNKHOUND_RESEARCH_QUERY_EXPANSION_ENABLED` | LLM-based query expansion |
| `relevance_threshold` | `number` | `0.5` | `CHUNKHOUND_RESEARCH_RELEVANCE_THRESHOLD` | Min rerank score for inclusion |

```json
{
  "research": {
    "algorithm": "v3",
    "exhaustive_mode": false,
    "target_tokens": 20000,
    "query_expansion_enabled": true,
    "relevance_threshold": 0.5
  }
}
```

The full list of parameters is available in `research_config.py`.

### Algorithm Versions

The `algorithm` setting controls how ChunkHound explores your codebase to answer a research question. All three versions produce the same output format; they differ only in how thoroughly they search.

**New to ChunkHound? Start with `"v3"` (the default).**

| Version | Strategy | LLM calls | Best for |
|---|---|---|---|
| `v1` | BFS — generates follow-up questions, explores one level deep | Minimal | Quick lookups, simple codebases |
| `v2` | Wide coverage — depth-first on top files, then gap detection | Medium | Balanced discovery; most production use cases |
| `v3` *(default)* | Runs v1 + v2 in parallel, merges results | Most (parallel, not sequential) | Complex codebases where missing context is costly |

**v3 is not slower than v2** — both strategies run concurrently via `asyncio.gather`, so the wall-clock time is roughly the same as v2 alone while covering more ground.

**When to switch away from v3:**
- Use `v1` when cost matters and the question is narrow and self-contained ("explain how the config loader works")
- Use `v2` when you want a good balance without the extra LLM spend of dual-strategy merging
- `v3` is the right default for open-ended research questions ("how does authentication flow through this system?")

Gap detection parameters (`min_gaps`, `max_gaps`, `gap_similarity_threshold`) only affect v2 and v3. They are silently ignored for v1.

## Environment Variables

Most environment variables use the `CHUNKHOUND_` prefix with `__` (double underscore) as the section delimiter. The LLM section uses a single underscore (`CHUNKHOUND_LLM_*`).

| Variable | Description |
|---|---|
| `CHUNKHOUND_EMBEDDING__PROVIDER` | Embedding provider name |
| `CHUNKHOUND_EMBEDDING__MODEL` | Embedding model name |
| `CHUNKHOUND_EMBEDDING__API_KEY` | API key for embedding provider |
| `CHUNKHOUND_EMBEDDING__BASE_URL` | Base URL for OpenAI-compatible endpoints |
| `CHUNKHOUND_EMBEDDING__SSL_VERIFY` | Verify TLS certificates for embedding requests sent to `base_url` |
| `CHUNKHOUND_EMBEDDING__RERANK_SSL_VERIFY` | Verify TLS certificates for rerank requests (overrides `ssl_verify`) |
| `CHUNKHOUND_DATABASE__PROVIDER` | Database backend (`duckdb` or `lancedb`) |
| `CHUNKHOUND_DATABASE__PATH` | Database storage path |
| `CHUNKHOUND_LLM_PROVIDER` | LLM provider for research |
| `CHUNKHOUND_LLM_UTILITY_MODEL` | LLM model for utility tasks (fast, lower cost) |
| `CHUNKHOUND_LLM_SYNTHESIS_MODEL` | LLM model for synthesis tasks (primary output) |
| `CHUNKHOUND_LLM_API_KEY` | API key for LLM provider |
| `CHUNKHOUND_LLM_BASE_URL` | Base URL for LLM provider (proxy / custom endpoint) |
| `CHUNKHOUND_LLM_SSL_VERIFY` | Verify TLS certificates for requests sent to `llm.base_url` |
| `CHUNKHOUND_INDEXING__EXCLUDE_MODE` | Exclusion mode (`combined`, `config_only`, `gitignore_only`) |
| `CHUNKHOUND_INDEXING__PER_FILE_TIMEOUT_SECONDS` | Per-file parse timeout |
| `CHUNKHOUND_INDEXING__DETECT_EMBEDDED_SQL` | Enable embedded SQL detection |
| `CHUNKHOUND_INDEXING__GIT_PATHSPEC_CAP` | Max git pathspec entries (default: 128) |
| `CHUNKHOUND_DB_EXECUTE_TIMEOUT` | Database executor timeout |
| `CHUNKHOUND_YAML_ENGINE` | YAML parser engine (`rapid` or `tree`) |
| `CHUNKHOUND_LLM_CODEX_REASONING_EFFORT` | Reasoning effort for Codex models (`minimal`, `low`, `medium`, `high`, `xhigh`) |
| `CHUNKHOUND_CONFIG_FILE` | Path to config file (alternative to `--config`) |
| `CHUNKHOUND_DEBUG` | Enable debug logging |
| `CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB` | Max database size in GB |
| `CHUNKHOUND_INDEXING__FORCE_REINDEX` | Force re-indexing |
| `CHUNKHOUND_INDEXING__MAX_CONCURRENT` | Max concurrent workers |
| `CHUNKHOUND_EMBEDDING__RERANK_MODEL` | Reranking model |
| `VOYAGE_API_KEY` | Fallback API key for VoyageAI provider |

## Advanced routing

The homepage configurator emits the 30-second onboarding shape. Real enterprise deployments often need to hit Azure, a self-hosted endpoint, or an LLM proxy. Below is what ChunkHound actually wires through, and what it doesn't.

### TLS verification for custom endpoints

`ssl_verify` is explicit now. ChunkHound does **not** disable certificate verification automatically.

- `embedding.ssl_verify` only affects requests sent to an explicit `embedding.base_url`.
- `embedding.rerank_ssl_verify` only affects rerank requests and overrides inherited `ssl_verify` when set.
- `llm.ssl_verify` only affects requests sent to an explicit `llm.base_url`.
- If `base_url` is unset, `ssl_verify` is ignored for security.
- If `rerank_url` is unset, `rerank_ssl_verify` is ignored.
- Prefer a proper CA trust chain when possible. Use `false` only for local endpoints or trusted internal networks with self-signed/private certificates.

### Azure OpenAI (embeddings)

ChunkHound's OpenAI embedding provider speaks Azure OpenAI natively. Supply the four Azure fields and omit `base_url` — the two are mutually exclusive.

```json
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "<YOUR_AZURE_KEY>",
    "azure_endpoint": "https://<resource>.openai.azure.com",
    "api_version": "2024-02-01",
    "azure_deployment": "<your-deployment-name>"
  }
}
```

LLM-side Azure OpenAI is **not supported yet** — the `llm` section has no Azure fields. Use a proxy (see below) if you need to route LLM traffic through Azure.

### VoyageAI on Azure ML / AI Foundry

VoyageAI models are available on the Azure Marketplace and in Microsoft Foundry. ChunkHound can target an Azure-hosted Voyage deployment via `base_url`:

```json
{
  "embedding": {
    "provider": "voyageai",
    "model": "voyage-3.5",
    "api_key": "<YOUR_AZURE_VOYAGE_KEY>",
    "base_url": "https://<your-resource>.services.ai.azure.com/models",
    "ssl_verify": true,
    "rerank_url": "https://<your-rerank-endpoint>/rerank",
    "rerank_ssl_verify": true,
    "rerank_format": "tei"
  }
}
```

Caveats:

- **Native Voyage API required.** The Azure deployment must expose `/v1/embeddings` with the native Voyage shape (true for Voyage marketplace listings; verify your specific deployment).
- **Bundled reranker unavailable.** VoyageAI's `rerank-*` models are not accessible through a custom `base_url` — the embedding endpoint doesn't expose `/rerank`. Run a separate reranker and point `rerank_url` at it. vLLM with `Qwen/Qwen3-Reranker-0.6B` is a drop-in option:
  ```bash
  vllm serve Qwen/Qwen3-Reranker-0.6B --task score --port 8000
  ```
- **TLS disablement is primarily for the HTTP reranker path.** The separate `rerank_url` path respects `ssl_verify` / `rerank_ssl_verify`. For the VoyageAI SDK path, prefer trusted CA configuration such as `REQUESTS_CA_BUNDLE`.
- **Concurrency throttled to 1 by default** when `base_url` is set, to respect Azure serverless rate limits. Override via `max_concurrent_batches` if your SKU permits.
- **`api_key` still required.** The validator doesn't enforce it when `base_url` is present, but Azure-hosted endpoints still need their own key — supply it.

### LLM via proxy (Anthropic, OpenAI, Grok)

The Anthropic, OpenAI, and Grok LLM providers all forward `base_url` to their SDK. Point them at a gateway like [LiteLLM](https://github.com/BerriAI/litellm) to centralize auth, logging, and rate limiting:

```json
{
  "llm": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5-20250929",
    "api_key": "<YOUR_GATEWAY_KEY>",
    "base_url": "https://your-gateway.example.com",
    "ssl_verify": true
  }
}
```

The gateway must preserve each provider's native request/response shape — ChunkHound uses the vendor SDKs, not a generic HTTP client.

### Local OpenAI-compatible servers (Ollama, vLLM)

Local inference servers that speak the OpenAI API work via `provider: "openai"` with `base_url` pointing at the local endpoint. No `api_key` is needed for servers that don't enforce auth, but you must set an explicit `model`.

#### Ollama

Ollama provides embeddings, reranking, and LLM inference in a single process. Pull the models you need, then point ChunkHound at the Ollama endpoint:

```bash
# Embedding + reranker models
ollama pull qwen3-embedding && ollama pull qwen3-reranker

# LLM — pick one
ollama pull qwen3-coder:30b
ollama pull gemma4:27b
```

Embedding and reranker config (`.chunkhound.json`):

```json
{
  "embedding": {
    "provider": "openai",
    "model": "qwen3-embedding",
    "base_url": "http://localhost:11434/v1",
    "ssl_verify": false,
    "rerank_model": "qwen3-reranker",
    "rerank_format": "cohere"
  }
}
```

No `rerank_url` is needed — it is auto-derived from `base_url`.

LLM config:

```json
{
  "llm": {
    "provider": "openai",
    "model": "qwen3-coder:30b",
    "base_url": "http://localhost:11434/v1",
    "ssl_verify": false
  }
}
```

Use whichever model you pulled in `llm.model`. For example, set `"model": "gemma4:27b"` if you want the Gemma 4 path instead of Qwen. ChunkHound does not infer a local default model from `base_url`.

If your embeddings stay on the official provider but reranking goes to a local HTTPS service with a self-signed certificate, override the reranker only:

```json
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "<YOUR_OPENAI_KEY>",
    "rerank_model": "Qwen/Qwen3-Reranker-0.6B",
    "rerank_url": "https://localhost:8001/rerank",
    "rerank_ssl_verify": false,
    "rerank_format": "tei"
  }
}
```

#### vLLM

vLLM gives you dedicated processes per model, which is better for throughput and lets you serve HuggingFace model IDs directly. When embeddings and reranking are served from the same OpenAI-compatible endpoint, ChunkHound infers the reranker path from `base_url` just like it does for Ollama:

```bash
# Embedding + reranker server
vllm serve Qwen/Qwen3-Embedding-0.6B --port 8000

# LLM server
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 11434
```

Embedding and reranker config (`.chunkhound.json`):

```json
{
  "embedding": {
    "provider": "openai",
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "base_url": "http://localhost:8000/v1",
    "rerank_model": "Qwen/Qwen3-Reranker-0.6B",
    "rerank_format": "cohere"
  }
}
```

No `rerank_url` is needed when the reranker lives behind the same OpenAI-compatible endpoint. ChunkHound auto-derives `/rerank` from `base_url`.

If you split embeddings and reranking across different services, keep `base_url` pointed at the embedding server and set `rerank_url` explicitly:

```json
{
  "embedding": {
    "provider": "openai",
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "base_url": "http://localhost:8025/v1",
    "rerank_model": "Qwen/Qwen3-Reranker-0.6B",
    "rerank_url": "http://localhost:8000/rerank",
    "rerank_format": "cohere"
  }
}
```

LLM config:

```json
{
  "llm": {
    "provider": "openai",
    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "base_url": "http://localhost:11434/v1"
  }
}
```

> **Ollama vs vLLM:** Ollama is simpler — one process, one command per model. vLLM is better for throughput and gives you full control over each serving process. Both work equally well with ChunkHound as long as `llm.model` is set explicitly.

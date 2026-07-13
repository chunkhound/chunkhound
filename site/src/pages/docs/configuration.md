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

### Global Defaults (cross-project configuration)

To avoid copying the same settings into every project, place a global config in one of these locations (checked in order):

- `~/.config/chunkhound/chunkhound.json`
- `~/.config/chunkhound/.chunkhound.json`
- `~/.chunkhound/chunkhound.json`
- `~/.chunkhound/.chunkhound.json`
- `~/chunkhound.json`
- `~/.chunkhound.json`

You can also point to an arbitrary file with the `CHUNKHOUND_GLOBAL_CONFIG_FILE` environment variable.

Example global config (`~/.config/chunkhound/chunkhound.json`):

```json
{
  "embedding": {
    "provider": "voyageai",
    "model": "voyage-3.5"
  },
  "llm": {
    "provider": "anthropic"
  },
  "indexing": {
    "exclude": ["**/node_modules/**", "**/.git/**", "**/dist/**"]
  }
}
```

Then in a specific project you only need a minimal `.chunkhound.json` for project-specific overrides (or nothing at all if the globals are sufficient):

```json
{
  "embedding": {
    "api_key": "sk-..."   // only the key differs per machine
  }
}
```

Global values are deep-merged; project files win for any keys they specify.

Merging behavior:

- **Nested objects** (`embedding`, `llm`, `research`, `database`, etc.): deep merge. You only need to specify the keys you want to change in a higher-priority file (local, explicit config, or CLI). Siblings from global (or lower layers) are preserved. This lets a project override just an `api_key`, switch `llm.model`, or change `research.algorithm` without losing other global settings.

- **Lists** (`indexing.exclude`, `indexing.include`): the list supplied by the higher-priority source completely replaces any list from a lower layer (including global). ChunkHound's built-in safe defaults are still layered on top of your list in the *effective* excludes (see `get_effective_config_excludes()`), and `.gitignore` interaction is controlled by `exclude_mode`.

Example — project overrides only the secret while inheriting provider + model from global (or switches providers entirely):

```json
{
  "embedding": {
    "api_key": "sk-..."
  }
}
```

or to use a completely different provider for this project:

```json
{
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-large",
    "api_key": "sk-..."
  }
}
```

Example — project uses its own exclude list (replaces global's list at the raw config level):

```json
{
  "indexing": {
    "exclude": ["**/my-vendor/**", "**/generated/**"]
  }
}
```

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
4. **Global defaults** -- `CHUNKHOUND_GLOBAL_CONFIG_FILE` or auto-discovered in `~/.config/chunkhound/` (or `~/.chunkhound/`)
5. **Environment variables** -- `CHUNKHOUND_*` prefixed variables
6. **Defaults** -- built-in fallback values

Global defaults let you maintain shared settings (e.g. embedding provider + API key, common exclude patterns, LLM roles) in a single file so you do not need to copy `.chunkhound.json` into every project. Any project-local `.chunkhound.json` (or explicit config/CLI) overrides values from the global layer. Nested objects (embedding, llm, research, database, ...) are deep-merged: specify only the keys you want to change and siblings from global survive. Lists such as `indexing.exclude` / `include` from a higher layer fully replace lower ones (built-in defaults are still applied on top; see Global Defaults above for details and examples).

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
| `fragmentation_threshold_pct` | `number` | `30` | Background/auto-compaction trigger: file-size overhead above the provider's estimated live DB size (%). 30 = compact when the DB is ~30% larger than live data. 0 = always, null = never. This does not disable the fixed `chunkhound index` compaction boundaries. CLI: `--fragmentation-threshold-pct`. |
| `lancedb_index_type` | `string` | `null` | LanceDB vector index type: `auto`, `ivf_hnsw_sq`, or `ivf_rq` |
| `lancedb_optimize_fragment_threshold` | `number` | `100` | Fragment count to trigger LanceDB compaction |

DuckDB also compacts during `chunkhound index` at two fixed batch boundaries: once after chunking and before embedding generation, then again at the end of the indexing pass. Those boundary calls are unconditional, including `--no-embeddings` and noop re-index runs. If a batch compaction fails with `status: "error"`, the index run is aborted. Providers that report compaction as unsupported/`skipped` keep indexing normally. Sampled/background auto-compaction (triggered by fragmentation threshold during normal operations) does NOT fail the original operation — failures are logged and skipped.

DuckDB compaction rebuilds a fresh canonical ChunkHound database file and swaps it into place atomically. This is intentionally **not** a generic DuckDB passthrough: only ChunkHound-owned canonical tables (`schema_version`, `files`, `chunks`, `embeddings_*`) are preserved. Any unknown or non-canonical tables are dropped during compaction by design.

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
| Gemini | `gemini` | `CHUNKHOUND_LLM_API_KEY` | Must be set explicitly via `CHUNKHOUND_LLM_MODEL` or `llm.model` (configurator defaults to `gemini-3.5-flash`) | Must be set explicitly via `CHUNKHOUND_LLM_MODEL` or `llm.model` (configurator defaults to `gemini-3.5-flash`) | Google Gemini API. Migration: `CHUNKHOUND_GEMINI_MODEL` was removed in v4.x — rename to `CHUNKHOUND_LLM_MODEL`. |
| Grok | `grok` | `CHUNKHOUND_LLM_API_KEY` | Must be set explicitly (configurator defaults to `grok-4.3`) | Must be set explicitly (configurator defaults to `grok-4.3`) | xAI API. Registry providers require explicit `model`. |
| DeepSeek | `deepseek` | `CHUNKHOUND_LLM_API_KEY` | Must be set explicitly (configurator defaults to `deepseek-v4-flash`) | Must be set explicitly (configurator defaults to `deepseek-v4-flash`) | DeepSeek API. Registry providers require explicit `model`. |
| Antigravity SDK | `antigravity-sdk` | `CHUNKHOUND_LLM_API_KEY` | `gemini-3.5-flash` | `gemini-3.1-pro` | Uses official Google Antigravity SDK |
| Antigravity CLI | `antigravity-cli` | -- | `gemini-3.5-flash` | `gemini-3.1-pro` | Uses local agy / antigravity CLI command |

> [!NOTE]
> The `google-antigravity` SDK dependency is optional to keep base installations lightweight and compatible.
> If you configure `provider: "antigravity-sdk"`, install the optional extra into the **same environment ChunkHound runs from**:
> - Source checkout (uv): `uv sync --extra antigravity` (or `uv pip install -e ".[antigravity]"`)
> - Pip venv: `pip install "chunkhound[antigravity]"`
> - Global tool: `uv tool install "chunkhound[antigravity]"`
>
> **Platform support:** `google-antigravity` is published as platform-specific wheels with no source distribution, so which OS/architectures it installs on depends on the wheels the SDK version you resolve happens to ship. If `pip install "chunkhound[antigravity]"` (or `uv sync --extra antigravity`) reports no compatible wheel for your platform, use the `antigravity-cli` provider instead — it needs only the external `agy`/`antigravity` CLI, no Python extra. See the [`google-antigravity` page on PyPI](https://pypi.org/project/google-antigravity/#files) for the current wheel list.

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

These apply when the active provider (or a role provider) is `anthropic`. Each option also has a matching `CHUNKHOUND_LLM_ANTHROPIC_<OPTION>` environment variable (single underscore, uppercased).

| Option | Type | Default | Description |
|---|---|---|---|
| `anthropic_thinking_enabled` | `boolean` | `false` | Enable extended thinking. |
| `anthropic_thinking_mode` | `string` | `null` | `auto` (default when unset), `off`, `manual`, or `adaptive`. `auto` selects adaptive on Opus 4.6+/Sonnet 4.6 and manual on older models. |
| `anthropic_thinking_budget_tokens` | `number` | `10000` | Manual-mode thinking budget (min 1024). Ignored in adaptive mode (Opus 4.6+). |
| `anthropic_thinking_display` | `string` | `null` | Adaptive-mode thinking text: `summarized` or `omitted`. Opus 4.7/4.8 omit by default. |
| `anthropic_interleaved_thinking` | `boolean` | `false` | Manual-mode interleaved thinking between tool calls. Auto-enabled in adaptive mode. |
| `anthropic_effort` | `string` | `null` | Token-usage effort: `low`, `medium`, `high`, `xhigh`, `max`. `xhigh` is Opus 4.7/4.8 only; `max` is Opus 4.6+. Unsupported levels are dropped with a warning. |
| `anthropic_task_budget_tokens` | `number` | `null` | Advisory agentic-loop token budget (beta). Opus 4.7/4.8 only; minimum 20000. |
| `anthropic_prompt_caching` | `boolean` | `false` | Send `cache_control` so the Messages API can cache prompt prefixes. |
| `anthropic_cache_ttl` | `string` | `null` | Prompt-cache TTL such as `1h`. `null` uses the API default of 5 minutes. |
| `anthropic_context_management_enabled` | `boolean` | `false` | Automatic clearing of tool results and thinking blocks (beta). |
| `anthropic_clear_thinking_keep_turns` | `number` | `null` | Thinking turns to keep when context management clears them. `null` keeps all. |
| `anthropic_clear_tool_uses_trigger_tokens` | `number` | `null` | Input-token threshold that triggers tool-result clearing. |
| `anthropic_clear_tool_uses_keep` | `number` | `null` | Number of recent tool-use pairs to keep after clearing. |

### Gemini-specific Options

These apply when the active provider (or a role provider) is `gemini`. Matching environment variables use the `CHUNKHOUND_LLM_GEMINI_*` prefix, and the CLI exposes `--llm-gemini-thinking-level` / `--llm-gemini-thinking-budget`.

| Option | Type | Default | Description |
|---|---|---|---|
| `gemini_thinking_level` | `string` | `null` | Adaptive thinking depth for Gemini 3+ models. Allowed values: `low`, `medium`, `high`. Forwarded to the Google Gen AI SDK as `thinking_level`. |
| `gemini_thinking_budget` | `number` | `null` | Fixed thinking token budget for Gemini 2.5+ models. Forwarded to the Google Gen AI SDK as `thinking_budget`. |

Both options can be set independently — `thinking_level` controls adaptive depth (Gemini 3+), while `thinking_budget` sets a fixed token cap (Gemini 2.5+).
If both are unset, ChunkHound sends no Gemini thinking config and the model uses its own defaults.

## Research Configuration

Controls the `code_research` MCP tool and `chunkhound research` command.

| Option | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `algorithm` | `"v1"\|"v2"\|"v3"` | `"v3"` | `CHUNKHOUND_RESEARCH_ALGORITHM` | Research algorithm version |
| `query_expansion_enabled` | `bool` | `true` | `CHUNKHOUND_RESEARCH_QUERY_EXPANSION_ENABLED` | LLM-based query expansion for broader coverage |
| `num_expanded_queries` | `int` | `2` | `CHUNKHOUND_RESEARCH_NUM_EXPANDED_QUERIES` | Number of additional queries to generate (1-5) |
| `initial_page_size` | `int` | `30` | `CHUNKHOUND_RESEARCH_INITIAL_PAGE_SIZE` | Results per vector query in multi-hop search (10-100) |
| `relevance_threshold` | `number` | `0.5` | `CHUNKHOUND_RESEARCH_RELEVANCE_THRESHOLD` | Min rerank score for chunk inclusion (0.3-0.8) |
| `max_symbols` | `int` | `5` | `CHUNKHOUND_RESEARCH_MAX_SYMBOLS` | Max symbols to extract for regex search augmentation (1-20) |
| `regex_augmentation_ratio` | `number` | `0.3` | `CHUNKHOUND_RESEARCH_REGEX_AUGMENTATION_RATIO` | Regex target as fraction of semantic count (0.1-1.0) |
| `regex_min_results` | `int` | `20` | `CHUNKHOUND_RESEARCH_REGEX_MIN_RESULTS` | Min regex results regardless of augmentation ratio (10-100) |
| `regex_scan_page_size` | `int` | `100` | `CHUNKHOUND_RESEARCH_REGEX_SCAN_PAGE_SIZE` | Internal pagination batch size for regex exclusion scanning (50-200) |
| `multi_hop_time_limit` | `number` | `5.0` | `CHUNKHOUND_RESEARCH_MULTI_HOP_TIME_LIMIT` | Max seconds for evidence expansion (1.0-15.0) |
| `multi_hop_result_limit` | `int` | `500` | `CHUNKHOUND_RESEARCH_MULTI_HOP_RESULT_LIMIT` | Max chunks accumulated during multi-hop expansion (100-2000) |
| `multi_hop_min_candidates` | `int` | `5` | `CHUNKHOUND_RESEARCH_MULTI_HOP_MIN_CANDIDATES` | Min candidates above threshold to continue expansion (1-20) |
| `multi_hop_score_degradation` | `number` | `0.15` | `CHUNKHOUND_RESEARCH_MULTI_HOP_SCORE_DEGRADATION` | Max score drop in top-5 before terminating expansion (0.05-0.5) |
| `multi_hop_min_relevance` | `number` | `0.3` | `CHUNKHOUND_RESEARCH_MULTI_HOP_MIN_RELEVANCE` | Quality floor for expansion candidates (0.1-0.8) |
| `depth_exploration_enabled` | `bool` | `true` | `CHUNKHOUND_RESEARCH_DEPTH_EXPLORATION_ENABLED` | Enable depth exploration to find more chunks in discovered files |
| `max_exploration_files` | `int` | `5` | `CHUNKHOUND_RESEARCH_MAX_EXPLORATION_FILES` | Max files to explore for additional aspects, top-K by score (1-15) |
| `exploration_queries_per_file` | `int` | `2` | `CHUNKHOUND_RESEARCH_EXPLORATION_QUERIES_PER_FILE` | Number of aspect-based queries to generate per file (1-3) |
| `depth_exploration_max_completion_tokens` | `int` | `10000` | `CHUNKHOUND_RESEARCH_DEPTH_EXPLORATION_MAX_COMPLETION_TOKENS` | Token budget for depth exploration query generation (1-50000) |
| `min_gaps` | `int` | `1` | `CHUNKHOUND_RESEARCH_MIN_GAPS` | Minimum gaps to process after selection (0-5) |
| `max_gaps` | `int` | `10` | `CHUNKHOUND_RESEARCH_MAX_GAPS` | Maximum gaps to fill after selection (5-30) |
| `gap_similarity_threshold` | `number` | `0.25` | `CHUNKHOUND_RESEARCH_GAP_SIMILARITY_THRESHOLD` | Cosine distance threshold for clustering similar gaps (0.1-0.5) |
| `shard_budget` | `int` | `40000` | `CHUNKHOUND_RESEARCH_SHARD_BUDGET` | Token budget per gap detection shard for LLM processing (20000-60000) |
| `min_cluster_size` | `int` | `5` | `CHUNKHOUND_RESEARCH_MIN_CLUSTER_SIZE` | Minimum cluster size for HDBSCAN clustering (1-20) |
| `target_tokens` | `int` | `20000` | `CHUNKHOUND_RESEARCH_TARGET_TOKENS` | Output token budget for final synthesis (10000-100000) |
| `max_compression_iterations` | `int` | `5` | `CHUNKHOUND_RESEARCH_MAX_COMPRESSION_ITERATIONS` | Max compression loop iterations before error (1-10) |
| `max_boundary_expansion_lines` | `int` | `300` | `CHUNKHOUND_RESEARCH_MAX_BOUNDARY_EXPANSION_LINES` | Max lines to expand for complete functions/classes (50-500) |
| `max_chunks_per_file_repr` | `int` | `5` | `CHUNKHOUND_RESEARCH_MAX_CHUNKS_PER_FILE_REPR` | Top chunks per file for representative document creation (1-10) |
| `max_tokens_per_file_repr` | `int` | `2000` | `CHUNKHOUND_RESEARCH_MAX_TOKENS_PER_FILE_REPR` | Token limit per file representative document (500-5000) |
| `context_window` | `int` | `150000` | `CHUNKHOUND_RESEARCH_CONTEXT_WINDOW` | Max tokens for LLM context window (50000-200000) |
| `compression_max_depth` | `int` | `10` | `CHUNKHOUND_RESEARCH_COMPRESSION_MAX_DEPTH` | Max recursion depth for hierarchical compression (1-20) |
| `final_synthesis_threshold` | `int` | `75000` | `CHUNKHOUND_RESEARCH_FINAL_SYNTHESIS_THRESHOLD` | Max tokens for final synthesis LLM call (30000-200000) |
| `window_expansion_enabled` | `bool` | `true` | `CHUNKHOUND_RESEARCH_WINDOW_EXPANSION_ENABLED` | Enable neighboring chunk expansion for context |
| `window_expansion_lines` | `int` | `50` | `CHUNKHOUND_RESEARCH_WINDOW_EXPANSION_LINES` | Lines to expand before/after retrieved chunks (10-200) |
| `import_resolution_enabled` | `bool` | `true` | `CHUNKHOUND_RESEARCH_IMPORT_RESOLUTION_ENABLED` | Automatically fetch source files for imports in retrieved chunks |
| `import_resolution_max_files` | `int` | `10` | `CHUNKHOUND_RESEARCH_IMPORT_RESOLUTION_MAX_FILES` | Max import source files to fetch per synthesis (1-50) |
| `exhaustive_mode` | `bool` | `false` | `CHUNKHOUND_RESEARCH_EXHAUSTIVE_MODE` | Enable exhaustive retrieval (no result limit, 600s timeout) |
| `exhaustive_time_limit` | `number` | `600.0` | `CHUNKHOUND_RESEARCH_EXHAUSTIVE_TIME_LIMIT` | Safety timeout for exhaustive mode in seconds (60-1800) |

```json
{
  "research": {
    "algorithm": "v3",
    "exhaustive_mode": false,
    "target_tokens": 20000,
    "query_expansion_enabled": true,
    "depth_exploration_max_completion_tokens": 10000,
    "relevance_threshold": 0.5
  }
}
```

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
| `CHUNKHOUND_EMBEDDING__RERANK_MODEL` | Reranking model name |
| `CHUNKHOUND_EMBEDDING__RERANK_URL` | Separate rerank endpoint URL |
| `CHUNKHOUND_EMBEDDING__RERANK_SSL_VERIFY` | Verify TLS certificates for rerank requests (overrides `ssl_verify`) |
| `CHUNKHOUND_EMBEDDING__RERANK_FORMAT` | Reranking API format: `cohere`, `tei`, or `auto` |
| `CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE` | Max documents per rerank request |
| `CHUNKHOUND_EMBEDDING__TIMEOUT` | Request timeout in seconds (default: 30) |
| `CHUNKHOUND_EMBEDDING__MAX_RETRIES` | Max retry attempts on failure (default: 3) |
| `CHUNKHOUND_EMBEDDING__API_VERSION` | Azure OpenAI API version (`YYYY-MM-DD`) |
| `CHUNKHOUND_EMBEDDING__AZURE_ENDPOINT` | Azure OpenAI endpoint |
| `CHUNKHOUND_EMBEDDING__AZURE_DEPLOYMENT` | Azure OpenAI deployment name |
| `CHUNKHOUND_DATABASE__PROVIDER` | Database backend (`duckdb` or `lancedb`) |
| `CHUNKHOUND_DATABASE__PATH` | Database storage path |
| `CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB` | Max database size in GB |
| `CHUNKHOUND_LLM_PROVIDER` | LLM provider for research |
| `CHUNKHOUND_LLM_MODEL` | LLM model shorthand that sets both utility and synthesis roles |
| `CHUNKHOUND_LLM_UTILITY_MODEL` | LLM model for utility tasks (fast, lower cost) |
| `CHUNKHOUND_LLM_SYNTHESIS_MODEL` | LLM model for synthesis tasks (primary output) |
| `CHUNKHOUND_LLM_API_KEY` | API key for LLM provider |
| `CHUNKHOUND_LLM_BASE_URL` | Base URL for LLM provider (proxy / custom endpoint) |
| `CHUNKHOUND_LLM_SSL_VERIFY` | Verify TLS certificates for requests sent to `llm.base_url` |
| `CHUNKHOUND_LLM_UTILITY_PROVIDER` | Override provider for utility operations |
| `CHUNKHOUND_LLM_SYNTHESIS_PROVIDER` | Override provider for synthesis operations |
| `CHUNKHOUND_LLM_TIMEOUT` | LLM request timeout in seconds (default: 120) |
| `CHUNKHOUND_LLM_MAX_RETRIES` | Max retry attempts (default: 3) |
| `CHUNKHOUND_LLM_CODEX_REASONING_EFFORT` | Reasoning effort for Codex models (`minimal`, `low`, `medium`, `high`, `xhigh`) |
| `CHUNKHOUND_LLM_CODEX_REASONING_EFFORT_UTILITY` | Reasoning effort override for utility stage |
| `CHUNKHOUND_LLM_CODEX_REASONING_EFFORT_SYNTHESIS` | Reasoning effort override for synthesis stage |
| `CHUNKHOUND_LLM_ANTHROPIC_THINKING_ENABLED` | Enable extended thinking |
| `CHUNKHOUND_LLM_ANTHROPIC_THINKING_MODE` | Thinking mode: `auto`, `off`, `manual`, or `adaptive` |
| `CHUNKHOUND_LLM_ANTHROPIC_THINKING_BUDGET_TOKENS` | Manual-mode thinking budget (min 1024, default: 10000) |
| `CHUNKHOUND_LLM_ANTHROPIC_THINKING_DISPLAY` | Adaptive-mode thinking text: `summarized` or `omitted` |
| `CHUNKHOUND_LLM_ANTHROPIC_INTERLEAVED_THINKING` | Manual-mode interleaved thinking between tool calls |
| `CHUNKHOUND_LLM_ANTHROPIC_EFFORT` | Token-usage effort: `low`, `medium`, `high`, `xhigh`, `max` |
| `CHUNKHOUND_LLM_ANTHROPIC_TASK_BUDGET_TOKENS` | Advisory agentic-loop token budget (beta) |
| `CHUNKHOUND_LLM_ANTHROPIC_PROMPT_CACHING` | Send `cache_control` for prompt caching |
| `CHUNKHOUND_LLM_ANTHROPIC_CACHE_TTL` | Prompt-cache TTL (e.g. `1h`) |
| `CHUNKHOUND_LLM_ANTHROPIC_CONTEXT_MANAGEMENT_ENABLED` | Automatic clearing of tool results and thinking blocks (beta) |
| `CHUNKHOUND_LLM_ANTHROPIC_CLEAR_THINKING_KEEP_TURNS` | Thinking turns to keep when context management clears them |
| `CHUNKHOUND_LLM_ANTHROPIC_CLEAR_TOOL_USES_TRIGGER_TOKENS` | Input-token threshold that triggers tool-result clearing |
| `CHUNKHOUND_LLM_ANTHROPIC_CLEAR_TOOL_USES_KEEP` | Number of recent tool-use pairs to keep after clearing |
| `CHUNKHOUND_LLM_GEMINI_THINKING_LEVEL` | Gemini thinking depth (`low`, `medium`, `high`) |
| `CHUNKHOUND_LLM_GEMINI_THINKING_BUDGET` | Gemini fixed thinking token budget |
| `CHUNKHOUND_INDEXING__EXCLUDE_MODE` | Exclusion mode (`combined`, `config_only`, `gitignore_only`) |
| `CHUNKHOUND_INDEXING__EXCLUDE` | Glob patterns to exclude from indexing |
| `CHUNKHOUND_INDEXING__INCLUDE` | Glob patterns limiting which files are indexed |
| `CHUNKHOUND_INDEXING__CLEANUP` | Remove orphaned DB records after indexing (default: true) |
| `CHUNKHOUND_INDEXING__FORCE_REINDEX` | Force re-indexing of all files (default: false) |
| `CHUNKHOUND_INDEXING__MAX_FILE_SIZE_MB` | Skip files larger than this (MB, default: 10) |
| `CHUNKHOUND_INDEXING__CONFIG_FILE_SIZE_THRESHOLD_KB` | Skip structured config files larger than this (KB, default: 20) |
| `CHUNKHOUND_INDEXING__PER_FILE_TIMEOUT_SECONDS` | Per-file parse timeout (default: 3.0) |
| `CHUNKHOUND_INDEXING__PER_FILE_TIMEOUT_MIN_SIZE_KB` | Only apply per-file timeout to files at least this large (KB, default: 128) |
| `CHUNKHOUND_INDEXING__BATCH_SIZE` | Files per parsing batch (default: 50) |
| `CHUNKHOUND_INDEXING__DB_BATCH_SIZE` | Chunks per database write batch (default: 100) |
| `CHUNKHOUND_INDEXING__MAX_CONCURRENT` | Max concurrent parser workers (default: 5) |
| `CHUNKHOUND_INDEXING__CHUNK_OVERLAP` | Internal chunk overlap (default: 50) |
| `CHUNKHOUND_INDEXING__MIN_CHUNK_SIZE` | Internal min chunk size (default: 50) |
| `CHUNKHOUND_INDEXING__INDEX_UNKNOWN_FILES` | Index files with unrecognized extensions as plain text (default: false) |
| `CHUNKHOUND_INDEXING__DETECT_EMBEDDED_SQL` | Enable embedded SQL detection (default: true) |
| `CHUNKHOUND_INDEXING__DISCOVERY_BACKEND` | File discovery backend: `auto`, `python`, `git`, `git_only` (default: `auto`) |
| `CHUNKHOUND_INDEXING__GITIGNORE_BACKEND` | Backend for gitignore evaluation: `python` or `libgit2` (default: `python`) |
| `CHUNKHOUND_INDEXING__CHIGNORE_FILE` | ChunkHound-specific ignore file name (default: `.chignore`) |
| `CHUNKHOUND_INDEXING__GIT_PATHSPEC_CAP` | Max git pathspec entries (default: 128) |
| `CHUNKHOUND_INDEXING__MTIME_EPSILON_SECONDS` | Tolerance for file mtime comparison (seconds, default: 0.01) |
| `CHUNKHOUND_INDEXING__PARALLEL_DISCOVERY` | Enable parallel directory traversal for large codebases (default: true) |
| `CHUNKHOUND_INDEXING__MIN_DIRS_FOR_PARALLEL` | Minimum top-level directories to activate parallel discovery (default: 4) |
| `CHUNKHOUND_INDEXING__MAX_DISCOVERY_WORKERS` | Maximum worker processes for parallel discovery (default: 16) |
| `CHUNKHOUND_INDEXING__WORKSPACE_GITIGNORE_OVERLAY` | Apply CH root .gitignore as global overlay across repos (default: false) |
| `CHUNKHOUND_INDEXING__WORKSPACE_GITIGNORE_NONREPO` | Use CH root .gitignore only for non-repo paths (default: true) |
| `CHUNKHOUND_INDEXING__REALTIME_BACKEND` | Filesystem monitoring backend: `watchman`, `watchdog`, or `polling` |
| `CHUNKHOUND_DB_EXECUTE_TIMEOUT` | Database executor timeout |
| `CHUNKHOUND_YAML_ENGINE` | YAML parser engine (`rapid` or `tree`) |
| `CHUNKHOUND_CONFIG_FILE` | Path to config file (alternative to `--config`) |
| `CHUNKHOUND_WEBSEARCH_TIMEOUT_SECONDS` | Web search subprocess timeout in seconds (default: 600) |
| `CHUNKHOUND_DEBUG` | Enable debug logging |
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

> **Migration note:** Do not set `llm.provider` to `"ollama"`.
> ChunkHound treats Ollama as an OpenAI-compatible endpoint, so use
> `provider: "openai"` with the Ollama `base_url` and an explicit `model`.

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

## Web Search

The `websearch` tool searches the web via DuckDuckGo, fetches the top pages, indexes the fetched content in memory, and runs the same deep research pipeline used for local code search. It is available as an MCP tool and as `chunkhound websearch`.

### Requirements

The web search tool requires all three provider capabilities to be configured:

- **Embedding provider** — e.g. `embedding.provider: "voyageai"` or `"openai"`
- **LLM provider** — for query expansion and answer synthesis
- **Reranking** — `embedding.rerank_model` must be set for relevance-aware multi-hop search

If any of these are missing, the MCP `websearch` tool is not registered (capability gating) and the CLI command will fail.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Natural-language or keyword query sent to DuckDuckGo |
| `--limit` / `limit` | int | 30 | Max results to fetch (1–100). CLI uses `--limit`, MCP uses `limit`. |

### Environment Variables

| Variable | Description |
|---|---|
| `CHUNKHOUND_WEBSEARCH_TIMEOUT_SECONDS` | Wall-clock timeout (seconds) for the research subprocess. Default: 600. Also returned for malformed values. |

### Fixed Constants

| Constant | Value | Description |
|---|---|---|
| `MAX_FETCH_CONCURRENCY` | `5` | Max concurrent page fetches (defined in `chunkhound.utils.websearch_core`) |
| `WEBSEARCH_LIMIT_MAX` | `100` | Upper bound for the `--limit` / `limit` parameter |

### Browser Dependency

The fetch path uses **zendriver** (v0.15.3, core dependency — no extra install needed) to drive the system-installed Google Chrome for rich page rendering. Chrome >=124 is required. If Chrome is not found or too old, fetches fall back to `urllib` (less capable — may miss JS-rendered content and cannot fetch PDFs).

### Research Config Linkage

The web search tool delegates to the same deep research pipeline as `code_research`. All settings in the [Research Configuration](#research-configuration) section apply: `algorithm`, `multi_hop_time_limit`, `relevance_threshold`, `query_expansion_enabled`, `target_tokens`, etc.

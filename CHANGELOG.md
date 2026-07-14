# Changelog

All notable changes to ChunkHound will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [5.2.0] - 2026-07-12

### Breaking Changes
- **DeepSeek, Grok, and Gemini require explicit model** — The baked-in default
  models (`deepseek-v4-flash`, `grok-4-1-fast-reasoning`, and Gemini's prior
  fallback/default path) have been removed. Users must now set `llm.model` (or
  per-role model override) explicitly for these providers.

### Added
- **Rust-native file scanner** — Directory scanning can now use a Rust extension (`chunkhound_native`) built on the `ignore` crate's parallel `WalkBuilder`, replacing the Python `os.walk`-based scanner. Enabled by default when the native extension is installed (`CHUNKHOUND_USE_RUST=0` to opt out); falls back to the Python scanner automatically when the extension is absent or for patterns the fast path can't handle.
- **Database compaction** — DuckDB databases now auto-compact via an atomic swap when fragmentation crosses `database.fragmentation_threshold_pct`, reclaiming disk space with no indexing downtime and no disruption to concurrent MCP/daemon access.
- **PowerShell language support** — `.ps1`/`.psm1` files are now parsed with function/class/method/property/enum-aware chunks instead of the generic text fallback.
- **Metal language support** — `.metal` shader files (MLX-style kernels) now parse via the C++ grammar for struct/function-aware chunks.
- **HAML indexing** — `.haml` files, previously skipped entirely, are now indexed and searchable as text.
- **Global configuration file** — Cross-project defaults (embedding provider/key, LLM configuration, research options, etc.) can now be set once via a global config file, loaded from `CHUNKHOUND_GLOBAL_CONFIG_FILE` or auto-discovered from standard user locations.
- **Python 3.14 support** — ChunkHound now runs on Python 3.14; `tree-sitter-language-pack` upgraded 0.9.0 → 0.13.0 and DuckDB pinned to 1.4.4.
- **Gemini configurable thinking** — The Gemini provider now supports configurable `thinking_level`/`thinking_budget` parameters.
- **`start-proxy.sh`** — Helper script for running ChunkHound's stdio MCP server behind a streamable-HTTP proxy, for MCP clients that require HTTP transport.
- **Git diff search** — Semantic search over git history: `--last-n <N>`, `--commit-range <A..B>`, `--commit-hash <hash>`. Available in CLI (`chunkhound search`, `chunkhound research`) and MCP tools (`search`, `code_research`). `--vector-source` controls scope: `diff` (default), `both` (merge with DB index), `db` (DB only). Staged-changes search (`git diff --cached`) is not included in this release.
- **Claude Opus 4.8 support**: the Anthropic provider now gives Opus 4.8 full Opus 4.7 capability parity: adaptive-only extended thinking (an explicit `manual` request auto-resolves to adaptive), effort levels `low`/`medium`/`high`/`xhigh`/`max`, and the task-budgets beta. The pinned `claude-opus` offline fallback was bumped to `claude-opus-4-8`. Fixes a `400 "thinking.type.enabled is not supported for this model"` error when targeting `claude-opus-4-8` with thinking enabled.
- **Websearch CLI command and MCP tool** — New `chunkhound websearch "<query>"` CLI subcommand and matching `websearch` MCP tool let coding agents pull external context into the loop without leaving it. Searches DuckDuckGo, fetches the top results, converts them to Markdown, runs `code_research` over a transient in-memory index of the fetched pages, and returns a single cited answer with local tmpdir paths rewritten back to source URLs. Per-request timeout configurable via `CHUNKHOUND_WEBSEARCH_TIMEOUT_SECONDS` (default 600s). `markdownify` and `zendriver==0.15.3` are new core dependencies.
  - **System Chrome via zendriver over CDP** — Rich page fetches drive the system-installed Google Chrome directly through zendriver's raw CDP bindings. No Node runtime, no bundled Chromium download, no `playwright install` step, and no `chunkhound[browser]` extra to opt into — it works out of the box when Chrome ≥124 is present. Chrome resolution probes known install paths and version-checks the binary (zendriver's CDP binding requires `Response.charset`, which only Chrome ≥124 emits); any verification failure (missing binary, too old, unparseable `--version`) collapses to a warning and the urllib fallback rather than raising.
  - **Robust PDF handling** — PDF detection inspects real response headers before Chrome's PDF viewer engages, verifies the `%PDF-` magic bytes (so endpoints returning HTML under an `application/pdf` Content-Type fall through to HTML→markdown instead of corrupting the result set), and refetches PDFs via urllib with Chrome's cookies forwarded so cookie-gated PDFs (signed URLs, session-protected docs) still work — redirects on the cookie-bearing request are blocked to prevent cross-origin credential leakage. Fetches whose rendered markdown is whitespace-only fail explicitly instead of writing zero-byte files into the result set.
  - **LLM query expansion** — Each websearch query is now expanded via the utility LLM into 3 DuckDuckGo-optimized variants, run and merged with URL-based deduplication, improving result coverage for a single search. Falls back to the original single query when no LLM is configured or the expansion call fails.
- **`chunkhound mcp --read-only` (DuckDB)** — Point an MCP client at a pre-indexed database — shared, mounted read-only, or otherwise untouchable — without risking writes from the indexer or daemon. Regex search still works; writes through the provider raise. Also exposed as `database.read_only`; rejected for non-`mcp` subcommands.

### Changed
- **DeepSeek/Grok refactored to data-driven registry** — Per-provider
  subclasses replaced with a unified `OpenAICompatibleSpec` registry.
  Adding a new OpenAI-compatible provider is now a data entry + a few
  type annotations instead of a full subclass. Net: -527 lines.
- **Vue template metadata schema for v-model directives** — The shape of metadata produced for `v-model` (and `v-model:arg`) directives has changed:
  - Old: `model_modifier` (string)
  - New: `modifiers` (list of strings) + `model_argument` (optional string for `v-model:foo` style bindings)

  The previous `model_modifier` field was already semantically incorrect on many real cases (it was derived from the tree-sitter `directive_argument` node and could not properly distinguish arguments from modifiers or represent multiple modifiers). The new fields are more correct and consistent with how `parse_vue_directive` works. This is a breaking change for any code or LLM prompts that directly inspected `chunk.metadata["model_modifier"]` on Vue chunks. Most other Vue directive metadata keys (`event_name`, `property_name`, `slot_name`, etc.) are unchanged in meaning.

### Enhanced
- **Depth exploration research performance** — Deep code research's Phase 1.5 depth-exploration step no longer fires one concurrent HTTP rerank call per exploration query (which could overload the reranking server); it now scores chunks via in-database cosine similarity, tracks found chunks globally across iterations to avoid re-scanning covered ground, and parallelizes gap-cluster unification — reducing wall-clock time on large research runs while improving output coverage. Per-step timings are now visible in the terminal progress tree, and the exploration query token budget is configurable.

### Performance
- **Skipped-file re-scan avoidance** — Files skipped during parsing (binary, unknown type, oversized config, timeout) are now recorded in the database with a skip reason, so unchanged skipped files are bypassed on subsequent indexing runs instead of being re-queued and re-skipped every time — saves up to ~10 minutes per run on large codebases.

### Fixed
- **Multi-source URL provenance in fact extraction** — URL-backed facts in
  multi-source clusters are now correctly skipped instead of being stored
  with the URL as a file-path, which broke file-scoped retrieval. New-style
  facts missing a `location` field are also skipped instead of defaulting
  to line 1-1.
- **MCP websearch traceback pollution** — Error output from research
  subprocesses strips internal Python traceback frames, keeping only the
  last meaningful error line.
- **MCP string return passthrough** — Tools returning raw strings (e.g.
  websearch results) now pass through as markdown instead of being wrapped
  in JSON objects.
- **OpenAI-compatible embedding config forwarding** — `timeout`, `batch_size`, and `max_retries` config values are now correctly forwarded to the OpenAI embedding provider factory path; previously they were silently dropped and every OpenAI-compatible provider ran with hardcoded defaults regardless of configuration.
- **Indexing batch_size forwarding** (issue #244) — `embedding_batch_size`, `db_batch_size`, and `max_concurrent_batches` config are now honored by `process_directory`, fixing silently-ignored `batch_size: 1` workarounds for limited providers.
- **tiktoken special-token crash** (issue #315) — Source files containing tokenizer special-token literals (e.g. `<|endoftext|>`) no longer abort the entire embedding batch with a "disallowed special token" error.
- **tiktoken stall on blocked networks** — Token estimation no longer stalls on the connect timeout when the tiktoken BPE-file download host is network-blocked; falls back to a character-count heuristic and negative-caches the failure so the warning fires at most once per model per process.
- **Embedding rate-limit backoff** (issue #298) — 429 responses now honor the `retry-after`/`x-ratelimit-reset-requests` header (or a parsed "try again in X seconds" message) with exponential backoff capped at 120s and jitter, instead of a flat 1–2s linear retry that hammered the provider.
- **Orphaned subprocess cleanup** — OpenCode CLI, the internal `_quickresearch` helper, and the MCP proxy no longer leave orphaned child processes behind when their parent exits, is cancelled, or dies unexpectedly.
- **Claude Code CLI subprocess auto-updates** — The Claude Code CLI auto-updater is now disabled for ChunkHound's subprocess invocations, preventing unexpected version changes mid-run.
- **Path-filter hidden file/directory misclassification** — Hidden directories (e.g. `.github`) and hidden files with extensions (e.g. `.eslintrc.js`) are no longer misclassified against each other in `path_filter` matching for regex/semantic search.
- **Multi-hop search result cap** — Multi-hop search now correctly bounds its initial fetch by `result_limit` (previously bypassed it), and a crash when `result_limit` is `None` in exhaustive mode is fixed.
- **LanceDB migration on PyArrow 23** — Metadata is now serialized correctly during schema migration, preventing an `ArrowTypeError` when restoring existing chunks after an embedding schema change.
- **Gitignore respected at repository roots** — Directory scans now honor `.gitignore` rules, including nested-gitignore-driven pruning, when scanning from a repository root; previously some ignored trees were still walked.

### Security
- **Path-filter SQL substring leak** — Fixed a path-filter normalization bug in `DuckDBProvider` where substring matches could leak results across directory boundaries (e.g. a filter for `src` could match `src_utils/`).

## [5.1.0] - 2026-05-20

### Breaking Changes
- **MCP `search` response format changed to markdown** — The `search` tool now returns lean
  markdown strings instead of JSON objects, with syntax-highlighted code fences, similarity
  percentages (semantic search), and a pagination footer. MCP clients that parse raw search
  output as JSON must migrate to the new format.

### Added
- **`--index-unknown-files` flag** — Files with unrecognized extensions are now indexable as
  plain text (binary files are still skipped). Enabled via `--index-unknown-files` CLI flag,
  `indexing.index_unknown_files` config key, or `CHUNKHOUND_INDEXING__INDEX_UNKNOWN_FILES` env var.
- **Proto, GraphQL, XML, config, and Dockerfile support** — `.proto`, `.graphql`, `.gql`,
  `.xml`, `.ini`, `.properties`, `.conf`, `.cfg`, and extensionless `Dockerfile`/`Jenkinsfile`
  files are now indexed by default. `.env` files are explicitly excluded to prevent secret leakage.
- **`chunkhound.ai` onboarding** — Interactive CLI setup is replaced with guided onboarding at
  chunkhound.ai; local backend is now configured explicitly rather than through prompts.

### Fixed
- **MCP startup HNSW crash** — MCP server no longer fails with a `CreateDeltaIndex` assertion
  on startup against databases missing the unique `(chunk_id, provider, model)` index from v5.0.0.
  HNSW recreation now runs outside the transaction (issue #280).
- **WAL validation HNSW crash** — WAL pre-flight validation now uses in-memory+ATTACH, preventing
  a C++ abort when the WAL contained HNSW operations from a prior session (issue #273).
- **`--db` nested directory bug** — Passing an explicit file path (e.g. `--db /path/to/chunks.db`)
  no longer creates a `chunks.db/chunks.db` nested directory; known DB extensions (`.db`,
  `.duckdb`) are now correctly identified as file paths (issue #215).
- **Parser install hints** — C# error messages now show the correct PyPI package
  `tree-sitter-c-sharp` (was `tree-sitter-csharp`); Makefile shows `tree-sitter-make`;
  SCSS points to `tree-sitter-language-pack` (issue #267).

## [5.0.0] - 2026-05-05

### Breaking Changes
- **Config precedence reordered** — Local `.chunkhound.json` now takes precedence over
  environment variables. If you relied on env vars overriding project-level settings,
  use CLI arguments instead.
- **`--config` now overrides local `.chunkhound.json`** — Previously, a project-local
  `.chunkhound.json` took precedence over an explicit `--config` path. Now `--config`
  wins. If you relied on local `.chunkhound.json` shadowing a shared config file, move
  that override into CLI arguments.
- **Missing `--config` / `CHUNKHOUND_CONFIG_FILE` path now raises** — A non-existent
  config file path used to be silently ignored; it now raises `ValueError` with an
  actionable message.
- **`DEFAULT_LLM_TIMEOUT` doubled** — Default LLM request timeout increased from 60 s
  to 120 s for all providers (was already 120 s for Gemini; now uniform).
- **HTTP MCP server removed** — ChunkHound now supports stdio transport only for MCP connections
  - `chunkhound mcp http` command removed
  - `--http`, `--port`, `--host` CLI flags removed
  - FastMCP dependency removed
  - Migration: Use `chunkhound mcp` (stdio) instead. All major MCP clients (Claude Code, Claude Desktop, VS Code) support stdio transport.
  - Rationale: Simplified codebase, reduced dependencies, focused on primary use case (stdio is the standard for MCP)
- **Unsupported file types no longer indexed as plain text** — Files with unrecognized extensions are now skipped instead of being force-parsed as plain text. Files with known text extensions (.txt, .log, .cfg, .conf, .ini) are unaffected.
- **Claude Code CLI default model changed** from `claude-sonnet-4-5-20250929` to `claude-haiku-4-5-20251001`. Users who relied on the default model will see different cost/quality characteristics. Set `llm.model`, `llm.utility_model`, or `llm.synthesis_model` explicitly to retain previous behavior.
- **Anthropic provider upgraded to Claude Opus 4.7/4.6 and Sonnet 4.6**
  - `anthropic` dependency minimum bumped to `>=0.96.0,<1.0.0`
  - Default Anthropic utility and synthesis models changed to ChunkHound's `claude-haiku` sentinel. This is intentional: current Claude Haiku is capable enough for synthesis, is Anthropic's cheapest available Claude model, and Anthropic does not currently offer a true low-cost utility tier. Users who prefer maximum synthesis quality can override `synthesis_model`.
  - Default Claude Code CLI model changed to the same `claude-haiku` sentinel. ChunkHound still honors its Claude env overrides first; otherwise it preserves the sentinel so Claude Code can resolve the latest matching alias itself.
  - Removed module symbols `BETA_EFFORT` and `EFFORT_SUPPORTED_MODELS`. Callers should use the `supports_effort(model)` / `supports_effort_level(model, level)` predicates instead.
  - `thinking_enabled=True` with `thinking_mode="auto"` resolves to adaptive only for adaptive-capable models such as Opus 4.6/4.7, Sonnet 4.6, and Mythos. The pinned Haiku fallback remains manual-mode thinking.
  - `anthropic_prompt_caching` defaults to `false` because ChunkHound requests rarely reuse prompt prefixes enough to offset Anthropic cache-write costs. To opt in, set `CHUNKHOUND_LLM_ANTHROPIC_PROMPT_CACHING=true` or pass `--llm-anthropic-prompt-caching`.
  - Invalid `thinking_mode` values and sub-20000 `task_budget_tokens` now raise `ValueError` instead of warning-and-coercing.

### Added
- **Elixir language support** — Full Elixir parsing (32nd language) via tree-sitter-elixir: modules, functions, macros, protocols, structs, specs, and import/alias/require statements.
- **TwinCAT/Structured Text parser** — IEC 61131-3 Structured Text (`.TcPOU`) files for PLC development are now fully searchable.
- **HTML, CSS, SCSS, and Jinja parsers** — Full tree-sitter parsing for web languages: HTML (`.html`, `.htm`, `.xhtml`), CSS (`.css`), SCSS/Sass (`.scss`, `.sass`), and Jinja templates (`.jinja`, `.j2`, `.njk`, `.erb`, `.ejs`). SCSS preprocessing handles `#{...}` interpolations for correct AST byte offsets. Import resolution is supported for all four languages.
- **Grok (xAI) LLM provider** — xAI Grok models are now supported for deep code research via the `code_research` tool.
- **Matryoshka embeddings** — OpenAI and VoyageAI providers now support Matryoshka truncation for flexible vector dimensions; default OpenAI model upgraded to `text-embedding-3-large`.
- **`openai_compatible` embedding provider** — Connect any OpenAI-compatible embedding endpoint with configurable SSL verification, auth, and dimension support.
- **Azure OpenAI embeddings** — Native Azure OpenAI embedding support with `azure_endpoint`, `api_version`, and `azure_deployment` configuration options.
- **VoyageAI ranking support** — VoyageAI provider now supports reranking for improved search result quality.
- **Claude Opus 4.7 / Opus 4.6 / Sonnet 4.6 support** — Adaptive thinking mode (auto / off / manual / adaptive selector), expanded effort levels (`low`, `medium`, `high`, `xhigh` (Opus 4.7 only), `max` (4.6+)), opt-in prompt caching with configurable TTL (`5m` / `1h`), and the task-budgets beta (Opus 4.7 only, advisory cap for agentic loops, min 20000 tokens).
- **New `LLMConfig` fields** — `anthropic_thinking_mode`, `anthropic_thinking_display`, `anthropic_prompt_caching`, `anthropic_cache_ttl`, `anthropic_task_budget_tokens` (and matching `CHUNKHOUND_LLM_ANTHROPIC_*` env vars and `--llm-anthropic-*` CLI flags). The pre-existing `anthropic_thinking_enabled`, `anthropic_thinking_budget_tokens`, `anthropic_interleaved_thinking`, `anthropic_effort`, `anthropic_context_management_enabled`, and `anthropic_clear_*` fields are now also readable from env and CLI.
- **Embedded SQL detection** — SQL embedded in string literals is detected and indexed by default across Python, Java, JavaScript, TypeScript, C#, Go, Rust, and PHP. Disable with `--no-detect-embedded-sql` or `CHUNKHOUND_INDEXING__DETECT_EMBEDDED_SQL=false`.
- **OpenAI Responses API** — Deep code research now supports reasoning models (gpt-5.1, gpt-5.1-codex, o-series, gpt-5-pro) via the Responses API, with automatic routing based on model compatibility across 30+ models.
- **Reasoning effort control** — Configurable LLM reasoning effort (`none`/`minimal`/`low`/`medium`/`high`) for deep research via `CHUNKHOUND_LLM_CODEX_REASONING_EFFORT` with per-role overrides.
- **Structured JSON output** — Responses API maintains schema validation consistency across both Chat Completions and Responses endpoints.
- **Multi-client MCP daemon** — Multiple MCP clients can share a single DuckDB connection via a background daemon, eliminating lock conflicts in multi-session workflows.
- **`--perf-diagnostics` mode** — `chunkhound index --perf-diagnostics` collects per-batch timing metrics and detects performance regressions via linear regression and z-score analysis, outputting a JSON diagnostics file.
- **`--path-filter` for research** — `chunkhound research --path-filter <dir>` scopes deep code research to a subdirectory.
- **PHP config-literal parsing** — PHP files with top-level `return [...]` arrays are now searchable.
- **Universal config-literal parsing** — Exported configuration objects and arrays in Python, JavaScript, TypeScript, and JSX/TSX are now discoverable through semantic search.
- **Watchman live-indexing operator docs** — Documents the private `.chunkhound/watchman/` sidecar, fail-fast startup/no-implicit-fallback behavior, `daemon_status` health interpretation, and the rollout/default-switch gate for making Watchman the primary backend.
- **Dart language support** — `.dart` files are now fully searchable via tree-sitter parsing: classes, functions, methods, constructors, and import/export statements (33rd language).
- **Lua language support** — `.lua` files are now parsed and indexed via tree-sitter, covering functions, tables, and module patterns.
- **T-SQL (SQL Server) parser** — SQL Server T-SQL (`.sql`) files are now fully parsed and searchable via tree-sitter.
- **`chunkhound autodoc` command** — Generates a static Astro documentation site from codebase research, with provenance citations linked to source references and byte-stable output across platforms.
- **`chunkhound codemap` command** — Maps areas of interest (POIs) in a codebase through deep code research; the `-j` flag enables parallel POI processing with automatic backoff to serial on failure.
- **Configurable disk storage limit** — `database.max_disk_usage_mb` config option (`--max-disk-usage-gb` CLI flag, `CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB` env var) caps database growth and raises a clear error instead of filling the disk.
- **Anthropic native structured outputs** — Anthropic provider now uses the `structured-outputs-2025-11-13` beta API for guaranteed schema-compliant JSON via constrained decoding, with type-safe Pydantic model responses and extended thinking compatibility.
- **Global gitignore support** — ChunkHound now reads the user's global gitignore file (via `git config --global core.excludesFile`) when building the exclusion list during indexing.

### Changed
- **Watchman default backend** — Watchman is now the default realtime backend on supported native-runtime platforms; `watchdog` and `polling` remain explicit fallback backends.

### Enhanced
- **MCP tool routing** — `code_research` and `search` tool descriptions rewritten for improved LLM routing; cross-references between tools are shown or hidden dynamically based on whether an LLM provider is configured.
- **Daemon overlap guard** — A user-scoped daemon registry is now validated against each project's `daemon.lock` before startup, preventing live parent/child root overlaps (e.g., running daemons for `/workspace` and `/workspace/project` simultaneously). Exact-root reuse across restarts is preserved; sibling roots are allowed.
- **`ChunkType.IMPORT`** — Import statements across all languages now use a dedicated chunk type instead of falling through to `UNKNOWN`, improving search precision.
- **Chunk size enforcement** — All parsers now enforce a central size guard before DB persistence; oversized chunks are split automatically, preventing embedding API failures.
- **Windows compatibility** — Cross-platform temp directory handling for Claude Code CLI provider; `shutil.which` replaces Unix-only `which` for git binary detection.
- **Version management** — Supports PEP 440 pre-release formats (alpha, beta, RC) with safety checks to prevent accidental releases from uncommitted work.
- **Multi-client MCP daemon — index lock conflict handling** — `chunkhound index` now detects a running daemon's lock file on DuckDB conflict: a healthy daemon prints an informational message and exits cleanly; an unresponsive daemon prompts the user to kill it and retry.
- **Python import resolution** — Import statements are now resolved more accurately in Python code research, improving cross-file symbol discovery.

### Performance
- **LanceDB dimension detection** — Table creation now detects embedding dimensions upfront from the configured provider, eliminating the O(n) table recreation penalty during first embedding insertion for large codebases (e.g. 16,000+ chunks no longer require full table migration).

### Fixed
- **Cross-repo data loss** — Re-indexing a subdirectory in a shared workspace no longer deletes other repositories' data from the database (fixes #87).
- **Global gitignore false exclusions** — `~/.gitignore` was incorrectly used as a global excludes fallback, causing all files to be excluded when a dotfiles repo contained broad patterns like `*` (fixes #216).
- **MCP startup error visibility** — DuckDB lock conflicts and config validation errors now surface as JSON-RPC error responses instead of silently exiting, with a specific hint to kill stale processes on lock conflicts.
- **Gemini LLM timeout** — All `code_research` calls no longer fail immediately; the 120s timeout was being passed as 120ms to the google-genai SDK.
- **Gemini LLM initialization** — Gemini provider no longer fails to register when `base_url` is present in config, restoring `code_research` availability.
- **VoyageAI `api_base`→`base_url`** — voyageai ≥0.3.7 renamed the parameter; ChunkHound now detects the correct key at runtime, preventing Azure ML endpoint rejections.
- **`tree-sitter-language-pack` 1.0.0 incompatibility** — Pinned to `<1.0.0` to prevent fresh installs from pulling the breaking release that made YAML, MATLAB, Swift, and other language-pack parsers fail at startup.
- **Global chunk deduplication** — YAML and Universal parsers now participate in chunk deduplication, preventing duplicate chunk IDs that caused indexing failures on repeated config values.
- **`hdbscan` startup crash under numpy 2.x** — Replaced `hdbscan` package (which uses the numpy 1.x ABI) with `sklearn.cluster.HDBSCAN` (already a dependency), eliminating MCP daemon startup failures on systems running numpy 2.x.
- **Windows MCP unicode safety** — MCP server stdout on Windows is now reconfigured with `errors='backslashreplace'` to prevent crashes when source files contain non-UTF-8 bytes; applied to both `main()` and `main_sync()` entry points (fixes #225).
- **HDBSCAN outlier cluster assignment** — Outliers in Phase 2 cluster merging were mapped to incorrect final cluster indices, causing code research results to be grouped with unrelated code. Fixed by threading the cluster-id-to-final-index mapping through the outlier merge step.
- **Symlink path preservation** — Worktree and repository symlink paths are now stored as their symlink paths during indexing instead of being silently resolved to their targets (fixes #102).

### Removed
- **`CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY`** — Database optimization now runs once at indexing end; the per-batch frequency config option is removed.

## [4.0.1] - 2025-11-12

### Fixed
- Package build configuration now excludes test fixtures from distribution, reducing package size and removing unnecessary test data from published releases

## [4.0.0] - 2025-11-12

### Added
- Map-reduce synthesis for dramatically improved research accuracy - clusters related files and synthesizes them separately before combining insights
- Compact numbered citation system `[1][2][3]` replacing verbose `file.py:123` references for better readability
- Automatic query expansion with intelligent deduplication to find more relevant results
- Structured JSON output support for LLM providers enabling programmatic research workflows
- Tree progress display with event system for visual research feedback
- `chunkhound research <query>` command for direct code research without starting MCP server
- `chunkhound index --simulate [--json]` - Dry-run mode showing which files would be indexed without making changes
- `chunkhound diagnose [--json]` - Troubleshooting command comparing ChunkHound's decisions vs git's ignore rules
- `chunkhound calibrate` - Automatic batch size performance tuning for Qwen3 reranker
- `--show-sizes` flag for file size reporting during indexing
- Swift language support with tree-sitter parsing for classes, protocols, functions, and properties (`.swift`, `.swiftinterface`)
- Objective-C support with content detection to disambiguate from MATLAB (`.m` files)
- Zig language support with comprehensive tree-sitter parsing
- Haskell language support for functions, types, classes, and modules (`.hs`, `.lhs`, `.hs-boot`, `.hsig`, `.hsc`)
- HCL (HashiCorp Configuration Language) support for Terraform with nested object parsing (`.hcl`, `.tf`, `.tfvars`)
- Vue.js Single File Component (SFC) support with specialized parsing for template, script, and style sections
- Svelte Single File Component support with specialized parsing for template, script, and style sections (`.svelte`)
- Vue cross-reference tracking between template elements and script definitions for enhanced semantic understanding
- PHP language support with comprehensive parsing for classes, interfaces, traits, functions, methods, namespaces, and PHPDoc comments
- RapidYAML parser using native bindings (10-100x faster than tree-sitter for large YAML files)
- Helm template sanitizer for Go template syntax in Kubernetes manifests
- Automatic fallback to tree-sitter parser when RapidYAML encounters issues
- Benchmark harness comparing PyYAML, universal, and RapidYAML performance (`scripts/bench_yaml.py`)
- Repo-aware ignore engine respecting repository boundaries and preventing rule leakage between sibling repos
- Workspace overlay mode collecting .gitignore rules from root and nested files with correct anchoring
- Combined exclusion modes: `indexing.exclude_mode` supports `"combined"`, `"config_only"`, or `"gitignore_only"`
- Wildcard directory segment matching for patterns like `**/.venv*/` and `**/*.phar/`
- Git pathspec capping with fallback to prevent pathspec explosion (default: 128, env: `CHUNKHOUND_INDEXING__GIT_PATHSPEC_CAP`)
- Real-time telemetry for git pathspec usage and exclusion sources
- TEI (Text Embeddings Inference) reranking format support alongside Cohere format
- Automatic reranker format detection from response field names (Cohere vs TEI)
- Thread-safe format caching for consistent reranker behavior across requests
- Authorization header support for TEI endpoints with `--api-key` flag
- Qwen3 reranker with automatic batch size calibration for optimal performance
- Async regex search methods for concurrent search operations
- Claude Code CLI provider with direct integration (`claude-code-cli`)
- Codex CLI provider for synthesis workflows
- AWS Anthropic Bedrock provider using official Anthropic SDK
- Provider-specific synthesis concurrency limits: OpenAI (3), Bedrock (5), Claude CLI (1)
- Smart change detection using checksums for verification when mtime/size differ
- Content hash support in both DuckDB and LanceDB providers
- DuckDB schema migration with `files.content_hash` column (idempotent via `ALTER TABLE IF NOT EXISTS`)
- LanceDB execute_query adapter for lightweight batch SELECT operations
- In-memory database mode for simulate on fresh workspaces (no .chunkhound/ directory created)
- Checkpointing and recovery for more robust indexing coordinator
- Per-file timeout controls: `indexing.per_file_timeout_seconds`, `indexing.per_file_timeout_min_size_kb`
- Configurable host parameter for HTTP MCP server (`--host` for binding to specific interfaces)
- Size-based filtering threshold for structured config files (JSON/YAML/TOML)
- Environment variable override for DB executor timeout: `CHUNKHOUND_DB_EXECUTE_TIMEOUT`
- Comprehensive test suites for Swift, Objective-C, Zig, Java, C#, Python, PHP, Vue, HCL
- Test fixtures for refactored research modules with fake providers and better mocks

### Enhanced
- Native git bindings for gitignore exclusions replacing Python-based pattern matching (10-100x faster indexing)
- Parallel directory discovery with auto-scaling for enterprise monorepos
- Concurrent file parsing using ProcessPoolExecutor across CPU cores
- Lazy parser instantiation reducing startup time
- Single-file fast path using in-process handling (no ProcessPool overhead)
- Single-read checksum verification eliminating redundant file I/O
- Provider-aware embedding concurrency: OpenAI (8 concurrent batches), VoyageAI (40 concurrent batches)
- Automatic retry logic for VoyageAI embedding provider
- Real-time embedding pass: dedicated "embed" phase after quick parse/store for new chunks
- Removed redundant reranking passes from deep research pipeline
- xxHash3-64 replacing SHA-256 for faster file change detection
- Git pathspec capping preventing pathspec explosion (configurable via env)
- In-memory DuckDB for simulate mode on fresh workspaces
- Automatic parser worker auto-scaling to CPU count when timeouts enabled (capped at 32)
- Split progress reporting: "Parsing files" vs "Handling files" with live cumulative info
- Better error messages and truncation detection for LLM responses
- Non-TTY progress fallback properly working in CI environments
- Improved diagnostics for parse/store errors with clearer failure messages
- Post-run prompt to add timed-out files to `indexing.exclude` when interactive
- Skipped file counts broken out into "Unchanged" and "Filtered" buckets
- Raw markdown output from code_research tool for better formatting in Claude
- Lazy imports for MCP-safe stdio operation
- Proper JSON-RPC handshake reliability
- Test-mode patches for Codex CLI integration (env-gated, no production impact)
- Increased startup wait time for Mac CI stability (3s → 5s)
- TEI reranking format comprehensive guide in CLAUDE.md
- Test coverage documentation with refactoring progress
- README improvements with startup profile CAP notes and exclusions section updates
- Benchmark instructions for YAML parser performance testing
- MCP setup improvements with multi-client support and `--show-setup` flag

### Changed
- **BREAKING**: Removed `depth` parameter from `code_research` MCP tool - system now auto-scales synthesis budgets based on repository size
- **BREAKING**: Checksum algorithm switched from SHA-256 to xxHash3-64 for faster file change detection - all files will be reindexed on first run after upgrade
- **BREAKING**: Default exclusion behavior changed - providing `indexing.exclude` list no longer disables .gitignore (use `exclude_mode: "config_only"` for legacy behavior)
- **BREAKING**: RapidYAML is now the default YAML parser (set `CHUNKHOUND_YAML_ENGINE=tree` to revert to tree-sitter)
- **BREAKING**: LanceDB provider now requires `content_hash` column in files schema
- Default per-file timeout enabled: `indexing.per_file_timeout_seconds=3.0` (previously `0`, disabled)
- Parser workers auto-scale to CPU count when timeouts enabled (capped at 32)
- Combined exclusion mode is now default: overlays gitignore + config excludes instead of replacing
- Model defaults updated to Haiku 4.5 for claude-code-cli and bedrock providers
- Deep research service refactored into specialized modules: question_generator, synthesis_engine, budget_calculator, citation_manager, quality_validator
- Search service refactored into strategies: context_retriever, single_hop_strategy, multi_hop_strategy, result_enhancer
- Extracted research pipeline modules: unified_search, query_expander, file_reader, context_manager

### Fixed
- Fixed double "**/" prefix preventing root file matches in default excludes
- Fixed real-time indexing for newly added languages
- Fixed file diversity collapse in deep research using proper reranking
- Fixed TOML parser to extract only matched node content instead of entire file
- Fixed tree-sitter language names for C# and Makefile parsers
- Fixed .gitignore pattern handling and error logging
- Fixed symbol validation inconsistency in Chunk.from_dict()
- Fixed Config.__init__ to respect target_dir kwarg in tests
- Fixed DuckDB `get_file_by_path(as_model=True)` to return correct mtime and size_bytes for accurate skip checks
- Fixed registry provider instance handling (was storing lambda instead of provider)
- Fixed orphaned embeddings cleanup with proper per-call db_path configuration
- Fixed LanceDB optimize() API usage for 0.21.0+ (cleanup_older_than parameter)
- Fixed single-file indexing to use in-process path and call on_batch for immediate storage
- Fixed missing sources in synthesis by using correct chunk.content field (was chunk.code)
- Fixed flaky multi-hop semantic chain test
- Fixed reranker single-batch top_k filtering for consistency across backends
- Fixed concurrent rerank calls using aiohttp (replaced custom socket-based HTTP)
- Fixed MCP stdio flow for code_research end-to-end reliability
- Fixed non-TTY progress manager regression (added minimal Progress shim for CI)
- Fixed exception classes to allow __traceback__ assignment (removed frozen dataclass)
- Fixed Windows path separator issues in gitignore pattern generation and matching
- Fixed ProcessPoolExecutor segfault on Linux by forcing spawn multiprocessing
- Fixed flaky QA test with file processing completion polling
- Fixed real-time indexing flakiness with proper timeout handling and task cleanup

### Removed
- Removed AWS Bedrock provider (consolidated to Anthropic SDK-based Bedrock provider)
- Removed research tools setup section from CONTRIBUTING.md (obsolete)
- Removed obsolete tests incompatible with refactored modular architecture

### Security
- Removed embedded API key from `.chunkhound.json` - use environment variables instead (e.g., `CHUNKHOUND_EMBEDDING__API_KEY`)

## [3.3.1] - 2025-09-25

### Enhanced
- Dependency updates to latest stable versions for improved stability and performance
- Test infrastructure reliability with better provider detection and error handling

### Fixed
- Tree-sitter 0.25.x API compatibility ensuring parsing works with latest language parsers
- Code formatting and import organization for cleaner, more maintainable codebase

## [3.3.0] - 2025-09-21

### Added
- Official Windows support with full CI testing across Windows, macOS, and Ubuntu
- Command-line search functionality (`chunkhound search`) for semantic and regex queries without starting MCP
- CONTRIBUTING.md guidelines
- Setup wizard when `.chunkhound.json` isn't found in the directory

### Fixed
- File exclude patterns (**/tmp/**) on Linux systems
- Regex search path resolution across platforms

## [3.2.0] - 2025-08-24

### Enhanced
- Semantic search upgraded from two-hop to dynamic multi-hop expansion with intelligent stopping criteria, delivering more comprehensive and contextually relevant results while avoiding search explosion

## [3.1.0] - 2025-08-21

### Added
- PDF document parsing and indexing with full text extraction using PyMuPDF integration

### Enhanced
- Language support expanded to 29 languages with comprehensive documentation breakdown

### Fixed
- JSON file parsing now extracts specific node content instead of entire file content, improving search precision and reducing noise

## [3.0.1] - 2025-08-21

### Enhanced
- Documentation site improved with cross-linking between pages and hero image for better navigation
- OpenAI-compatible endpoint flexibility increased by making API keys optional for local deployments
- Test infrastructure reliability improved with comprehensive CI fixes and timeout handling

### Fixed
- JSON file parsing now handles empty chunks correctly, eliminating indexing failures on common JSON patterns
- Test suite stability enhanced with proper background task cleanup and configuration isolation
- GitHub Actions workflow simplified and made more reliable by removing redundant processes

## [3.0.0] - 2025-08-20

### Added
- VoyageAI embedding provider with advanced two-hop semantic search and reranking capabilities
- GitHub Pages documentation site with interactive examples and improved navigation
- Intelligent file exclusion system with .gitignore support and JSON size filtering
- Advanced makefile parsing with dependency analysis for better code comprehension
- Comprehensive test suite for database consistency and integration testing
- Real-time filesystem indexing with MCP integration for live code monitoring

### Enhanced
- Parsing system completely rebuilt with cAST (Code AST) algorithm for universal language support
- Configuration system dramatically simplified with fewer user-facing options for easier setup
- OpenAI provider unified to handle both standard and custom OpenAI-compatible endpoints
- MCP server reliability improved with proper initialization sequencing and watchdog coordination
- Test infrastructure enhanced with Ollama compatibility and extended timeouts
- Directory indexing consolidated between CLI and MCP with shared service architecture

### Fixed
- MCP server initialization blocking resolved - no more startup deadlocks during directory scanning
- Custom OpenAI endpoint configuration now properly recognized and applied
- Real-time indexing now generates missing embeddings for unchanged code chunks
- SSL verification disabled for custom OpenAI-compatible endpoints to support local deployments
- Watchdog filesystem monitoring no longer blocks MCP server startup process
- MCP server properly respects target directory path arguments across all operations

### Removed
- TEI (Text Embeddings Inference) provider support - simplified provider ecosystem
- BGE provider support - consolidated to core providers for better maintenance
- Legacy parsing system replaced with modern cAST algorithm
- Obsolete configuration documentation and setup files cleaned up

## [2.8.1] - 2025-07-20

### Enhanced
- Architecture documentation significantly improved for better LLM comprehension and AI-assisted development workflows

### Fixed
- Type annotation syntax errors that could cause import failures in Python 3.10+ environments
- Enhanced smoke tests now detect forward reference type annotation issues early

## [2.8.0] - 2025-07-20

### Added
- MCP HTTP transport support alongside stdio transport for flexible deployment options

### Enhanced
- Configuration system unified across CLI and MCP components for consistent behavior
- File change processing reliability improved in MCP servers with better debouncing and coordination
- Database portability enhanced with relative path storage

### Fixed
- MCP server initialization deadlocks and startup crashes resolved with proper async coordination
- File deletion handling improved using IndexingCoordinator for better reliability
- MCP server tool discovery enhanced with fallback logic for better error recovery
- File path resolution improved in DuckDB provider for cross-platform consistency

## [2.7.0] - 2025-07-12

### Fixed
- MCP server now uses configured embedding model instead of hardcoded text-embedding-3-small default, ensuring semantic search works with any configured model
- MCP test environment improvements with comprehensive test data and configuration files

## [2.6.3] - 2025-07-10

### Fixed
- Configuration merge precedence now correctly preserves environment variables over JSON config values
- MCP server semantic search now works properly when running from different directories

### Removed
- Removed obsolete Ubuntu 20 Dockerfile as issue was resolved in configuration system

## [2.6.2] - 2025-07-10

### Fixed
- MCP server now properly loads embedding provider configuration from target directory

## [2.6.1] - 2025-07-10

### Fixed
- MCP server now properly respects CLI-provided project root directory for configuration loading
- Configuration files (.chunkhound.json) are now correctly loaded when running MCP server from different directories

## [2.6.0] - 2025-07-10

### Fixed
- MCP server crashes on Ubuntu and Linux systems when running from different directories by fixing database path resolution and process coordination
- Enhanced TaskGroup error reporting to show underlying causes instead of generic wrapper errors
- Configuration file loading in MCP server now properly respects .chunkhound.json files in target directories
- Database lock conflicts between multiple MCP instances resolved with proper process detection

### Enhanced
- Docker test infrastructure for MCP server validation to prevent future regressions
- Improved error messages for debugging MCP server issues with detailed analysis

## [2.5.4] - 2025-07-10

### Fixed
- MCP server reliability on Ubuntu and other Linux distributions when running from different directories
- Database path resolution consistency across all MCP server components

## [2.5.3] - 2025-07-10

### Fixed
- MCP server communication reliability improved by removing debug logging that interfered with JSON-RPC protocol

## [2.5.2] - 2025-07-10

### Added
- Automatic database optimization during embedding generation to maintain performance with large datasets (every 1000 batches, configurable via `CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY`)

### Fixed
- MCP server compatibility on Ubuntu and other strict platforms by preserving virtual environment context in subprocesses
- OpenAI embedding provider crash on Ubuntu due to async resource creation outside event loop context

## [2.5.1] - 2025-01-09

### Fixed
- Project detection now properly respects CHUNKHOUND_PROJECT_ROOT environment variable, ensuring MCP command works correctly when launched from any directory
- Removed duplicate MCP parser function that could cause confusion

## [2.5.0] - 2025-01-09

### Enhanced
- MCP positional path argument now controls complete project scope - database location, config file search, and watch paths are all set to the specified directory instead of just watch paths

### Fixed
- MCP launcher import path resolution when running from different directories, eliminating TaskGroup errors on Ubuntu and other strict platforms

## [2.4.4] - 2025-01-09

### Fixed
- Ubuntu TaskGroup crash fixed by removing problematic directory change in MCP launcher

## [2.4.3] - 2025-01-09

### Fixed
- MCP server now works correctly when launched from any directory, not just the project root
- Fixed path resolution inconsistencies that caused TaskGroup errors on Ubuntu deployments

## [2.4.2] - 2025-01-09

### Added
- MCP command now accepts optional path argument to specify directory for indexing and watching (defaults to current directory)

### Fixed
- Parser architecture inconsistencies resolved across C, Bash, and Makefile parsers for consistent search functionality
- MCP server database duplication eliminated through proper async task isolation
- LanceDB storage growth controlled with automatic optimization during quiet periods
- MCP server reliability improved with corrected import structure and dependency resolution
- Python parser behavior now consistent between CLI and MCP modes
- Search operation freezes after file deletion resolved with proper thread safety

## [2.4.1] - 2025-01-09

### Fixed
- Package structure consolidated under chunkhound/ directory for improved import reliability and Python packaging best practices

## [2.4.0] - 2025-01-09

### Fixed
- LanceDB storage growth issue resolved with automatic database optimization during quiet periods
- Configuration system project root detection for .chunkhound.json files improved

### Changed
- Enhanced database provider architecture with capability detection and activity tracking
- Modernized configuration system by removing legacy registry config building

## [2.3.1] - 2025-07-09

### Fixed
- MCP server communication reliability improved by preventing stderr output from corrupting JSON-RPC messages
- Enhanced configuration documentation with automatic .chunkhound.json detection examples

## [2.3.0] - 2025-07-08

### Changed
- **BREAKING**: Configuration system completely refactored with centralized management and clear precedence hierarchy
- **BREAKING**: Automatic configuration file loading removed - config files now only load with explicit `--config` flag
- **BREAKING**: Environment variables standardized to `CHUNKHOUND_*` prefix with `__` delimiters (e.g., `CHUNKHOUND_EMBEDDING__API_KEY`)
- **BREAKING**: Legacy `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables no longer supported

### Added
- Complete CLI argument coverage for all configuration options
- Centralized configuration precedence: CLI args → Config file → Environment variables → Defaults
- Comprehensive migration guide for updating existing configurations
- Database file gitignore pattern for Lance database files

### Fixed
- MCP server database duplication caused by shared transaction state across async tasks
- Parser architecture inconsistencies for C, Bash, and Makefile language parsers
- Configuration auto-detection issues that caused deployment complexity

## [2.2.0] - 2025-01-07

### Fixed
- Database freezing during concurrent file operations through proper async/sync boundary handling
- Thread safety issues in DuckDB provider with synchronized WAL cleanup and operation timeouts
- LanceDB duplicate file entries through atomic merge operations and path normalization
- File deletion operations now properly handle async contexts without blocking the event loop

### Changed
- Aligned LanceDB provider with serial executor pattern for consistency with DuckDB
- Improved path normalization to handle symlinks and different path representations
- Enhanced database operation reliability with proper thread isolation

### Added
- Support for complete configuration storage including API keys in .chunkhound.json files
- Consolidated embedding provider creation system for consistent behavior across CLI and config files

## [2.1.4] - 2025-07-03

### Fixed
- CLI argument defaults no longer override config file values
- Updated dependencies via uv.lock

## [2.1.3] - 2025-07-03

### Changed
- Consolidated embedding provider creation to use single factory pattern for consistency
- Reduced embedding provider log verbosity for cleaner output

## [2.1.2] - 2025-07-03

### Fixed
- API key configuration loading from .chunkhound.json files
- Configuration precedence documentation to match actual behavior

### Added
- Complete configuration examples with API key and security guidance

## [2.1.1] - 2025-07-03

### Added
- Centralized version management system for consistent versioning across all components

### Changed
- Simplified version updates through automated scripts
- Enhanced installation and development documentation
- Code formatting improvements and linting cleanup

### Fixed
- Version consistency across CLI, MCP server, and package initialization
- Import statement in package `__init__.py` for better module exposure

## [2.1.0] - 2025-07-02

### Fixed
- Database duplication in MCP server by implementing single-threaded executor pattern
- WAL corruption handling during DuckDB catalog replay
- Parser architecture inconsistencies for C, Bash, and Makefile parsers
- DuckDB foreign key constraint transaction limitations
- Python parser CLI/MCP divergence through unified factory pattern
- Connection management architectural violations

### Changed
- Consolidated database operations through DuckDBProvider executor pattern
- Simplified ConnectionManager to handle only connection lifecycle
- Updated file discovery patterns to include all 16 supported languages
- Removed deprecated connection methods and schema fields
- Enhanced transaction handling with contextvars for task isolation

### Added
- Automatic database migration system for schema updates
- Enhanced parser functionality for C pointer functions and Bash function bodies
- Task-local transaction state management
- Comprehensive executor methods for database operations

## [2.0.0] - 2025-06-26

### Added
- 10 new language parsers: Rust, Go, C++, C, Kotlin, Groovy, Bash, TOML, Makefile, Matlab
- Search pagination with response size limits
- Registry-based parser architecture
- MCP search task coordinator
- Test coverage for file modification tracking
- Comment and docstring indexing for all language parsers
- Background periodic indexing for better performance
- Path filtering support for targeted searches
- HNSW index WAL recovery with enhanced checkpoints
- Embedding cache optimization with CRC32-based content tracking

### Changed
- **BREAKING**: 'run' command renamed to 'index' with current directory default
- **BREAKING**: Parser system refactored to registry pattern
- Centralized language support in Language enum
- Optimized embedding performance with token-aware batching
- Enhanced PyInstaller compatibility
- Improved cross-platform build support (Windows, Ubuntu Docker)
- Enhanced MCP server JSON-RPC communication with logging suppression

### Fixed
- Parser error handling and registry integration
- OpenAI token limit handling
- PyInstaller module path resolution
- Database WAL corruption issues on server exit
- File watcher cancellation responsiveness
- Signal handler safety by removing unsafe database operations
- Windows PyInstaller and MATLAB dependency issues
- Build workflow reliability across platforms

## [1.2.3] - 2025-06-23

### Changed
- Default database location changed to current directory for better persistence

### Fixed
- OpenAI token limit exceeded error with dynamic batching for large embedding requests
- Empty chunk filtering to reduce noise in search results
- Python parser validation for empty symbol names
- Windows build support with comprehensive GitHub Actions workflow
- macOS Intel build issues with UV package manager installation
- Cross-platform build workflow reliability

### Added
- Windows build support with automated testing
- Enhanced debugging for build processes across platforms

## [1.2.2] - 2024-12-15

### Added
- File watching CLI for real-time code monitoring

### Changed
- Unified JavaScript and TypeScript parsers
- Default database location to current directory

### Fixed
- Empty symbol validation in Python parser

## [1.2.1] - 2024-11-28

### Added
- Ubuntu 20.04 build support
- Token limit management for MCP search

### Fixed
- Duplicate chunks after file edits
- File modification detection race conditions

## [1.2.0] - 2024-11-15

### Added
- C# language support
- JSON, YAML, and plain text file support
- File watching with real-time indexing

### Fixed
- File deletion handling
- Database connection issues

## [1.1.0] - 2025-06-12

### Added
- Multi-language support: TypeScript, JavaScript, C#, Java, and Markdown
- Comprehensive CLI interface
- Binary distribution with faster startup

### Changed
- Improved CLI startup performance (90% faster)
- Binary startup performance (16x faster)

### Fixed
- Version display consistency
- Cross-platform build issues

## [1.0.1] - 2025-06-11

### Added
- Python 3.10+ compatibility
- PyPI publishing
- Standalone executable support
- MCP server integration

### Fixed
- Dependency conflicts
- OpenAI model parameter handling
- Binary compilation issues

## [1.0.0] - 2025-06-10

### Added
- Initial release of ChunkHound
- Python parsing with tree-sitter
- DuckDB backend for storage and search
- OpenAI embeddings for semantic search
- CLI interface for indexing and searching
- MCP server for AI assistant integration
- File watching for real-time indexing
- Regex search capabilities

For more information, visit: https://github.com/chunkhound/chunkhound

[Unreleased]: https://github.com/chunkhound/chunkhound/compare/v5.2.0...HEAD
[5.2.0]: https://github.com/chunkhound/chunkhound/compare/v5.1.0...v5.2.0
[5.1.0]: https://github.com/chunkhound/chunkhound/compare/v5.0.0...v5.1.0
[5.0.0]: https://github.com/chunkhound/chunkhound/compare/v4.0.1...v5.0.0
[4.0.1]: https://github.com/chunkhound/chunkhound/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/chunkhound/chunkhound/compare/v3.3.1...v4.0.0
[3.3.1]: https://github.com/chunkhound/chunkhound/compare/v3.3.0...v3.3.1
[3.3.0]: https://github.com/chunkhound/chunkhound/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/chunkhound/chunkhound/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/chunkhound/chunkhound/compare/v3.0.1...v3.1.0
[3.0.1]: https://github.com/chunkhound/chunkhound/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/chunkhound/chunkhound/compare/v2.8.1...v3.0.0
[2.8.1]: https://github.com/chunkhound/chunkhound/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/chunkhound/chunkhound/compare/v2.7.0...v2.8.0
[2.7.0]: https://github.com/chunkhound/chunkhound/compare/v2.6.3...v2.7.0
[2.6.3]: https://github.com/chunkhound/chunkhound/compare/v2.6.2...v2.6.3
[2.6.2]: https://github.com/chunkhound/chunkhound/compare/v2.6.1...v2.6.2
[2.6.1]: https://github.com/chunkhound/chunkhound/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/chunkhound/chunkhound/compare/v2.5.4...v2.6.0
[2.5.4]: https://github.com/chunkhound/chunkhound/compare/v2.5.3...v2.5.4
[2.5.3]: https://github.com/chunkhound/chunkhound/compare/v2.5.2...v2.5.3
[2.5.2]: https://github.com/chunkhound/chunkhound/compare/v2.5.1...v2.5.2
[2.5.1]: https://github.com/chunkhound/chunkhound/compare/v2.5.0...v2.5.1
[2.5.0]: https://github.com/chunkhound/chunkhound/compare/v2.4.4...v2.5.0
[2.4.4]: https://github.com/chunkhound/chunkhound/compare/v2.4.3...v2.4.4
[2.4.3]: https://github.com/chunkhound/chunkhound/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/chunkhound/chunkhound/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/chunkhound/chunkhound/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/chunkhound/chunkhound/compare/v2.3.1...v2.4.0
[2.3.1]: https://github.com/chunkhound/chunkhound/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/chunkhound/chunkhound/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/chunkhound/chunkhound/compare/v2.1.4...v2.2.0
[2.1.4]: https://github.com/chunkhound/chunkhound/compare/v2.1.3...v2.1.4
[2.1.3]: https://github.com/chunkhound/chunkhound/compare/v2.1.2...v2.1.3
[2.1.2]: https://github.com/chunkhound/chunkhound/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/chunkhound/chunkhound/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/chunkhound/chunkhound/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/chunkhound/chunkhound/compare/v1.2.3...v2.0.0
[1.2.3]: https://github.com/chunkhound/chunkhound/compare/v1.2.2...v1.2.3
[1.2.2]: https://github.com/chunkhound/chunkhound/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/chunkhound/chunkhound/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/chunkhound/chunkhound/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/chunkhound/chunkhound/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/chunkhound/chunkhound/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/chunkhound/chunkhound/releases/tag/v1.0.0

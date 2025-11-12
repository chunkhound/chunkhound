<p align="center">
  <a href="https://chunkhound.github.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="public/wordmark-centered-dark.svg">
      <img src="public/wordmark-centered.svg" alt="ChunkHound" width="400">
    </picture>
  </a>
</p>

<p align="center">
  <strong>Deep Research for Code & Files</strong>
</p>

<p align="center">
  <a href="https://github.com/chunkhound/chunkhound/actions/workflows/smoke-tests.yml"><img src="https://github.com/chunkhound/chunkhound/actions/workflows/smoke-tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/100%25%20AI-Generated-ff69b4.svg" alt="100% AI Generated">
  <a href="https://discord.gg/BAepHEXXnX"><img src="https://img.shields.io/badge/Discord-Join_Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

Transform your codebase into a searchable knowledge base for AI assistants using [semantic search via cAST algorithm](https://arxiv.org/pdf/2506.15655) and regex search. Integrates with AI assistants via the [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/).

## Features

- **[cAST Algorithm](https://arxiv.org/pdf/2506.15655)** - Research-backed semantic code chunking
- **[Multi-Hop Semantic Search](https://chunkhound.github.io/under-the-hood/#multi-hop-semantic-search)** - Discovers interconnected code relationships beyond direct matches
- **Semantic search** - Natural language queries like "find authentication code"
- **Regex search** - Pattern matching without API keys
- **Local-first** - Your code stays on your machine
- **29 languages** with structured parsing
  - **Programming** (via [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)): Python, JavaScript, TypeScript, JSX, TSX, Java, Kotlin, Groovy, C, C++, C#, Go, Rust, Haskell, Swift, Bash, MATLAB, Makefile, Objective-C, PHP, Vue, Zig
  - **Configuration** (via Tree-sitter): JSON, YAML, TOML, HCL, Markdown
  - **Text-based** (custom parsers): Text files, PDF
- **[MCP integration](https://spec.modelcontextprotocol.io/)** - Works with Claude, VS Code, Cursor, Windsurf, Zed, etc

## Documentation

**Visit [chunkhound.github.io](https://chunkhound.github.io) for complete guides:**
- [Tutorial](https://chunkhound.github.io/tutorial/)
- [Configuration Guide](https://chunkhound.github.io/configuration/)
- [Architecture Deep Dive](https://chunkhound.github.io/under-the-hood/)

## Requirements

- Python 3.10+
- [uv package manager](https://docs.astral.sh/uv/)
- API keys (optional - regex search works without any keys)
  - **Embeddings**: [OpenAI](https://platform.openai.com/api-keys) | [VoyageAI](https://dash.voyageai.com/) | [Local with Ollama](https://ollama.ai/)
  - **LLM (for deep research)**: [Anthropic](https://console.anthropic.com/) | [OpenAI](https://platform.openai.com/api-keys) | [Local with Ollama](https://ollama.ai/)

## Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ChunkHound
uv tool install chunkhound
```

## Quick Start

1. Create `.chunkhound.json` in project root file
```json
{
  "embedding": {
    "provider": "openai",
    "api_key": "your-api-key-here"
  }
}
```
2. Index your codebase
```bash
chunkhound index
```

**For configuration, IDE setup, and advanced usage, see the [documentation](https://chunkhound.github.io).**

## YAML Parsing Benchmarks

Use the reproducible benchmark harness to compare PyYAML, tree-sitter/cAST, and RapidYAML bindings on representative YAML workloads.

```bash
# Default synthetic cases with all available backends
uv run python scripts/bench_yaml.py

# Use your own fixtures or disable specific backends
uv run python scripts/bench_yaml.py \
  --cases-dir ./benchmarks/yaml \
  --backends pyyaml_safe_load tree_sitter_universal \
  --iterations 10
```

## Real-Time Indexing

**Automatic File Watching**: MCP servers monitor your codebase and update the index automatically as you edit files. No manual re-indexing required.

**Smart Content Diffs**: Only changed code chunks get re-processed. Unchanged chunks keep their existing embeddings, making updates efficient even for large codebases.

**Seamless Branch Switching**: When you switch git branches, ChunkHound automatically detects and re-indexes only the files that actually changed between branches.

**Live Memory Systems**: Index markdown notes or documentation that updates in real-time while you work, creating a dynamic knowledge base.

## Why ChunkHound?

**Research Foundation**: Built on the [cAST (Chunking via Abstract Syntax Trees)](https://arxiv.org/pdf/2506.15655) algorithm from Carnegie Mellon University, providing:
- **4.3 point gain** in Recall@5 on RepoEval retrieval
- **2.67 point gain** in Pass@1 on SWE-bench generation
- **Structure-aware chunking** that preserves code meaning

**Local-First Architecture**:
- Your code never leaves your machine
- Works offline with [Ollama](https://ollama.ai/) local models
- No per-token charges for large codebases

**Universal Language Support**:
- Structured parsing for 29 languages (Tree-sitter + custom parsers)
- Same semantic concepts across all programming languages

**Intelligent Code Discovery**:
- Multi-hop search follows semantic relationships to find related implementations
- Automatically discovers complete feature patterns: find "authentication" to get password hashing, token validation, session management
- Convergence detection prevents semantic drift while maximizing discovery

## License

MIT

## Startup profile (discovery diagnostics)

Use `--profile-startup` to emit a JSON block with discovery and startup timing diagnostics to stderr. This works for both simulate and full index runs.

Examples:

```bash
# Simulate (discovery only) — file list on stdout, JSON profile on stderr
CHUNKHOUND_NO_RICH=1 \
chunkhound index --simulate . --sort path --profile-startup 2>profile.json

# Full run (no embeddings) — JSON profile on stderr
CHUNKHOUND_NO_RICH=1 \
chunkhound index . --no-embeddings --profile-startup 2>profile.json
```

Fields in `startup_profile` (JSON):

- `discovery_ms` — discovery time in milliseconds
- `cleanup_ms` — orphan cleanup time in milliseconds
- `change_scan_ms` — change-scan time in milliseconds
- `resolved_backend` — discovery backend actually used: `python | git | git_only`
- `resolved_reasons` — reasons for the decision (e.g., `no_repos`, `all_repos`, `mixed`, `explicit`)
- `git_rows_tracked` — number of paths from `git ls-files` (tracked)
- `git_rows_others` — number of paths from `git ls-files --others --exclude-standard`
- `git_rows_total` — sum of the two above
- `git_pathspecs` — number of pathspecs (`:(glob) ...`) pushed down to Git for pre-filtering
  - CAP: set `CHUNKHOUND_INDEXING__GIT_PATHSPEC_CAP` (default: 128). If the number of synthesized specs would exceed the cap, ChunkHound falls back to a subtree-only pathspec to guarantee coverage. The profile reflects the actual `git_pathspecs` used; an optional `git_pathspecs_capped: true` may appear.

Notes:

- `git_*` counters appear only when the backend is `git` or `git_only`.
- In `auto` mode, the backend is chosen heuristically (`git_only` for all‑repo trees, `git` for mixed trees, `python` when no repos are found).
- For scripting, set `CHUNKHOUND_NO_RICH=1` and read stderr; each JSON block appears on its own line near the end of the run.

Example snippet:

```json
{
  "startup_profile": {
    "discovery_ms": 154.2,
    "cleanup_ms": 12.7,
    "change_scan_ms": 3.1,
    "resolved_backend": "git_only",
    "resolved_reasons": ["all_repos"],
    "git_rows_tracked": 420,
    "git_rows_others": 17,
    "git_rows_total": 437,
    "git_pathspecs": 4
  }
}
```

## Exclusions (gitignore, config, defaults)

ChunkHound combines repository–aware ignores with safe defaults. The behavior depends on how you set `indexing.exclude` in `.chunkhound.json`:

- Not set (default) → gitignore only
  - The `.gitignore` files inside repositories are honored (repo‑aware engine). Default ChunkHound excludes (e.g., `.git/`, `node_modules/`, `.chunkhound/`, caches) always apply to prevent self‑indexing and noise.
- String sentinel `.gitignore` → gitignore only
  - Same as the default: only `.gitignore` rules are used as the exclusion source (plus ChunkHound’s default excludes).
- Explicit list (array) → combined (gitignore + config) [default]
  - Your glob patterns in `indexing.exclude` are layered on top of `.gitignore` rather than replacing it. ChunkHound’s default excludes are also applied. This avoids surprising loss of `.gitignore` behavior when you accept prompts to add slow files to excludes.
  - To restore legacy behavior, set `indexing.exclude_mode: "config_only"`. To force only gitignore even when a list exists, set `indexing.exclude_mode: "gitignore_only"` (rare).

Workspace overlay for non‑repo paths (default: on)
- When the directory you index contains non‑repo subtrees, ChunkHound can apply the root workspace `.gitignore` only to those non‑repo paths. This is controlled by `indexing.workspace_gitignore_nonrepo` (default: `true`).
- Repository subtrees always use their own `.gitignore` and Git’s native semantics.

Examples

```jsonc
// Default: gitignore only (+ safe defaults)
{
  "indexing": {
    // exclude omitted
    "workspace_gitignore_nonrepo": true
  }
}

// Gitignore only (explicit sentinel)
{
  "indexing": {
    "exclude": ".gitignore",
    "workspace_gitignore_nonrepo": true
  }
}

// Explicit list layered ON TOP of gitignore (default)
{
  "indexing": {
    "exclude": ["**/dist/**", "**/*.min.js"],
    "workspace_gitignore_nonrepo": false
  }
}

// Legacy behavior: config only (gitignore ignored)
{
  "indexing": {
    "exclude": ["**/dist/**", "**/*.min.js"],
    "exclude_mode": "config_only"
  }
}

// Force gitignore-only even with a list (rare)
{
  "indexing": {
    "exclude": ["**/dist/**"],
    "exclude_mode": "gitignore_only"
  }
}

### Root semantics for config patterns

- Config `include` and `exclude` patterns are always evaluated relative to the ChunkHound root (the directory you pass to `chunkhound index`).
- Git’s own `.gitignore` patterns remain repo‑aware (anchored to their respective repository roots), but your config overlay applies uniformly from the CH root across all subtrees (including Git repos).
- Examples:
  - CH root is `/workspaces`; Git repo lives under `/workspaces/monorepo`. To exclude a file inside that repo using config, prefer a CH‑root‑relative path (e.g., `"**/monorepo/path/inside/repo/file.txt"`).
  - When using anchored includes like `"src/**/*.ts"`, ensure the anchor is correct from the CH root perspective (e.g., `"monorepo/src/**/*.ts"` when the repo is nested).
```

CLI toggle for the workspace overlay
- `--nonrepo-gitignore` enables the root `.gitignore` overlay for non‑repo paths for the current run.
- To disable overlay persistently, set `"workspace_gitignore_nonrepo": false` in `.chunkhound.json`.

## Simulate and diagnostics

Simulate a discovery run without writing to the database. Useful for verifying include/exclude rules, sorting, and sizes.

```bash
# List discovered files (sorted by path)
chunkhound index --simulate . --sort path

# Show sizes and sort by size (descending)
chunkhound index --simulate . --show-sizes --sort size_desc

# Emit JSON instead of plain text
chunkhound index --simulate . --json > files.json

# Add discovery timing/profile to stderr (JSON)
CHUNKHOUND_NO_RICH=1 chunkhound index --simulate . --profile-startup 2>profile.json

# Print debug info about ignores (to stderr): CH root, sources, first N defaults
chunkhound index --simulate . --debug-ignores
```

Diagnostics (ignore decisions):

```bash
# Compare ChunkHound’s ignore decision vs Git for the current tree
chunkhound index --check-ignores --vs git --json > ignore_diff.json
```

Notes:
- When piping simulate output to tools like `head`, BrokenPipe is handled gracefully; prefer `CHUNKHOUND_NO_RICH=1` for easy JSON parsing.

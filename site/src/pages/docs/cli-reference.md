---
layout: ../../layouts/DocsLayout.astro
title: "CLI Reference"
description: "Complete reference for all ChunkHound CLI commands and flags."
order: 3
section: "manual"
---

# CLI Reference

All ChunkHound commands and their options.

## `chunkhound index`

Index a directory for code search. Parses source files, generates embeddings, and stores chunks in the database.

```bash
chunkhound index [path] [options]
```

| Argument | Description |
|---|---|
| `path` | Directory to index (default: `.`) |

**Options:**

| Flag | Description |
|---|---|
| `--config PATH` | Path to configuration file |
| `--simulate` | Dry-run: show which files would be indexed without making changes |
| `--json` | Output as JSON (with `--simulate` or `--check-ignores`) |
| `--all-discovered` | Show files before change-detection pruning |
| `--show-sizes` | Include file sizes in output |
| `--sort {path,size,size_desc}` | Sort output |
| `--check-ignores` | Compare ignore decisions vs git |
| `--vs {git}` | Sentinel for `--check-ignores` |
| `--debug-ignores` | Print ignore context to stderr |
| `--profile-startup` | Emit phase timings as JSON to stderr |
| `--discovery-backend {auto,python,git,git_only}` | Override file discovery backend |
| `--perf-diagnostics` | Collect per-batch timing metrics |
| `--perf-output PATH` | Output path for performance JSON |
| `--verbose` | Verbose output |
| `--debug` | Debug output |

**Config-override flags** (override values from `.chunkhound.json`):

Database, embedding, and indexing options can be set via CLI flags. These follow the pattern `--database-provider`, `--embedding-model`, `--indexing-exclude`, etc.

**Examples:**

```bash
# Index current directory
chunkhound index

# Index a specific project
chunkhound index /path/to/project

# Dry-run to see what would be indexed
chunkhound index --simulate

# Dry-run with JSON output
chunkhound index --simulate --json

# Compare ignore decisions against git
chunkhound index --check-ignores --vs git
```

## `chunkhound search`

Search an indexed codebase using semantic or regex search.

```bash
chunkhound search <query> [path] [options]
```

| Argument | Description |
|---|---|
| `query` | Search query (required) |
| `path` | Project directory (default: `.`) |

**Options:**

| Flag | Description |
|---|---|
| `--semantic` | Semantic search (default) |
| `--single-hop` | Force single-hop semantic search |
| `--multi-hop` | Force multi-hop semantic search |
| `--regex` | Regex pattern search (no embeddings required) |
| `--page-size N` | Results per page (default: 10) |
| `--offset N` | Pagination offset |
| `--path-filter PATH` | Filter results by file path |
| `--config PATH` | Path to configuration file |
| `--verbose` | Verbose output |
| `--debug` | Debug output |

**Examples:**

```bash
# Semantic search
chunkhound search "authentication flow"

# Regex search (no API key needed)
chunkhound search --regex "def.*auth"

# Filter by path
chunkhound search "database connection" --path-filter src/db/

# Paginate results
chunkhound search "error handling" --page-size 5 --offset 10
```

## `chunkhound research`

Deep code research. Generates a synthesized answer with citations by searching the codebase, reading relevant files, and using an LLM to analyze the results.

```bash
chunkhound research <query> [path] [options]
```

| Argument | Description |
|---|---|
| `query` | Research question (required) |
| `path` | Project directory (default: `.`) |

**Options:**

| Flag | Description |
|---|---|
| `--path-filter PATH` | Filter results by file path |
| `--config PATH` | Path to configuration file |
| `--verbose` | Verbose output |
| `--debug` | Debug output |

**Examples:**

```bash
# Research a topic
chunkhound research "How does the auth system work?"

# Scoped to a subdirectory
chunkhound research "How are database migrations handled?" --path-filter src/db/
```

## `chunkhound mcp`

Run ChunkHound as an MCP (Model Context Protocol) server for AI assistant integration.

```bash
chunkhound mcp [path] [options]
```

| Argument | Description |
|---|---|
| `path` | Project directory (default: `.`) |

**Options:**

| Flag | Description |
|---|---|
| `--no-daemon` | Run without daemon (single client mode) |
| `--config PATH` | Path to configuration file |
| `--verbose` | Verbose output |
| `--debug` | Debug output |

**Examples:**

```bash
# Start MCP server for current directory
chunkhound mcp

# Start MCP server for a specific project
chunkhound mcp /path/to/project
```

## `chunkhound map`

Generate agent-facing documentation from your codebase using Code Mapper.

```bash
chunkhound map [path] [options]
```

| Argument | Description |
|---|---|
| `path` | Directory to document |

**Options:**

| Flag | Description |
|---|---|
| `--out DIR` | Output directory (required) |
| `--plan` | Only run the planning pass |
| `--audience {technical,balanced,end-user}` | Target audience |
| `--context PATH` | Authoritative context file |
| `--combined / --no-combined` | Write combined markdown output |
| `-j, --jobs N` | Concurrent research jobs |
| `--comprehensiveness {minimal,low,medium,high,ultra}` | Mapping depth |
| `--minimal` | Alias for `--comprehensiveness minimal` |
| `--low` | Alias for `--comprehensiveness low` |
| `--medium` | Alias for `--comprehensiveness medium` |
| `--high` | Alias for `--comprehensiveness high` |
| `--ultra` | Alias for `--comprehensiveness ultra` |
| `--config PATH` | Path to configuration file |
| `--verbose` | Verbose output |
| `--debug` | Debug output |

**Examples:**

```bash
# Generate documentation
chunkhound map /path/to/project --out docs/

# Planning pass only
chunkhound map /path/to/project --out docs/ --plan

# High-detail documentation
chunkhound map /path/to/project --out docs/ --high -j 4
```

## `chunkhound autodoc`

Generate an Astro documentation site from Code Mapper output.

```bash
chunkhound autodoc [map-in] [options]
```

| Argument | Description |
|---|---|
| `map-in` | Directory with Code Mapper outputs |

**Options:**

| Flag | Description |
|---|---|
| `--out-dir DIR` | Output directory (required) |
| `--force` | Allow deletion of existing topics |
| `--assets-only` | Update only Astro assets |
| `--site-title TEXT` | Override site title |
| `--site-tagline TEXT` | Override site tagline |
| `--cleanup-mode {llm}` | Cleanup pass mode |
| `--cleanup-batch-size N` | Sections per LLM batch |
| `--cleanup-max-tokens N` | Max tokens per cleanup |
| `--audience {technical,balanced,end-user}` | Target audience |
| `--index-pattern GLOB` | Override index globs |
| `--map-out-dir DIR` | Output directory for auto-generated maps |
| `--map-comprehensiveness {minimal,low,medium,high,ultra}` | Mapping depth |
| `--map-audience` | Audience for auto-generated maps |
| `--map-context PATH` | Context file for mapper |
| `--config PATH` | Path to configuration file |
| `--verbose` | Verbose output |
| `--debug` | Debug output |

**Examples:**

```bash
# Generate docs site from existing map output
chunkhound autodoc map-output/ --out-dir docs-site/

# Update only the generated docs site assets
chunkhound autodoc --assets-only --out-dir docs-site/

# Full pipeline: map and generate docs
chunkhound autodoc --out-dir docs-site/ --map-out-dir map-output/ --map-comprehensiveness high
```

## `chunkhound calibrate`

Calibrate embedding and reranking batch sizes for optimal performance.

```bash
chunkhound calibrate [options]
```

**Options:**

| Flag | Description |
|---|---|
| `--embedding-batch-sizes N [N ...]` | Embedding batch sizes to test |
| `--reranking-batch-sizes N [N ...]` | Reranking batch sizes to test |
| `--test-document-count N` | Number of test documents (default: 500) |
| `--num-test-runs N` | Runs per size (default: 5) |
| `--output-format {text,json}` | Output format |
| `--output-file PATH` | Write results to file |
| `--config PATH` | Path to configuration file |
| `--verbose` | Verbose output |
| `--debug` | Debug output |

**Examples:**

```bash
# Run calibration with defaults
chunkhound calibrate

# Test specific batch sizes
chunkhound calibrate --embedding-batch-sizes 64 128 256 512

# Output as JSON
chunkhound calibrate --output-format json --output-file calibration.json
```

## Common Flags

These flags are available on all commands:

| Flag | Description |
|---|---|
| `--config PATH` | Path to `.chunkhound.json` configuration file |
| `--verbose` | Enable verbose output |
| `--debug` | Enable debug output (implies verbose) |
| `--version` | Show version and exit |
| `--help` | Show help and exit |

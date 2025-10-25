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
- **24 languages** with structured parsing
  - **Programming** (via [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)): Python, JavaScript, TypeScript, JSX, TSX, Java, Kotlin, Groovy, C, C++, C#, Go, Rust, Bash, MATLAB, Makefile, PHP, Vue
  - **Configuration** (via Tree-sitter): JSON, YAML, TOML, Markdown
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
- API key for semantic search (optional - regex search works without any keys)
  - [OpenAI](https://platform.openai.com/api-keys) | [VoyageAI](https://dash.voyageai.com/) | [Local with Ollama](https://ollama.ai/)

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

## Codex CLI Synthesis (Advanced)

ChunkHound can use the Codex CLI as the synthesis LLM for deep research. This lets you reuse your local Codex agent configuration while ChunkHound handles retrieval and context assembly.

- Prerequisites: Codex CLI installed and authenticated. Authentication is discovered from your `CODEX_HOME` (defaults to `~/.codex`).
- ChunkHound never prints Codex output directly to MCP stdout; results are returned via tool responses.

Run MCP stdio with Codex for synthesis:

```bash
uv run chunkhound mcp . --stdio \
  --llm-synthesis-provider codex-cli \
  --llm-synthesis-model codex
```

Use Codex for the research CLI:

```bash
chunkhound research "How does indexing deduplicate chunks?" \
  --llm-synthesis-provider codex-cli \
  --llm-synthesis-model codex
```

Implementation notes:
- The provider shells out to `codex exec` and captures output. It never writes to stdout from within the MCP server process.
- Each Codex run uses a sandboxed `CODEX_HOME`. ChunkHound copies only auth/session files, writes a fresh no-history/no-MCP `config.toml`, and points Codex at that sandbox so nothing from your primary agent config leaks in. The symbolic model name `codex` resolves to the Codex CLI default (`gpt-5-codex`) unless overridden.
- Pass an explicit Codex CLI model via `--llm-synthesis-model` (for example `gpt-5-codex-pro`) to switch variants; all overlay modes honor the requested model.
- Control the Codex reasoning effort with `--llm-codex-reasoning-effort` (`minimal` | `low` | `medium` | `high`) or `CHUNKHOUND_LLM_CODEX_REASONING_EFFORT=high` to change the Responses API “thinking” depth used during synthesis. Use `--llm-codex-reasoning-effort-utility` / `--llm-codex-reasoning-effort-synthesis` (or the `CHUNKHOUND_LLM_CODEX_REASONING_EFFORT_UTILITY` / `CHUNKHOUND_LLM_CODEX_REASONING_EFFORT_SYNTHESIS` env vars) for stage-specific overrides.
- Prompts are sent via stdin (privacy-first) rather than argv by default. Set `CHUNKHOUND_CODEX_STDIN_FIRST=0` to prefer argv for short prompts.
- Environment variables:
  - `CHUNKHOUND_CODEX_CONFIG_OVERRIDE` — `env` (default) uses `CODEX_CONFIG`; `flag` adds `--config <path>`
  - `CHUNKHOUND_CODEX_CONFIG_ENV` — env key for config path (default: `CODEX_CONFIG`)
  - `CHUNKHOUND_CODEX_CONFIG_FLAG` — config flag name when using `flag` override (default: `--config`)
  - `CHUNKHOUND_CODEX_DEFAULT_MODEL` — fallback Codex CLI model when config/model is `codex` (default: `gpt-5-codex`)
  - `CHUNKHOUND_CODEX_REASONING_EFFORT` — reasoning effort written to temp config (`minimal`, `low`, `medium`, `high`; default: `low`)
  - `CHUNKHOUND_CODEX_AUTH_ENV` — comma-list of auth envs to forward (default: `OPENAI_API_KEY,CODEX_API_KEY,ANTHROPIC_API_KEY,BEARER_TOKEN`)
  - `CHUNKHOUND_CODEX_PASSTHROUGH_ENV` — comma-list of additional envs to forward
  - `CHUNKHOUND_CODEX_BIN` — path to the `codex` binary (default: `codex` on PATH)
  - `CHUNKHOUND_CODEX_ARG_LIMIT` — max characters to pass via argv before falling back to stdin (default: 200000)
  - `CHUNKHOUND_CODEX_COPY_ALL` — copy entire `CODEX_HOME` in overlay mode (default: minimal selective copy)
  - `CHUNKHOUND_CODEX_MAX_COPY_BYTES` — per-file copy cap in overlay mode (default: 1,000,000 bytes)
  - `CHUNKHOUND_CODEX_LOG_MAX_ERR` — max characters of stderr included in error messages (default: 800; secrets are redacted)

If Codex is unavailable or unauthenticated, ChunkHound’s other providers (OpenAI, Ollama, Claude Code CLI, Bedrock) remain available.

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
- Structured parsing for 24 languages (Tree-sitter + custom parsers)
- Same semantic concepts across all programming languages

**Intelligent Code Discovery**:
- Multi-hop search follows semantic relationships to find related implementations
- Automatically discovers complete feature patterns: find "authentication" to get password hashing, token validation, session management
- Convergence detection prevents semantic drift while maximizing discovery

## License

MIT

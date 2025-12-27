resume te current epic with
codex resume 019b5e94-c01c-7d22-9a51-225b5b0a9e33

<p align="center">
  <a href="https://chunkhound.github.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="public/wordmark-centered-dark.svg">
      <img src="public/wordmark-centered.svg" alt="ChunkHound" width="400">
    </picture>
  </a>
</p>

<p align="center">
  <strong>Don't search your code. Research it.</strong>
</p>

<p align="center">
  <a href="https://github.com/chunkhound/chunkhound/actions/workflows/smoke-tests.yml"><img src="https://github.com/chunkhound/chunkhound/actions/workflows/smoke-tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/100%25%20AI-Generated-ff69b4.svg" alt="100% AI Generated">
  <a href="https://discord.gg/BAepHEXXnX"><img src="https://img.shields.io/badge/Discord-Join_Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

Your AI assistant searches code but doesn't understand it. ChunkHound researches your codebase—extracting architecture, patterns, and institutional knowledge at any scale. Integrates via [MCP](https://spec.modelcontextprotocol.io/).

## Features

- **[cAST Algorithm](https://arxiv.org/pdf/2506.15655)** - Research-backed semantic code chunking
- **[Multi-Hop Semantic Search](https://chunkhound.github.io/under-the-hood/#multi-hop-semantic-search)** - Discovers interconnected code relationships beyond direct matches
- **Semantic search** - Natural language queries like "find authentication code"
- **Regex search** - Pattern matching without API keys
- **Local-first** - Your code stays on your machine
- **29 languages** with structured parsing
  - **Programming** (via [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)): Python, JavaScript, TypeScript, JSX, TSX, Java, Kotlin, Groovy, C, C++, C#, Go, Rust, Haskell, Swift, Bash, MATLAB, Makefile, Objective-C, PHP, Vue, Zig
  - **Configuration**: JSON, YAML, TOML, HCL, Markdown
  - **Text-based** (custom parsers): Text files, PDF
- **[MCP integration](https://spec.modelcontextprotocol.io/)** - Works with Claude, VS Code, Cursor, Windsurf, Zed, etc
- **Real-time indexing** - Automatic file watching, smart diffs, seamless branch switching

## Documentation

**Visit [chunkhound.github.io](https://chunkhound.github.io) for complete guides:**
- [Tutorial](https://chunkhound.github.io/tutorial/)
- [Configuration Guide](https://chunkhound.github.io/configuration/)
- [Architecture Deep Dive](https://chunkhound.github.io/under-the-hood/)

## Requirements

- Python 3.10+
- [uv package manager](https://docs.astral.sh/uv/)
- API keys (optional - regex search works without any keys)
  - **Embeddings**: [VoyageAI](https://dash.voyageai.com/) (recommended) | [OpenAI](https://platform.openai.com/api-keys) | [Local with Ollama](https://ollama.ai/)
  - **LLM (for Code Research)**: Claude Code CLI or Codex CLI (no API key needed) | [Anthropic](https://console.anthropic.com/) | [OpenAI](https://platform.openai.com/api-keys)

## Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ChunkHound
uv tool install chunkhound
```

## Quick Start

1. Create `.chunkhound.json` in project root
```json
{
  "embedding": {
    "provider": "voyageai",
    "api_key": "your-voyageai-key"
  },
  "llm": {
    "provider": "claude-code-cli"
  }
}
```
> **Note:** Use `"codex-cli"` instead if you prefer Codex. Both work equally well and require no API key.
2. Index your codebase
```bash
chunkhound index
```

**For configuration, IDE setup, and advanced usage, see the [documentation](https://chunkhound.github.io).**

## Why ChunkHound?

| Approach | Capability | Scale | Maintenance |
|----------|------------|-------|-------------|
| Keyword Search | Exact matching | Fast | None |
| Traditional RAG | Semantic search | Scales | Re-index files |
| Knowledge Graphs | Relationship queries | Expensive | Continuous sync |
| **ChunkHound** | Semantic + Regex + Code Research | Automatic | Incremental + realtime |

**Ideal for:**
- Large monorepos with cross-team dependencies
- Security-sensitive codebases (local-only, no cloud)
- Multi-language projects needing consistent search
- Offline/air-gapped development environments

**Stop recreating code. Start with deep understanding.**

## License

MIT

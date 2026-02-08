<p align="center">
  <a href="https://chunkhound.github.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="public/wordmark-centered-dark.svg">
      <img src="public/wordmark-centered.svg" alt="ChunkHound" width="400">
    </picture>
  </a>
</p>

<p align="center">
  <strong>Local-first codebase intelligence</strong>
</p>

<p align="center">
  Open-source codebase intelligence for AI coding agents
</p>

<p align="center">
  <a href="https://github.com/chunkhound/chunkhound/actions/workflows/smoke-tests.yml"><img src="https://github.com/chunkhound/chunkhound/actions/workflows/smoke-tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/100%25%20AI-Generated-ff69b4.svg" alt="100% AI Generated">
  <a href="https://discord.gg/BAepHEXXnX"><img src="https://img.shields.io/badge/Discord-Join_Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

---

ChunkHound gives AI agents the same structural understanding of your codebase that your best engineers have—complete functions, architectural context, and semantic relationships.

## The Problem

AI coding assistants are powerful, but they struggle with real codebases:

- **Lost context** — Agents see file fragments, not architectural relationships
- **Repeated questions** — "Where is authentication handled?" asked in every session
- **Broken changes** — Edits that work in isolation but break dependencies elsewhere

Your AI can write code. It just doesn't understand *your* code.

## The Solution

ChunkHound transforms your codebase into a searchable knowledge base that AI agents can query through MCP. Instead of feeding your agent random file snippets, give it the ability to research your architecture, discover patterns, and understand how components connect.

## Features

### Code Research

The flagship capability. Your AI agent asks complex questions about your codebase—"How does the authentication system work?" or "What patterns does this repo use for error handling?"—and gets synthesized answers with citations, not just file matches.

### Semantic + Regex Search

Natural language queries like "find rate limiting logic" combined with precise pattern matching like `def test_.*async`. Use what fits the task.

### CAST Algorithm

Research-backed semantic code chunking that preserves function boundaries and context. Based on [published research](https://arxiv.org/pdf/2506.15655). Your AI sees complete, meaningful code units—not arbitrary line splits.

### 32 Languages

Structured parsing via [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) for Python, TypeScript, Go, Rust, Java, C++, and 23 more. Plus configuration formats (JSON, YAML, TOML, HCL) and documents (Markdown, PDF).

### Real-time Indexing

Automatic file watching, smart diffs, seamless branch switching. Your index stays current without manual intervention.

### MCP Integration

Native [Model Context Protocol](https://spec.modelcontextprotocol.io/) support. Works with Claude Code, VS Code Copilot, Cursor, Windsurf, Zed, and any MCP-compatible client.

### Local-first Privacy

Your code never leaves your machine. Bring your own embedding provider—VoyageAI, OpenAI, or run fully local with Ollama.

## Quick Start

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ChunkHound
uv tool install chunkhound

# Index your codebase
chunkhound index
```

Create `.chunkhound.json` in your project root:

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

> **Note:** Use `"codex-cli"` for OpenAI's Codex CLI. Both work without separate API keys.

For configuration, IDE setup, and advanced usage, see the [documentation](https://chunkhound.github.io).

## Why ChunkHound?

Traditional code search finds files. ChunkHound understands architecture.

When your AI agent needs to modify a feature, it doesn't just need to find the file—it needs to understand the module boundaries, the data flow, the error handling patterns, and the test coverage expectations. ChunkHound provides that structural context.

**Works best with:**
- Large monorepos with cross-team dependencies
- Security-sensitive codebases requiring local-only operation
- Multi-language projects needing consistent search across stacks
- Teams wanting AI agents that learn their conventions

## Built for Production

ChunkHound handles real-world scale:

- **Kubernetes codebase**: 4.8M lines indexed in 56 minutes on a MacBook Pro M4
- **Incremental updates**: Re-indexing after edits takes seconds, not minutes
- **Branch switching**: Automatic diff-based updates when you change branches

## Requirements

- Python 3.10+
- [uv package manager](https://docs.astral.sh/uv/)
- API keys (optional—regex search works without any keys)
  - **Embeddings**: [VoyageAI](https://dash.voyageai.com/) (recommended) | [OpenAI](https://platform.openai.com/api-keys) | [Local with Ollama](https://ollama.ai/)
  - **LLM (for Code Research)**: Claude Code CLI or Codex CLI (no API key needed) | [Anthropic](https://console.anthropic.com/) | [OpenAI](https://platform.openai.com/api-keys)

## Documentation

- [Tutorial](https://chunkhound.github.io/tutorial/) — Get started in 5 minutes
- [Configuration Guide](https://chunkhound.github.io/configuration/) — Providers, ignore patterns, advanced options
- [Architecture Deep Dive](https://chunkhound.github.io/under-the-hood/) — How CAST and multi-hop search work

## Community

Questions, ideas, or want to contribute?

- [Join our Discord](https://discord.gg/BAepHEXXnX) — Chat with maintainers and users
- [GitHub Discussions](https://github.com/chunkhound/chunkhound/discussions) — Feature requests and Q&A
- [Contributing Guide](https://github.com/chunkhound/chunkhound/blob/main/CONTRIBUTING.md) — Pull requests welcome

## License

MIT

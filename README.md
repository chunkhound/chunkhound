<p align="center">
  <a href="https://chunkhound.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="site/public/wordmark-centered-dark.svg">
      <img src="site/public/wordmark-centered.svg" alt="ChunkHound" width="400">
    </picture>
  </a>
</p>

<p align="center">
  <strong>Your entire codebase, deeply understood</strong>
</p>

<p align="center">
  <a href="https://github.com/chunkhound/chunkhound/actions/workflows/smoke-tests.yml"><img src="https://github.com/chunkhound/chunkhound/actions/workflows/smoke-tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/100%25%20AI-Generated-ff69b4.svg" alt="100% AI Generated">
  <a href="https://discord.gg/BAepHEXXnX"><img src="https://img.shields.io/badge/Discord-Join_Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

ChunkHound gives AI assistants deep codebase understanding — architecture, patterns, and cross-file relationships — via [MCP](https://spec.modelcontextprotocol.io/). Works with Claude, VS Code, Cursor, Windsurf, and Zed.

## Install

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ChunkHound
uv tool install chunkhound
```

## Quick Start

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

Then index your codebase:

```bash
chunkhound index
```

## Features

- **[cAST Algorithm](https://arxiv.org/pdf/2506.15655)** — Research-backed semantic code chunking
- **[Multi-Hop Semantic Search](https://chunkhound.ai/#capabilities)** — Discovers interconnected code relationships beyond direct matches
- **Semantic + Regex search** — Natural language queries or exact pattern matching
- **Local-first** — Your code stays on your machine
- **32 languages** — Python, TypeScript, Rust, Go, Java, C/C++, and [more](https://chunkhound.ai/docs/getting-started/)
- **Real-time indexing** — File watching, smart diffs, seamless branch switching

## Why ChunkHound?

| Approach | Capability | Scale | Maintenance |
|----------|------------|-------|-------------|
| Keyword Search | Exact matching | Fast | None |
| Traditional RAG | Semantic search | Scales | Re-index files |
| Knowledge Graphs | Relationship queries | Expensive | Continuous sync |
| **ChunkHound** | Semantic + Regex + Code Research | Automatic | Incremental + realtime |

## Documentation

**[chunkhound.ai](https://chunkhound.ai)** — Quickstart, configuration, onboarding, and docs.

## License

MIT

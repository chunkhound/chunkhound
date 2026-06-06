<p align="center">
    <a href="https://chunkhound.ai">
    <picture>
        <source media="(prefers-color-scheme: dark)"
srcset="site/public/logo-light.svg">
        <img src="site/public/logo.svg"
alt="ChunkHound" width="140">
    </picture>
    </a>
</p>

<p align="center">
    <a href="https://chunkhound.ai">
    <picture>
        <source media="(prefers-color-scheme: dark)"
srcset="site/public/wordmark-text-dark.svg">
        <img src="site/public/wordmark-text.svg"
alt="ChunkHound" width="300">
    </picture>
    </a>
</p>

<p align="center">
  <strong>Your entire codebase, deeply understood.</strong>
</p>

<p align="center">
  Code research, semantic search, and auto-generated docs for AI agents.
</p>

<p align="center">
  Local-first · Dozens of languages & file types · MIT licensed · Free forever
</p>

<p align="center">
  <a href="https://github.com/chunkhound/chunkhound/actions/workflows/ci.yml"><img src="https://github.com/chunkhound/chunkhound/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/chunkhound/"><img src="https://img.shields.io/pypi/v/chunkhound.svg" alt="PyPI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/100%25%20AI-Generated-ff69b4.svg" alt="100% AI Generated">
  <a href="https://discord.gg/BAepHEXXnX"><img src="https://img.shields.io/badge/Discord-Join_Community-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://chunkhound.ai/docs/getting-started/">Getting Started</a>
  ·
  <a href="https://chunkhound.ai/docs/configuration/">Configuration</a>
  ·
  <a href="https://chunkhound.ai/docs/cli-reference/">CLI Reference</a>
</p>

---

## Requirements

- **Python 3.10+**
- **uv** — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **API keys** (optional — regex search works without any):
  - Embeddings: [VoyageAI](https://dash.voyageai.com/) (recommended) | [OpenAI](https://platform.openai.com/api-keys) | [Ollama](https://ollama.ai/) (local)
  - LLM: Claude Code CLI or Codex CLI (no key needed) | [Anthropic](https://console.anthropic.com/) | [OpenAI](https://platform.openai.com/api-keys) | [Grok](https://console.x.ai)

---

## AI writes code blind

Agents can search code, but that is not the same as understanding how a system works.

They miss that auth flows through three files, that the utility they need already exists, or that the data model changed in a way that makes the obvious implementation wrong.

ChunkHound gives agents grounded, cross-file understanding of a codebase: where concepts live, how files relate, and which architectural paths matter before code gets written.

## One semantic index, three lenses

ChunkHound parses your code and builds one semantic index. That index powers three different workflows:

- **Search** — drill into exact code with semantic and regex search
- **Research** — explain how the system works with cited reports that trace behavior across files
- **Autodoc** — turn that understanding into generated documentation from the code itself

Not three separate tools. Three lenses into one index.

## Why this helps

When an agent lacks context, it tends to loop:

1. search broadly
2. read partial results
3. guess
4. generate
5. backtrack
6. search again

That costs time, burns tokens, and pollutes context with failed attempts.

ChunkHound shifts that work earlier. Build the index once, research before editing, then use grounded search and cited answers on top of it. In practice that usually means:

- fewer dead-end searches
- fewer retries
- less context pollution
- better first-pass answers

## What it can do

- cited code research that explains how a system works across files
- multi-hop semantic search across architectural relationships
- hybrid semantic + regex workflows for discovery plus exact tracing
- gap-filling and query expansion during research
- auto-generated docs from the indexed codebase
- local-first indexing and search
- Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, and more via Tree-sitter

## Install

```bash
uv tool install chunkhound
```

## Try it

```bash
chunkhound index .
chunkhound research "How does authentication work?"
```

Index once, ask a real architecture question, and get a grounded answer with citations. Semantic search requires an embedding provider; research requires an LLM provider and an embedding provider with reranking support. Choose local providers for zero-code-egress setups.

For a full configurable setup, create `.chunkhound.json` in your project root:

```json
{
  "embedding": { "provider": "voyageai", "api_key": "your-key" },
  "llm": { "provider": "claude-code-cli" }
}
```

For editor integration, all provider options, and advanced configuration:

**→ [chunkhound.ai/docs/getting-started](https://chunkhound.ai/docs/getting-started/)**

## Good fit

ChunkHound is especially useful for:

- large repos and monorepos
- multi-language codebases
- legacy systems
- local-only or security-sensitive environments
- agent workflows that need architecture understanding, not just text matches

## Community

ChunkHound is MIT licensed, open source, and community built.

- [Docs](https://chunkhound.ai/docs/getting-started/)
- [Contributing](https://chunkhound.ai/docs/contributing/)
- [Discord](https://discord.gg/BAepHEXXnX)
- [Issues](https://github.com/chunkhound/chunkhound/issues)

## License

MIT

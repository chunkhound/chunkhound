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
  <strong>Your entire engineering context, deeply understood.</strong>
</p>

<p align="center">
  Open-source codebase intelligence that gives agents and teams cited context across current code, git history, and technical web research.
</p>

<!-- Keep in sync with site/src/components/Hero.astro amplifier line -->
<p align="center">
  Local-first · Dozens of languages & file types · Cited answers · Git history research · Pinpoint web research
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

Agents can generate code, but they still miss the context that makes software safe to change: how behavior flows across files, what changed across a branch or release, and which external constraints matter.

Reviewers, support, and product teams hit the same wall when large PRs, merge conflicts, bugs, and release notes need implementation-backed explanation instead of guesses.

ChunkHound turns current code, git history, and technical web research into cited context before anyone edits, reviews, debugs, or explains software.

## Deep understanding for four context-heavy jobs

ChunkHound applies codebase understanding to the workflows where missing context hurts most.

### Research before editing

Give coding agents grounded architecture context, relevant files, recent changes, and external constraints before they write code.

### Understand large PRs and releases

Turn branch diffs, commit ranges, tags, and specific commits into cited engineering briefs for review, release notes, and changelog drafts.

### Trace bugs and incidents

Turn symptoms, stack traces, and customer reports into likely code paths, recent changes, and external constraints.

### Reconcile code with external docs

Pinpoint the technical docs, APIs, issues, and articles your implementation depends on, then connect that external evidence to local code research.

## What you can ask

### Ground an agent before edits

```bash
chunkhound research "How does authentication work?"
chunkhound search "JWT refresh token validation"
chunkhound research "What changed in auth recently?" --last-n 20
```

### Understand a large PR or release

```bash
chunkhound research "Summarize the behavior changes on this branch for reviewers" --commit-range main..HEAD
chunkhound research "Draft changelog bullets for billing since v2.4" --commit-range v2.4..HEAD
chunkhound search "database migration" --commit-hash abc1234
```

### Get context before resolving conflicts

```bash
chunkhound research "Why did auth session handling change on each side?" --commit-range main..feature/auth
chunkhound search "session refresh conflict" --last-n 50
```

### Trace a bug with external constraints

```bash
chunkhound research "why would webhook retries fail?"
chunkhound research "what changed in webhook handling this week?" --last-n 30
chunkhound websearch "Stripe webhook retry schedule"
```

### Explain product behavior

```bash
chunkhound research "What happens when a user cancels a subscription?"
chunkhound research "What changed in billing since v2.4?" --commit-range v2.4..HEAD
```

## What powers deep understanding

- **Semantic code search** — find relevant code by meaning, not only exact text
- **Cited code research** — explain behavior across files with source citations
- **Git history research** — ask by last N commits, commit hash, tag, branch, or range to understand large PRs and releases
- **Pinpoint web research** — bring cited external docs, APIs, issues, and articles into the same workflow as local code research
- **Autodoc** — generate shareable docs from code-backed research
- **Local-first indexing** — keep code search and indexing under your control
- **Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, and more** via Tree-sitter

## Install

```bash
uv tool install chunkhound
```

## Try it

```bash
chunkhound index .
chunkhound research "How does authentication work?"
```

Index once, ask a real architecture question, and get a grounded answer with citations. Regex search works without providers. Semantic search requires an embedding provider. Deep research requires an LLM provider and an embedding provider with reranking support; web research uses the same provider stack. Choose local providers for zero-code-egress setups.

For a full configurable setup, create `.chunkhound.json` in your project root:

```json
{
  "embedding": { "provider": "voyageai", "api_key": "your-key" },
  "llm": { "provider": "claude-code-cli" }
}
```

For editor integration, all provider options, and advanced configuration:

**→ [chunkhound.ai/docs/getting-started](https://chunkhound.ai/docs/getting-started/)**

---

## Search git history

In addition to searching your indexed codebase, ChunkHound can search
code changes across git history — useful for understanding what changed
in a PR, a release, or since a specific commit.

```bash
# Last N commits
chunkhound search "authentication changes" --last-n 20

# Changes introduced by a specific commit
chunkhound search "database migration" --commit-hash abc1234

# Custom git range
chunkhound search "API changes" --commit-range v2.0..HEAD

# Deep research over recent changes
chunkhound research "what changed in the auth module?" --last-n 50
```

> `--vector-source` controls scope: `diff` (default, changed code only),
> `both` (merges diff + DB), `db` (ignore diff).

## Good fit

ChunkHound is especially useful for:

- large repos and monorepos
- multi-language codebases
- legacy systems
- local-only or security-sensitive environments
- engineering teams that want agents, support, and product questions grounded in the same code index

## Community

ChunkHound is MIT licensed, open source, and community built.

- [Docs](https://chunkhound.ai/docs/getting-started/)
- [Contributing](https://chunkhound.ai/docs/contributing/)
- [Discord](https://discord.gg/BAepHEXXnX)
- [Issues](https://github.com/chunkhound/chunkhound/issues)

## License

MIT

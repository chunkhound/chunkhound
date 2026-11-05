"""Semantic rendering contracts for the configurator output.

Samples embedding x llm x editor x platform x compact/full through the
public barrel (`site/src/components/configurator/index.ts`) via the real TS
path (`tests/site/tsx_runner.py:run_tsx_json`). No mocks.

The corpus is sampled, so the tests assert coverage and user-visible
tokenization: every (embedding, llm) pair remains exercised, posix and
PowerShell shells keep their forms, the preview wraps exactly the copied text,
and the highlighter's token classes and HTML escaping are stable. No byte
goldens, so harmless renderer refactors do not fail the suite.
"""

from __future__ import annotations

import html
import re

from tests.site.tsx_runner import run_tsx_json

# Barrel-only imports: goldens pin the public contract, not internals.
RENDER_SCRIPT = """
import {
  buildCompactConfiguratorOutput,
  buildFullConfiguratorOutput,
  editors,
  embeddingProviders,
  llmProviders,
  PLATFORM_OPTIONS,
} from './site/src/components/configurator/index.ts';

const out = {};
const modes = {
  compact: buildCompactConfiguratorOutput,
  full: buildFullConfiguratorOutput,
};
// Named embedding slots make catalog reordering irrelevant. Both modes use
// the same fixed shell assignments; the guards below fail when the catalog
// drifts. Pair coverage is asserted by
// test_highlight_corpus_covers_every_embedding_llm_pair in the Python tests.
function assignments(voyageai, openai, ollama, vllm) {
  return {
    voyageai,
    'openai-embed': openai,
    'ollama-embed': ollama,
    'vllm-embed': vllm,
  };
}
const shellAssignments = {
  'posix/pi': assignments('vercel', 'openrouter', 'orcarouter', 'anthropic'),
  'posix/cursor': assignments('openrouter', 'orcarouter', 'anthropic', 'openai-llm'),
  'posix/claude-code': assignments(
    'orcarouter', 'anthropic', 'openai-llm', 'codex-cli'),
  'posix/vscode': assignments(
    'anthropic', 'openai-llm', 'codex-cli', 'claude-code-cli'),
  'posix/opencode': assignments('openai-llm', 'codex-cli', 'claude-code-cli', 'gemini'),
  'posix/codex': assignments('codex-cli', 'claude-code-cli', 'gemini', 'deepseek'),
  'posix/windsurf': assignments('claude-code-cli', 'gemini', 'deepseek', 'grok'),
  'posix/roo-code': assignments('gemini', 'deepseek', 'grok', 'ollama-llm'),
  'posix/zed': assignments('deepseek', 'grok', 'ollama-llm', 'vllm-llm'),
  'powershell/pi': assignments('grok', 'ollama-llm', 'vllm-llm', 'opencode-cli'),
  'powershell/cursor': assignments('ollama-llm', 'vllm-llm', 'opencode-cli', 'vercel'),
  'powershell/claude-code': assignments(
    'vllm-llm', 'opencode-cli', 'vercel', 'openrouter'),
  'powershell/vscode': assignments(
    'opencode-cli', 'vercel', 'openrouter', 'orcarouter'),
  // Duplicate of posix/pi, so its four pairs stay covered when all four slots
  // switch to requesty: this closes the catalog-coverage gap with one shard.
  'powershell/opencode': assignments('requesty', 'requesty', 'requesty', 'requesty'),
  'powershell/codex': assignments(
    'openrouter', 'orcarouter', 'anthropic', 'openai-llm'),
  'powershell/windsurf': assignments(
    'orcarouter', 'anthropic', 'openai-llm', 'codex-cli'),
  'powershell/roo-code': assignments(
    'anthropic', 'openai-llm', 'codex-cli', 'claude-code-cli'),
  'powershell/zed': assignments('openai-llm', 'codex-cli', 'claude-code-cli', 'gemini'),
};
function requireAssignedProviders(providers, assigned, kind) {
  const catalog = providers.map(provider => provider.id).sort();
  const expected = [...new Set(assigned)].sort();
  if (JSON.stringify(catalog) !== JSON.stringify(expected)) {
    throw new Error(`Update golden ${kind} assignments: ${catalog} != ${expected}`);
  }
}
const samples = Object.values(shellAssignments);
requireAssignedProviders(embeddingProviders, samples.flatMap(Object.keys), 'embedding');
requireAssignedProviders(llmProviders, samples.flatMap(Object.values), 'LLM');
function renderShard(mode, build, platform, editor) {
  const assigned = shellAssignments[`${platform.id}/${editor.id}`];
  if (!assigned) {
    throw new Error(`Missing golden assignments: ${platform.id}/${editor.id}`);
  }
  for (const embedding of embeddingProviders) {
    const llm = llmProviders.find(provider => provider.id === assigned[embedding.id]);
    if (!llm) throw new Error(`Missing golden LLM assignment: ${embedding.id}`);
    const key = `${mode}/${platform.id}/${editor.id}/${embedding.id}/${llm.id}`;
    out[key] = build(embedding, llm, editor.id, platform.id);
  }
}
for (const [mode, build] of Object.entries(modes)) {
  for (const platform of PLATFORM_OPTIONS) {
    for (const editor of editors) renderShard(mode, build, platform, editor);
  }
}
console.log(JSON.stringify(out));
"""

# The preview is the copied text wrapped in style spans; stripping tags must
# reproduce `copy` exactly.
_TAG = re.compile(r"<[^>]+>")


# Module-local lazy cache: the three tests share one tsx subprocess instead
# of each spawning its own. Session-local to this module — no global state.
_rendered_cache: dict[str, dict[str, str]] | None = None


def _render_all() -> dict[str, dict[str, str]]:
    global _rendered_cache
    if _rendered_cache is None:
        _rendered_cache = run_tsx_json(RENDER_SCRIPT)
    return _rendered_cache


def test_highlight_tokens_and_escaping_are_stable() -> None:
    """The highlighter emits user-visible token classes and escapes markup.

    These are the classes the shipped CSS paints. The test pins the contract,
    not the full byte output, so a renderer refactor that keeps the same tokens
    and escaping stays green.
    """
    rendered = run_tsx_json(
        """
import { highlightJsonLine, highlightShellLine, highlightCommentLine }
  from './site/src/components/configurator/highlight.ts';
console.log(JSON.stringify({
  json: highlightJsonLine('  "provider": "vercel",'),
  jsonEscaped: highlightJsonLine('  "api_key": "<script>&",'),
  shell: highlightShellLine('chunkhound index ./src >> out.log'),
  shellEscaped: highlightShellLine('chunkhound index "<src>"'),
  comment: highlightCommentLine('# <note>'),
}));
"""
    )
    assert 'class="json-key"' in rendered["json"]
    assert 'class="json-string"' in rendered["json"]
    assert "&lt;script&gt;" in rendered["jsonEscaped"]
    assert "&amp;" in rendered["jsonEscaped"]
    assert "<script>" not in rendered["jsonEscaped"]
    assert 'class="sh-cmd"' in rendered["shell"]
    assert 'class="sh-op"' in rendered["shell"]
    assert "&lt;src&gt;" in rendered["shellEscaped"]
    assert 'class="sh-comment"' in rendered["comment"]
    assert "&lt;note&gt;" in rendered["comment"]


def test_highlight_corpus_covers_every_embedding_llm_pair() -> None:
    """The corpus samples providers, so it must still exercise every
    (embedding, llm) pair — otherwise the trim silently drops a rendering path
    the full cross product used to freeze."""
    rendered = _render_all()
    pairs = {(key.split("/")[3], key.split("/")[4]) for key in rendered}
    embeddings = {embedding for embedding, _ in pairs}
    llms = {llm for _, llm in pairs}
    assert len(pairs) == len(embeddings) * len(llms), (
        "sampled golden corpus misses an (embedding, llm) pair: "
        f"{len(pairs)} for {len(embeddings)}x{len(llms)} providers"
    )
    # The square check above stays green when a provider disappears from the
    # corpus and the catalog together, so pin the catalog itself. Mirrors the
    # TS `requireAssignedProviders` guard; drift here means lost rendering paths.
    llm_catalog = {
        "anthropic",
        "claude-code-cli",
        "codex-cli",
        "deepseek",
        "gemini",
        "grok",
        "ollama-llm",
        "openai-llm",
        "opencode-cli",
        "openrouter",
        "orcarouter",
        "requesty",
        "vercel",
        "vllm-llm",
    }
    assert llms == llm_catalog, (
        "golden corpus LLM set diverged from the catalog: "
        f"missing {sorted(llm_catalog - llms)}, unexpected {sorted(llms - llm_catalog)}"
    )


def test_highlight_corpus_covers_both_shell_forms() -> None:
    rendered = _render_all()
    posix = {k: v for k, v in rendered.items() if k.startswith("compact/posix/")}
    assert posix and all("<<'CHUNKHOUND_EOF'" in v["copy"] for v in posix.values())
    powershell = {k: v for k, v in rendered.items() if "/powershell/" in k}
    assert powershell and all(
        "'@ | Set-Content" in v["copy"] for v in powershell.values()
    )


def test_highlight_preview_only_wraps_the_copied_text() -> None:
    """The preview is the copied text plus style spans, and nothing else.

    Stripping every tag and unescaping entities has to reproduce ``copy``
    byte-for-byte, so a renderer change cannot silently drop, reorder, or
    rewrite what the user pastes into a shell.
    """
    rendered = _render_all()
    assert rendered, "barrel rendered no combinations"
    for key, entry in rendered.items():
        stripped = html.unescape(_TAG.sub("", entry["html"]))
        assert stripped == entry["copy"], f"highlighted preview diverges at {key}"

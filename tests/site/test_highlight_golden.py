"""Byte-identical rendering goldens for the configurator output.

Samples embedding x llm x editor x platform x compact/full through the
public barrel (`site/src/components/configurator/index.ts`) via the real TS
path (`tests/site/tsx_runner.py:run_tsx_json`). No mocks.

Goldens live in `tests/site/__goldens__/highlight/`, one shard per
mode/platform/editor (`{mode}.{platform}.{editor}.json`) mapping
`{mode}/{platform}/{editor}/{embedding}/{llm}` -> {"html": ..., "copy": ...}.

Each shard keeps every embedding provider (so a provider-specific rendering
bug is caught in all 36 shells) and advances the llm provider one step per
shard, so the corpus renders every 4x12 (embedding, llm) pair across its 36
shards without storing the full cross product in every shard. Regenerate with
`UPDATE_GOLDENS=1`; `git diff --exit-code` then proves parity for the frozen
`highlight.ts` renderer.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

from tests.site.tsx_runner import run_tsx_json

GOLDEN_DIR = Path(__file__).resolve().parent / "__goldens__" / "highlight"
UPDATE_GOLDENS = os.environ.get("UPDATE_GOLDENS") == "1"

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
// Representative corpus: every embedding provider in every shard, with the llm
// provider advancing one step per shard (plus the embedding offset), so the 36
// shards together render all 48 (embedding, llm) pairs. Rendering the full 4x12
// cross product in every shard would only repeat those pairs; the structural
// tests below pin the rest of the contract, including shared API-key dedupe
// (see test_output_deduplicates_shared_api_key_requirements in the rendering
// tests).
let shard = 0;
for (const [mode, build] of Object.entries(modes)) {
  for (const platform of PLATFORM_OPTIONS) {
    for (const editor of editors) {
      for (let e = 0; e < embeddingProviders.length; e++) {
        const embedding = embeddingProviders[e];
        const llm =
          llmProviders[
            (shard + e) % llmProviders.length
          ];
        const rendered = build(embedding, llm, editor.id, platform.id);
        const key =
          `${mode}/${platform.id}/${editor.id}/${embedding.id}/${llm.id}`;
        out[key] = rendered;
      }
      shard += 1;
    }
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


def _shard_of(key: str) -> str:
    mode, platform, editor, _embedding, _llm = key.split("/")
    return f"{mode}.{platform}.{editor}.json"


def _group_by_shard(rendered: dict[str, dict[str, str]]) -> dict[str, dict]:
    shards: dict[str, dict] = {}
    for key in sorted(rendered):
        shards.setdefault(_shard_of(key), {})[key] = rendered[key]
    return shards


def _write_shard(name: str, entries: dict) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    (GOLDEN_DIR / name).write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _disk_shards() -> set[str]:
    """Shard file names currently on disk (empty when the dir is absent)."""
    if not GOLDEN_DIR.exists():
        return set()
    return {p.name for p in GOLDEN_DIR.glob("*.json")}


def _load_shard(name: str) -> dict:
    path = GOLDEN_DIR / name
    assert path.exists(), f"missing golden {name}; rerun with UPDATE_GOLDENS=1"
    return json.loads(path.read_text(encoding="utf-8"))


def test_highlight_goldens_match_exact_rendering() -> None:
    rendered = _render_all()
    assert rendered, "barrel rendered no combinations"
    shards = _group_by_shard(rendered)
    if UPDATE_GOLDENS:
        # Regeneration is the fix path for intentional option removals.
        for orphan in _disk_shards() - set(shards):
            (GOLDEN_DIR / orphan).unlink()
        for name, entries in shards.items():
            _write_shard(name, entries)
        return
    orphans = _disk_shards() - set(shards)
    assert not orphans, (
        f"orphan golden shards {sorted(orphans)}; "
        "rerun with UPDATE_GOLDENS=1 to regenerate"
    )
    for name, entries in shards.items():
        assert _load_shard(name) == entries, (
            f"golden drift in {name}; rerun with UPDATE_GOLDENS=1 to regenerate"
        )
    stored_keys = {key for name in shards for key in _load_shard(name)}
    assert stored_keys == set(rendered), "golden key set diverged from barrel"


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

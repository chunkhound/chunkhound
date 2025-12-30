from __future__ import annotations

from pathlib import Path

from chunkhound.autodoc.models import CodeMapperIndex, DocsitePage, DocsiteSite
from chunkhound.autodoc.site_writer import write_astro_assets_only, write_astro_site
from tests.autodoc.site_tree_manifest import build_tree_manifest

_FULL_SITE_EXPECTED_MANIFEST: dict[str, str] = {
    "README.md": "0aaed6dd0601528919f86524ec9fed4f730f193aed8e445035b88472fbd6d203",
    "astro.config.mjs": (
        "7fcafc68489f3a8262965976a470c0d86da51979cbc64edb7efde601c5de4c32"
    ),
    "package.json": "0530749628cdd0f893114e9a8c9d507cecb989a8ae314d4692d40024a3c62341",
    "public/favicon.ico": (
        "d014edc031656dd8a5cb7740ed900d658ba3108ff6fcb977fc3ff4f758c10f0b"
    ),
    "src/data/nav.json": (
        "17cd99edb59031fb7a7459c15f7771e1c30b9f26a811fc975eb56a46ba5b8c8a"
    ),
    "src/data/search.json": (
        "b3c47239cb696caa23652666ad76a72cdb00d81ba5223f0b0529d1d3fc1876ad"
    ),
    "src/data/site.json": (
        "3bb38a8c41b09bdd0b4681c8d3461fa213c40f8940041a4fdd83c096674dc993"
    ),
    "src/layouts/DocLayout.astro": (
        "fde0db562238b8f202b935011b8563fc03e785a9d19a7d451f88dc4367a225ac"
    ),
    "src/pages/index.md": (
        "52103c6295cc25902523f0e0211f95ad4b679a0842396f8dd7778e0aa2584f0f"
    ),
    "src/pages/topics/01-topic-one.md": (
        "afba982e414b91f87f9c5ba06a35acc13de490459fe8532942d4c75f599d09ac"
    ),
    "src/pages/topics/02-topic-two.md": (
        "2ea825bb4f5a16d6e422cc2193514a90ee94585fe594028c795eead8781be03d"
    ),
    "src/styles/global.css": (
        "cc160bacd9f2702fa7b305fe798f380f25135639426451bbd886e276220bc565"
    ),
    "tsconfig.json": "ec0d7fbe45c5b2efb1c617eec3a7ee249d72974163c5309843420904703ee0a4",
}


_ASSETS_ONLY_EXPECTED_MANIFEST: dict[str, str] = {
    "README.md": "0aaed6dd0601528919f86524ec9fed4f730f193aed8e445035b88472fbd6d203",
    "astro.config.mjs": (
        "7fcafc68489f3a8262965976a470c0d86da51979cbc64edb7efde601c5de4c32"
    ),
    "package.json": "0530749628cdd0f893114e9a8c9d507cecb989a8ae314d4692d40024a3c62341",
    "public/favicon.ico": (
        "d014edc031656dd8a5cb7740ed900d658ba3108ff6fcb977fc3ff4f758c10f0b"
    ),
    "src/data/site.json": (
        "367608f04d1ad213ec25b9cbd83f53c39b8f7c1a765a8eeacf360f4d5603dc27"
    ),
    "src/layouts/DocLayout.astro": (
        "fde0db562238b8f202b935011b8563fc03e785a9d19a7d451f88dc4367a225ac"
    ),
    "src/pages/topics/sentinel.md": (
        "4a1b4fb33e6c8a7d979d87905829dbba93ac1dbb4c98105c5527d08e90e7996c"
    ),
    "src/styles/global.css": (
        "cc160bacd9f2702fa7b305fe798f380f25135639426451bbd886e276220bc565"
    ),
    "tsconfig.json": "ec0d7fbe45c5b2efb1c617eec3a7ee249d72974163c5309843420904703ee0a4",
}


def test_write_astro_site_emits_byte_stable_tree(tmp_path: Path) -> None:
    metadata_block = """agent_doc_metadata:
  created_from_sha: abc123
  generated_at: 2025-01-01T00:00:00Z
  llm_config:
    provider: codex-cli
    synthesis_provider: codex-cli
    synthesis_model: gpt-5
    utility_model: gpt-5-mini
    codex_reasoning_effort_synthesis: high
    codex_reasoning_effort_utility: low
    map_hyde_provider: codex-cli
    map_hyde_model: gpt-5
    map_hyde_reasoning_effort: medium
  generation_stats:
    generator_mode: code_research
    total_research_calls: 10
    autodoc_comprehensiveness: medium
    files:
      referenced: 5
      total_indexed: 20
      basis: scope
      coverage: 25.0%
      referenced_in_scope: 5
      unreferenced_in_scope: 15
      unreferenced_list_file: unref.txt
    chunks:
      referenced: 30
      total_indexed: 200
      basis: scope
      coverage: 15.0%
"""

    site = DocsiteSite(
        title="My Docs",
        tagline="Tagline",
        scope_label="repo",
        generated_at="2025-01-01T00:00:00Z",
        source_dir="/repo",
        topic_count=2,
    )
    pages = [
        DocsitePage(
            order=1,
            title="Topic One",
            slug="01-topic-one",
            description="First topic",
            body_markdown="## Overview\n\nHello `world`.\n",
            source_path="input/topic-one.md",
            scope_label="repo",
            references_count=1,
        ),
        DocsitePage(
            order=2,
            title="Topic Two",
            slug="02-topic-two",
            description="Second topic",
            body_markdown="## Overview\n\nSecond body.\n\n## References\n\n- [x](y)\n",
            source_path="input/topic-two.md",
            scope_label="repo",
            references_count=0,
        ),
    ]
    index = CodeMapperIndex(
        title="AutoDoc Topics",
        scope_label="repo",
        metadata_block=metadata_block,
        topics=[],
    )

    output_dir = tmp_path / "site"
    write_astro_site(output_dir=output_dir, site=site, pages=pages, index=index)

    assert build_tree_manifest(output_dir) == _FULL_SITE_EXPECTED_MANIFEST


def test_write_astro_assets_only_emits_byte_stable_tree(tmp_path: Path) -> None:
    output_dir = tmp_path / "site"

    topics_dir = output_dir / "src" / "pages" / "topics"
    topics_dir.mkdir(parents=True)
    topic_path = topics_dir / "sentinel.md"
    topic_path.write_text("sentinel topic\n", encoding="utf-8")

    data_dir = output_dir / "src" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "site.json").write_text(
        "{\n"
        '  "title": "My Docs",\n'
        '  "tagline": "Tagline",\n'
        '  "scopeLabel": "repo",\n'
        '  "generatedAt": "2025-01-01T00:00:00Z",\n'
        '  "sourceDir": "/repo",\n'
        '  "topicCount": 1\n'
        "}\n",
        encoding="utf-8",
    )

    write_astro_assets_only(output_dir=output_dir)

    assert topic_path.read_text(encoding="utf-8") == "sentinel topic\n"
    assert build_tree_manifest(output_dir) == _ASSETS_ONLY_EXPECTED_MANIFEST

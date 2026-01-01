import json
from pathlib import Path

import pytest

from chunkhound.autodoc.cleanup import _cleanup_with_llm
from chunkhound.autodoc.index_loader import find_index_file
from chunkhound.autodoc.models import (
    CleanupConfig,
    CodeMapperIndex,
    CodeMapperTopic,
    DocsitePage,
    DocsiteSite,
)
from chunkhound.autodoc.references import flatten_sources_block
from chunkhound.autodoc.site_writer import (
    _render_astro_config,
    _render_doc_layout,
    _render_index_page,
    _render_search_index,
    write_astro_site,
)


def test_write_astro_site_removes_stale_topics(tmp_path: Path) -> None:
    output_dir = tmp_path / "site"
    topics_dir = output_dir / "src" / "pages" / "topics"
    topics_dir.mkdir(parents=True)
    stale_path = topics_dir / "old-topic.md"
    stale_path.write_text("stale", encoding="utf-8")

    site = DocsiteSite(
        title="Test Site",
        tagline="Tagline",
        scope_label="/",
        generated_at="2025-12-22T00:00:00Z",
        source_dir=str(tmp_path),
        topic_count=1,
    )
    pages = [
        DocsitePage(
            order=1,
            title="New Topic",
            slug="new-topic",
            description="New topic description.",
            body_markdown="Body",
        )
    ]
    index = CodeMapperIndex(
        title="Index",
        scope_label="/",
        metadata_block=None,
        topics=[],
    )

    write_astro_site(
        output_dir=output_dir,
        site=site,
        pages=pages,
        index=index,
        allow_delete_topics_dir=True,
    )

    assert not stale_path.exists()
    assert (topics_dir / "new-topic.md").exists()


def test_find_index_file_warns_on_multiple_candidates(tmp_path: Path) -> None:
    code_mapper = tmp_path / "scope_code_mapper_index.md"
    autodoc = tmp_path / "scope_autodoc_index.md"
    code_mapper.write_text("# Index", encoding="utf-8")
    autodoc.write_text("# Index", encoding="utf-8")

    warnings: list[str] = []

    def log_warning(message: str) -> None:
        warnings.append(message)

    selected = find_index_file(tmp_path, log_warning=log_warning)

    assert selected == code_mapper
    assert warnings
    assert "Multiple AutoDoc index files found" in warnings[0]


def test_render_astro_config_disables_dev_toolbar() -> None:
    config = _render_astro_config()

    assert "devToolbar" in config
    assert "enabled: false" in config


def test_render_index_page_wraps_generation_details() -> None:
    metadata_block = "\n".join(
        [
            "agent_doc_metadata:",
            "  generated_at: 2025-01-01T00:00:00Z",
        ]
    )
    index = CodeMapperIndex(
        title="AutoDoc Topics",
        scope_label="/",
        metadata_block=metadata_block,
        topics=[],
    )

    output = _render_index_page(
        site=DocsiteSite(
            title="Test",
            tagline="Tag",
            scope_label="/",
            generated_at="2025-12-22T00:00:00Z",
            source_dir=".",
            topic_count=0,
        ),
        pages=[],
        index=index,
    )

    assert "<details" in output
    assert "Generation Details" in output


def test_flatten_sources_block_wraps_paths_in_code() -> None:
    sources_block = "\n".join(
        [
            "## Sources",
            "",
            "└── repo/",
            "\t└── [1] __init__.py (1 chunks: L1-2)",
        ]
    )

    flattened = flatten_sources_block(sources_block)

    assert flattened == ["- [1] `repo/__init__.py` (1 chunks: L1-2)"]


def test_render_doc_layout_includes_search_and_toc() -> None:
    layout = _render_doc_layout()

    assert "data-search-input" in layout
    assert "data-nav-filter" in layout
    assert "data-toc" in layout
    assert "aria-current" in layout
    assert "page-nav" in layout
    assert "navGroups" in layout
    assert "navData" in layout


def test_render_doc_layout_avoids_dom_injection_and_cdn_imports() -> None:
    layout = _render_doc_layout()

    assert "innerHTML" not in layout
    assert "cdn.jsdelivr.net" not in layout
    assert "import('mermaid')" in layout
    assert ".replace(/</g, '\\\\u003c')" in layout


def test_render_search_index_includes_body_text() -> None:
    pages = [
        DocsitePage(
            order=1,
            title="Topic",
            slug="topic",
            description="Desc",
            body_markdown="## Overview\nHello `world`.\n\n## Details\nMore text.",
        )
    ]

    payload = json.loads(_render_search_index(pages))

    assert payload[0]["title"] == "Topic"
    assert "Hello world" in payload[0]["body"]


class _NoHeadingProvider:
    async def batch_complete(  # type: ignore[no-untyped-def]
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ):
        from chunkhound.interfaces.llm_provider import LLMResponse

        return [
            LLMResponse(
                content="First paragraph.\n\n- item one\n- item two",
                tokens_used=0,
                model="fake",
                finish_reason="stop",
            )
            for _ in prompts
        ]


@pytest.mark.asyncio
async def test_llm_cleanup_inserts_overview_heading_when_missing() -> None:
    topics = [
        CodeMapperTopic(
            order=1,
            title="Topic",
            source_path=Path("topic.md"),
            raw_markdown="Body",
            body_markdown="First paragraph.\n\n- item one\n- item two",
        )
    ]

    cleaned = await _cleanup_with_llm(
        topics=topics,
        provider=_NoHeadingProvider(),  # type: ignore[arg-type]
        config=CleanupConfig(mode="llm", batch_size=1, max_completion_tokens=512),
        log_info=None,
        log_warning=None,
    )

    assert cleaned[0].startswith("## Overview")

import json
from pathlib import Path

from chunkhound.autodoc import docsite


def test_write_astro_site_removes_stale_topics(tmp_path: Path) -> None:
    output_dir = tmp_path / "site"
    topics_dir = output_dir / "src" / "pages" / "topics"
    topics_dir.mkdir(parents=True)
    stale_path = topics_dir / "old-topic.md"
    stale_path.write_text("stale", encoding="utf-8")

    site = docsite.DocsiteSite(
        title="Test Site",
        tagline="Tagline",
        scope_label="/",
        generated_at="2025-12-22T00:00:00Z",
        source_dir=str(tmp_path),
        topic_count=1,
    )
    pages = [
        docsite.DocsitePage(
            order=1,
            title="New Topic",
            slug="new-topic",
            description="New topic description.",
            body_markdown="Body",
        )
    ]
    index = docsite.CodeMapperIndex(
        title="Index",
        scope_label="/",
        metadata_block=None,
        topics=[],
    )

    docsite.write_astro_site(
        output_dir=output_dir,
        site=site,
        pages=pages,
        index=index,
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

    selected = docsite.find_index_file(tmp_path, log_warning=log_warning)

    assert selected == code_mapper
    assert warnings
    assert "Multiple AutoDoc index files found" in warnings[0]


def test_render_astro_config_disables_dev_toolbar() -> None:
    config = docsite._render_astro_config()

    assert "devToolbar" in config
    assert "enabled: false" in config


def test_render_index_page_wraps_generation_details() -> None:
    metadata_block = "\n".join(
        [
            "agent_doc_metadata:",
            "  generated_at: 2025-01-01T00:00:00Z",
        ]
    )
    index = docsite.CodeMapperIndex(
        title="AutoDoc Topics",
        scope_label="/",
        metadata_block=metadata_block,
        topics=[],
    )

    output = docsite._render_index_page(
        site=docsite.DocsiteSite(
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

    flattened = docsite.flatten_sources_block(sources_block)

    assert flattened == ["- [1] `repo/__init__.py` (1 chunks: L1-2)"]


def test_render_doc_layout_includes_search_and_toc() -> None:
    layout = docsite._render_doc_layout()

    assert "data-search-input" in layout
    assert "data-nav-filter" in layout
    assert "data-toc" in layout
    assert "aria-current" in layout
    assert "page-nav" in layout
    assert "navGroups" in layout
    assert "navData" in layout


def test_render_search_index_includes_body_text() -> None:
    pages = [
        docsite.DocsitePage(
            order=1,
            title="Topic",
            slug="topic",
            description="Desc",
            body_markdown="## Overview\nHello `world`.\n\n## Details\nMore text.",
        )
    ]

    payload = json.loads(docsite._render_search_index(pages))

    assert payload[0]["title"] == "Topic"
    assert "Hello world" in payload[0]["body"]


def test_minimal_cleanup_inserts_overview_heading_when_missing() -> None:
    topic = docsite.CodeMapperTopic(
        order=1,
        title="Topic",
        source_path=Path("topic.md"),
        raw_markdown="Body",
        body_markdown="First paragraph.\n\n- item one\n- item two",
    )

    output = docsite._minimal_cleanup(topic)

    assert output.startswith("## Overview")

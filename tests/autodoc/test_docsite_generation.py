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

from pathlib import Path

from chunkhound.autodoc import docsite


def test_render_topic_page_emits_v2_frontmatter_when_present() -> None:
    page = docsite.DocsitePage(
        order=1,
        title="Topic",
        slug="topic",
        description="Desc",
        body_markdown="## Overview\nHello.\n\n## References\n- [1] `x.py` (1 chunks: L1-2)",
        source_path=str(Path("input/topic.md")),
        scope_label="/repo",
        references_count=1,
    )

    output = docsite._render_topic_page(page)

    assert 'sourcePath: "input/topic.md"' in output
    assert 'scope: "/repo"' in output
    assert "referencesCount: 1" in output
    assert "tags:" not in output


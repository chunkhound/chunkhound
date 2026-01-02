from pathlib import Path

from chunkhound.autodoc.models import DocsitePage
from chunkhound.autodoc.site_writer import _render_topic_page


def test_render_topic_page_emits_v2_frontmatter_when_present() -> None:
    page = DocsitePage(
        order=1,
        title="Topic",
        slug="topic",
        description="Desc",
        body_markdown=(
            "## Overview\nHello.\n\n## References\n"
            "- [1] `x.py` (1 chunks: L1-2)"
        ),
        source_path=str(Path("input/topic.md")),
        scope_label="/repo",
        references_count=1,
    )

    output = _render_topic_page(page)

    assert 'sourcePath: "input/topic.md"' in output
    assert 'scope: "/repo"' in output
    assert "referencesCount: 1" in output
    assert "tags:" not in output

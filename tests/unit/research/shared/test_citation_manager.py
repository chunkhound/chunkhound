"""Tests for citation source rendering."""

from chunkhound.services.research.shared.citation_manager import CitationManager


def test_sources_footer_keeps_web_urls_intact() -> None:
    manager = CitationManager()
    url = "https://example.com/docs/auth?lang=en"

    footer = manager.build_sources_footer(
        chunks=[
            {
                "file_path": url,
                "start_line": 1,
                "end_line": 4,
            }
        ],
        files={url: "page content"},
        file_reference_map={url: 1},
    )

    assert f"[1] <{url}> (1 chunks: L1-4)" in footer
    assert "https:/" not in footer.replace(f"<{url}>", "")


def test_sources_footer_keeps_repository_paths_as_tree() -> None:
    footer = CitationManager().build_sources_footer(
        chunks=[
            {"file_path": "src/main.py", "start_line": 2, "end_line": 5}
        ],
        files={"src/main.py": "print('hello')"},
    )

    assert "└── src/" in footer
    assert "└── main.py (1 chunks: L2-5)" in footer

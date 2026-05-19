"""Unit tests for ``_html_to_markdown``.

Covers the two conversion contracts enforced by the implementation:

1. The *bodies* of ``<script>``, ``<style>``, and ``<svg>`` are dropped.
   markdownify's ``strip=`` only removes the wrapper tag and would otherwise
   leak raw JS/CSS/SVG source into the markdown — the converter in
   ``_html_to_markdown`` overrides ``convert_script``/``convert_style``/
   ``convert_svg`` to return an empty string.
2. Structural chrome tags listed in ``strip=`` are removed from the HTML
   (so ``<nav>``, ``<form>`` etc. do not appear as raw tags in the output).
"""

from __future__ import annotations

from chunkhound.api.cli.commands.websearch import _html_to_markdown


def test_strips_script_body_not_just_tag() -> None:
    html = "<html><body><script>alert(1)</script><p>Hello</p></body></html>"
    md = _html_to_markdown(html)
    assert "alert(1)" not in md
    assert "Hello" in md


def test_strips_style_body_not_just_tag() -> None:
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><p>Visible</p></body></html>"
    )
    md = _html_to_markdown(html)
    assert "color:red" not in md
    assert "body{" not in md
    assert "Visible" in md


def test_strips_svg_body_not_just_tag() -> None:
    html = (
        "<html><body>"
        '<svg xmlns="http://www.w3.org/2000/svg"><path d="M10 10h80v80H10z"/></svg>'
        "<p>After svg</p>"
        "</body></html>"
    )
    md = _html_to_markdown(html)
    assert "M10 10h80v80H10z" not in md
    assert "<path" not in md
    assert "After svg" in md


def test_strips_structural_chrome_wrapper_tags() -> None:
    # markdownify's strip= drops the wrapper tags — verify the raw tags do
    # not survive conversion and that main/body text still flows through.
    html = """
    <html>
      <head><title>HEADTITLE</title></head>
      <body>
        <header>HEADER</header>
        <nav>NAVTEXT</nav>
        <aside>ASIDETEXT</aside>
        <form>FORMTEXT <button>BUTTONTEXT</button></form>
        <iframe>IFRAMETEXT</iframe>
        <noscript>NOSCRIPTTEXT</noscript>
        <main><p>MAINCONTENT</p></main>
        <footer>FOOTERTEXT</footer>
      </body>
    </html>
    """
    md = _html_to_markdown(html)
    for tag in (
        "<head", "<nav", "<footer", "<header", "<aside",
        "<form", "<button", "<iframe", "<noscript",
    ):
        assert tag not in md, f"wrapper {tag!r} should have been stripped"
    assert "MAINCONTENT" in md


def test_preserves_main_content_with_atx_headings() -> None:
    html = "<h1>Title</h1><h2>Sub</h2><p>Body text.</p>"
    md = _html_to_markdown(html)
    assert "# Title" in md
    assert "## Sub" in md
    assert "Body text." in md

"""Unit tests for HTML parser."""

import pytest
from pathlib import Path

from chunkhound.core.types.common import Language, ChunkType
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.fixture
def html_parser():
    """Create an HTML parser instance."""
    factory = ParserFactory()
    return factory.create_parser(Language.HTML)


@pytest.fixture
def comprehensive_html():
    """Load comprehensive HTML test file."""
    fixture_path = Path(__file__).parent / "fixtures" / "html" / "comprehensive.html"
    if not fixture_path.exists():
        pytest.skip(f"Test fixture not found: {fixture_path}")
    return fixture_path


def test_parses_semantic_elements_as_block(html_parser):
    """Semantic landmark elements are extracted as BLOCK chunks."""
    code = """<!DOCTYPE html>
<html>
<body>
  <header id="top">
    <h1>Site Title</h1>
  </header>
  <main>
    <article class="post">
      <p>Content</p>
    </article>
    <aside>Sidebar</aside>
  </main>
  <footer>Footer</footer>
</body>
</html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks found for semantic elements"
    symbols = {c.symbol for c in block_chunks}
    assert any("header" in s for s in symbols), f"header not found in {symbols}"
    assert any("main" in s or "article" in s or "aside" in s or "footer" in s for s in symbols)


def test_parses_custom_elements_as_block(html_parser):
    """Custom elements (with hyphen) are extracted as BLOCK chunks."""
    code = """<html><body>
  <my-component id="widget" data-value="42">
    <span>Content</span>
  </my-component>
  <app-header class="top-bar">Title</app-header>
</body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks for custom elements"
    symbols = {c.symbol for c in block_chunks}
    assert any("my-component" in s or "app-header" in s for s in symbols), (
        f"Custom element not found in {symbols}"
    )


def test_parses_link_stylesheet_as_import(html_parser):
    """<link rel='stylesheet'> elements are extracted as IMPORT chunks."""
    code = """<html><head>
  <link rel="stylesheet" href="styles/main.css">
  <link rel="stylesheet" href="theme.css">
</head><body></body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) > 0, "No import chunks found for link[rel=stylesheet]"
    symbols = {c.symbol for c in import_chunks}
    assert any("main.css" in s or "theme.css" in s for s in symbols), (
        f"CSS href not in import symbols: {symbols}"
    )


def test_parses_script_src_as_block(html_parser):
    """<script src='...'> elements are extracted as BLOCK chunks (not IMPORT).

    Note: script_element is captured by the BLOCK query. The IMPORT query only
    captures (element) nodes, but <script> maps to script_element in the grammar.
    """
    code = """<html><head>
  <script src="js/app.js"></script>
  <script src="vendor/lodash.min.js"></script>
</head><body></body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks found for script[src]"
    symbols = {c.symbol for c in block_chunks}
    assert any("script" in s and "src=" in s for s in symbols), (
        f"script[src=...] block not found in {symbols}"
    )


def test_parses_inline_script_as_block(html_parser):
    """Inline <script> blocks (no src) are extracted as BLOCK chunks."""
    code = """<html><body>
<script>
  var x = 1;
  console.log(x);
</script>
</body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunk for inline script"
    assert any("script" in c.symbol for c in block_chunks), "No script block chunk found"


def test_parses_inline_style_as_block(html_parser):
    """Inline <style> blocks are extracted as BLOCK chunks."""
    code = """<html><head>
<style>
  body { margin: 0; }
  .container { max-width: 1200px; }
</style>
</head><body></body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunk for inline style"
    assert any("style" in c.symbol for c in block_chunks), "No style block chunk found"


def test_parses_comments(html_parser):
    """HTML comments are extracted as COMMENT chunks."""
    code = """<html><body>
<!-- Main navigation comment -->
<nav>
  <!-- nested comment -->
  <a href="/">Home</a>
</nav>
<!-- Footer comment -->
</body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    comment_chunks = [c for c in chunks if c.chunk_type == ChunkType.COMMENT]
    assert len(comment_chunks) > 0, "No COMMENT chunks found"
    assert all(c.symbol.startswith("comment_line") for c in comment_chunks)


def test_parses_doctype_as_structure(html_parser):
    """DOCTYPE is extracted as a STRUCTURE (NAMESPACE) chunk."""
    code = """<!DOCTYPE html>
<html><head><title>Test</title></head><body></body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    structure_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
    assert len(structure_chunks) > 0, "No STRUCTURE chunk for DOCTYPE"
    assert any(c.symbol == "doctype" for c in structure_chunks)


def test_non_semantic_elements_not_extracted(html_parser):
    """Non-semantic elements (div, span, p) are NOT extracted as BLOCK chunks."""
    code = """<html><body>
  <div class="wrapper">
    <span class="label">Text</span>
    <p>A paragraph</p>
    <ul><li>Item</li></ul>
  </div>
</body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    # None of div/span/p/ul/li should appear as block chunks
    for chunk in block_chunks:
        tag = chunk.metadata.get("tag_name", "") if chunk.metadata else ""
        assert tag not in ("div", "span", "p", "ul", "li"), (
            f"Non-semantic element '{tag}' was incorrectly extracted as BLOCK"
        )


def test_element_name_uses_id(html_parser):
    """Element symbol includes id attribute when present."""
    code = """<html><body>
  <section id="intro">Content</section>
</body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert any("section#intro" in c.symbol for c in block_chunks), (
        f"Expected 'section#intro' in symbols, got: {[c.symbol for c in block_chunks]}"
    )


def test_element_name_uses_class_when_no_id(html_parser):
    """Element symbol uses first class when no id present."""
    code = """<html><body>
  <article class="blog-post featured">Content</article>
</body></html>"""
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert any("article.blog-post" in c.symbol for c in block_chunks), (
        f"Expected 'article.blog-post' in symbols, got: {[c.symbol for c in block_chunks]}"
    )


def test_comprehensive_file(html_parser, comprehensive_html):
    """Parse the comprehensive HTML fixture and check coverage."""
    chunks = html_parser.parse_file(comprehensive_html, file_id=1)

    assert len(chunks) > 5, f"Expected more than 5 chunks, got {len(chunks)}"

    chunk_types = {c.chunk_type for c in chunks}

    # Must have BLOCK (semantic elements - cAST may merge adjacent elements)
    assert ChunkType.BLOCK in chunk_types, f"No BLOCK chunks found. Types: {chunk_types}"

    # Must have COMMENT
    assert ChunkType.COMMENT in chunk_types, f"No COMMENT chunks found. Types: {chunk_types}"

    # Must have NAMESPACE (DOCTYPE = STRUCTURE)
    assert ChunkType.NAMESPACE in chunk_types, f"No NAMESPACE/STRUCTURE chunks. Types: {chunk_types}"

    # Must have IMPORT (link[rel=stylesheet] imports)
    assert ChunkType.IMPORT in chunk_types, f"No IMPORT chunks. Types: {chunk_types}"

    # Verify that DOCTYPE chunk is present
    assert any(c.symbol == "doctype" for c in chunks), "doctype chunk not found"

    # Verify stylesheet imports are captured (may be merged into one chunk)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) >= 1, f"Expected at least 1 import chunk, got {len(import_chunks)}"
    import_code = " ".join(c.code for c in import_chunks)
    assert "css" in import_code.lower() or "stylesheet" in import_code.lower(), (
        "No CSS import found in import chunks"
    )

    # Verify semantic content is present somewhere in block chunk code
    all_block_code = " ".join(c.code for c in chunks if c.chunk_type == ChunkType.BLOCK)
    assert "main" in all_block_code or "article" in all_block_code, (
        "No main/article content in BLOCK chunks"
    )

    # Must have multiple COMMENT chunks
    comment_chunks = [c for c in chunks if c.chunk_type == ChunkType.COMMENT]
    assert len(comment_chunks) >= 3, f"Expected at least 3 comment chunks, got {len(comment_chunks)}"


def test_unquoted_attribute_value():
    """_get_attribute handles unquoted attribute values."""
    from chunkhound.parsers.parser_factory import ParserFactory
    from chunkhound.core.types.common import Language
    factory = ParserFactory()
    parser = factory.create_parser(Language.HTML)
    # Unquoted id attribute
    code = '<section id=intro><h1>Hello</h1></section>'
    chunks = parser.parse_content(code, "test.html", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunk for section with unquoted id"
    assert any("intro" in c.symbol for c in block_chunks), (
        f"section#intro not found in {[c.symbol for c in block_chunks]}"
    )


def test_jinja_language_produces_chunks():
    """Language.JINJA (HTML grammar) parses templates and produces BLOCK chunks."""
    from chunkhound.parsers.parser_factory import ParserFactory
    from chunkhound.core.types.common import Language
    factory = ParserFactory()
    parser = factory.create_parser(Language.JINJA)
    # Jinja {{ }} expressions are treated as plain text by the HTML grammar
    code = """<!DOCTYPE html>
<html>
  <body>
    <section id="main">
      <h1>{{ title }}</h1>
      {% for item in items %}<p>{{ item }}</p>{% endfor %}
    </section>
  </body>
</html>"""
    chunks = parser.parse_content(code, "template.html", file_id=1)
    assert len(chunks) > 0, "Language.JINJA parser returned no chunks"
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks for Jinja template"
    # All chunks must be labeled JINJA, not HTML
    for chunk in chunks:
        assert chunk.language == Language.JINJA, (
            f"Chunk language should be JINJA, got {chunk.language}"
        )


def test_resolve_import_paths_html(tmp_path):
    """resolve_import_paths resolves a relative href to an absolute path."""
    from chunkhound.parsers.mappings.html import HtmlMapping
    html = HtmlMapping()
    (tmp_path / "style.css").write_text("body{}")
    resolved = html.resolve_import_paths("style.css", tmp_path, tmp_path / "index.html")
    assert len(resolved) == 1
    assert resolved[0] == tmp_path / "style.css"


def test_resolve_import_paths_html_query_string(tmp_path):
    """resolve_import_paths strips cache-busting query strings before resolving."""
    from chunkhound.parsers.mappings.html import HtmlMapping
    html = HtmlMapping()
    (tmp_path / "style.css").write_text("body{}")
    # href with cache-busting suffix — should still resolve to the bare file
    link_tag = '<link rel="stylesheet" href="style.css?v=1.2.3">'
    resolved = html.resolve_import_paths(link_tag, tmp_path, tmp_path / "index.html")
    assert len(resolved) == 1, f"Expected 1 resolved path, got {resolved}"
    assert resolved[0] == tmp_path / "style.css"


def test_resolve_import_paths_html_fragment(tmp_path):
    """resolve_import_paths strips URL fragments (#...) before resolving."""
    from chunkhound.parsers.mappings.html import HtmlMapping
    html = HtmlMapping()
    (tmp_path / "style.css").write_text("body{}")
    link_tag = '<link rel="stylesheet" href="style.css#section">'
    resolved = html.resolve_import_paths(link_tag, tmp_path, tmp_path / "index.html")
    assert len(resolved) == 1, f"Expected 1 resolved path, got {resolved}"
    assert resolved[0] == tmp_path / "style.css"


def test_stylesheet_import_name_strips_query_string(html_parser):
    """IMPORT chunk name for <link rel=stylesheet> strips cache-busting query strings."""
    code = '<link rel="stylesheet" href="main.css?v=2.0.0">'
    chunks = html_parser.parse_content(code, "test.html", file_id=1)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) > 0, "No IMPORT chunk for link[rel=stylesheet]"
    assert all("?" not in c.symbol for c in import_chunks), (
        f"Query string leaked into symbol: {[c.symbol for c in import_chunks]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

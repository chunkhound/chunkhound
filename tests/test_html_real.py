"""Real-world HTML parser tests using Python docs page."""

import pytest
from pathlib import Path

from chunkhound.core.types.common import Language, ChunkType
from chunkhound.parsers.parser_factory import ParserFactory

REAL_HTML_DIR = Path(__file__).parent / "fixtures" / "real" / "html"


@pytest.fixture
def html_parser():
    return ParserFactory().create_parser(Language.HTML)


# --- python-docs.html tests ---

@pytest.fixture
def python_docs_chunks(html_parser):
    path = REAL_HTML_DIR / "python-docs.html"
    if not path.exists():
        pytest.skip("Download python-docs.html first: see plan Phase 1")
    return html_parser.parse_file(path, file_id=1)


def test_python_docs_has_doctype(python_docs_chunks):
    namespace_chunks = [c for c in python_docs_chunks if c.chunk_type == ChunkType.NAMESPACE]
    assert len(namespace_chunks) > 0, "Expected DOCTYPE as NAMESPACE chunk"
    assert any("doctype" in (c.symbol or "").lower() for c in namespace_chunks), (
        f"Expected 'doctype' symbol; found: {[c.symbol for c in namespace_chunks]}"
    )


def test_python_docs_has_nav_block(python_docs_chunks):
    block_chunks = [c for c in python_docs_chunks if c.chunk_type == ChunkType.BLOCK]
    symbols = [c.symbol or "" for c in block_chunks]
    assert any("nav" in s.lower() for s in symbols), (
        f"Expected a <nav> block; found symbols: {symbols[:20]}"
    )


def test_python_docs_has_stylesheet_imports(python_docs_chunks):
    import_chunks = [c for c in python_docs_chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) > 0, (
        "Expected <link rel='stylesheet'> as IMPORT chunks"
    )


def test_python_docs_has_script_blocks(python_docs_chunks):
    block_chunks = [c for c in python_docs_chunks if c.chunk_type == ChunkType.BLOCK]
    script_chunks = [c for c in block_chunks if "script" in (c.symbol or "").lower()]
    assert len(script_chunks) > 0, "Python docs includes <script> tags"


def test_python_docs_no_div_in_blocks(python_docs_chunks):
    block_chunks = [c for c in python_docs_chunks if c.chunk_type == ChunkType.BLOCK]
    symbols = [c.symbol or "" for c in block_chunks]
    div_chunks = [s for s in symbols if s.startswith("div")]
    assert len(div_chunks) == 0, (
        f"Divs should NOT be extracted as BLOCK; found: {div_chunks}"
    )


def test_python_docs_has_semantic_elements(python_docs_chunks):
    block_chunks = [c for c in python_docs_chunks if c.chunk_type == ChunkType.BLOCK]
    symbols = [c.symbol or "" for c in block_chunks]
    semantic_found = [
        s for s in symbols
        if any(tag in s.lower() for tag in ("section", "article", "main", "header", "footer", "nav", "aside"))
    ]
    assert len(semantic_found) > 0, (
        f"Expected at least one semantic element BLOCK; found: {symbols[:20]}"
    )

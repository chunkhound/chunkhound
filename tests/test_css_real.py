"""Real-world CSS parser tests using normalize.css and Bootstrap 5."""

import pytest
from pathlib import Path

from chunkhound.core.types.common import Language, ChunkType
from chunkhound.parsers.parser_factory import ParserFactory

REAL_CSS_DIR = Path(__file__).parent / "fixtures" / "real" / "css"


@pytest.fixture
def css_parser():
    return ParserFactory().create_parser(Language.CSS)


# --- normalize.css tests ---

@pytest.fixture
def normalize_chunks(css_parser):
    path = REAL_CSS_DIR / "normalize.css"
    if not path.exists():
        pytest.skip("Download normalize.css first: see plan Phase 1")
    return css_parser.parse_file(path, file_id=1)


def test_normalize_extracts_html_rule(normalize_chunks):
    symbols = {c.symbol for c in normalize_chunks}
    assert "html" in symbols, f"Expected 'html' selector, got: {symbols}"


def test_normalize_extracts_body_rule(normalize_chunks):
    symbols = {c.symbol for c in normalize_chunks}
    assert "body" in symbols, f"Expected 'body' selector, got: {symbols}"


def test_normalize_extracts_h1_rule(normalize_chunks):
    symbols = {c.symbol for c in normalize_chunks}
    assert "h1" in symbols, f"Expected 'h1' selector, got: {symbols}"


def test_normalize_has_comment_with_license(normalize_chunks):
    comment_chunks = [c for c in normalize_chunks if c.chunk_type == ChunkType.COMMENT]
    assert len(comment_chunks) > 0, "Expected at least one COMMENT chunk"
    combined = " ".join(c.code for c in comment_chunks)
    assert "normalize" in combined.lower() or "MIT" in combined, (
        f"Expected license comment; found: {combined[:200]}"
    )


def test_normalize_no_media_queries(normalize_chunks):
    # normalize.css has no @media — verify parser doesn't create spurious ones
    block_chunks = [c for c in normalize_chunks if c.chunk_type == ChunkType.BLOCK]
    media_chunks = [c for c in block_chunks if "@media" in (c.symbol or "")]
    assert len(media_chunks) == 0, (
        f"normalize.css has no @media but parser found: {[c.symbol for c in media_chunks]}"
    )


def test_normalize_extracts_multiple_rules(normalize_chunks):
    block_chunks = [c for c in normalize_chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) >= 5, (
        f"normalize.css has many rules, got only {len(block_chunks)} BLOCK chunks"
    )


# --- bootstrap.css tests ---

@pytest.fixture
def bootstrap_chunks(css_parser):
    path = REAL_CSS_DIR / "bootstrap.css"
    if not path.exists():
        pytest.skip("Download bootstrap.css first: see plan Phase 1")
    return css_parser.parse_file(path, file_id=1)


def test_bootstrap_extracts_root_block(bootstrap_chunks):
    # Bootstrap uses ':root,\n[data-bs-theme=light]' as selector (not plain ':root')
    # so _is_root_vars doesn't match; the block is emitted as BLOCK with the selector as symbol
    block_chunks = [c for c in bootstrap_chunks if c.chunk_type == ChunkType.BLOCK]
    root_chunks = [c for c in block_chunks if ":root" in (c.symbol or "")]
    assert len(root_chunks) > 0, (
        f"No ':root' rule found; symbols sample: {[c.symbol for c in block_chunks[:30]]}"
    )


def test_bootstrap_has_media_queries(bootstrap_chunks):
    block_chunks = [c for c in bootstrap_chunks if c.chunk_type == ChunkType.BLOCK]
    media_chunks = [c for c in block_chunks if "@media" in (c.symbol or "")]
    assert len(media_chunks) > 0, "Expected @media blocks from Bootstrap responsive CSS"


def test_bootstrap_has_keyframes(bootstrap_chunks):
    block_chunks = [c for c in bootstrap_chunks if c.chunk_type == ChunkType.BLOCK]
    keyframe_chunks = [c for c in block_chunks if "@keyframes" in (c.symbol or "")]
    assert len(keyframe_chunks) > 0, "Bootstrap has spinner animations via @keyframes"


def test_bootstrap_extracts_many_rules(bootstrap_chunks):
    block_chunks = [c for c in bootstrap_chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 50, (
        f"Bootstrap has hundreds of rules, got {len(block_chunks)}"
    )

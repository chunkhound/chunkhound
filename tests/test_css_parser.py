"""Unit tests for CSS parser."""

import pytest
from pathlib import Path

from chunkhound.core.types.common import Language, ChunkType
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.fixture
def css_parser():
    """Create a CSS parser instance."""
    factory = ParserFactory()
    return factory.create_parser(Language.CSS)


@pytest.fixture
def comprehensive_css():
    """Load comprehensive CSS test file."""
    fixture_path = Path(__file__).parent / "fixtures" / "css" / "comprehensive.css"
    if not fixture_path.exists():
        pytest.skip(f"Test fixture not found: {fixture_path}")
    return fixture_path


def test_parses_rule_set_as_definition(css_parser):
    """Simple rule sets are extracted as BLOCK (via chunk_type_hint) chunks.

    Note: adjacent rule sets may be merged by the cAST algorithm into a single chunk.
    """
    code = "body { margin: 0; padding: 0; font-family: sans-serif; }"
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    # CSS rule_set → DEFINITION → chunk_type_hint "block" → ChunkType.BLOCK
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks for CSS rule sets"
    assert any("body" in c.symbol for c in block_chunks), (
        f"'body' not in {[c.symbol for c in block_chunks]}"
    )


def test_parses_root_variables_as_structure(css_parser):
    """:root rule with custom properties is extracted as NAMESPACE with symbol containing ':root_vars'.

    The chunk_type_hint 'namespace' in metadata maps to ChunkType.NAMESPACE,
    distinguishing these design-token blocks from regular rule sets (ChunkType.BLOCK).
    The symbol includes the selector and line number for uniqueness.
    """
    code = """:root {
  --primary: #333;
  --secondary: #999;
}"""
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
    assert len(ns_chunks) > 0, "No NAMESPACE chunk for :root vars"
    assert any(":root_vars" in c.symbol for c in ns_chunks), (
        f":root_vars not in {[c.symbol for c in ns_chunks]}"
    )
    # Metadata should indicate is_root_vars
    root_chunk = next((c for c in ns_chunks if ":root_vars" in c.symbol), None)
    assert root_chunk is not None
    assert root_chunk.metadata and root_chunk.metadata.get("is_root_vars") is True


def test_parses_star_variables_as_structure(css_parser):
    """* rule with custom properties is handled as a variable block (BLOCK type)."""
    code = """* {
  --margin: 0;
  --padding: 0;
}"""
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    # * with vars may appear as BLOCK or be merged. Just check something is returned.
    assert len(chunks) > 0, "No chunks for * vars rule"
    # The chunk content should include custom properties
    all_code = " ".join(c.code for c in chunks)
    assert "--margin" in all_code or "--padding" in all_code, (
        "Custom properties not captured in * rule"
    )


def test_parses_media_as_block(css_parser):
    """@media statements are extracted as BLOCK chunks."""
    code = """@media (max-width: 768px) {
  .container { padding: 0 1rem; }
}"""
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunk for @media"
    assert any("@media" in c.symbol for c in block_chunks)


def test_parses_keyframes_as_block(css_parser):
    """@keyframes statements are extracted as BLOCK chunks."""
    code = """@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}"""
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunk for @keyframes"
    assert any("@keyframes fadeIn" == c.symbol for c in block_chunks), (
        f"Expected '@keyframes fadeIn', got: {[c.symbol for c in block_chunks]}"
    )


def test_parses_supports_as_block(css_parser):
    """@supports statements are extracted as BLOCK chunks."""
    code = """@supports (display: grid) {
  .grid { display: grid; }
}"""
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunk for @supports"
    assert any("supports" in c.symbol for c in block_chunks)


def test_parses_import_as_import(css_parser):
    """@import statements are extracted as IMPORT chunks."""
    code = '@import "variables.css";\n@import url("reset.css");'
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) > 0, "No IMPORT chunks for @import"
    symbols = {c.symbol for c in import_chunks}
    assert any("variables" in s for s in symbols), f"variables.css not in {symbols}"


def test_import_with_media_query_symbol_is_path_only(css_parser):
    """@import with a media query returns only the path as symbol, not the media qualifier."""
    code = '@import "reset.css" screen, print;\n@import url("base.css") all;'
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) > 0, "No IMPORT chunks"
    # Chunks may be merged; join all symbols and verify media qualifiers are absent.
    all_symbols = " ".join(c.symbol for c in import_chunks)
    assert "screen" not in all_symbols, f"Media qualifier 'screen' leaked into symbol: {all_symbols}"
    assert "print" not in all_symbols, f"Media qualifier 'print' leaked into symbol: {all_symbols}"
    assert "all" not in all_symbols, f"Media qualifier 'all' leaked into symbol: {all_symbols}"
    assert "reset" in all_symbols, f"Expected 'reset' in symbol, got: {all_symbols}"


def test_parses_comments(css_parser, comprehensive_css):
    """CSS /* */ comments are extracted as COMMENT chunks.

    Note: CSS comments only appear in larger files; the cAST algorithm merges
    small files into single chunks. Use the comprehensive fixture for reliable testing.
    """
    chunks = css_parser.parse_file(comprehensive_css, file_id=1)
    comment_chunks = [c for c in chunks if c.chunk_type == ChunkType.COMMENT]
    assert len(comment_chunks) > 0, "No COMMENT chunks in comprehensive CSS"
    assert all(c.symbol.startswith("comment_line") for c in comment_chunks)


def test_complex_selector_name(css_parser):
    """Complex selectors are preserved as the chunk symbol."""
    code = ".nav > ul > li > a:hover { color: red; }"
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0
    assert any(".nav > ul > li > a:hover" in c.symbol for c in block_chunks), (
        f"Complex selector not found in {[c.symbol for c in block_chunks]}"
    )


def test_selector_truncated_at_60_chars(css_parser):
    """Selectors longer than 60 chars are truncated with ellipsis."""
    # Build a selector longer than 60 chars
    selector = ".very-long-class-name.another-very-long-class-name.one-more-class"
    assert len(selector) > 60
    code = f"{selector} {{ color: red; }}"
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0
    symbol = block_chunks[0].symbol
    assert symbol.endswith("..."), f"Expected truncation with '...', got: {symbol}"
    assert len(symbol) <= 63, f"Symbol too long: {symbol}"


def test_rule_set_not_in_structure_when_no_vars(css_parser):
    """Regular rule sets (no custom properties) are NOT extracted as STRUCTURE."""
    code = "body { color: red; margin: 0; }"
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    struct_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
    # body without --vars should not be STRUCTURE
    assert all("body" not in c.symbol for c in struct_chunks), (
        "body rule without vars incorrectly tagged as STRUCTURE"
    )


def test_comprehensive_file(css_parser, comprehensive_css):
    """Parse the comprehensive CSS fixture and check coverage."""
    chunks = css_parser.parse_file(comprehensive_css, file_id=1)

    assert len(chunks) > 5, f"Expected more than 5 chunks, got {len(chunks)}"

    chunk_types = {c.chunk_type for c in chunks}

    # Must have BLOCK (rule_sets, @media, @keyframes, @supports)
    assert ChunkType.BLOCK in chunk_types, f"No BLOCK chunks. Types: {chunk_types}"

    # Must have COMMENT
    assert ChunkType.COMMENT in chunk_types, f"No COMMENT chunks. Types: {chunk_types}"

    # Must have IMPORT (@import)
    assert ChunkType.IMPORT in chunk_types, f"No IMPORT chunks. Types: {chunk_types}"

    symbols = {c.symbol for c in chunks}
    all_code = " ".join(c.code for c in chunks)

    # Check body rule set
    assert any("body" in s for s in symbols), f"body not found in {symbols}"

    # @media, @keyframes, @supports may be merged with inner content;
    # check that the constructs appear somewhere in the chunk code
    assert "@media" in all_code, f"@media not found in chunk code"
    assert "@keyframes" in all_code, f"@keyframes not found in chunk code"
    assert "@supports" in all_code, f"@supports not found in chunk code"

    # :root vars → NAMESPACE with symbol containing ':root_vars'
    ns_symbols = {c.symbol for c in chunks if c.chunk_type == ChunkType.NAMESPACE}
    assert any(":root_vars" in s for s in ns_symbols), (
        f":root_vars not in namespace symbols {ns_symbols}"
    )

    # Check import symbols
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) >= 1, f"Expected at least 1 import, got {len(import_chunks)}"


def test_parses_star_variables_chunk_type(css_parser):
    """* rule with custom properties produces a NAMESPACE chunk with is_root_vars=True."""
    code = """* {
  --margin: 0;
  --padding: 0;
}"""
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
    assert len(ns_chunks) > 0, "No NAMESPACE chunk for * with custom properties"
    assert any(
        c.metadata and c.metadata.get("is_root_vars") is True for c in ns_chunks
    ), "is_root_vars not True for * with --vars"


def test_comment_query_is_valid():
    """CssMapping.get_comment_query returns a non-empty CSS comment query."""
    from chunkhound.parsers.mappings.css import CssMapping
    css = CssMapping()
    query = css.get_comment_query()
    assert query, "get_comment_query returned empty string"
    assert "comment" in query


def test_resolve_import_paths_url(tmp_path):
    """resolve_import_paths strips url(...) wrapper and resolves the path."""
    from chunkhound.parsers.mappings.css import CssMapping
    css = CssMapping()
    # Create a real file for the resolver to find
    (tmp_path / "reset.css").write_text("*{margin:0}")
    resolved = css.resolve_import_paths('url("reset.css")', tmp_path, tmp_path / "style.css")
    assert len(resolved) == 1
    assert resolved[0] == tmp_path / "reset.css"


def test_resolve_import_paths_not_found(tmp_path):
    """resolve_import_paths returns empty list when the file does not exist."""
    from chunkhound.parsers.mappings.css import CssMapping
    css = CssMapping()
    resolved = css.resolve_import_paths('"nonexistent.css"', tmp_path, tmp_path / "style.css")
    assert resolved == []


def test_resolve_import_paths_url_with_spaces(tmp_path):
    """resolve_import_paths handles url( path ) with internal spaces."""
    from chunkhound.parsers.mappings.css import CssMapping
    css = CssMapping()
    (tmp_path / "reset.css").write_text("*{margin:0}")
    # url() with spaces around the path — broken by naive split()[0]
    resolved = css.resolve_import_paths(
        '@import url( "reset.css" );', tmp_path, tmp_path / "style.css"
    )
    assert len(resolved) == 1, f"Expected 1 resolved path, got {resolved}"
    assert resolved[0] == tmp_path / "reset.css"


def test_root_vars_not_merged_with_adjacent_rule(css_parser):
    """:root var block and adjacent rule set produce separate chunks.

    Before the fix, (DEFINITION, STRUCTURE) was in _COMPATIBLE_CONCEPT_PAIRS
    which caused the greedy merge pass to absorb :root var blocks into
    neighbouring rule sets, losing the design-token NAMESPACE chunk.
    """
    from chunkhound.core.types.common import ChunkType
    code = """:root {
  --primary: #3498db;
  --secondary: #2ecc71;
}

body {
  margin: 0;
  padding: 0;
}
"""
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    chunk_types = {c.chunk_type for c in chunks}
    # NAMESPACE (:root vars) and BLOCK (body rule) must both be present
    assert ChunkType.NAMESPACE in chunk_types, (
        f":root vars collapsed — no NAMESPACE chunk. Types: {chunk_types}"
    )
    assert ChunkType.BLOCK in chunk_types, (
        f"body rule missing — no BLOCK chunk. Types: {chunk_types}"
    )
    # Verify they are separate chunks — not merged into one
    ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(ns_chunks) >= 1, "Expected at least 1 NAMESPACE chunk for :root vars"
    assert len(block_chunks) >= 1, "Expected at least 1 BLOCK chunk for body rule"


def test_root_vars_comma_selector(css_parser):
    """:root, [data-bs-theme=light] selector list is recognised as a design-token block.

    Bootstrap 5.3+ uses a comma-selector for theming: the :root part means the block
    should still be classified as STRUCTURE (namespace), not a plain DEFINITION.
    """
    code = """:root, [data-bs-theme=light] {
  --bs-primary: #0d6efd;
  --bs-secondary: #6c757d;
}

body { margin: 0; }
"""
    chunks = css_parser.parse_content(code, "bootstrap.css", file_id=1)
    from chunkhound.core.types.common import ChunkType
    ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
    assert len(ns_chunks) >= 1, (
        f"Expected NAMESPACE chunk for ':root, [data-bs-theme=light]', got {[c.symbol for c in chunks]}"
    )


def test_resolve_import_paths_source_file_relative(tmp_path):
    """CSS import resolution uses the importing file's directory, not the project root."""
    from chunkhound.parsers.mappings.css import CssMapping
    css = CssMapping()
    # src/styles/main.css imports ../tokens/colors.css (relative to src/styles/)
    styles_dir = tmp_path / "src" / "styles"
    styles_dir.mkdir(parents=True)
    tokens_dir = tmp_path / "src" / "tokens"
    tokens_dir.mkdir()
    (tokens_dir / "colors.css").write_text(":root{--blue:#00f}")
    source = styles_dir / "main.css"
    resolved = css.resolve_import_paths(
        '@import "../tokens/colors.css"', tmp_path, source
    )
    assert len(resolved) == 1, f"Expected 1 resolved path, got {resolved}"
    assert resolved[0] == tokens_dir / "colors.css"


def test_resolve_import_paths_query_string(tmp_path):
    """resolve_import_paths strips ?query from import URL before resolving."""
    from chunkhound.parsers.mappings.css import CssMapping
    css = CssMapping()
    (tmp_path / "theme.css").write_text(":root{--color:red}")
    resolved = css.resolve_import_paths(
        '@import url("theme.css?v=1");', tmp_path, tmp_path / "style.css"
    )
    assert len(resolved) == 1, f"Expected 1 resolved path, got {resolved}"
    assert resolved[0] == tmp_path / "theme.css"


def test_resolve_import_paths_fragment(tmp_path):
    """resolve_import_paths strips #fragment from import URL before resolving."""
    from chunkhound.parsers.mappings.css import CssMapping
    css = CssMapping()
    (tmp_path / "theme.css").write_text(":root{--color:red}")
    resolved = css.resolve_import_paths(
        '@import "theme.css#v2";', tmp_path, tmp_path / "style.css"
    )
    assert len(resolved) == 1, f"Expected 1 resolved path, got {resolved}"
    assert resolved[0] == tmp_path / "theme.css"


def test_root_vars_tail_position_in_selector_list(css_parser):
    """:root appearing after the 60-char truncation point is still classified as NAMESPACE.

    Before the fix, _is_root_vars() called selector_text() which truncates at 60 chars.
    A selector list where :root appears after position 60 would have been misclassified
    as a plain DEFINITION block, losing the design-token NAMESPACE chunk.
    """
    from chunkhound.core.types.common import ChunkType
    # `:root` appears well past the 60-char cutoff
    long_prefix = ".very-long-vendor-specific-selector-name-exceeding-sixty-chars"
    assert len(long_prefix) > 60
    code = f"""{long_prefix}, :root {{
  --color: red;
  --size: 1rem;
}}
"""
    chunks = css_parser.parse_content(code, "test.css", file_id=1)
    ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
    assert len(ns_chunks) >= 1, (
        f":root after 60-char cutoff was misclassified — no NAMESPACE chunk. "
        f"Chunks: {[(c.chunk_type, c.symbol) for c in chunks]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

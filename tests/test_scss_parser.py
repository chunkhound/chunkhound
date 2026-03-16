"""Unit tests for SCSS parser."""

import pytest
from pathlib import Path

from chunkhound.core.types.common import Language, ChunkType
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.fixture
def scss_parser():
    """Create an SCSS parser instance."""
    factory = ParserFactory()
    return factory.create_parser(Language.SCSS)


@pytest.fixture
def comprehensive_scss():
    """Load comprehensive SCSS test file."""
    fixture_path = Path(__file__).parent / "fixtures" / "scss" / "comprehensive.scss"
    if not fixture_path.exists():
        pytest.skip(f"Test fixture not found: {fixture_path}")
    return fixture_path


def test_parses_mixin_as_definition(scss_parser):
    """@mixin definitions are extracted as FUNCTION chunks."""
    code = """@mixin flex-center {
  display: flex;
  align-items: center;
  justify-content: center;
}"""
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    # mixin_statement → DEFINITION → chunk_type_hint "function" → ChunkType.FUNCTION
    func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
    assert len(func_chunks) > 0, "No FUNCTION chunks for @mixin"
    assert any("@mixin flex-center" in c.symbol for c in func_chunks), (
        f"@mixin flex-center not found in {[c.symbol for c in func_chunks]}"
    )


def test_parses_function_as_definition(scss_parser):
    """@function definitions are extracted as FUNCTION chunks."""
    code = """@function rem($px) {
  @return $px / 16px * 1rem;
}"""
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
    assert len(func_chunks) > 0, "No FUNCTION chunks for @function"
    assert any("@function rem" in c.symbol for c in func_chunks), (
        f"@function rem not found in {[c.symbol for c in func_chunks]}"
    )


def test_parses_variable_as_structure(scss_parser):
    """$variable declarations are extracted as STRUCTURE (NAMESPACE) chunks."""
    code = "$primary: #3498db;\n$secondary: #2ecc71;\n$font-size: 16px;"
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    struct_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
    assert len(struct_chunks) > 0, "No STRUCTURE chunks for $variables"
    symbols = {c.symbol for c in struct_chunks}
    assert any("$primary" in s for s in symbols), f"$primary not in {symbols}"
    assert any("$secondary" in s for s in symbols), f"$secondary not in {symbols}"


def test_parses_rule_set_as_definition(scss_parser):
    """Plain rule sets are extracted as BLOCK (via chunk_type_hint) chunks.

    Note: adjacent rule sets may be merged by the cAST algorithm into one chunk.
    """
    code = "body { margin: 0; padding: 0; font-family: sans-serif; }"
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    # rule_set → DEFINITION → chunk_type_hint "block" → ChunkType.BLOCK
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks for SCSS rule sets"
    assert any("body" in c.symbol for c in block_chunks), (
        f"body not in {[c.symbol for c in block_chunks]}"
    )


def test_parses_include_as_block(scss_parser):
    """@include statements are extracted as BLOCK chunks.

    When @include appears inside a rule set, it becomes part of that rule's chunk.
    Standalone @include produces a BLOCK with symbol '@include_line...'
    """
    # Standalone @include produces dedicated BLOCK chunk
    code = "@include flex-center;\n@include button-variant(red);"
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks for @include"
    assert any("include" in c.symbol for c in block_chunks), (
        f"No @include block in {[c.symbol for c in block_chunks]}"
    )


def test_parses_each_as_block(scss_parser):
    """@each loops are extracted as BLOCK chunks.

    Note: The tree-sitter SCSS grammar may produce unusual symbols for @each
    (e.g. iteration values). Check that the @each content appears in a BLOCK chunk.
    """
    code = """@each $color in red, green, blue {
  .text-#{$color} { color: $color; }
}"""
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks for @each"
    # The @each content should appear somewhere in the block chunks' code
    all_code = " ".join(c.code for c in block_chunks)
    assert "@each" in all_code or ".text-" in all_code, (
        f"@each content not found in block code: {all_code[:200]}"
    )


def test_parses_for_as_block(scss_parser):
    """@for loops are extracted as BLOCK chunks."""
    code = """@for $i from 1 through 3 {
  .col-#{$i} { width: 33.33% * $i; }
}"""
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks for @for"
    assert any("for" in c.symbol for c in block_chunks), (
        f"No @for block in {[c.symbol for c in block_chunks]}"
    )


def test_parses_if_as_block(scss_parser):
    """@if statements are extracted as BLOCK chunks."""
    code = """@if $size > 14px {
  .large { line-height: 1.6; }
} @else {
  .large { line-height: 1.4; }
}"""
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks for @if"
    assert any("if" in c.symbol for c in block_chunks), (
        f"No @if block in {[c.symbol for c in block_chunks]}"
    )


def test_parses_media_as_block(scss_parser):
    """@media statements are extracted as BLOCK chunks."""
    code = """@media (max-width: 768px) {
  .container { padding: 0 1rem; }
}"""
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunks for @media"
    assert any("@media" in c.symbol for c in block_chunks)


def test_parses_import_as_import(scss_parser):
    """@import statements are extracted as IMPORT chunks."""
    code = '@import "variables";\n@import "mixins";'
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) > 0, "No IMPORT chunks for @import"
    symbols = {c.symbol for c in import_chunks}
    assert any("variables" in s for s in symbols), f"variables not in {symbols}"


def test_parses_use_as_import(scss_parser):
    """@use statements are extracted as IMPORT chunks."""
    code = '@use "sass:math";\n@use "sass:color";'
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) > 0, "No IMPORT chunks for @use"
    symbols = {c.symbol for c in import_chunks}
    assert any("math" in s for s in symbols), f"sass:math not in {symbols}"


def test_parses_forward_as_import(scss_parser):
    """@forward statements are extracted as IMPORT chunks."""
    code = '@forward "mixins";\n@forward "functions";'
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) > 0, "No IMPORT chunks for @forward"
    symbols = {c.symbol for c in import_chunks}
    assert any("mixins" in s or "functions" in s for s in symbols), (
        f"forward targets not in {symbols}"
    )


def test_parses_comments(scss_parser):
    """SCSS /* */ block comments are extracted as COMMENT chunks.

    Note: Single-line // comments are not captured by the tree-sitter SCSS grammar's
    comment query. Only /* */ block comments produce COMMENT chunks.
    """
    code = "/* Block comment */\n$var: red;\n/* Another block comment */\n.btn { color: $var; }"
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    comment_chunks = [c for c in chunks if c.chunk_type == ChunkType.COMMENT]
    assert len(comment_chunks) > 0, "No COMMENT chunks found for /* */ comments"
    assert all(c.symbol.startswith("comment_line") for c in comment_chunks)


def test_mixin_metadata(scss_parser):
    """@mixin chunks have name in metadata."""
    code = """@mixin my-mixin($arg) {
  color: $arg;
}"""
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
    assert len(func_chunks) > 0
    meta = func_chunks[0].metadata
    assert meta is not None
    assert meta.get("name") == "my-mixin", f"Expected name='my-mixin', got {meta}"


def test_comprehensive_file(scss_parser, comprehensive_scss):
    """Parse the comprehensive SCSS fixture and check coverage."""
    chunks = scss_parser.parse_file(comprehensive_scss, file_id=1)

    assert len(chunks) > 5, f"Expected more than 5 chunks, got {len(chunks)}"

    chunk_types = {c.chunk_type for c in chunks}

    # Must have FUNCTION (at least @mixin)
    assert ChunkType.FUNCTION in chunk_types, f"No FUNCTION chunks. Types: {chunk_types}"

    # Must have BLOCK (rule_sets, @media, @keyframes, @include, etc.)
    assert ChunkType.BLOCK in chunk_types, f"No BLOCK chunks. Types: {chunk_types}"

    # Must have NAMESPACE (STRUCTURE: $variables)
    assert ChunkType.NAMESPACE in chunk_types, f"No NAMESPACE/STRUCTURE. Types: {chunk_types}"

    # Must have IMPORT (@import, @use, @forward)
    assert ChunkType.IMPORT in chunk_types, f"No IMPORT chunks. Types: {chunk_types}"

    symbols = {c.symbol for c in chunks}

    # At least one @mixin appears
    assert any("@mixin" in s for s in symbols), f"No @mixin in {symbols}"

    # $variables appear somewhere
    assert any("$" in s for s in symbols), f"No $variable in {symbols}"

    # @media appears somewhere (may be merged with other blocks)
    all_code = " ".join(c.code for c in chunks)
    assert "@media" in all_code, "No @media in chunk code"
    assert "@keyframes" in all_code, "No @keyframes in chunk code"

    # Check imports are captured (may be merged into one chunk)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) >= 1, f"Expected at least 1 import, got {len(import_chunks)}"
    import_code = " ".join(c.symbol for c in import_chunks)
    assert "variables" in import_code or "math" in import_code or "sass" in import_code, (
        f"Expected @import/@use targets in {import_code}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

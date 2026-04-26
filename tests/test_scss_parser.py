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


def test_use_with_alias_symbol_is_path_only(scss_parser):
    """@use with 'as' alias returns only the module path as symbol, not the alias."""
    code = '@use "colors" as c;\n@use "typography" as t;\n@forward "mixins" as m-*;'
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
    assert len(import_chunks) > 0, "No IMPORT chunks"
    symbols = {c.symbol for c in import_chunks}
    assert any("colors" in s for s in symbols), f"colors not in {symbols}"
    assert any("typography" in s for s in symbols), f"typography not in {symbols}"
    assert not any(" as " in s or 'as c' in s or '"' in s for s in symbols), (
        f"Alias or quotes leaked into symbol: {symbols}"
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


def test_interpolated_custom_props_not_empty(scss_parser):
    """Bootstrap-style --#{$prefix}name custom props don't produce zero chunks.

    The tree-sitter SCSS grammar cannot parse ``--#{$var}name`` natively.
    ScssMapping.preprocess_for_ast() replaces #{...} with same-length
    placeholders so the grammar sees a valid token and extracts rule sets.
    The stored chunk code must still contain the original #{$prefix} syntax.
    """
    code = """.btn {
  --#{$prefix}btn-color: #{$btn-color};
  --#{$prefix}btn-bg: transparent;
  display: inline-block;
  padding: var(--#{$prefix}btn-padding-y) var(--#{$prefix}btn-padding-x);
}
.btn-lg {
  padding: var(--#{$prefix}btn-padding-y-lg) var(--#{$prefix}btn-padding-x-lg);
}
"""
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    # Both rule sets are captured (cAST may merge adjacent small rules into one chunk)
    assert len(block_chunks) >= 1, (
        f"Expected >=1 BLOCK chunk for interpolated custom-prop file, got {len(block_chunks)}"
    )
    all_code = " ".join(c.code for c in block_chunks)
    # Both selectors appear in the extracted code
    assert ".btn" in all_code, ".btn selector not captured"
    assert ".btn-lg" in all_code, ".btn-lg selector not captured"
    # Original interpolation syntax must be preserved (not replaced with x's)
    assert "#{$prefix}" in all_code, "#{$prefix} not preserved in stored chunk code"
    assert "xxxxxxxx" not in all_code, "placeholder leaked into stored chunk code"


def test_parses_while_as_block(scss_parser):
    """@while loops are extracted as BLOCK chunks."""
    code = """$i: 1;
@while $i <= 3 {
  .item-#{$i} { width: 10px * $i; }
  $i: $i + 1;
}"""
    chunks = scss_parser.parse_content(code, "test.scss", file_id=1)
    block_chunks = [c for c in chunks if c.chunk_type == ChunkType.BLOCK]
    assert len(block_chunks) > 0, "No BLOCK chunk for @while"
    assert any("while" in c.symbol for c in block_chunks), (
        f"@while not found in {[c.symbol for c in block_chunks]}"
    )


def test_resolve_import_paths_partial(tmp_path):
    """resolve_import_paths resolves SCSS underscore-prefixed partials."""
    from chunkhound.parsers.mappings.scss import ScssMapping
    scss = ScssMapping()
    # Create the partial file (underscore prefix convention)
    (tmp_path / "_colors.scss").write_text("$primary: red;")
    # Import without the underscore or extension
    resolved = scss.resolve_import_paths("colors", tmp_path, tmp_path / "main.scss")
    assert len(resolved) == 1
    assert resolved[0] == tmp_path / "_colors.scss"


def test_resolve_import_paths_direct(tmp_path):
    """resolve_import_paths resolves a direct SCSS path first."""
    from chunkhound.parsers.mappings.scss import ScssMapping
    scss = ScssMapping()
    (tmp_path / "variables.scss").write_text("$size: 16px;")
    resolved = scss.resolve_import_paths("variables.scss", tmp_path, tmp_path / "main.scss")
    assert len(resolved) == 1
    assert resolved[0] == tmp_path / "variables.scss"


def test_resolve_import_paths_builtin_sass_module(tmp_path):
    """resolve_import_paths returns [] for built-in Sass modules (sass:xxx)."""
    from chunkhound.parsers.mappings.scss import ScssMapping
    scss = ScssMapping()
    # Built-in modules have no filesystem path — should return empty, not error
    assert scss.resolve_import_paths('@use "sass:math"', tmp_path, tmp_path / "main.scss") == []
    assert scss.resolve_import_paths('@use "sass:color"', tmp_path, tmp_path / "main.scss") == []
    assert scss.resolve_import_paths('@use "sass:list"', tmp_path, tmp_path / "main.scss") == []


def test_preprocess_scss_deep_nesting():
    """Interpolation preprocessor handles arbitrarily nested braces."""
    from chunkhound.parsers.mappings.scss import _preprocess_scss_interpolations
    # Three levels of nesting
    source = ".a { --#{fn(#{inner($x)})}name: red; }"
    result = _preprocess_scss_interpolations(source)
    # All interpolation characters replaced with 'x' (except newlines)
    # Span is #{fn(#{inner($x)})} — starts at #{, ends after the outer }
    # Verify the surrounding non-interpolation text is unchanged
    assert result.startswith(".a { --")
    assert result.endswith("name: red; }")
    # No literal #{...} should remain in the output
    assert "#{" not in result, f"Unprocessed interpolation in: {result}"


def test_preprocess_scss_interpolation_with_quoted_brace():
    """Preprocessor does not prematurely close on } inside a string argument."""
    from chunkhound.parsers.mappings.scss import _preprocess_scss_interpolations
    # The } inside "a}" is part of a string, not the interpolation close
    source = '.a { --#{if($c, "a}", "b")}name: red; }'
    result = _preprocess_scss_interpolations(source)
    assert result.startswith(".a { --")
    assert result.endswith("name: red; }")
    assert "#{" not in result, f"Unprocessed interpolation in: {result}"


def test_preprocess_scss_preserves_newlines():
    """Preprocessor keeps newlines intact to preserve AST line numbers."""
    from chunkhound.parsers.mappings.scss import _preprocess_scss_interpolations
    source = ".a {\n  --#{$var\n}name: red;\n}"
    result = _preprocess_scss_interpolations(source)
    assert result.count("\n") == source.count("\n"), "Newline count changed"


def test_preprocess_scss_no_interpolation_unchanged():
    """Content without any #{...} passes through unchanged."""
    from chunkhound.parsers.mappings.scss import _preprocess_scss_interpolations
    source = ".a { color: red; }"
    assert _preprocess_scss_interpolations(source) == source


def test_resolve_import_paths_comma_separated(tmp_path):
    """@import "a", "b"; resolves both paths."""
    from chunkhound.parsers.mappings.scss import ScssMapping
    scss = ScssMapping()
    (tmp_path / "_variables.scss").write_text("$size: 16px;")
    (tmp_path / "_mixins.scss").write_text("@mixin flex{display:flex}")
    resolved = scss.resolve_import_paths(
        '@import "variables", "mixins"', tmp_path, tmp_path / "main.scss"
    )
    assert len(resolved) == 2, f"Expected 2 resolved paths for comma import, got {resolved}"
    names = {p.name for p in resolved}
    assert "_variables.scss" in names
    assert "_mixins.scss" in names


def test_resolve_import_paths_source_file_relative(tmp_path):
    """SCSS import resolution uses the importing file's directory, not the project root."""
    from chunkhound.parsers.mappings.scss import ScssMapping
    scss = ScssMapping()
    # src/components/button.scss imports ../../tokens/colors (relative to src/components/)
    comp_dir = tmp_path / "src" / "components"
    comp_dir.mkdir(parents=True)
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    (tokens_dir / "_colors.scss").write_text("$blue: #00f;")
    source = comp_dir / "button.scss"
    resolved = scss.resolve_import_paths(
        '@import "../../tokens/colors"', tmp_path, source
    )
    assert len(resolved) == 1, f"Expected 1 resolved path, got {resolved}"
    assert resolved[0] == tokens_dir / "_colors.scss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

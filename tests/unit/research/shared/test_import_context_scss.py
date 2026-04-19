"""Integration tests for ImportContextService with SCSS parser.

Verifies that get_file_imports() routes through preprocess_for_ast() so that
#{...} interpolations are neutralised before tree-sitter parsing.  Without the
preprocessing hook, the SCSS grammar fails on interpolations and returns no
@import/@use/@forward edges — the regression that ofriw's review caught.
"""

import pytest

from chunkhound.parsers._grammar_availability import SCSS_AVAILABLE
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.services.research.shared.import_context import ImportContextService

pytestmark = pytest.mark.skipif(
    not SCSS_AVAILABLE, reason="tree-sitter-scss not installed"
)


# SCSS content whose @import lines appear *before* the interpolation block.
# A parser that skips preprocess_for_ast() will trip on `#{$name}` and may
# return an incomplete or empty parse tree, losing the import edges.
SCSS_WITH_INTERPOLATION = """\
@import "variables";
@use "sass:math";
@use "sass:color" as color-utils;
@forward "mixins";

$colors: (primary: #3498db, secondary: #2ecc71);

@each $name, $value in $colors {
  .text-#{$name} {
    color: $value;
  }
  .bg-#{$name} {
    background-color: $value;
  }
}
"""

# Minimal SCSS with no interpolation — baseline to confirm extraction works
# without any preprocessing at all.
SCSS_PLAIN = """\
@import "base";
@use "sass:list";

.container {
  display: flex;
}
"""


@pytest.fixture
def service() -> ImportContextService:
    return ImportContextService(ParserFactory())


def test_imports_extracted_from_scss_with_interpolation(service):
    """@import/@use/@forward are found even when #{...} interpolations are present."""
    imports = service.get_file_imports("styles/theme.scss", SCSS_WITH_INTERPOLATION)

    import_text = "\n".join(imports)
    assert "@import" in import_text or "@use" in import_text or "@forward" in import_text, (
        "No imports found — preprocess_for_ast() was likely bypassed"
    )
    assert any('"variables"' in line or "variables" in line for line in imports)
    assert any("sass:math" in line for line in imports)
    assert any("sass:color" in line for line in imports)
    assert any("mixins" in line for line in imports)


def test_imports_extracted_from_plain_scss(service):
    """Baseline: imports are extracted from SCSS without any interpolation."""
    imports = service.get_file_imports("styles/base.scss", SCSS_PLAIN)

    assert any('"base"' in line or "base" in line for line in imports)
    assert any("sass:list" in line for line in imports)


def test_results_are_cached(service):
    """Second call for the same path returns the cached list without re-parsing."""
    first = service.get_file_imports("styles/theme.scss", SCSS_WITH_INTERPOLATION)
    second = service.get_file_imports("styles/theme.scss", SCSS_WITH_INTERPOLATION)
    assert first is second  # same list object — came from cache


def test_clear_cache_removes_scss_entry(service):
    """clear_cache() removes the cached entry so the next call re-parses."""
    service.get_file_imports("styles/theme.scss", SCSS_WITH_INTERPOLATION)
    service.clear_cache()
    assert "styles/theme.scss" not in service._import_cache


def test_interpolation_content_not_present_in_output(service):
    """Chunk content should reflect the original source, not preprocessed placeholders."""
    imports = service.get_file_imports("styles/theme.scss", SCSS_WITH_INTERPOLATION)
    for line in imports:
        # Preprocessing replaces #{...} with same-length placeholders like __SCSS_0__.
        # Import chunks should not contain those placeholders.
        assert "__SCSS_" not in line, f"Preprocessor placeholder leaked into output: {line!r}"

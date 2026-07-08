"""Tests for _fnmatch_to_gitignore — the fnmatch→gitignore pattern converter.

These run without the Rust extension; they exercise pure-Python logic.
"""

import pytest

from chunkhound.utils.file_patterns import _fnmatch_to_gitignore


@pytest.mark.unit
@pytest.mark.parametrize(
    "pattern, expected",
    [
        # Directory subtree patterns: keep '/**' so gi.matched(file, is_dir=false)
        # matches files inside the directory, not just the directory node itself.
        ("**/node_modules/**", "**/node_modules/**"),
        ("**/__pycache__/**", "**/__pycache__/**"),
        ("**/.git/**", "**/.git/**"),
        ("**/.venv/**", "**/.venv/**"),
        ("**/dist/**", "**/dist/**"),
        ("**/build/**", "**/build/**"),
        ("**/target/**", "**/target/**"),
        ("**/.yarn/cache/**", "**/.yarn/cache/**"),
        ("**/.yarn/unplugged/**", "**/.yarn/unplugged/**"),
        ("**/.vuepress/dist/**", "**/.vuepress/dist/**"),
        ("**/.mypy_cache/**", "**/.mypy_cache/**"),
        ("**/.pytest_cache/**", "**/.pytest_cache/**"),
        # Non-directory patterns: keep '**/' for multi-segment paths (root-anchor risk),
        # drop '**/' for simple names/extensions (gitignore bare names match anywhere).
        ("**/.yarn/build-state.yml", "**/.yarn/build-state.yml"),
        ("**/.yarn/install-state.gz", "**/.yarn/install-state.gz"),
        ("**/*.swp", "*.swp"),
        ("**/*.pyc", "*.pyc"),
        # Patterns without a leading '**/' pass through unchanged.
        ("tmp/**", "tmp/**"),
        (".chunkhound.json", ".chunkhound.json"),
    ],
)
def test_fnmatch_to_gitignore(pattern: str, expected: str) -> None:
    assert _fnmatch_to_gitignore(pattern) == expected


@pytest.mark.unit
def test_multi_segment_patterns_not_root_anchored() -> None:
    """
    Multi-segment patterns must keep '**/' so
    gitignore doesn't anchor them to root.
    """
    result = _fnmatch_to_gitignore("**/.yarn/cache/**")
    assert result.startswith("**/"), (
        f"Expected '**/' prefix to prevent root-anchoring, got: {result!r}"
    )
    result2 = _fnmatch_to_gitignore("**/.vuepress/dist/**")
    assert result2.startswith("**/"), (
        f"Expected '**/' prefix to prevent root-anchoring, got: {result2!r}"
    )

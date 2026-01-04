from __future__ import annotations


def test_extract_include_prefixes_gets_anchors():
    from chunkhound.utils.file_patterns import _extract_include_prefixes

    inc = [
        "src/**/*.ts",
        "docs/api/**/*.md",
        "**/*.py",  # wide, no anchor
        "README",  # filename pattern, no anchor
        "**/Makefile",  # filename pattern, no anchor
    ]
    prefixes = _extract_include_prefixes(inc)
    assert prefixes == {"src", "docs/api"}


def test_can_prune_by_prefix_logic():
    from chunkhound.utils.file_patterns import (
        _can_prune_dir_by_prefix,
        _extract_include_prefixes,
    )

    inc = ["src/**/*.ts", "docs/api/**/*.md"]
    prefixes = _extract_include_prefixes(inc)

    # At root: unrelated top-level dir should be pruned
    assert _can_prune_dir_by_prefix(prefixes, current_rel="pkg") is True
    # 'src' is exactly a prefix → do not prune
    assert _can_prune_dir_by_prefix(prefixes, current_rel="src") is False
    # 'docs' is ancestor of 'docs/api' → do not prune
    assert _can_prune_dir_by_prefix(prefixes, current_rel="docs") is False
    # 'docs/api' exact → do not prune
    assert _can_prune_dir_by_prefix(prefixes, current_rel="docs/api") is False
    # 'docsx' unrelated → prune
    assert _can_prune_dir_by_prefix(prefixes, current_rel="docsx") is True


def test_heavy_dir_prune_respects_explicit_prefix():
    from chunkhound.utils.file_patterns import _should_prune_heavy_dir

    heavy = {"node_modules", ".venv"}

    # No anchors → prune heavy dir
    assert (
        _should_prune_heavy_dir(
            heavy, include_prefixes=set(), current_name="node_modules"
        )
        is True
    )
    # Anchored to node_modules → do not prune
    assert (
        _should_prune_heavy_dir(
            heavy, include_prefixes={"node_modules"}, current_name="node_modules"
        )
        is False
    )

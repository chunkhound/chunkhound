from pathlib import Path

import pytest

from chunkhound.utils.file_patterns import should_exclude_path


@pytest.mark.unit
def test_exclude_pattern_with_double_star_prefix_matches_anywhere_in_tree():
    base = Path(".")
    # Simulate repo path
    target = Path(
        "monorepo/workflows-engine/camunda-cockpit-plugins/instance-route-history.js"
    )

    # User-specified exclude; should match regardless of leading segments
    patterns = ["**/workflows-engine/camunda-cockpit-plugins/instance-route-history.js"]

    assert should_exclude_path(target, base, patterns, {}) is True


@pytest.mark.unit
def test_exclude_pattern_without_double_star_needs_exact_prefix():
    base = Path(".")
    target = Path(
        "monorepo/workflows-engine/camunda-cockpit-plugins/instance-route-history.js"
    )

    patterns = ["workflows-engine/camunda-cockpit-plugins/instance-route-history.js"]

    # Depending on implementation, this may or may not match. This test
    # documents that the double-star form is the portable one for matching
    # the file at any depth, while the bare relative form should not match
    # when there's a leading segment like 'monorepo/'.
    assert should_exclude_path(target, base, patterns, {}) is False

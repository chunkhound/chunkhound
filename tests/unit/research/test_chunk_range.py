"""Tests for smart boundary expansion edge cases."""

from chunkhound.services.research.shared.chunk_range import (
    expand_to_natural_boundaries,
)


def test_boundary_expansion_skips_invalid_start() -> None:
    lines = ["line"] * 5
    start, end = expand_to_natural_boundaries(
        lines=lines,
        start_line=10,
        end_line=12,
        chunk={},
        file_path="sample.py",
    )
    assert (start, end) == (0, 0)


def test_boundary_expansion_skips_invalid_end() -> None:
    lines = ["line"] * 5
    start, end = expand_to_natural_boundaries(
        lines=lines,
        start_line=1,
        end_line=0,
        chunk={},
        file_path="sample.py",
    )
    assert (start, end) == (0, 0)


def test_boundary_expansion_skips_reversed_range() -> None:
    lines = ["line"] * 5
    start, end = expand_to_natural_boundaries(
        lines=lines,
        start_line=4,
        end_line=2,
        chunk={},
        file_path="sample.py",
    )
    assert (start, end) == (0, 0)

"""Unit tests for normalize_path_for_lookup contract.

Tests the external invariant that normalize_path_for_lookup always returns:
  - Relative (never absolute)
  - Forward-slash separated

These tests verify the user-facing contract, not implementation details.
A refactor that changes internal normalization logic but preserves these
invariants should not break these tests.
"""

import os
from pathlib import Path

import pytest

from chunkhound.core.utils.path_utils import normalize_path_for_lookup


# ---------------------------------------------------------------------------
# normalize_path_for_lookup: relative paths
# ---------------------------------------------------------------------------


class TestNormalizePathForLookupRelative:
    """normalize_path_for_lookup preserves relative paths via as_posix()."""

    @pytest.mark.parametrize(
        "input_path,expected",
        [
            ("src/utils.py", "src/utils.py"),
            ("./src/utils.py", "src/utils.py"),
            ("src//utils.py", "src/utils.py"),
            ("src/./utils.py", "src/utils.py"),
            ("a/b/c/d.py", "a/b/c/d.py"),
        ],
        ids=[
            "already-relative",
            "leading-dot-slash",
            "double-slash",
            "middle-dot-slash",
            "deeply-nested",
        ],
    )
    def test_relative_paths_preserved(self, input_path: str, expected: str) -> None:
        """Relative paths are preserved with as_posix() normalization."""
        result = normalize_path_for_lookup(input_path, base_dir=Path("/project"))
        assert result == expected

    def test_single_component(self) -> None:
        """Single filename without directories."""
        result = normalize_path_for_lookup("file.py", base_dir=Path("/project"))
        assert result == "file.py"

    def test_hidden_directory(self) -> None:
        """Path in hidden directory."""
        result = normalize_path_for_lookup(
            ".github/workflows/ci.yml", base_dir=Path("/project")
        )
        assert result == ".github/workflows/ci.yml"

    def test_path_with_spaces(self) -> None:
        """Path with spaces is preserved."""
        result = normalize_path_for_lookup(
            "my folder/my file.py", base_dir=Path("/project")
        )
        assert result == "my folder/my file.py"

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only native path case")
    def test_windows_relative_path_uses_forward_slashes(self) -> None:
        """Windows relative paths normalize to forward slashes."""
        result = normalize_path_for_lookup(r"src\utils.py", base_dir=Path("C:/project"))
        assert result == "src/utils.py"


# ---------------------------------------------------------------------------
# normalize_path_for_lookup: absolute paths
# ---------------------------------------------------------------------------


class TestNormalizePathForLookupAbsolute:
    """Absolute paths require base_dir and are converted to relative."""

    def test_absolute_requires_base_dir(self, tmp_path: Path) -> None:
        """Absolute path without base_dir raises ValueError."""
        absolute_path = tmp_path / "absolute" / "path.py"
        with pytest.raises(ValueError, match="Cannot normalize absolute path"):
            normalize_path_for_lookup(str(absolute_path))

    def test_absolute_under_base_dir(self, tmp_path: Path) -> None:
        """Absolute path under base_dir is converted to relative."""
        base = tmp_path / "project"
        absolute_path = base / "src" / "utils.py"
        result = normalize_path_for_lookup(str(absolute_path), base_dir=base)
        assert result == "src/utils.py"
        assert not result.startswith("/")

    def test_absolute_outside_base_dir_raises(self, tmp_path: Path) -> None:
        """Absolute path outside base_dir raises ValueError."""
        base = tmp_path / "project"
        outside = tmp_path / "other" / "project" / "file.py"
        with pytest.raises(ValueError, match="not under base directory"):
            normalize_path_for_lookup(str(outside), base_dir=base)

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only native path case")
    def test_windows_absolute_under_base_dir(self) -> None:
        """Windows absolute paths normalize to forward-slash relative paths."""
        base = Path("C:/project")
        result = normalize_path_for_lookup(r"C:\project\src\utils.py", base_dir=base)
        assert result == "src/utils.py"


# ---------------------------------------------------------------------------
# Contract: invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    """Invariants that must hold for all paths from normalize_path_for_lookup."""

    def test_lookup_never_returns_absolute(self) -> None:
        """No path from normalize_path_for_lookup is ever absolute."""
        paths = ["src/utils.py", "./src/utils.py", "a/b/c.py"]
        for p in paths:
            result = normalize_path_for_lookup(p, base_dir=Path("/project"))
            assert not Path(result).is_absolute(), f"Path is absolute: {result}"

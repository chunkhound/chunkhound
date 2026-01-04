"""Tests for path traversal validation in MCP server.

These tests verify that the path validation functions correctly
reject path traversal attacks while allowing legitimate paths.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from chunkhound.mcp_server.common import is_safe_relative_path, validate_and_join_path


class TestIsSafeRelativePath:
    """Tests for is_safe_relative_path function."""

    def test_safe_simple_path(self):
        """Simple relative paths should be allowed."""
        assert is_safe_relative_path("src") is True
        assert is_safe_relative_path("src/components") is True
        assert is_safe_relative_path("tests/unit/test_foo.py") is True

    def test_safe_current_dir_prefix(self):
        """Paths with ./ prefix should be allowed."""
        assert is_safe_relative_path("./src") is True
        assert is_safe_relative_path("./tests/unit") is True

    def test_reject_parent_directory_traversal(self):
        """Parent directory traversal should be rejected."""
        assert is_safe_relative_path("..") is False
        assert is_safe_relative_path("../etc/passwd") is False
        assert is_safe_relative_path("../../secret") is False
        assert is_safe_relative_path("foo/../..") is False
        assert is_safe_relative_path("foo/bar/../../../etc") is False

    def test_reject_absolute_paths(self):
        """Absolute paths should be rejected."""
        assert is_safe_relative_path("/etc/passwd") is False
        assert is_safe_relative_path("/home/user/.ssh") is False

    def test_reject_sneaky_traversal(self):
        """Paths that normalize to parent traversal should be rejected."""
        assert is_safe_relative_path("foo/../../bar") is False
        assert is_safe_relative_path("./foo/../../../etc") is False

    def test_safe_internal_dots(self):
        """Paths with dots that don't escape should be allowed."""
        assert is_safe_relative_path("foo/bar/..") is True  # stays in foo
        assert is_safe_relative_path("foo/./bar") is True
        assert is_safe_relative_path(".hidden") is True
        assert is_safe_relative_path("foo/.hidden/bar") is True


class TestValidateAndJoinPath:
    """Tests for validate_and_join_path function."""

    def test_safe_path_join(self):
        """Safe paths should be joined correctly."""
        with tempfile.TemporaryDirectory() as base:
            result = validate_and_join_path(base, "src")
            assert result is not None
            assert result == str(Path(base) / "src")

    def test_safe_nested_path(self):
        """Nested safe paths should work."""
        with tempfile.TemporaryDirectory() as base:
            result = validate_and_join_path(base, "src/components/Button.tsx")
            assert result is not None
            assert "src/components/Button.tsx" in result

    def test_reject_parent_traversal(self):
        """Parent traversal should return None."""
        with tempfile.TemporaryDirectory() as base:
            result = validate_and_join_path(base, "../etc/passwd")
            assert result is None

    def test_reject_deep_traversal(self):
        """Deep traversal attempts should return None."""
        with tempfile.TemporaryDirectory() as base:
            result = validate_and_join_path(base, "foo/../../..")
            assert result is None

    def test_reject_absolute_path(self):
        """Absolute paths should return None."""
        with tempfile.TemporaryDirectory() as base:
            result = validate_and_join_path(base, "/etc/passwd")
            assert result is None

    def test_safe_with_internal_navigation(self):
        """Internal navigation that stays in bounds should work."""
        with tempfile.TemporaryDirectory() as base:
            # Create subdirectory
            subdir = Path(base) / "foo" / "bar"
            subdir.mkdir(parents=True)

            # foo/bar/.. resolves to foo, which is still in base
            result = validate_and_join_path(base, "foo/bar/..")
            assert result is not None
            # Result should be foo (normalized)
            assert Path(result).name == "foo" or "foo" in result

    def test_rejects_symlink_escape(self):
        """Symlinks that escape should be caught by resolve check."""
        with tempfile.TemporaryDirectory() as base:
            with tempfile.TemporaryDirectory() as outside:
                # Create a symlink inside base pointing outside
                link_path = Path(base) / "escape_link"
                try:
                    link_path.symlink_to(outside)

                    # Following the symlink would escape the base
                    result = validate_and_join_path(base, "escape_link/secret")
                    # Should be rejected because resolved path is outside base
                    assert result is None
                except OSError:
                    # Symlink creation may fail on some systems
                    pytest.skip("Cannot create symlinks on this system")


class TestPathTraversalVectors:
    """Test various path traversal attack vectors."""

    ATTACK_VECTORS = [
        # Basic parent traversal
        "..",
        "../",
        "../..",
        "../../",
        "../../../etc/passwd",
        # Hidden in middle
        "foo/../..",
        "foo/bar/../../../etc",
        "foo/bar/baz/../../../../..",
        # URL-style encoding (after normpath these become literal characters)
        "%2e%2e/",
        "%2e%2e%2f",
        "..%2f",
        # Unicode variants
        "..﹒/",
        "..．/",
        # Double encoding
        "%252e%252e/",
        # Null byte (if not handled)
        "../\x00.txt",
        "foo\x00/../..",
        # Windows-style paths (on Unix these are literal)
        "..\\",
        "..\\..\\etc",
        # Mixed separators
        "../..\\etc",
        "foo\\..\\../..",
    ]

    @pytest.mark.parametrize("attack_path", ATTACK_VECTORS)
    def test_attack_vector_rejected(self, attack_path: str):
        """Various attack vectors should be rejected or harmless."""
        # After normpath, these should either:
        # 1. Be rejected by is_safe_relative_path
        # 2. Stay within bounds when resolved

        normalized = os.path.normpath(attack_path)

        # If normpath produces a parent traversal, it should be rejected
        if normalized.startswith("..") or normalized.startswith(os.sep):
            assert is_safe_relative_path(attack_path) is False

        # Double-check with actual path joining
        with tempfile.TemporaryDirectory() as base:
            result = validate_and_join_path(base, attack_path)
            if result is not None:
                # If we got a result, verify it's actually within base
                resolved = Path(result).resolve()
                base_resolved = Path(base).resolve()
                assert str(resolved).startswith(str(base_resolved))


class TestEdgeCases:
    """Edge case tests for path validation."""

    def test_empty_path(self):
        """Empty path should be safe (represents current dir)."""
        assert is_safe_relative_path("") is True

    def test_dot_path(self):
        """Single dot should be safe."""
        assert is_safe_relative_path(".") is True

    def test_hidden_files(self):
        """Hidden files/dirs (starting with .) should be safe."""
        assert is_safe_relative_path(".git") is True
        assert is_safe_relative_path(".github/workflows") is True
        assert is_safe_relative_path("src/.hidden") is True

    def test_spaces_in_path(self):
        """Paths with spaces should be handled correctly."""
        assert is_safe_relative_path("my project/src") is True
        assert is_safe_relative_path("path with spaces/file.txt") is True

    def test_special_characters(self):
        """Paths with special characters should be handled."""
        assert is_safe_relative_path("src-v2/main.py") is True
        assert is_safe_relative_path("tests_unit/test_foo.py") is True
        assert is_safe_relative_path("foo@bar/baz") is True

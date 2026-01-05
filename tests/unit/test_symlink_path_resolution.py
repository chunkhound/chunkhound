"""Tests for symlink path resolution in indexing.

Tests the fix for git worktree symlinks where symlinks may point outside
the indexed base directory. Symlinks should preserve their logical paths
while regular files are resolved (for Windows 8.3 compatibility).
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from chunkhound.core.utils.path_utils import (
    get_relative_path_safe,
    resolve_path_for_relative,
)


@pytest.fixture
def temp_dirs():
    """Create two temporary directories for testing symlinks."""
    main_repo = Path(tempfile.mkdtemp())
    worktree = Path(tempfile.mkdtemp())
    yield main_repo, worktree
    shutil.rmtree(main_repo, ignore_errors=True)
    shutil.rmtree(worktree, ignore_errors=True)


class TestResolvePathForRelative:
    """Tests for resolve_path_for_relative helper."""

    def test_regular_file_is_resolved(self, temp_dirs):
        """Regular files should be resolved for Windows 8.3 compatibility."""
        main_repo, _ = temp_dirs
        test_file = main_repo / "test.py"
        test_file.write_text("# test")

        path_to_use, resolved_base = resolve_path_for_relative(test_file, main_repo)

        # Regular file should be resolved
        assert path_to_use == test_file.resolve()
        assert resolved_base == main_repo.resolve()

    @pytest.mark.skipif(os.name == "nt", reason="Symlinks require admin on Windows")
    def test_symlink_preserves_logical_path(self, temp_dirs):
        """Symlinks should preserve their logical path, not resolve to target."""
        main_repo, worktree = temp_dirs

        # Create target file in main repo
        target_file = main_repo / "shared_config.py"
        target_file.write_text("# shared config")

        # Create symlink in worktree pointing to main repo
        symlink_file = worktree / "config.py"
        symlink_file.symlink_to(target_file)

        path_to_use, resolved_base = resolve_path_for_relative(symlink_file, worktree)

        # Symlink should NOT be resolved - keep logical path
        assert path_to_use == symlink_file
        assert path_to_use != target_file  # Not the target


class TestGetRelativePathSafe:
    """Tests for get_relative_path_safe helper."""

    def test_regular_file_relative_path(self, temp_dirs):
        """Regular file should return correct relative path."""
        main_repo, _ = temp_dirs
        subdir = main_repo / "src"
        subdir.mkdir()
        test_file = subdir / "module.py"
        test_file.write_text("# module")

        rel_path = get_relative_path_safe(test_file, main_repo)

        assert rel_path == Path("src/module.py")

    @pytest.mark.skipif(os.name == "nt", reason="Symlinks require admin on Windows")
    def test_symlink_within_base_returns_logical_path(self, temp_dirs):
        """Symlink within base directory returns its logical path."""
        main_repo, _ = temp_dirs

        # Create target and symlink both within main_repo
        target = main_repo / "real.py"
        target.write_text("# real")
        symlink = main_repo / "alias.py"
        symlink.symlink_to(target)

        rel_path = get_relative_path_safe(symlink, main_repo)

        # Should return "alias.py", not "real.py"
        assert rel_path == Path("alias.py")

    @pytest.mark.skipif(os.name == "nt", reason="Symlinks require admin on Windows")
    def test_worktree_symlink_pointing_outside(self, temp_dirs):
        """Symlink in worktree pointing outside should preserve logical path."""
        main_repo, worktree = temp_dirs

        # Create target in main repo (outside worktree)
        target = main_repo / "shared.py"
        target.write_text("# shared")

        # Create symlink in worktree pointing to main repo
        symlink = worktree / "linked.py"
        symlink.symlink_to(target)

        # This should work - symlink path is under worktree
        rel_path = get_relative_path_safe(symlink, worktree)

        assert rel_path == Path("linked.py")

    @pytest.mark.skipif(os.name == "nt", reason="Symlinks require admin on Windows")
    def test_worktree_symlink_in_subdirectory(self, temp_dirs):
        """Symlink in worktree subdirectory pointing outside."""
        main_repo, worktree = temp_dirs

        # Create target in main repo
        target = main_repo / "config.py"
        target.write_text("# config")

        # Create symlink in worktree/src/
        subdir = worktree / "src"
        subdir.mkdir()
        symlink = subdir / "config.py"
        symlink.symlink_to(target)

        rel_path = get_relative_path_safe(symlink, worktree)

        assert rel_path == Path("src/config.py")

    def test_path_not_under_base_raises(self, temp_dirs):
        """Path not under base directory should raise ValueError."""
        main_repo, worktree = temp_dirs

        # Create file in main_repo
        test_file = main_repo / "test.py"
        test_file.write_text("# test")

        # Try to get relative path with wrong base
        with pytest.raises(ValueError):
            get_relative_path_safe(test_file, worktree)


class TestWindowsCompatibility:
    """Tests for Windows 8.3 short name handling."""

    def test_resolved_paths_match(self, temp_dirs):
        """Test that resolved paths allow relative_to to work."""
        main_repo, _ = temp_dirs
        test_file = main_repo / "test.py"
        test_file.write_text("# test")

        # Even if paths have different representations, resolved should match
        rel_path = get_relative_path_safe(test_file, main_repo)

        assert rel_path == Path("test.py")

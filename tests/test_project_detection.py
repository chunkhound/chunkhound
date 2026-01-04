"""Test project root detection logic."""

import os
import tempfile
from pathlib import Path

import pytest

from chunkhound.utils.project_detection import find_project_root


class TestProjectRootDetection:
    """Test find_project_root() tree-walking behavior."""

    def test_finds_chunkhound_json_in_current_directory(self):
        """Should find .chunkhound.json in current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()  # Resolve symlinks for macOS /var
            (project_root / ".chunkhound.json").touch()

            # Change to project root
            original_cwd = Path.cwd()
            try:
                os.chdir(project_root)
                result = find_project_root()
                assert result == project_root
            finally:
                os.chdir(original_cwd)

    def test_finds_chunkhound_json_in_parent_directory(self):
        """Should walk up tree to find .chunkhound.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()  # Resolve symlinks for macOS /var
            nested_dir = project_root / "src" / "services"
            nested_dir.mkdir(parents=True)
            (project_root / ".chunkhound.json").touch()

            # Change to nested directory
            original_cwd = Path.cwd()
            try:
                os.chdir(nested_dir)
                result = find_project_root()
                assert result == project_root
            finally:
                os.chdir(original_cwd)

    def test_finds_chunkhound_db_in_parent_directory(self):
        """Should walk up tree to find .chunkhound/db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()  # Resolve symlinks for macOS /var
            nested_dir = project_root / "src" / "services"
            nested_dir.mkdir(parents=True)
            db_dir = project_root / ".chunkhound" / "db"
            db_dir.mkdir(parents=True)

            # Change to nested directory
            original_cwd = Path.cwd()
            try:
                os.chdir(nested_dir)
                result = find_project_root()
                assert result == project_root
            finally:
                os.chdir(original_cwd)

    def test_finds_git_root_in_parent_directory(self):
        """Should walk up tree to find .git."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()  # Resolve symlinks for macOS /var
            nested_dir = project_root / "src" / "services"
            nested_dir.mkdir(parents=True)
            (project_root / ".git").mkdir()

            # Change to nested directory
            original_cwd = Path.cwd()
            try:
                os.chdir(nested_dir)
                result = find_project_root()
                assert result == project_root
            finally:
                os.chdir(original_cwd)

    def test_chunkhound_json_overrides_git(self):
        """Should prioritize .chunkhound.json over .git."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create monorepo structure
            monorepo_root = Path(tmpdir).resolve()  # Resolve symlinks for macOS /var
            project_a = monorepo_root / "project-a"
            project_a_nested = project_a / "src"
            project_a_nested.mkdir(parents=True)

            # Git at monorepo root
            (monorepo_root / ".git").mkdir()

            # .chunkhound.json at project-a (should win)
            (project_a / ".chunkhound.json").touch()

            # Change to nested directory in project-a
            original_cwd = Path.cwd()
            try:
                os.chdir(project_a_nested)
                result = find_project_root()
                # Should find project-a, not monorepo root
                assert result == project_a
            finally:
                os.chdir(original_cwd)

    def test_chunkhound_json_overrides_db(self):
        """Should prioritize .chunkhound.json over .chunkhound/db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()  # Resolve symlinks for macOS /var
            nested_dir = project_root / "src"
            nested_dir.mkdir(parents=True)

            # Both markers present
            (project_root / ".chunkhound.json").touch()
            db_dir = project_root / ".chunkhound" / "db"
            db_dir.mkdir(parents=True)

            # Change to nested directory
            original_cwd = Path.cwd()
            try:
                os.chdir(nested_dir)
                result = find_project_root()
                assert result == project_root
            finally:
                os.chdir(original_cwd)

    def test_chunkhound_db_overrides_git(self):
        """Should prioritize .chunkhound/db over .git."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure with both markers at different levels
            git_root = Path(tmpdir).resolve()  # Resolve symlinks for macOS /var
            project_dir = git_root / "project"
            nested_dir = project_dir / "src"
            nested_dir.mkdir(parents=True)

            # Git at top level
            (git_root / ".git").mkdir()

            # Database at project level (should win)
            db_dir = project_dir / ".chunkhound" / "db"
            db_dir.mkdir(parents=True)

            # Change to nested directory
            original_cwd = Path.cwd()
            try:
                os.chdir(nested_dir)
                result = find_project_root()
                # Should find project_dir (with db), not git_root
                assert result == project_dir
            finally:
                os.chdir(original_cwd)

    def test_explicit_path_overrides_auto_detection(self):
        """CLI args should override auto-detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()  # Resolve symlinks for macOS /var
            other_project = project_root / "other"
            other_project.mkdir()

            # Markers in project_root
            (project_root / ".chunkhound.json").touch()

            # Explicit path to other_project should win
            result = find_project_root(other_project)
            assert result == other_project

    def test_fails_when_no_markers_found(self):
        """Should exit with error when no markers found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()

            # Change to empty directory
            original_cwd = Path.cwd()
            try:
                os.chdir(empty_dir)
                # Should call sys.exit(1)
                with pytest.raises(SystemExit) as exc_info:
                    find_project_root()
                assert exc_info.value.code == 1
            finally:
                os.chdir(original_cwd)

    def test_stops_at_home_directory(self):
        """Should not walk above home directory."""
        # This test is hard to verify without mocking, but we document the behavior
        # The code has: while current != current.parent and current != home
        pass

    def test_multiple_nested_levels(self):
        """Should work from deeply nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()  # Resolve symlinks for macOS /var
            deeply_nested = project_root / "a" / "b" / "c" / "d" / "e"
            deeply_nested.mkdir(parents=True)
            (project_root / ".chunkhound.json").touch()

            # Change to deeply nested directory
            original_cwd = Path.cwd()
            try:
                os.chdir(deeply_nested)
                result = find_project_root()
                assert result == project_root
            finally:
                os.chdir(original_cwd)

    def test_explicit_path_with_nonexistent_directory(self):
        """Should exit with error for nonexistent explicit path."""
        nonexistent = Path("/nonexistent/path/to/project")

        with pytest.raises(SystemExit) as exc_info:
            find_project_root(nonexistent)
        assert exc_info.value.code == 1

    def test_explicit_path_with_file_not_directory(self):
        """Should exit with error when explicit path is a file, not directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "file.txt"
            file_path.touch()

            with pytest.raises(SystemExit) as exc_info:
                find_project_root(file_path)
            assert exc_info.value.code == 1

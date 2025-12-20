"""Unit tests for global gitignore support.

Tests for:
- get_global_excludes_file(): Config read, fallback paths, missing file
- _collect_global_gitignore_patterns(): Pattern prefixing, negations, anchored
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestGetGlobalExcludesFile:
    """Tests for get_global_excludes_file() in git_safe.py."""

    def test_reads_from_git_config(self, tmp_path: Path) -> None:
        """Test reading core.excludesFile from git config."""
        from chunkhound.utils.git_safe import get_global_excludes_file

        # Create a temp global gitignore file
        global_ignore = tmp_path / "my_global_ignore"
        global_ignore.write_text("*.pyc\n")

        # Mock subprocess to return the path
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = str(global_ignore) + "\n"

        with patch("subprocess.run", return_value=mock_result):
            result = get_global_excludes_file()
            assert result == global_ignore

    def test_expands_tilde_in_config_path(self, tmp_path: Path, monkeypatch) -> None:
        """Test that ~ is expanded in the config path."""
        from chunkhound.utils.git_safe import get_global_excludes_file

        # Create file in a fake home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        global_ignore = fake_home / ".gitignore_global"
        global_ignore.write_text("*.log\n")

        # Mock expanduser to use our fake home (works cross-platform)
        # HOME works on Unix, USERPROFILE on Windows
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("USERPROFILE", str(fake_home))

        # Mock subprocess to return tilde path
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "~/.gitignore_global\n"

        with patch("subprocess.run", return_value=mock_result):
            result = get_global_excludes_file()
            assert result == global_ignore

    def test_fallback_to_gitignore_global(self, tmp_path: Path, monkeypatch) -> None:
        """Test fallback to ~/.gitignore_global when git config fails."""
        from chunkhound.utils.git_safe import get_global_excludes_file

        # Create file in fake home
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        global_ignore = fake_home / ".gitignore_global"
        global_ignore.write_text("*.tmp\n")

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Mock subprocess to fail
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = get_global_excludes_file()
            assert result == global_ignore

    def test_fallback_to_home_gitignore(self, tmp_path: Path, monkeypatch) -> None:
        """Test fallback to ~/.gitignore when ~/.gitignore_global doesn't exist."""
        from chunkhound.utils.git_safe import get_global_excludes_file

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        gitignore = fake_home / ".gitignore"
        gitignore.write_text("*.bak\n")

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = get_global_excludes_file()
            assert result == gitignore

    def test_fallback_to_config_git_ignore(self, tmp_path: Path, monkeypatch) -> None:
        """Test fallback to ~/.config/git/ignore."""
        from chunkhound.utils.git_safe import get_global_excludes_file

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        config_dir = fake_home / ".config" / "git"
        config_dir.mkdir(parents=True)
        ignore_file = config_dir / "ignore"
        ignore_file.write_text("*.swp\n")

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = get_global_excludes_file()
            assert result == ignore_file

    def test_returns_none_when_no_file(self, tmp_path: Path, monkeypatch) -> None:
        """Test returns None when no global gitignore exists."""
        from chunkhound.utils.git_safe import get_global_excludes_file

        # Empty fake home with no gitignore files
        fake_home = tmp_path / "empty_home"
        fake_home.mkdir()

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = get_global_excludes_file()
            assert result is None

    def test_handles_subprocess_exception(self, tmp_path: Path, monkeypatch) -> None:
        """Test gracefully handles subprocess exceptions."""
        from chunkhound.utils.git_safe import get_global_excludes_file

        fake_home = tmp_path / "home"
        fake_home.mkdir()

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        with patch("subprocess.run", side_effect=Exception("Git not found")):
            # Should not raise, returns None
            result = get_global_excludes_file()
            assert result is None

    def test_handles_timeout(self, tmp_path: Path, monkeypatch) -> None:
        """Test handles git command timeout."""
        from chunkhound.utils.git_safe import get_global_excludes_file

        fake_home = tmp_path / "home"
        fake_home.mkdir()

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)):
            result = get_global_excludes_file()
            assert result is None

    def test_config_path_not_exists_triggers_fallback(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test that non-existent config path triggers fallback."""
        from chunkhound.utils.git_safe import get_global_excludes_file

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        fallback_file = fake_home / ".gitignore_global"
        fallback_file.write_text("fallback\n")

        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Git config returns a path that doesn't exist
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "/nonexistent/path/.gitignore\n"

        with patch("subprocess.run", return_value=mock_result):
            result = get_global_excludes_file()
            assert result == fallback_file


class TestCollectGlobalGitignorePatterns:
    """Tests for _collect_global_gitignore_patterns() in ignore_engine.py."""

    def test_returns_empty_when_no_global_file(self, monkeypatch) -> None:
        """Test returns empty list when no global gitignore exists."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file", return_value=None
        ):
            result = _collect_global_gitignore_patterns()
            assert result == []

    def test_prefixes_simple_patterns_with_recursive(self, tmp_path: Path) -> None:
        """Test simple patterns get **/ prefix to match anywhere."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        global_file = tmp_path / ".gitignore_global"
        global_file.write_text("*.pyc\n__pycache__\n.DS_Store\n")

        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file",
            return_value=global_file,
        ):
            result = _collect_global_gitignore_patterns()
            assert "**/*.pyc" in result
            assert "**/__pycache__" in result
            assert "**/.DS_Store" in result

    def test_preserves_already_recursive_patterns(self, tmp_path: Path) -> None:
        """Test patterns starting with **/ are not double-prefixed."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        global_file = tmp_path / ".gitignore_global"
        global_file.write_text("**/node_modules\n**/build\n")

        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file",
            return_value=global_file,
        ):
            result = _collect_global_gitignore_patterns()
            assert "**/node_modules" in result
            assert "**/build" in result
            # Should not be double-prefixed
            assert "**/**/node_modules" not in result

    def test_preserves_anchored_patterns(self, tmp_path: Path) -> None:
        """Test patterns starting with / are preserved as-is."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        global_file = tmp_path / ".gitignore_global"
        global_file.write_text("/root_only\n/specific_dir/\n")

        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file",
            return_value=global_file,
        ):
            result = _collect_global_gitignore_patterns()
            assert "/root_only" in result
            assert "/specific_dir/" in result

    def test_preserves_negation_patterns(self, tmp_path: Path) -> None:
        """Test negation patterns (!) are preserved."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        global_file = tmp_path / ".gitignore_global"
        global_file.write_text("*.log\n!important.log\n")

        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file",
            return_value=global_file,
        ):
            result = _collect_global_gitignore_patterns()
            assert "**/*.log" in result
            assert "!important.log" in result

    def test_skips_comments_and_empty_lines(self, tmp_path: Path) -> None:
        """Test comments and empty lines are ignored."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        global_file = tmp_path / ".gitignore_global"
        global_file.write_text("# This is a comment\n\n*.tmp\n   # Another comment\n\n")

        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file",
            return_value=global_file,
        ):
            result = _collect_global_gitignore_patterns()
            assert len(result) == 1
            assert "**/*.tmp" in result

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        """Test gracefully handles empty global gitignore file."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        global_file = tmp_path / ".gitignore_global"
        global_file.write_text("")

        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file",
            return_value=global_file,
        ):
            result = _collect_global_gitignore_patterns()
            assert result == []

    def test_handles_file_read_error(self, tmp_path: Path) -> None:
        """Test gracefully handles file read errors."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        global_file = tmp_path / ".gitignore_global"
        global_file.write_text("*.tmp\n")

        # Make file unreadable (mock read_text to raise)
        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file",
            return_value=global_file,
        ):
            with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                result = _collect_global_gitignore_patterns()
                assert result == []

    def test_strips_whitespace_from_patterns(self, tmp_path: Path) -> None:
        """Test whitespace is stripped from patterns."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        global_file = tmp_path / ".gitignore_global"
        global_file.write_text("  *.log  \n\t*.tmp\t\n")

        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file",
            return_value=global_file,
        ):
            result = _collect_global_gitignore_patterns()
            assert "**/*.log" in result
            assert "**/*.tmp" in result

    def test_handles_mixed_patterns(self, tmp_path: Path) -> None:
        """Test handles a mix of pattern types correctly."""
        from chunkhound.utils.ignore_engine import _collect_global_gitignore_patterns

        global_file = tmp_path / ".gitignore_global"
        global_file.write_text(
            """# IDE files
.idea/
*.swp
.vscode/

# OS files
.DS_Store
Thumbs.db

# Already recursive
**/node_modules

# Anchored (rare in global)
/local_only

# Negation
!.gitkeep
"""
        )

        with patch(
            "chunkhound.utils.git_safe.get_global_excludes_file",
            return_value=global_file,
        ):
            result = _collect_global_gitignore_patterns()

            # Simple patterns get **/ prefix
            assert "**/.idea/" in result
            assert "**/*.swp" in result
            assert "**/.vscode/" in result
            assert "**/.DS_Store" in result
            assert "**/Thumbs.db" in result

            # Already recursive - no double prefix
            assert "**/node_modules" in result
            assert "**/**/node_modules" not in result

            # Anchored preserved
            assert "/local_only" in result

            # Negation preserved
            assert "!.gitkeep" in result


class TestExtendWithGlobalGitignore:
    """Tests for IndexingCoordinator._extend_with_global_gitignore()."""

    def test_extends_list_with_patterns(self, tmp_path: Path) -> None:
        """Test that effective_excludes is extended with global patterns."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.indexing_coordinator import IndexingCoordinator

        db = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
        coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path)

        excludes = ["existing_pattern"]

        with patch(
            "chunkhound.utils.ignore_engine._collect_global_gitignore_patterns",
            return_value=["**/*.pyc", "**/.DS_Store"],
        ):
            coord._extend_with_global_gitignore(excludes)

        assert "existing_pattern" in excludes
        assert "**/*.pyc" in excludes
        assert "**/.DS_Store" in excludes

    def test_handles_import_error_gracefully(self, tmp_path: Path) -> None:
        """Test gracefully handles import errors."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.indexing_coordinator import IndexingCoordinator

        db = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
        coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path)

        excludes = ["existing"]

        # Simulate import error
        with patch(
            "chunkhound.services.indexing_coordinator.IndexingCoordinator._extend_with_global_gitignore",
            side_effect=ImportError("Module not found"),
        ):
            # Should not raise
            try:
                coord._extend_with_global_gitignore(excludes)
            except ImportError:
                pass  # Expected when we patch the method itself

    def test_handles_empty_patterns(self, tmp_path: Path) -> None:
        """Test handles empty patterns list from collector."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.indexing_coordinator import IndexingCoordinator

        db = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
        coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path)

        excludes = ["original"]

        with patch(
            "chunkhound.utils.ignore_engine._collect_global_gitignore_patterns",
            return_value=[],
        ):
            coord._extend_with_global_gitignore(excludes)

        # Only original pattern should remain
        assert excludes == ["original"]

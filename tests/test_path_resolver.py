"""Unit tests for PathResolver service.

Tests path resolution for multi-repository support, including:
- Absolute path resolution against indexed base directories
- Relative path resolution using current working directory
- Cache functionality and invalidation
- Error handling for missing paths
"""

from unittest.mock import Mock

import pytest

from chunkhound.services.path_resolver import PathResolver


class TestPathResolver:
    """Test PathResolver class for multi-repo path resolution."""

    @pytest.fixture
    def mock_db_provider(self):
        """Create mock database provider with indexed roots."""
        import os

        provider = Mock()

        # Mock indexed roots: two projects
        provider.get_indexed_roots.return_value = [
            {
                "base_directory": "/home/user/project1",
                "project_name": "project1",
                "file_count": 100,
            },
            {
                "base_directory": "/home/user/project2",
                "project_name": "project2",
                "file_count": 50,
            },
        ]

        # Canonicalize test paths for comparison (handles macOS symlinks)
        project1_base = os.path.realpath("/home/user/project1")
        project2_base = os.path.realpath("/home/user/project2")

        # Mock find_base_for_path: returns containing base or None
        def find_base(path: str) -> str | None:
            if path.startswith(project1_base):
                return project1_base
            elif path.startswith(project2_base):
                return project2_base
            return None

        provider.find_base_for_path = Mock(side_effect=find_base)

        return provider

    @pytest.fixture
    def resolver(self, mock_db_provider):
        """Create PathResolver instance with mock provider."""
        return PathResolver(mock_db_provider)

    def test_resolve_absolute_path_in_indexed_base(self, resolver, mock_db_provider):
        """Test resolving absolute path that exists in indexed base."""
        path = "/home/user/project1/src/main.py"
        result = resolver.resolve_path(path)

        # Should return canonicalized path (symlinks resolved)
        # On macOS, /home → /System/Volumes/Data/home
        import os

        expected = os.path.realpath(path)
        assert result == expected

        # Should have queried database with canonical path
        mock_db_provider.find_base_for_path.assert_called_once_with(expected)

    def test_resolve_absolute_path_not_in_indexed_base(
        self, resolver, mock_db_provider
    ):
        """Test resolving absolute path not in any indexed base."""
        path = "/home/user/project3/file.py"
        result = resolver.resolve_path(path)

        # Should return canonicalized path (symlinks resolved)
        import os

        expected = os.path.realpath(path)
        assert result == expected

        # Should have queried database with canonical path
        mock_db_provider.find_base_for_path.assert_called_once_with(expected)

    def test_resolve_relative_path_with_cwd_in_indexed_base(self, resolver):
        """Test resolving relative path when CWD is in indexed base."""
        query_path = "src/main.py"
        cwd = "/home/user/project1/subdir"

        result = resolver.resolve_path(query_path, current_working_dir=cwd)

        # Should resolve to base + relative path (not CWD + relative)
        # PathResolver finds base for CWD, then joins relative path
        assert result == "/home/user/project1/src/main.py"

    def test_resolve_relative_path_without_cwd_raises_error(self, resolver):
        """Test that relative path without CWD raises ValueError."""
        query_path = "src/main.py"

        with pytest.raises(
            ValueError,
            match="Cannot resolve relative path.*without current_working_dir",
        ):
            resolver.resolve_path(query_path, current_working_dir=None)

    def test_resolve_relative_path_when_cwd_not_in_indexed_base(self, resolver):
        """Test resolving relative path when CWD is not in any indexed base."""
        query_path = "src/main.py"
        cwd = "/home/user/other_project"

        result = resolver.resolve_path(query_path, current_working_dir=cwd)

        # Should fallback to CWD + relative path
        assert result == "/home/user/other_project/src/main.py"

    def test_cache_hit_for_repeated_lookups(self, resolver, mock_db_provider):
        """Test that repeated lookups use cache."""
        path = "/home/user/project1/src/main.py"

        # First call
        result1 = resolver.resolve_path(path)

        # Second call (should use cache)
        result2 = resolver.resolve_path(path)

        assert result1 == result2

        # Database should only be called once (cached second time)
        assert mock_db_provider.find_base_for_path.call_count == 1

    def test_clear_cache_invalidates_cached_entries(self, resolver, mock_db_provider):
        """Test that clear_cache() invalidates cache."""
        path = "/home/user/project1/src/main.py"

        # First call (populates cache)
        resolver.resolve_path(path)

        # Clear cache
        resolver.clear_cache()

        # Second call (should query DB again)
        resolver.resolve_path(path)

        # Database should be called twice (cache was cleared)
        assert mock_db_provider.find_base_for_path.call_count == 2

    def test_get_indexed_roots_returns_all_roots(self, resolver, mock_db_provider):
        """Test get_indexed_roots() returns all indexed bases."""
        roots = resolver.get_indexed_roots()

        assert len(roots) == 2
        assert roots[0]["base_directory"] == "/home/user/project1"
        assert roots[1]["base_directory"] == "/home/user/project2"

    def test_get_indexed_roots_handles_legacy_provider(
        self, resolver, mock_db_provider
    ):
        """Test get_indexed_roots() gracefully handles provider without support."""
        # Remove method to simulate legacy provider
        delattr(mock_db_provider, "get_indexed_roots")

        roots = resolver.get_indexed_roots()

        # Should return empty list without error
        assert roots == []

    def test_resolve_path_handles_provider_without_find_base(
        self, resolver, mock_db_provider
    ):
        """Test resolve_path() handles provider without find_base_for_path."""
        import os

        # Remove method to simulate legacy provider
        delattr(mock_db_provider, "find_base_for_path")

        path = "/home/user/project1/src/main.py"
        result = resolver.resolve_path(path)

        # Should return canonicalized path (fallback behavior)
        # Path is still resolved for consistency
        assert result == os.path.realpath(path)

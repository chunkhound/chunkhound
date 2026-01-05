"""Tests for multi-path search functionality.

These tests verify that the path list resolution works correctly
for multi-project searches in global database mode.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chunkhound.mcp_server.tools import _resolve_paths


class MockProject:
    """Mock project for testing."""

    def __init__(self, name: str, base_dir: str, tags: list[str] | None = None):
        self.project_name = name
        self.base_directory = Path(base_dir)
        self.tags = tags or []


class MockProjectRegistry:
    """Mock project registry for testing."""

    def __init__(self, projects: list[MockProject]):
        self._projects = {p.project_name: p for p in projects}

    def get_project(self, name: str) -> MockProject | None:
        return self._projects.get(name)

    def get_all_projects(self) -> list[MockProject]:
        return list(self._projects.values())

    def get_projects_by_tags(self, tags: list[str]) -> list[MockProject]:
        """Return projects that have ALL specified tags."""
        result = []
        for p in self._projects.values():
            if all(t in p.tags for t in tags):
                result.append(p)
        return result


class TestResolvePaths:
    """Tests for the _resolve_paths function."""

    def test_no_paths_no_context_returns_unchanged(self) -> None:
        """When no paths and no context, arguments unchanged."""
        args: dict[str, Any] = {"pattern": "test"}
        result = _resolve_paths(args, None, None)
        assert result == {"pattern": "test"}

    def test_no_paths_with_context_adds_project_path(self) -> None:
        """When no paths but client context, scope to client project."""
        args: dict[str, Any] = {"pattern": "test"}
        context = {"project": "/home/user/my-project"}

        result = _resolve_paths(args, context, None)

        assert result["path"] == ["/home/user/my-project"]
        assert result["pattern"] == "test"

    def test_absolute_path_unchanged(self) -> None:
        """Absolute paths are used as-is."""
        args: dict[str, Any] = {
            "path": ["/home/user/project-a", "/home/user/project-b"]
        }

        result = _resolve_paths(args, None, None)

        assert result["path"] == ["/home/user/project-a", "/home/user/project-b"]

    def test_relative_path_resolved_against_context(self) -> None:
        """Relative paths are resolved against client context."""
        args: dict[str, Any] = {"path": ["src/", "tests/"]}
        context = {"project": "/home/user/my-project"}

        result = _resolve_paths(args, context, None)

        # Note: Path joining normalizes trailing slashes
        assert result["path"] == [
            "/home/user/my-project/src",
            "/home/user/my-project/tests",
        ]

    def test_mixed_paths_resolved_correctly(self) -> None:
        """Mix of absolute and relative paths handled correctly."""
        args: dict[str, Any] = {"path": ["/other/project", "src/"]}
        context = {"project": "/home/user/my-project"}

        result = _resolve_paths(args, context, None)

        assert result["path"] == [
            "/other/project",
            "/home/user/my-project/src",
        ]

    def test_relative_path_no_context_unchanged(self) -> None:
        """Relative paths without context are used as-is."""
        args: dict[str, Any] = {"path": ["src/"]}

        result = _resolve_paths(args, None, None)

        assert result["path"] == ["src/"]

    def test_tags_filter_projects(self) -> None:
        """Tags filter to matching projects."""
        projects = [
            MockProject("frontend", "/home/user/frontend", tags=["web", "typescript"]),
            MockProject("backend", "/home/user/backend", tags=["api", "python"]),
            MockProject("shared", "/home/user/shared", tags=["python", "web"]),
        ]
        registry = MockProjectRegistry(projects)

        args: dict[str, Any] = {"tags": ["python"]}
        result = _resolve_paths(args, None, registry)

        # Should include backend and shared (both have "python" tag)
        assert set(result["path"]) == {"/home/user/backend", "/home/user/shared"}
        assert "tags" not in result  # tags should be removed after processing

    def test_tags_with_multiple_required(self) -> None:
        """Multiple tags require ALL to match (AND logic)."""
        projects = [
            MockProject("frontend", "/home/user/frontend", tags=["web", "typescript"]),
            MockProject("backend", "/home/user/backend", tags=["api", "python"]),
            MockProject("shared", "/home/user/shared", tags=["python", "web"]),
        ]
        registry = MockProjectRegistry(projects)

        args: dict[str, Any] = {"tags": ["python", "web"]}
        result = _resolve_paths(args, None, registry)

        # Only shared has both python AND web
        assert result["path"] == ["/home/user/shared"]

    def test_tags_with_relative_paths(self) -> None:
        """Relative paths are applied to each tagged project."""
        projects = [
            MockProject("project-a", "/home/user/project-a", tags=["work"]),
            MockProject("project-b", "/home/user/project-b", tags=["work"]),
        ]
        registry = MockProjectRegistry(projects)

        args: dict[str, Any] = {
            "path": ["src/", "tests/"],
            "tags": ["work"],
        }
        result = _resolve_paths(args, None, registry)

        # src/ and tests/ applied to each work project
        expected = {
            "/home/user/project-a/src",
            "/home/user/project-a/tests",
            "/home/user/project-b/src",
            "/home/user/project-b/tests",
        }
        assert set(result["path"]) == expected

    def test_tags_with_absolute_path_union(self) -> None:
        """Absolute paths are added directly (union with tagged projects)."""
        projects = [
            MockProject("backend", "/home/user/backend", tags=["python"]),
        ]
        registry = MockProjectRegistry(projects)

        args: dict[str, Any] = {
            "path": ["/other/library/"],
            "tags": ["python"],
        }
        result = _resolve_paths(args, None, registry)

        # Tagged project root + absolute path
        assert set(result["path"]) == {"/home/user/backend", "/other/library/"}

    def test_tags_with_mixed_paths(self) -> None:
        """Mixed relative and absolute paths handled correctly with tags."""
        projects = [
            MockProject("project-a", "/home/user/project-a", tags=["work"]),
            MockProject("project-b", "/home/user/project-b", tags=["work"]),
        ]
        registry = MockProjectRegistry(projects)

        args: dict[str, Any] = {
            "path": ["src/", "/external/lib/"],
            "tags": ["work"],
        }
        result = _resolve_paths(args, None, registry)

        # Relative "src/" applied to each work project, absolute added directly
        expected = {
            "/home/user/project-a/src",
            "/home/user/project-b/src",
            "/external/lib/",
        }
        assert set(result["path"]) == expected

    def test_empty_path_list_scopes_to_client(self) -> None:
        """Empty path list scopes to client project (empty list is falsy)."""
        args: dict[str, Any] = {"path": [], "pattern": "test"}
        context = {"project": "/home/user/my-project"}

        result = _resolve_paths(args, context, None)

        # Empty list is falsy in Python, so falls through to client project
        assert result["path"] == ["/home/user/my-project"]

    def test_tags_no_match_returns_empty_path(self) -> None:
        """When tags are specified but no projects match, return empty path list."""
        projects = [
            MockProject("frontend", "/home/user/frontend", tags=["web"]),
            MockProject("backend", "/home/user/backend", tags=["api"]),
        ]
        registry = MockProjectRegistry(projects)

        # Search for a tag that doesn't exist
        args: dict[str, Any] = {"tags": ["nonexistent"], "pattern": "test"}
        context = {"project": "/home/user/some-project"}  # Should NOT be used
        result = _resolve_paths(args, context, registry)

        # Should return empty path list (no matches), not fall through to client project
        assert result["path"] == []
        assert "tags" not in result  # tags should be removed


class TestPathListIntegration:
    """Integration tests for path list in search tools."""

    @pytest.mark.asyncio
    async def test_search_regex_accepts_path_list(self) -> None:
        """search_regex accepts path as list of strings."""
        from chunkhound.mcp_server.tools import execute_tool

        services = MagicMock()
        services.provider.is_connected = True
        services.search_service.search_regex.return_value = (
            [],
            {"offset": 0, "page_size": 10, "has_more": False},
        )

        result = await execute_tool(
            tool_name="search_regex",
            services=services,
            embedding_manager=None,
            arguments={
                "pattern": "test",
                "path": ["/project-a/src/", "/project-b/src/"],
            },
        )

        assert "results" in result
        # Verify the search was called with path_prefixes for multiple paths
        call_args = services.search_service.search_regex.call_args
        assert call_args.kwargs["path_prefixes"] == [
            "/project-a/src/",
            "/project-b/src/",
        ]

    @pytest.mark.asyncio
    async def test_search_regex_single_path(self) -> None:
        """search_regex with single path uses path_filter."""
        from chunkhound.mcp_server.tools import execute_tool

        services = MagicMock()
        services.provider.is_connected = True
        services.search_service.search_regex.return_value = (
            [],
            {"offset": 0, "page_size": 10, "has_more": False},
        )

        result = await execute_tool(
            tool_name="search_regex",
            services=services,
            embedding_manager=None,
            arguments={
                "pattern": "test",
                "path": ["/project-a/src/"],
            },
        )

        assert "results" in result
        # Single path should use path_filter, not path_prefixes
        call_args = services.search_service.search_regex.call_args
        assert call_args.kwargs["path_filter"] == "/project-a/src/"
        assert call_args.kwargs["path_prefixes"] is None

"""Unit tests for SearchContext service.

Tests search context creation from tool arguments and scope resolution.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chunkhound.services.project_registry import ProjectInfo
from chunkhound.services.search_context import (
    SearchContext,
    SearchScope,
    resolve_search_path_filter,
)


@pytest.fixture
def mock_registry():
    """Create a mock project registry."""
    registry = MagicMock()
    registry.list_projects.return_value = []
    registry.get_project.return_value = None
    registry.find_project_for_path.return_value = None
    registry.get_projects_by_tags.return_value = []
    return registry


@pytest.fixture
def sample_project():
    """Create a sample ProjectInfo."""
    return ProjectInfo(
        base_directory=Path("/home/user/myproject"),
        project_name="myproject",
        tags=["backend", "python"],
    )


@pytest.fixture
def sample_project_b():
    """Create another sample ProjectInfo."""
    return ProjectInfo(
        base_directory=Path("/home/user/project-b"),
        project_name="project-b",
        tags=["frontend", "typescript"],
    )


class TestSearchContextFromToolArguments:
    """Tests for SearchContext.from_tool_arguments()."""

    def test_from_tool_arguments_minimal(self):
        """Test creating context with minimal arguments."""
        arguments = {"query": "authentication"}

        context = SearchContext.from_tool_arguments(arguments)

        assert context.current_project_path is None
        assert context.target_projects is None
        assert context.target_paths is None
        assert context.search_all is False
        assert context.path_filter is None
        assert context.tags is None

    def test_from_tool_arguments_with_projects(self):
        """Test creating context with explicit project list."""
        arguments = {
            "query": "authentication",
            "projects": ["project-a", "project-b"],
        }

        context = SearchContext.from_tool_arguments(arguments)

        assert context.target_projects == ["project-a", "project-b"]
        assert context.search_all is False

    def test_from_tool_arguments_with_single_project_string(self):
        """Test that single project string is converted to list."""
        arguments = {
            "query": "authentication",
            "projects": "single-project",
        }

        context = SearchContext.from_tool_arguments(arguments)

        assert context.target_projects == ["single-project"]

    def test_from_tool_arguments_with_tags(self):
        """Test creating context with tag filtering."""
        arguments = {
            "query": "authentication",
            "tags": ["backend", "python"],
        }

        context = SearchContext.from_tool_arguments(arguments)

        assert context.tags == ["backend", "python"]

    def test_from_tool_arguments_with_single_tag_string(self):
        """Test that single tag string is converted to list."""
        arguments = {
            "query": "authentication",
            "tags": "backend",
        }

        context = SearchContext.from_tool_arguments(arguments)

        assert context.tags == ["backend"]

    def test_from_tool_arguments_with_absolute_path(self):
        """Test creating context with absolute path."""
        arguments = {
            "query": "authentication",
            "path": "/home/user/project/src/",
        }

        context = SearchContext.from_tool_arguments(arguments)

        assert context.target_paths == [Path("/home/user/project/src/")]
        assert context.path_filter is None  # Not a relative path

    def test_from_tool_arguments_with_relative_path(self):
        """Test creating context with relative path filter."""
        arguments = {
            "query": "authentication",
            "path": "src/auth/",
        }

        context = SearchContext.from_tool_arguments(arguments)

        assert context.target_paths is None
        # Path() normalizes away trailing slash
        assert context.path_filter == "src/auth"

    def test_from_tool_arguments_with_search_all(self):
        """Test creating context with search_all flag."""
        arguments = {
            "query": "authentication",
            "search_all": True,
        }

        context = SearchContext.from_tool_arguments(arguments)

        assert context.search_all is True

    def test_from_tool_arguments_with_client_context(self):
        """Test extracting current project from client context."""
        arguments = {"query": "authentication"}
        client_context = {"project": "/home/user/current-project"}

        context = SearchContext.from_tool_arguments(arguments, client_context)

        assert context.current_project_path == Path("/home/user/current-project")

    def test_from_tool_arguments_full(self):
        """Test creating context with all options."""
        arguments = {
            "query": "authentication",
            "path": "src/",
            "projects": ["project-a"],
            "search_all": False,
            "tags": ["backend"],
        }
        client_context = {"project": "/home/user/my-project"}

        context = SearchContext.from_tool_arguments(arguments, client_context)

        assert context.current_project_path == Path("/home/user/my-project")
        assert context.target_projects == ["project-a"]
        # Path() normalizes away trailing slash
        assert context.path_filter == "src"
        assert context.tags == ["backend"]


class TestSearchContextToDict:
    """Tests for SearchContext.to_dict()."""

    def test_to_dict_minimal(self):
        """Test serializing minimal context."""
        context = SearchContext()
        result = context.to_dict()

        assert result["current_project_path"] is None
        assert result["target_projects"] is None
        assert result["target_paths"] is None
        assert result["search_all"] is False
        assert result["tags"] is None

    def test_to_dict_full(self):
        """Test serializing full context."""
        context = SearchContext(
            current_project_path=Path("/home/user/project"),
            target_projects=["a", "b"],
            target_paths=[Path("/home/user/other")],
            path_filter="src/",
            search_all=False,
            tags=["backend"],
        )
        result = context.to_dict()

        assert result["current_project_path"] == "/home/user/project"
        assert result["target_projects"] == ["a", "b"]
        assert result["target_paths"] == ["/home/user/other"]
        assert result["path_filter"] == "src/"
        assert result["tags"] == ["backend"]


class TestSearchScopeResolve:
    """Tests for SearchContext.resolve_scope()."""

    def test_resolve_scope_no_registry(self):
        """Test that no registry returns empty scope."""
        context = SearchContext(search_all=True)

        scope = context.resolve_scope(None)

        assert scope.projects == []
        assert scope.is_all_projects is False

    def test_resolve_scope_with_tags(self, mock_registry, sample_project):
        """Test resolving scope with tag filtering."""
        mock_registry.get_projects_by_tags.return_value = [sample_project]
        context = SearchContext(tags=["backend", "python"])

        scope = context.resolve_scope(mock_registry)

        assert len(scope.projects) == 1
        assert scope.projects[0].project_name == "myproject"
        mock_registry.get_projects_by_tags.assert_called_with(
            ["backend", "python"], match_all=True
        )

    def test_resolve_scope_with_tags_no_match(self, mock_registry):
        """Test resolving scope with tags that match nothing."""
        mock_registry.get_projects_by_tags.return_value = []
        context = SearchContext(tags=["nonexistent"])

        scope = context.resolve_scope(mock_registry)

        assert scope.projects == []

    def test_resolve_scope_with_search_all(
        self, mock_registry, sample_project, sample_project_b
    ):
        """Test resolving scope with search_all flag."""
        mock_registry.list_projects.return_value = [sample_project, sample_project_b]
        context = SearchContext(search_all=True)

        scope = context.resolve_scope(mock_registry)

        assert len(scope.projects) == 2
        assert scope.is_all_projects is True
        mock_registry.list_projects.assert_called_once()

    def test_resolve_scope_with_search_all_and_tags(
        self, mock_registry, sample_project, sample_project_b
    ):
        """Test search_all combined with tag filtering."""
        # When tags are specified, get_projects_by_tags is called first
        mock_registry.get_projects_by_tags.return_value = [sample_project]
        context = SearchContext(search_all=True, tags=["backend"])

        scope = context.resolve_scope(mock_registry)

        # Only sample_project has "backend" tag
        assert len(scope.projects) == 1
        assert scope.projects[0].project_name == "myproject"
        assert scope.is_all_projects is True
        mock_registry.get_projects_by_tags.assert_called_with(
            ["backend"], match_all=True
        )

    def test_resolve_scope_with_projects(
        self, mock_registry, sample_project, sample_project_b
    ):
        """Test resolving scope with explicit project names."""
        mock_registry.get_project.side_effect = lambda name: (
            sample_project
            if name == "myproject"
            else sample_project_b
            if name == "project-b"
            else None
        )
        context = SearchContext(target_projects=["myproject", "project-b"])

        scope = context.resolve_scope(mock_registry)

        assert len(scope.projects) == 2
        assert scope.is_all_projects is False

    def test_resolve_scope_with_projects_not_found(self, mock_registry):
        """Test resolving scope with non-existent project names."""
        mock_registry.get_project.return_value = None
        context = SearchContext(target_projects=["nonexistent"])

        scope = context.resolve_scope(mock_registry)

        assert scope.projects == []

    def test_resolve_scope_with_projects_and_tags(self, mock_registry, sample_project):
        """Test that projects are filtered by tags."""
        mock_registry.get_project.return_value = sample_project
        # sample_project has tags ["backend", "python"]
        context = SearchContext(target_projects=["myproject"], tags=["frontend"])

        scope = context.resolve_scope(mock_registry)

        # myproject doesn't have "frontend" tag
        assert scope.projects == []

    def test_resolve_scope_with_absolute_path(self, mock_registry, sample_project):
        """Test resolving scope with absolute path."""
        mock_registry.find_project_for_path.return_value = sample_project
        target_path = Path("/home/user/myproject/src/auth/")
        context = SearchContext(target_paths=[target_path])

        scope = context.resolve_scope(mock_registry)

        assert len(scope.projects) == 1
        assert scope.projects[0].project_name == "myproject"
        # Path filter should be relative portion
        assert scope.path_filter == "src/auth"

    def test_resolve_scope_with_absolute_path_root(self, mock_registry, sample_project):
        """Test resolving scope with absolute path that is project root."""
        mock_registry.find_project_for_path.return_value = sample_project
        target_path = Path("/home/user/myproject")
        context = SearchContext(target_paths=[target_path])

        scope = context.resolve_scope(mock_registry)

        assert len(scope.projects) == 1
        # Path filter should be None (not "." for root)
        assert scope.path_filter is None

    def test_resolve_scope_with_current_project(self, mock_registry, sample_project):
        """Test resolving scope with current project context."""
        mock_registry.find_project_for_path.return_value = sample_project
        context = SearchContext(
            current_project_path=Path("/home/user/myproject"),
            path_filter="src/",
        )

        scope = context.resolve_scope(mock_registry)

        assert len(scope.projects) == 1
        assert scope.projects[0].project_name == "myproject"
        assert scope.path_filter == "src/"

    def test_resolve_scope_with_current_project_not_indexed(self, mock_registry):
        """Test resolving scope when current project is not indexed."""
        mock_registry.find_project_for_path.return_value = None
        context = SearchContext(
            current_project_path=Path("/home/user/unindexed-project"),
        )

        scope = context.resolve_scope(mock_registry)

        assert scope.projects == []

    def test_resolve_scope_empty(self, mock_registry):
        """Test resolving scope with no constraints returns empty."""
        context = SearchContext()

        scope = context.resolve_scope(mock_registry)

        assert scope.projects == []
        assert scope.path_filter is None

    def test_resolve_scope_tags_combined_with_projects(
        self, mock_registry, sample_project, sample_project_b
    ):
        """Test that tags filter explicit project list."""
        # When tags are specified, get_projects_by_tags is called first
        # It returns projects matching tags, then target_projects filters further
        mock_registry.get_projects_by_tags.return_value = [sample_project]
        mock_registry.get_project.side_effect = lambda name: (
            sample_project
            if name == "myproject"
            else sample_project_b
            if name == "project-b"
            else None
        )
        # sample_project has tags ["backend", "python"], sample_project_b has ["frontend", "typescript"]
        context = SearchContext(
            target_projects=["myproject", "project-b"],
            tags=["python"],
        )

        scope = context.resolve_scope(mock_registry)

        # With tags set, get_projects_by_tags is called first
        # Since search_all=False and target_projects is set, it goes to priority 3
        # But priority 1 tags path intercepts when tags are set
        # Looking at the code flow: tags are set -> get_projects_by_tags called
        # Then check: if search_all or no other scope -> return tagged
        # But target_projects is set, so it continues to priority 3 with tag filtering
        assert len(scope.projects) == 1
        assert scope.projects[0].project_name == "myproject"

    def test_resolve_scope_priority_tags_first(self, mock_registry, sample_project):
        """Test that tags take priority when specified alone."""
        mock_registry.get_projects_by_tags.return_value = [sample_project]
        context = SearchContext(tags=["backend"])

        scope = context.resolve_scope(mock_registry)

        assert len(scope.projects) == 1
        mock_registry.get_projects_by_tags.assert_called_once()
        mock_registry.list_projects.assert_not_called()


class TestSearchScope:
    """Tests for SearchScope dataclass."""

    def test_project_paths(self, sample_project, sample_project_b):
        """Test getting list of project paths."""
        scope = SearchScope(projects=[sample_project, sample_project_b])

        paths = scope.project_paths

        assert len(paths) == 2
        assert paths[0] == Path("/home/user/myproject")
        assert paths[1] == Path("/home/user/project-b")

    def test_project_names(self, sample_project, sample_project_b):
        """Test getting list of project names."""
        scope = SearchScope(projects=[sample_project, sample_project_b])

        names = scope.project_names

        assert names == ["myproject", "project-b"]

    def test_empty_scope(self):
        """Test empty scope properties."""
        scope = SearchScope()

        assert scope.project_paths == []
        assert scope.project_names == []
        assert scope.is_all_projects is False


class TestResolveSearchPathFilter:
    """Tests for resolve_search_path_filter helper function."""

    def test_all_projects_with_filter(self, sample_project):
        """Test path filter for all projects search."""
        context = SearchContext(path_filter="src/")
        scope = SearchScope(
            projects=[sample_project], is_all_projects=True, path_filter="src/"
        )

        result = resolve_search_path_filter(context, scope)

        assert result == "src/"

    def test_all_projects_no_filter(self, sample_project):
        """Test no filter for all projects search."""
        context = SearchContext()
        scope = SearchScope(projects=[sample_project], is_all_projects=True)

        result = resolve_search_path_filter(context, scope)

        assert result is None

    def test_empty_scope(self):
        """Test empty scope uses context path filter."""
        context = SearchContext(path_filter="src/")
        scope = SearchScope()

        result = resolve_search_path_filter(context, scope)

        assert result == "src/"

    def test_single_project_with_filter(self, sample_project):
        """Test single project with path filter."""
        context = SearchContext(path_filter="src")
        scope = SearchScope(projects=[sample_project], path_filter="src")

        result = resolve_search_path_filter(context, scope)

        # Should combine project path with filter (no trailing slash)
        assert result == "/home/user/myproject/src"

    def test_single_project_no_filter(self, sample_project):
        """Test single project without path filter."""
        context = SearchContext()
        scope = SearchScope(projects=[sample_project])

        result = resolve_search_path_filter(context, scope)

        # Should use project path as filter
        assert result == "/home/user/myproject"

    def test_multiple_projects(self, sample_project, sample_project_b):
        """Test multiple projects returns None (caller handles)."""
        context = SearchContext()
        scope = SearchScope(projects=[sample_project, sample_project_b])

        result = resolve_search_path_filter(context, scope)

        # Multiple projects need special handling
        assert result is None

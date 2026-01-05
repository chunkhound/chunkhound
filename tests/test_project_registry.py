"""Unit tests for ProjectRegistry service.

Tests project registration, lookup, unregistration, and tag management.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chunkhound.services.project_registry import ProjectInfo, ProjectRegistry


@pytest.fixture
def mock_db_provider():
    """Create a mock database provider with multi-repo methods."""
    provider = MagicMock()
    provider.supports_multi_repo = True  # Enable multi-repo by default
    provider.get_indexed_roots.return_value = []
    provider.register_base_directory.return_value = None
    provider.remove_base_directory.return_value = None
    provider.update_indexed_root_stats.return_value = None
    provider.update_indexed_root_watcher_status.return_value = None
    provider.update_indexed_root_tags.return_value = None
    provider.add_indexed_root_tags.return_value = None
    provider.remove_indexed_root_tags.return_value = None
    provider.get_indexed_roots_by_tags.return_value = []
    provider.get_all_tags.return_value = []
    return provider


@pytest.fixture
def registry(mock_db_provider):
    """Create a ProjectRegistry with mocked database."""
    return ProjectRegistry(mock_db_provider)


@pytest.fixture
def temp_project_path(tmp_path):
    """Create a temporary project directory."""
    project_dir = tmp_path / "test-project"
    project_dir.mkdir()
    return project_dir


@pytest.fixture
def sample_project_row():
    """Sample database row for a project."""
    return {
        "base_directory": "/home/user/myproject",
        "project_name": "myproject",
        "indexed_at": datetime(2024, 1, 15, 10, 30, 0),
        "updated_at": datetime(2024, 1, 16, 14, 0, 0),
        "file_count": 150,
        "watcher_active": True,
        "last_error": None,
        "error_count": 0,
        "config_snapshot": {"provider": "openai", "model": "text-embedding-3-small"},
        "tags": ["backend", "python"],
    }


class TestProjectInfo:
    """Tests for ProjectInfo dataclass."""

    def test_from_db_row(self, sample_project_row):
        """Test creating ProjectInfo from database row."""
        project = ProjectInfo.from_db_row(sample_project_row)

        assert project.base_directory == Path("/home/user/myproject")
        assert project.project_name == "myproject"
        assert project.indexed_at == datetime(2024, 1, 15, 10, 30, 0)
        assert project.updated_at == datetime(2024, 1, 16, 14, 0, 0)
        assert project.file_count == 150
        assert project.watcher_active is True
        assert project.last_error is None
        assert project.error_count == 0
        assert project.config_snapshot == {
            "provider": "openai",
            "model": "text-embedding-3-small",
        }
        assert project.tags == ["backend", "python"]

    def test_from_db_row_minimal(self):
        """Test creating ProjectInfo from minimal database row."""
        row = {"base_directory": "/tmp/project"}
        project = ProjectInfo.from_db_row(row)

        assert project.base_directory == Path("/tmp/project")
        assert project.project_name == "project"  # Derived from path
        assert project.indexed_at is None
        assert project.file_count == 0
        assert project.tags == []

    def test_to_dict(self, sample_project_row):
        """Test converting ProjectInfo to dictionary."""
        project = ProjectInfo.from_db_row(sample_project_row)
        result = project.to_dict()

        assert result["base_directory"] == "/home/user/myproject"
        assert result["project_name"] == "myproject"
        assert result["indexed_at"] == "2024-01-15T10:30:00"
        assert result["file_count"] == 150
        assert result["tags"] == ["backend", "python"]

    def test_to_dict_no_dates(self):
        """Test to_dict with None dates."""
        project = ProjectInfo(
            base_directory=Path("/tmp/test"),
            project_name="test",
        )
        result = project.to_dict()

        assert result["indexed_at"] is None
        assert result["updated_at"] is None


class TestProjectRegistration:
    """Tests for project registration."""

    def test_register_project_creates_entry(
        self, registry, mock_db_provider, temp_project_path
    ):
        """Test registering a new project creates a database entry."""
        project = registry.register_project(temp_project_path)

        mock_db_provider.register_base_directory.assert_called_once()
        call_args = mock_db_provider.register_base_directory.call_args
        assert call_args[0][0] == str(temp_project_path)
        assert call_args[0][1] == temp_project_path.name  # Default name

    def test_register_project_with_custom_name(
        self, registry, mock_db_provider, temp_project_path
    ):
        """Test registering with a custom project name."""
        registry.register_project(temp_project_path, name="my-custom-name")

        call_args = mock_db_provider.register_base_directory.call_args
        assert call_args[0][1] == "my-custom-name"

    def test_register_project_with_config(
        self, registry, mock_db_provider, temp_project_path
    ):
        """Test registering with configuration snapshot."""
        config = {"provider": "openai", "model": "text-embedding-3-small"}
        registry.register_project(temp_project_path, config=config)

        call_args = mock_db_provider.register_base_directory.call_args
        assert call_args[0][2] == config

    def test_register_project_non_existent_relative_path(self, registry, tmp_path):
        """Test that non-existent relative paths raise ValueError."""
        # A relative path that doesn't exist when resolved
        fake_relative = Path("nonexistent_dir_xyz123")
        with pytest.raises(ValueError, match="does not exist"):
            registry.register_project(fake_relative)

    def test_register_project_non_existent_path(self, registry, tmp_path):
        """Test that non-existent paths raise ValueError."""
        fake_path = tmp_path / "does-not-exist"
        with pytest.raises(ValueError, match="does not exist"):
            registry.register_project(fake_path)

    def test_register_project_file_path(self, registry, tmp_path):
        """Test that file paths (not directories) raise ValueError."""
        file_path = tmp_path / "test.txt"
        file_path.touch()
        with pytest.raises(ValueError, match="must be a directory"):
            registry.register_project(file_path)

    def test_register_subfolder_of_existing_project_rejected(
        self, registry, mock_db_provider, tmp_path
    ):
        """Test that registering a subfolder of an existing project is rejected."""
        # Create parent and child directories
        parent_dir = tmp_path / "project"
        parent_dir.mkdir()
        child_dir = parent_dir / "subdir"
        child_dir.mkdir()

        # Mock existing project in cache
        mock_db_provider.get_indexed_roots.return_value = [
            {
                "base_directory": str(parent_dir),
                "project_name": "project",
                "file_count": 10,
                "indexed_at": None,
                "updated_at": None,
                "watcher_active": False,
                "tags": [],
            }
        ]

        # Attempt to register subfolder should fail
        with pytest.raises(ValueError, match="subfolder of existing project"):
            registry.register_project(child_dir)

    def test_register_parent_of_existing_project_rejected(
        self, registry, mock_db_provider, tmp_path
    ):
        """Test that registering a parent of an existing project is rejected."""
        # Create parent and child directories
        parent_dir = tmp_path / "project"
        parent_dir.mkdir()
        child_dir = parent_dir / "subdir"
        child_dir.mkdir()

        # Mock existing project at child path
        mock_db_provider.get_indexed_roots.return_value = [
            {
                "base_directory": str(child_dir),
                "project_name": "subdir",
                "file_count": 10,
                "indexed_at": None,
                "updated_at": None,
                "watcher_active": False,
                "tags": [],
            }
        ]

        # Attempt to register parent should fail
        with pytest.raises(ValueError, match="is a subfolder"):
            registry.register_project(parent_dir)

    def test_reregister_same_path_allowed(
        self, registry, mock_db_provider, tmp_path
    ):
        """Test that re-registering the exact same path is allowed (update case)."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Mock existing project at same path
        mock_db_provider.get_indexed_roots.return_value = [
            {
                "base_directory": str(project_dir),
                "project_name": "project",
                "file_count": 10,
                "indexed_at": None,
                "updated_at": None,
                "watcher_active": False,
                "tags": [],
            }
        ]

        # Re-registration of same path should succeed
        project = registry.register_project(project_dir)
        mock_db_provider.register_base_directory.assert_called()


class TestProjectUnregistration:
    """Tests for project unregistration."""

    def test_unregister_project_by_name(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test unregistering a project by name."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        result = registry.unregister_project("myproject")

        assert result is True
        mock_db_provider.remove_base_directory.assert_called_once_with(
            "/home/user/myproject", cascade=False
        )

    def test_unregister_project_by_path(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test unregistering a project by path."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        result = registry.unregister_project("/home/user/myproject")

        assert result is True
        mock_db_provider.remove_base_directory.assert_called_once()

    def test_unregister_with_cascade(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test unregistering with cascade=True deletes files/chunks."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        result = registry.unregister_project("myproject", cascade=True)

        assert result is True
        mock_db_provider.remove_base_directory.assert_called_once_with(
            "/home/user/myproject", cascade=True
        )

    def test_unregister_project_not_found(self, registry, mock_db_provider):
        """Test unregistering a non-existent project returns False."""
        mock_db_provider.get_indexed_roots.return_value = []

        result = registry.unregister_project("nonexistent")

        assert result is False
        mock_db_provider.remove_base_directory.assert_not_called()

    def test_unregister_stops_watcher(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test unregistering stops the file watcher."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]
        mock_watcher = MagicMock()
        registry.set_watcher_manager(mock_watcher)

        registry.unregister_project("myproject")

        mock_watcher.stop_watcher.assert_called_once_with(Path("/home/user/myproject"))


class TestProjectLookup:
    """Tests for project lookup operations."""

    def test_get_project_by_name(self, registry, mock_db_provider, sample_project_row):
        """Test getting a project by its name."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        project = registry.get_project("myproject")

        assert project is not None
        assert project.project_name == "myproject"
        assert project.base_directory == Path("/home/user/myproject")

    def test_get_project_by_path(self, registry, mock_db_provider, sample_project_row):
        """Test getting a project by its path."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        project = registry.get_project("/home/user/myproject")

        assert project is not None
        assert project.project_name == "myproject"

    def test_get_project_by_directory_name(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test getting a project by its directory name (fallback)."""
        sample_project_row["project_name"] = "different-name"
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        project = registry.get_project("myproject")

        assert project is not None
        assert project.base_directory.name == "myproject"

    def test_get_project_not_found(self, registry, mock_db_provider):
        """Test getting a non-existent project returns None."""
        mock_db_provider.get_indexed_roots.return_value = []

        project = registry.get_project("nonexistent")

        assert project is None

    def test_list_projects_empty(self, registry, mock_db_provider):
        """Test listing projects when none are registered."""
        mock_db_provider.get_indexed_roots.return_value = []

        projects = registry.list_projects()

        assert projects == []

    def test_list_projects_multiple(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test listing multiple projects."""
        project2 = {
            "base_directory": "/home/user/other-project",
            "project_name": "other-project",
            "file_count": 50,
            "tags": ["frontend"],
        }
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row, project2]

        projects = registry.list_projects()

        assert len(projects) == 2
        assert projects[0].project_name == "myproject"
        assert projects[1].project_name == "other-project"


class TestFindProjectForPath:
    """Tests for finding which project contains a path."""

    def test_find_project_for_path_exact_match(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test finding project when path matches exactly."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        project = registry.find_project_for_path(Path("/home/user/myproject"))

        assert project is not None
        assert project.project_name == "myproject"

    def test_find_project_for_path_nested(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test finding project for a nested file path."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        project = registry.find_project_for_path(
            Path("/home/user/myproject/src/main.py")
        )

        assert project is not None
        assert project.project_name == "myproject"

    def test_find_project_for_path_not_found(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test finding project for a path not in any project."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        project = registry.find_project_for_path(
            Path("/home/user/different-project/main.py")
        )

        assert project is None

    def test_find_project_for_path_longest_match(self, registry, mock_db_provider):
        """Test that longest prefix match is used for nested projects."""
        parent_project = {
            "base_directory": "/home/user/project",
            "project_name": "parent",
        }
        nested_project = {
            "base_directory": "/home/user/project/subproject",
            "project_name": "nested",
        }
        mock_db_provider.get_indexed_roots.return_value = [
            parent_project,
            nested_project,
        ]

        # Path in nested project should match nested, not parent
        project = registry.find_project_for_path(
            Path("/home/user/project/subproject/src/main.py")
        )

        assert project is not None
        assert project.project_name == "nested"


class TestTagManagement:
    """Tests for project tag management."""

    def test_get_projects_by_tags_match_all(self, registry, mock_db_provider):
        """Test getting projects that have ALL specified tags (AND logic)."""
        matching_rows = [
            {
                "base_directory": "/path/a",
                "project_name": "a",
                "tags": ["backend", "python"],
            },
        ]
        mock_db_provider.get_indexed_roots_by_tags.return_value = matching_rows

        projects = registry.get_projects_by_tags(["backend", "python"], match_all=True)

        assert len(projects) == 1
        mock_db_provider.get_indexed_roots_by_tags.assert_called_with(
            ["backend", "python"], True
        )

    def test_get_projects_by_tags_match_any(self, registry, mock_db_provider):
        """Test getting projects that have ANY specified tag (OR logic)."""
        matching_rows = [
            {"base_directory": "/path/a", "project_name": "a", "tags": ["backend"]},
            {"base_directory": "/path/b", "project_name": "b", "tags": ["python"]},
        ]
        mock_db_provider.get_indexed_roots_by_tags.return_value = matching_rows

        projects = registry.get_projects_by_tags(["backend", "python"], match_all=False)

        assert len(projects) == 2
        mock_db_provider.get_indexed_roots_by_tags.assert_called_with(
            ["backend", "python"], False
        )

    def test_get_projects_by_tags_empty_list(self, registry, mock_db_provider):
        """Test that empty tag list returns empty result."""
        projects = registry.get_projects_by_tags([])

        assert projects == []
        mock_db_provider.get_indexed_roots_by_tags.assert_not_called()

    def test_get_projects_by_tags_fallback(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test fallback to in-memory filtering when DB doesn't support multi-repo."""
        mock_db_provider.supports_multi_repo = False
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        projects = registry.get_projects_by_tags(["backend", "python"], match_all=True)

        assert len(projects) == 1
        assert projects[0].project_name == "myproject"

    def test_set_project_tags(self, registry, mock_db_provider, sample_project_row):
        """Test setting tags replaces existing tags."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        result = registry.set_project_tags("myproject", ["new-tag", "another-tag"])

        assert result is True
        mock_db_provider.update_indexed_root_tags.assert_called_once_with(
            "/home/user/myproject", ["new-tag", "another-tag"]
        )

    def test_set_project_tags_not_found(self, registry, mock_db_provider):
        """Test setting tags on non-existent project returns False."""
        mock_db_provider.get_indexed_roots.return_value = []

        result = registry.set_project_tags("nonexistent", ["tag"])

        assert result is False
        mock_db_provider.update_indexed_root_tags.assert_not_called()

    def test_add_project_tags(self, registry, mock_db_provider, sample_project_row):
        """Test adding tags preserves existing tags."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        result = registry.add_project_tags("myproject", ["new-tag"])

        assert result is True
        mock_db_provider.add_indexed_root_tags.assert_called_once_with(
            "/home/user/myproject", ["new-tag"]
        )

    def test_remove_project_tags(self, registry, mock_db_provider, sample_project_row):
        """Test removing tags from a project."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        result = registry.remove_project_tags("myproject", ["backend"])

        assert result is True
        mock_db_provider.remove_indexed_root_tags.assert_called_once_with(
            "/home/user/myproject", ["backend"]
        )

    def test_get_all_tags(self, registry, mock_db_provider):
        """Test getting all unique tags across projects."""
        mock_db_provider.get_all_tags.return_value = ["backend", "frontend", "python"]

        tags = registry.get_all_tags()

        assert tags == ["backend", "frontend", "python"]
        mock_db_provider.get_all_tags.assert_called_once()

    def test_get_all_tags_fallback(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test fallback to cache when DB doesn't support multi-repo."""
        mock_db_provider.supports_multi_repo = False
        sample_project_row["tags"] = ["backend", "python"]
        project2 = {
            "base_directory": "/path/b",
            "project_name": "b",
            "tags": ["frontend", "python"],
        }
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row, project2]

        tags = registry.get_all_tags()

        assert sorted(tags) == ["backend", "frontend", "python"]


class TestProjectStats:
    """Tests for project statistics updates."""

    def test_update_project_stats(self, registry, mock_db_provider, sample_project_row):
        """Test updating file count and timestamps."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        result = registry.update_project_stats("myproject")

        assert result is True
        mock_db_provider.update_indexed_root_stats.assert_called_once_with(
            "/home/user/myproject"
        )

    def test_update_project_stats_not_found(self, registry, mock_db_provider):
        """Test updating stats for non-existent project."""
        mock_db_provider.get_indexed_roots.return_value = []

        result = registry.update_project_stats("nonexistent")

        assert result is False

    def test_get_project_count(self, registry, mock_db_provider, sample_project_row):
        """Test getting the count of registered projects."""
        project2 = {
            "base_directory": "/home/user/other-project",
            "project_name": "other-project",
            "file_count": 50,
        }
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row, project2]

        count = registry.get_project_count()

        assert count == 2


class TestWatcherCoordination:
    """Tests for watcher lifecycle coordination."""

    def test_set_watcher_status(self, registry, mock_db_provider, sample_project_row):
        """Test setting watcher status."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        result = registry.set_watcher_status("myproject", active=True)

        assert result is True
        mock_db_provider.update_indexed_root_watcher_status.assert_called_once()

    def test_set_watcher_status_with_error(
        self, registry, mock_db_provider, sample_project_row
    ):
        """Test setting watcher status with error message."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        result = registry.set_watcher_status(
            "myproject", active=False, error="Connection failed"
        )

        assert result is True
        mock_db_provider.update_indexed_root_watcher_status.assert_called_with(
            "/home/user/myproject", False, "Connection failed"
        )

    def test_start_all_watchers(self, registry, mock_db_provider, sample_project_row):
        """Test starting watchers for all projects."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]
        mock_watcher = MagicMock()
        registry.set_watcher_manager(mock_watcher)

        results = registry.start_all_watchers()

        assert "myproject" in results
        assert results["myproject"] is True
        mock_watcher.start_watcher.assert_called_once()

    def test_start_all_watchers_no_manager(self, registry, mock_db_provider):
        """Test start_all_watchers returns empty when no manager set."""
        results = registry.start_all_watchers()

        assert results == {}

    def test_stop_all_watchers(self, registry, mock_db_provider, sample_project_row):
        """Test stopping all watchers."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]
        mock_watcher = MagicMock()
        registry.set_watcher_manager(mock_watcher)

        registry.stop_all_watchers()

        mock_watcher.stop_all.assert_called_once()


class TestCacheManagement:
    """Tests for cache invalidation and refresh."""

    def test_cache_invalidation_on_register(
        self, registry, mock_db_provider, temp_project_path
    ):
        """Test that registering a project invalidates cache."""
        # Prime the cache
        mock_db_provider.get_indexed_roots.return_value = []
        registry.list_projects()
        assert mock_db_provider.get_indexed_roots.call_count == 1

        # Register should invalidate
        registry.register_project(temp_project_path)
        registry.list_projects()

        # Should have called again due to cache invalidation
        assert mock_db_provider.get_indexed_roots.call_count == 2

    def test_cache_reuse(self, registry, mock_db_provider, sample_project_row):
        """Test that cache is reused when valid."""
        mock_db_provider.get_indexed_roots.return_value = [sample_project_row]

        # Multiple calls should only hit DB once
        registry.list_projects()
        registry.list_projects()
        registry.list_projects()

        assert mock_db_provider.get_indexed_roots.call_count == 1

    def test_provider_without_multi_repo_support(self, registry, mock_db_provider):
        """Test graceful handling when provider lacks multi-repo methods."""
        del mock_db_provider.get_indexed_roots

        # Should not raise, just return empty
        projects = registry.list_projects()
        assert projects == []

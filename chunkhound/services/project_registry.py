"""Project registry service for multi-repository support.

This service manages registered projects in global database mode, providing:
- Project registration and unregistration
- Project lookup by name or path
- Watcher lifecycle coordination
- Project metadata management

The ProjectRegistry is the central coordinator for multi-repo operations,
working with the database's indexed_roots table and the WatcherManager.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from chunkhound.services.watcher_manager import WatcherManager

from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.services.base_service import BaseService


@dataclass
class ProjectInfo:
    """Information about an indexed project.

    Attributes:
        base_directory: Absolute path to the project root
        project_name: Human-readable name (defaults to directory name)
        indexed_at: When the project was first indexed
        updated_at: When the project was last updated
        file_count: Number of files indexed
        watcher_active: Whether file watcher is running
        last_error: Last error message (if any)
        error_count: Number of consecutive errors
        config_snapshot: Configuration used for indexing (embedding provider, model, etc.)
        tags: User-defined tags for categorization and filtering
    """

    base_directory: Path
    project_name: str
    indexed_at: datetime | None = None
    updated_at: datetime | None = None
    file_count: int = 0
    watcher_active: bool = False
    last_error: str | None = None
    error_count: int = 0
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_db_row(cls, row: dict[str, Any]) -> "ProjectInfo":
        """Create ProjectInfo from database row.

        Args:
            row: Dictionary with keys from indexed_roots table

        Returns:
            ProjectInfo instance
        """
        return cls(
            base_directory=Path(row["base_directory"]),
            project_name=row.get("project_name", Path(row["base_directory"]).name),
            indexed_at=row.get("indexed_at"),
            updated_at=row.get("updated_at"),
            file_count=row.get("file_count", 0),
            watcher_active=row.get("watcher_active", False),
            last_error=row.get("last_error"),
            error_count=row.get("error_count", 0),
            config_snapshot=row.get("config_snapshot") or {},
            tags=row.get("tags") or [],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "base_directory": str(self.base_directory),
            "project_name": self.project_name,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "file_count": self.file_count,
            "watcher_active": self.watcher_active,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "config_snapshot": self.config_snapshot,
            "tags": self.tags,
        }


class ProjectRegistry(BaseService):
    """Manages registered projects and their watchers.

    The ProjectRegistry is responsible for:
    - Registering and unregistering projects
    - Looking up projects by name or path
    - Finding which project contains a given file path
    - Coordinating with WatcherManager for file monitoring

    Thread-safe: All operations protected by RLock for concurrent access.

    Usage:
        registry = ProjectRegistry(db_provider)

        # Register a new project
        project = registry.register_project(Path("/home/user/myproject"))

        # Find project by name
        project = registry.get_project("myproject")

        # Find project containing a file
        project = registry.find_project_for_path(Path("/home/user/myproject/src/main.py"))

        # List all projects
        for project in registry.list_projects():
            print(project.project_name)
    """

    def __init__(self, database_provider: DatabaseProvider):
        """Initialize project registry.

        Args:
            database_provider: Database provider for persistence
        """
        super().__init__(database_provider)
        self._lock = RLock()
        # In-memory cache of projects (path -> ProjectInfo)
        self._cache: dict[str, ProjectInfo] = {}
        self._cache_valid = False
        # Reference to watcher manager (set later to avoid circular import)
        self._watcher_manager: "WatcherManager | None" = None

    def set_watcher_manager(self, watcher_manager: "WatcherManager") -> None:
        """Set the watcher manager for coordinating file watchers.

        Args:
            watcher_manager: WatcherManager instance
        """
        self._watcher_manager = watcher_manager

    def _invalidate_cache(self) -> None:
        """Invalidate the in-memory cache."""
        with self._lock:
            self._cache_valid = False
            self._cache.clear()

    def _refresh_cache(self) -> None:
        """Refresh cache from database if invalid."""
        with self._lock:
            if self._cache_valid:
                return

            try:
                rows = self._db.get_indexed_roots()
                self._cache.clear()
                for row in rows:
                    project = ProjectInfo.from_db_row(row)
                    self._cache[str(project.base_directory)] = project
                self._cache_valid = True
                logger.debug(f"Refreshed project cache: {len(self._cache)} projects")
            except AttributeError:
                # Provider doesn't support get_indexed_roots (per-repo mode)
                logger.debug("Database provider does not support multi-repo features")
                self._cache_valid = True

    def validate_path_not_nested(self, path: Path) -> None:
        """Validate that a path doesn't conflict with existing projects.

        Checks that the path is not a subfolder of any existing project,
        and that no existing project is a subfolder of the path.

        Args:
            path: Path to validate (will be resolved to absolute)

        Raises:
            ValueError: If path is nested within or contains an existing project
        """
        path = path.resolve()
        path_str = str(path)

        with self._lock:
            self._refresh_cache()
            for existing in self._cache.values():
                existing_str = str(existing.base_directory)

                # Skip exact match (re-registration is OK)
                if path_str == existing_str:
                    continue

                # Check if new path is subfolder of existing project
                if path_str.startswith(existing_str + "/"):
                    raise ValueError(
                        f"Cannot index '{path}': it is a subfolder of existing "
                        f"project '{existing.project_name}' at {existing.base_directory}"
                    )

                # Check if existing project is subfolder of new path
                if existing_str.startswith(path_str + "/"):
                    raise ValueError(
                        f"Cannot index '{path}': existing project "
                        f"'{existing.project_name}' at {existing.base_directory} "
                        f"is a subfolder. Remove it first with: "
                        f"chunkhound repos remove {existing.project_name}"
                    )

    def register_project(
        self,
        path: Path,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> ProjectInfo:
        """Register a new project or update existing registration.

        Args:
            path: Absolute path to the project root directory
            name: Optional project name (defaults to directory name)
            config: Optional configuration snapshot (embedding provider, model, etc.)

        Returns:
            ProjectInfo for the registered project

        Raises:
            ValueError: If path is not absolute or doesn't exist
        """
        path = path.resolve()

        if not path.is_absolute():
            raise ValueError(f"Project path must be absolute: {path}")

        if not path.exists():
            raise ValueError(f"Project path does not exist: {path}")

        if not path.is_dir():
            raise ValueError(f"Project path must be a directory: {path}")

        project_name = name or path.name

        with self._lock:
            # Validate no nested path conflicts (re-registration is OK)
            self.validate_path_not_nested(path)

            # Register in database
            try:
                self._db.register_base_directory(str(path), project_name, config)
                logger.info(f"Registered project: {project_name} at {path}")
            except AttributeError:
                logger.warning(
                    "Database provider does not support register_base_directory"
                )

            # Invalidate cache to pick up changes
            self._invalidate_cache()

            # Return fresh project info
            return self.get_project(str(path)) or ProjectInfo(
                base_directory=path,
                project_name=project_name,
                config_snapshot=config or {},
            )

    def unregister_project(
        self,
        name_or_path: str,
        cascade: bool = False,
    ) -> bool:
        """Unregister a project.

        Args:
            name_or_path: Project name or absolute path
            cascade: If True, also delete all indexed files/chunks/embeddings

        Returns:
            True if project was unregistered, False if not found
        """
        with self._lock:
            project = self.get_project(name_or_path)
            if not project:
                logger.warning(f"Project not found for unregister: {name_or_path}")
                return False

            # Stop watcher if active
            if self._watcher_manager and project.watcher_active:
                self._watcher_manager.stop_watcher(project.base_directory)

            # Remove from database
            try:
                self._db.remove_base_directory(
                    str(project.base_directory), cascade=cascade
                )
                logger.info(
                    f"Unregistered project: {project.project_name} (cascade={cascade})"
                )
            except AttributeError:
                logger.warning(
                    "Database provider does not support remove_base_directory"
                )
                return False

            # Invalidate cache
            self._invalidate_cache()

            return True

    def get_project(self, name_or_path: str) -> ProjectInfo | None:
        """Get project by name or path.

        Args:
            name_or_path: Project name or absolute path

        Returns:
            ProjectInfo if found, None otherwise
        """
        with self._lock:
            self._refresh_cache()

            # Try as absolute path first
            path_key = (
                str(Path(name_or_path).resolve())
                if name_or_path.startswith("/")
                else None
            )
            if path_key and path_key in self._cache:
                return self._cache[path_key]

            # Try as project name
            for project in self._cache.values():
                if project.project_name == name_or_path:
                    return project

            # Try as directory name (last component of path)
            for project in self._cache.values():
                if project.base_directory.name == name_or_path:
                    return project

            return None

    def list_projects(self) -> list[ProjectInfo]:
        """Get all registered projects.

        Returns:
            List of ProjectInfo for all registered projects
        """
        with self._lock:
            self._refresh_cache()
            return list(self._cache.values())

    def find_project_for_path(self, path: Path) -> ProjectInfo | None:
        """Find which project contains the given path.

        Uses longest prefix matching to find the most specific project
        that contains the given path.

        Args:
            path: Absolute path to check

        Returns:
            ProjectInfo for containing project, or None if not found

        Example:
            # Given projects:
            #   /home/user/project1
            #   /home/user/project1/subproject
            #
            # find_project_for_path("/home/user/project1/subproject/src/main.py")
            # Returns: project1/subproject (longest match)
        """
        if not path.is_absolute():
            path = path.resolve()

        path_str = str(path)

        with self._lock:
            self._refresh_cache()

            best_match: ProjectInfo | None = None
            best_match_len = 0

            for project in self._cache.values():
                base_str = str(project.base_directory)

                # Check if path is under this project
                if path_str == base_str or path_str.startswith(base_str + "/"):
                    # Prefer longer (more specific) match
                    if len(base_str) > best_match_len:
                        best_match = project
                        best_match_len = len(base_str)

            return best_match

    def update_project_stats(self, name_or_path: str) -> bool:
        """Update file count and timestamps for a project.

        Args:
            name_or_path: Project name or path

        Returns:
            True if updated successfully, False if project not found
        """
        with self._lock:
            project = self.get_project(name_or_path)
            if not project:
                return False

            try:
                self._db.update_indexed_root_stats(str(project.base_directory))
                self._invalidate_cache()
                logger.debug(f"Updated stats for project: {project.project_name}")
                return True
            except AttributeError:
                logger.warning(
                    "Database provider does not support update_indexed_root_stats"
                )
                return False

    def set_watcher_status(
        self,
        name_or_path: str,
        active: bool,
        error: str | None = None,
    ) -> bool:
        """Update watcher status for a project.

        Args:
            name_or_path: Project name or path
            active: Whether watcher is active
            error: Error message if watcher failed

        Returns:
            True if updated successfully, False if project not found
        """
        with self._lock:
            project = self.get_project(name_or_path)
            if not project:
                return False

            # Update in database
            try:
                if getattr(self._db, "supports_multi_repo", False):
                    self._db.update_indexed_root_watcher_status(
                        str(project.base_directory),
                        active,
                        error,
                    )
                self._invalidate_cache()
                logger.debug(
                    f"Updated watcher status for {project.project_name}: "
                    f"active={active}, error={error}"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to update watcher status: {e}")
                return False

    def get_project_count(self) -> int:
        """Get the number of registered projects.

        Returns:
            Number of registered projects
        """
        with self._lock:
            self._refresh_cache()
            return len(self._cache)

    def start_all_watchers(self) -> dict[str, bool]:
        """Start file watchers for all registered projects.

        Requires WatcherManager to be set via set_watcher_manager().

        Returns:
            Dictionary mapping project names to success status
        """
        if not self._watcher_manager:
            logger.warning("WatcherManager not set, cannot start watchers")
            return {}

        results = {}
        with self._lock:
            for project in self.list_projects():
                try:
                    self._watcher_manager.start_watcher(project)
                    results[project.project_name] = True
                    logger.info(f"Started watcher for: {project.project_name}")
                except Exception as e:
                    results[project.project_name] = False
                    logger.error(
                        f"Failed to start watcher for {project.project_name}: {e}"
                    )
                    self.set_watcher_status(
                        str(project.base_directory),
                        active=False,
                        error=str(e),
                    )

        return results

    def stop_all_watchers(self) -> None:
        """Stop all file watchers.

        Requires WatcherManager to be set via set_watcher_manager().
        """
        if not self._watcher_manager:
            logger.warning("WatcherManager not set, cannot stop watchers")
            return

        with self._lock:
            self._watcher_manager.stop_all()
            # Update status for all projects
            for project in self.list_projects():
                self.set_watcher_status(str(project.base_directory), active=False)

        logger.info("Stopped all project watchers")

    # =========================================================================
    # Tag Management
    # =========================================================================

    def get_projects_by_tags(
        self, tags: list[str], match_all: bool = True
    ) -> list[ProjectInfo]:
        """Get projects that have the specified tags.

        Args:
            tags: List of tags to filter by
            match_all: If True, projects must have ALL tags (AND).
                      If False, projects must have ANY tag (OR).

        Returns:
            List of matching ProjectInfo objects
        """
        if not tags:
            return []

        with self._lock:
            try:
                if getattr(self._db, "supports_multi_repo", False):
                    rows = self._db.get_indexed_roots_by_tags(tags, match_all)
                    return [ProjectInfo.from_db_row(row) for row in rows]
                else:
                    # Fallback: filter in memory
                    self._refresh_cache()
                    if match_all:
                        return [
                            p
                            for p in self._cache.values()
                            if all(tag in p.tags for tag in tags)
                        ]
                    else:
                        return [
                            p
                            for p in self._cache.values()
                            if any(tag in p.tags for tag in tags)
                        ]
            except Exception as e:
                logger.error(f"Failed to get projects by tags: {e}")
                return []

    def set_project_tags(self, name_or_path: str, tags: list[str]) -> bool:
        """Set tags for a project (replaces existing tags).

        Args:
            name_or_path: Project name or path
            tags: New list of tags

        Returns:
            True if updated successfully, False if project not found
        """
        with self._lock:
            project = self.get_project(name_or_path)
            if not project:
                logger.warning(f"Project not found: {name_or_path}")
                return False

            try:
                if getattr(self._db, "supports_multi_repo", False):
                    self._db.update_indexed_root_tags(str(project.base_directory), tags)
                    self._invalidate_cache()
                    logger.info(f"Set tags for {project.project_name}: {tags}")
                    return True
                else:
                    logger.warning("Database provider does not support tag management")
                    return False
            except Exception as e:
                logger.error(f"Failed to set tags: {e}")
                return False

    def add_project_tags(self, name_or_path: str, tags: list[str]) -> bool:
        """Add tags to a project (preserves existing tags).

        Args:
            name_or_path: Project name or path
            tags: Tags to add

        Returns:
            True if updated successfully, False if project not found
        """
        with self._lock:
            project = self.get_project(name_or_path)
            if not project:
                logger.warning(f"Project not found: {name_or_path}")
                return False

            try:
                if getattr(self._db, "supports_multi_repo", False):
                    self._db.add_indexed_root_tags(str(project.base_directory), tags)
                    self._invalidate_cache()
                    logger.info(f"Added tags to {project.project_name}: {tags}")
                    return True
                else:
                    logger.warning("Database provider does not support tag management")
                    return False
            except Exception as e:
                logger.error(f"Failed to add tags: {e}")
                return False

    def remove_project_tags(self, name_or_path: str, tags: list[str]) -> bool:
        """Remove tags from a project.

        Args:
            name_or_path: Project name or path
            tags: Tags to remove

        Returns:
            True if updated successfully, False if project not found
        """
        with self._lock:
            project = self.get_project(name_or_path)
            if not project:
                logger.warning(f"Project not found: {name_or_path}")
                return False

            try:
                if getattr(self._db, "supports_multi_repo", False):
                    self._db.remove_indexed_root_tags(str(project.base_directory), tags)
                    self._invalidate_cache()
                    logger.info(f"Removed tags from {project.project_name}: {tags}")
                    return True
                else:
                    logger.warning("Database provider does not support tag management")
                    return False
            except Exception as e:
                logger.error(f"Failed to remove tags: {e}")
                return False

    def get_all_tags(self) -> list[str]:
        """Get all unique tags across all projects.

        Returns:
            Sorted list of unique tag names
        """
        with self._lock:
            try:
                if getattr(self._db, "supports_multi_repo", False):
                    return self._db.get_all_tags()
                else:
                    # Fallback: collect from cache
                    self._refresh_cache()
                    all_tags = set()
                    for project in self._cache.values():
                        all_tags.update(project.tags)
                    return sorted(all_tags)
            except Exception as e:
                logger.error(f"Failed to get all tags: {e}")
                return []

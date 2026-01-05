"""Search context model for multi-project search operations.

This module provides the SearchContext dataclass that encapsulates all
scoping information for search operations in global database mode.

Search Scope Resolution:
    1. search_all=True → all indexed projects
    2. projects=[...] → those specific projects by name
    3. path="/absolute/..." → project containing that path
    4. path="relative/..." → current project + relative path
    5. (nothing) → current project only
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from chunkhound.services.project_registry import ProjectInfo, ProjectRegistry


@dataclass
class SearchScope:
    """Resolved search scope with project paths to query.

    Attributes:
        projects: List of project infos to search
        path_filter: Optional path filter within projects (relative)
        is_all_projects: Whether searching all indexed projects
    """

    projects: list["ProjectInfo"] = field(default_factory=list)
    path_filter: str | None = None
    is_all_projects: bool = False

    @property
    def project_paths(self) -> list[Path]:
        """Get list of project base directory paths."""
        return [p.base_directory for p in self.projects]

    @property
    def project_names(self) -> list[str]:
        """Get list of project names."""
        return [p.project_name for p in self.projects]


@dataclass
class SearchContext:
    """Context for search operations in global database mode.

    This class encapsulates all scoping information for a search operation,
    including the current project context, target projects/paths, and options.

    Attributes:
        current_project_path: The client's current project path (from context header)
        target_projects: Explicit list of project names to search
        target_paths: Explicit list of absolute paths to search
        path_filter: Path filter within the project(s)
        search_all: Whether to search all indexed projects
        tags: List of tags to filter projects (AND logic)
    """

    # Project scope
    current_project_path: Path | None = None
    target_projects: list[str] | None = None
    target_paths: list[Path] | None = None
    search_all: bool = False

    # Tag filtering
    tags: list[str] | None = None

    # Path filtering within project(s)
    path_filter: str | None = None

    @classmethod
    def from_tool_arguments(
        cls,
        arguments: dict[str, Any],
        client_context: dict[str, Any] | None = None,
    ) -> "SearchContext":
        """Create SearchContext from MCP tool arguments.

        Args:
            arguments: Tool call arguments dict
            client_context: Client context dict (from HTTP header or session)

        Returns:
            SearchContext instance

        Example arguments:
            {
                "query": "authentication",
                "path": "/home/user/project/src/",  # or "src/" for relative
                "projects": ["project-a", "project-b"],
                "search_all": True,
            }
        """
        # Extract current project from client context
        current_project = None
        if client_context:
            project_str = client_context.get("project")
            if project_str:
                current_project = Path(project_str)

        # Extract target projects
        target_projects = arguments.get("projects")
        if isinstance(target_projects, str):
            target_projects = [target_projects]

        # Extract path (could be absolute or relative)
        path_arg = arguments.get("path")
        target_paths = None
        path_filter = None

        if path_arg:
            path = Path(path_arg)
            if path.is_absolute():
                # Absolute path - treat as target path
                target_paths = [path]
            else:
                # Relative path - use as filter within current project
                path_filter = str(path)

        # Handle search_all flag
        search_all = arguments.get("search_all", False)

        # Extract tags
        tags = arguments.get("tags")
        if isinstance(tags, str):
            tags = [tags]

        return cls(
            current_project_path=current_project,
            target_projects=target_projects,
            target_paths=target_paths,
            path_filter=path_filter,
            search_all=search_all,
            tags=tags,
        )

    def resolve_scope(
        self,
        registry: "ProjectRegistry | None",
    ) -> SearchScope:
        """Resolve this context to a concrete search scope.

        Resolution priority:
            1. tags → projects matching all tags
            2. search_all=True → all indexed projects
            3. target_projects → those specific projects
            4. target_paths → projects containing those paths
            5. current_project_path → that project
            6. (nothing) → empty scope (will use default per-repo behavior)

        Note: tags can be combined with other scopes to further filter.

        Args:
            registry: ProjectRegistry for resolving project info

        Returns:
            SearchScope with resolved projects and filters
        """
        scope = SearchScope()

        # If no registry, return empty scope (per-repo mode will handle)
        if registry is None:
            logger.debug("No registry available, returning empty scope")
            return scope

        # Priority 1: Tags (can also be combined with other scopes)
        if self.tags:
            # Get projects matching all tags
            tagged_projects = registry.get_projects_by_tags(self.tags, match_all=True)
            if not tagged_projects:
                logger.warning(f"No projects found with tags: {self.tags}")
                return scope  # Empty scope

            # If search_all or no other scope, use all tagged projects
            if self.search_all or (
                not self.target_projects
                and not self.target_paths
                and not self.current_project_path
            ):
                scope.projects = tagged_projects
                scope.is_all_projects = self.search_all
                scope.path_filter = self.path_filter
                logger.debug(
                    f"Resolved to {len(scope.projects)} projects with tags {self.tags}"
                )
                return scope

            # Continue to other priorities with tag filtering applied inline
            # (Each priority check filters by tags if self.tags is set)

        # Priority 2: search_all
        if self.search_all:
            scope.projects = registry.list_projects()
            # Filter by tags if specified
            if self.tags:
                scope.projects = [
                    p for p in scope.projects if all(tag in p.tags for tag in self.tags)
                ]
            scope.is_all_projects = True
            scope.path_filter = self.path_filter
            logger.debug(f"Resolved to all {len(scope.projects)} projects")
            return scope

        # Priority 3: explicit project names
        if self.target_projects:
            for name in self.target_projects:
                project = registry.get_project(name)
                if project:
                    # Filter by tags if specified
                    if self.tags and not all(tag in project.tags for tag in self.tags):
                        logger.debug(f"Project {name} excluded by tag filter")
                        continue
                    scope.projects.append(project)
                else:
                    logger.warning(f"Project not found: {name}")
            scope.path_filter = self.path_filter
            logger.debug(f"Resolved to {len(scope.projects)} target projects")
            return scope

        # Priority 4: absolute paths
        if self.target_paths:
            for path in self.target_paths:
                project = registry.find_project_for_path(path)
                if project:
                    # Filter by tags if specified
                    if self.tags and not all(tag in project.tags for tag in self.tags):
                        logger.debug(f"Project for {path} excluded by tag filter")
                        continue
                    if project not in scope.projects:
                        scope.projects.append(project)
                    # Set path filter to the relative portion (first match only)
                    # NOTE: SearchScope only supports a single path_filter.
                    # Multiple absolute paths use the first non-root relative.
                    # Multi-path filtering requires database OR-based queries.
                    if scope.path_filter is None:
                        rel_path = path.relative_to(project.base_directory)
                        if str(rel_path) != ".":
                            scope.path_filter = str(rel_path)
                else:
                    logger.warning(f"No project found for path: {path}")
            logger.debug(f"Resolved absolute paths to {len(scope.projects)} projects")
            return scope

        # Priority 5: current project context
        if self.current_project_path:
            project = registry.find_project_for_path(self.current_project_path)
            if project:
                # Filter by tags if specified
                if self.tags and not all(tag in project.tags for tag in self.tags):
                    logger.debug("Current project excluded by tag filter")
                    return scope  # Empty
                scope.projects.append(project)
                scope.path_filter = self.path_filter
                logger.debug(f"Resolved to current project: {project.project_name}")
            else:
                logger.warning(
                    f"Current project not indexed: {self.current_project_path}"
                )
            return scope

        # Priority 6: no scope (empty - let per-repo behavior handle)
        logger.debug("No scope specified, returning empty")
        return scope

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_project_path": str(self.current_project_path)
            if self.current_project_path
            else None,
            "target_projects": self.target_projects,
            "target_paths": [str(p) for p in self.target_paths]
            if self.target_paths
            else None,
            "path_filter": self.path_filter,
            "search_all": self.search_all,
            "tags": self.tags,
        }


def resolve_search_path_filter(
    context: SearchContext,
    scope: SearchScope,
) -> str | None:
    """Resolve the final path filter string for database query.

    Combines the search scope with any explicit path filter to produce
    the final path filter for the database query.

    Args:
        context: Original search context
        scope: Resolved search scope

    Returns:
        Path filter string for database query, or None for no filter
    """
    if scope.is_all_projects:
        # All projects - only use explicit path filter
        return scope.path_filter

    if not scope.projects:
        # No projects in scope - use path filter as-is (per-repo mode)
        return context.path_filter

    # Single project - prefix path filter with project path
    if len(scope.projects) == 1:
        project = scope.projects[0]
        if scope.path_filter:
            # Combine project path with filter
            return str(project.base_directory / scope.path_filter)
        else:
            # Filter by project path
            return str(project.base_directory)

    # Multiple projects - will need to search each separately
    # Return None and let caller handle multi-project case
    return None

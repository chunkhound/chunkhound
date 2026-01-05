"""Tags command module - handles project tag management operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from chunkhound.core.config.config import Config

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.database import DaemonProxy
    from chunkhound.services.project_registry import ProjectRegistry

from ..utils.rich_output import RichOutputFormatter


async def tags_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the tags command.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))

    # Check if global mode is enabled
    if not (config.database.multi_repo and config.database.multi_repo.enabled):
        formatter.error(
            "Tags require global mode. Enable with:\n"
            "  export CHUNKHOUND_DATABASE__MULTI_REPO__ENABLED=true\n"
            "  export CHUNKHOUND_DATABASE__MULTI_REPO__MODE=global"
        )
        sys.exit(1)

    # Route to subcommand
    subcommand = getattr(args, "tags_command", None)

    if subcommand is None:
        formatter.error(
            "No tags subcommand specified. Use: tags list|add|remove|set|clear"
        )
        sys.exit(1)

    # Check if we should proxy through daemon
    from chunkhound.api.cli.utils.database import get_daemon_proxy_if_running

    proxy = get_daemon_proxy_if_running(config)
    if proxy:
        try:
            if subcommand == "list":
                await _tags_list_proxy(args, proxy, formatter)
            elif subcommand == "add":
                await _tags_add_proxy(args, proxy, formatter)
            elif subcommand == "remove":
                await _tags_remove_proxy(args, proxy, formatter)
            elif subcommand == "set":
                await _tags_set_proxy(args, proxy, formatter)
            elif subcommand == "clear":
                await _tags_clear_proxy(args, proxy, formatter)
            else:
                formatter.error(f"Unknown tags subcommand: {subcommand}")
                sys.exit(1)
            return
        except Exception as e:
            formatter.error(f"Tags command failed: {e}")
            logger.exception("Tags command error details")
            sys.exit(1)

    # Direct database access (daemon not running)
    try:
        from chunkhound.database_factory import create_services
        from chunkhound.services.project_registry import ProjectRegistry

        # Create services to access database (handles global mode automatically)
        db_path = config.database.get_db_path(current_dir=Path.cwd())
        services = create_services(db_path=db_path, config=config)
        services.provider.connect()

        try:
            # Create registry
            registry = ProjectRegistry(services.provider)

            if subcommand == "list":
                await _tags_list(args, registry, formatter)
            elif subcommand == "add":
                await _tags_add(args, registry, formatter)
            elif subcommand == "remove":
                await _tags_remove(args, registry, formatter)
            elif subcommand == "set":
                await _tags_set(args, registry, formatter)
            elif subcommand == "clear":
                await _tags_clear(args, registry, formatter)
            else:
                formatter.error(f"Unknown tags subcommand: {subcommand}")
                sys.exit(1)
        finally:
            services.provider.disconnect()

    except Exception as e:
        formatter.error(f"Tags command failed: {e}")
        logger.exception("Tags command error details")
        sys.exit(1)


async def _tags_list(
    args: argparse.Namespace,
    registry: ProjectRegistry,
    formatter: RichOutputFormatter,
) -> None:
    """List tags."""
    project_name = getattr(args, "project", None)
    as_json = getattr(args, "json", False)

    if project_name:
        # List tags for specific project
        project = registry.get_project(project_name)
        if not project:
            formatter.error(f"Project not found: {project_name}")
            sys.exit(1)

        if as_json:
            print(
                json.dumps(
                    {
                        "project": project.project_name,
                        "path": str(project.base_directory),
                        "tags": project.tags,
                    },
                    indent=2,
                )
            )
        else:
            if project.tags:
                formatter.section_header(f"Tags for {project.project_name}")
                for tag in sorted(project.tags):
                    print(f"  {tag}")
            else:
                formatter.info(f"No tags for {project.project_name}")
    else:
        # List all unique tags
        all_tags = registry.get_all_tags()

        if as_json:
            # Include tag counts
            tag_counts = {}
            for project in registry.list_projects():
                for tag in project.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            print(
                json.dumps(
                    {
                        "tags": [
                            {"name": tag, "project_count": tag_counts.get(tag, 0)}
                            for tag in all_tags
                        ]
                    },
                    indent=2,
                )
            )
        else:
            if all_tags:
                formatter.section_header("All Tags")
                # Show with counts
                tag_counts = {}
                for project in registry.list_projects():
                    for tag in project.tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

                for tag in all_tags:
                    count = tag_counts.get(tag, 0)
                    print(f"  {tag} ({count} project{'s' if count != 1 else ''})")
            else:
                formatter.info("No tags defined")


async def _tags_add(
    args: argparse.Namespace,
    registry: ProjectRegistry,
    formatter: RichOutputFormatter,
) -> None:
    """Add tags to a project."""
    project_name = args.project
    tags = args.tags

    project = registry.get_project(project_name)
    if not project:
        formatter.error(f"Project not found: {project_name}")
        sys.exit(1)

    if registry.add_project_tags(project_name, tags):
        # Get updated project
        updated = registry.get_project(project_name)
        formatter.success(
            f"Added tags to {project.project_name}: {', '.join(tags)}\n"
            f"Current tags: {', '.join(updated.tags) if updated else 'unknown'}"
        )
    else:
        formatter.error(f"Failed to add tags to {project_name}")
        sys.exit(1)


async def _tags_remove(
    args: argparse.Namespace,
    registry: ProjectRegistry,
    formatter: RichOutputFormatter,
) -> None:
    """Remove tags from a project."""
    project_name = args.project
    tags = args.tags

    project = registry.get_project(project_name)
    if not project:
        formatter.error(f"Project not found: {project_name}")
        sys.exit(1)

    if registry.remove_project_tags(project_name, tags):
        # Get updated project
        updated = registry.get_project(project_name)
        remaining = updated.tags if updated else []
        formatter.success(
            f"Removed tags from {project.project_name}: {', '.join(tags)}\n"
            f"Remaining tags: {', '.join(remaining) if remaining else '(none)'}"
        )
    else:
        formatter.error(f"Failed to remove tags from {project_name}")
        sys.exit(1)


async def _tags_set(
    args: argparse.Namespace,
    registry: ProjectRegistry,
    formatter: RichOutputFormatter,
) -> None:
    """Set (replace) tags for a project."""
    project_name = args.project
    tags = args.tags

    project = registry.get_project(project_name)
    if not project:
        formatter.error(f"Project not found: {project_name}")
        sys.exit(1)

    old_tags = project.tags.copy() if project.tags else []

    if registry.set_project_tags(project_name, tags):
        formatter.success(
            f"Set tags for {project.project_name}: {', '.join(tags)}\n"
            f"Previous tags: {', '.join(old_tags) if old_tags else '(none)'}"
        )
    else:
        formatter.error(f"Failed to set tags for {project_name}")
        sys.exit(1)


async def _tags_clear(
    args: argparse.Namespace,
    registry: ProjectRegistry,
    formatter: RichOutputFormatter,
) -> None:
    """Clear all tags from a project."""
    project_name = args.project

    project = registry.get_project(project_name)
    if not project:
        formatter.error(f"Project not found: {project_name}")
        sys.exit(1)

    old_tags = project.tags.copy() if project.tags else []

    if not old_tags:
        formatter.info(f"No tags to clear for {project.project_name}")
        return

    if registry.set_project_tags(project_name, []):
        formatter.success(
            f"Cleared all tags from {project.project_name}\n"
            f"Removed: {', '.join(old_tags)}"
        )
    else:
        formatter.error(f"Failed to clear tags from {project_name}")
        sys.exit(1)


# Proxy implementations for daemon mode


async def _tags_list_proxy(
    args: argparse.Namespace,
    proxy: DaemonProxy,
    formatter: RichOutputFormatter,
) -> None:
    """List tags via daemon proxy."""
    project_name = getattr(args, "project", None)
    as_json = getattr(args, "json", False)

    if project_name:
        tags = proxy.get_project_tags(project_name)
        if as_json:
            print(json.dumps({"project": project_name, "tags": tags}, indent=2))
        else:
            if tags:
                formatter.section_header(f"Tags for {project_name}")
                for tag in sorted(tags):
                    print(f"  {tag}")
            else:
                formatter.info(f"No tags for {project_name}")
    else:
        all_tags = proxy.list_all_tags()
        if as_json:
            print(json.dumps({"tags": [{"name": tag} for tag in all_tags]}, indent=2))
        else:
            if all_tags:
                formatter.section_header("All Tags")
                for tag in all_tags:
                    print(f"  {tag}")
            else:
                formatter.info("No tags defined")


async def _tags_add_proxy(
    args: argparse.Namespace,
    proxy: DaemonProxy,
    formatter: RichOutputFormatter,
) -> None:
    """Add tags via daemon proxy."""
    project_name = args.project
    tags = args.tags

    result = proxy.add_project_tags(project_name, tags)
    if result.get("status") == "success":
        formatter.success(
            f"Added tags to {project_name}: {', '.join(tags)}\n"
            f"Current tags: {', '.join(result.get('tags', []))}"
        )
    else:
        formatter.error(f"Failed to add tags: {result.get('error')}")
        sys.exit(1)


async def _tags_remove_proxy(
    args: argparse.Namespace,
    proxy: DaemonProxy,
    formatter: RichOutputFormatter,
) -> None:
    """Remove tags via daemon proxy."""
    project_name = args.project
    tags = args.tags

    result = proxy.remove_project_tags(project_name, tags)
    if result.get("status") == "success":
        remaining = result.get("tags", [])
        formatter.success(
            f"Removed tags from {project_name}: {', '.join(tags)}\n"
            f"Remaining tags: {', '.join(remaining) if remaining else '(none)'}"
        )
    else:
        formatter.error(f"Failed to remove tags: {result.get('error')}")
        sys.exit(1)


async def _tags_set_proxy(
    args: argparse.Namespace,
    proxy: DaemonProxy,
    formatter: RichOutputFormatter,
) -> None:
    """Set tags via daemon proxy."""
    project_name = args.project
    tags = args.tags

    result = proxy.set_project_tags(project_name, tags)
    if result.get("status") == "success":
        formatter.success(f"Set tags for {project_name}: {', '.join(tags)}")
    else:
        formatter.error(f"Failed to set tags: {result.get('error')}")
        sys.exit(1)


async def _tags_clear_proxy(
    args: argparse.Namespace,
    proxy: DaemonProxy,
    formatter: RichOutputFormatter,
) -> None:
    """Clear tags via daemon proxy."""
    project_name = args.project

    # First get current tags to show what's being cleared
    current_tags = proxy.get_project_tags(project_name)

    if not current_tags:
        formatter.info(f"No tags to clear for {project_name}")
        return

    result = proxy.set_project_tags(project_name, [])
    if result.get("status") == "success":
        formatter.success(
            f"Cleared all tags from {project_name}\nRemoved: {', '.join(current_tags)}"
        )
    else:
        formatter.error(f"Failed to clear tags: {result.get('error')}")
        sys.exit(1)


__all__ = ["tags_command"]

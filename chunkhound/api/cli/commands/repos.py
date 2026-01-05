"""CLI commands for managing indexed repositories."""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.core.config.config import Config


def _format_time_ago(dt: datetime | str | None) -> str:
    """Format datetime as relative time (e.g., '2h ago', '3d ago')."""
    if dt is None:
        return "never"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return str(dt)[:16] if dt else "never"

    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt

    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days}d ago"
    else:
        weeks = int(seconds / 604800)
        return f"{weeks}w ago"


async def list_command(
    args, formatter: RichOutputFormatter, config: Config | None = None
) -> None:
    """List all indexed repositories/base directories.

    Args:
        args: Parsed command-line arguments
        formatter: Output formatter for displaying results
        config: Configuration object
    """
    try:
        # Get config if not provided
        if config is None:
            from chunkhound.api.cli.utils.config_factory import create_validated_config

            config, _ = create_validated_config(args, "repos")

        # Check if global mode is enabled
        if not (config.database.multi_repo and config.database.multi_repo.enabled):
            formatter.error(
                "Repos command requires global mode. Enable with:\n"
                "  export CHUNKHOUND_DATABASE__MULTI_REPO__ENABLED=true\n"
                "  export CHUNKHOUND_DATABASE__MULTI_REPO__MODE=global"
            )
            sys.exit(1)

        # In global mode, must use daemon proxy (no direct DB access to prevent locks)
        from chunkhound.api.cli.utils.database import require_daemon_or_exit

        proxy = require_daemon_or_exit(config, formatter)
        roots = proxy.list_projects()

        # Process indexed roots
        if not roots:
            formatter.info("No indexed repositories found")
            return

        # Apply filters
        filter_tags = getattr(args, "tags", None)
        filter_active = getattr(args, "active", False)
        filter_errors = getattr(args, "errors", False)

        filtered = []
        for root in roots:
            # Tag filter (AND logic - must have all specified tags)
            if filter_tags:
                root_tags = set(root.get("tags", []))
                if not all(t in root_tags for t in filter_tags):
                    continue

            # Active watcher filter
            if filter_active and not root.get("watcher_active"):
                continue

            # Errors filter
            if filter_errors and not root.get("last_error"):
                continue

            filtered.append(root)

        if not filtered:
            formatter.info("No repositories match filters")
            return

        # Sort
        sort_key = getattr(args, "sort", "name")
        reverse = getattr(args, "reverse", False)

        sort_map = {
            "name": lambda r: (r.get("project_name") or "").lower(),
            "updated": lambda r: r.get("updated_at") or "",
            "files": lambda r: r.get("file_count", 0),
            "path": lambda r: r.get("base_directory", "").lower(),
        }
        filtered.sort(key=sort_map.get(sort_key, sort_map["name"]), reverse=reverse)

        # Plain output mode
        if getattr(args, "plain", False):
            for root in filtered:
                base_dir = root.get("base_directory", "Unknown")
                project_name = root.get("project_name") or Path(base_dir).name
                tags = root.get("tags", [])
                tags_str = f" [{', '.join(tags)}]" if tags else ""
                print(f"{project_name}{tags_str}")
                print(f"    {base_dir}")
            return

        # Rich table output
        _display_table(filtered, formatter)

    except Exception as e:
        formatter.error(f"Failed to list repositories: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


def _display_table(roots: list, formatter: RichOutputFormatter) -> None:
    """Display repos as a Rich table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = formatter.console or Console()

        table = Table(
            title=f"Indexed Repositories ({len(roots)})",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )

        # Columns
        table.add_column("Name", style="bold white", no_wrap=True)
        table.add_column("Path", style="dim")
        table.add_column("Tags", style="yellow")
        table.add_column("Files", justify="right", style="green")
        table.add_column("Updated", justify="right", style="dim")

        home = str(Path.home())

        for root in roots:
            base_dir = root.get("base_directory", "Unknown")
            project_name = root.get("project_name") or Path(base_dir).name
            tags = root.get("tags", [])
            file_count = root.get("file_count", 0)
            updated_at = root.get("updated_at")
            watcher_active = root.get("watcher_active", False)
            last_error = root.get("last_error")

            # Shorten path with ~
            display_path = (
                base_dir.replace(home, "~") if base_dir.startswith(home) else base_dir
            )

            # Format values
            tags_str = ", ".join(tags) if tags else "-"
            updated_str = _format_time_ago(updated_at)

            # Status indicator in name
            if last_error:
                name_display = f"[red]✗[/red] {project_name}"
            elif watcher_active:
                name_display = f"[green]●[/green] {project_name}"
            else:
                name_display = project_name

            table.add_row(
                name_display,
                display_path,
                tags_str,
                str(file_count),
                updated_str,
            )

        console.print(table)

    except ImportError:
        # Fallback if Rich not available
        for root in roots:
            base_dir = root.get("base_directory", "Unknown")
            project_name = root.get("project_name") or Path(base_dir).name
            tags = root.get("tags", [])
            tags_str = f" [{', '.join(tags)}]" if tags else ""
            formatter.info(f"• {project_name}{tags_str}")
            formatter.info(f"    {base_dir}")


async def show_command(
    args, formatter: RichOutputFormatter, config: Config | None = None
) -> None:
    """Show details for a specific indexed repository.

    Args:
        args: Parsed command-line arguments (must have 'name' attribute)
        formatter: Output formatter for displaying results
        config: Configuration object
    """
    try:
        # Get config if not provided
        if config is None:
            from chunkhound.api.cli.utils.config_factory import create_validated_config

            config, _ = create_validated_config(args, "repos")

        # Check if global mode is enabled
        if not (config.database.multi_repo and config.database.multi_repo.enabled):
            formatter.error(
                "Repos command requires global mode. Enable with:\n"
                "  export CHUNKHOUND_DATABASE__MULTI_REPO__ENABLED=true\n"
                "  export CHUNKHOUND_DATABASE__MULTI_REPO__MODE=global"
            )
            sys.exit(1)

        # In global mode, must use daemon proxy (no direct DB access to prevent locks)
        from chunkhound.api.cli.utils.database import require_daemon_or_exit

        proxy = require_daemon_or_exit(config, formatter)
        roots = proxy.list_projects()

        # Find matching root
        matching_root = None
        for root in roots:
            project_name = root.get("project_name", "")
            base_directory = root.get("base_directory", "")

            # Match by project name or base directory path
            if (
                project_name == args.name
                or base_directory == args.name
                or Path(base_directory).name == args.name
            ):
                matching_root = root
                break

        if not matching_root:
            formatter.error(f"Repository not found: {args.name}")
            sys.exit(1)

        # Display detailed information
        formatter.section_header(f"Repository: {matching_root['project_name']}")
        formatter.info(f"Base Directory: {matching_root['base_directory']}")
        formatter.info(f"File Count: {matching_root.get('file_count', 0)}")
        formatter.info(f"Indexed At: {matching_root.get('indexed_at', 'Unknown')}")
        formatter.info(f"Updated At: {matching_root.get('updated_at', 'Unknown')}")

        if config_snapshot := matching_root.get("config_snapshot"):
            formatter.verbose_info(f"Config Snapshot: {config_snapshot}")

    except Exception as e:
        formatter.error(f"Failed to show repository details: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


async def remove_command(
    args, formatter: RichOutputFormatter, config: Config | None = None
) -> None:
    """Remove an indexed repository from tracking.

    Args:
        args: Parsed command-line arguments (must have 'name' and optional 'force')
        formatter: Output formatter for displaying results
        config: Configuration object
    """
    try:
        # Get config if not provided
        if config is None:
            from chunkhound.api.cli.utils.config_factory import create_validated_config

            config, _ = create_validated_config(args, "repos")

        # Check if global mode is enabled
        if not (config.database.multi_repo and config.database.multi_repo.enabled):
            formatter.error(
                "Repos command requires global mode. Enable with:\n"
                "  export CHUNKHOUND_DATABASE__MULTI_REPO__ENABLED=true\n"
                "  export CHUNKHOUND_DATABASE__MULTI_REPO__MODE=global"
            )
            sys.exit(1)

        # In global mode, must use daemon proxy (no direct DB access to prevent locks)
        from chunkhound.api.cli.utils.database import require_daemon_or_exit

        proxy = require_daemon_or_exit(config, formatter)

        # Confirm removal unless --force
        if not getattr(args, "force", False):
            formatter.warning(f"About to remove repository: {args.name}")
            formatter.warning("This will NOT delete files, only remove from tracking")
            response = input("Continue? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                formatter.info("Removal cancelled")
                return

        cascade = getattr(args, "cascade", False)
        result = proxy.remove_project(args.name, cascade=cascade)
        if result.get("status") == "success":
            if cascade:
                formatter.success(
                    f"Removed repository and deleted data: {result.get('removed')}"
                )
            else:
                formatter.success(
                    f"Removed repository (tracking only): {result.get('removed')}"
                )
        else:
            formatter.error(f"Failed to remove: {result.get('error')}")
            sys.exit(1)

    except Exception as e:
        formatter.error(f"Failed to remove repository: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

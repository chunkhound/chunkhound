"""Tags command argument parser for ChunkHound CLI."""

import argparse
from typing import Any, cast


def add_tags_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add tags command subparser to the main parser.

    Args:
        subparsers: Subparsers object from the main argument parser

    Returns:
        The configured tags subparser
    """
    tags_parser = subparsers.add_parser(
        "tags",
        help="Manage project tags (global mode)",
        description=(
            "Manage tags for indexed projects in global database mode. "
            "Tags enable flexible categorization and filtering of projects."
        ),
    )

    # Create subparsers for tags subcommands
    tags_subparsers = tags_parser.add_subparsers(
        dest="tags_command",
        help="Tags management commands",
    )

    # tags list
    list_parser = tags_subparsers.add_parser(
        "list",
        help="List all tags or a project's tags",
        description="Show all unique tags across projects, or tags for a specific project.",
    )
    list_parser.add_argument(
        "--project",
        "-p",
        type=str,
        default=None,
        help="Show tags for a specific project (name or path)",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # tags add
    add_parser = tags_subparsers.add_parser(
        "add",
        help="Add tags to a project",
        description="Add one or more tags to an indexed project (preserves existing tags).",
    )
    add_parser.add_argument(
        "project",
        type=str,
        help="Project name or path",
    )
    add_parser.add_argument(
        "tags",
        nargs="+",
        type=str,
        help="Tags to add (space-separated, e.g., backend python work)",
    )

    # tags remove
    remove_parser = tags_subparsers.add_parser(
        "remove",
        help="Remove tags from a project",
        description="Remove one or more tags from an indexed project.",
    )
    remove_parser.add_argument(
        "project",
        type=str,
        help="Project name or path",
    )
    remove_parser.add_argument(
        "tags",
        nargs="+",
        type=str,
        help="Tags to remove (space-separated, e.g., old-tag deprecated)",
    )

    # tags set
    set_parser = tags_subparsers.add_parser(
        "set",
        help="Set tags for a project (replaces existing)",
        description="Replace all tags on a project with the specified tags.",
    )
    set_parser.add_argument(
        "project",
        type=str,
        help="Project name or path",
    )
    set_parser.add_argument(
        "tags",
        nargs="+",
        type=str,
        help="Tags to set (space-separated, replaces all existing tags)",
    )

    # tags clear
    clear_parser = tags_subparsers.add_parser(
        "clear",
        help="Remove all tags from a project",
        description="Remove all tags from an indexed project.",
    )
    clear_parser.add_argument(
        "project",
        type=str,
        help="Project name or path",
    )

    return cast(argparse.ArgumentParser, tags_parser)


__all__ = ["add_tags_subparser"]

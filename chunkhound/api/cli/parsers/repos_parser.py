"""Parser for repos command."""

import argparse


def add_repos_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add repos subcommand parser.

    Args:
        subparsers: Subparsers object to add to
    """
    repos_parser = subparsers.add_parser(
        "repos",
        help="Manage indexed repositories",
        description="View and manage indexed repositories/base directories",
    )

    repos_subparsers = repos_parser.add_subparsers(
        dest="repos_command",
        help="Repos subcommands",
        required=True,
    )

    # repos list
    list_parser = repos_subparsers.add_parser(
        "list",
        help="List all indexed repositories",
        description="Show all repositories that have been indexed",
    )
    list_parser.add_argument(
        "--tag",
        "-t",
        action="append",
        dest="tags",
        help="Filter by tag (can specify multiple: -t work -t python)",
    )
    list_parser.add_argument(
        "--active",
        action="store_true",
        help="Show only projects with active file watchers",
    )
    list_parser.add_argument(
        "--errors", action="store_true", help="Show only projects with errors"
    )
    list_parser.add_argument(
        "--sort",
        choices=["name", "updated", "files", "path"],
        default="name",
        help="Sort by field (default: name)",
    )
    list_parser.add_argument(
        "--reverse", "-r", action="store_true", help="Reverse sort order"
    )
    list_parser.add_argument(
        "--plain", action="store_true", help="Plain text output (no table)"
    )

    # repos show
    show_parser = repos_subparsers.add_parser(
        "show",
        help="Show details for a specific repository",
        description="Display detailed information about an indexed repository",
    )
    show_parser.add_argument(
        "name",
        help="Repository name or base directory path",
    )
    show_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )

    # repos remove
    remove_parser = repos_subparsers.add_parser(
        "remove",
        help="Remove a repository from tracking",
        description=(
            "Remove a repository from indexed_roots tracking. "
            "This does NOT delete files, only removes the tracking entry."
        ),
    )
    remove_parser.add_argument(
        "name",
        help="Repository name or base directory path to remove",
    )
    remove_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )
    remove_parser.add_argument(
        "--cascade",
        action="store_true",
        help="Also delete all files, chunks, and embeddings belonging to this repository",
    )

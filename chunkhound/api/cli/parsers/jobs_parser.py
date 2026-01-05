"""Parser for jobs command - manage background indexing jobs."""

import argparse


def add_jobs_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add jobs command parser.

    Args:
        subparsers: Subparsers object from main parser

    Returns:
        The created parser
    """
    jobs_parser = subparsers.add_parser(
        "jobs",
        help="Manage background indexing jobs",
        description="List, monitor, and cancel background indexing jobs.",
    )

    jobs_subparsers = jobs_parser.add_subparsers(
        dest="jobs_subcommand",
        title="subcommands",
        description="Available operations",
    )

    # jobs list
    list_parser = jobs_subparsers.add_parser(
        "list",
        help="List indexing jobs",
        description="List all indexing jobs with status.",
    )
    list_parser.add_argument(
        "--active-only",
        "-a",
        action="store_true",
        help="Only show active (running/queued) jobs",
    )
    list_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Maximum number of jobs to show (default: 20)",
    )

    # jobs status
    status_parser = jobs_subparsers.add_parser(
        "status",
        help="Show job status",
        description="Show detailed status of a specific job.",
    )
    status_parser.add_argument(
        "job_id",
        help="Job ID to check",
    )

    # jobs cancel
    cancel_parser = jobs_subparsers.add_parser(
        "cancel",
        help="Cancel a running job",
        description="Cancel a running or queued indexing job.",
    )
    cancel_parser.add_argument(
        "job_id",
        help="Job ID to cancel",
    )

    # Set default subcommand to list
    jobs_parser.set_defaults(jobs_subcommand="list")

    return jobs_parser

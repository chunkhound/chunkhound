"""Jobs command module - manage background indexing jobs."""

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chunkhound.core.config.config import Config

from ..utils.database import DaemonProxy, get_daemon_proxy_if_running
from ..utils.rich_output import RichOutputFormatter


def get_proxy_or_exit(config: "Config", formatter: RichOutputFormatter) -> DaemonProxy:
    """Get daemon proxy or exit with error message.

    Args:
        config: Configuration object
        formatter: Rich output formatter

    Returns:
        DaemonProxy if daemon is running

    Exits:
        If daemon is not running or not in global mode
    """
    proxy = get_daemon_proxy_if_running(config)
    if not proxy:
        formatter.error("Daemon not running. Start with: chunkhound daemon start")
        formatter.info(
            "Jobs are only available when the daemon is running in global mode."
        )
        sys.exit(1)
    return proxy


async def jobs_command(args: argparse.Namespace, config: "Config") -> None:
    """Execute the jobs command.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))
    subcommand = getattr(args, "jobs_subcommand", "list")

    if subcommand == "list":
        await _jobs_list(args, config, formatter)
    elif subcommand == "status":
        await _jobs_status(args, config, formatter)
    elif subcommand == "cancel":
        await _jobs_cancel(args, config, formatter)
    else:
        formatter.error(f"Unknown subcommand: {subcommand}")
        sys.exit(1)


async def _jobs_list(
    args: argparse.Namespace, config: "Config", formatter: RichOutputFormatter
) -> None:
    """List indexing jobs.

    Args:
        args: Parsed arguments
        config: Configuration
        formatter: Output formatter
    """
    proxy = get_proxy_or_exit(config, formatter)

    try:
        include_completed = not getattr(args, "active_only", False)
        limit = getattr(args, "limit", 20)
        jobs = proxy.list_jobs(include_completed=include_completed, limit=limit)

        if not jobs:
            formatter.info("No indexing jobs found.")
            return

        # Format output
        formatter.info(f"Found {len(jobs)} job(s):\n")

        for job in jobs:
            job_id = job.get("job_id", "?")
            status = job.get("status", "unknown")
            phase = job.get("phase", "")
            project = job.get("project_name") or job.get("project_path", "?")
            elapsed = job.get("elapsed_seconds", 0)
            progress = job.get("progress", {})

            # Status indicator
            if status == "running":
                status_str = f"[yellow]RUNNING[/yellow] ({phase})"
            elif status == "embedding":
                status_str = "[blue]EMBEDDING[/blue]"
            elif status == "completed":
                status_str = "[green]COMPLETED[/green]"
            elif status == "failed":
                status_str = "[red]FAILED[/red]"
            elif status == "cancelled":
                status_str = "[dim]CANCELLED[/dim]"
            elif status == "queued":
                status_str = "[cyan]QUEUED[/cyan]"
            else:
                status_str = status

            # Progress info
            files = progress.get("files_processed", 0)
            chunks = progress.get("chunks_created", 0)
            embeds = progress.get("embeddings_generated", 0)
            progress_str = f"{files} files, {chunks} chunks, {embeds} embeddings"

            # Print job info
            print(f"  {job_id}: {status_str}")
            print(f"    Project: {project}")
            print(f"    Progress: {progress_str}")
            print(f"    Elapsed: {elapsed:.1f}s")
            if job.get("error"):
                print(f"    Error: {job['error']}")
            print()

    except Exception as e:
        formatter.error(f"Failed to list jobs: {e}")
        sys.exit(1)


async def _jobs_status(
    args: argparse.Namespace, config: "Config", formatter: RichOutputFormatter
) -> None:
    """Show status of a specific job.

    Args:
        args: Parsed arguments (must have job_id)
        config: Configuration
        formatter: Output formatter
    """
    proxy = get_proxy_or_exit(config, formatter)
    job_id = getattr(args, "job_id", None)

    if not job_id:
        formatter.error("Job ID required. Usage: chunkhound jobs status <job_id>")
        sys.exit(1)

    try:
        job = proxy.get_job_status(job_id)

        # Format output
        formatter.info(f"Job: {job_id}\n")

        status = job.get("status", "unknown")
        phase = job.get("phase", "")
        project = job.get("project_name") or job.get("project_path", "?")
        elapsed = job.get("elapsed_seconds", 0)
        progress = job.get("progress", {})

        print(f"  Status: {status}")
        if phase:
            print(f"  Phase: {phase}")
        print(f"  Project: {project}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print()
        print("  Progress:")
        print(f"    Files discovered: {progress.get('files_discovered', 0)}")
        print(f"    Files processed: {progress.get('files_processed', 0)}")
        print(f"    Chunks created: {progress.get('chunks_created', 0)}")
        print(
            f"    Embeddings: {progress.get('embeddings_generated', 0)}/{progress.get('embeddings_total', 0)}"
        )

        if job.get("error"):
            print(f"\n  Error: {job['error']}")

        if job.get("result"):
            result = job["result"]
            print("\n  Result:")
            print(f"    Files processed: {result.get('files_processed', 0)}")
            print(f"    Embeddings generated: {result.get('embeddings_generated', 0)}")

    except Exception as e:
        if "404" in str(e):
            formatter.error(f"Job not found: {job_id}")
        else:
            formatter.error(f"Failed to get job status: {e}")
        sys.exit(1)


async def _jobs_cancel(
    args: argparse.Namespace, config: "Config", formatter: RichOutputFormatter
) -> None:
    """Cancel a running job.

    Args:
        args: Parsed arguments (must have job_id)
        config: Configuration
        formatter: Output formatter
    """
    proxy = get_proxy_or_exit(config, formatter)
    job_id = getattr(args, "job_id", None)

    if not job_id:
        formatter.error("Job ID required. Usage: chunkhound jobs cancel <job_id>")
        sys.exit(1)

    try:
        result = proxy.cancel_job(job_id)
        if result.get("status") == "cancelled":
            formatter.success(f"Job {job_id} cancelled")
        else:
            formatter.warning(f"Unexpected result: {result}")

    except Exception as e:
        if "404" in str(e):
            formatter.error(f"Job not found or already completed: {job_id}")
        else:
            formatter.error(f"Failed to cancel job: {e}")
        sys.exit(1)

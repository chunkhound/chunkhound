"""Quickresearch command — index into memory, then research."""

import argparse
import sys

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.directory_indexing_service import DirectoryIndexingService

from ..utils.rich_output import RichOutputFormatter
from .research import run_research, setup_embedding_llm


async def quickresearch_command(args: argparse.Namespace, config: Config) -> None:
    """Index a directory into an in-memory database and perform code research.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    formatter = RichOutputFormatter(verbose=args.verbose)

    if not args.path.is_dir():
        formatter.error(f"Path does not exist or is not a directory: {args.path}")
        sys.exit(1)

    # _quickresearch is only invoked by `chunkhound websearch` and the websearch
    # MCP tool, which point it at a tempdir of fetched pages. User-supplied
    # include/exclude (from .chunkhound.json, --config, env vars, or global
    # config) would zero out files in that tempdir. Reset to library defaults.
    # System tempdirs typically have no .gitignore/.chignore, so the remaining
    # ignore knobs (workspace_gitignore_*, chignore_file) are de-facto inert.
    config.indexing.reset_user_include_exclude()

    embedding_manager, llm_manager = setup_embedding_llm(formatter, config)

    # Single create_services call owns the only :memory: connection.
    # Unlike file-backed DBs (where a second connection hits the same data and
    # causes a hard DuckDB lock error), each duckdb.connect(":memory:") creates
    # a fully isolated, independent database — a second call would silently
    # return zero results with no error.  Pass services.indexing_coordinator
    # directly so indexing and research share the exact same connection.
    try:
        services = create_services(":memory:", config, embedding_manager)
    except Exception as e:
        formatter.error(f"Failed to initialize services: {e}")
        sys.exit(1)

    formatter.info(f"Indexing {args.path} into memory...")
    try:
        svc = DirectoryIndexingService(
            indexing_coordinator=services.indexing_coordinator,
            config=config,
            progress_callback=formatter.progress_indicator,
        )
        await svc.process_directory(args.path)
    except Exception as e:
        formatter.error(f"Indexing failed: {e}")
        sys.exit(1)

    formatter.info(
        f"Researching {args.path}"
        + (f" (filter: {args.path_filter})" if args.path_filter else "")
    )
    await run_research(
        services,
        embedding_manager,
        llm_manager,
        args.query,
        args.path_filter,
        config,
        formatter,
    )

"""Quickresearch command — index into memory, then research."""

import argparse
import os
import sys
import threading

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.directory_indexing_service import DirectoryIndexingService

from ..utils.rich_output import RichOutputFormatter
from .research import run_research, setup_embedding_llm

_ORPHAN_POLL_INTERVAL = 2.0


def _orphan_exit() -> None:
    """Die immediately — parent is gone, no reason to keep spending tokens.

    Uses os._exit so we bypass atexit and coroutine finally blocks that would
    otherwise try to write to now-dead stdio pipes. The diagnostic print is
    guarded because on the MCP path stderr is a PIPE whose read end closes
    once the parent dies — without the guard, BrokenPipeError would unwind
    before os._exit ran and the watchdog task would fail silently.
    """
    try:
        print("_quickresearch: parent died, exiting", file=sys.stderr, flush=True)
    except (BrokenPipeError, OSError):
        pass
    os._exit(1)


def _orphan_watchdog_thread(initial_ppid: int, stop: threading.Event) -> None:
    """Poll getppid() from a daemon thread; call _orphan_exit when it changes.

    Runs off the event loop so it fires even when the main thread is blocked
    in sync file I/O — the indexing_coordinator change-detection phase does
    per-file stat/hash without yielding, which would delay an asyncio-based
    watchdog. Windows is skipped by the caller because getppid() there is not
    updated on parent death (no PID-1 reparent semantics).
    """
    while not stop.wait(_ORPHAN_POLL_INTERVAL):
        if os.getppid() != initial_ppid:
            _orphan_exit()
            return  # unreachable in prod; keeps the loop testable


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

    watchdog_stop: threading.Event | None = None
    if sys.platform != "win32":
        # Immediate check closes the startup race: if our parent died during
        # interpreter startup, getppid() already returns the reparent PID and
        # would never equal the authoritative value the parent handed us.
        if os.getppid() != args.parent_pid:
            _orphan_exit()
        watchdog_stop = threading.Event()
        threading.Thread(
            target=_orphan_watchdog_thread,
            args=(args.parent_pid, watchdog_stop),
            daemon=True,
            name="quickresearch-orphan-watchdog",
        ).start()
    try:
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
    finally:
        if watchdog_stop is not None:
            # Unblock the polling loop; daemon=True means we skip the join.
            watchdog_stop.set()

"""Stdio MCP server implementation using the base class pattern.

This module implements the stdio (stdin/stdout) JSON-RPC protocol for MCP,
inheriting common initialization and lifecycle management from MCPServerBase.

CRITICAL: NO stdout output allowed - breaks JSON-RPC protocol
ARCHITECTURE: Global state required for stdio communication model
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import warnings
from importlib.metadata import version as package_version

# CRITICAL: Suppress SWIG warnings that break JSON-RPC protocol in CI
# The DuckDB Python bindings generate a DeprecationWarning that goes to stdout
# in some environments (Ubuntu CI with Python 3.12), breaking MCP protocol
warnings.filterwarnings(
    "ignore", message=".*swigvarlink.*", category=DeprecationWarning
)
from collections.abc import AsyncIterator, Iterator  # noqa: E402
from contextlib import asynccontextmanager, contextmanager, nullcontext  # noqa: E402
from typing import Any, NoReturn  # noqa: E402

from loguru import logger as loguru_logger  # noqa: E402

from chunkhound.core.config.config import Config  # noqa: E402
from chunkhound.utils.windows_constants import IS_WINDOWS  # noqa: E402
from chunkhound.version import __version__  # noqa: E402

from .base import _MCP_AVAILABLE, MCPServerBase  # noqa: E402

_ActivationSnapshot = tuple[list[tuple[str, bool]], bool]


def _snapshot_loguru_activation(loguru_logger: Any) -> _ActivationSnapshot:
    """Snapshot activation, failing loudly if loguru's private shape drifts."""
    core = getattr(loguru_logger, "_core", None)
    with getattr(core, "lock", nullcontext()):
        entries = getattr(core, "activation_list", None)
        none_status = getattr(core, "activation_none", None)
    if not isinstance(entries, list) or not isinstance(none_status, bool):
        _raise_activation_drift()
    if not all(
        isinstance(entry, tuple)
        and len(entry) == 2
        and isinstance(entry[0], str)
        and isinstance(entry[1], bool)
        for entry in entries
    ):
        _raise_activation_drift()
    return entries, none_status


def _raise_activation_drift() -> NoReturn:
    raise RuntimeError(
        f"loguru {package_version('loguru')} drifted from the pin in pyproject.toml: "
        "need a list[(str, bool)] activation_list and a bool activation_none. "
        "Re-test tests/unit/test_loguru_compat.py before widening the pin."
    )


def _restore_loguru_activation(loguru_logger: Any, saved: _ActivationSnapshot) -> None:
    """Replay through public setters; ``enable("")`` also resets the cache."""
    loguru_logger.enable("")
    setter = loguru_logger.enable if saved[1] else loguru_logger.disable
    setter(None)
    for name, status in sorted(saved[0], key=lambda item: item[0].count(".")):
        setter = loguru_logger.enable if status else loguru_logger.disable
        setter(name.removesuffix("."))  # Saved names carry a trailing dot.


@contextmanager
def _silence_loguru() -> Iterator[None]:
    """Mute loguru for the stdio run, then restore activation state.

    Sinks stay installed: ``Handler.stop()`` closes queues and drops records
    silently, so removing them would permanently break delivery for any
    in-process caller that survives ``main()``. Deferred to ``main()`` (not
    import) so importers keep their sinks and per-module disables.
    """
    saved = _snapshot_loguru_activation(loguru_logger)
    try:
        loguru_logger.disable("")
        loguru_logger.disable(None)  # Records without ``__name__`` need the flag.
        yield
    finally:
        _restore_loguru_activation(loguru_logger, saved)


@contextmanager
def _silence_stdlib_logging() -> Iterator[None]:
    """Mute stdlib logging for the stdio server run, then restore it.

    ``logging.disable`` blocks every level up to CRITICAL for all loggers, so
    per-logger thresholds add nothing. Deferred to the entry point because the
    setting is process-global: mutating it on import would silence log delivery
    inside embedded, daemon, and test processes that only import this module.
    """
    saved_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(saved_disable)


class StdioMCPServer(MCPServerBase):
    """MCP server implementation for stdio protocol.

    Uses global state as required by the stdio protocol's persistent
    connection model. All initialization happens eagerly during startup.
    """

    def __init__(self, config: Config, args: Any = None):
        """Initialize stdio MCP server.

        Args:
            config: Validated configuration object
            args: Original CLI arguments for direct path access
        """
        super().__init__(config, args=args)

        # Test-only hook: allow E2E tests to inject a sitecustomize from PYTHONPATH
        # to stub Codex CLI and force synthesis without requiring real binaries.
        # This is guarded behind CH_TEST_PATCH_CODEX and is a no-op otherwise.
        try:
            if os.getenv("CH_TEST_PATCH_CODEX") == "1":
                pp = os.environ.get("PYTHONPATH", "")
                if pp:
                    for path in pp.split(os.pathsep):
                        if path and path not in sys.path:
                            sys.path.insert(0, path)
                # Best-effort: import test helper if available
                try:
                    __import__("sitecustomize")
                except Exception:
                    pass

                # Also patch Codex provider directly to guarantee stubbed exec
                try:
                    from chunkhound.providers.llm.codex_cli_provider import (
                        CodexCLIProvider,
                    )

                    async def _stub_run_exec(  # type: ignore[override]
                        self,
                        text,
                        cwd=None,
                        max_tokens=1024,
                        timeout=None,
                        model=None,
                    ):
                        mark = os.getenv("CH_TEST_CODEX_MARK_FILE")
                        if mark:
                            try:
                                with open(mark, "a", encoding="utf-8") as f:
                                    f.write("CALLED\n")
                            except Exception:
                                pass
                        return "SYNTH_OK: codex-cli invoked"

                    def _stub_available(self) -> bool:  # pragma: no cover
                        return True

                    CodexCLIProvider._run_exec = _stub_run_exec  # type: ignore[attr-defined]
                    CodexCLIProvider._codex_available = _stub_available  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            # Silent by design in MCP mode
            pass

    def _register_tools(self) -> None:
        """Register tool handlers with the stdio server."""
        self._register_common_tool_handlers()

    @asynccontextmanager
    async def server_lifespan(self) -> AsyncIterator[dict]:
        """Manage server lifecycle with proper initialization and cleanup."""
        try:
            # Initialize services
            await self.initialize()
            self._initialization_complete.set()
            self.debug_log("Server initialization complete")

            # Yield control to server
            yield {"services": self.services, "embeddings": self.embedding_manager}

        finally:
            # Cleanup on shutdown
            await self.cleanup()

    async def run(self) -> None:
        """Run the stdio server with proper lifecycle management."""
        try:
            if _MCP_AVAILABLE:
                async with self.server_lifespan():
                    import contextlib

                    from chunkhound.utils import stdio_lines

                    frames = stdio_lines.iter_json_frames()
                    read_frame = stdio_lines.make_stdin_frame_reader(frames)

                    self.debug_log("Stdio server started, awaiting requests")
                    async with contextlib.aclosing(frames):
                        await self.serve_connection(
                            read_frame, stdio_lines.write_json_frame
                        )
            else:
                # Minimal fallback stdio: read initialize request,
                # respond with matching ID so tests can proceed
                # without the official MCP SDK.
                import json

                # Read one line from stdin (the initialize request)
                request_id = 1
                try:
                    line = sys.stdin.readline()
                    if line:
                        request = json.loads(line)
                        request_id = request.get("id", 1)
                except Exception:
                    pass

                resp = {
                    "jsonrpc": "2.0",
                    "id": request_id,  # Match client's request ID per JSON-RPC spec
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {
                            "name": "ChunkHound Code Search",
                            "version": __version__,
                        },
                        "capabilities": {"tools": {}},  # Advertise tools capability
                    },
                }
                try:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
                except Exception:
                    pass
                # Keep process alive briefly; tests terminate the process
                await asyncio.sleep(1.0)

        except KeyboardInterrupt:
            self.debug_log("Server interrupted by user")
        except Exception as e:
            # NOTE: This handler intentionally does NOT re-raise after calling
            # _respond_with_startup_error.  main() has its own except block that
            # also calls _respond_with_startup_error — if run() ever re-raises,
            # two JSON-RPC error objects would be written to stdout, corrupting
            # the protocol.  Keep this invariant: run() swallows, main() catches
            # only __init__() / pre-run() failures.
            self.debug_log(f"Server error: {e}")
            if self.debug_mode:
                import traceback

                traceback.print_exc(file=sys.stderr)
            _respond_with_startup_error(e, getattr(self, "config", None))


def _respond_with_startup_error(error: Exception, config: Any = None) -> None:
    """Write startup error to a log file and send a JSON-RPC error response.

    Called when the server crashes before the MCP stdio loop starts, so the
    MCP client (e.g. Claude Code) receives a readable error instead of a
    silent process exit.
    """
    import json
    import select
    import traceback as _tb
    from pathlib import Path as _Path

    error_msg = str(error)
    error_traceback = _tb.format_exc()

    # Generate a hint for well-known failure modes
    hint = ""
    if "Could not set lock" in error_msg or "Conflicting lock" in error_msg:
        hint = (
            "Another chunkhound process is holding a lock on the database. "
            "Run: kill $(pgrep -f 'chunkhound mcp') to stop stale processes, "
            "then reconnect."
        )

    # Always write to an error log (regardless of --debug flag)
    try:
        log_path = "/tmp/chunkhound_mcp_error.log"
        if (
            config is not None
            and getattr(config, "database", None) is not None
            and getattr(config.database, "path", None) is not None
        ):
            db_dir = _Path(config.database.path).parent
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                log_path = str(db_dir / "startup_error.log")
            except Exception:
                pass
        with open(log_path, "a") as f:
            f.write(f"ChunkHound MCP startup error: {error_msg}\n")
            if error_traceback and error_traceback != "NoneType: None\n":
                f.write(f"Traceback:\n{error_traceback}\n")
            if hint:
                f.write(f"Hint: {hint}\n")
    except Exception:
        pass

    # Attempt to respond to the pending initialize request so the MCP client
    # can surface a human-readable error message.
    try:
        # Use select with a short timeout so we don't block if stdin is empty.
        # NOTE: On Windows, select.select() only accepts sockets, not file
        # objects like sys.stdin — the call will fail and fall through to
        # the outer except, which is acceptable (best-effort semantics).
        ready, _, _ = select.select([sys.stdin], [], [], 2.0)
        if ready:
            line = sys.stdin.readline()
            if line.strip():
                req = json.loads(line)
                if req.get("method") == "initialize":
                    full_msg = f"ChunkHound startup failed: {error_msg}"
                    if hint:
                        full_msg += f"  |  {hint}"
                    error_resp = {
                        "jsonrpc": "2.0",
                        "id": req.get("id"),
                        "error": {
                            "code": -32000,
                            "message": full_msg,
                        },
                    }
                    sys.stdout.write(json.dumps(error_resp) + "\n")
                    sys.stdout.flush()
    except Exception:
        pass


async def main(args: Any = None, config: Config | None = None) -> None:
    """Main entry point for the MCP stdio server.

    Args:
        args: Pre-parsed arguments. If None, will parse from sys.argv.
        config: Pre-validated configuration. If provided (e.g. by
            ``mcp_command``, which already validated it once), validation is
            not repeated; otherwise a fresh one is built and validated here.
    """
    with _silence_stdlib_logging(), _silence_loguru():
        await _serve(args, config)


async def _serve(args: Any = None, config: Config | None = None) -> None:
    import argparse

    from chunkhound.api.cli.utils.config_factory import create_validated_config
    from chunkhound.mcp_server.common import add_common_mcp_arguments

    if args is None:
        # Direct invocation - parse arguments
        parser = argparse.ArgumentParser(
            description="ChunkHound MCP stdio server",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Add common MCP arguments
        add_common_mcp_arguments(parser)
        # Parse arguments
        args = parser.parse_args()

    # Mark process as MCP mode so downstream code avoids interactive prompts
    os.environ["CHUNKHOUND_MCP_MODE"] = "1"

    if config is None:
        # Create and validate configuration
        config, validation_errors = await create_validated_config(args, "mcp")

        if validation_errors:
            msg = "; ".join(str(e) for e in validation_errors)
            _respond_with_startup_error(
                Exception(f"Configuration errors: {msg}"), config
            )
            sys.exit(1)

    # Create and run the stdio server
    try:
        server = StdioMCPServer(config, args=args)
        await server.run()
    except Exception as e:
        _respond_with_startup_error(e, config)
        sys.exit(1)


def main_sync() -> None:
    """Synchronous wrapper for CLI entry point."""
    if IS_WINDOWS:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="backslashreplace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(errors="backslashreplace")
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()

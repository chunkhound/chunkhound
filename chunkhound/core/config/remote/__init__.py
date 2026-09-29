"""Remote-configuration fetch pipeline.

Public entry: ``run_remote_config_fetch(args, command)``. Idempotent and
silent on any recoverable failure — the caller (``create_validated_config``)
proceeds with whatever the loader sees on disk. Only disk-write failures
during backup+write escalate to ``sys.exit(1)``: a broken persistence step
means the pipeline's view of "what's on disk" no longer matches reality,
and continuing would silently apply the wrong config on the next run.
"""

from typing import Any

from chunkhound.utils.logging_guard import log_if_not_mcp

from . import pipeline

# Internal subcommands whose parents (``mcp``, ``websearch``) already ran the
# pipeline and persisted the result to disk. Without this gate, each parent →
# child hop would pay the 10s fetch tax a second time. Add any new child that
# inherits its parent's fetched config here.
_SUBPROCESS_SKIP: frozenset[str] = frozenset({"_daemon", "_quickresearch"})

# Flag dests checked when warning that a skipped subcommand was invoked with
# remote-config args — those flags are accepted by the shared parser but
# never acted on here, so passing them by hand is a silent no-op without
# this warning.
_REMOTE_CONFIG_DESTS: tuple[str, ...] = (
    "remote_config_url",
    "remote_config_auth_header",
)


async def run_remote_config_fetch(args: Any, command: str) -> None:
    """Fetch, validate, and persist remote configuration for this invocation.

    Delegates to ``pipeline.run``. See that module for the step-by-step
    contract; this function is a thin public entry point.
    """
    if command in _SUBPROCESS_SKIP:
        ignored = [
            f"--{dest.replace('_', '-')}"
            for dest in _REMOTE_CONFIG_DESTS
            if getattr(args, dest, None) is not None
        ]
        if ignored:
            log_if_not_mcp(
                "WARNING",
                "Remote-config flag(s) ignored — {} inherits fetched config "
                "from its parent and never runs the fetch itself: {}",
                command,
                ", ".join(ignored),
            )
        return
    await pipeline.run(args, command)


__all__ = ["run_remote_config_fetch"]

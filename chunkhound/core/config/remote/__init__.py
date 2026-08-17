"""Remote-configuration fetch pipeline.

Public entry: ``run_remote_config_fetch(args, command)``. Idempotent and
silent on any recoverable failure — the caller (``create_validated_config``)
proceeds with whatever the loader sees on disk. Only disk-write failures
during backup+write escalate to ``sys.exit(1)``: a broken persistence step
means the pipeline's view of "what's on disk" no longer matches reality,
and continuing would silently apply the wrong config on the next run.
"""

from typing import Any

from . import pipeline


async def run_remote_config_fetch(args: Any, command: str) -> None:
    """Fetch, validate, and persist remote configuration for this invocation.

    Delegates to ``pipeline.run``. See that module for the step-by-step
    contract; this function is a thin public entry point.
    """
    await pipeline.run(args, command)


__all__ = ["run_remote_config_fetch"]

"""MCP-safe logging guard for DuckDB provider modules.

MCP uses stdout for JSON-RPC message framing — any stray log output
breaks the protocol.  All user-facing log messages in compaction and
disconnect code paths must go through :func:`log_if_not_mcp`.

The canonical value is ``"1"`` — set by ``mcp_command()`` and
``StdioMCPServer``.  This guard uses strict ``== "1"`` rather than
a truthy check to match ``run.py`` and to avoid false suppression
when the variable is set to other values (e.g. ``"true"``).
"""

import os
from typing import Any

from loguru import logger


def log_if_not_mcp(level: str, message: str, *args: Any, **kwargs: Any) -> None:
    """Emit a log message unless running in MCP mode.

    MCP uses stdout for JSON-RPC message framing — any stray log output
    breaks the protocol.  All user-facing log messages in compaction and
    disconnect code paths must go through this guard.

    Uses ``logger.opt(depth=1)`` so the log source reflects the *caller*,
    not this function.  Do NOT wrap ``log_if_not_mcp`` in another helper
    — the depth offset would break caller attribution.
    """
    if os.environ.get("CHUNKHOUND_MCP_MODE") == "1":
        return
    logger.opt(depth=1).log(level.upper(), message, *args, **kwargs)

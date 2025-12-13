"""Lightweight MCP package initializer with lazy imports.

Avoid importing heavy transport dependencies (e.g., MCP SDK) at import
time to prevent environment-specific failures during test collection.
Transport servers are resolved lazily either via getters or module-level
__getattr__.
"""

from typing import TYPE_CHECKING

from .base import MCPServerBase
from .tools import TOOL_REGISTRY

if TYPE_CHECKING:  # type checkers only; avoid runtime hard deps
    from .stdio import StdioMCPServer as _StdioMCPServer  # noqa: F401


def get_stdio_server_class():
    """Return the stdio MCP server class with a lazy import."""
    from .stdio import StdioMCPServer

    return StdioMCPServer


def __getattr__(name: str):  # PEP 562: lazy attribute access on module
    if name == "StdioMCPServer":
        return get_stdio_server_class()
    raise AttributeError(name)


__all__ = [
    "MCPServerBase",
    "TOOL_REGISTRY",
    # Legacy names resolved lazily for compatibility
    "StdioMCPServer",
    # Explicit getters
    "get_stdio_server_class",
]

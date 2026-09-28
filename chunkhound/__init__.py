"""Local-first research engine for AI agents via MCP, with a CLI companion."""

from .version import __version__

__author__ = "Ofri Wolfus"

# Import modules only when needed to avoid dependency issues during setup
__all__ = [
    "Database",
    "CodeParser",
    "__version__",
]


def __getattr__(name: str):
    """Lazy import to avoid dependency issues during setup."""
    if name == "Database":
        from .database import Database

        return Database
    elif name == "CodeParser":
        from .parser import CodeParser

        return CodeParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

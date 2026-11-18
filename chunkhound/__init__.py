"""ChunkHound - Local-first semantic code search with vector and regex capabilities."""
# ci: merge queue test

from .platform_guard import disable_platform_wmi_queries
from .version import __version__

# First, before any module calls platform: see chunkhound/platform_guard.py.
disable_platform_wmi_queries()

__author__ = "Ofri Wolfus"
__description__ = "Local-first semantic code search with vector and regex capabilities"

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

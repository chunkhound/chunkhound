"""Local-first research engine for AI agents via MCP, with a CLI companion."""

from .version import __version__

__author__ = "Ofri Wolfus"

# pyproject.toml `project.description` is the single source of truth; this
# docstring copy is decorative and kept in sync only by
# tests/unit/test_package_description.py. -OO strips docstrings, so fall back
# to installed package metadata (built from pyproject by the installer).
def _package_description() -> str:
    if __doc__:
        return __doc__.rstrip(".")
    try:
        from importlib.metadata import metadata

        return metadata("chunkhound")["Summary"] or ""
    except Exception:  # metadata absent while the package itself is installing
        return ""


__description__ = _package_description()

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

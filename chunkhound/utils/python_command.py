"""Build commands that run installed Python modules from a project directory."""

import sys


def python_module_command(module: str) -> list[str]:
    """Keep the project cwd for file access while loading installed modules."""
    if sys.version_info >= (3, 11):
        return [sys.executable, "-P", "-m", module]
    # Python 3.10 lacks -P. Starting elsewhere and calling os.chdir at daemon
    # startup would change cwd handling for config that falls back to Path.cwd().
    return [sys.executable, "-m", module]

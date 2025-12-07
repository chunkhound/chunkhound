"""Compatibility shim for the ChunkHound search evaluation harness.

The implementation has been moved to ``chunkhound.tools.eval.search`` to keep
this module small and focused. This wrapper preserves the original entrypoint:

    uv run python -m chunkhound.tools.eval_search --help
"""

from __future__ import annotations

from chunkhound.tools.eval.search import main as _main


def main() -> None:
    """Entry point for python -m chunkhound.tools.eval_search."""
    _main()


if __name__ == "__main__":
    main()


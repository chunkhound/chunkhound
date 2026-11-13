"""ChunkHound CLI API package - modular command-line interface."""

from chunkhound.utils.blas_threads import limit_blas_threads

# First, before anything imports numpy (see chunkhound/utils/blas_threads.py).
limit_blas_threads()

# Removed eager imports to eliminate 958-module import cascade
# Commands are now imported lazily in main.py when needed

__all__ = [
    "run_command",
    "mcp_command",
]

"""Entry point for running ChunkHound as a module: python -m chunkhound.

This enables:
    python -m chunkhound [command] [args]
    python -m chunkhound mcp http --port 5173

Used by the daemon manager when spawning background processes.
"""

from chunkhound.api.cli.main import main

if __name__ == "__main__":
    main()

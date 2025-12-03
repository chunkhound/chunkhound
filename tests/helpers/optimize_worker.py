"""Worker script for testing optimize_tables() in subprocess.

This script is used by test_lancedb_optimize_lock.py to test multi-process
FileLock behavior. It connects to a LanceDB database and calls optimize_tables().

Exit codes:
- 0: Success (optimization completed or skipped due to lock)
- 1: Error during execution

Usage:
    python optimize_worker.py <db_path> [--debug]
"""

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: optimize_worker.py <db_path> [--debug]", file=sys.stderr)
        return 1

    db_path = Path(sys.argv[1])
    debug = "--debug" in sys.argv

    if debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    try:
        from chunkhound.providers.database.lancedb_provider import LanceDBProvider

        provider = LanceDBProvider(db_path=db_path, base_directory=db_path.parent)
        provider.connect()
        provider.optimize_tables()  # Will skip silently if lock held (Timeout caught)
        provider.close()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

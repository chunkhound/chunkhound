"""Project directory detection utilities for MCP server."""

import os
from pathlib import Path


def _exists_or_false(path: Path) -> bool:
    # Path.exists() on Python <3.12 propagates PermissionError (EACCES /
    # ERROR_ACCESS_DENIED) — e.g. probing a locked-down C:\Users on Windows
    # aborts the entire walk. Treat any OSError as "not present" so discovery
    # continues past inaccessible parents.
    try:
        return path.exists()
    except OSError:
        return False


def find_project_root(start_path: Path | None = None) -> Path:
    """
    Find project root directory using strict requirements.

    Project root is determined ONLY by:
    1. A positional CLI argument (passed as start_path)
    2. The presence of .chunkhound.json in the current directory
    3. Everything else is considered an error and terminates the process

    Args:
        start_path: Starting directory from CLI positional argument (optional)

    Returns:
        Path to project root directory

    Raises:
        SystemExit: If no valid project root can be determined
    """
    import sys

    if start_path is not None:
        # CLI positional argument provided - use it directly
        project_root = Path(start_path).resolve()
        if not _exists_or_false(project_root):
            print(
                f"Error: Specified project directory does not exist or is not accessible: {project_root}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not project_root.is_dir():
            print(
                f"Error: Specified project path is not a directory: {project_root}",
                file=sys.stderr,
            )
            sys.exit(1)
        return project_root

    # No CLI argument - walk up tree looking for project markers
    current = Path.cwd()
    home = Path.home()

    # Inspect home before breaking (Windows: CWD may equal Path.home()).
    while True:
        # Priority 1: Explicit .chunkhound.json marker
        if _exists_or_false(current / ".chunkhound.json"):
            return current

        # Priority 2: Existing database directory
        if _exists_or_false(current / ".chunkhound" / "db"):
            return current

        # Priority 3: Git repository root
        if _exists_or_false(current / ".git"):
            return current

        if current == home or current == current.parent:
            break
        current = current.parent

    # No markers found - provide helpful error
    print("Error: No ChunkHound project found.", file=sys.stderr)
    print(f"Searched upward from: {Path.cwd()}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Expected to find one of:", file=sys.stderr)
    print("  - .chunkhound.json (explicit project marker)", file=sys.stderr)
    print("  - .chunkhound/db (indexed database)", file=sys.stderr)
    print("  - .git (git repository root)", file=sys.stderr)
    print("", file=sys.stderr)
    print("Solutions:", file=sys.stderr)
    print("  1. Create .chunkhound.json in your project root", file=sys.stderr)
    print("  2. Run 'chunkhound index .' from project root first", file=sys.stderr)
    print(
        "  3. Pass explicit path: chunkhound <command> /path/to/project",
        file=sys.stderr,
    )
    sys.exit(1)


def get_project_database_path() -> Path:
    """
    Get the database path for the current project.

    NOTE: This function is deprecated. The Config class now handles
    database path resolution internally. Use Config().database.path instead.

    Returns:
        Path to database file in project root
    """
    # Check environment variable first
    db_path_env = os.environ.get("CHUNKHOUND_DATABASE__PATH") or os.environ.get(
        "CHUNKHOUND_DB_PATH"
    )
    if db_path_env:
        return Path(db_path_env)

    # Find project root and use default database name
    project_root = find_project_root()
    return project_root / ".chunkhound" / "db"

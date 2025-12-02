"""Path utility functions for ChunkHound."""

from pathlib import Path


def normalize_path_for_lookup(
    input_path: str | Path, base_dir: Path | None = None
) -> str:
    """Normalize path for database lookup operations.

    Converts absolute paths to relative paths using base directory,
    and ensures forward slash normalization for cross-platform compatibility.

    Resolves regular files to handle platform quirks (Windows 8.3 short names,
    /var -> /private/var on macOS), but preserves symlink logical paths to
    support git worktrees where symlinks may point outside the base directory.

    Args:
        input_path: Path to normalize (can be absolute or relative)
        base_dir: Base directory for relative path calculation (required for absolute paths)

    Returns:
        Normalized relative path with forward slashes

    Raises:
        ValueError: If absolute path is provided without base_dir, or if path is not under base_dir
    """
    path_obj = Path(input_path)

    # If path is already relative, just normalize slashes
    if not path_obj.is_absolute():
        return path_obj.as_posix()

    # For absolute paths, base_dir is REQUIRED
    if base_dir is None:
        raise ValueError(
            f"Cannot normalize absolute path without base_dir: {input_path}. "
            f"This indicates a bug - base directory should always be available from config."
        )

    try:
        # Resolve base_dir to canonical form (handles /var -> /private/var on macOS)
        resolved_base_dir = base_dir.resolve()

        # Only resolve non-symlinks to preserve logical paths for worktree symlinks
        # Regular files need resolution for Windows 8.3 short name compatibility
        if path_obj.is_symlink():
            path_to_use = path_obj
        else:
            path_to_use = path_obj.resolve()

        try:
            relative_path = path_to_use.relative_to(resolved_base_dir)
            return relative_path.as_posix()
        except ValueError:
            # Fallback for symlinks with unresolved base
            relative_path = path_obj.relative_to(base_dir)
            return relative_path.as_posix()
    except ValueError:
        # Path is not under base_dir - this should not happen in normal operation
        raise ValueError(
            f"Path {input_path} is not under base directory {base_dir}. "
            f"This indicates a configuration or indexing issue."
        )

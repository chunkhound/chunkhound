"""Path utility functions for ChunkHound."""

from pathlib import Path


def resolve_path_for_relative(path: Path, base_dir: Path) -> tuple[Path, Path]:
    """Resolve path and base_dir for relative_to() computation.

    Preserves symlink logical paths (git worktree support).
    Resolves regular files (Windows 8.3 short names, macOS /var -> /private/var).

    Args:
        path: File path to resolve
        base_dir: Base directory for relative path calculation

    Returns:
        Tuple of (path_to_use, resolved_base_dir) ready for relative_to()
    """
    resolved_base = base_dir.resolve()
    if path.is_symlink():
        return path, resolved_base
    return path.resolve(), resolved_base


def get_relative_path_safe(path: Path, base_dir: Path) -> Path:
    """Get relative path, handling symlinks and platform quirks.

    Preserves symlink logical paths (git worktree support).
    Resolves regular files (Windows 8.3 short names, macOS /var -> /private/var).

    Args:
        path: File path (absolute)
        base_dir: Base directory for relative path calculation

    Returns:
        Relative path from base_dir to path

    Raises:
        ValueError: If path is not under base_dir
    """
    path_to_use, resolved_base = resolve_path_for_relative(path, base_dir)
    try:
        return path_to_use.relative_to(resolved_base)
    except ValueError:
        # Fallback for edge cases (e.g., symlink with different base resolution)
        return path.relative_to(base_dir)


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
        return get_relative_path_safe(path_obj, base_dir).as_posix()
    except ValueError:
        # Path is not under base_dir - this should not happen in normal operation
        raise ValueError(
            f"Path {input_path} is not under base directory {base_dir}. "
            f"This indicates a configuration or indexing issue."
        )

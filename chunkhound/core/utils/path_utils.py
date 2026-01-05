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
    input_path: str | Path, base_dir: Path | None = None, use_absolute: bool = False
) -> str:
    """Normalize path for database lookup operations.

    In per-repo mode (use_absolute=False): Converts to relative paths
    In global mode (use_absolute=True): Keeps as absolute paths

    Converts absolute paths to relative paths using base directory,
    and ensures forward slash normalization for cross-platform compatibility.

    Resolves regular files to handle platform quirks (Windows 8.3 short names,
    /var -> /private/var on macOS), but preserves symlink logical paths to
    support git worktrees where symlinks may point outside the base directory.

    Args:
        input_path: Path to normalize (can be absolute or relative)
        base_dir: Base directory for relative path calculation
        use_absolute: If True, return absolute paths (global mode).
                     If False, return relative paths (per-repo mode, default)

    Returns:
        Normalized path with forward slashes (relative or absolute based on mode)

    Raises:
        ValueError: If path normalization fails
    """
    path_obj = Path(input_path)

    # GLOBAL MODE: Return absolute paths
    if use_absolute:
        # Resolve to canonical form (handles symlinks)
        resolved_path = path_obj.resolve()
        return resolved_path.as_posix()

    # PER-REPO MODE: Return relative paths
    # If path is already relative, just normalize slashes
    if not path_obj.is_absolute():
        return path_obj.as_posix()

    # For absolute paths, base_dir is REQUIRED in per-repo mode
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

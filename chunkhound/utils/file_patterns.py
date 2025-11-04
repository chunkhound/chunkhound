"""File pattern matching utilities for directory traversal.

# FILE_CONTEXT: Shared pattern matching logic for file discovery
# ROLE: Provides picklable helper functions for both sequential and parallel file discovery
# RATIONALE: Eliminates code duplication between worker processes and main thread
#
# PERFORMANCE OPTIMIZATIONS:
# - Compiled regex patterns cached for 2-3x speedup vs fnmatch on repeated matches
# - Module-level functions enable multiprocessing serialization (picklability)
# - Early pattern matching avoids expensive directory traversal of excluded subtrees
"""

import os
import re
from fnmatch import translate, fnmatch
from pathlib import Path
from typing import Pattern


def compile_pattern(pattern: str, cache: dict[str, Pattern[str]]) -> Pattern[str]:
    """Compile fnmatch pattern to regex with caching.

    Args:
        pattern: fnmatch-style pattern (e.g., "*.py", "**/*.md")
        cache: Dictionary to cache compiled patterns

    Returns:
        Compiled regex pattern

    Note:
        Modifies cache as side effect for performance
    """
    if pattern not in cache:
        regex_pattern = translate(pattern)
        cache[pattern] = re.compile(regex_pattern)
    return cache[pattern]


def should_exclude_path(
    path: Path,
    base_dir: Path,
    patterns: list[str],
    cache: dict[str, Pattern[str]],
) -> bool:
    """Check if a path should be excluded based on patterns.

    Args:
        path: Path to check
        base_dir: Base directory for relative path calculation
        patterns: Exclusion patterns to match against
        cache: Compiled pattern cache

    Returns:
        True if path should be excluded, False otherwise
    """
    try:
        rel_path = path.relative_to(base_dir)
    except ValueError:
        # Path is not under base directory, use as-is
        rel_path = path

    rel_path_str = rel_path.as_posix()
    path_name = path.name

    for exclude_pattern in patterns:
        # Handle **/ prefix and /** suffix patterns (directory subtree excludes)
        if exclude_pattern.startswith("**/") and exclude_pattern.endswith("/**"):
            # Example: **/.venv/**, **/.venv*/**, **/node_modules/**
            target = exclude_pattern[3:-3]

            # If the target segment contains wildcard characters, evaluate it
            # against each path component using fnmatch so patterns like
            # "**/.venv*/**" correctly match ".venv-docling" directories.
            if "*" in target or "?" in target or ("[" in target and "]" in target):
                # Check both relative and absolute component views for robustness
                for part in rel_path.parts:
                    if fnmatch(part, target):
                        return True
                for part in path.parts:
                    if fnmatch(part, target):
                        return True
            else:
                # Fast path for exact directory segment matches
                if target in rel_path.parts or target in path.parts:
                    return True
        elif exclude_pattern.startswith("**/"):
            # Treat "**/..." like include logic: try full and simple variants
            compiled_full = compile_pattern(exclude_pattern, cache)
            compiled_simple = compile_pattern(exclude_pattern[3:], cache)
            if (
                compiled_full.match(rel_path_str)
                or compiled_simple.match(rel_path_str)
                or compiled_simple.match(path_name)
            ):
                return True
        else:
            # Regular pattern - use compiled regex for faster matching
            compiled = compile_pattern(exclude_pattern, cache)
            if compiled.match(rel_path_str) or compiled.match(path_name):
                return True
    return False


def should_include_file(
    file_path: Path,
    root_dir: Path,
    patterns: list[str],
    cache: dict[str, Pattern[str]],
) -> bool:
    """Check if a file matches any of the include patterns.

    Args:
        file_path: File path to check
        root_dir: Root directory for relative path calculation
        patterns: Include patterns to match against
        cache: Compiled pattern cache

    Returns:
        True if file should be included, False otherwise
    """
    rel_path = file_path.relative_to(root_dir)
    rel_path_str = rel_path.as_posix()
    filename = file_path.name

    for pattern in patterns:
        # Handle **/ prefix patterns (common from CLI conversion)
        if pattern.startswith("**/"):
            simple_pattern = pattern[3:]  # Remove **/ prefix (e.g., *.md from **/*.md)

            # Use compiled patterns for consistent performance
            compiled_full = compile_pattern(pattern, cache)
            compiled_simple = compile_pattern(simple_pattern, cache)

            # Match against:
            # 1. Full relative path with **/ for nested files
            # 2. Simple pattern for any depth
            # 3. Filename only for simple patterns
            if (
                compiled_full.match(rel_path_str)
                or compiled_simple.match(rel_path_str)
                or compiled_simple.match(filename)
            ):
                return True
        else:
            # Use compiled regex for faster matching
            compiled = compile_pattern(pattern, cache)
            if compiled.match(rel_path_str) or compiled.match(filename):
                return True
    return False


def load_gitignore_patterns(dir_path: Path, root_dir: Path) -> list[str]:
    """Load and parse .gitignore file for a directory.

    Args:
        dir_path: Directory containing .gitignore
        root_dir: Root directory for relative pattern resolution

    Returns:
        List of gitignore patterns converted to exclude patterns

    Note:
        Patterns starting with / are made relative to root_dir.
        Other patterns are made recursive with **/ prefix.
    """
    gitignore_path = dir_path / ".gitignore"
    if not gitignore_path.exists():
        return []

    try:
        with open(gitignore_path, encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()

        patterns_from_gitignore = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Handle trailing slash (directory-only patterns)
            if line.endswith("/"):
                line = line[:-1]

            # Convert gitignore pattern to our exclude pattern format
            if line.startswith("/"):
                # Pattern relative to gitignore's directory
                rel_from_root = dir_path.relative_to(root_dir)
                if rel_from_root == Path("."):
                    # At root level
                    patterns_from_gitignore.append(line[1:])
                    patterns_from_gitignore.append(f"{line[1:]}/**")
                else:
                    # In subdirectory
                    patterns_from_gitignore.append(
                        (rel_from_root / line[1:]).as_posix()
                    )
                    patterns_from_gitignore.append(
                        f"{(rel_from_root / line[1:]).as_posix()}/**"
                    )
            else:
                # Recursive pattern
                rel_from_root = dir_path.relative_to(root_dir)
                if rel_from_root == Path("."):
                    # Use shared normalization utility to avoid double prefixing
                    patterns_from_gitignore.append(normalize_include_pattern(line))
                else:
                    patterns_from_gitignore.append(
                        f"{rel_from_root.as_posix()}/**/{line}"
                    )
                    patterns_from_gitignore.append(f"{rel_from_root.as_posix()}/{line}")

        return patterns_from_gitignore
    except (OSError, Exception) as e:
        # Return empty list on error - caller can log if needed
        return []


def scan_directory_files(
    directory: Path,
    patterns: list[str],
    exclude_patterns: list[str],
    gitignore_patterns: list[str] | None = None,
) -> list[Path]:
    """Scan files in a single directory (non-recursive) with pattern filtering.

    Args:
        directory: Directory to scan
        patterns: Include patterns
        exclude_patterns: Exclude patterns
        gitignore_patterns: Optional gitignore patterns to apply

    Returns:
        List of file paths that match criteria
    """
    files = []
    pattern_cache: dict[str, Pattern[str]] = {}

    try:
        for item in directory.iterdir():
            if item.is_file():
                # Check against exclude patterns
                if should_exclude_path(
                    item, directory, exclude_patterns, pattern_cache
                ):
                    continue

                # Check against gitignore patterns
                if gitignore_patterns and should_exclude_path(
                    item, directory, gitignore_patterns, pattern_cache
                ):
                    continue

                # Check against include patterns
                if should_include_file(item, directory, patterns, pattern_cache):
                    files.append(item)
    except (PermissionError, OSError):
        # Silently skip on errors - caller can log if needed
        pass

    return files


def walk_directory_tree(
    start_path: Path,
    root_directory: Path,
    patterns: list[str],
    exclude_patterns: list[str],
    parent_gitignores: dict[Path, list[str]],
    use_inode_ordering: bool = False,
) -> tuple[list[Path], dict[Path, list[str]]]:
    """Core directory traversal logic shared by sequential and parallel discovery.

    DESIGN: Single source of truth for directory walking with gitignore support.
    This function contains all the os.walk logic that was previously duplicated.

    RACE CONDITION SAFETY: Handles directories deleted during traversal gracefully.

    Args:
        start_path: Starting directory to traverse
        root_directory: Root directory for relative path resolution
        patterns: File patterns to include
        exclude_patterns: Patterns to exclude
        parent_gitignores: Pre-loaded gitignore patterns from parent directories
        use_inode_ordering: Sort directories by inode for disk locality

    Returns:
        Tuple of (list of file paths found, updated gitignore_patterns dict)
    """
    files = []
    gitignore_patterns: dict[Path, list[str]] = parent_gitignores.copy()
    pattern_cache: dict[str, Pattern[str]] = {}  # Local pattern cache

    # Walk the directory tree using os.walk() for efficiency
    # SAFETY: Handle race condition where start_path is deleted before walk begins
    try:
        walk_iter = os.walk(start_path, topdown=True)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        # Directory deleted, became a file, or permission denied before walk started
        return files, gitignore_patterns

    for dirpath, dirnames, filenames in walk_iter:
        current_dir = Path(dirpath)

        # Load gitignore for current directory
        gitignore_patterns[current_dir] = load_gitignore_patterns(
            current_dir, root_directory
        )

        # Combine gitignore patterns from parents
        all_gitignore_patterns = []
        check_dir = current_dir
        while check_dir >= root_directory:
            if check_dir in gitignore_patterns:
                all_gitignore_patterns.extend(gitignore_patterns[check_dir])
            if check_dir == root_directory:
                break
            check_dir = check_dir.parent

        # Filter directories in-place (topdown=True enables this optimization)
        # This is KEY: excluded directories are not traversed at all
        dirs_to_remove = []
        for dirname in dirnames:
            dir_path = current_dir / dirname
            if should_exclude_path(
                dir_path, root_directory, exclude_patterns, pattern_cache
            ) or (
                all_gitignore_patterns
                and should_exclude_path(
                    dir_path, root_directory, all_gitignore_patterns, pattern_cache
                )
            ):
                dirs_to_remove.append(dirname)

        for dirname in dirs_to_remove:
            dirnames.remove(dirname)

        # Optional inode ordering for HDD disk locality
        if use_inode_ordering:
            try:
                dirnames.sort(key=lambda d: os.stat(current_dir / d).st_ino)
            except (OSError, AttributeError):
                # st_ino not available on all platforms (Windows)
                pass

        # Process files
        for filename in filenames:
            file_path = current_dir / filename

            if should_exclude_path(
                file_path, root_directory, exclude_patterns, pattern_cache
            ):
                continue

            if all_gitignore_patterns and should_exclude_path(
                file_path, root_directory, all_gitignore_patterns, pattern_cache
            ):
                continue

            if should_include_file(file_path, root_directory, patterns, pattern_cache):
                files.append(file_path)

    return files, gitignore_patterns

# ---------------------------------------------------------------------------
# Normalization helpers (shared across services)
# ---------------------------------------------------------------------------

def normalize_include_pattern(pattern: str) -> str:
    """Ensure include pattern starts with "**/" prefix without double-prefixing.

    Examples:
    - "*.py"      -> "**/*.py"
    - "**/*.py"   -> "**/*.py" (unchanged)
    - "README"    -> "**/README"
    """
    return pattern if pattern.startswith("**/") else f"**/{pattern}"


def normalize_include_patterns(patterns: list[str]) -> list[str]:
    """Normalize a list of include patterns with consistent "**/" prefixing."""
    return [normalize_include_pattern(p) for p in patterns]


def walk_subtree_worker(
    subtree_path: Path,
    root_directory: Path,
    patterns: list[str],
    exclude_patterns: list[str],
    parent_gitignores: dict[Path, list[str]],
    use_inode_ordering: bool = False,
) -> tuple[list[Path], list[str]]:
    """Worker function for parallel directory traversal (must be module-level for pickling).

    MULTIPROCESSING: This function must remain at module level to be picklable by ProcessPoolExecutor.
    Each worker process creates its own pattern cache to avoid sharing state.

    RACE CONDITION SAFETY: Gracefully handles directories deleted during traversal.

    Args:
        subtree_path: Directory subtree to traverse
        root_directory: Root directory for relative path resolution
        patterns: File patterns to include
        exclude_patterns: Patterns to exclude
        parent_gitignores: Pre-loaded gitignore patterns from parent directories
        use_inode_ordering: Sort directories by inode for disk locality (HDD optimization)

    Returns:
        Tuple of (list of file paths found, list of error messages)
    """
    errors = []

    try:
        # Use shared directory traversal logic
        files, _ = walk_directory_tree(
            subtree_path,
            root_directory,
            patterns,
            exclude_patterns,
            parent_gitignores,
            use_inode_ordering,
        )
        return files, errors
    except (FileNotFoundError, NotADirectoryError) as e:
        # Subtree was deleted/moved after being queued for processing
        error_msg = f"Subtree {subtree_path} deleted during traversal: {e}"
        errors.append(error_msg)
        return [], errors
    except PermissionError as e:
        # Permission denied accessing subtree
        error_msg = f"Permission denied accessing {subtree_path}: {e}"
        errors.append(error_msg)
        return [], errors
    except Exception as e:
        # Unexpected error - capture for debugging
        error_msg = (
            f"Unexpected error in worker for {subtree_path}: {type(e).__name__}: {e}"
        )
        errors.append(error_msg)
        return [], errors

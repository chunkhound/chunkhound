"""Shared path-filter validation, normalization, and matching.

Single source of truth for interpreting a user-supplied ``path_filter``:
DB-backed search translates the normalized filter into a SQL LIKE pattern,
diff-backed search matches it against in-memory chunk paths. Both layers must
agree on file-vs-directory classification, otherwise the same query returns
inconsistent results depending on the vector source.
"""

# Rejected outright: traversal (``..``, ``~``) and glob/control characters that
# would otherwise leak into LIKE patterns or log injection.
_FORBIDDEN_PATTERNS = ("..", "~", "*", "?", "[", "]", "\0", "\n", "\r")


def validate_and_normalize_path_filter(path_filter: str | None) -> str | None:
    """Validate and normalize a path filter for security and consistency.

    Args:
        path_filter: User-provided path filter

    Returns:
        Normalized relative path; directory filters carry a trailing slash.
        None when the filter is absent or blank.

    Raises:
        ValueError: If path contains dangerous patterns
    """
    if path_filter is None:
        return None

    normalized = path_filter.strip()
    if not normalized:
        return None

    for pattern in _FORBIDDEN_PATTERNS:
        if pattern in normalized:
            raise ValueError(f"Path filter contains forbidden pattern: {pattern}")

    # Forward slashes only, relative paths only.
    normalized = normalized.replace("\\", "/").lstrip("/")
    if not normalized:
        return None

    if not normalized.endswith("/"):
        normalized += "" if _has_file_extension(normalized) else "/"

    return normalized


def _has_file_extension(normalized: str) -> bool:
    """Classify the last path component as a file (True) or directory (False).

    A file extension is a dot appearing AFTER the first character. Leading-dot
    names without a second dot (.github, .env) are directories; leading-dot
    names with an extension (.eslintrc.js) are files.
    """
    return "." in normalized.split("/")[-1][1:]


def path_matches_filter(file_path: str, normalized_filter: str) -> bool:
    """In-memory equivalent of ``CONCAT('/', path) LIKE <pattern>`` filtering.

    Directory filters match anywhere in the path; file filters are
    right-anchored so ``module.py`` does not match ``module.py.bak``.
    """
    anchored = "/" + str(file_path).replace("\\", "/").lstrip("/")
    needle = "/" + normalized_filter
    if normalized_filter.endswith("/"):
        return needle in anchored
    return anchored.endswith(needle)

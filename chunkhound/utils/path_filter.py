"""Shared path-filter semantics for database and transient search.

Database providers translate a normalized filter into a SQL ``LIKE`` pattern;
transient (in-memory) search uses :func:`path_matches_filter`. Both must agree
so ``vector_source="both"`` filters diff and indexed results identically.
"""

_FORBIDDEN_PATTERNS = ("..", "~", "*", "?", "[", "]", "\0", "\n", "\r")


def normalize_path_filter(path_filter: str | None) -> str | None:
    """Validate and normalize a user-provided path filter.

    Separators become ``/`` and leading slashes are removed. Directory filters
    (no extension in the last segment) end with ``/``; file filters do not.
    A file extension is a dot after the first character, so ``.github`` is a
    directory while ``.eslintrc.js`` is a file.

    Args:
        path_filter: User-provided path filter.

    Returns:
        The normalized filter, or ``None`` when no filter applies.

    Raises:
        ValueError: If the filter contains traversal or wildcard patterns.
    """
    if path_filter is None:
        return None

    normalized = path_filter.strip()
    if not normalized:
        return None

    for pattern in _FORBIDDEN_PATTERNS:
        if pattern in normalized:
            raise ValueError(f"Path filter contains forbidden pattern: {pattern}")

    normalized = normalized.replace("\\", "/").lstrip("/")

    if normalized and not normalized.endswith("/"):
        last = normalized.split("/")[-1]
        if "." not in last[1:]:
            normalized += "/"

    return normalized or None


def path_matches_filter(file_path: str, normalized_filter: str) -> bool:
    """Return whether ``file_path`` matches a filter from normalize_path_filter.

    Mirrors the database rule ``CONCAT('/', path) LIKE pattern``: directory
    filters match a complete path-segment run anywhere (``%/dir/%``); file
    filters must match the path suffix on a segment boundary (``%/file.py``).
    """
    candidate = "/" + file_path.replace("\\", "/").lstrip("/")
    if normalized_filter.endswith("/"):
        return f"/{normalized_filter}" in candidate
    return candidate.endswith(f"/{normalized_filter}")

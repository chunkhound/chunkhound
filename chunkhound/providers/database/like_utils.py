"""Utilities for building safe SQL LIKE patterns."""


def escape_like_pattern(value: str, *, escape_quotes: bool = False) -> str:
    """Escape SQL LIKE metacharacters for literal prefix/substring matching.

    Uses backslash as the escape character (paired with ESCAPE '\\' in SQL).
    Set escape_quotes=True when interpolating the value into a SQL string.
    """
    escaped = (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("[", "\\[")
    )
    if escape_quotes:
        escaped = escaped.replace("'", "''")
    return escaped

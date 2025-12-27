def safe_scope_label(scope_label: str) -> str:
    """Normalize a scope label for use in filenames."""
    safe_scope = scope_label.replace("/", "_")
    return safe_scope or "root"


def compute_scope_prefix(scope_label: str) -> str | None:
    """Convert a scope label into a normalized prefix for path filtering.

    Returns None for root scope ("/"), otherwise ensures a trailing slash.
    """
    if scope_label == "/":
        return None
    normalized = scope_label.rstrip("/") + "/"
    return normalized

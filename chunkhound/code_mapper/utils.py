def safe_scope_label(scope_label: str) -> str:
    """Normalize a scope label for use in filenames."""
    safe_scope = scope_label.replace("/", "_")
    return safe_scope or "root"

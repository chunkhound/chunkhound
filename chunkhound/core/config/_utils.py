"""Shared utilities for config modules."""


def _parse_env_bool(value: str) -> bool | None:
    """Parse a boolean environment variable value."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None

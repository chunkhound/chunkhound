"""Shared utilities for config modules."""

import json
from pathlib import Path
from typing import Any


def read_json_file(path: Path) -> Any:
    """Read JSON while accepting the UTF-8 BOM emitted by Windows tools."""
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def _parse_env_bool(value: str) -> bool | None:
    """Parse a boolean environment variable value."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None

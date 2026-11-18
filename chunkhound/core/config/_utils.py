"""Shared utilities for config modules."""

import json
import locale
from pathlib import Path
from typing import Any


def read_json_file(path: Path) -> Any:
    """Read JSON with BOM tolerance and base-compatible locale decoding.

    Base readers used the platform locale. Decode as UTF-8-with-signature
    first (strips an optional BOM), then retry with the locale encoding only
    when UTF-8 decoding fails. This keeps ANSI/ACP configs on Windows loading
    while still accepting BOM-prefixed UTF-8. Undecodable input still raises.

    A corrupt UTF-8 file can decode as locale text and load with mojibake.
    This is accepted to keep legacy ANSI/ACP configs working.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode(locale.getpreferredencoding(False))
    return json.loads(text)


def _parse_env_bool(value: str) -> bool | None:
    """Parse a boolean environment variable value."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None

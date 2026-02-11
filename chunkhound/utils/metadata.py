"""Metadata normalization helpers.

Chunk metadata is produced by multiple language mappings and persisted in the DB.
Over time, some keys have accumulated multiple possible shapes (e.g., legacy string
vs structured lists). These helpers centralize shape-tolerant parsing so consumers
don't re-implement ad-hoc logic.
"""

import re
from typing import Any

ParameterItem = str | dict[str, str]

_IDENTIFIER_LIKE_RE = re.compile(r"^[A-Za-z_$][\w$]*$")


def _split_comma_separated(value: str) -> list[str]:
    parts = [part.strip() for part in value.split(",")]
    stripped_parts = [part for part in parts if part]
    if not stripped_parts:
        return []

    if all(_IDENTIFIER_LIKE_RE.match(part) for part in stripped_parts):
        return stripped_parts

    return [value]


def normalize_parameters(value: Any) -> list[ParameterItem]:
    """Normalize a chunk's ``metadata['parameters']`` value.

    Supported shapes:
    - ``str``: legacy comma-joined values (split on commas)
    - ``list[str]``: parameter names
    - ``list[dict]``: parameter info dicts (e.g., ``{'name': str, 'type': str}``)

    Any other shape returns an empty list.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if "," in stripped:
            return _split_comma_separated(stripped)
        return [stripped]

    if not isinstance(value, list):
        return []

    normalized: list[ParameterItem] = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
            continue

        if isinstance(item, dict):
            param: dict[str, str] = {}
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                param["name"] = name.strip()

            param_type = item.get("type")
            if isinstance(param_type, str) and param_type.strip():
                param["type"] = param_type.strip()

            if param:
                normalized.append(param)

    return normalized


def extract_parameter_names(value: Any) -> list[str]:
    """Extract parameter names from a parameters metadata value."""
    names: list[str] = []
    for item in normalize_parameters(value):
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def extract_parameter_types(value: Any) -> list[str]:
    """Extract parameter type values from a parameters metadata value.

    Only emits types from dict-shaped parameters (e.g., ``{'name': str, 'type': str}``).
    """
    types: list[str] = []
    for item in normalize_parameters(value):
        if isinstance(item, dict):
            param_type = item.get("type")
            if isinstance(param_type, str) and param_type:
                types.append(param_type)
    return types

"""Metadata normalization helpers.

Chunk metadata is produced by multiple language mappings and persisted in the DB.
Over time, some keys have accumulated multiple possible shapes (e.g., legacy string
vs structured lists). These helpers centralize shape-tolerant parsing so consumers
don't re-implement ad-hoc logic.
"""

from typing import Any

ParameterItem = str | dict[str, str]


def _split_comma_separated(value: str) -> list[str]:
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


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


def iter_parameter_names(value: Any) -> list[str]:
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


def iter_parameter_symbols(value: Any) -> list[str]:
    """Extract searchable parameter symbols from a parameters metadata value.

    For dict-shaped parameters, emits ``name`` first (when present), then ``type``.
    """
    symbols: list[str] = []
    for item in normalize_parameters(value):
        if isinstance(item, str):
            symbols.append(item)
        elif isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name:
                symbols.append(name)

            param_type = item.get("type")
            if isinstance(param_type, str) and param_type:
                symbols.append(param_type)
    return symbols


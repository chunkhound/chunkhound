from __future__ import annotations

import importlib.resources
from pathlib import PurePosixPath

_ASSETS_PACKAGE = "chunkhound.autodoc.assets"


def _validate_relative_path(relative_path: str) -> PurePosixPath:
    candidate = PurePosixPath(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"Asset path must be relative: {relative_path}")
    if ".." in candidate.parts:
        raise ValueError(f"Asset path must not traverse parents: {relative_path}")
    if not candidate.parts:
        raise ValueError("Asset path must not be empty")
    return candidate


def load_text(relative_path: str) -> str:
    asset_path = _validate_relative_path(relative_path)
    with (
        importlib.resources.files(_ASSETS_PACKAGE)
        .joinpath(*asset_path.parts)
        .open("r", encoding="utf-8") as handle
    ):
        return handle.read()


def load_bytes(relative_path: str) -> bytes:
    asset_path = _validate_relative_path(relative_path)
    with (
        importlib.resources.files(_ASSETS_PACKAGE)
        .joinpath(*asset_path.parts)
        .open("rb") as handle
    ):
        return handle.read()


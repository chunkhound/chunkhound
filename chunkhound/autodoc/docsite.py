"""AutoDoc CLI facade.

This module is intentionally small: it exposes a minimal import surface for the
`chunkhound autodoc` CLI command. Tests should import implementation details
directly from `chunkhound.autodoc.*` modules.
"""

from __future__ import annotations

from chunkhound.autodoc.generator import generate_docsite
from chunkhound.autodoc.models import CleanupConfig
from chunkhound.autodoc.site_writer import write_astro_assets_only

__all__: list[str] = [
    "CleanupConfig",
    "generate_docsite",
    "write_astro_assets_only",
]

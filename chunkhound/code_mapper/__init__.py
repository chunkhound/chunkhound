"""Code Mapper support library.

This package contains the reusable building blocks for the `chunkhound map`
CLI command (scope discovery, HyDE prompt construction, and metadata helpers).
"""

from .models import AgentDocMetadata, HydeConfig
from .scope import collect_scope_files

__all__: list[str] = ["AgentDocMetadata", "HydeConfig", "collect_scope_files"]

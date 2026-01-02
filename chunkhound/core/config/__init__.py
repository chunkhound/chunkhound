"""
Configuration management package for ChunkHound.

This package provides a unified configuration system that supports:
- Multiple configuration sources (environment variables, config files, CLI args)
- Type-safe configuration validation using Pydantic
- Consistent embedding provider configuration across MCP and indexing flows
- Secure handling of sensitive configuration data
"""

from .embedding_config import EmbeddingConfig
from .embedding_factory import EmbeddingProviderFactory
from .research_config import ResearchConfig
from .settings_sources import (
    FilteredCliSettingsSource,
    JsonConfigSettingsSource,
    TomlConfigSettingsSource,
    YamlConfigSettingsSource,
    create_config_sources,
    find_config_files,
)

__all__ = [
    "EmbeddingConfig",
    "EmbeddingProviderFactory",
    "ResearchConfig",
    "YamlConfigSettingsSource",
    "TomlConfigSettingsSource",
    "JsonConfigSettingsSource",
    "FilteredCliSettingsSource",
    "create_config_sources",
    "find_config_files",
]

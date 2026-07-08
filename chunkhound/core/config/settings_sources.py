"""
Custom settings sources for ChunkHound configuration management.

This module provides custom Pydantic settings sources that extend the default
configuration loading capabilities to support YAML and TOML configuration files,
as well as filtered CLI argument parsing.
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


class BaseFileConfigSettingsSource(PydanticBaseSettingsSource, ABC):
    """
    Abstract base class for file-based configuration sources.

    This class provides the common framework for loading configuration
    from various file formats (YAML, TOML, JSON) with consistent behavior.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        config_file: str | Path | list[str | Path],
    ):
        """
        Initialize file-based configuration source.

        Args:
            settings_cls: The settings class
            config_file: Path(s) to configuration file(s)
        """
        super().__init__(settings_cls)

        # Handle multiple config files
        if isinstance(config_file, (str, Path)):
            self.config_files = [Path(config_file)]
        else:
            self.config_files = [Path(f) for f in config_file]

        self._data = self._load_files()

    def _load_files(self) -> dict[str, Any]:
        """Load and merge data from all configuration files."""
        merged_data = {}

        for config_file in self.config_files:
            if config_file.exists():
                try:
                    file_data = self.load_file(config_file)
                    if file_data:
                        # Later files override earlier ones
                        merged_data.update(file_data)
                except Exception as e:
                    # Log warning but continue with other files
                    # Skip stderr output in MCP mode to avoid JSON-RPC interference
                    if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                        print(
                            f"Warning: Failed to load config file {config_file}: {e}",
                            file=sys.stderr,
                        )
            else:
                # Only warn if it's the first/only config file
                if len(self.config_files) == 1:
                    # Skip stderr output in MCP mode to avoid JSON-RPC interference
                    if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                        print(
                            f"Warning: Config file {config_file} not found",
                            file=sys.stderr,
                        )

        return merged_data

    @abstractmethod
    def load_file(self, path: Path) -> dict[str, Any]:
        """
        Load configuration data from a specific file.

        Args:
            path: Path to the configuration file

        Returns:
            Dictionary containing configuration data
        """
        pass

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Get field value from configuration data."""
        if field_name in self._data:
            return self._data[field_name], field_name, True
        else:
            # Check for nested field access (e.g., database.host)
            field_parts = field_name.split(".")
            current_data = self._data

            try:
                for part in field_parts:
                    current_data = current_data[part]
                return current_data, field_name, True
            except (KeyError, TypeError):
                pass

        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return the loaded configuration data."""
        return self._data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"config_files={[str(f) for f in self.config_files]})"
        )


class YamlConfigSettingsSource(BaseFileConfigSettingsSource):
    """Configuration source for YAML files."""

    def load_file(self, path: Path) -> dict[str, Any]:
        """Load YAML configuration file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML configuration files. "
                "Install with: pip install pyyaml"
            )

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}


class TomlConfigSettingsSource(BaseFileConfigSettingsSource):
    """Configuration source for TOML files."""

    def load_file(self, path: Path) -> dict[str, Any]:
        """Load TOML configuration file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError(
                    "tomllib (Python 3.11+) or tomli is "
                    "required for TOML configuration files. "
                    "Install with: pip install tomli"
                )

        with open(path, "rb") as f:
            return tomllib.load(f)


class JsonConfigSettingsSource(BaseFileConfigSettingsSource):
    """Configuration source for JSON files."""

    def load_file(self, path: Path) -> dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)


class FilteredCliSettingsSource(PydanticBaseSettingsSource):
    """
    CLI settings source that can include or exclude specific fields.

    This is useful when you need some CLI arguments to have different
    precedence levels (e.g., bootstrap arguments vs regular arguments).
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        cli_args: list[str] | None = None,
        cli_prefix: str = "",
        cli_includes: list[str] | None = None,
        cli_excludes: list[str] | None = None,
        cli_nested_delimiter: str = "__",
    ):
        """
        Initialize filtered CLI settings source.

        Args:
            settings_cls: The settings class
            cli_args: CLI arguments to parse (uses sys.argv if None)
            cli_prefix: Prefix for CLI arguments
            cli_includes: Field names to include (None = include all)
            cli_excludes: Field names to exclude (None = exclude none)
            cli_nested_delimiter: Delimiter for nested fields
        """
        super().__init__(settings_cls)

        self.cli_args = cli_args or sys.argv[1:]
        self.cli_prefix = cli_prefix
        self.cli_includes = set(cli_includes) if cli_includes else None
        self.cli_excludes = set(cli_excludes) if cli_excludes else set()
        self.cli_nested_delimiter = cli_nested_delimiter

        self._parsed_args = self._parse_cli_args()

    def _parse_cli_args(self) -> dict[str, Any]:
        """Parse CLI arguments into a dictionary."""
        parsed = {}
        i = 0

        while i < len(self.cli_args):
            arg = self.cli_args[i]

            # Check if this is a flag argument
            if arg.startswith("--"):
                # Remove -- prefix
                arg_name = arg[2:]

                # Remove prefix if present
                if self.cli_prefix and arg_name.startswith(self.cli_prefix):
                    arg_name = arg_name[len(self.cli_prefix) :]

                # Check if argument should be included/excluded
                if not self._should_include_field(arg_name):
                    i += 1
                    continue

                # Handle --key=value format
                if "=" in arg_name:
                    key, value = arg_name.split("=", 1)
                    parsed[key] = self._parse_value(value)
                    i += 1
                # Handle --key value format
                elif i + 1 < len(self.cli_args) and not self.cli_args[i + 1].startswith(
                    "--"
                ):
                    key = arg_name
                    value = self.cli_args[i + 1]
                    parsed[key] = self._parse_value(value)
                    i += 2
                # Handle boolean flags
                else:
                    parsed[arg_name] = True
                    i += 1
            else:
                i += 1

        # Convert nested keys (key__subkey) to nested dictionaries
        return self._convert_nested_keys(parsed)

    def _should_include_field(self, field_name: str) -> bool:
        """Check if a field should be included based on include/exclude rules."""
        # Remove nested delimiter parts for checking
        base_field = field_name.split(self.cli_nested_delimiter)[0]

        # Check excludes first
        if base_field in self.cli_excludes:
            return False

        # Check includes (if specified, only include listed fields)
        if self.cli_includes is not None:
            return base_field in self.cli_includes

        return True

    def _parse_value(self, value: str) -> Any:
        """Parse a string value to appropriate Python type."""
        # Handle boolean values
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False

        # Handle numeric values
        try:
            # Try integer first
            if "." not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass

        # Handle JSON values
        if value.startswith(("{", "[", '"')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Return as string
        return value

    def _convert_nested_keys(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert nested keys (key__subkey) to nested dictionaries."""
        result = {}

        for key, value in data.items():
            if self.cli_nested_delimiter in key:
                # Split into parts and create nested structure
                parts = key.split(self.cli_nested_delimiter)
                current = result

                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                current[parts[-1]] = value
            else:
                result[key] = value

        return result

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Get field value from parsed CLI arguments."""
        if field_name in self._parsed_args:
            return self._parsed_args[field_name], field_name, True
        else:
            # Check for nested field access (e.g., database.host)
            field_parts = field_name.split(".")
            current_data = self._parsed_args

            try:
                for part in field_parts:
                    current_data = current_data[part]
                return current_data, field_name, True
            except (KeyError, TypeError):
                pass

        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return the parsed CLI arguments."""
        return self._parsed_args

    def __repr__(self) -> str:
        return (
            f"FilteredCliSettingsSource("
            f"prefix={self.cli_prefix}, "
            f"includes={self.cli_includes}, "
            f"excludes={self.cli_excludes})"
        )


def create_config_sources(
    settings_cls: type[BaseSettings],
    config_files: list[str | Path] | None = None,
    cli_args: list[str] | None = None,
    cli_prefix: str = "",
    cli_includes: list[str] | None = None,
    cli_excludes: list[str] | None = None,
) -> list[PydanticBaseSettingsSource]:
    """
    Create a list of configuration sources with reasonable defaults.

    Args:
        settings_cls: Settings class
        config_files: List of configuration files to load
        cli_args: CLI arguments (uses sys.argv if None)
        cli_prefix: Prefix for CLI arguments
        cli_includes: CLI fields to include
        cli_excludes: CLI fields to exclude

    Returns:
        List of configured settings sources
    """
    sources = []

    # Add CLI source if requested
    if cli_args is not None or cli_prefix or cli_includes or cli_excludes:
        sources.append(
            FilteredCliSettingsSource(
                settings_cls,
                cli_args=cli_args,
                cli_prefix=cli_prefix,
                cli_includes=cli_includes,
                cli_excludes=cli_excludes,
            )
        )

    # Add config file sources
    if config_files:
        for config_file in config_files:
            config_path = Path(config_file)

            if config_path.suffix.lower() in (".yaml", ".yml"):
                sources.append(YamlConfigSettingsSource(settings_cls, config_path))
            elif config_path.suffix.lower() == ".toml":
                sources.append(TomlConfigSettingsSource(settings_cls, config_path))
            elif config_path.suffix.lower() == ".json":
                sources.append(JsonConfigSettingsSource(settings_cls, config_path))
            else:
                # Skip stderr output in MCP mode to avoid JSON-RPC interference
                if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                    print(
                        f"Warning: Unknown config file format: {config_path}",
                        file=sys.stderr,
                    )

    return sources


def find_config_files(
    base_dirs: list[str | Path] | None = None,
    config_names: list[str] | None = None,
) -> list[Path]:
    """
    Find configuration files in common locations.

    Args:
        base_dirs: Directories to search (defaults to common config locations)
        config_names: Config file names to look for

    Returns:
        List of found configuration files in priority order
    """
    if base_dirs is None:
        base_dirs = [
            Path.cwd(),
            Path.home() / ".config" / "chunkhound",
            Path.home() / ".chunkhound",
        ]
    else:
        base_dirs = [Path(d) for d in base_dirs]

    if config_names is None:
        config_names = [
            "chunkhound.yaml",
            "chunkhound.yml",
            "chunkhound.toml",
            "chunkhound.json",
            ".chunkhound.yaml",
            ".chunkhound.yml",
            ".chunkhound.toml",
            ".chunkhound.json",
        ]

    found_files = []

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        for config_name in config_names:
            config_path = base_dir / config_name
            if config_path.exists() and config_path.is_file():
                found_files.append(config_path)

    return found_files

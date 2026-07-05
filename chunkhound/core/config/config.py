"""Centralized configuration management for ChunkHound.

This module provides a unified configuration system with clear precedence:
1. CLI arguments (highest priority)
2. Explicit config file (via --config path or CHUNKHOUND_CONFIG_FILE)
3. Local .chunkhound.json in target directory
4. Global defaults (CHUNKHOUND_GLOBAL_CONFIG_FILE or auto-discovered under
   ~/.config/chunkhound/ or ~/.chunkhound/)
5. Environment variables
6. Default values (lowest priority)

Global config support means you can maintain common settings (API keys,
indexing rules, etc.) in one place instead of copying .chunkhound.json
to every project directory. Project-local files override the global layer.
"""

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .database_config import DatabaseConfig
from .embedding_config import EmbeddingConfig
from .indexing_config import IndexingConfig
from .llm_config import LLMConfig
from .mcp_config import MCPConfig
from .research_config import ResearchConfig


class Config(BaseModel):
    """Centralized configuration for ChunkHound.

    Precedence (highest first):
    1. CLI arguments
    2. Explicit config file (--config or CHUNKHOUND_CONFIG_FILE)
    3. Local .chunkhound.json in target directory
    4. Global defaults (CHUNKHOUND_GLOBAL_CONFIG_FILE or
       ~/.config/chunkhound/*.json etc.)
    5. Environment variables (CHUNKHOUND_*)
    6. Defaults
    """

    model_config = ConfigDict(validate_assignment=True)

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig | None = Field(default=None)
    llm: LLMConfig | None = Field(default=None)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    debug: bool = Field(default=False)

    # Private field to store the target directory from CLI args
    target_dir: Path | None = Field(default=None, exclude=True)
    # Private field to track if embeddings were explicitly disabled
    embeddings_disabled: bool = Field(default=False, exclude=True)
    # Private field to store the auto-discovered local .chunkhound.json path
    local_config_file: Path | None = Field(default=None, exclude=True)
    # Private field to store the global defaults config path
    # (from home or CHUNKHOUND_GLOBAL_CONFIG_FILE)
    global_config_file: Path | None = Field(default=None, exclude=True)
    # Private field to store the explicit --config path (absolute) when provided
    config_file: Path | None = Field(default=None, exclude=True)

    @staticmethod
    def _get_global_config_candidates() -> list[Path]:
        """Return preferred locations for global/user defaults config files.

        These provide defaults that apply across projects without needing
        a .chunkhound.json in every directory. Project-local .chunkhound.json
        (and explicit --config) override values from global configs.
        """
        home = Path.home()
        return [
            home / ".config" / "chunkhound" / "chunkhound.json",
            home / ".config" / "chunkhound" / ".chunkhound.json",
            home / ".chunkhound" / "chunkhound.json",
            home / ".chunkhound" / ".chunkhound.json",
            home / "chunkhound.json",
            home / ".chunkhound.json",
        ]

    def __init__(self, args: Any | None = None, **kwargs: Any) -> None:
        """Universal configuration initialization that handles all contexts.

        Automatically applies correct precedence order:
        1. CLI arguments (highest priority)
        2. Explicit config file (via --config path or CHUNKHOUND_CONFIG_FILE)
        3. Local .chunkhound.json in target directory
        4. Global defaults config (CHUNKHOUND_GLOBAL_CONFIG_FILE or discovered in
           ~/.config/chunkhound/ or ~/.chunkhound/)
        5. Environment variables
        6. Default values (lowest priority)

        Global config enables one set of defaults (e.g. embedding provider + key,
        common excludes) without copying .chunkhound.json into every project.
        Local project config and explicit/CLI always override globals.

        Args:
            args: Optional argparse.Namespace from command line parsing
            **kwargs: Direct overrides for testing or special cases
        """
        # Start with defaults
        config_data: dict[str, Any] = {}

        # 1. Smart config file resolution (before env vars)
        config_file = None
        command = getattr(args, "command", None) if args else None
        is_map = command == "map"
        is_autodoc = command == "autodoc"
        is_map_or_autodoc = is_map or is_autodoc

        # Extract target_dir from kwargs first (for testing)
        target_dir = kwargs.pop("target_dir", None)
        if target_dir is not None:
            target_dir = Path(target_dir)

        # Extract config file and target directory from args if provided
        if args:
            # Get config file from --config if present
            if hasattr(args, "config") and args.config:
                config_file = Path(args.config)
                config_data["config_file"] = config_file.resolve()

            # For most commands, args.path represents the project root used for config
            # discovery. For map, args.path is a documentation scope and must
            # not change config discovery. AutoDoc has no positional path.
            if not is_map_or_autodoc:
                # Get target directory from args.path (overrides kwargs)
                if hasattr(args, "path") and args.path:
                    target_dir = Path(args.path)
            elif target_dir is None and config_file is not None:
                # For map/autodoc, treat explicit --config as the workspace root.
                target_dir = config_file.parent

        # If no config file from args, check environment variable
        if not config_file:
            env_config_file = os.getenv("CHUNKHOUND_CONFIG_FILE")
            if env_config_file:
                config_file = Path(env_config_file)

        if is_map_or_autodoc and target_dir is None and config_file is not None:
            # For map/autodoc, treat CHUNKHOUND_CONFIG_FILE as the workspace root.
            target_dir = config_file.parent

        # Only detect project root if target_dir not provided
        if target_dir is None:
            from chunkhound.utils.project_detection import find_project_root

            target_dir = find_project_root(
                None if is_map_or_autodoc else (getattr(args, "path", None) if args else None)
            )

        # 2. Load environment variables
        env_vars = self._load_env_vars()
        self._deep_merge(config_data, env_vars)

        # 2.5 Load global defaults config (env var or auto-discovered).
        # Merged before local so project-local .chunkhound.json overrides globals.
        # Lets users keep common settings (keys, excludes) in one place instead
        # of copying .chunkhound.json into every project.
        global_config_file = None
        env_global = os.getenv("CHUNKHOUND_GLOBAL_CONFIG_FILE")
        if env_global:
            global_config_file = Path(env_global)
            if not global_config_file.exists():
                raise ValueError(
                    f"Global config file not found: {global_config_file}. "
                    "Check the path or remove CHUNKHOUND_GLOBAL_CONFIG_FILE."
                )
            config_data["global_config_file"] = global_config_file.resolve()
        else:
            for candidate in self._get_global_config_candidates():
                if candidate.exists() and candidate.is_file():
                    global_config_file = candidate
                    config_data["global_config_file"] = global_config_file.resolve()
                    break

        if global_config_file:
            try:
                with open(global_config_file) as f:
                    global_config = json.load(f)
                    self._deep_merge(config_data, global_config)
                    self._mark_exclude_user_supplied(config_data)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in global config file {global_config_file}: {e}. "
                    "Please check the file format and try again."
                )

        # 3. Check for local .chunkhound.json (overrides env vars and globals)
        if target_dir and target_dir.exists():
            local_config_path = target_dir / ".chunkhound.json"
            if local_config_path.exists() and local_config_path != config_file:
                config_data["local_config_file"] = local_config_path.resolve()
                try:
                    with open(local_config_path) as f:
                        local_config = json.load(f)
                        self._deep_merge(config_data, local_config)
                        self._mark_exclude_user_supplied(config_data)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in config file {local_config_path}: {e}. "
                        "Please check the file format and try again."
                    )

        # 4. Load explicit config file last so it wins over auto-discovered local config
        if config_file and not config_file.exists():
            raise ValueError(
                f"Config file not found: {config_file}. "
                "Check the path or visit https://chunkhound.ai to generate a config."
            )
        if config_file:
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    self._deep_merge(config_data, file_config)
                    self._mark_exclude_user_supplied(config_data)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in config file {config_file}: {e}. "
                    "Please check the file format and try again."
                )

        # 5. Apply CLI arguments (highest precedence)
        if args:
            cli_overrides = self._extract_cli_overrides(args)
            self._mark_exclude_user_supplied(cli_overrides)
            self._deep_merge(config_data, cli_overrides)

        # 6. Apply any direct kwargs (for testing)
        if kwargs:
            self._mark_exclude_user_supplied(kwargs)
            self._deep_merge(config_data, kwargs)

        # Special handling for EmbeddingConfig
        if "embedding" in config_data and isinstance(config_data["embedding"], dict):
            # Create EmbeddingConfig instance with the data
            config_data["embedding"] = EmbeddingConfig(**config_data["embedding"])

        # Special handling for LLMConfig
        if "llm" in config_data and isinstance(config_data["llm"], dict):
            # Create LLMConfig instance with the data
            config_data["llm"] = LLMConfig(**config_data["llm"])

        # Special handling for ResearchConfig
        if "research" in config_data and isinstance(config_data["research"], dict):
            # Create ResearchConfig instance with the data
            config_data["research"] = ResearchConfig(**config_data["research"])

        # Add target_dir to config_data for initialization
        config_data["target_dir"] = target_dir

        # Initialize the model
        super().__init__(**config_data)

    @staticmethod
    def _mark_exclude_user_supplied(data: dict[str, Any]) -> None:
        """Mark exclude list as user-supplied when present in indexing config."""
        idx = data.get("indexing")
        if isinstance(idx, dict) and isinstance(idx.get("exclude"), list):
            idx["exclude_user_supplied"] = True

    def _load_env_vars(self) -> dict[str, Any]:
        """Load configuration from environment variables.

        Supports both legacy and new environment variable names.
        Uses CHUNKHOUND_ prefix with __ delimiter for nested values.
        """
        config: dict[str, Any] = {}

        # Debug mode
        if os.getenv("CHUNKHOUND_DEBUG"):
            config["debug"] = os.getenv("CHUNKHOUND_DEBUG", "").lower() in (
                "true",
                "1",
                "yes",
            )

        # Delegate to each config class
        if db_config := DatabaseConfig.load_from_env():
            config["database"] = db_config
        if embedding_config := EmbeddingConfig.load_from_env():
            config["embedding"] = embedding_config
        if llm_config := LLMConfig.load_from_env():
            config["llm"] = llm_config
        if mcp_config := MCPConfig.load_from_env():
            config["mcp"] = mcp_config
        if indexing_config := IndexingConfig.load_from_env():
            config["indexing"] = indexing_config
        if research_config := ResearchConfig.load_from_env():
            config["research"] = research_config

        return config

    def _extract_cli_overrides(self, args: Any) -> dict[str, Any]:
        """Extract configuration overrides from CLI arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            Dictionary of configuration overrides
        """
        overrides: dict[str, Any] = {}

        # Common CLI args
        if hasattr(args, "debug") and args.debug:
            overrides["debug"] = args.debug
        elif hasattr(args, "verbose") and args.verbose:
            overrides["debug"] = args.verbose

        # Delegate to each config class
        if db_overrides := DatabaseConfig.extract_cli_overrides(args):
            overrides["database"] = db_overrides
        if embedding_overrides := EmbeddingConfig.extract_cli_overrides(args):
            # Handle special case for --no-embeddings
            if embedding_overrides.get("disabled"):
                overrides["embedding"] = None
                overrides["embeddings_disabled"] = True
            else:
                overrides["embedding"] = embedding_overrides
        if llm_overrides := LLMConfig.extract_cli_overrides(args):
            overrides["llm"] = llm_overrides
        if mcp_overrides := MCPConfig.extract_cli_overrides(args):
            overrides["mcp"] = mcp_overrides
        if indexing_overrides := IndexingConfig.extract_cli_overrides(args):
            overrides["indexing"] = indexing_overrides
        if research_overrides := ResearchConfig.extract_cli_overrides(args):
            overrides["research"] = research_overrides

        return overrides

    def _deep_merge(self, base: dict[str, Any], update: dict[str, Any]) -> None:
        """Deep merge update dictionary into base dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        """Validate the configuration after initialization."""
        # Ensure target_dir is always set and resolved (never None)
        if self.target_dir is None:
            from chunkhound.utils.project_detection import find_project_root

            detected_root = find_project_root(None)
            # Fallback to current working directory if no project root found
            resolved_target = (
                detected_root.resolve() if detected_root else Path.cwd().resolve()
            )
            # Use object.__setattr__ to avoid Pydantic validation recursion
            object.__setattr__(self, "target_dir", resolved_target)
        else:
            # Ensure target_dir is resolved to canonical path (handles symlinks)
            # Use object.__setattr__ to avoid Pydantic validation recursion
            object.__setattr__(self, "target_dir", self.target_dir.resolve())

        # Ensure the tracked config file paths are also resolved to canonical form.
        # This handles Windows short names (8.3 like RUNNER~1), symlinks, and
        # any differences in how temp dirs / constructed Paths are represented
        # before vs after .resolve().
        for field_name in ("local_config_file", "global_config_file", "config_file"):
            val = getattr(self, field_name, None)
            if val is not None:
                try:
                    object.__setattr__(self, field_name, val.resolve())
                except Exception:
                    # Best effort; leave as-is if resolve is not possible.
                    pass

        # Ensure database path is set
        if not self.database.path:
            # Try to detect project root from target_dir or auto-detect
            from chunkhound.utils.project_detection import find_project_root

            # Use the target_dir if it was provided during initialization
            start_path = self.target_dir
            project_root = find_project_root(start_path)

            # Set default database path in project root
            self.database.path = project_root / ".chunkhound" / "db"

        # Ensure database path is resolved to canonical form (handles symlinks)
        if self.database.path:
            self.database.path = self.database.path.resolve()

        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_environment(cls) -> "Config":
        """Construct Config using environment and defaults (no CLI args).

        Convenience for legacy call sites expecting a simple way to obtain a
        fully-initialized Config without command-specific CLI parsing.
        """
        return cls(args=None)

    def validate_for_command(self, command: str, args: Any | None = None) -> list[str]:
        """
        Validate configuration for a specific command.

        Args:
            command: Command name ('index', 'mcp', etc.)

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[str] = []

        # Check for missing configuration
        missing_config = self.get_missing_config()
        if missing_config:
            errors.extend(
                f"Missing required configuration: {item}" for item in missing_config
            )

        # websearch only spawns _quickresearch as a subprocess, but we validate
        # LLM/embedding config in the parent so misconfiguration fails fast
        # before fetching DDG results and writing tempfiles.
        requires_llm = (
            command in ("research", "websearch", "_quickresearch")
            or (command == "map" and not getattr(args, "overview_only", False))
            or (command == "autodoc" and not getattr(args, "assets_only", False))
        )
        if requires_llm:
            if self.llm is None:
                errors.append("No LLM provider configured")
            else:
                llm_roles = ["utility", "synthesis"]
                if command == "map" and (
                    self.llm.map_hyde_provider
                    or self.llm.map_hyde_model
                    or self.llm.map_hyde_reasoning_effort
                ):
                    llm_roles.append("map_hyde")
                if command == "autodoc":
                    llm_roles.append("autodoc_cleanup")

                llm_missing = self.llm.get_missing_config_for_roles(tuple(llm_roles))
                if llm_missing:
                    errors.extend(
                        f"Missing required configuration: llm.{item}"
                        for item in llm_missing
                    )

        # Validate embedding provider requirements for commands that index code
        if command in ("index", "websearch", "_quickresearch"):
            # Skip embedding validation if embeddings were explicitly disabled
            if not self.embeddings_disabled:
                if self.embedding is None:
                    errors.append("No embedding provider configured")
                elif self.embedding and not self.embedding.is_provider_configured():
                    errors.append("Embedding provider not properly configured")

        # For MCP command, embedding is optional
        elif command == "mcp":
            if self.embedding and not self.embedding.is_provider_configured():
                errors.append("Embedding provider not properly configured")

        # For search command, embedding is optional but must be valid if present
        elif command == "search":
            if self.embedding and not self.embedding.is_provider_configured():
                errors.append("Embedding provider not properly configured")

        if self.database.read_only:
            # _quickresearch always uses a :memory: DB (see quickresearch.py),
            # so a project-level read_only setting inherited via --config is
            # structurally inapplicable — accept it here and let the connection
            # manager drop read_only for :memory: paths.
            if command not in ("mcp", "_quickresearch"):
                errors.append(
                    "database.read_only=True is only valid for the 'mcp' subcommand"
                )
            elif command == "mcp" and self.database.provider != "duckdb":
                errors.append(
                    "database.read_only=True is only supported with the DuckDB "
                    f"provider (got '{self.database.provider}'). "
                    "Use the DuckDB provider, or omit --read-only."
                )

        return errors

    def get_missing_config(self) -> list[str]:
        """
        Get list of missing required configuration parameters.

        Returns:
            List of missing configuration parameter names
        """
        missing = []

        # Check embedding configuration if it exists
        if self.embedding:
            if hasattr(self.embedding, "get_missing_config"):
                embedding_missing = self.embedding.get_missing_config()
                for item in embedding_missing:
                    missing.append(f"embedding.{item}")
        # Note: If embedding is None, we don't assume a default provider
        # Commands like index and search can work without embeddings

        return missing

    def is_fully_configured(self) -> bool:
        """
        Check if all required configuration is present.

        Returns:
            True if fully configured, False otherwise
        """
        return self.embedding is not None and self.embedding.is_provider_configured()

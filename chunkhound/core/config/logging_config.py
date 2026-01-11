"""Logging configuration models for ChunkHound."""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class FileLoggingConfig(BaseModel):
    """Configuration for file-based logging."""

    enabled: bool = Field(default=True, description="Enable file logging")
    path: str = Field(default="chunkhound.log", description="Path to log file")
    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    rotation: str = Field(default="10 MB", description="Log rotation size (e.g., '10 MB', '1 week')")
    retention: str = Field(default="1 week", description="Log retention period (e.g., '1 week', '30 days')")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        description="Log message format"
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level '{v}'. Must be one of: {', '.join(valid_levels)}")
        return v.upper()

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate log file path."""
        if not v.strip():
            raise ValueError("Log file path cannot be empty")
        # Basic path validation
        try:
            Path(v)
        except Exception as e:
            raise ValueError(f"Invalid log file path '{v}': {e}")
        return v


class PerformanceLoggingConfig(BaseModel):
    """Configuration for performance timing logs."""

    enabled: bool = Field(default=True, description="Enable performance logging")
    path: str = Field(default="chunkhound-performance.log", description="Path to performance log file")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate performance log file path."""
        if not v.strip():
            raise ValueError("Performance log file path cannot be empty")
        try:
            Path(v)
        except Exception as e:
            raise ValueError(f"Invalid performance log file path '{v}': {e}")
        return v


class LoggingConfig(BaseModel):
    """Top-level logging configuration."""

    file: FileLoggingConfig = Field(default_factory=FileLoggingConfig)
    performance: PerformanceLoggingConfig = Field(default_factory=PerformanceLoggingConfig)
    console_level: str = Field(default="WARNING", description="Console logging level (DEBUG, INFO, WARNING, ERROR)")
    progress_display_log_level: str = Field(default="WARNING", description="Log level for progress display messages (DEBUG, INFO, WARNING, ERROR)")
    max_log_messages: int = Field(default=100, description="Maximum number of log messages to buffer for progress display")
    log_panel_ratio: float = Field(default=0.3, description="Ratio of terminal height allocated to log panel (0.0-1.0)")

    @field_validator("console_level")
    @classmethod
    def validate_console_level(cls, v: str) -> str:
        """Validate console logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid console log level '{v}'. Must be one of: {', '.join(valid_levels)}")
        return v.upper()

    @field_validator("progress_display_log_level")
    @classmethod
    def validate_progress_display_log_level(cls, v: str) -> str:
        """Validate progress display logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid progress display log level '{v}'. Must be one of: {', '.join(valid_levels)}")
        return v.upper()

    @field_validator("max_log_messages")
    @classmethod
    def validate_max_log_messages(cls, v: int) -> int:
        """Validate maximum log messages."""
        if v <= 0:
            raise ValueError(f"max_log_messages must be positive, got {v}")
        return v

    @field_validator("log_panel_ratio")
    @classmethod
    def validate_log_panel_ratio(cls, v: float) -> float:
        """Validate log panel ratio."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"log_panel_ratio must be between 0.0 and 1.0, got {v}")
        return v

    def is_enabled(self) -> bool:
        """Check if any logging is enabled."""
        return self.file.enabled

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any] | None:
        """Extract logging configuration overrides from CLI arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            Dictionary of logging configuration overrides, or None if no overrides
        """
        overrides: dict[str, Any] = {}

        # File logging overrides
        file_overrides: dict[str, Any] = {}
        if hasattr(args, "log_file") and args.log_file:
            file_overrides["enabled"] = True
            file_overrides["path"] = args.log_file
        if hasattr(args, "log_level") and args.log_level:
            file_overrides["level"] = args.log_level

        if file_overrides:
            overrides["file"] = file_overrides

        # Performance logging overrides
        if hasattr(args, "performance_log") and args.performance_log:
            overrides["performance"] = {
                "enabled": True,
                "path": args.performance_log
            }

        # Progress display logging overrides
        if hasattr(args, "progress_display_log_level") and args.progress_display_log_level:
            overrides["progress_display_log_level"] = args.progress_display_log_level
        if hasattr(args, "max_log_messages") and args.max_log_messages is not None:
            overrides["max_log_messages"] = args.max_log_messages
        if hasattr(args, "log_panel_ratio") and args.log_panel_ratio is not None:
            overrides["log_panel_ratio"] = args.log_panel_ratio

        return overrides if overrides else None
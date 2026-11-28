"""Websearch configuration for the ChunkHound multi-engine web search.

Governs which search engines back ``websearch``, how they are selected
(priority vs random among healthy), and how long a refused engine cools down
before it is retried within a process. Configuration sources match the rest
of ChunkHound: CLI arguments > config files > environment variables >
defaults.
"""

import argparse
import os
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Engines the websearch pipeline can drive. Adding an engine means
# registering its adapter in chunkhound.utils.websearch_backends and listing
# its name here; unknown names are rejected loudly at config load.
BACKEND_NAMES: tuple[str, ...] = ("duckduckgo", "bing")

_DEFAULT_BACKENDS = ["duckduckgo", "bing"]
_DEFAULT_SELECTION: Literal["priority", "random_healthy"] = "priority"
_DEFAULT_COOLDOWN_SECONDS = 900


class WebSearchConfig(BaseSettings):
    """
    Websearch configuration.

    Configuration Sources (in order of precedence):
    1. CLI arguments
    2. Config files
    3. Environment variables (CHUNKHOUND_WEBSEARCH_*)
    4. Default values

    Environment Variables:
        CHUNKHOUND_WEBSEARCH_BACKENDS=duckduckgo,bing
        CHUNKHOUND_WEBSEARCH_SELECTION=priority
        CHUNKHOUND_WEBSEARCH_COOLDOWN_SECONDS=900
    """

    model_config = SettingsConfigDict(
        env_prefix="CHUNKHOUND_WEBSEARCH_",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_default=True,
        extra="ignore",  # Ignore unknown fields for forward compatibility
    )

    # NoDecode: the env source would otherwise demand JSON for a list field;
    # we split the raw comma string in the mode="before" validator instead.
    backends: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(_DEFAULT_BACKENDS),
        description=(
            "Ordered list of search engines; the first healthy engine serves "
            "and the rest are fallbacks. Recognized engines: "
            f"{', '.join(BACKEND_NAMES)}."
        ),
    )

    selection: Literal["priority", "random_healthy"] = Field(
        default=_DEFAULT_SELECTION,
        description=(
            "'priority' uses the listed order and skips cooling engines; "
            "'random_healthy' shuffles only the healthy engines."
        ),
    )

    cooldown_seconds: int = Field(
        default=_DEFAULT_COOLDOWN_SECONDS,
        ge=0,
        description=(
            "Seconds a refused engine stays out of rotation before it is "
            "retried (per process)."
        ),
    )

    @field_validator("backends", mode="before")
    @classmethod
    def _coerce_backends(cls, value: object) -> object:
        # The env source hands us the raw comma string
        # (CHUNKHOUND_WEBSEARCH_BACKENDS=duckduckgo,bing); split it before
        # list validation.
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("backends")
    @classmethod
    def _validate_and_dedupe_backends(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw in value:
            name = raw.strip().lower()
            if not name:
                continue
            if name not in BACKEND_NAMES:
                raise ValueError(
                    f"Unknown websearch backend {raw!r}; recognized engines: "
                    f"{', '.join(BACKEND_NAMES)}"
                )
            if name not in normalized:
                normalized.append(name)
        if not normalized:
            raise ValueError("websearch.backends must name at least one engine")
        return normalized

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add websearch-related CLI arguments."""
        parser.add_argument(
            "--websearch-backends",
            type=str,
            help=(
                "Comma-separated ordered search engines (default: "
                "duckduckgo,bing)"
            ),
        )
        parser.add_argument(
            "--websearch-selection",
            choices=["priority", "random_healthy"],
            help="Engine selection: priority (listed order) or random_healthy",
        )
        parser.add_argument(
            "--websearch-cooldown-seconds",
            type=int,
            help="Seconds a refused engine cools down before retry (default: 900)",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load websearch config from environment variables."""
        config: dict[str, Any] = {}

        if raw_backends := os.getenv("CHUNKHOUND_WEBSEARCH_BACKENDS"):
            config["backends"] = [
                part.strip()
                for part in raw_backends.split(",")
                if part.strip()
            ]

        if selection := os.getenv("CHUNKHOUND_WEBSEARCH_SELECTION"):
            config["selection"] = selection.strip().lower()

        if cooldown := os.getenv("CHUNKHOUND_WEBSEARCH_COOLDOWN_SECONDS"):
            # Keep the raw string; final WebSearchConfig validation coerces it.
            # Eager int() here would let an invalid env value raise before
            # CLI/file overrides (higher precedence) can replace it. Deferred
            # coercion is the intentional pattern; eager sibling coercion is
            # out of scope.
            config["cooldown_seconds"] = cooldown

        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract websearch config from CLI arguments."""
        overrides: dict[str, Any] = {}

        if (
            hasattr(args, "websearch_backends")
            and args.websearch_backends is not None
        ):
            overrides["backends"] = [
                part.strip()
                for part in args.websearch_backends.split(",")
                if part.strip()
            ]

        if (
            hasattr(args, "websearch_selection")
            and args.websearch_selection is not None
        ):
            overrides["selection"] = args.websearch_selection

        if (
            hasattr(args, "websearch_cooldown_seconds")
            and args.websearch_cooldown_seconds is not None
        ):
            overrides["cooldown_seconds"] = args.websearch_cooldown_seconds

        return overrides

    def __repr__(self) -> str:
        """String representation with key settings."""
        return (
            f"WebSearchConfig("
            f"backends={self.backends}, "
            f"selection={self.selection}, "
            f"cooldown_seconds={self.cooldown_seconds})"
        )

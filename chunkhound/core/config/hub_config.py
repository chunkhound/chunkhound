"""Opaque manifest URL and configurable client-side archive resource limits."""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArchiveLimits(BaseModel):
    """Finite archive budgets; environment values are decoded before validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_members: int = Field(default=100_000, strict=True, gt=0)
    max_path_depth: int = Field(default=32, strict=True, gt=0)
    max_path_bytes: int = Field(default=4096, strict=True, gt=0)
    max_total_path_bytes: int = Field(default=32 * 1024 * 1024, strict=True, gt=0)
    max_extracted_bytes: int = Field(default=32 * 1024**3, strict=True, gt=0)
    max_decompressed_bytes: int = Field(default=40 * 1024**3, strict=True, gt=0)


class HubConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str | None = Field(default=None, repr=False)
    archive_limits: ArchiveLimits = Field(default_factory=ArchiveLimits)

    @field_validator("url", mode="before")
    @classmethod
    def empty_is_missing(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "CHUNKHOUND_HUB__URL" in os.environ:
            result["url"] = os.environ["CHUNKHOUND_HUB__URL"] or None
        limits: dict[str, int | str] = {}
        for name in ArchiveLimits.model_fields:
            key = f"CHUNKHOUND_HUB__ARCHIVE_LIMITS__{name.upper()}"
            if key not in os.environ:
                continue
            raw = os.environ[key]
            value: int | str = raw
            if raw.isascii() and raw.isdecimal():
                try:
                    value = int(raw)
                except ValueError:
                    # Preserve invalid input until normal layered validation.
                    value = raw
            limits[name] = value
        if limits:
            result["archive_limits"] = limits
        return result

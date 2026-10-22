"""Opaque committed manifest URL configuration for the indexing hub."""

import os

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HubConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str | None = Field(default=None, repr=False)

    @field_validator("url", mode="before")
    @classmethod
    def empty_is_missing(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @classmethod
    def load_from_env(cls) -> dict[str, str | None]:
        if "CHUNKHOUND_HUB__URL" not in os.environ:
            return {}
        return {"url": os.environ["CHUNKHOUND_HUB__URL"] or None}

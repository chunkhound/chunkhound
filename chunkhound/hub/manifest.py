"""Strict manifest selection using an offline, exact embedding identity."""

from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.embedding_dimensions import configured_embedding_dims
from chunkhound.core.index_format import INDEX_FORMAT_VERSION
from chunkhound.hub.errors import HubUnavailable, safe_label


@dataclass(frozen=True)
class MatchKey:
    db_provider: str
    provider: str
    model: str
    dims: int
    index_format_version: int


class WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class Artifact(WireModel):
    url: str = Field(repr=False)
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class Version(WireModel):
    rel: Literal["latest", "-1", "-2"]
    artifact: Artifact
    build_id: str = ""
    revision: str = ""
    created_at: str = ""


class Variant(WireModel):
    embedding_id: str
    db_provider: str
    provider: str
    model: str
    dims: int = Field(gt=0)
    index_format_version: int = Field(gt=0)
    versions: list[Version] = Field(max_length=3)

    def key(self) -> MatchKey:
        return MatchKey(
            self.db_provider,
            self.provider,
            self.model,
            self.dims,
            self.index_format_version,
        )


class Manifest(WireModel):
    manifest_version: Literal[2]
    project_id: str = ""
    index_root: Any = None
    updated_at: str = ""
    embeddings: list[Variant]


@dataclass(frozen=True)
class Selected:
    manifest: Manifest
    variant: Variant
    version: Version


def select_manifest(raw: bytes, key: MatchKey, rel: str) -> Selected:
    try:
        manifest = Manifest.model_validate_json(raw)
    except ValidationError:
        raise HubUnavailable("Hub manifest is invalid or unsupported") from None
    variants = [entry for entry in manifest.embeddings if entry.key() == key]
    if len(variants) != 1:
        logger.warning(
            "No unique published base matches local={} offers={}",
            safe_label(key),
            safe_label([v.key() for v in manifest.embeddings]),
        )
        raise HubUnavailable("No unique published base matches local configuration")
    versions = [entry for entry in variants[0].versions if entry.rel == rel]
    if len(versions) != 1:
        raise HubUnavailable("Requested relative hub version is missing or ambiguous")
    return Selected(manifest, variants[0], versions[0])


def configured_key(config: Config) -> MatchKey:
    if config.embeddings_disabled or config.embedding is None:
        raise HubUnavailable("Hub pull requires embedding configuration")
    try:
        provider = EmbeddingProviderFactory.create_provider(config.embedding)
        if provider.name != config.embedding.provider:
            raise HubUnavailable(
                "Hub storage-provider interoperability is not enabled for this endpoint"
            )
        dims = configured_embedding_dims(provider)
        if type(dims) is not int or dims <= 0:
            raise HubUnavailable("Hub pull requires known embedding dimensions")
        return MatchKey(
            config.database.provider,
            config.embedding.provider,
            provider.model,
            dims,
            INDEX_FORMAT_VERSION,
        )
    except HubUnavailable:
        raise
    except Exception:
        raise HubUnavailable(
            "Cannot resolve configured embedding identity offline"
        ) from None

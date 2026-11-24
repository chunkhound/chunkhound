"""Strict manifest selection using an offline, exact embedding identity."""

from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.embedding_dimensions import configured_embedding_dims
from chunkhound.core.index_format import INDEX_FORMAT_VERSION
from chunkhound.hub.errors import HubUnavailable, safe_label
from chunkhound.utils.http_redirects import http_destination_ok


@dataclass(frozen=True)
class MatchKey:
    """What a published base must match: backend, provider, model, dims, format."""

    db_provider: str
    provider: str
    model: str
    dims: int
    index_format_version: int


class WireModel(BaseModel):
    """Base for manifest records: strict types, unknown fields ignored."""

    model_config = ConfigDict(extra="ignore", strict=True)


class Artifact(WireModel):
    """A downloadable archive with its size and SHA-256."""

    url: str = Field(repr=False)
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Reject artifact URLs that are not allowed destinations."""
        if not http_destination_ok(value):
            raise ValueError("artifact URL is unsupported")
        return value


class Version(WireModel):
    """One published release of a base (latest, -1, or -2)."""

    rel: Literal["latest", "-1", "-2"]
    artifact: Artifact
    build_id: str = ""
    revision: str = ""
    created_at: str = ""


class Variant(WireModel):
    """A base published for one embedding provider, model, and format."""

    embedding_id: str
    db_provider: str
    provider: str
    model: str
    dims: int = Field(gt=0)
    index_format_version: int = Field(gt=0)
    versions: list[Version] = Field(max_length=3)

    def key(self) -> MatchKey:
        """The match key this variant publishes."""
        return MatchKey(
            self.db_provider,
            self.provider,
            self.model,
            self.dims,
            self.index_format_version,
        )


class Manifest(WireModel):
    """The published list of base variants."""

    manifest_version: Literal[2]
    project_id: str = ""
    index_root: Any = None
    updated_at: str = ""
    embeddings: list[Variant]


@dataclass(frozen=True)
class Selected:
    """The manifest entry chosen for this checkout."""

    manifest: Manifest
    variant: Variant
    version: Version


def parse_manifest(raw: bytes) -> Manifest:
    """Parse manifest JSON, refusing invalid or unsupported documents."""
    try:
        return Manifest.model_validate_json(raw)
    except ValidationError:
        raise HubUnavailable("Hub manifest is invalid or unsupported") from None


def select_manifest(manifest: Manifest, key: MatchKey, rel: str) -> Selected:
    """Choose the unique variant and version matching a key and release."""
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
    """Build the match key from local configuration without touching local files."""
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

"""Detect when the configured embedding setup differs from the indexed one.

An index built with one embedding configuration cannot be searched with
another. Search embeds the query with the configured provider, reads the
``embeddings_<dims>`` table matching the query vector's length, and filters it
by ``(provider, model)``. A different model, or the same model at different
dimensions, therefore matches nothing. Re-indexing does not repair it either:
chunks that already hold vectors for the provider and model are skipped at any
dimension.

The index therefore owns its model and dimensions. A configuration change is a
proposal that has to be accepted, not an instruction that silently rewrites or
strands the database. Callers find drift with :func:`detect_model_drift`; the
registry decides what to pin, and only the CLI ever asks a human.
"""

import re
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from chunkhound.providers.database.duckdb.schema_constants import (
    EMBEDDING_TABLE_SIMILAR_PATTERN,
)

# Table names cannot be bound as query parameters, so the only defense against
# interpolating something unexpected is to refuse anything that is not a
# dimension-suffixed embedding table. Checked here rather than trusting the
# information_schema filter, so the guarantee lives next to the interpolation.
_EMBEDDING_TABLE_NAME = re.compile(r"\Aembeddings_[0-9]+\Z")


class _QueryableDatabase(Protocol):
    """The slice of a database provider this module needs."""

    def execute_query(
        self, query: str, params: list[Any] | None = ...
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class IndexedEmbeddingModel:
    """The provider, model and dimensions the existing embeddings were built with."""

    provider: str
    model: str
    dims: int
    embedding_count: int

    def __str__(self) -> str:
        return f"{self.provider}/{self.model} @ {self.dims} dims"


@dataclass(frozen=True)
class ModelDrift:
    """Embedding configuration that disagrees with what the index was built with.

    ``configured_dims`` is None when the configured provider cannot know its
    dimensions before its first call (an unknown model on a custom endpoint).
    Dimensions are then left out of the comparison rather than guessed.
    """

    indexed: IndexedEmbeddingModel
    configured_provider: str
    configured_model: str
    configured_dims: int | None

    @property
    def model_changed(self) -> bool:
        """True when the provider or model differs, not only the dimensions.

        Only a model change can be re-embedded in place. Chunks that already
        hold vectors for the indexed provider and model are skipped at any
        dimension, so accepting a dimensions-only change would strand search.
        """
        return (self.indexed.provider, self.indexed.model) != (
            self.configured_provider,
            self.configured_model,
        )

    @property
    def dims_changed(self) -> bool:
        return (
            self.configured_dims is not None
            and self.configured_dims != self.indexed.dims
        )

    @property
    def configured(self) -> str:
        dims = (
            f"{self.configured_dims} dims"
            if self.configured_dims is not None
            else "dims unknown until first call"
        )
        return f"{self.configured_provider}/{self.configured_model} @ {dims}"


def detect_indexed_model(
    db: _QueryableDatabase,
) -> IndexedEmbeddingModel | None:
    """Return the dominant provider/model/dims in the index, or None if empty.

    "Dominant" means the combination with the most stored vectors. A database
    can hold several at once, which is exactly what a half-finished switch
    leaves behind; picking the largest keeps the answer stable across an
    interrupted re-embed rather than flip-flopping on row order.

    A query failure returns None, meaning "cannot tell". Callers treat that as
    "no drift detected" and proceed, because refusing to index over an
    unreadable count would be a worse failure than the one being prevented.
    """
    try:
        tables = db.execute_query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' "
            f"AND table_name SIMILAR TO '{EMBEDDING_TABLE_SIMILAR_PATTERN}'"
        )
    except Exception as e:
        logger.debug(f"Could not enumerate embedding tables for drift check: {e}")
        return None

    rows: list[dict[str, Any]] = []
    for table in tables:
        table_name = table["table_name"]
        if not _EMBEDDING_TABLE_NAME.match(table_name):
            logger.debug(f"Skipping unexpected embedding table name: {table_name!r}")
            continue
        try:
            rows.extend(
                db.execute_query(
                    "SELECT provider, model, dims, COUNT(*) AS count "
                    f"FROM {table_name} GROUP BY provider, model, dims"
                )
            )
        except Exception as e:
            logger.debug(f"Could not read {table_name} for drift check: {e}")
            return None

    if not rows:
        return None

    dominant = max(rows, key=lambda row: row["count"])
    return IndexedEmbeddingModel(
        provider=dominant["provider"],
        model=dominant["model"],
        dims=dominant["dims"],
        embedding_count=dominant["count"],
    )


def detect_model_drift(
    db: _QueryableDatabase,
    configured_provider: str,
    configured_model: str,
    configured_dims: int | None,
) -> ModelDrift | None:
    """Return drift between the index and the configuration, if any.

    None covers both "the index agrees" and "there is nothing indexed yet",
    which callers treat identically: proceed with the configuration.
    """
    indexed = detect_indexed_model(db)
    if indexed is None:
        return None
    drift = ModelDrift(
        indexed=indexed,
        configured_provider=configured_provider,
        configured_model=configured_model,
        configured_dims=configured_dims,
    )
    if not drift.model_changed and not drift.dims_changed:
        return None
    return drift


def format_drift_warning(drift: ModelDrift) -> str:
    """Human-readable explanation of what switching costs, or why it cannot."""
    count = drift.indexed.embedding_count
    header = (
        "Embedding configuration changed since this index was built.\n"
        f"  indexed with: {drift.indexed}  ({count:,} embeddings)\n"
        f"  configured:   {drift.configured}\n"
    )
    if drift.model_changed:
        return header + (
            f"Re-embedding rewrites all {count:,} vectors, and the "
            f"{drift.indexed.model} vectors stay in the database until removed. "
            "Until then, searches run against the indexed model."
        )
    return header + (
        "Changing dimensions on the same model cannot be re-embedded in place, "
        f"so the index keeps {drift.indexed.dims} dims. To change them, delete "
        "the database directory and re-index."
    )

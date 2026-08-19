"""Detect when the configured embedding model differs from the indexed one.

An index built with one embedding model cannot be searched with another.
``SearchService`` filters stored vectors by ``(provider, model)``, so a mismatch
returns nothing at all until every chunk has been re-embedded under the new
model. That re-embed costs tokens and wall-clock time, and the superseded
vectors stay in the database until something removes them.

The index therefore owns the model. A configuration change is a proposal that
has to be accepted, not an instruction that silently rewrites the database.
Callers detect drift with :func:`detect_indexed_model`, then decide with
:func:`resolve_model_drift`; only the CLI ever asks a human.
"""

import re
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from chunkhound.providers.database.duckdb.schema_constants import (
    EMBEDDING_TABLE_SIMILAR_PATTERN,
)

# Table names cannot be bound as query parameters, so the only defence against
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
    """The provider/model pair the existing embeddings were built with."""

    provider: str
    model: str
    dims: int
    embedding_count: int

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class ModelDrift:
    """A configured model that disagrees with the indexed one."""

    indexed: IndexedEmbeddingModel
    configured_provider: str
    configured_model: str

    @property
    def configured(self) -> str:
        return f"{self.configured_provider}/{self.configured_model}"


def detect_indexed_model(
    db: _QueryableDatabase,
) -> IndexedEmbeddingModel | None:
    """Return the dominant provider/model in the index, or None if it is empty.

    "Dominant" means the pair with the most stored vectors. A database can hold
    several pairs at once, which is exactly what a half-finished model switch
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
) -> ModelDrift | None:
    """Return drift between the index and the configured model, if any.

    None covers both "the index agrees" and "there is nothing indexed yet",
    which callers treat identically: proceed with the configured model.
    """
    indexed = detect_indexed_model(db)
    if indexed is None:
        return None
    if (indexed.provider, indexed.model) == (configured_provider, configured_model):
        return None
    return ModelDrift(
        indexed=indexed,
        configured_provider=configured_provider,
        configured_model=configured_model,
    )


def format_drift_warning(drift: ModelDrift) -> str:
    """Human-readable explanation of what a re-embed would cost."""
    return (
        "Embedding model changed since this index was built.\n"
        f"  indexed with: {drift.indexed}"
        f"  ({drift.indexed.embedding_count:,} embeddings)\n"
        f"  configured:   {drift.configured}\n"
        f"Re-embedding rewrites all {drift.indexed.embedding_count:,} vectors, "
        f"and the {drift.indexed.model} vectors stay in the database until "
        "removed. Until then, searches run against the indexed model."
    )

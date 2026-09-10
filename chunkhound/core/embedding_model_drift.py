"""Detect when the configured embedding setup differs from the indexed one.

An index built with one embedding configuration cannot be searched with
another. Search embeds the query with the configured provider, reads the
stored vectors whose length matches the query, and filters them by
``(provider, model)``. A different model, or the same model at different
dimensions, therefore matches nothing. Re-indexing does not repair it either:
chunks that already hold vectors for the provider and model are skipped at any
dimension.

The index therefore owns its model and dimensions. They are recorded in a
small file beside the database, so which model an index uses never depends on
counting vectors. A configuration change is a proposal that has to be
accepted, not an instruction that silently rewrites or strands the database.
The registry decides what to pin, and only the CLI ever asks a human.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

ACTIVE_MODEL_SUFFIX = ".embedding.json"
_ACTIVE_MODEL_VERSION = 1


class _EmbeddingInventory(Protocol):
    """The slice of a database provider this module needs."""

    def get_embedding_model_counts(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ModelGroup:
    """Stored vectors sharing one provider, model and dimension count.

    ``latest`` is the newest write as a POSIX timestamp, or -inf when the
    backend records none.
    """

    provider: str
    model: str
    dims: int
    count: int
    latest: float


@dataclass(frozen=True)
class ActiveModel:
    """The embedding setup an index is committed to, as recorded on disk.

    ``dims`` is None when a switch was accepted before the new model's
    dimensions were known; they are then read from its stored vectors.
    """

    provider: str
    model: str
    dims: int | None


@dataclass(frozen=True)
class IndexedEmbeddingModel:
    """The provider, model and dimensions the index is using."""

    provider: str
    model: str
    dims: int | None
    embedding_count: int

    def __str__(self) -> str:
        dims = f"{self.dims} dims" if self.dims is not None else "dims unknown"
        return f"{self.provider}/{self.model} @ {dims}"


@dataclass(frozen=True)
class ModelDrift:
    """Embedding configuration that disagrees with what the index uses.

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
            and self.indexed.dims is not None
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


class ActiveModelStateError(Exception):
    """The active-model record exists but cannot be trusted."""


def read_index_groups(db: _EmbeddingInventory) -> list[ModelGroup] | None:
    """Summarize the index's stored vectors, or None if it cannot be read.

    None and an empty list mean different things: an empty index has nothing
    to protect, while an unreadable one leaves drift protection off for this
    run, which is logged rather than guessed at.
    """
    try:
        rows = db.get_embedding_model_counts()
    except Exception as e:
        logger.warning(
            "Could not read the index's embedding models, so drift protection "
            f"is off for this run: {e}"
        )
        return None
    return [
        ModelGroup(
            provider=row["provider"],
            model=row["model"],
            dims=int(row["dims"]),
            count=int(row["count"]),
            latest=_timestamp(row.get("latest")),
        )
        for row in rows
        if row["count"]
    ]


def _timestamp(value: object) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, int | float):
        return float(value)
    return float("-inf")


def active_model_path(db_path: Path | str) -> Path | None:
    """Where a database's active-model record lives, or None in memory."""
    if str(db_path) == ":memory:":
        return None
    path = Path(db_path)
    return path.with_name(path.name + ACTIVE_MODEL_SUFFIX)


def read_active_model(path: Path) -> ActiveModel | None:
    """Return the recorded active model, or None if none has been recorded.

    Raises ActiveModelStateError when a record exists but is unreadable,
    malformed, or in a newer format, so a caller never overwrites what it does
    not understand.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ActiveModelStateError(f"{path} could not be read: {e}") from e
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ActiveModelStateError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(payload, dict) or payload.get("version") != (
        _ACTIVE_MODEL_VERSION
    ):
        raise ActiveModelStateError(f"{path} is not a version 1 record")
    provider = payload.get("provider")
    model = payload.get("model")
    dims = payload.get("dims")
    valid_dims = dims is None or (
        isinstance(dims, int) and not isinstance(dims, bool) and dims > 0
    )
    if not isinstance(provider, str) or not isinstance(model, str) or not valid_dims:
        raise ActiveModelStateError(f"{path} is missing provider, model or dims")
    return ActiveModel(provider=provider, model=model, dims=dims)


def write_active_model(path: Path, active: ActiveModel) -> None:
    """Record the active model atomically, so a crash never leaves half a file."""
    payload = {
        "version": _ACTIVE_MODEL_VERSION,
        "provider": active.provider,
        "model": active.model,
        "dims": active.dims,
    }
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resolve_indexed_model(
    groups: list[ModelGroup], active: ActiveModel | None
) -> IndexedEmbeddingModel | None:
    """Decide which of the stored models the index is using.

    A recorded active model wins outright, even before any of its vectors
    exist: that is what an accepted but unfinished switch looks like, and the
    next index run fills it in. Without a record, the model with the most
    vectors wins, and a tie goes to the most recently written one, which is
    the model a completed switch moved to.
    """
    if not groups:
        return None
    if active is not None:
        matching = [
            group
            for group in groups
            if (group.provider, group.model) == (active.provider, active.model)
            and (active.dims is None or group.dims == active.dims)
        ]
        dims = active.dims
        if dims is None and matching:
            dims = max(matching, key=lambda group: group.count).dims
        return IndexedEmbeddingModel(
            provider=active.provider,
            model=active.model,
            dims=dims,
            embedding_count=sum(group.count for group in matching),
        )
    dominant = max(groups, key=lambda group: (group.count, group.latest))
    return IndexedEmbeddingModel(
        provider=dominant.provider,
        model=dominant.model,
        dims=dominant.dims,
        embedding_count=dominant.count,
    )


def detect_model_drift(
    indexed: IndexedEmbeddingModel,
    configured_provider: str,
    configured_model: str,
    configured_dims: int | None,
) -> ModelDrift | None:
    """Return drift between the index and the configuration, if any."""
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

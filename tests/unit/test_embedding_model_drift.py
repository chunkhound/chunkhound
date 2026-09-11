"""Contract tests for embedding drift against an existing index.

The invariant under test: an index built with one embedding model and
dimensions is never silently re-embedded under, or stranded by, another.
Search reads the stored vectors matching the query's length and filters them
by provider and model, so an unannounced change returns nothing at all, and
re-indexing skips chunks that already hold vectors for the provider and model.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.constants import EMBEDDING_MODEL_UPGRADES
from chunkhound.core.embedding_model_drift import (
    ActiveModel,
    ActiveModelStateError,
    IndexedEmbeddingModel,
    ModelDrift,
    ModelGroup,
    active_model_path,
    detect_model_drift,
    format_drift_warning,
    read_active_model,
    read_index_groups,
    resolve_indexed_model,
    write_active_model,
)
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.embeddings.voyageai_provider import VOYAGE_MODEL_CONFIG
from chunkhound.registry import ModelDriftDecision, ProviderRegistry

# ---------------------------------------------------------------------------
# Which model the index uses, and whether the configuration disagrees
# ---------------------------------------------------------------------------


def _group(
    model: str = "voyage-code-3",
    *,
    dims: int = 1024,
    count: int = 12431,
    latest: float = 0.0,
) -> ModelGroup:
    return ModelGroup(
        provider="voyageai", model=model, dims=dims, count=count, latest=latest
    )


_CODE_3 = IndexedEmbeddingModel(
    provider="voyageai", model="voyage-code-3", dims=1024, embedding_count=12431
)


class _ExplodingDatabase:
    def get_embedding_model_counts(self) -> list[dict[str, Any]]:
        raise RuntimeError("database is locked")


class TestWhichModelTheIndexUses:
    def test_empty_index_has_no_model(self):
        assert resolve_indexed_model([], None) is None

    def test_model_with_the_most_vectors_wins(self):
        groups = [_group("voyage-code-4", count=300), _group(count=12431)]

        indexed = resolve_indexed_model(groups, None)

        assert indexed is not None
        assert indexed.model == "voyage-code-3"

    @pytest.mark.parametrize("reverse", [False, True], ids=["old-first", "new-first"])
    def test_a_tie_goes_to_the_most_recently_written(self, reverse: bool):
        """A completed switch leaves every chunk embedded under both models.

        The newer one is the model the switch moved to, whatever order the
        backend happens to return the groups in.
        """
        groups = [
            _group("voyage-code-3", count=50, latest=1.0),
            _group("voyage-code-4", count=50, latest=2.0),
        ]

        indexed = resolve_indexed_model(groups[::-1] if reverse else groups, None)

        assert indexed is not None
        assert indexed.model == "voyage-code-4"

    def test_recorded_model_wins_over_vector_counts(self):
        """An accepted switch interrupted early has few new vectors, but it is
        still the model the operator chose."""
        groups = [_group(count=12431), _group("voyage-code-4", dims=2048, count=3)]

        indexed = resolve_indexed_model(
            groups, ActiveModel("voyageai", "voyage-code-4", 2048)
        )

        assert indexed is not None
        assert (indexed.model, indexed.dims, indexed.embedding_count) == (
            "voyage-code-4",
            2048,
            3,
        )

    def test_unreadable_index_is_reported_not_guessed(self):
        assert read_index_groups(_ExplodingDatabase()) is None


class TestDriftDetection:
    def test_reports_drift_when_configured_model_differs(self):
        drift = detect_model_drift(_CODE_3, "voyageai", "voyage-code-4", 1024)

        assert drift is not None
        assert drift.model_changed
        assert not drift.dims_changed

    def test_no_drift_when_index_and_config_agree(self):
        assert detect_model_drift(_CODE_3, "voyageai", "voyage-code-3", 1024) is None

    def test_dims_change_on_the_same_model_is_drift(self):
        """Same model at other dims queries vectors the index never wrote."""
        drift = detect_model_drift(_CODE_3, "voyageai", "voyage-code-3", 512)

        assert drift is not None
        assert drift.dims_changed
        assert not drift.model_changed

    def test_unknown_configured_dims_compare_model_only(self):
        """A provider that cannot know its dims before a call is not guessed at."""
        assert detect_model_drift(_CODE_3, "voyageai", "voyage-code-3", None) is None

    def test_warning_states_the_cost_of_a_model_switch(self):
        drift = detect_model_drift(_CODE_3, "voyageai", "voyage-code-4", 1024)
        assert drift is not None

        message = format_drift_warning(drift)

        assert "voyage-code-3" in message
        assert "voyage-code-4" in message
        assert "12,431" in message

    def test_warning_says_a_dims_change_cannot_be_reembedded(self):
        drift = detect_model_drift(_CODE_3, "voyageai", "voyage-code-3", 512)
        assert drift is not None

        message = format_drift_warning(drift)

        assert "keeps 1024 dims" in message
        assert "delete the database directory" in message


class TestActiveModelRecord:
    """The record beside the database is a durable, versioned format."""

    def test_absent_record_reads_as_none(self, tmp_path: Path):
        assert read_active_model(tmp_path / "chunks.db.embedding.json") is None

    def test_record_round_trips(self, tmp_path: Path):
        path = tmp_path / "chunks.db.embedding.json"
        active = ActiveModel("voyageai", "voyage-code-4", 2048)

        write_active_model(path, active)

        assert read_active_model(path) == active

    @pytest.mark.parametrize(
        "content",
        ["{not json", json.dumps({"version": 2, "provider": "x", "model": "y"})],
        ids=["malformed", "newer-format"],
    )
    def test_a_record_it_cannot_understand_is_refused(
        self, tmp_path: Path, content: str
    ):
        """Refusing, rather than reading it as absent, keeps it from being
        overwritten, which would discard what a newer ChunkHound recorded."""
        path = tmp_path / "chunks.db.embedding.json"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(ActiveModelStateError):
            read_active_model(path)

    def test_in_memory_database_has_no_record(self):
        assert active_model_path(":memory:") is None


# ---------------------------------------------------------------------------
# Resolution: configure() against a real index
# ---------------------------------------------------------------------------

_CODE_3_INDEX = [("voyageai", "voyage-code-3", 1024)]


def _db_config(tmp_path: Path, *, read_only: bool = False) -> DatabaseConfig:
    return DatabaseConfig(
        provider="duckdb", path=tmp_path / ".chunkhound" / "db", read_only=read_only
    )


def _seed_index(
    tmp_path: Path, indexed: list[tuple[str, str, int]], *, vectors: int = 2
) -> DatabaseConfig:
    """Write ``vectors`` per (provider, model, dims) into a real DuckDB index."""
    db_config = _db_config(tmp_path)
    db_path = db_config.get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    database = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
    database.connect()
    try:
        for provider, model, dims in indexed:
            database.insert_embeddings_batch(
                [
                    {
                        "chunk_id": chunk_id,
                        "provider": provider,
                        "model": model,
                        "embedding": [0.1] * dims,
                        "dims": dims,
                    }
                    for chunk_id in range(1, vectors + 1)
                ]
            )
    finally:
        database.disconnect()
    return db_config


class _Index:
    """Seeds a real DuckDB index and runs ``ProviderRegistry.configure``."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self._open: list[ProviderRegistry] = []

    def seed(
        self, indexed: list[tuple[str, str, int]], *, vectors: int = 2
    ) -> DatabaseConfig:
        return _seed_index(self.tmp_path, indexed, vectors=vectors)

    def configure(
        self,
        *,
        model: str,
        output_dims: int | None = None,
        on_model_drift: ModelDriftDecision | None = None,
        read_only: bool = False,
    ) -> tuple[ProviderRegistry, Config]:
        config = Config(target_dir=self.tmp_path)
        config.database = _db_config(self.tmp_path, read_only=read_only)
        config.embedding = EmbeddingConfig(
            provider="voyageai",
            model=model,
            api_key="test-key",
            output_dims=output_dims,
        )
        registry = ProviderRegistry()
        self._open.append(registry)
        registry.configure(config, on_model_drift)
        return registry, config

    def close(self, registry: ProviderRegistry) -> None:
        registry.get_provider("database").disconnect(skip_checkpoint=True)
        self._open.remove(registry)

    def close_all(self) -> None:
        for registry in list(self._open):
            self.close(registry)


@pytest.fixture
def index(tmp_path: Path, clean_environment: None) -> Iterator[_Index]:
    harness = _Index(tmp_path)
    yield harness
    harness.close_all()


def _decline(drift: ModelDrift) -> bool:
    return False


def _accept(drift: ModelDrift) -> bool:
    return True


def _recorder(consulted: list[ModelDrift]) -> Callable[[ModelDrift], bool]:
    def record(drift: ModelDrift) -> bool:
        consulted.append(drift)
        return False

    return record


class TestConfigureKeepsTheIndexUsable:
    @pytest.mark.parametrize(
        "decision", [None, _decline], ids=["no-hook-as-mcp", "declined"]
    )
    def test_unaccepted_switch_keeps_indexed_model_and_dims(
        self, index: _Index, decision: ModelDriftDecision | None
    ):
        index.seed(_CODE_3_INDEX)

        registry, _ = index.configure(
            model="voyage-code-4", output_dims=2048, on_model_drift=decision
        )

        provider = registry.get_provider("embedding")
        assert (provider.model, provider.dims) == ("voyage-code-3", 1024)

    def test_pin_reaches_providers_rebuilt_from_config(self, index: _Index):
        """``search`` and ``create_services`` build providers from config.

        If only the registry's instance were pinned, those would embed queries
        for a model and dims the index does not hold and find nothing.
        """
        index.seed(_CODE_3_INDEX)
        _, config = index.configure(model="voyage-code-4", output_dims=2048)
        assert config.embedding is not None

        rebuilt = EmbeddingProviderFactory.create_provider(config.embedding)

        assert (rebuilt.model, rebuilt.dims) == ("voyage-code-3", 1024)

    def test_unknown_configured_model_decline_restores_indexed_dims(
        self, index: _Index
    ):
        """An unknown model reports no dims, so nothing flags a dims change.

        The pin must still restore the index's dims, or the restored model
        embeds queries at its default size and search stays empty.
        """
        index.seed([("voyageai", "voyage-code-3", 2048)])

        registry, _ = index.configure(model="acme-custom", on_model_drift=_decline)

        provider = registry.get_provider("embedding")
        assert (provider.model, provider.dims) == ("voyage-code-3", 2048)

    def test_dims_change_on_the_same_model_keeps_indexed_dims_even_if_accepted(
        self, index: _Index
    ):
        """Re-indexing skips chunks already embedded under the model, so the
        new dims could never be written: accepting must not strand search."""
        index.seed(_CODE_3_INDEX)

        registry, config = index.configure(
            model="voyage-code-3", output_dims=512, on_model_drift=_accept
        )
        assert config.embedding is not None

        assert registry.get_provider("embedding").dims == 1024
        assert config.embedding.output_dims == 1024

    def test_accepted_model_switch_uses_configured_model_and_dims(self, index: _Index):
        index.seed(_CODE_3_INDEX)

        registry, config = index.configure(
            model="voyage-code-4", output_dims=2048, on_model_drift=_accept
        )
        assert config.embedding is not None

        provider = registry.get_provider("embedding")
        assert (provider.model, provider.dims) == ("voyage-code-4", 2048)
        assert (config.embedding.model, config.embedding.output_dims) == (
            "voyage-code-4",
            2048,
        )

    def test_completed_switch_survives_a_fresh_start(self, index: _Index):
        """Once the re-embed finishes, both models hold every chunk. The next
        start must keep the model the operator switched to, not the old one."""
        index.seed(_CODE_3_INDEX)
        first, _ = index.configure(
            model="voyage-code-4", output_dims=2048, on_model_drift=_accept
        )
        index.close(first)
        index.seed([("voyageai", "voyage-code-4", 2048)])

        consulted: list[ModelDrift] = []
        registry, _ = index.configure(
            model="voyage-code-4", output_dims=2048, on_model_drift=_recorder(consulted)
        )

        assert consulted == []
        provider = registry.get_provider("embedding")
        assert (provider.model, provider.dims) == ("voyage-code-4", 2048)

    def test_interrupted_switch_keeps_the_chosen_model(self, index: _Index):
        """Counting vectors would pick the old model here; the record does not."""
        index.seed(_CODE_3_INDEX, vectors=4)
        first, _ = index.configure(
            model="voyage-code-4", output_dims=2048, on_model_drift=_accept
        )
        index.close(first)
        index.seed([("voyageai", "voyage-code-4", 2048)], vectors=1)

        consulted: list[ModelDrift] = []
        registry, _ = index.configure(
            model="voyage-code-4", output_dims=2048, on_model_drift=_recorder(consulted)
        )

        assert consulted == []
        assert registry.get_provider("embedding").model == "voyage-code-4"

    def test_first_start_records_the_active_model(self, index: _Index):
        db_config = index.seed(_CODE_3_INDEX)

        index.configure(model="voyage-code-3")

        path = active_model_path(db_config.get_db_path())
        assert path is not None
        assert read_active_model(path) == ActiveModel("voyageai", "voyage-code-3", 1024)

    def test_read_only_database_records_nothing(self, index: _Index):
        db_config = index.seed(_CODE_3_INDEX)

        index.configure(model="voyage-code-3", read_only=True)

        path = active_model_path(db_config.get_db_path())
        assert path is not None
        assert not path.exists()

    def test_matching_index_is_left_alone(self, index: _Index):
        index.seed(_CODE_3_INDEX)
        consulted: list[ModelDrift] = []

        registry, _ = index.configure(
            model="voyage-code-3", on_model_drift=_recorder(consulted)
        )

        assert consulted == []
        assert registry.get_provider("embedding").dims == 1024

    def test_custom_indexed_model_is_pinned_with_its_dims(self, index: _Index):
        index.seed([("voyageai", "acme-custom", 777)])

        registry, _ = index.configure(model="voyage-3.5")

        provider = registry.get_provider("embedding")
        assert (provider.model, provider.dims) == ("acme-custom", 777)

    def test_index_dims_invalid_for_its_model_do_not_crash_startup(self, index: _Index):
        """``voyage-law-2`` only produces 1024 dims, so 2048 cannot be pinned.

        configure() also runs at MCP startup, so it keeps the model and warns
        rather than raising.
        """
        index.seed([("voyageai", "voyage-law-2", 2048)])

        registry, _ = index.configure(model="voyage-3.5")

        assert registry.get_provider("embedding").model == "voyage-law-2"

    def test_provider_change_is_not_pinned(self, index: _Index):
        """A different provider has other credentials and dimensions, so there
        is nothing to fall back to and the configured provider stands."""
        index.seed([("openai", "text-embedding-3-small", 1536)])

        registry, _ = index.configure(model="voyage-3.5")

        assert registry.get_provider("embedding").model == "voyage-3.5"

    def test_provider_built_before_configure_is_held_to_the_index(
        self, tmp_path: Path, clean_environment: None
    ):
        """The MCP server and ``research`` build their provider first.

        They create it from config, then call ``create_services``, which
        configures the registry and registers that earlier instance. It must be
        pinned too, or MCP search queries a model and dims the index does not
        hold and returns nothing.
        """
        config = Config(target_dir=tmp_path)
        config.database = _seed_index(tmp_path, _CODE_3_INDEX)
        config.embedding = EmbeddingConfig(
            provider="voyageai",
            model="voyage-code-4",
            api_key="test-key",
            output_dims=2048,
        )
        early = EmbeddingProviderFactory.create_provider(config.embedding)
        embedding_manager = EmbeddingManager()
        embedding_manager.register_provider(early, set_default=True)

        services = create_services(
            db_path=Path(config.database.path),
            config=config,
            embedding_manager=embedding_manager,
        )
        try:
            assert (early.model, early.dims) == ("voyage-code-3", 1024)
        finally:
            services.provider.disconnect(skip_checkpoint=True)

    def test_config_without_embeddings_leaves_other_providers_alone(
        self, tmp_path: Path, clean_environment: None
    ):
        """``create_services`` registers a caller's provider after configure().

        The next configure() without an embedding section built nothing of its
        own, so it must not treat that leftover as its provider and fail on it.
        """
        leftover = SimpleNamespace(name="dummy", model="dummy")
        config = Config(target_dir=tmp_path)
        config.database = _seed_index(tmp_path, _CODE_3_INDEX)
        config.embedding = None
        registry = ProviderRegistry()
        registry.register_provider("embedding", leftover)
        try:
            registry.configure(config)

            assert registry.get_provider("embedding") is leftover
        finally:
            registry.get_provider("database").disconnect(skip_checkpoint=True)


class TestLanceDBIndex:
    def test_lancedb_index_keeps_its_model(
        self, tmp_path: Path, clean_environment: None
    ):
        """Drift protection reads the index through the provider, so LanceDB,
        which has no SQL catalogue, is covered like DuckDB."""
        pytest.importorskip("lancedb")
        from chunkhound.core.models import Chunk, File
        from chunkhound.core.types.common import ChunkType, Language
        from chunkhound.providers.database.lancedb_provider import LanceDBProvider

        db_config = DatabaseConfig(
            provider="lancedb", path=tmp_path / ".chunkhound" / "db"
        )
        seeder = LanceDBProvider(str(db_config.get_db_path()), base_directory=tmp_path)
        seeder.connect()
        try:
            file_id = seeder.insert_file(
                File(path="a.py", mtime=1.0, language=Language.PYTHON, size_bytes=1)
            )
            chunk_ids = seeder.insert_chunks_batch(
                [
                    Chunk(
                        file_id=file_id,
                        code=f"def f{i}():\n    return {i}",
                        start_line=i + 1,
                        end_line=i + 1,
                        chunk_type=ChunkType.FUNCTION,
                        language=Language.PYTHON,
                        symbol=f"f{i}",
                    )
                    for i in range(2)
                ]
            )
            seeder.insert_embeddings_batch(
                [
                    {
                        "chunk_id": chunk_id,
                        "provider": "voyageai",
                        "model": "voyage-code-3",
                        "dims": 1024,
                        "embedding": [0.1] * 1024,
                    }
                    for chunk_id in chunk_ids
                ]
            )
        finally:
            seeder.disconnect()

        config = Config(target_dir=tmp_path)
        config.database = db_config
        config.embedding = EmbeddingConfig(
            provider="voyageai",
            model="voyage-code-4",
            api_key="test-key",
            output_dims=2048,
        )
        registry = ProviderRegistry()
        registry.configure(config, _decline)
        try:
            provider = registry.get_provider("embedding")
            assert (provider.model, provider.dims) == ("voyage-code-3", 1024)
        finally:
            registry.get_provider("database").disconnect()


class TestUpgradeSuggestions:
    def test_every_suggestion_names_models_the_provider_knows(self):
        """A hint pointing at a model the provider cannot batch is worse than none."""
        for superseded, successor in EMBEDDING_MODEL_UPGRADES["voyageai"].items():
            assert superseded in VOYAGE_MODEL_CONFIG
            assert successor in VOYAGE_MODEL_CONFIG

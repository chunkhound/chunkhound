"""Contract tests for embedding drift against an existing index.

The invariant under test: an index built with one embedding model and
dimensions is never silently re-embedded under, or stranded by, another.
Search reads the ``embeddings_<dims>`` table matching the query and filters it
by provider and model, so an unannounced change returns nothing at all, and
re-indexing skips chunks that already hold vectors for the provider and model.
"""

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
    ModelDrift,
    detect_indexed_model,
    detect_model_drift,
    format_drift_warning,
)
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.embeddings.voyageai_provider import VOYAGE_MODEL_CONFIG
from chunkhound.registry import ModelDriftDecision, ProviderRegistry

# ---------------------------------------------------------------------------
# Detection: pure functions over what the index reports
# ---------------------------------------------------------------------------


class _FakeDatabase:
    """Answers the two queries the drift check makes, and nothing else."""

    def __init__(self, rows_by_table: dict[str, list[dict[str, Any]]]):
        self._rows_by_table = rows_by_table

    def execute_query(
        self, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        if "information_schema" in query:
            return [{"table_name": name} for name in self._rows_by_table]
        for name, rows in self._rows_by_table.items():
            if f"FROM {name} " in query:
                return rows
        raise AssertionError(f"unexpected query: {query}")


class _ExplodingDatabase:
    def execute_query(
        self, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        raise RuntimeError("database is locked")


def _db(provider: str = "voyageai", model: str = "voyage-code-3", count: int = 12431):
    return _FakeDatabase(
        {
            "embeddings_1024": [
                {"provider": provider, "model": model, "dims": 1024, "count": count}
            ]
        }
    )


class TestDriftDetection:
    def test_reports_drift_when_configured_model_differs(self):
        drift = detect_model_drift(_db(), "voyageai", "voyage-code-4", 1024)

        assert drift is not None
        assert drift.model_changed
        assert not drift.dims_changed
        assert drift.indexed.model == "voyage-code-3"
        assert drift.indexed.embedding_count == 12431

    def test_no_drift_when_index_and_config_agree(self):
        assert detect_model_drift(_db(), "voyageai", "voyage-code-3", 1024) is None

    def test_dims_change_on_the_same_model_is_drift(self):
        """Same model at other dims queries a table the index never wrote."""
        drift = detect_model_drift(_db(), "voyageai", "voyage-code-3", 512)

        assert drift is not None
        assert drift.dims_changed
        assert not drift.model_changed

    def test_unknown_configured_dims_compare_model_only(self):
        """A provider that cannot know its dims before a call is not guessed at."""
        assert detect_model_drift(_db(), "voyageai", "voyage-code-3", None) is None

    def test_no_drift_on_empty_index(self):
        empty = _FakeDatabase({"embeddings_1024": []})

        assert detect_indexed_model(empty) is None
        assert detect_model_drift(empty, "voyageai", "voyage-code-4", 1024) is None

    def test_unreadable_database_reports_no_drift(self):
        """An unreadable count must not block indexing outright."""
        assert detect_indexed_model(_ExplodingDatabase()) is None

    def test_half_finished_switch_reports_the_larger_model(self):
        """A partial re-embed leaves two models; the dominant one wins.

        Picking by row count keeps the answer stable instead of depending on
        whichever row the database happens to return first.
        """
        partial = _FakeDatabase(
            {
                "embeddings_1024": [
                    {
                        "provider": "voyageai",
                        "model": "voyage-code-3",
                        "dims": 1024,
                        "count": 12431,
                    },
                    {
                        "provider": "voyageai",
                        "model": "voyage-code-4",
                        "dims": 1024,
                        "count": 300,
                    },
                ]
            }
        )

        indexed = detect_indexed_model(partial)

        assert indexed is not None
        assert indexed.model == "voyage-code-3"

    def test_warning_states_the_cost_of_a_model_switch(self):
        drift = detect_model_drift(_db(), "voyageai", "voyage-code-4", 1024)
        assert drift is not None

        message = format_drift_warning(drift)

        assert "voyage-code-3" in message
        assert "voyage-code-4" in message
        assert "12,431" in message

    def test_warning_says_a_dims_change_cannot_be_reembedded(self):
        drift = detect_model_drift(_db(), "voyageai", "voyage-code-3", 512)
        assert drift is not None

        message = format_drift_warning(drift)

        assert "keeps 1024 dims" in message
        assert "delete the database directory" in message


# ---------------------------------------------------------------------------
# Resolution: configure() against a real DuckDB index
# ---------------------------------------------------------------------------

_CODE_3_INDEX = [("voyageai", "voyage-code-3", 1024)]


def _seed_index(tmp_path: Path, indexed: list[tuple[str, str, int]]) -> DatabaseConfig:
    """Write two vectors per (provider, model, dims) into a real DuckDB index."""
    db_config = DatabaseConfig(provider="duckdb", path=tmp_path / ".chunkhound" / "db")
    db_path = db_config.get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    database = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
    database.connect()
    try:
        chunk_id = 1
        for provider, model, dims in indexed:
            database.insert_embeddings_batch(
                [
                    {
                        "chunk_id": chunk_id + offset,
                        "provider": provider,
                        "model": model,
                        "embedding": [0.1] * dims,
                        "dims": dims,
                    }
                    for offset in range(2)
                ]
            )
            chunk_id += 2
    finally:
        database.disconnect()
    return db_config


ConfigureIndex = Callable[..., tuple[ProviderRegistry, Config]]


@pytest.fixture
def configure_index(
    tmp_path: Path, clean_environment: None
) -> Iterator[ConfigureIndex]:
    """Seed an index, then run ``ProviderRegistry.configure`` against it."""
    registries: list[ProviderRegistry] = []

    def _configure(
        indexed: list[tuple[str, str, int]],
        *,
        model: str,
        output_dims: int | None = None,
        on_model_drift: ModelDriftDecision | None = None,
    ) -> tuple[ProviderRegistry, Config]:
        config = Config(target_dir=tmp_path)
        config.database = _seed_index(tmp_path, indexed)
        config.embedding = EmbeddingConfig(
            provider="voyageai",
            model=model,
            api_key="test-key",
            output_dims=output_dims,
        )
        registry = ProviderRegistry()
        registries.append(registry)
        registry.configure(config, on_model_drift)
        return registry, config

    yield _configure

    for registry in registries:
        registry.get_provider("database").disconnect(skip_checkpoint=True)


def _decline(drift: ModelDrift) -> bool:
    return False


def _accept(drift: ModelDrift) -> bool:
    return True


class TestConfigureKeepsTheIndexUsable:
    @pytest.mark.parametrize(
        "decision", [None, _decline], ids=["no-hook-as-mcp", "declined"]
    )
    def test_unaccepted_switch_keeps_indexed_model_and_dims(
        self,
        configure_index: ConfigureIndex,
        decision: ModelDriftDecision | None,
    ):
        registry, _ = configure_index(
            _CODE_3_INDEX,
            model="voyage-code-4",
            output_dims=2048,
            on_model_drift=decision,
        )

        provider = registry.get_provider("embedding")
        assert (provider.model, provider.dims) == ("voyage-code-3", 1024)

    def test_pin_reaches_providers_rebuilt_from_config(
        self, configure_index: ConfigureIndex
    ):
        """``search`` and ``create_services`` build providers from config.

        If only the registry's instance were pinned, those would embed queries
        for a model and table the index does not hold and find nothing.
        """
        _, config = configure_index(
            _CODE_3_INDEX, model="voyage-code-4", output_dims=2048
        )
        assert config.embedding is not None

        rebuilt = EmbeddingProviderFactory.create_provider(config.embedding)

        assert (rebuilt.model, rebuilt.dims) == ("voyage-code-3", 1024)

    def test_dims_change_on_the_same_model_keeps_indexed_dims_even_if_accepted(
        self, configure_index: ConfigureIndex
    ):
        """Re-indexing skips chunks already embedded under the model, so the
        new dims could never be written: accepting must not strand search."""
        registry, config = configure_index(
            _CODE_3_INDEX,
            model="voyage-code-3",
            output_dims=512,
            on_model_drift=_accept,
        )
        assert config.embedding is not None

        assert registry.get_provider("embedding").dims == 1024
        assert config.embedding.output_dims == 1024

    def test_accepted_model_switch_uses_configured_model_and_dims(
        self, configure_index: ConfigureIndex
    ):
        registry, config = configure_index(
            _CODE_3_INDEX,
            model="voyage-code-4",
            output_dims=2048,
            on_model_drift=_accept,
        )
        assert config.embedding is not None

        provider = registry.get_provider("embedding")
        assert (provider.model, provider.dims) == ("voyage-code-4", 2048)
        assert (config.embedding.model, config.embedding.output_dims) == (
            "voyage-code-4",
            2048,
        )

    def test_matching_index_is_left_alone(self, configure_index: ConfigureIndex):
        consulted: list[ModelDrift] = []

        def record(drift: ModelDrift) -> bool:
            consulted.append(drift)
            return False

        registry, _ = configure_index(
            _CODE_3_INDEX, model="voyage-code-3", on_model_drift=record
        )

        assert consulted == []
        assert registry.get_provider("embedding").dims == 1024

    def test_custom_indexed_model_is_pinned_with_its_dims(
        self, configure_index: ConfigureIndex
    ):
        registry, _ = configure_index(
            [("voyageai", "acme-custom", 777)], model="voyage-3.5"
        )

        provider = registry.get_provider("embedding")
        assert (provider.model, provider.dims) == ("acme-custom", 777)

    def test_index_dims_invalid_for_its_model_do_not_crash_startup(
        self, configure_index: ConfigureIndex
    ):
        """``voyage-law-2`` only produces 1024 dims, so 2048 cannot be pinned.

        configure() also runs at MCP startup, so it keeps the model and warns
        rather than raising.
        """
        registry, _ = configure_index(
            [("voyageai", "voyage-law-2", 2048)], model="voyage-3.5"
        )

        assert registry.get_provider("embedding").model == "voyage-law-2"

    def test_provider_built_before_configure_is_held_to_the_index(
        self, tmp_path: Path, clean_environment: None
    ):
        """The MCP server and ``research`` build their provider first.

        They create it from config, then call ``create_services``, which
        configures the registry and registers that earlier instance. It must be
        pinned too, or MCP search queries a model and table the index does not
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

    def test_provider_change_is_not_pinned(self, configure_index: ConfigureIndex):
        """A different provider has other credentials and dimensions, so there
        is nothing to fall back to and the configured provider stands."""
        registry, _ = configure_index(
            [("openai", "text-embedding-3-small", 1536)], model="voyage-3.5"
        )

        assert registry.get_provider("embedding").model == "voyage-3.5"


class TestUpgradeSuggestions:
    def test_every_suggestion_names_models_the_provider_knows(self):
        """A hint pointing at a model the provider cannot batch is worse than none."""
        for superseded, successor in EMBEDDING_MODEL_UPGRADES["voyageai"].items():
            assert superseded in VOYAGE_MODEL_CONFIG
            assert successor in VOYAGE_MODEL_CONFIG

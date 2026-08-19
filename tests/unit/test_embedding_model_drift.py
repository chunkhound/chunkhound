"""Contract tests for embedding-model drift against an existing index.

The invariant under test: an index built with one model is never silently
re-embedded under another. Search filters stored vectors by (provider, model),
so an unannounced switch both bills the user for a full re-embed and returns
nothing until that re-embed finishes.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.core.embedding_model_drift import (
    detect_indexed_model,
    detect_model_drift,
    format_drift_warning,
)
from chunkhound.registry import ProviderRegistry


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


class _FakeEmbeddingProvider:
    def __init__(self, name: str, model: str):
        self._name = name
        self.model = model
        self.update_calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    def update_config(self, **kwargs: Any) -> None:
        self.update_calls.append(kwargs)
        if "model" in kwargs:
            self.model = kwargs["model"]


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
        drift = detect_model_drift(_db(), "voyageai", "voyage-code-4")

        assert drift is not None
        assert drift.indexed.model == "voyage-code-3"
        assert drift.indexed.embedding_count == 12431
        assert drift.configured_model == "voyage-code-4"

    def test_no_drift_when_index_and_config_agree(self):
        assert detect_model_drift(_db(), "voyageai", "voyage-code-3") is None

    def test_no_drift_on_empty_index(self):
        empty = _FakeDatabase({"embeddings_1024": []})

        assert detect_indexed_model(empty) is None
        assert detect_model_drift(empty, "voyageai", "voyage-code-4") is None

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

    def test_warning_states_the_cost(self):
        drift = detect_model_drift(_db(), "voyageai", "voyage-code-4")
        assert drift is not None

        message = format_drift_warning(drift)

        assert "voyage-code-3" in message
        assert "voyage-code-4" in message
        assert "12,431" in message


class TestRegistryKeepsIndexedModel:
    """Without an operator decision, the index's model wins."""

    def _registry(self, configured_model: str, indexed_model: str = "voyage-code-3"):
        registry = ProviderRegistry()
        provider = _FakeEmbeddingProvider("voyageai", configured_model)
        registry.register_provider("embedding", provider)
        registry.register_provider("database", _db(model=indexed_model))
        return registry, provider

    def test_no_callback_pins_provider_to_indexed_model(self):
        registry, provider = self._registry("voyage-code-4")

        registry._resolve_embedding_model_drift(None)

        assert provider.model == "voyage-code-3"

    def test_declining_keeps_indexed_model(self):
        registry, provider = self._registry("voyage-code-4")

        registry._resolve_embedding_model_drift(lambda drift: False)

        assert provider.model == "voyage-code-3"

    def test_accepting_keeps_configured_model(self):
        registry, provider = self._registry("voyage-code-4")

        registry._resolve_embedding_model_drift(lambda drift: True)

        assert provider.model == "voyage-code-4"
        assert provider.update_calls == []

    def test_matching_model_is_left_alone(self):
        registry, provider = self._registry("voyage-code-3")

        registry._resolve_embedding_model_drift(None)

        assert provider.model == "voyage-code-3"
        assert provider.update_calls == []

    def test_pin_is_written_back_to_config(self):
        """Pinning the live provider alone is not enough.

        ``search`` and ``create_services`` build their own provider from
        ``config.embedding`` instead of asking the registry. If the config
        still names the configured model, those instances query a model the
        index does not hold and the search returns nothing at all.
        """
        from chunkhound.core.config.embedding_config import EmbeddingConfig

        registry, provider = self._registry("voyage-code-4")
        registry._config = SimpleNamespace(
            embeddings_disabled=False,
            embedding=EmbeddingConfig(
                provider="voyageai", model="voyage-code-4", api_key="test-key"
            ),
        )

        registry._resolve_embedding_model_drift(None)

        assert provider.model == "voyage-code-3"
        assert registry._config.embedding.model == "voyage-code-3"

    def test_provider_change_is_not_pinned(self):
        """A different provider has different credentials and dimensions.

        There is nothing to fall back to, so the configured provider stands.
        """
        registry = ProviderRegistry()
        provider = _FakeEmbeddingProvider("openai", "text-embedding-3-small")
        registry.register_provider("embedding", provider)
        registry.register_provider("database", _db(provider="voyageai"))

        registry._resolve_embedding_model_drift(None)

        assert provider.model == "text-embedding-3-small"
        assert provider.update_calls == []


class TestUpgradeSuggestion:
    def test_every_superseded_model_maps_to_a_known_model(self):
        """A hint pointing at a model the provider cannot batch is worse than none."""
        from chunkhound.core.constants import EMBEDDING_MODEL_UPGRADES
        from chunkhound.providers.embeddings.voyageai_provider import (
            VOYAGE_MODEL_CONFIG,
        )

        for superseded, successor in EMBEDDING_MODEL_UPGRADES.items():
            assert superseded in VOYAGE_MODEL_CONFIG
            assert successor in VOYAGE_MODEL_CONFIG

    @pytest.mark.parametrize(
        ("superseded", "successor"),
        [
            ("voyage-code-3", "voyage-code-4"),
            ("voyage-3.5", "voyage-4"),
            ("voyage-3.5-lite", "voyage-4-lite"),
            ("voyage-3-large", "voyage-4-large"),
        ],
    )
    def test_upgrade_targets_stay_in_family(self, superseded: str, successor: str):
        from chunkhound.core.constants import EMBEDDING_MODEL_UPGRADES

        assert EMBEDDING_MODEL_UPGRADES[superseded] == successor

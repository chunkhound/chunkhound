"""Provider registry and dependency injection container for ChunkHound.

Simplified from 540 lines to ~250 lines by:
- Removing string-based class detection
- Eliminating factory-within-factory pattern
- Using explicit provider creation methods
- Removing unused generic service creation
"""

import os
from collections.abc import Callable, MutableMapping
from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger

# Import centralized configuration
from chunkhound.core.config.config import Config

# Import embedding factory for unified provider creation
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.embedding_model_drift import ModelDrift, detect_model_drift

# Import core types
from chunkhound.core.types.common import Language
from chunkhound.embeddings import EmbeddingManager

# Import new unified parser system
from chunkhound.parsers.parser_factory import get_parser_factory

# Answers "the configured model differs from the indexed one; switch to it?".
# True re-embeds under the configured model, False keeps the indexed one.
ModelDriftDecision = Callable[[ModelDrift], bool]


class LazyLanguageParsers(MutableMapping[Language, Any]):
    """Mapping that lazily materializes language parsers on first access."""

    def __init__(self):
        self._factories: dict[Language, Callable[[], Any]] = {}
        self._instances: dict[Language, Any] = {}
        self._lock = Lock()

    def register_factory(self, language: Language, factory: Callable[[], Any]) -> None:
        """Register a factory used to materialize a parser lazily."""
        self._factories[language] = factory

    def materialized_copy(self) -> dict[Language, Any]:
        """Return copy of parsers that have already been instantiated."""
        with self._lock:
            return dict(self._instances)

    def __getitem__(self, key: Language) -> Any:
        with self._lock:
            if key in self._instances:
                return self._instances[key]

        factory = self._factories.get(key)
        if factory is None:
            raise KeyError(key)

        parser = factory()
        with self._lock:
            existing = self._instances.get(key)
            if existing is not None:
                return existing
            self._instances[key] = parser
            return parser

    def __setitem__(self, key: Language, value: Any) -> None:
        with self._lock:
            self._instances[key] = value
            self._factories.pop(key, None)

    def __delitem__(self, key: Language) -> None:
        with self._lock:
            self._instances.pop(key, None)
            self._factories.pop(key, None)

    def __iter__(self):
        with self._lock:
            combined = set(self._instances.keys()) | set(self._factories.keys())
        return iter(combined)

    def __len__(self) -> int:
        with self._lock:
            return len(set(self._instances.keys()) | set(self._factories.keys()))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, Language):
            return False
        with self._lock:
            return key in self._instances or key in self._factories

    def clear(self) -> None:
        with self._lock:
            self._instances.clear()
            self._factories.clear()


# Import services
from chunkhound.services.embedding_service import EmbeddingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService


class ProviderRegistry:
    """Registry for managing provider implementations and dependency injection."""

    def __init__(self):
        """Initialize the provider registry."""
        self._providers: dict[str, Any] = {}
        self._language_parsers: LazyLanguageParsers = LazyLanguageParsers()
        self._config: Config | None = None
        self._embedding_manager: EmbeddingManager | None = None

    def configure(
        self,
        config: Config,
        on_model_drift: "ModelDriftDecision | None" = None,
    ) -> None:
        """Configure the registry with application settings.

        ``on_model_drift`` is consulted only when the configured embedding
        model disagrees with the one the index was built with. Omitting it
        keeps the indexed model, which is what MCP and other non-interactive
        entry points want; the CLI passes a callback that asks the operator.
        """
        self._config = config

        # Create and register providers based on configuration
        self._setup_embedding_provider()
        self._setup_database_provider()
        self._setup_language_parsers()
        self._resolve_embedding_model_drift(on_model_drift)

    def _resolve_embedding_model_drift(
        self, on_model_drift: "ModelDriftDecision | None"
    ) -> None:
        """Keep using the model the index was built with, unless told otherwise.

        Search filters stored vectors by ``(provider, model)``, so pointing a
        different model at an existing index returns nothing until every chunk
        has been re-embedded. Pinning to the indexed model means a changed
        default never silently rewrites, or breaks, an existing database.
        """
        if self._config and getattr(self._config, "embeddings_disabled", False):
            return

        embedding_provider = self._providers.get("embedding")
        database_provider = self._providers.get("database")
        if embedding_provider is None or database_provider is None:
            return

        drift = detect_model_drift(
            database_provider,
            embedding_provider.name,
            embedding_provider.model,
        )
        if drift is None:
            return

        # A different provider cannot be pinned: its credentials, endpoint and
        # dimensions all differ, so there is nothing to fall back to. Say so
        # and let the configured provider re-embed.
        if drift.indexed.provider != embedding_provider.name:
            logger.warning(
                f"Index was built with {drift.indexed} but {drift.configured} "
                "is configured. Switching providers re-embeds every chunk."
            )
            return

        if on_model_drift is not None and on_model_drift(drift):
            logger.info(
                f"Re-embedding {drift.indexed.embedding_count:,} chunks "
                f"with {drift.configured} (accepted by operator)."
            )
            return

        # A caller that supplied a decision hook has already told the operator
        # what is happening, on its own stream; repeating it here duplicates
        # the message and interleaves badly with the CLI's own output.
        report = logger.debug if on_model_drift is not None else logger.warning

        embedding_provider.update_config(model=drift.indexed.model)

        # Pin the config as well as the live provider. Several callers build
        # their own provider straight from ``config.embedding`` rather than
        # asking the registry (``search`` and ``create_services`` both do), and
        # those instances would otherwise query a model the index does not hold
        # and quietly return nothing.
        if self._config is not None and self._config.embedding is not None:
            self._config.embedding.model = drift.indexed.model

        report(
            f"Using {drift.indexed} because the index was built with it "
            f"({drift.indexed.embedding_count:,} embeddings). Configured "
            f"{drift.configured} was not applied; nothing was re-embedded."
        )

    def register_provider(
        self, name: str, provider: Any, singleton: bool = True
    ) -> None:
        """Register a provider instance directly.

        Simplified: Takes actual instances instead of classes.
        """
        self._providers[name] = provider

        if not os.environ.get("CHUNKHOUND_MCP_MODE"):
            logger.debug(f"Registered {type(provider).__name__} as {name}")

    def register_language_parser(self, language: Language, parser_class: Any) -> None:
        """Register a language parser for a specific programming language."""
        # Create and setup parser instance
        parser = parser_class()
        if hasattr(parser, "setup"):
            parser.setup()

        self._language_parsers[language] = parser

        if not os.environ.get("CHUNKHOUND_MCP_MODE"):
            logger.debug(f"Registered {parser_class.__name__} for {language.value}")

    def get_provider(self, name: str) -> Any:
        """Get a provider instance by name."""
        logger.debug(
            f"[REGISTRY] Attempting to get provider '{name}', available providers: {list(self._providers.keys())}"
        )
        if name not in self._providers:
            logger.warning(
                f"[REGISTRY] No provider registered for {name}, available: {list(self._providers.keys())}"
            )
            raise ValueError(f"No provider registered for {name}")
        logger.debug(
            f"[REGISTRY] Successfully retrieved provider '{name}': {type(self._providers[name])}"
        )
        return self._providers[name]

    def get_language_parser(self, language: Language) -> Any | None:
        """Get parser for specified programming language."""
        try:
            return self._language_parsers[language]
        except KeyError:
            return None

    def get_all_language_parsers(self) -> dict[Language, Any]:
        """Get all registered language parsers."""
        return self._language_parsers.materialized_copy()

    def create_indexing_coordinator(self) -> IndexingCoordinator:
        """Create an IndexingCoordinator with all dependencies."""
        logger.debug("[REGISTRY] Creating IndexingCoordinator")
        database_provider = self.get_provider("database")
        embedding_provider = None

        # Respect explicit --no-embeddings: do not attempt lookup and avoid warnings
        if self._config and getattr(self._config, "embeddings_disabled", False):
            embedding_provider = None
            logger.debug("[REGISTRY] Embeddings disabled; skipping embedding provider")
        else:
            try:
                logger.debug(
                    "[REGISTRY] Attempting to get embedding provider for IndexingCoordinator"
                )
                embedding_provider = self.get_provider("embedding")
                logger.debug(
                    f"[REGISTRY] Successfully got embedding provider: {type(embedding_provider)}"
                )
            except ValueError as e:
                logger.warning(
                    f"[REGISTRY] No embedding provider configured for IndexingCoordinator: {e}"
                )
                pass  # No embedding provider configured

        # Get base directory from config (guaranteed to be set) or fallback to cwd
        base_directory = self._config.target_dir if self._config else Path.cwd()

        logger.debug(
            f"[REGISTRY] Creating IndexingCoordinator with embedding_provider={embedding_provider}"
        )
        return IndexingCoordinator(
            database_provider=database_provider,
            base_directory=base_directory,
            embedding_provider=embedding_provider,
            language_parsers=self._language_parsers,
            config=self._config,
        )

    def create_search_service(self) -> SearchService:
        """Create a SearchService with all dependencies."""
        database_provider = self.get_provider("database")
        embedding_provider = None

        if self._config and getattr(self._config, "embeddings_disabled", False):
            embedding_provider = None
            logger.debug(
                "[REGISTRY] Embeddings disabled; search service will run without embeddings"
            )
        else:
            try:
                embedding_provider = self.get_provider("embedding")
            except ValueError:
                logger.warning("No embedding provider configured for search service")

        return SearchService(
            database_provider=database_provider,
            embedding_provider=embedding_provider,
            config=self._config.research if self._config else None,
        )

    def create_embedding_service(self) -> EmbeddingService:
        """Create an EmbeddingService with all dependencies."""
        database_provider = self.get_provider("database")
        embedding_provider = None

        if self._config and getattr(self._config, "embeddings_disabled", False):
            embedding_provider = None
            logger.debug(
                "[REGISTRY] Embeddings disabled; embedding service will be inert"
            )
        else:
            try:
                embedding_provider = self.get_provider("embedding")
            except ValueError:
                logger.warning("No embedding provider configured for embedding service")

        # Get batch configuration from config
        if self._config and self._config.embedding:
            embedding_batch_size = self._config.embedding.batch_size
            max_concurrent = self._config.embedding.max_concurrent_batches
        else:
            embedding_batch_size = 1000
            max_concurrent = None

        db_batch_size = 5000
        if self._config and self._config.indexing:
            db_batch_size = self._config.indexing.db_batch_size

        return EmbeddingService(
            database_provider=database_provider,
            embedding_provider=embedding_provider,
            embedding_batch_size=embedding_batch_size,
            db_batch_size=db_batch_size,
            max_concurrent_batches=max_concurrent,
        )

    # Private setup methods - explicit provider creation

    def _setup_database_provider(self) -> None:
        """Create and register the database provider based on configuration."""
        if not self._config:
            # Default to DuckDB if no config
            from pathlib import Path

            from chunkhound.providers.database.duckdb_provider import DuckDBProvider

            provider = DuckDBProvider(
                db_path=".chunkhound/db", base_directory=Path.cwd()
            )
            provider.connect()
            self.register_provider("database", provider, singleton=True)
            return

        provider_type = self._config.database.provider
        # Use get_db_path() to get the actual database location (includes provider-specific transformations)
        db_path = str(self._config.database.get_db_path())

        # Get base directory from config (guaranteed to be set)
        base_directory = self._config.target_dir

        # Create the appropriate provider
        if provider_type == "duckdb":
            from chunkhound.providers.database.duckdb_provider import DuckDBProvider

            provider = DuckDBProvider(
                db_path, base_directory, config=self._config.database
            )
        elif provider_type == "lancedb":
            from chunkhound.providers.database.lancedb_provider import LanceDBProvider

            # Get embedding_manager if available for dimension detection
            embedding_manager = getattr(self, "_embedding_manager", None)

            provider = LanceDBProvider(
                db_path,
                base_directory,
                embedding_manager=embedding_manager,
                config=self._config.database,
            )
        else:
            logger.warning(f"Unknown provider {provider_type}, defaulting to DuckDB")
            from chunkhound.providers.database.duckdb_provider import DuckDBProvider

            provider = DuckDBProvider(
                db_path, base_directory, config=self._config.database
            )

        # Connect and register
        provider.connect()
        self.register_provider("database", provider, singleton=True)

    def _setup_embedding_provider(self) -> None:
        """Create and register the embedding provider if configured."""
        logger.debug("[REGISTRY] Setting up embedding provider")

        # Skip if no config at all
        if not self._config:
            logger.debug(
                "[REGISTRY] No config available, skipping embedding provider setup"
            )
            return

        # Skip if embeddings were explicitly disabled
        if (
            hasattr(self._config, "embeddings_disabled")
            and self._config.embeddings_disabled
        ):
            logger.debug(
                "[REGISTRY] Embeddings explicitly disabled, skipping embedding provider setup"
            )
            return

        # Skip if no embedding config found
        if not self._config.embedding:
            logger.debug(
                "[REGISTRY] No embedding config found, skipping embedding provider setup"
            )
            return

        logger.debug(
            f"[REGISTRY] Found embedding config: provider={self._config.embedding.provider}"
        )
        try:
            # Create EmbeddingManager and store as instance variable
            self._embedding_manager = EmbeddingManager()

            # Use the factory to create the provider
            logger.debug("[REGISTRY] Creating embedding provider from factory")
            provider = EmbeddingProviderFactory.create_provider(self._config.embedding)
            logger.debug(f"[REGISTRY] Created provider: {type(provider)}")

            # Register provider with the manager (enables dimension detection)
            self._embedding_manager.register_provider(provider, set_default=True)

            # Also register provider in registry for backward compatibility
            logger.debug("[REGISTRY] Registering embedding provider")
            self.register_provider("embedding", provider, singleton=True)
            logger.debug("[REGISTRY] Successfully registered embedding provider")

            if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                logger.info(
                    f"Registered {self._config.embedding.provider} embedding provider"
                )
        except Exception as e:
            logger.error(f"[REGISTRY] Failed to create embedding provider: {e}")
            raise  # Re-raise to see the actual error

    def _setup_language_parsers(self) -> None:
        """Register all available language parsers."""
        parser_factory = get_parser_factory()
        available_languages = parser_factory.get_available_languages()

        for language, is_available in available_languages.items():
            if is_available:
                try:
                    self._language_parsers.register_factory(
                        language,
                        lambda lang=language: parser_factory.create_parser(lang),
                    )
                    if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                        logger.debug(f"Registered parser factory for {language.value}")
                except Exception as e:
                    if not os.environ.get("CHUNKHOUND_MCP_MODE"):
                        logger.warning(
                            f"Failed to register parser for {language.value}: {e}"
                        )

    # Transaction management - delegates to database provider

    def begin_transaction(self) -> None:
        """Begin transaction on registered database provider."""
        database_provider = self.get_provider("database")
        if hasattr(database_provider, "begin_transaction"):
            database_provider.begin_transaction()

    def commit_transaction(self) -> None:
        """Commit transaction on registered database provider."""
        database_provider = self.get_provider("database")
        if hasattr(database_provider, "commit_transaction"):
            database_provider.commit_transaction()

    def rollback_transaction(self) -> None:
        """Rollback transaction on registered database provider."""
        database_provider = self.get_provider("database")
        if hasattr(database_provider, "rollback_transaction"):
            database_provider.rollback_transaction()

    def get_config(self) -> Config | None:
        """Get the current configuration instance."""
        return self._config


# Global registry instance
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def configure_registry(
    config: Config | dict[str, Any],
    on_model_drift: ModelDriftDecision | None = None,
) -> None:
    """Configure the global provider registry.

    ``on_model_drift`` lets an interactive caller decide what to do when the
    configured embedding model disagrees with the indexed one. Callers that
    cannot prompt (MCP, daemons) omit it and keep the indexed model.
    """
    if isinstance(config, dict):
        from chunkhound.core.config.config import Config as ConfigClass

        config_obj = ConfigClass(**config)
        get_registry().configure(config_obj, on_model_drift)
    else:
        get_registry().configure(config, on_model_drift)


# Convenience functions for common operations


def get_provider(name: str) -> Any:
    """Get a provider from the global registry."""
    return get_registry().get_provider(name)


def create_indexing_coordinator() -> IndexingCoordinator:
    """Create an IndexingCoordinator from the global registry."""
    return get_registry().create_indexing_coordinator()


def create_search_service() -> SearchService:
    """Create a SearchService from the global registry."""
    return get_registry().create_search_service()


def create_embedding_service() -> EmbeddingService:
    """Create an EmbeddingService from the global registry."""
    return get_registry().create_embedding_service()


__all__ = [
    "ProviderRegistry",
    "get_registry",
    "configure_registry",
    "get_provider",
    "create_indexing_coordinator",
    "create_search_service",
    "create_embedding_service",
]

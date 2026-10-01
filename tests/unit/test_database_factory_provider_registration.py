"""Contracts for best-effort embedding-provider registration in the factory."""

from unittest.mock import MagicMock, Mock

import chunkhound.database_factory as database_factory


def test_provider_registration_failure_warns_and_still_creates_services(monkeypatch):
    registration_error = RuntimeError("provider is unavailable")
    embedding_manager = Mock()
    embedding_manager.get_default_provider.side_effect = registration_error

    provider = object()
    coordinator = object()
    search_service = object()
    embedding_service = object()
    registry = MagicMock()
    registry.get_provider.return_value = provider
    registry.create_indexing_coordinator.return_value = coordinator
    registry.create_search_service.return_value = search_service
    registry.create_embedding_service.return_value = embedding_service

    logger = MagicMock()
    configure = Mock()
    monkeypatch.setattr(database_factory, "get_registry", lambda: registry)
    monkeypatch.setattr(database_factory, "configure_registry", configure)
    monkeypatch.setattr(database_factory, "logger", logger)

    services = database_factory.create_services(
        db_path=":memory:", config={}, embedding_manager=embedding_manager
    )

    assert services == database_factory.DatabaseServices(
        provider, coordinator, search_service, embedding_service
    )
    registry.get_config.assert_not_called()
    configure.assert_called_once()
    registry.create_indexing_coordinator.assert_called_once_with()
    registry.create_search_service.assert_called_once_with()
    registry.create_embedding_service.assert_called_once_with()
    logger.opt.assert_called_once_with(exception=True)
    logger.opt.return_value.warning.assert_called_once_with(
        "Provider registration failed: {}", registration_error
    )

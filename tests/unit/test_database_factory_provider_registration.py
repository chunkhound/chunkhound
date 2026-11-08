"""Factory behavior with an explicitly empty embedding owner."""

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager


def test_factory_with_empty_embedding_manager_still_serves_regex(tmp_path):
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "db"},
        embeddings_disabled=True,
        analytics={"enabled": False},
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )
    services = create_services(
        config.database.get_db_path(), config, embedding_manager=EmbeddingManager()
    )
    try:
        assert services.search_service.search_regex("never_matches")[0] == []
    finally:
        services.provider.close()

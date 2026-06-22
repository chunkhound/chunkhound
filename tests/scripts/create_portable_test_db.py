"""Create a portable DuckDB test DB from the real indexing pipeline.

The database contents must remain portable across OSs when chunks.db.root.json
is absent.
"""

import asyncio
import shutil
from pathlib import Path

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
# Imports from tests/ rely on PEP 420 implicit namespace packages
# (tests/fixtures/ has no __init__.py). This works when run via
# 'uv run python tests/scripts/...' from the project root.
from tests.fixtures.fake_providers import ConstantEmbeddingProvider


def main() -> None:
    repo = Path(".portable-db-source")
    db_root = Path(".chunkhound")
    db_path = db_root / "db" / "chunks.db"

    if repo.exists():
        shutil.rmtree(repo)
    if db_root.exists():
        shutil.rmtree(db_root)

    (repo / "src" / "utils").mkdir(parents=True)
    (repo / "lib" / "core").mkdir(parents=True)
    (repo / "src" / "utils" / "helpers.py").write_text(
        'def helper() -> str:\n    """Portable helper."""\n    return "ok"\n',
        encoding="utf-8",
    )
    (repo / "lib" / "core" / "engine.py").write_text(
        'class Engine:\n    def run(self) -> str:\n        return "running"\n',
        encoding="utf-8",
    )

    db_path.parent.mkdir(parents=True)
    provider = DuckDBProvider(db_path, base_directory=repo)
    provider.connect()

    try:
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=repo,
            embedding_provider=ConstantEmbeddingProvider(model="test-model", dims=8),
        )

        loop = asyncio.new_event_loop()
        try:
            stats = loop.run_until_complete(
                coordinator.process_directory(
                    repo,
                    patterns=["**/*.py"],
                    exclude_patterns=[],
                )
            )
            assert stats.get("files_processed", 0) == 2, stats
            assert stats.get("total_chunks", 0) >= 2, stats

            embed_stats = loop.run_until_complete(
                coordinator.generate_missing_embeddings()
            )
            assert embed_stats.get("generated", 0) > 0, embed_stats
        finally:
            loop.close()
    finally:
        provider.disconnect()

    sidecar = db_path.with_name(db_path.name + ".root.json")
    if sidecar.exists():
        sidecar.unlink()

    shutil.rmtree(repo)
    print(
        "Portable test database created successfully from real indexing without sidecar"
    )


if __name__ == "__main__":
    main()

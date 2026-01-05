from pathlib import Path

import pytest

from chunkhound.code_mapper.coverage import compute_db_scope_stats
from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from tests.integration.code_mapper_scope_helpers import (
    build_config,
    index_repo,
    patch_code_mapper_dependencies,
    run_code_mapper,
    write_scope_repo_layout,
)


@pytest.mark.asyncio
async def test_code_mapper_lancedb_scope_real_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    write_scope_repo_layout(repo_root)

    config = build_config(repo_root, provider="lancedb")
    db_path = config.database.get_db_path()

    embedding_provider = FakeEmbeddingProvider(batch_size=100)
    embedding_manager = EmbeddingManager()
    embedding_manager.register_provider(embedding_provider, set_default=True)

    provider = LanceDBProvider(
        db_path,
        base_directory=repo_root,
        embedding_manager=embedding_manager,
        config=config.database,
    )
    provider.connect()
    try:
        await index_repo(provider, repo_root, embedding_provider)
    finally:
        provider.disconnect()

    referenced_files = ["scope/a.py", "scope/b.py"]
    referenced_chunks = [
        {"file_path": "scope/a.py", "start_line": 1, "end_line": 2},
        {"file_path": "scope/b.py", "start_line": 1, "end_line": 2},
    ]
    patch_code_mapper_dependencies(
        monkeypatch,
        referenced_files=referenced_files,
        referenced_chunks=referenced_chunks,
    )

    out_dir = tmp_path / "out"
    await run_code_mapper(
        scope_path=repo_root / "scope", out_dir=out_dir, config=config
    )

    combined_docs = list(out_dir.glob("*_code_mapper.md"))
    assert combined_docs, "Expected combined Code Mapper document"
    combined_content = combined_docs[0].read_text(encoding="utf-8")
    assert "# Code Mapper for scope" in combined_content
    assert "other/" not in combined_content

    index_files = list(out_dir.glob("*_code_mapper_index.md"))
    assert index_files, "Expected Code Mapper index file"
    index_content = index_files[0].read_text(encoding="utf-8")
    assert "total_indexed: 3" in index_content
    assert "coverage: 66.67%" in index_content
    assert "basis: scope" in index_content
    assert "scope_scope_unreferenced_files.txt" in index_content
    assert "other/" not in index_content

    topic_files = list(out_dir.glob("*_topic_*.md"))
    assert topic_files, "Expected per-topic files"
    for topic_path in topic_files:
        content = topic_path.read_text(encoding="utf-8")
        assert "other/" not in content

    unref_files = list(out_dir.glob("*_scope_unreferenced_files.txt"))
    assert unref_files, "Expected unreferenced files artifact"
    unref_content = unref_files[0].read_text(encoding="utf-8")
    assert "scope/c.py" in unref_content
    assert "other/" not in unref_content


@pytest.mark.asyncio
async def test_lancedb_scope_stats_sanity(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    write_scope_repo_layout(repo_root)

    config = build_config(repo_root, provider="lancedb")
    db_path = config.database.get_db_path()

    embedding_provider = FakeEmbeddingProvider(batch_size=100)
    embedding_manager = EmbeddingManager()
    embedding_manager.register_provider(embedding_provider, set_default=True)

    provider = LanceDBProvider(
        db_path,
        base_directory=repo_root,
        embedding_manager=embedding_manager,
        config=config.database,
    )
    provider.connect()
    try:
        await index_repo(provider, repo_root, embedding_provider)

        class Services:
            def __init__(self, provider: LanceDBProvider) -> None:
                self.provider = provider

        services = Services(provider)
        scope_files, scope_chunks, scoped_set = compute_db_scope_stats(
            services, "scope"
        )
        root_files, root_chunks, _ = compute_db_scope_stats(services, "/")

        assert scope_files < root_files
        assert scope_chunks < root_chunks
        if scoped_set:
            assert all(path.startswith("scope/") for path in scoped_set)
    finally:
        provider.disconnect()

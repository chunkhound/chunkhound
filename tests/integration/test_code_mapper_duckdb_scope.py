from pathlib import Path

import pytest

from chunkhound.code_mapper.coverage import compute_db_scope_stats
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from tests.integration.code_mapper_scope_helpers import (
    build_config,
    index_repo,
    patch_code_mapper_dependencies,
    run_code_mapper,
    write_scope_repo_layout,
)


@pytest.mark.asyncio
async def test_code_mapper_duckdb_scope_real_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    write_scope_repo_layout(repo_root)

    config = build_config(repo_root, provider="duckdb")
    db_path = config.database.get_db_path()

    embedding_provider = FakeEmbeddingProvider(batch_size=100)
    provider = DuckDBProvider(db_path, base_directory=repo_root)
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
async def test_duckdb_scope_stats_sanity(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    write_scope_repo_layout(repo_root)

    config = build_config(repo_root, provider="duckdb")
    db_path = config.database.get_db_path()

    embedding_provider = FakeEmbeddingProvider(batch_size=100)
    provider = DuckDBProvider(db_path, base_directory=repo_root)
    provider.connect()
    try:
        await index_repo(provider, repo_root, embedding_provider)

        class Services:
            def __init__(self, provider: DuckDBProvider) -> None:
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


@pytest.mark.asyncio
async def test_duckdb_scope_like_escapes_metacharacters(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    scope_dir = repo_root / "scope_"
    other_dir = repo_root / "scopeX"
    scope_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    (scope_dir / "a.py").write_text("def alpha():\n    return 'a'\n", encoding="utf-8")
    (other_dir / "b.py").write_text("def beta():\n    return 'b'\n", encoding="utf-8")

    config = build_config(repo_root, provider="duckdb")
    db_path = config.database.get_db_path()

    embedding_provider = FakeEmbeddingProvider(batch_size=100)
    provider = DuckDBProvider(db_path, base_directory=repo_root)
    provider.connect()
    try:
        await index_repo(provider, repo_root, embedding_provider)

        total_files, total_chunks = provider.get_scope_stats("scope_/")
        assert total_files == 1
        assert total_chunks > 0

        scoped_paths = provider.get_scope_file_paths("scope_/")
        assert scoped_paths == ["scope_/a.py"]

        results, _ = provider.search_regex(
            pattern="def",
            page_size=50,
            offset=0,
            path_filter="scope_",
        )
        file_paths = {row["file_path"] for row in results}
        assert file_paths == {"scope_/a.py"}
    finally:
        provider.disconnect()

"""Contract and integration tests for diff chunk caching."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import chunkhound.mcp_server.tools as mcp_tools
from chunkhound.core.git_diff import (
    DIFF_CHUNK_PIPELINE_VERSION,
    resolve_diff_cache_identity,
    stream_git_diff_fragments,
)
from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import (
    ChunkType,
    FileId,
    FilePath,
    Language,
    LineNumber,
)
from chunkhound.mcp_server.tools import MAX_DIFF_CHUNK_CHARS, _inject_diff_service
from chunkhound.services.diff_chunk_cache import DiffChunkCache
from tests.integration.test_deep_research_with_diff import (
    _make_embedding_manager,
    _make_original_search_service,
    _make_services,
)


def _chunk(symbol: str, code: str = "x = 1\n", file_path: str = "a.py") -> Chunk:
    return Chunk(
        symbol=symbol,
        start_line=LineNumber(1),
        end_line=LineNumber(2),
        code=code,
        chunk_type=ChunkType.FUNCTION,
        file_id=FileId(0),
        language=Language.PYTHON,
        file_path=FilePath(file_path),
    )


def _key(
    *,
    repo: str = "/repo/.git",
    op: str = "..",
    start: str = "a" * 40,
    end: str = "b" * 40,
    max_chars: int = MAX_DIFF_CHUNK_CHARS,
    batch_size: int = 100,
    version: str = DIFF_CHUNK_PIPELINE_VERSION,
):
    return (repo, op, start, end, max_chars, batch_size, version)


def _git(cwd: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()


def _init_repo_with_commits(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")

    for i in range(1, 5):
        (repo / f"mod_{i}.py").write_text(
            f"def func_{i}():\n    return {i}\n",
            encoding="utf-8",
        )
        _git(repo, "add", f"mod_{i}.py")
        _git(repo, "commit", "-m", f"commit {i}")
    return repo


# --- DiffChunkCache unit tests ------------------------------------------------


def test_diff_chunk_cache_preserves_batch_boundaries_and_chunks() -> None:
    cache = DiffChunkCache(max_entries=4, ttl_seconds=60)
    b1 = (_chunk("f1"), _chunk("f2"))
    b2 = (_chunk("f3"),)
    key = _key()

    assert cache.get(key) is None
    assert cache.put(key, (b1, b2)) is True

    cached = cache.get(key)
    assert cached == (b1, b2)
    assert cached is not None
    assert len(cached) == 2
    assert cached[0][0] is b1[0]
    assert cached[1][0] is b2[0]


def test_diff_chunk_cache_key_isolation() -> None:
    cache = DiffChunkCache(max_entries=10, ttl_seconds=60)
    batches = ((_chunk("f1"),),)
    base = _key()
    cache.put(base, batches)

    assert cache.get(_key(repo="/other/.git")) is None
    assert cache.get(_key(op="...")) is None
    assert cache.get(_key(start="c" * 40)) is None
    assert cache.get(_key(end="d" * 40)) is None
    assert cache.get(_key(max_chars=5_000)) is None
    assert cache.get(_key(batch_size=50)) is None
    assert cache.get(_key(version="diff-chunks-v2")) is None
    assert cache.get(base) == batches


def test_diff_chunk_cache_lru_and_byte_budget_eviction() -> None:
    small_chunk = _chunk("s", code="x" * 100)
    cache_lru = DiffChunkCache(max_entries=2, ttl_seconds=60)
    k1, k2, k3 = _key(end="1" * 40), _key(end="2" * 40), _key(end="3" * 40)

    cache_lru.put(k1, ((small_chunk,),))
    cache_lru.put(k2, ((small_chunk,),))
    assert cache_lru.get(k1) is not None  # refresh k1
    cache_lru.put(k3, ((small_chunk,),))

    assert cache_lru.get(k1) is not None
    assert cache_lru.get(k2) is None
    assert cache_lru.get(k3) is not None

    big_batches = ((_chunk("big", code="y" * 2_000),),)
    # Budget large enough for 1 big_batches (~2.5KB) but not 2.
    cache_bytes = DiffChunkCache(max_entries=10, ttl_seconds=60, max_bytes=3_500)
    assert cache_bytes.put(k1, big_batches) is True
    assert cache_bytes.put(k2, big_batches) is True
    assert len(cache_bytes) == 1
    assert cache_bytes.get(k1) is None
    assert cache_bytes.get(k2) is not None
    assert cache_bytes.evictions == 1

    # Single entry exceeding max_bytes is rejected without evicting existing entry.
    huge = ((_chunk("huge", code="z" * 10_000),),)
    assert cache_bytes.put(k3, huge) is False
    assert cache_bytes.get(k2) is not None


def test_diff_chunk_cache_ttl_expiry_and_zero_disable(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(
        "chunkhound.services.diff_chunk_cache.time.monotonic", lambda: now
    )

    cache = DiffChunkCache(max_entries=4, ttl_seconds=10)
    key = _key()
    cache.put(key, ((_chunk("f1"),),))

    now = 106.0
    assert cache.get(key) is not None  # refreshes TTL
    now = 113.0
    assert cache.get(key) is not None  # 13s since insert, 7s since last access
    now = 125.0
    assert cache.get(key) is None  # expired after 12s idle

    disabled = DiffChunkCache(max_entries=0, ttl_seconds=60)
    assert disabled.put(key, ((_chunk("f1"),),)) is False
    assert disabled.get(key) is None
    assert len(disabled) == 0


# --- Integration tests with real git repo and chunk_stream choke point -------


@pytest.mark.asyncio
async def test_resolve_diff_cache_identity_and_root_commit(tmp_path: Path) -> None:
    repo = _init_repo_with_commits(tmp_path)
    root_sha = _git(repo, "rev-list", "--max-parents=0", "HEAD")

    ident_rel = await resolve_diff_cache_identity("HEAD~1..HEAD", repo)
    assert ident_rel is not None
    assert ident_rel[1] == ".."
    assert ident_rel[2] == _git(repo, "rev-parse", "HEAD~1")
    assert ident_rel[3] == _git(repo, "rev-parse", "HEAD")

    # Root commit <root>^..<root> resolves start OID to empty tree OID.
    ident_root = await resolve_diff_cache_identity(f"{root_sha}^..{root_sha}", repo)
    assert ident_root is not None
    empty_tree = _git(repo, "hash-object", "-t", "tree", "--stdin")
    assert ident_root[2] == empty_tree
    assert ident_root[3] == root_sha

    # Mutable working-tree diff must not be cached.
    assert await resolve_diff_cache_identity("HEAD~1", repo) is None


@pytest.mark.asyncio
async def test_chunk_stream_caches_once_and_invalidates_when_head_moves(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_repo_with_commits(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(mcp_tools, "_TRANSIENT_BATCH_SIZE", 2)

    stream_calls = 0

    def counting_stream(commit_range: str, cwd=None):
        nonlocal stream_calls
        stream_calls += 1
        return stream_git_diff_fragments(commit_range, cwd=cwd)

    provider = MagicMock()
    provider.get_base_directory.return_value = repo
    services = _make_services(_make_original_search_service())._replace(
        provider=provider
    )
    em = _make_embedding_manager()

    with patch("chunkhound.core.git_diff.stream_git_diff_fragments", counting_stream):
        # First call misses and runs git diff once.
        svc1 = await _inject_diff_service(services, "HEAD~3..HEAD", "diff", em)
        r1, _ = await svc1.search_service.search_semantic("func_2", page_size=10)
        assert stream_calls == 1

        # Second call on the same resolved range hits cache with zero git calls.
        svc2 = await _inject_diff_service(services, "HEAD~3..HEAD", "diff", em)
        r2, _ = await svc2.search_service.search_semantic("func_2", page_size=10)
        assert stream_calls == 1
        assert r2 == r1

        # Verify batch boundaries match fresh streaming directly.
        fresh_batches = [
            list(batch)
            async for batch in svc1.search_service._chunk_stream()  # type: ignore[attr-defined]
        ]
        assert stream_calls == 1
        assert [len(b) for b in fresh_batches] == [2, 1]

        # Move HEAD forward: same range string "HEAD~3..HEAD" must miss!
        (repo / "mod_5.py").write_text(
            "def func_5():\n    return 5\n", encoding="utf-8"
        )
        _git(repo, "add", "mod_5.py")
        _git(repo, "commit", "-m", "commit 5")

        svc3 = await _inject_diff_service(services, "HEAD~3..HEAD", "diff", em)
        r3, _ = await svc3.search_service.search_semantic("func_5", page_size=10)
        assert stream_calls == 2
        assert any("mod_5.py" in row["file_path"] for row in r3)


@pytest.mark.asyncio
async def test_incomplete_stream_not_cached_and_aclose_on_replay_is_safe(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_repo_with_commits(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(mcp_tools, "_TRANSIENT_BATCH_SIZE", 1)

    stream_calls = 0

    def counting_stream(commit_range: str, cwd=None):
        nonlocal stream_calls
        stream_calls += 1
        return stream_git_diff_fragments(commit_range, cwd=cwd)

    provider = MagicMock()
    provider.get_base_directory.return_value = repo
    services = _make_services(_make_original_search_service())._replace(
        provider=provider
    )
    em = _make_embedding_manager()

    with patch("chunkhound.core.git_diff.stream_git_diff_fragments", counting_stream):
        svc = await _inject_diff_service(services, "HEAD~2..HEAD", "diff", em)
        # Abort after first batch: must NOT cache partial results.
        partial_stream = svc.search_service._chunk_stream()  # type: ignore[attr-defined]
        first = await anext(partial_stream)
        assert len(first) == 1
        await partial_stream.aclose()
        assert stream_calls == 1
        assert len(mcp_tools._TRANSIENT_DIFF_CHUNK_CACHE) == 0

        # Full consumption populates the cache.
        full_batches = [
            batch
            async for batch in svc.search_service._chunk_stream()  # type: ignore[attr-defined]
        ]
        assert stream_calls == 2
        assert len(full_batches) == 2
        assert len(mcp_tools._TRANSIENT_DIFF_CHUNK_CACHE) == 1

        # Closing a replayed stream early does not spawn git diff or corrupt cache.
        replay = svc.search_service._chunk_stream()  # type: ignore[attr-defined]
        replayed_first = await anext(replay)
        assert replayed_first == full_batches[0]
        await replay.aclose()
        assert stream_calls == 2
        full_replay = [
            batch
            async for batch in svc.search_service._chunk_stream()  # type: ignore[attr-defined]
        ]
        assert full_replay == full_batches


@pytest.mark.asyncio
async def test_max_entries_zero_env_disables_diff_chunk_cache(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_repo_with_commits(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("CHUNKHOUND_VECTOR_CACHE_MAX_ENTRIES", "0")

    stream_calls = 0

    def counting_stream(commit_range: str, cwd=None):
        nonlocal stream_calls
        stream_calls += 1
        return stream_git_diff_fragments(commit_range, cwd=cwd)

    provider = MagicMock()
    provider.get_base_directory.return_value = repo
    services = _make_services(_make_original_search_service())._replace(
        provider=provider
    )
    em = _make_embedding_manager()

    with patch("chunkhound.core.git_diff.stream_git_diff_fragments", counting_stream):
        svc1 = await _inject_diff_service(services, "HEAD~2..HEAD", "diff", em)
        r1, _ = await svc1.search_service.search_semantic("func_3", page_size=10)
        svc2 = await _inject_diff_service(services, "HEAD~2..HEAD", "diff", em)
        r2, _ = await svc2.search_service.search_semantic("func_3", page_size=10)

    assert stream_calls == 2
    assert r1 == r2
    assert len(mcp_tools._TRANSIENT_DIFF_CHUNK_CACHE) == 0

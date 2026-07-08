"""End-to-end DB portability contract tests.

Verifies that ChunkHound stores portable DB contents:
  1. Indexing stores relative unix paths
  2. Search output remains convertible to native OS paths
  3. DuckDB contents are portable when chunks.db.root.json is removed
  4. DuckDB sidecar mismatches fail fast until removed or updated
  5. LanceDB remains portable without DuckDB sidecar semantics
"""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest

from chunkhound.providers.database.duckdb_provider import (
    DuckDBIndexedRootMismatchError,
    DuckDBProvider,
)

# Optional - used in type annotations and at runtime inside _create_provider
try:
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider
except ImportError:  # pragma: no cover - optional dependency
    LanceDBProvider = None  # type: ignore[assignment,misc]

from chunkhound.services.indexing_coordinator import IndexingCoordinator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

try:
    from tests.fixtures.fake_providers import ConstantEmbeddingProvider

    FAKES_AVAILABLE = True
except ImportError:
    FAKES_AVAILABLE = False


@pytest.fixture
def fake_embedding():
    """Constant embedding provider — all chunks get identical unit vectors."""
    if not FAKES_AVAILABLE:
        pytest.skip("tests.fixtures.fake_providers not importable")
    return ConstantEmbeddingProvider(model="test-model", dims=8)


@pytest.fixture
def constant_query_vec() -> list[float]:
    """Query vector that matches ConstantEmbeddingProvider output."""
    value = 1.0 / (8**0.5)
    return [value] * 8


@pytest.fixture
def dummy_repo(tmp_path: Path) -> Path:
    """Create a small Python codebase at tmp_path/repo."""
    repo = tmp_path / "repo"
    sub = repo / "utils"
    sub.mkdir(parents=True)
    (repo / "__init__.py").write_text("")
    (repo / "main.py").write_text(
        "def greet(name: str) -> str:\n"
        '    """Greet someone."""\n'
        '    return f"Hello, {name}!"\n'
        "\n"
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers."""\n'
        "    return a + b\n"
    )
    (sub / "__init__.py").write_text("")
    (sub / "helpers.py").write_text(
        "class Calculator:\n"
        '    """A simple calculator."""\n'
        "\n"
        "    def multiply(self, x: float, y: float) -> float:\n"
        "        return x * y\n"
    )
    return repo


def _create_provider(backend: str, db_path: Path, base_directory: Path):
    """Create and connect a provider for the given backend."""
    if backend == "duckdb":
        p = DuckDBProvider(db_path=db_path, base_directory=base_directory)
    elif backend == "lancedb":
        from chunkhound.providers.database.lancedb_provider import LanceDBProvider

        p = LanceDBProvider(db_path, base_directory=base_directory)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    p.connect()
    return p


def _remove_sidecar(backend: str, db_path: Path) -> None:
    """Remove indexed-root sidecar for DuckDB; no-op for LanceDB."""
    if backend == "duckdb":
        sidecar = db_path.with_name(db_path.name + ".root.json")
        if sidecar.exists():
            sidecar.unlink()


@pytest.fixture(params=["duckdb", "lancedb"])
def portability_backend(request: pytest.FixtureRequest) -> str:
    """Parametrized backend name for DuckDB/LanceDB portability tests.

    LanceDB tests are skipped when lancedb is not installed.
    """
    backend: str = request.param
    if backend == "lancedb":
        pytest.importorskip("lancedb")
    return backend


def _index_all(coordinator: IndexingCoordinator, repo: Path) -> dict[str, Any]:
    """Run full indexing — discover, chunk, then generate embeddings."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        stats = loop.run_until_complete(
            coordinator.process_directory(
                repo,
                patterns=["**/*.py"],
                exclude_patterns=[],
            )
        )
        # Generate embeddings when chunks exist.  generate_missing_embeddings
        # is a no-op when no embedding_provider is configured.
        if stats.get("total_chunks", 0) > 0:
            embed_stats = loop.run_until_complete(
                coordinator.generate_missing_embeddings()
            )
            stats["embeddings_generated"] = embed_stats.get("generated", 0)
        return stats
    finally:
        loop.close()


def _search_regex(
    provider: DuckDBProvider | LanceDBProvider, pattern: str = "def"
) -> list[dict[str, Any]]:
    """Run regex search and return results."""
    results, _ = provider.search_regex(
        pattern=pattern,
        page_size=100,
        offset=0,
    )
    return results


def _search_semantic(
    provider: DuckDBProvider | LanceDBProvider,
    query_vec: list[float],
    provider_name: str = "fake",
    model: str = "test-model",
) -> list[dict[str, Any]]:
    """Run semantic search and return results."""
    results, _ = provider.search_semantic(
        query_embedding=query_vec,
        provider=provider_name,
        model=model,
        page_size=100,
        offset=0,
    )
    return results


def _assert_stored_paths_are_relative_unix(results: list[dict[str, Any]]) -> None:
    """Assert all file_path values are relative unix (forward slashes, not absolute)."""
    assert results, "Expected at least one result"
    for r in results:
        fp = r.get("file_path", "")
        assert fp, "Expected non-empty file_path"
        # Root-level files ("main.py") have no slash — that's normal.
        # But if there IS a slash, it must be forward, never backslash.
        assert "\\" not in fp, f"Expected no backslashes in stored path: {fp!r}"
        assert not fp.startswith("/"), f"Expected relative path, got absolute: {fp!r}"
        assert ".." not in fp.split("/"), f"Expected no directory traversal: {fp!r}"


def _assert_native_format(results: list[dict[str, Any]]) -> None:
    """Assert file_path values convert to a valid native relative path.

    On any OS, Path(fp) must produce a relative path without crashing.
    On Windows this verifies forward-slash paths are accepted; on
    Unix this is a no-op structural check.
    """
    for r in results:
        fp = r.get("file_path", "")
        # Convert stored unix path to native OS path via pathlib
        native = Path(fp)
        # Must remain relative (no drive roots, no leading /)
        assert not native.is_absolute(), (
            f"Expected relative native path, got absolute: {native!r}"
        )
        # Must not raise — PurePath(fp) parses on all platforms
        PurePosixPath(fp)  # forward-slash parse must succeed
        PureWindowsPath(fp)  # Windows parse must not raise
        # On Windows, Path(fp) must not produce a drive-absolute path
        assert not str(native).startswith("\\"), (
            f"Expected relative path, got UNC: {native!r}"
        )


def _paths_from(results: list[dict[str, Any]]) -> list[str]:
    """Extract sorted file_path values from search results."""
    return sorted({r.get("file_path", "") for r in results})


def _repo_python_paths(repo: Path) -> list[str]:
    """Return repo Python file paths in stored-path format."""
    return sorted(path.relative_to(repo).as_posix() for path in repo.rglob("*.py"))


def _raw_stored_paths(provider: DuckDBProvider | LanceDBProvider) -> list[str]:
    """Return raw file paths exactly as stored in the backing database."""
    rows = provider.execute_query("SELECT path FROM files")
    paths: list[str] = []
    for row in rows:
        path = row.get("path")
        assert path is not None, f"Row missing 'path' key: {row}"
        paths.append(str(path))
    return sorted(paths)


def _assert_stored_paths_match(
    provider: DuckDBProvider | LanceDBProvider, expected_paths: list[str]
) -> None:
    """Assert raw stored file paths match expectations exactly."""
    stored_paths = _raw_stored_paths(provider)
    assert stored_paths == sorted(expected_paths), (
        f"Stored paths mismatch. Expected {sorted(expected_paths)}, got {stored_paths}"
    )
    for p in stored_paths:
        assert "\\" not in p, f"Expected no backslash in stored path: {p!r}"
        assert not p.startswith("/"), (
            f"Expected relative stored path, got absolute: {p!r}"
        )
        assert ".." not in p.split("/"), (
            f"Expected no directory traversal in stored path: {p!r}"
        )


def _verify_same_results(
    results_a: list[dict[str, Any]], results_b: list[dict[str, Any]]
) -> None:
    """Verify two result sets have the same number of results and paths."""
    assert len(results_a) == len(results_b), (
        f"Result count mismatch: {len(results_a)} vs {len(results_b)}"
    )
    assert _paths_from(results_a) == _paths_from(results_b), "Paths differ between runs"


def _move_repo_and_db(
    tmp_path: Path, repo: Path, db_dir: Path, db_name: str
) -> tuple[Path, Path]:
    """Copy the repo + DB to a new location and return their moved paths."""
    new_repo = tmp_path / "new_repo"
    new_db_dir = tmp_path / "new_db"
    shutil.copytree(repo, new_repo)
    shutil.copytree(db_dir, new_db_dir)
    return new_repo, new_db_dir / db_name


# ===========================================================================
# DB portability: stored path format
# ===========================================================================


@pytest.mark.integration
def test_path_format_contract(
    tmp_path: Path,
    dummy_repo: Path,
    fake_embedding: Any,
    constant_query_vec: list[float],
    portability_backend: str,
) -> None:
    """Paths stored as relative unix, native on output.

    Phase 1 — Index at original location.
    Phase 2 — Verify stored paths are relative unix (forward slashes).
    Phase 3 — Regex search returns relative unix, native-convertible paths.
    Phase 4 — Semantic search returns relative unix, native-convertible paths.
    """
    backend = portability_backend
    ext = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    db_path = tmp_path / "db" / ext
    db_path.parent.mkdir(parents=True)

    provider = _create_provider(backend, db_path, dummy_repo)

    try:
        # Phase 1: Index
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=dummy_repo,
            embedding_provider=fake_embedding,
        )
        stats = _index_all(coordinator, dummy_repo)
        assert stats.get("files_processed", 0) > 0, "Expected files to be indexed"
        assert stats.get("total_chunks", 0) > 0, "Expected chunks to be created"

        # Phase 2: Verify stored paths are relative unix
        expected_paths = _repo_python_paths(dummy_repo)
        _assert_stored_paths_match(provider, expected_paths)

        # Phase 3: Regex search — verify paths
        regex_results = _search_regex(provider, "def")
        _assert_stored_paths_are_relative_unix(regex_results)
        _assert_native_format(regex_results)

        # Phase 4: Semantic search — verify paths
        semantic_results = _search_semantic(provider, constant_query_vec)
        _assert_stored_paths_are_relative_unix(semantic_results)
        _assert_native_format(semantic_results)

    finally:
        provider.disconnect()


@pytest.mark.integration
def test_portable_db_remains_searchable_immediately_after_move(
    tmp_path: Path,
    dummy_repo: Path,
    fake_embedding: Any,
    constant_query_vec: list[float],
    portability_backend: str,
) -> None:
    """Moved DB stays searchable before any repair step like re-indexing.

    Phase 1 — Index at original location.
    Phase 2 — Run baseline queries.
    Phase 3 — Close DB, move DB + codebase.
    Phase 4 — Delete indexed-root sidecar (DuckDB) or no-op (LanceDB).
    Phase 5 — Re-open at new location without re-indexing.
    Phase 6 — Run same queries, verify identical results.
    """
    backend = portability_backend
    db_ext = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"

    db_path = tmp_path / "db" / db_ext
    db_path.parent.mkdir(parents=True)
    provider = _create_provider(backend, db_path, dummy_repo)

    try:
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=dummy_repo,
            embedding_provider=fake_embedding,
        )
        _index_all(coordinator, dummy_repo)

        baseline_regex = _search_regex(provider, "def")
        baseline_semantic = _search_semantic(provider, constant_query_vec)
        baseline_stored_paths = _raw_stored_paths(provider)
        assert len(baseline_regex) > 0, "Expected regex results"
        assert len(baseline_semantic) > 0, "Expected semantic results"
        expected_paths = _repo_python_paths(dummy_repo)
    finally:
        provider.disconnect()

    new_repo, new_db_path = _move_repo_and_db(
        tmp_path, dummy_repo, db_path.parent, db_ext
    )
    _remove_sidecar(backend, new_db_path)

    moved_provider = _create_provider(backend, new_db_path, new_repo)
    try:
        moved_expected_paths = _repo_python_paths(new_repo)
        assert moved_expected_paths == expected_paths
        _assert_stored_paths_match(moved_provider, moved_expected_paths)
        assert _raw_stored_paths(moved_provider) == baseline_stored_paths

        moved_regex = _search_regex(moved_provider, "def")
        _verify_same_results(baseline_regex, moved_regex)
        _assert_stored_paths_are_relative_unix(moved_regex)
        _assert_native_format(moved_regex)

        moved_semantic = _search_semantic(moved_provider, constant_query_vec)
        _verify_same_results(baseline_semantic, moved_semantic)
        _assert_stored_paths_are_relative_unix(moved_semantic)
        _assert_native_format(moved_semantic)
    finally:
        moved_provider.disconnect()


@pytest.mark.integration
def test_reindex_after_move_is_idempotent(
    tmp_path: Path,
    dummy_repo: Path,
    fake_embedding: Any,
    constant_query_vec: list[float],
    portability_backend: str,
) -> None:
    """Re-indexing after a portable move must preserve search results.

    This stays separate from the move-portability contract so re-indexing cannot
    mask a moved-DB regression before the first post-move query.

    Phase 1 — Index at original location.
    Phase 2 — Run baseline queries.
    Phase 3 — Close DB, move DB + codebase.
    Phase 4 — Delete indexed-root sidecar (DuckDB) or no-op (LanceDB).
    Phase 5 — Re-open at new location, verify pre-reindex baseline.
    Phase 6 — Re-index at new location.
    Phase 7 — Run same queries, verify idempotent results.
    """
    backend = portability_backend
    db_ext = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"

    db_path = tmp_path / "db" / db_ext
    db_path.parent.mkdir(parents=True)
    provider = _create_provider(backend, db_path, dummy_repo)

    try:
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=dummy_repo,
            embedding_provider=fake_embedding,
        )
        _index_all(coordinator, dummy_repo)

        baseline_regex = _search_regex(provider, "def")
        baseline_semantic = _search_semantic(provider, constant_query_vec)
        baseline_stored_paths = _raw_stored_paths(provider)
        assert len(baseline_regex) > 0, "Expected regex results"
        assert len(baseline_semantic) > 0, "Expected semantic results"
        expected_paths = _repo_python_paths(dummy_repo)
    finally:
        provider.disconnect()

    new_repo, new_db_path = _move_repo_and_db(
        tmp_path, dummy_repo, db_path.parent, db_ext
    )
    _remove_sidecar(backend, new_db_path)

    moved_provider = _create_provider(backend, new_db_path, new_repo)
    try:
        # Verify the moved DB works before re-indexing so the idempotency check
        # starts from a known-portable baseline rather than repairing first.
        pre_reindex_regex = _search_regex(moved_provider, "def")
        pre_reindex_semantic = _search_semantic(moved_provider, constant_query_vec)
        _verify_same_results(baseline_regex, pre_reindex_regex)
        _verify_same_results(baseline_semantic, pre_reindex_semantic)

        coordinator2 = IndexingCoordinator(
            database_provider=moved_provider,
            base_directory=new_repo,
            embedding_provider=fake_embedding,
        )
        _index_all(coordinator2, new_repo)

        moved_expected_paths = _repo_python_paths(new_repo)
        assert moved_expected_paths == expected_paths
        _assert_stored_paths_match(moved_provider, moved_expected_paths)
        assert _raw_stored_paths(moved_provider) == baseline_stored_paths

        moved_regex = _search_regex(moved_provider, "def")
        _verify_same_results(baseline_regex, moved_regex)
        _assert_stored_paths_are_relative_unix(moved_regex)
        _assert_native_format(moved_regex)

        moved_semantic = _search_semantic(moved_provider, constant_query_vec)
        _verify_same_results(baseline_semantic, moved_semantic)
        _assert_stored_paths_are_relative_unix(moved_semantic)
        _assert_native_format(moved_semantic)
    finally:
        moved_provider.disconnect()


# ===========================================================================
# DuckDB sidecar portability guard
# ===========================================================================


@pytest.mark.integration
def test_duckdb_move_with_sidecar_mismatch_fails(
    tmp_path: Path,
    dummy_repo: Path,
    fake_embedding: Any,
) -> None:
    """DuckDB: moving a DB without removing/updating the sidecar must fail."""
    db_file = tmp_path / "db" / "chunks.db"
    db_file.parent.mkdir(parents=True)
    provider = DuckDBProvider(db_path=db_file, base_directory=dummy_repo)
    provider.connect()

    try:
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=dummy_repo,
            embedding_provider=fake_embedding,
        )
        stats = _index_all(coordinator, dummy_repo)
        assert stats.get("files_processed", 0) > 0, "Expected files to be indexed"
    finally:
        provider.disconnect()

    new_repo = tmp_path / "new_repo"
    new_db_dir = tmp_path / "new_db"
    shutil.copytree(dummy_repo, new_repo)
    shutil.copytree(db_file.parent, new_db_dir)
    new_db_file = new_db_dir / "chunks.db"

    provider2 = DuckDBProvider(db_path=new_db_file, base_directory=new_repo)
    with pytest.raises(DuckDBIndexedRootMismatchError):
        provider2.connect()


# LanceDB portability is covered by parametrized tests above
# (test_path_format_contract and test_portability_after_move).


# ===========================================================================
# Case sensitivity contract test
# ===========================================================================


@pytest.mark.integration
def test_case_sensitivity_contract(
    tmp_path: Path,
    fake_embedding: Any,
) -> None:
    """Case handling is consistent with the host filesystem.

    On case-insensitive FS (macOS/Windows): File.py and file.py are same file
    On case-sensitive FS (Linux): they are different files

    The contract: whatever the FS allows, the DB stores correctly.
    """
    # Use names that differ only by case so the test follows host FS rules.
    repo = tmp_path / "repo_case"
    repo.mkdir()
    upper = repo / "File.py"
    lower = repo / "file.py"
    upper.write_text(
        'def uppercase_start():\n    """Function in uppercase file."""\n    return 1\n'
    )
    lower.write_text(
        'def lowercase_start():\n    """Function in lowercase file."""\n    return 2\n'
    )
    fs_visible_paths = sorted(p.name for p in repo.glob("*.py"))
    assert fs_visible_paths in (["File.py"], ["File.py", "file.py"], ["file.py"])

    db_file = tmp_path / "db" / "chunks_case.db"
    db_file.parent.mkdir(parents=True)
    provider = DuckDBProvider(db_path=db_file, base_directory=repo)
    provider.connect()

    try:
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=repo,
            embedding_provider=fake_embedding,
        )
        _index_all(coordinator, repo)

        # The DB must reflect exactly what the host filesystem exposed.
        _assert_stored_paths_match(provider, fs_visible_paths)

        results, _ = provider.search_regex(pattern="def", page_size=100, offset=0)
        assert sorted({r["file_path"] for r in results}) == fs_visible_paths

    finally:
        provider.disconnect()


# ===========================================================================
# Unicode path contract test
# ===========================================================================


@pytest.mark.integration
def test_unicode_path_contract(
    tmp_path: Path,
    fake_embedding: Any,
) -> None:
    """Unicode file paths are stored and returned correctly."""
    repo = tmp_path / "repo_unicode"
    repo.mkdir()
    (repo / "données.py").write_text(
        'def analyser():\n    """Analyse des données."""\n    return "données"\n'
    )
    (repo / "engine.py").write_text(
        'def run():\n    """Run the engine."""\n    return "engine"\n'
    )

    db_file = tmp_path / "db" / "chunks_unicode.db"
    db_file.parent.mkdir(parents=True)
    provider = DuckDBProvider(db_path=db_file, base_directory=repo)
    provider.connect()

    try:
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=repo,
            embedding_provider=fake_embedding,
        )
        _index_all(coordinator, repo)

        # Verify unicode path is stored
        expected_paths = _repo_python_paths(repo)
        _assert_stored_paths_match(provider, expected_paths)
        assert any("données" in p for p in expected_paths), (
            f"Unicode path not found in repo paths: {expected_paths}"
        )

        # Verify search works with unicode files
        results, _ = provider.search_regex(pattern="def", page_size=100, offset=0)
        assert len(results) >= 2

        # All paths must be forward-slash relative
        for r in results:
            assert "\\" not in r["file_path"]
            assert not r["file_path"].startswith("/")

    finally:
        provider.disconnect()


# ===========================================================================
# Deeply nested path contract test
# ===========================================================================


@pytest.mark.integration
def test_deeply_nested_path_contract(
    tmp_path: Path,
    fake_embedding: Any,
) -> None:
    """Deeply nested paths are stored and returned correctly."""
    repo = tmp_path / "repo_deep"
    deep = repo / "src" / "lib" / "core" / "utils" / "helpers"
    deep.mkdir(parents=True)
    (repo / "src").mkdir(exist_ok=True)
    (deep / "__init__.py").write_text("")
    (deep / "assistants.py").write_text(
        'def deep_function():\n    """A deeply nested function."""\n    return 42\n'
    )

    db_file = tmp_path / "db" / "chunks_deep.db"
    db_file.parent.mkdir(parents=True)
    provider = DuckDBProvider(db_path=db_file, base_directory=repo)
    provider.connect()

    try:
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=repo,
            embedding_provider=fake_embedding,
        )
        _index_all(coordinator, repo)

        # Verify deep path is stored correctly
        expected_paths = _repo_python_paths(repo)
        _assert_stored_paths_match(provider, expected_paths)
        deep_paths = [p for p in expected_paths if "assistants" in p]
        assert len(deep_paths) == 1
        assert deep_paths[0] == "src/lib/core/utils/helpers/assistants.py"

        # Verify search works
        results, _ = provider.search_regex(pattern="def", page_size=100, offset=0)
        assert len(results) >= 1
        for r in results:
            assert "\\" not in r["file_path"]
            assert not r["file_path"].startswith("/")

    finally:
        provider.disconnect()

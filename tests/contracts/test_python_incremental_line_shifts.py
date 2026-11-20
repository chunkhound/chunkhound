"""Contract coverage for Python incremental indexing after line shifts."""

from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pytest

from tests.contracts.mock_embed import MOCK_MODEL, MOCK_PROVIDER, MockEmbeddingProvider
from tests.contracts.pipeline_harness import (
    assert_chunk_multiset_identical,
    assert_embedding_multiset_identical,
    disconnect_registry_db,
    index_with_python,
)

pytestmark = pytest.mark.integration

BASE_TS = """export const headerG = "G";
export function fnOne(a: number): number {
  return a + 1;
}
export function fnTwo(a: number): number {
  return a + 2;
}
export function fnThree(a: number): number {
  return a + 3;
}
export const footerG = 0;
"""
TOP_INSERT = "// insert 1\n// insert 2\n// insert 3\n"
STABLE_TS = 'export const stable = "unchanged";\n'


@pytest.fixture(autouse=True)
def isolate_environment(clean_environment, monkeypatch, tmp_path: Path):
    """Prevent global configuration from creating a real embedding provider."""
    home = tmp_path / "home"
    config = tmp_path / "xdg-config"
    cache = tmp_path / "xdg-cache"
    for directory in (home, config, cache):
        directory.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    yield
    disconnect_registry_db()


def _write_project(project_dir: Path, g_contents: str) -> None:
    project_dir.mkdir()
    (project_dir / "G.ts").write_text(g_contents)
    (project_dir / "stable.ts").write_text(STABLE_TS)


def _payloads(chunks: list[tuple]) -> Counter:
    return Counter(chunk[:4] for chunk in chunks)


def _ranges_by_payload(chunks: list[tuple]) -> dict[tuple, set[tuple[int, int]]]:
    ranges: dict[tuple, set[tuple[int, int]]] = defaultdict(set)
    for path, chunk_type, symbol, code, start_line, end_line in chunks:
        ranges[(path, chunk_type, symbol, code)].add((start_line, end_line))
    return ranges


def _assert_setup_guards(initial, fresh) -> None:
    assert initial.errors == []
    assert fresh.errors == []
    assert initial.chunk_tuples
    assert fresh.chunk_tuples

    initial_ranges = _ranges_by_payload(initial.chunk_tuples)
    fresh_ranges = _ranges_by_payload(fresh.chunk_tuples)
    assert any(
        initial_ranges[payload] != fresh_ranges[payload]
        for payload in initial_ranges.keys() & fresh_ranges.keys()
    ), "fixture must retain a payload at a different line range"

    initial_payloads = _payloads(initial.chunk_tuples)
    fresh_payloads = _payloads(fresh.chunk_tuples)
    assert initial_payloads - fresh_payloads, "fixture must remove a payload"
    assert fresh_payloads - initial_payloads, "fixture must add a payload"

    initial_stable = [
        chunk for chunk in initial.chunk_tuples if chunk[0] == "stable.ts"
    ]
    fresh_stable = [chunk for chunk in fresh.chunk_tuples if chunk[0] == "stable.ts"]
    assert initial_stable
    assert initial_stable == fresh_stable


def _assert_mock_embedding_associations(db_file: Path) -> None:
    """Every live chunk has one mock embedding and no embedding is orphaned."""
    conn = duckdb.connect(str(db_file))
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
        assert tables
        quoted_tables = [
            f'"{name.replace(chr(34), chr(34) * 2)}"' for (name,) in tables
        ]
        all_embeddings = " UNION ALL ".join(
            f"SELECT chunk_id, provider, model FROM {table}" for table in quoted_tables
        )
        association_counts = conn.execute(
            f"""
            SELECT c.id, COUNT(e.chunk_id)
            FROM chunks c
            LEFT JOIN ({all_embeddings}) e
              ON e.chunk_id = c.id AND e.provider = ? AND e.model = ?
            GROUP BY c.id
            HAVING COUNT(e.chunk_id) != 1
            """,
            [MOCK_PROVIDER, MOCK_MODEL],
        ).fetchall()
        assert association_counts == []
        orphans = conn.execute(
            f"""
            SELECT e.chunk_id
            FROM ({all_embeddings}) e
            LEFT JOIN chunks c ON c.id = e.chunk_id
            WHERE c.id IS NULL
            """
        ).fetchall()
        assert orphans == []
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_contents", "final_contents"),
    [
        pytest.param(BASE_TS, TOP_INSERT + BASE_TS, id="insert-above"),
        pytest.param(TOP_INSERT + BASE_TS, BASE_TS, id="delete-above"),
    ],
)
async def test_incremental_shift_final_state_matches_fresh_reindex(
    tmp_path: Path, initial_contents: str, final_contents: str
) -> None:
    project = tmp_path / "project"
    _write_project(project, initial_contents)
    incremental_db = tmp_path / "db_incremental"
    fresh_db = tmp_path / "db_fresh"

    initial = await index_with_python(project, incremental_db, skip_embeddings=True)
    (project / "G.ts").write_text(final_contents)
    fresh = await index_with_python(project, fresh_db, skip_embeddings=True)
    _assert_setup_guards(initial, fresh)

    incremental = await index_with_python(project, incremental_db, skip_embeddings=True)
    assert incremental.errors == []
    assert_chunk_multiset_identical(
        incremental.chunk_tuples,
        fresh.chunk_tuples,
        label_a="incremental",
        label_b="fresh",
    )

    rerun = await index_with_python(project, incremental_db, skip_embeddings=True)
    assert rerun.errors == []
    assert_chunk_multiset_identical(
        rerun.chunk_tuples,
        fresh.chunk_tuples,
        label_a="incremental rerun",
        label_b="fresh",
    )


@pytest.mark.asyncio
async def test_incremental_shift_embedding_final_state_matches_fresh_reindex(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_project(project, BASE_TS)
    incremental_db = tmp_path / "db_incremental"
    fresh_db = tmp_path / "db_fresh"
    provider = MockEmbeddingProvider()

    initial = await index_with_python(
        project, incremental_db, skip_embeddings=False, embedding_provider=provider
    )
    (project / "G.ts").write_text(TOP_INSERT + BASE_TS)
    fresh = await index_with_python(
        project, fresh_db, skip_embeddings=False, embedding_provider=provider
    )
    _assert_setup_guards(initial, fresh)

    incremental = await index_with_python(
        project, incremental_db, skip_embeddings=False, embedding_provider=provider
    )
    assert incremental.errors == []
    assert_chunk_multiset_identical(
        incremental.chunk_tuples,
        fresh.chunk_tuples,
        label_a="incremental",
        label_b="fresh",
    )
    assert_embedding_multiset_identical(
        incremental.embedding_tuples,
        fresh.embedding_tuples,
        label_a="incremental",
        label_b="fresh",
    )
    disconnect_registry_db()
    _assert_mock_embedding_associations(incremental_db / "chunks.db")


@pytest.mark.asyncio
async def test_force_reindex_after_shift_matches_fresh_reindex(tmp_path: Path) -> None:
    """Cover forced indexing final state, not proof that force wiring was used."""
    project = tmp_path / "project"
    _write_project(project, BASE_TS)
    incremental_db = tmp_path / "incremental" / ".chhound.db"
    incremental_db.parent.mkdir()
    fresh_db = tmp_path / "db_fresh"

    initial = await index_with_python(
        project, incremental_db.parent, skip_embeddings=True, db_file=incremental_db
    )
    (project / "G.ts").write_text(TOP_INSERT + BASE_TS)
    fresh = await index_with_python(project, fresh_db, skip_embeddings=True)
    _assert_setup_guards(initial, fresh)

    incremental = await index_with_python(
        project,
        incremental_db.parent,
        skip_embeddings=True,
        force_reindex=True,
        db_file=incremental_db,
    )
    assert incremental.errors == []
    assert_chunk_multiset_identical(
        incremental.chunk_tuples,
        fresh.chunk_tuples,
        label_a="forced incremental",
        label_b="fresh",
    )

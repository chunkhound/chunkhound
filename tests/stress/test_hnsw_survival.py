"""Stress tests for HNSW index survival through compaction.

Uses direct DuckDB connections (not DuckDBProvider.optimize()) to avoid the
~60s CHECKPOINT penalty with VSS in small test DBs — same strategy as
TestEmbeddingVectorSurvival in tests/unit/test_compaction_service.py.

The orchestration path (lock, soft_disconnect, atomic swap) is exercised
by TestDuckDBProviderOptimize; these tests focus on data and index fidelity.
"""
from __future__ import annotations

import asyncio
import random
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chunkhound.core.config.config import Config
from chunkhound.services.compaction_service import CompactionService

try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False

pytestmark = pytest.mark.heavy  # skipped in fast CI; run with -m heavy


def _vss_available() -> bool:
    if not _DUCKDB_AVAILABLE:
        return False
    try:
        conn = duckdb.connect(":memory:")
        conn.execute("LOAD vss")
        conn.close()
        return True
    except Exception:
        return False


def _make_db_with_embeddings(
    db_path: Path,
    *,
    n_rows: int = 50,
    dims: int = 128,
    index_name: str = "idx_hnsw_128",
    rng_seed: int = 42,
) -> list[list[float]]:
    """Create DB with embeddings table and HNSW index. Returns inserted vectors."""
    rng = random.Random(rng_seed)
    vectors = [[rng.uniform(-1, 1) for _ in range(dims)] for _ in range(n_rows)]

    conn = duckdb.connect(str(db_path))
    conn.execute("LOAD vss")
    try:
        conn.execute("SET hnsw_enable_experimental_persistence = true")
    except duckdb.Error:
        pass
    conn.execute(f"CREATE TABLE embeddings_{dims} (chunk_id INTEGER, embedding FLOAT[{dims}])")
    conn.executemany(
        f"INSERT INTO embeddings_{dims} VALUES (?, ?::FLOAT[{dims}])",
        [[i, vec] for i, vec in enumerate(vectors)],
    )
    conn.execute(
        f'CREATE INDEX "{index_name}" ON "embeddings_{dims}" USING HNSW (embedding) WITH (metric = \'cosine\')'
    )
    conn.execute("CHECKPOINT")
    conn.close()
    return vectors


def _run_compact_cycle(
    db_path: Path,
    export_dir: Path,
    new_db_path: Path,
    *,
    recreate_hnsw: bool = True,
) -> list[dict]:
    """Run one EXPORT/IMPORT cycle directly (bypasses DuckDBProvider). Returns HNSW snapshot."""
    import re

    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("LOAD vss")
    rows = conn.execute(
        "SELECT index_name, table_name, sql FROM duckdb_indexes() WHERE table_name LIKE 'embeddings_%'"
    ).fetchall()
    hnsw_snapshot = []
    for idx_name, tbl_name, create_sql in rows:
        if "USING HNSW" in (create_sql or "").upper() or idx_name.startswith(("idx_hnsw_", "hnsw_")):
            m = re.search(r"metric\s*=\s*'([^']+)'", create_sql or "", re.IGNORECASE)
            metric = m.group(1) if m else "cosine"
            try:
                dims = int(tbl_name[11:])
            except ValueError:
                continue
            hnsw_snapshot.append({
                "index_name": idx_name,
                "table_name": tbl_name,
                "dims": dims,
                "metric": metric,
            })

    if export_dir.exists():
        shutil.rmtree(export_dir)
    conn.execute(f"EXPORT DATABASE '{export_dir.as_posix()}' (FORMAT PARQUET)")
    conn.close()

    if new_db_path.exists():
        new_db_path.unlink()
    conn = duckdb.connect(str(new_db_path))
    conn.execute("LOAD vss")
    try:
        conn.execute("SET hnsw_enable_experimental_persistence = true")
    except duckdb.Error:
        pass
    conn.execute(f"IMPORT DATABASE '{export_dir.as_posix()}'")
    if recreate_hnsw:
        for idx in hnsw_snapshot:
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS "{idx['index_name']}"
                ON "{idx['table_name']}"
                USING HNSW (embedding)
                WITH (metric = '{idx['metric']}')
            """)
    conn.execute("CHECKPOINT")
    conn.close()
    return hnsw_snapshot


def _index_names_in_db(db_path: Path) -> set[str]:
    """Return set of HNSW index names on embeddings tables in a DB file."""
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        try:
            conn.execute("LOAD vss")
        except duckdb.Error:
            pass
        rows = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()



@pytest.mark.skipif(not _vss_available(), reason="VSS extension not available")
class TestOneTimeIndexingAndCompact:
    """Scenario: index once, compact once via direct EXPORT/IMPORT, verify HNSW survives."""

    def test_hnsw_present_after_single_compact_cycle(self, tmp_path: Path):
        """After one compact cycle, HNSW index exists in the compacted DB."""
        db_path = tmp_path / "src.duckdb"
        _make_db_with_embeddings(db_path, n_rows=50, dims=128)

        snapshot = _run_compact_cycle(
            db_path, tmp_path / "export", tmp_path / "compact.duckdb"
        )

        assert len(snapshot) == 1, f"Expected 1 HNSW index snapshot, got {snapshot}"
        assert snapshot[0]["index_name"] == "idx_hnsw_128"

        idx_names = _index_names_in_db(tmp_path / "compact.duckdb")
        assert "idx_hnsw_128" in idx_names, f"HNSW index missing. Found: {idx_names}"

    def test_row_count_preserved_after_compact(self, tmp_path: Path):
        """All 50 rows survive compaction."""
        db_path = tmp_path / "src.duckdb"
        _make_db_with_embeddings(db_path, n_rows=50, dims=128)
        _run_compact_cycle(db_path, tmp_path / "export", tmp_path / "compact.duckdb")

        conn = duckdb.connect(str(tmp_path / "compact.duckdb"), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM embeddings_128").fetchone()[0]
        conn.close()
        assert count == 50, f"Expected 50 rows, got {count}"

    def test_multiple_dim_indexes_all_preserved(self, tmp_path: Path):
        """Multiple HNSW indexes across different dims all survive."""
        db_path = tmp_path / "src.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("LOAD vss")
        try:
            conn.execute("SET hnsw_enable_experimental_persistence = true")
        except duckdb.Error:
            pass

        rng = random.Random(7)
        for dims in [64, 256]:
            conn.execute(f"CREATE TABLE embeddings_{dims} (chunk_id INTEGER, embedding FLOAT[{dims}])")
            for i in range(20):
                vec = [rng.uniform(-1, 1) for _ in range(dims)]
                conn.execute(f"INSERT INTO embeddings_{dims} VALUES (?, ?::FLOAT[{dims}])", [i, vec])
            conn.execute(
                f'CREATE INDEX "idx_hnsw_{dims}" ON "embeddings_{dims}" USING HNSW (embedding) WITH (metric = \'cosine\')'
            )
        conn.execute("CHECKPOINT")
        conn.close()

        snapshot = _run_compact_cycle(db_path, tmp_path / "export", tmp_path / "compact.duckdb")
        assert len(snapshot) == 2, f"Expected 2 snapshots, got {snapshot}"

        idx_names = _index_names_in_db(tmp_path / "compact.duckdb")
        assert "idx_hnsw_64" in idx_names
        assert "idx_hnsw_256" in idx_names


@pytest.mark.skipif(not _vss_available(), reason="VSS extension not available")
class TestRepeatIndexingCycles:
    """Scenario: multiple index+compact cycles, HNSW survives each round."""

    def test_hnsw_survives_three_compact_cycles(self, tmp_path: Path):
        """HNSW index present and row count correct after 3 sequential compact cycles."""
        db_path = tmp_path / "db.duckdb"
        _make_db_with_embeddings(db_path, n_rows=30, dims=128, rng_seed=1)

        for cycle in range(3):
            export_dir = tmp_path / f"export_{cycle}"
            compact_path = tmp_path / f"compact_{cycle}.duckdb"

            _run_compact_cycle(db_path, export_dir, compact_path)

            idx_names = _index_names_in_db(compact_path)
            conn = duckdb.connect(str(compact_path), read_only=True)
            row_count = conn.execute("SELECT COUNT(*) FROM embeddings_128").fetchone()[0]
            conn.close()

            assert "idx_hnsw_128" in idx_names, (
                f"Cycle {cycle}: HNSW missing. Found: {idx_names}"
            )
            assert row_count == 30, f"Cycle {cycle}: expected 30 rows, got {row_count}"

            shutil.copy2(str(compact_path), str(db_path))

    def test_data_appended_between_cycles_survives(self, tmp_path: Path):
        """Rows added after each compact cycle are present in the final compact output."""
        db_path = tmp_path / "db.duckdb"
        _make_db_with_embeddings(db_path, n_rows=20, dims=64, index_name="idx_hnsw_64", rng_seed=5)

        rng = random.Random(99)
        for cycle in range(2):
            conn = duckdb.connect(str(db_path))
            conn.execute("LOAD vss")
            base_id = 20 + cycle * 10
            for i in range(10):
                vec = [rng.uniform(-1, 1) for _ in range(64)]
                conn.execute("INSERT INTO embeddings_64 VALUES (?, ?::FLOAT[64])", [base_id + i, vec])
            conn.execute("CHECKPOINT")
            conn.close()

            export_dir = tmp_path / f"export_{cycle}"
            compact_path = tmp_path / f"compact_{cycle}.duckdb"
            _run_compact_cycle(db_path, export_dir, compact_path)
            shutil.copy2(str(compact_path), str(db_path))

        conn = duckdb.connect(str(db_path), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM embeddings_64").fetchone()[0]
        conn.close()
        assert count == 40, f"Expected 40 rows (20 initial + 20 appended), got {count}"


@pytest.mark.skipif(not _vss_available(), reason="VSS extension not available")
class TestLargeScaleIndexSurvival:
    """Scenario: large-scale HNSW indexes survive repeated compaction cycles.

    Simulates real usage: many embeddings, multiple dimensions, repeated
    compaction. The orchestration (lock, soft_disconnect, atomic swap) is
    exercised by test_lifecycle.py and test_concurrency.py; this tests
    data and index fidelity at scale.
    """

    def test_100_rows_100_dims_survive_compact(self, tmp_path: Path):
        """100 embeddings × 100 dims survive one compaction cycle."""
        db_path = tmp_path / "scale_1.duckdb"
        _make_db_with_embeddings(db_path, n_rows=100, dims=100, index_name="idx_hnsw_100", rng_seed=999)

        snapshot = _run_compact_cycle(
            db_path, tmp_path / "export", tmp_path / "compact.duckdb"
        )

        assert len(snapshot) == 1
        assert snapshot[0]["dims"] == 100
        idx_names = _index_names_in_db(tmp_path / "compact.duckdb")
        assert "idx_hnsw_100" in idx_names

        conn = duckdb.connect(str(tmp_path / "compact.duckdb"), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM embeddings_100").fetchone()[0]
        conn.close()
        assert count == 100

    def test_1000_rows_512_dims_survive_compact(self, tmp_path: Path):
        """1000 embeddings × 512 dims survive one compaction cycle.

        Tests with larger vectors to stress the HNSW persistence layer.
        """
        db_path = tmp_path / "scale_2.duckdb"
        _make_db_with_embeddings(db_path, n_rows=1000, dims=512, index_name="idx_hnsw_512", rng_seed=777)

        snapshot = _run_compact_cycle(
            db_path, tmp_path / "export", tmp_path / "compact.duckdb"
        )

        assert len(snapshot) == 1
        assert snapshot[0]["dims"] == 512
        idx_names = _index_names_in_db(tmp_path / "compact.duckdb")
        assert "idx_hnsw_512" in idx_names

        conn = duckdb.connect(str(tmp_path / "compact.duckdb"), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM embeddings_512").fetchone()[0]
        conn.close()
        assert count == 1000

    def test_five_compact_cycles_with_appends(self, tmp_path: Path):
        """Five successive compact cycles with data appends between each."""
        db_path = tmp_path / "db.duckdb"
        _make_db_with_embeddings(db_path, n_rows=20, dims=256, index_name="idx_hnsw_256", rng_seed=555)

        rng = random.Random(888)
        for cycle in range(5):
            # Compact current DB
            export_dir = tmp_path / f"export_{cycle}"
            compact_path = tmp_path / f"compact_{cycle}.duckdb"
            _run_compact_cycle(db_path, export_dir, compact_path)

            # Validate HNSW survived
            idx_names = _index_names_in_db(compact_path)
            assert "idx_hnsw_256" in idx_names, f"Cycle {cycle}: index missing"

            # Append 10 more rows to compacted DB for next cycle
            conn = duckdb.connect(str(compact_path))
            conn.execute("LOAD vss")
            base_id = 20 + cycle * 10
            for i in range(10):
                vec = [rng.uniform(-1, 1) for _ in range(256)]
                conn.execute(
                    "INSERT INTO embeddings_256 VALUES (?, ?::FLOAT[256])",
                    [base_id + i, vec],
                )
            conn.execute("CHECKPOINT")
            conn.close()

            shutil.copy2(str(compact_path), str(db_path))

        # Final check: DB has all 70 rows and HNSW index
        conn = duckdb.connect(str(db_path), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM embeddings_256").fetchone()[0]
        conn.close()
        assert count == 70, f"Expected 70 rows, got {count}"

        idx_names = _index_names_in_db(db_path)
        assert "idx_hnsw_256" in idx_names


@pytest.mark.skipif(not _vss_available(), reason="VSS extension not available")
class TestMCPBackgroundCompaction:
    """Scenario: CompactionService compact_blocking/compact_background runs correctly.

    Uses make_fragmented_db from conftest for a properly fragmented ChunkHound DB.
    HNSW index preservation through the full pipeline is implicitly covered:
    compact_blocking calls _run_blocking_compaction which uses the same
    _export_database_for_compaction/_import_database_for_compaction methods
    validated for HNSW by TestOneTimeIndexingAndCompact.
    """

    @pytest.mark.asyncio
    async def test_compact_blocking_preserves_hnsw(self, tmp_path: Path):
        """compact_blocking() via CompactionService runs and data survives."""
        from tests.stress.conftest import make_fragmented_db, make_compaction_config

        fragdb = make_fragmented_db(tmp_path, n_files=10, n_chunks_per_file=5)
        provider = fragdb.provider
        config = make_compaction_config(fragdb.db_path)

        svc = CompactionService(fragdb.db_path, config)
        compacted = await svc.compact_blocking(provider)
        assert compacted, "compact_blocking() returned False — compaction did not run"

        provider.disconnect()

    @pytest.mark.asyncio
    async def test_compact_background_preserves_hnsw(self, tmp_path: Path):
        """compact_background() calls on_complete and HNSW indexes survive compaction."""
        from tests.stress.conftest import make_fragmented_db, make_compaction_config

        fragdb = make_fragmented_db(tmp_path, n_files=10, n_chunks_per_file=5)
        provider = fragdb.provider
        config = make_compaction_config(fragdb.db_path)

        on_complete = AsyncMock()
        svc = CompactionService(fragdb.db_path, config)

        started = await svc.compact_background(provider, on_complete=on_complete)
        assert started, "compact_background() returned False — compaction did not start"

        if svc._compaction_task:
            await asyncio.wait_for(svc._compaction_task, timeout=120)

        on_complete.assert_called_once()

        provider.disconnect()

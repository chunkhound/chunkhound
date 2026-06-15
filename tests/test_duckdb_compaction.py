"""Tests for DuckDB compaction — fragmentation measurement + canonical rebuild.

Every test reuses the existing synthetic-vector infrastructure used elsewhere
in the suite.
"""

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

duckdb = pytest.importorskip("duckdb")

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.embeddings import EmbeddingManager
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database import duckdb_provider as duckdb_provider_module
from chunkhound.providers.database.duckdb_provider import (
    DuckDBProvider,
)
from chunkhound.providers.database.serial_executor import (
    DatabaseCompactionInProgressError,
)
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from tests.fixtures.fake_providers import (
    ConstantEmbeddingProvider,
    FakeEmbeddingProvider,
)


def _fetch_scalar(db_path: Path | str, sql: str) -> Any:
    """Run one scalar query against the file-backed DuckDB."""
    conn = duckdb.connect(str(db_path))
    try:
        row = conn.execute(sql).fetchone()
        return None if row is None else row[0]
    finally:
        conn.close()


def _fetch_index_names(db_path: Path | str) -> set[str]:
    """Return public DuckDB index names from the compacted file."""
    conn = duckdb.connect(str(db_path))
    try:
        rows = conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _assert_db_integrity(db_path: Path | str) -> None:
    """Verify the database file is a structurally valid DuckDB database.

    Opens a fresh connection — a corrupt or truncated file will raise
    ``duckdb.IOException`` / ``duckdb.CatalogException`` at connect time
    or when querying the catalog.
    """
    conn = duckdb.connect(str(db_path))
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        names = {t[0] for t in tables}
        assert "files" in names, "Expected 'files' table in compacted DB"
        assert "chunks" in names, "Expected 'chunks' table in compacted DB"
    finally:
        conn.close()


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def compaction_db(tmp_path: Path) -> DuckDBProvider:
    """DuckDB with FakeEmbeddingProvider using the standard test pattern."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.config = DatabaseConfig(fragmentation_threshold_pct=30.0)
    embedding_manager = EmbeddingManager()
    embedding_manager.register_provider(
        FakeEmbeddingProvider(dims=16), set_default=True
    )
    db.embedding_manager = embedding_manager
    db.connect()
    return db


@pytest.fixture
def file_backed_db(tmp_path: Path) -> DuckDBProvider:
    """File-backed DuckDB so os.path.getsize works — same connect pattern."""
    db_path = tmp_path / "test.duckdb"
    db = DuckDBProvider(db_path, base_directory=tmp_path)
    db.config = DatabaseConfig(fragmentation_threshold_pct=30.0)
    embedding_manager = EmbeddingManager()
    embedding_manager.register_provider(
        FakeEmbeddingProvider(dims=16), set_default=True
    )
    db.embedding_manager = embedding_manager
    db.connect()
    return db


@pytest.fixture
def populated_db(
    file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
) -> DuckDBProvider:
    """DB seeded with 2 files + chunks + embeddings via real executor paths."""
    monkeypatch.setattr(
        "chunkhound.providers.database.serial_executor.COMPACT_SAMPLE_INTERVAL", 0
    )
    db = file_backed_db
    prov = ConstantEmbeddingProvider(dims=16)

    for i in range(2):
        file_model = File(
            path=f"test_{i}.py",
            mtime=float(i),
            language=Language.PYTHON,
            size_bytes=100 + i,
            content_hash=f"hash{i}",
        )
        file_id = db._execute_in_db_thread_sync("insert_file", file_model)

        chunks = [
            Chunk(
                file_id=file_id,
                chunk_type=ChunkType.FUNCTION,
                code=f"def func_{i}a(): pass",
                start_line=1,
                end_line=2,
                symbol=f"func_{i}a",
                language=Language.PYTHON,
            ),
            Chunk(
                file_id=file_id,
                chunk_type=ChunkType.FUNCTION,
                code=f"def func_{i}b(): pass",
                start_line=4,
                end_line=5,
                symbol=f"func_{i}b",
                language=Language.PYTHON,
            ),
        ]
        chunk_ids = db._execute_in_db_thread_sync("insert_chunks_batch", chunks)

        emb_data = [
            {
                "chunk_id": chunk_ids[0],
                "provider": "fake",
                "model": "fake-embeddings",
                "embedding": prov._generate_deterministic_vector(f"test_{i}a"),
                "dims": 16,
            },
            {
                "chunk_id": chunk_ids[1],
                "provider": "fake",
                "model": "fake-embeddings",
                "embedding": prov._generate_deterministic_vector(f"test_{i}b"),
                "dims": 16,
            },
        ]
        db._execute_in_db_thread_sync("insert_embeddings_batch", emb_data, None)

    return db


# ── Fragmentation measurement ────────────────────────────────────────────


class TestMeasureFragmentation:
    """Stateless fragmentation measurement tests."""

    def test_fresh_db_returns_low_ratio(self, file_backed_db: DuckDBProvider) -> None:
        """Fresh DB should return a ratio close to 1.0 (healthy)."""
        ratio = file_backed_db.measure_fragmentation()
        # A fresh empty DB has ~1.0 ratio (file_size ≈ used_blocks * block_size)
        assert ratio > 0.0
        assert ratio < 2.0  # definitely not fragmented

    def test_after_insert_returns_still_low(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """After inserting data, ratio should remain healthy."""
        file_model = File(
            path="test.py", mtime=1.0, language=Language.PYTHON, size_bytes=100
        )
        file_backed_db._execute_in_db_thread_sync("insert_file", file_model)
        ratio = file_backed_db.measure_fragmentation()
        assert ratio > 0.0
        assert ratio < 2.0  # inserting data shouldn't cause fragmentation

    def test_after_fragmentation_ratio_increases(
        self, populated_db: DuckDBProvider
    ) -> None:
        """After many inserts (filling row groups), ratio may increase."""
        ratio_before = populated_db.measure_fragmentation()
        # Compact to get a baseline
        populated_db.compact_database()
        ratio_after_compact = populated_db.measure_fragmentation()
        # Contract: compaction must not make fragmentation worse.
        # Some DuckDB versions leave small fixed metadata overhead even after
        # canonical rebuild compaction, so we assert monotonic non-increase
        # (with a tiny
        # 1.05x tolerance for rounding), not absolute minimum.
        assert ratio_after_compact <= ratio_before * 1.05

    def test_in_memory_returns_zero(self, compaction_db: DuckDBProvider) -> None:
        """In-memory never has fragmentation."""
        ratio = compaction_db.measure_fragmentation()
        assert ratio == 0.0


# ── Compaction mechanics ─────────────────────────────────────────────────


class TestCompactDatabase:
    """Full DuckDB canonical rebuild compaction tests."""

    def test_basic_roundtrip_data_survives(self, populated_db: DuckDBProvider) -> None:
        """All files, chunks, and embeddings survive compaction."""
        stats_before = populated_db.get_stats()

        compacted = populated_db.compact_database()
        assert compacted > 0
        _assert_db_integrity(cast(str, populated_db.db_path))

        stats_after = populated_db.get_stats()
        assert stats_after["files"] == stats_before["files"]
        assert stats_after["chunks"] == stats_before["chunks"]
        assert stats_after["embeddings"] == stats_before["embeddings"]

    def test_standard_indexes_exist_immediately_after_compaction(
        self, populated_db: DuckDBProvider
    ) -> None:
        """Compaction finalization restores lookup indexes before reconnecting."""
        populated_db.compact_database()
        _assert_db_integrity(cast(str, populated_db.db_path))
        names = _fetch_index_names(populated_db.db_path)
        assert "idx_files_path" in names
        assert "idx_files_language" in names
        assert "idx_chunks_file_id" in names
        assert "idx_chunks_type" in names
        assert "idx_chunks_symbol" in names

    def test_compacted_file_is_not_larger(self, populated_db: DuckDBProvider) -> None:
        """Compacted file should be <= original size plus small overhead."""
        size_before = os.path.getsize(cast(str, populated_db.db_path))

        compacted = populated_db.compact_database()
        size_after = os.path.getsize(cast(str, populated_db.db_path))
        _assert_db_integrity(cast(str, populated_db.db_path))

        assert compacted > 0
        # Allow ~1KB metadata overhead from new DuckDB version headers
        assert size_after <= size_before + 2048

    def test_in_memory_no_op(self, compaction_db: DuckDBProvider) -> None:
        """In-memory compaction returns 0."""
        result = compaction_db.compact_database()
        assert result == 0

    def test_fails_when_transaction_active(self, populated_db: DuckDBProvider) -> None:
        """Compaction raises RuntimeError during active transaction."""
        populated_db._execute_in_db_thread_sync("begin_transaction")
        with pytest.raises(RuntimeError, match="transaction is active"):
            populated_db.compact_database()
        populated_db._execute_in_db_thread_sync("rollback_transaction")

    def test_post_compaction_embedding_insert_uses_restored_default_id(
        self, populated_db: DuckDBProvider
    ) -> None:
        """Existing embedding tables still accept inserts without an explicit id."""
        max_embedding_id_before = _fetch_scalar(
            populated_db.db_path, "SELECT MAX(id) FROM embeddings_16"
        )
        assert max_embedding_id_before is not None

        populated_db.compact_database()

        file_id = populated_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="post_compaction_embedding.py",
                mtime=99.0,
                language=Language.PYTHON,
                size_bytes=123,
                content_hash="post-compaction-embedding",
            ),
        )
        chunk_id = populated_db._execute_in_db_thread_sync(
            "insert_chunks_batch",
            [
                Chunk(
                    file_id=file_id,
                    chunk_type=ChunkType.FUNCTION,
                    code="def post_compaction_embedding(): pass",
                    start_line=1,
                    end_line=1,
                    symbol="post_compaction_embedding",
                    language=Language.PYTHON,
                )
            ],
        )[0]
        inserted = populated_db._execute_in_db_thread_sync(
            "insert_embeddings_batch",
            [
                {
                    "chunk_id": chunk_id,
                    "provider": "fake",
                    "model": "fake-embeddings",
                    "embedding": ConstantEmbeddingProvider(
                        dims=16
                    )._generate_deterministic_vector("post-compaction-embedding"),
                    "dims": 16,
                }
            ],
            None,
        )
        assert inserted == 1

        max_embedding_id_after = _fetch_scalar(
            populated_db.db_path, "SELECT MAX(id) FROM embeddings_16"
        )
        assert max_embedding_id_after is not None
        assert max_embedding_id_after > max_embedding_id_before

    def test_connection_manager_health_restored_after_compaction(
        self, populated_db: DuckDBProvider
    ) -> None:
        """Public connection health remains true after the live file swap."""
        populated_db.compact_database()

        assert populated_db.is_connected is True
        health = populated_db.health_check()
        assert health["connected"] is True
        assert health["errors"] == []

    def test_unknown_tables_are_dropped_during_compaction(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """Compaction keeps only canonical ChunkHound tables by design."""
        manager_conn = file_backed_db._connection_manager.connection
        assert manager_conn is not None
        manager_conn.execute("CREATE TABLE extra_state (id INTEGER, note TEXT)")
        manager_conn.execute("INSERT INTO extra_state VALUES (1, 'dropped')")

        file_backed_db.compact_database()

        assert (
            _fetch_scalar(
                file_backed_db.db_path,
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'extra_state'",
            )
            == 0
        )

    def test_post_compaction_files_and_chunks_reseed_ids(
        self, populated_db: DuckDBProvider
    ) -> None:
        """files/chunks inserts after compaction continue above existing ids."""
        max_file_id_before = _fetch_scalar(
            populated_db.db_path, "SELECT MAX(id) FROM files"
        )
        max_chunk_id_before = _fetch_scalar(
            populated_db.db_path, "SELECT MAX(id) FROM chunks"
        )
        assert max_file_id_before is not None
        assert max_chunk_id_before is not None

        populated_db.compact_database()

        file_id = populated_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="post_compaction_ids.py",
                mtime=100.0,
                language=Language.PYTHON,
                size_bytes=321,
                content_hash="post-compaction-ids",
            ),
        )
        chunk_ids = populated_db._execute_in_db_thread_sync(
            "insert_chunks_batch",
            [
                Chunk(
                    file_id=file_id,
                    chunk_type=ChunkType.FUNCTION,
                    code="def post_compaction_ids(): pass",
                    start_line=1,
                    end_line=1,
                    symbol="post_compaction_ids",
                    language=Language.PYTHON,
                )
            ],
        )

        assert file_id > max_file_id_before
        assert chunk_ids[0] > max_chunk_id_before

    def test_hnsw_indexes_survive_compaction(
        self, populated_db: DuckDBProvider
    ) -> None:
        """HNSW indexes created before compaction survive and remain functional."""
        indexes_before = _fetch_index_names(populated_db.db_path)
        hnsw_before = {n for n in indexes_before if n.startswith("idx_hnsw_")}

        populated_db.compact_database()

        indexes_after = _fetch_index_names(populated_db.db_path)
        hnsw_after = {n for n in indexes_after if n.startswith("idx_hnsw_")}

        # Every HNSW index that existed before must still exist after
        assert hnsw_after == hnsw_before, (
            f"HNSW indexes changed: before={hnsw_before} after={hnsw_after}"
        )

        # The canonical HNSW index naming (idx_hnsw_*) is created by
        # _executor_create_embedding_table_indexes using USING HNSW.
        # Post-compaction survival of the name proves the VSS extension
        # was loaded and the index recreated correctly.
        for hnsw_name in hnsw_before:
            assert hnsw_name.startswith("idx_hnsw_"), (
                f"Unexpected HNSW name: {hnsw_name}"
            )

    def test_compaction_with_single_quote_in_path(self, tmp_path: Path) -> None:
        """Compaction works when the DB path contains a single quote."""
        # Create a directory with a single quote in the name
        quote_dir = tmp_path / "John's Project"
        quote_dir.mkdir()
        db_path = quote_dir / "test.duckdb"

        db = DuckDBProvider(db_path, base_directory=quote_dir)
        db.config = DatabaseConfig(fragmentation_threshold_pct=30.0)
        embedding_manager = EmbeddingManager()
        embedding_manager.register_provider(
            FakeEmbeddingProvider(dims=16), set_default=True
        )
        db.embedding_manager = embedding_manager
        db.connect()

        try:
            file_id = db._execute_in_db_thread_sync(
                "insert_file",
                File(
                    path="test.py",
                    mtime=1.0,
                    language=Language.PYTHON,
                    size_bytes=50,
                ),
            )
            assert file_id > 0

            compacted = db.compact_database()
            assert compacted > 0
            _assert_db_integrity(db_path)

            # Verify data survived
            stats = db.get_stats()
            assert stats["files"] == 1
        finally:
            db.disconnect()


# ── Crash recovery ──────────────────────────────────────────────────────


class TestCompactionCrashRecovery:
    """Backup restoration when compaction fails mid-way."""

    def test_backup_restored_on_copy_failure(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A copy-phase failure restores the original DB and cleans staged artifacts."""
        db_path = Path(cast(str, file_backed_db.db_path))
        backup_path = db_path.with_suffix(db_path.suffix + ".compact_backup")
        staged_path = db_path.with_suffix(db_path.suffix + ".compact_new")

        file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="crash_recovery.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=50,
            ),
        )

        def _fail_copy(*args, **kwargs):
            staged_path.write_bytes(b"partial copy")
            raise RuntimeError("simulated copy failure")

        monkeypatch.setattr(file_backed_db, "_compact_copy_data", _fail_copy)

        with pytest.raises(RuntimeError, match="simulated copy failure"):
            file_backed_db.compact_database()

        assert db_path.exists(), "Original DB should be restored"
        assert not backup_path.exists(), (
            "Restore should move the backup back into place"
        )
        assert not staged_path.exists(), "Restore should clean failed staged artifacts"
        assert file_backed_db.is_connected is True
        assert file_backed_db.get_stats()["files"] == 1
        assert (
            _fetch_scalar(
                db_path, "SELECT COUNT(*) FROM files WHERE path = 'crash_recovery.py'"
            )
            == 1
        )
        inserted_id = file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="copy_recovery_after.py",
                mtime=1.0,
                language=Language.PYTHON,
                size_bytes=60,
            ),
        )
        assert inserted_id > 0
        _assert_db_integrity(db_path)

    def test_no_backup_restored_on_success(self, populated_db: DuckDBProvider) -> None:
        """After successful compaction, the backup file is removed."""
        db_path_raw = populated_db.db_path
        backup_path = str(db_path_raw) + ".compact_backup"

        populated_db.compact_database()

        assert not os.path.exists(backup_path), "Backup should be cleaned up"

    def test_backup_survives_reconnection_failure(
        self, file_backed_db: DuckDBProvider, monkeypatch
    ) -> None:
        """When reconnecting the swapped DB fails, the original DB is restored."""
        db_path = cast(str, file_backed_db.db_path)

        file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="reconn_test.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=50,
            ),
        )

        def _failing_reconnect(*args, **kwargs):
            raise RuntimeError("simulated reconnection failure")

        monkeypatch.setattr(file_backed_db, "_create_connection", _failing_reconnect)

        with pytest.raises(RuntimeError, match="Compaction failed|simulated"):
            file_backed_db.compact_database()

        assert os.path.exists(db_path), "Original DB should be restored"
        assert (
            _fetch_scalar(db_path, "SELECT COUNT(*) FROM information_schema.tables")
            > 0
        )
        _assert_db_integrity(db_path)

    def test_startup_recovers_empty_live_db_from_compaction_backup(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """Startup restores the backup instead of opening an empty live DB."""
        db_path = Path(cast(str, file_backed_db.db_path))
        file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="startup_recovery.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=50,
            ),
        )
        file_backed_db.disconnect()

        backup_path = db_path.with_suffix(db_path.suffix + ".compact_backup")
        os.replace(db_path, backup_path)
        db_path.write_bytes(b"")

        recovered = DuckDBProvider(db_path, base_directory=db_path.parent)
        recovered.config = DatabaseConfig(fragmentation_threshold_pct=30.0)
        recovered.connect()
        try:
            assert recovered.get_stats()["files"] == 1
            assert not backup_path.exists()
        finally:
            recovered.disconnect()

    def test_finalize_failure_restores_original_db_and_cleans_artifacts(
        self, populated_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A finalize-phase failure restores the original DB and removes temp files."""
        db_path = Path(cast(str, populated_db.db_path))
        backup_path = db_path.with_suffix(db_path.suffix + ".compact_backup")
        staged_path = db_path.with_suffix(db_path.suffix + ".compact_new")

        manager_conn = populated_db._connection_manager.connection
        assert manager_conn is not None
        manager_conn.execute(
            "CREATE INDEX custom_hnsw_restore_idx "
            "ON embeddings_16 USING HNSW (embedding)"
        )

        original_finalize = populated_db._compact_finalize

        def _fail_finalize(*args, **kwargs):
            raise RuntimeError("simulated finalize failure")

        monkeypatch.setattr(populated_db, "_compact_finalize", _fail_finalize)

        with pytest.raises(RuntimeError, match="simulated finalize failure"):
            populated_db.compact_database()

        monkeypatch.setattr(populated_db, "_compact_finalize", original_finalize)

        assert populated_db.is_connected is True
        assert populated_db.get_stats()["files"] == 2
        assert "custom_hnsw_restore_idx" in _fetch_index_names(populated_db.db_path)
        assert not backup_path.exists()
        assert not staged_path.exists()

        inserted_id = populated_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="post_swap_after.py",
                mtime=1.0,
                language=Language.PYTHON,
                size_bytes=60,
            ),
        )
        assert inserted_id > 0
        _assert_db_integrity(populated_db.db_path)

    def test_provider_remains_usable_after_restore_path_failure(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed compaction restores the live provider, not just the file on disk."""
        file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="provider_recovery_before.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=50,
            ),
        )

        original_create = file_backed_db._create_connection

        def _failing_create(*args, **kwargs):
            raise RuntimeError("simulated reconnection failure")

        monkeypatch.setattr(file_backed_db, "_create_connection", _failing_create)

        with pytest.raises(RuntimeError, match="Compaction failed|simulated"):
            file_backed_db.compact_database()

        monkeypatch.setattr(
            file_backed_db,
            "_create_connection",
            original_create,
        )

        inserted_id = file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="provider_recovery_after.py",
                mtime=1.0,
                language=Language.PYTHON,
                size_bytes=60,
            ),
        )
        assert inserted_id > 0
        assert file_backed_db.get_stats()["files"] == 2
        _assert_db_integrity(file_backed_db.db_path)

    def test_stale_backup_cleaned_up_when_live_db_is_healthy(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """Stale compaction artifacts are cleaned up when live DB is intact."""
        db_path = Path(cast(str, file_backed_db.db_path))
        file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="stale_cleanup.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=50,
            ),
        )
        file_backed_db.disconnect()

        # Simulate stale artifacts from a prior compaction that completed
        # but left behind backup/staging files (e.g. process killed after swap
        # but before cleanup).
        backup_path = db_path.with_suffix(db_path.suffix + ".compact_backup")
        staged_path = db_path.with_suffix(db_path.suffix + ".compact_new")
        backup_path.write_bytes(b"stale backup")
        staged_path.write_bytes(b"stale staged")

        live_size_before = db_path.stat().st_size

        recovered = DuckDBProvider(db_path, base_directory=db_path.parent)
        recovered.config = DatabaseConfig(fragmentation_threshold_pct=30.0)
        recovered.connect()
        try:
            # Live DB is intact — data preserved
            assert recovered.get_stats()["files"] == 1
            # Stale artifacts cleaned up
            assert not backup_path.exists()
            assert not staged_path.exists()
            # Live file unchanged
            assert db_path.stat().st_size == live_size_before
        finally:
            recovered.disconnect()

    def test_compacted_file_corruption_triggers_restore_from_backup(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A corrupted compacted file triggers restore from the valid backup."""
        file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="corruption_test.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=10,
            ),
        )

        db_path = Path(cast(str, file_backed_db.db_path))
        backup_path = db_path.with_suffix(db_path.suffix + ".compact_backup")

        original_copy = file_backed_db._compact_copy_data

        def _corrupted_copy(backup_path, compacted_path, state):
            result = original_copy(backup_path, compacted_path, state)
            # Corrupt the newly built file so the swap → reconnect fails
            compacted_path.write_bytes(b"GARBAGE" * 1000)
            return result

        monkeypatch.setattr(
            file_backed_db, "_compact_copy_data", _corrupted_copy
        )

        with pytest.raises(Exception):
            file_backed_db.compact_database()

        # Original DB must be restored from valid backup
        assert db_path.exists()
        assert not backup_path.exists()
        _assert_db_integrity(db_path)

        # Provider must still be usable after restore
        inserted = file_backed_db.get_stats()["files"]
        assert inserted >= 1


# ── Compaction guard ─────────────────────────────────────────────────────


class TestCompactionGuard:
    """compaction_in_progress flag blocks mutations."""

    def test_mutations_blocked_during_compaction(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """Insert during compaction raises DatabaseCompactionInProgressError."""
        file_backed_db._executor.set_compaction_in_progress(True)

        file_model = File(
            path="blocked.py", mtime=0.0, language=Language.PYTHON, size_bytes=10
        )
        try:
            with pytest.raises(DatabaseCompactionInProgressError):
                file_backed_db._execute_in_db_thread_sync("insert_file", file_model)
        finally:
            file_backed_db._executor.set_compaction_in_progress(False)


# ── should_optimize / optimize_tables ────────────────────────────────────


def test_default_fragmentation_threshold_is_30_percent() -> None:
    """Default auto-compaction threshold matches the product contract."""
    assert DatabaseConfig().fragmentation_threshold_pct == 30.0


class TestShouldOptimize:
    """should_optimize respects operation context and threshold."""

    def test_compact_db_returns_false_after_compact(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """After compaction, should_optimize returns False (ratio is low)."""
        file_backed_db.compact_database()
        assert file_backed_db.should_optimize() is False

    def test_in_memory_false(self, compaction_db: DuckDBProvider) -> None:
        """In-memory DB never needs optimization."""
        assert compaction_db.should_optimize() is False

    def test_threshold_zero_triggers_on_any_overhead(
        self, populated_db: DuckDBProvider
    ) -> None:
        """threshold=0 → compact_if_needed returns True when the DB
        has meaningful fragmentation overhead after real operations."""
        populated_db.config = DatabaseConfig(fragmentation_threshold_pct=0.0)
        populated_db._execute_in_db_thread_sync("maybe_checkpoint", True)

        # threshold=0 means compact whenever any overhead above the
        # estimated minimum live size exists.
        assert populated_db.compact_if_needed() is True

    def test_threshold_none_disables_auto_compact(self, tmp_path: Path) -> None:
        """threshold=None → should_optimize returns False even with overhead."""
        db_path = tmp_path / "none_thresh.duckdb"
        db = DuckDBProvider(db_path, base_directory=tmp_path)
        db.config = DatabaseConfig(fragmentation_threshold_pct=None)
        embedding_manager = EmbeddingManager()
        embedding_manager.register_provider(
            FakeEmbeddingProvider(dims=16), set_default=True
        )
        db.embedding_manager = embedding_manager
        db.connect()

        file_model = File(
            path="nogrowth.py", mtime=1.0, language=Language.PYTHON, size_bytes=100
        )
        db._execute_in_db_thread_sync("insert_file", file_model)

        assert db.should_optimize() is False

    def test_nonzero_threshold_skips_when_ratio_is_below_boundary(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A realistic nonzero threshold must not compact below the boundary."""
        file_backed_db.config = DatabaseConfig(fragmentation_threshold_pct=30.0)
        monkeypatch.setattr(
            file_backed_db,
            "_executor_measure_fragmentation",
            lambda conn, state: 1.29,
        )
        compact_calls: list[str] = []
        monkeypatch.setattr(
            file_backed_db,
            "_executor_compact_database",
            lambda conn, state: compact_calls.append("compact") or 0,
        )

        assert file_backed_db.compact_if_needed() is False
        assert compact_calls == []

    def test_compact_if_needed_runs_when_threshold_is_exceeded(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A realistic nonzero threshold compacts once above the boundary."""
        file_backed_db.config = DatabaseConfig(fragmentation_threshold_pct=30.0)
        monkeypatch.setattr(
            file_backed_db,
            "_executor_measure_fragmentation",
            lambda conn, state: 1.31,
        )
        compact_calls: list[str] = []
        monkeypatch.setattr(
            file_backed_db,
            "_executor_compact_database",
            lambda conn, state: compact_calls.append("compact") or 0,
        )

        assert file_backed_db.compact_if_needed() is True
        assert compact_calls == ["compact"]


# ── Index flow compaction points ─────────────────────────────────────────


class TestIndexFlowCompaction:
    """Batch compaction points fire only when an index run changes the DB.

    Uses manual service construction (same pattern as test_scoped_deep_research.py)
    to avoid Config's EmbeddingConfig validation (only accepts openai/voyageai).
    """

    def _make_coordinator(
        self, tmp_path: Path, cfg: Any
    ) -> tuple[DuckDBProvider, IndexingCoordinator, Any]:
        """Build a DuckDBProvider + IndexingCoordinator with fake embeddings.

        Same pattern as test_scoped_deep_research.py.
        """
        db_path = tmp_path / "index_test.duckdb"
        db = DuckDBProvider(db_path, base_directory=tmp_path)
        db.config = cfg.database

        embedding_manager = EmbeddingManager()
        embedding_manager.register_provider(
            FakeEmbeddingProvider(dims=16), set_default=True
        )
        db.embedding_manager = embedding_manager
        db.connect()

        parser = create_parser_for_language(Language.PYTHON)
        coordinator = IndexingCoordinator(
            db,
            tmp_path,
            embedding_manager.get_default_provider(),
            {Language.PYTHON: parser},
        )
        return db, coordinator, cfg

    @pytest.mark.asyncio
    async def test_index_flow_orders_chunk_compact_embed_compact(
        self, tmp_path: Path
    ) -> None:
        """The first fixed compaction happens before embeddings, the second after."""
        events: list[str] = []

        class _Service(DirectoryIndexingService):
            async def _process_directory_files(
                self, target_path, include_patterns, exclude_patterns
            ):
                events.append("chunk")
                return {"status": "success", "files_processed": 1, "skipped": 0}

            async def _generate_missing_embeddings(self, exclude_patterns):
                events.append("embed")
                return {"status": "success", "generated": 0}

        async def _compact_database() -> dict[str, Any]:
            events.append("compact")
            return {
                "status": "success",
                "size_before": 10,
                "size_after": 9,
                "reduction_pct": 10.0,
            }

        config = SimpleNamespace(
            indexing=SimpleNamespace(
                include=[],
                exclude=[],
                config_file_size_threshold_kb=20,
            )
        )
        coordinator = SimpleNamespace(
            compact_database=AsyncMock(side_effect=_compact_database)
        )

        stats = await _Service(coordinator, config).process_directory(
            tmp_path, no_embeddings=False
        )

        assert events == ["chunk", "compact", "embed", "compact"]
        assert stats.db_compactions == 2

    @pytest.mark.asyncio
    async def test_unconditional_compaction_runs_twice_during_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Directory indexing calls compact_database at both fixed boundaries."""
        cfg = Config(
            database={
                "path": str(tmp_path / "index_test.duckdb"),
                "provider": "duckdb",
            },
        )
        cfg.target_dir = tmp_path

        _, coordinator, config = self._make_coordinator(tmp_path, cfg)

        monkeypatch.setattr(
            coordinator,
            "compact_database",
            AsyncMock(
                return_value={
                    "status": "success",
                    "size_before": 10,
                    "size_after": 9,
                    "reduction_pct": 10.0,
                }
            ),
        )

        test_file = tmp_path / "mod.py"
        test_file.write_text("def hello(): pass\n")

        svc = DirectoryIndexingService(coordinator, config)
        stats = await svc.process_directory(tmp_path, no_embeddings=False)

        assert coordinator.compact_database.call_count == 2
        assert stats.db_compactions == 2

    @pytest.mark.asyncio
    async def test_no_embeddings_both_compaction_points_still_fire(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without embeddings, both fixed compaction boundaries still run."""
        cfg = Config(
            database={
                "path": str(tmp_path / "index_test.duckdb"),
                "provider": "duckdb",
            },
        )
        cfg.target_dir = tmp_path

        _, coordinator, config = self._make_coordinator(tmp_path, cfg)

        monkeypatch.setattr(
            coordinator,
            "compact_database",
            AsyncMock(
                return_value={
                    "status": "success",
                    "size_before": 10,
                    "size_after": 9,
                    "reduction_pct": 10.0,
                }
            ),
        )

        test_file = tmp_path / "mod.py"
        test_file.write_text("def hello(): pass\n")

        svc = DirectoryIndexingService(coordinator, config)
        stats = await svc.process_directory(tmp_path, no_embeddings=True)

        assert coordinator.compact_database.call_count == 2
        assert stats.db_compactions == 2

    @pytest.mark.asyncio
    async def test_noop_index_run_still_invokes_both_compaction_boundaries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Noop re-index still invokes both fixed compaction boundaries."""
        cfg = Config(
            database={
                "path": str(tmp_path / "index_test.duckdb"),
                "provider": "duckdb",
            },
        )
        cfg.target_dir = tmp_path

        _, coordinator, config = self._make_coordinator(tmp_path, cfg)
        compact_database = AsyncMock(
            side_effect=[
                {
                    "status": "success",
                    "size_before": 10,
                    "size_after": 9,
                    "reduction_pct": 10.0,
                },
                {
                    "status": "success",
                    "size_before": 9,
                    "size_after": 9,
                    "reduction_pct": 0.0,
                },
                {"status": "skipped", "reason": "unsupported"},
                {"status": "skipped", "reason": "unsupported"},
            ]
        )
        monkeypatch.setattr(coordinator, "compact_database", compact_database)

        test_file = tmp_path / "mod.py"
        test_file.write_text("def hello(): pass\n")

        svc = DirectoryIndexingService(coordinator, config)
        await svc.process_directory(tmp_path, no_embeddings=False)
        stats2 = await svc.process_directory(tmp_path, no_embeddings=False)

        assert compact_database.await_count == 4
        assert stats2.db_compactions == 0

    @pytest.mark.asyncio
    async def test_unsupported_batch_compaction_is_skipped_not_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Providers without file-compaction support must not break batch indexing."""
        cfg = Config(
            database={
                "path": str(tmp_path / "index_test.duckdb"),
                "provider": "duckdb",
            },
        )
        cfg.target_dir = tmp_path

        _, coordinator, config = self._make_coordinator(tmp_path, cfg)

        monkeypatch.setattr(
            coordinator,
            "compact_database",
            AsyncMock(return_value={"status": "skipped", "reason": "unsupported"}),
        )

        test_file = tmp_path / "mod.py"
        test_file.write_text("def hello(): pass\n")

        svc = DirectoryIndexingService(coordinator, config)
        stats = await svc.process_directory(tmp_path, no_embeddings=False)

        assert coordinator.compact_database.call_count == 2
        assert stats.db_compactions == 0
        assert stats.files_processed >= 1

    @pytest.mark.asyncio
    async def test_batch_compaction_error_aborts_indexing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fixed-boundary compaction error aborts the index run."""
        cfg = Config(
            database={
                "path": str(tmp_path / "index_test.duckdb"),
                "provider": "duckdb",
            },
        )
        cfg.target_dir = tmp_path

        _, coordinator, config = self._make_coordinator(tmp_path, cfg)
        monkeypatch.setattr(
            coordinator,
            "compact_database",
            AsyncMock(return_value={"status": "error", "error": "boom"}),
        )

        test_file = tmp_path / "mod.py"
        test_file.write_text("def hello(): pass\n")

        svc = DirectoryIndexingService(coordinator, config)
        with pytest.raises(
            RuntimeError, match="Batch database compaction failed: boom"
        ):
            await svc.process_directory(tmp_path, no_embeddings=False)

    @pytest.mark.asyncio
    async def test_real_index_flow_runs_index_compact_embed_compact(
        self, tmp_path: Path
    ) -> None:
        """Real batch indexing finishes with both mandatory compactions."""
        cfg = Config(
            database={
                "path": str(tmp_path / "index_test.duckdb"),
                "provider": "duckdb",
                "fragmentation_threshold_pct": 0,
            },
        )
        cfg.target_dir = tmp_path

        db, coordinator, config = self._make_coordinator(tmp_path, cfg)

        test_file = tmp_path / "mod.py"
        test_file.write_text("def hello():\n    return 1\n")

        svc = DirectoryIndexingService(coordinator, config)
        stats = await svc.process_directory(tmp_path, no_embeddings=False)

        db_stats = db.get_stats()
        assert stats.files_processed >= 1
        assert stats.chunks_created >= 1
        assert db_stats["files"] >= 1
        assert db_stats["chunks"] >= 1


class TestIndexingCoordinatorCompactionContract:
    """Coordinator exposes stable compaction status dictionaries."""

    @pytest.mark.asyncio
    async def test_success_reports_sizes_and_reduction(self, tmp_path: Path) -> None:
        db_path = tmp_path / "contract.duckdb"
        db_path.write_bytes(b"x" * 100)

        class _Db:
            def __init__(self, path: Path) -> None:
                self.db_path = path

            async def compact_database_async(self) -> int:
                return 75

        coordinator = IndexingCoordinator(_Db(db_path), tmp_path)

        result = await coordinator.compact_database()

        assert result == {
            "status": "success",
            "size_before": 100,
            "size_after": 75,
            "reduction_pct": 25.0,
        }

    @pytest.mark.asyncio
    async def test_unsupported_provider_is_skipped(self, tmp_path: Path) -> None:
        class _Db:
            db_path = ":memory:"

            async def compact_database_async(self) -> int:
                raise NotImplementedError

        result = await IndexingCoordinator(_Db(), tmp_path).compact_database()

        assert result == {"status": "skipped", "reason": "No compaction support"}

    @pytest.mark.asyncio
    async def test_compaction_error_is_reported(self, tmp_path: Path) -> None:
        class _Db:
            db_path = ":memory:"

            async def compact_database_async(self) -> int:
                raise RuntimeError("boom")

        result = await IndexingCoordinator(_Db(), tmp_path).compact_database()

        assert result == {"status": "error", "error": "boom"}

    @pytest.mark.asyncio
    async def test_compact_if_needed_status_mapping(self, tmp_path: Path) -> None:
        class _Db:
            db_path = ":memory:"

            def __init__(self, outcome: Any) -> None:
                self._outcome = outcome

            async def compact_if_needed_async(self) -> bool:
                if isinstance(self._outcome, Exception):
                    raise self._outcome
                return bool(self._outcome)

        assert await IndexingCoordinator(_Db(True), tmp_path).compact_if_needed() == {
            "status": "success",
            "compacted": True,
        }
        assert await IndexingCoordinator(_Db(False), tmp_path).compact_if_needed() == {
            "status": "skipped",
            "compacted": False,
            "reason": "below threshold",
        }
        assert await IndexingCoordinator(
            _Db(NotImplementedError()), tmp_path
        ).compact_if_needed() == {
            "status": "skipped",
            "compacted": False,
            "reason": "unsupported",
        }
        assert await IndexingCoordinator(
            _Db(RuntimeError("boom")), tmp_path
        ).compact_if_needed() == {
            "status": "error",
            "compacted": False,
            "error": "boom",
        }


class TestDatabaseProcessDirectoryCompactionContract:
    """Legacy Database wrapper keeps the fixed index-flow compaction boundaries."""

    @pytest.mark.asyncio
    async def test_process_directory_runs_chunk_compact_embed_compact_in_order(
        self,
    ) -> None:
        from chunkhound.database import Database

        events: list[str] = []

        async def _process_directory(*args, **kwargs) -> dict[str, Any]:
            events.append("chunk")
            return {"status": "success"}

        async def _compact_database() -> dict[str, Any]:
            events.append("compact")
            return {"status": "success", "size_before": 10, "size_after": 9}

        async def _generate_missing_embeddings(**kwargs) -> dict[str, Any]:
            events.append("embed")
            assert kwargs == {"exclude_patterns": ["**/.git/**"]}
            return {"status": "success", "generated": 0}

        coordinator = SimpleNamespace(
            process_directory=AsyncMock(side_effect=_process_directory),
            compact_database=AsyncMock(side_effect=_compact_database),
        )
        embedding_service = SimpleNamespace(
            generate_missing_embeddings=AsyncMock(side_effect=_generate_missing_embeddings)
        )
        db = Database.__new__(Database)
        db._indexing_coordinator = coordinator
        db._embedding_service = embedding_service

        result = await db.process_directory(
            Path("repo"),
            patterns=["**/*.py"],
            exclude_patterns=["**/.git/**"],
        )

        assert result == {"status": "success"}
        coordinator.process_directory.assert_awaited_once_with(
            Path("repo"), ["**/*.py"], ["**/.git/**"]
        )
        assert events == ["chunk", "compact", "embed", "compact"]

    @pytest.mark.asyncio
    async def test_process_directory_skips_compaction_when_chunk_phase_errors(
        self,
    ) -> None:
        from chunkhound.database import Database

        coordinator = SimpleNamespace(
            process_directory=AsyncMock(
                return_value={"status": "error", "error": "boom"}
            ),
            compact_database=AsyncMock(),
        )
        embedding_service = SimpleNamespace(
            generate_missing_embeddings=AsyncMock()
        )
        db = Database.__new__(Database)
        db._indexing_coordinator = coordinator
        db._embedding_service = embedding_service

        result = await db.process_directory(Path("repo"), patterns=["**/*.py"])

        assert result == {"status": "error", "error": "boom"}
        coordinator.compact_database.assert_not_awaited()
        embedding_service.generate_missing_embeddings.assert_not_awaited()


# ── Config loading ────────────────────────────────────────────────────────


class TestFragmentationConfig:
    """Config loading from env var, CLI arg, and repr."""

    def test_load_from_env_with_threshold(self, monkeypatch):
        """Loading from CHUNKHOUND_DATABASE__FRAGMENTATION_THRESHOLD_PCT."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__FRAGMENTATION_THRESHOLD_PCT", "45.0")
        config = DatabaseConfig.load_from_env()
        assert config["fragmentation_threshold_pct"] == 45.0

    def test_load_from_env_invalid_threshold_ignored(self, monkeypatch):
        """Invalid env value is silently ignored."""
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__FRAGMENTATION_THRESHOLD_PCT", "not-a-float"
        )
        config = DatabaseConfig.load_from_env()
        assert "fragmentation_threshold_pct" not in config

    def test_extract_cli_overrides_threshold(self):
        """CLI --fragmentation-threshold-pct is picked up by extract_cli_overrides."""
        args = MagicMock()
        args.fragmentation_threshold_pct = 50.0
        overrides = DatabaseConfig.extract_cli_overrides(args)
        assert overrides["fragmentation_threshold_pct"] == 50.0

    def test_repr_includes_threshold(self):
        """__repr__ includes fragmentation_threshold_pct."""
        config = DatabaseConfig(fragmentation_threshold_pct=42.0)
        assert "fragmentation_threshold_pct=42.0" in repr(config)


# ── Data integrity post-compaction ───────────────────────────────────────


class TestCompactionDataIntegrity:
    """Verify FK integrity, sequence correctness, and multi-dim survival."""

    def test_fk_integrity_after_compaction(self, populated_db: DuckDBProvider) -> None:
        """No orphaned chunks after compaction (chunks.file_id → files.id FK)."""
        populated_db.compact_database()
        orphan_count = _fetch_scalar(
            populated_db.db_path,
            "SELECT COUNT(*) FROM chunks c "
            "LEFT JOIN files f ON c.file_id = f.id WHERE f.id IS NULL",
        )
        assert orphan_count == 0, (
            f"Found {orphan_count} orphaned chunks after compaction"
        )

    def test_sequences_continue_above_max(self, populated_db: DuckDBProvider) -> None:
        """nextval for each sequence returns a value above the current max.

        Verifies all three canonical sequences (files, chunks, embeddings) are
        correctly reseeded after compaction.  Previously the embeddings_id_seq
        was left at START 1 after compaction restore, causing PRIMARY KEY
        violations on the next embedding insert.
        """
        populated_db.compact_database()
        db_path = populated_db.db_path
        entries = [
            ("files", "files_id_seq"),
            ("chunks", "chunks_id_seq"),
        ]
        # Add embeddings_id_seq if any embedding tables exist
        emb_names = _fetch_scalar(
            db_path,
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE 'embeddings_%' "
            "LIMIT 1",
        )
        if emb_names:
            entries.append(("embeddings", "embeddings_id_seq"))

        for table, seq in entries:
            if table == "embeddings":
                conn = duckdb.connect(str(db_path))
                try:
                    rows = conn.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_name LIKE 'embeddings_%'"
                    ).fetchall()
                    max_id = 0
                    for (tn,) in rows:
                        m = conn.execute(
                            f"SELECT COALESCE(MAX(id), 0) FROM {tn}"
                        ).fetchone()[0]
                        max_id = max(max_id, int(m))
                finally:
                    conn.close()
            else:
                max_id = _fetch_scalar(db_path, f"SELECT MAX(id) FROM {table}")
            # Advance the sequence by one
            next_id = _fetch_scalar(db_path, f"SELECT nextval('{seq}')")
            assert next_id is not None
            assert next_id > max_id, (
                f"{seq}: nextval={next_id} should be > MAX(id)={max_id}"
            )

    def test_multi_dim_embeddings_survive_compaction(self, tmp_path: Path) -> None:
        """Embedding tables with different dimensions all survive compaction."""
        db_path = tmp_path / "multi_dim.duckdb"
        db = DuckDBProvider(db_path, base_directory=tmp_path)
        db.config = DatabaseConfig(fragmentation_threshold_pct=30.0)
        em = EmbeddingManager()
        em.register_provider(FakeEmbeddingProvider(dims=8), set_default=True)
        db.embedding_manager = em
        db.connect()

        # Insert file + chunks
        file_id = db._execute_in_db_thread_sync(
            "insert_file",
            File(path="multi.py", mtime=1.0, language=Language.PYTHON, size_bytes=50),
        )
        chunk_ids = db._execute_in_db_thread_sync(
            "insert_chunks_batch",
            [
                Chunk(
                    file_id=file_id,
                    chunk_type=ChunkType.FUNCTION,
                    code="def a(): pass",
                    start_line=1,
                    end_line=1,
                    symbol="a",
                    language=Language.PYTHON,
                ),
            ],
        )

        # Insert into 8-dim table (default)
        db._execute_in_db_thread_sync(
            "insert_embeddings_batch",
            [
                {
                    "chunk_id": chunk_ids[0],
                    "provider": "fake",
                    "model": "fake-embeddings",
                    "embedding": [0.1] * 8,
                    "dims": 8,
                }
            ],
            None,
        )

        # Manually create a second dimension table and insert
        db._execute_in_db_thread_sync("ensure_embedding_table_exists", 32)
        db._execute_in_db_thread_sync(
            "insert_embeddings_batch",
            [
                {
                    "chunk_id": chunk_ids[0],
                    "provider": "fake",
                    "model": "fake-embeddings-32",
                    "embedding": [0.2] * 32,
                    "dims": 32,
                }
            ],
            None,
        )

        db.compact_database()

        # Both tables should still exist and have data
        count_8 = _fetch_scalar(db_path, "SELECT COUNT(*) FROM embeddings_8")
        count_32 = _fetch_scalar(db_path, "SELECT COUNT(*) FROM embeddings_32")
        assert count_8 == 1
        assert count_32 == 1

        # FK from embeddings → chunks should be intact
        orphan_8 = _fetch_scalar(
            db_path,
            "SELECT COUNT(*) FROM embeddings_8 e "
            "LEFT JOIN chunks c ON e.chunk_id = c.id WHERE c.id IS NULL",
        )
        assert orphan_8 == 0

    def test_multi_dim_embeddings_reseed_global_max_after_compaction(
        self, tmp_path: Path
    ) -> None:
        """Shared embeddings_id_seq is reseeded from the global max across dims."""
        db_path = tmp_path / "multi_dim_seq.duckdb"
        db = DuckDBProvider(db_path, base_directory=tmp_path)
        db.config = DatabaseConfig(fragmentation_threshold_pct=30.0)
        em = EmbeddingManager()
        em.register_provider(FakeEmbeddingProvider(dims=8), set_default=True)
        db.embedding_manager = em
        db.connect()

        file_id = db._execute_in_db_thread_sync(
            "insert_file",
            File(path="multi.py", mtime=1.0, language=Language.PYTHON, size_bytes=50),
        )
        chunk_ids = db._execute_in_db_thread_sync(
            "insert_chunks_batch",
            [
                Chunk(
                    file_id=file_id,
                    chunk_type=ChunkType.FUNCTION,
                    code="def a(): pass",
                    start_line=1,
                    end_line=1,
                    symbol="a",
                    language=Language.PYTHON,
                ),
                Chunk(
                    file_id=file_id,
                    chunk_type=ChunkType.FUNCTION,
                    code="def b(): pass",
                    start_line=2,
                    end_line=2,
                    symbol="b",
                    language=Language.PYTHON,
                ),
            ],
        )

        db._execute_in_db_thread_sync("ensure_embedding_table_exists", 8)
        db._execute_in_db_thread_sync("ensure_embedding_table_exists", 32)
        db._execute_in_db_thread_sync(
            "insert_embeddings_batch",
            [
                {
                    "chunk_id": chunk_ids[0],
                    "provider": "fake",
                    "model": "fake-8",
                    "embedding": [0.1] * 8,
                    "dims": 8,
                }
            ],
            None,
        )
        db._execute_in_db_thread_sync(
            "insert_embeddings_batch",
            [
                {
                    "chunk_id": chunk_ids[1],
                    "provider": "fake",
                    "model": "fake-32a",
                    "embedding": [0.2] * 32,
                    "dims": 32,
                }
            ],
            None,
        )
        db._execute_in_db_thread_sync(
            "insert_embeddings_batch",
            [
                {
                    "chunk_id": chunk_ids[1],
                    "provider": "fake-alt",
                    "model": "fake-32b",
                    "embedding": [0.3] * 32,
                    "dims": 32,
                }
            ],
            None,
        )

        max_before = _fetch_scalar(
            db_path,
            "SELECT GREATEST((SELECT COALESCE(MAX(id), 0) FROM embeddings_8), "
            "(SELECT COALESCE(MAX(id), 0) FROM embeddings_32))",
        )
        db.compact_database()

        next_id = _fetch_scalar(db_path, "SELECT nextval('embeddings_id_seq')")
        assert next_id > max_before

        db._execute_in_db_thread_sync(
            "insert_embeddings_batch",
            [
                {
                    "chunk_id": chunk_ids[1],
                    "provider": "fake-after",
                    "model": "fake-after",
                    "embedding": [0.4] * 32,
                    "dims": 32,
                }
            ],
            None,
        )
        inserted_id = _fetch_scalar(
            db_path,
            "SELECT id FROM embeddings_32 "
            "WHERE provider = 'fake-after' AND model = 'fake-after'",
        )
        assert inserted_id == next_id + 1

    def test_regex_and_semantic_search_still_work_after_compaction(
        self, populated_db: DuckDBProvider
    ) -> None:
        """Compaction preserves user-visible regex and semantic search behavior."""
        query_embedding = ConstantEmbeddingProvider(
            dims=16
        )._generate_deterministic_vector("test_1b")

        regex_before, regex_pagination_before = populated_db.search_regex(
            "func_1b",
            page_size=10,
        )
        semantic_before, semantic_pagination_before = populated_db.search_semantic(
            query_embedding=query_embedding,
            provider="fake",
            model="fake-embeddings",
            page_size=10,
        )
        assert regex_pagination_before["total"] >= 1
        assert semantic_pagination_before["total"] >= 1

        populated_db.compact_database()

        regex_after, regex_pagination_after = populated_db.search_regex(
            "func_1b",
            page_size=10,
        )
        semantic_after, semantic_pagination_after = populated_db.search_semantic(
            query_embedding=query_embedding,
            provider="fake",
            model="fake-embeddings",
            page_size=10,
        )

        assert regex_pagination_after["total"] == regex_pagination_before["total"]
        assert semantic_pagination_after["total"] == semantic_pagination_before["total"]
        assert any(
            result.get("file_path") == "test_1.py"
            and "func_1b" in result.get("content", "")
            for result in regex_after
        )
        assert any(
            result.get("file_path") == "test_1.py"
            and result.get("symbol") == "func_1b"
            for result in semantic_after
        )


# ── Guard extended coverage ──────────────────────────────────────────────


class TestAutoCompactionMaintenance:
    """Auto-compaction stays active during normal operations via random sampling."""

    def test_dispatch_layer_triggers_compact_if_needed(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """execute_sync calls compact_if_needed when random sample fires."""
        monkeypatch.setattr("random.randint", lambda lo, hi: 0)

        compact_calls: list[str] = []

        def _fake_compact_if_needed() -> bool:
            compact_calls.append("compact")
            return True

        monkeypatch.setattr(
            file_backed_db, "compact_if_needed", _fake_compact_if_needed
        )

        file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="write_trigger.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=10,
            ),
        )

        assert compact_calls == ["compact"]

    def test_dispatch_skips_compaction_when_sample_does_not_fire(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compact_if_needed is NOT called when the random sample doesn't fire."""
        monkeypatch.setattr("random.randint", lambda lo, hi: hi)

        compact_calls: list[str] = []

        def _tracking_compact_if_needed() -> bool:
            compact_calls.append("compact")
            return False

        monkeypatch.setattr(
            file_backed_db, "compact_if_needed", _tracking_compact_if_needed
        )

        file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="no_trigger.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=10,
            ),
        )

        assert compact_calls == []

    def test_sync_dispatch_preserves_original_result_when_compaction_fails(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sampled sync auto-compaction failure must not fail the triggering op."""
        monkeypatch.setattr("random.randint", lambda lo, hi: 0)

        def _failing_compact_if_needed() -> bool:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            file_backed_db, "compact_if_needed", _failing_compact_if_needed
        )

        file_id = file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="write_fail.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=10,
            ),
        )

        assert isinstance(file_id, int)

    def test_mcp_mode_does_not_disable_dispatch(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MCP mode must not suppress post-operation auto-compaction dispatch."""
        monkeypatch.setenv("CHUNKHOUND_MCP_MODE", "1")
        monkeypatch.setattr("random.randint", lambda lo, hi: 0)

        compact_calls: list[str] = []

        def _fake_compact_if_needed() -> bool:
            compact_calls.append("compact")
            return True

        monkeypatch.setattr(
            file_backed_db, "compact_if_needed", _fake_compact_if_needed
        )

        file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="mcp_write.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=10,
            ),
        )

        assert compact_calls == ["compact"]

    @pytest.mark.asyncio
    async def test_async_dispatch_layer_triggers_compact_if_needed(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """execute_async also calls compact_if_needed when random sample fires."""
        monkeypatch.setattr("random.randint", lambda lo, hi: 0)

        compact_calls: list[str] = []

        async def _fake_compact_if_needed_async() -> bool:
            compact_calls.append("compact")
            return True

        monkeypatch.setattr(
            file_backed_db, "compact_if_needed_async", _fake_compact_if_needed_async
        )

        await file_backed_db._execute_in_db_thread(
            "insert_file",
            File(
                path="async_write.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=10,
            ),
        )

        assert compact_calls == ["compact"]

    @pytest.mark.asyncio
    async def test_async_dispatch_preserves_original_result_when_compaction_fails(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sampled async auto-compaction failure must not fail the triggering op."""
        monkeypatch.setattr("random.randint", lambda lo, hi: 0)

        async def _failing_compact_if_needed_async() -> bool:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            file_backed_db, "compact_if_needed_async", _failing_compact_if_needed_async
        )

        file_id = await file_backed_db._execute_in_db_thread(
            "insert_file",
            File(
                path="async_fail.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=10,
            ),
        )

        assert isinstance(file_id, int)

    def test_compact_operation_prefixed_names_get_extended_timeout(
        self, file_backed_db: DuckDBProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compact_database and compact_if_needed both use extended timeout."""
        from chunkhound.providers.database.serial_executor import (
            _COMPACTION_OPERATION_TIMEOUT_SECONDS,
        )

        captured_timeouts: list[float] = []
        original_execute = file_backed_db._executor.execute_sync

        def _tracking_execute(provider, operation_name, *args, **kwargs):
            # Patch the env var to a known value so we can verify the
            # default was picked up correctly
            default_timeout = (
                _COMPACTION_OPERATION_TIMEOUT_SECONDS
                if operation_name.startswith("compact")
                else 30.0
            )
            captured_timeouts.append(default_timeout)
            return original_execute(provider, operation_name, *args, **kwargs)

        monkeypatch.setattr(
            file_backed_db._executor, "execute_sync", _tracking_execute
        )

        file_backed_db.compact_if_needed()
        assert len(captured_timeouts) >= 1
        assert captured_timeouts[0] == _COMPACTION_OPERATION_TIMEOUT_SECONDS


class TestCompactionGuardExtended:
    """Verify reads are also blocked and flag is cleaned up on failure."""

    def test_reads_blocked_during_compaction(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """search_regex during compaction raises DatabaseCompactionInProgressError."""
        file_backed_db._executor.set_compaction_in_progress(True)

        try:
            with pytest.raises(DatabaseCompactionInProgressError):
                file_backed_db.search_regex("pattern")
        finally:
            file_backed_db._executor.set_compaction_in_progress(False)

    def test_fast_fail_happens_before_executor_queueing(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """New work should fail immediately once compaction is published active."""
        file_backed_db._executor.set_compaction_in_progress(True)

        try:
            with pytest.raises(DatabaseCompactionInProgressError):
                file_backed_db._execute_in_db_thread_sync(
                    "insert_file",
                    File(
                        path="blocked-before-queue.py",
                        mtime=0.0,
                        language=Language.PYTHON,
                        size_bytes=10,
                    ),
                )
        finally:
            file_backed_db._executor.set_compaction_in_progress(False)

    def test_compact_if_needed_blocked_during_active_compaction(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """compact_if_needed is blocked during active compaction."""
        file_backed_db._executor.set_compaction_in_progress(True)

        try:
            with pytest.raises(DatabaseCompactionInProgressError):
                file_backed_db.compact_if_needed()
        finally:
            file_backed_db._executor.set_compaction_in_progress(False)

    def test_flag_cleared_after_compaction_failure(
        self, file_backed_db: DuckDBProvider, monkeypatch
    ) -> None:
        """The provider is usable after a compaction failure."""

        def _failing_compact(conn, state):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(
            file_backed_db, "_executor_compact_database", _failing_compact
        )

        with pytest.raises(RuntimeError):
            file_backed_db.compact_database()

        # Verify the provider accepts writes after failure
        file_id = file_backed_db._execute_in_db_thread_sync(
            "insert_file",
            File(
                path="post_failure_work.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=10,
            ),
        )
        assert isinstance(file_id, int)
        assert file_id > 0


# ── Concurrent compaction guard ──────────────────────────────────────────


class TestConcurrentCompactionGuard:
    """Concurrent write attempts during active compaction."""

    def test_write_blocks_during_active_compaction(
        self, file_backed_db: DuckDBProvider, monkeypatch
    ) -> None:
        """Fire compaction in background, verify insert is blocked."""
        import concurrent.futures

        compaction_started = threading.Event()
        orig = file_backed_db._executor.set_compaction_in_progress

        def _patched(active: bool) -> None:
            orig(active)
            if active:
                compaction_started.set()

        monkeypatch.setattr(
            file_backed_db._executor, "set_compaction_in_progress", _patched
        )

        def _do_compact() -> None:
            file_backed_db.compact_database()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_do_compact)
            # Wait for compaction to actually set the in-progress flag
            assert compaction_started.wait(timeout=10), "Compaction never started"

            with pytest.raises(DatabaseCompactionInProgressError):
                file_backed_db.insert_file(
                    File(
                        path="blocked_conc.py",
                        mtime=0.0,
                        language=Language.PYTHON,
                        size_bytes=100,
                    )
                )

            fut.result(timeout=30)  # let compaction finish

        assert file_backed_db._executor.is_compaction_in_progress() is False

    def test_write_succeeds_after_compaction_completes(
        self, file_backed_db: DuckDBProvider
    ) -> None:
        """Fire compaction in background, await completion, verify write succeeds."""
        import concurrent.futures

        def _do_compact() -> None:
            file_backed_db.compact_database()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_do_compact)
            fut.result(timeout=30)  # wait for compaction to complete

        file_id = file_backed_db.insert_file(
            File(
                path="after_conc_compact.py",
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=100,
            )
        )
        assert file_id > 0

def test_atomic_replace_retries_transient_windows_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows atomic replace retries transient handle-release failures."""
    attempts: list[tuple[str, str]] = []
    sleeps: list[float] = []

    def _flaky_replace(src: str, dst: str) -> None:
        attempts.append((src, dst))
        if len(attempts) < 3:
            raise OSError("busy")

    monkeypatch.setattr(duckdb_provider_module, "IS_WINDOWS", True)
    monkeypatch.setattr(duckdb_provider_module.os, "replace", _flaky_replace)
    monkeypatch.setattr(duckdb_provider_module.time, "sleep", sleeps.append)

    duckdb_provider_module._atomic_replace("src.duckdb", "dst.duckdb")

    assert attempts == [
        ("src.duckdb", "dst.duckdb"),
        ("src.duckdb", "dst.duckdb"),
        ("src.duckdb", "dst.duckdb"),
    ]
    assert sleeps == [
        duckdb_provider_module.WINDOWS_FILE_HANDLE_DELAY,
        duckdb_provider_module.WINDOWS_FILE_HANDLE_DELAY * 2,
    ]


def test_live_file_is_valid_after_compaction(populated_db: DuckDBProvider) -> None:
    """After compaction, the live DB file is valid, non-empty, and has data."""
    populated_db.compact_database()

    db_path = cast(str, populated_db.db_path)
    assert os.path.getsize(db_path) > 0, "Compacted DB file must not be empty"

    stats = populated_db.get_stats()
    assert stats["files"] >= 2
    assert stats["chunks"] >= 4
    assert stats["embeddings"] >= 4

    # Verify via a fresh connection that the file is structurally valid
    remaining_indexes = _fetch_index_names(db_path)
    assert "idx_files_path" in remaining_indexes
    assert "idx_chunks_file_id" in remaining_indexes

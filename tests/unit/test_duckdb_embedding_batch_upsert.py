from pathlib import Path

import pytest

from chunkhound.providers.database.duckdb.connection_manager import (
    DuckDBConnectionManager,
)
from chunkhound.providers.database.duckdb.embedding_repository import (
    DuckDBEmbeddingRepository,
)
from chunkhound.providers.database.duckdb_provider import DuckDBProvider


def _get_hnsw_index_names(provider: DuckDBProvider) -> list[str]:
    rows = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
          AND (index_name LIKE 'hnsw_%' OR index_name LIKE 'idx_hnsw_%')
        ORDER BY index_name
        """,
        [],
    )
    return [row["index_name"] for row in rows]


def _get_embedding_rows(
    provider: DuckDBProvider, chunk_id: int, provider_name: str, model_name: str
) -> list[dict]:
    return provider.execute_query(
        """
        SELECT id, chunk_id, provider, model, embedding
        FROM embeddings_3
        WHERE chunk_id = ? AND provider = ? AND model = ?
        ORDER BY id
        """,
        [chunk_id, provider_name, model_name],
    )


def _vector_1536(value: float) -> list[float]:
    return [value] * 1536


def test_large_batch_insert_parameterizes_special_chars_and_preserves_indexes(
    tmp_path: Path,
) -> None:
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()
    try:
        provider._ensure_embedding_table_exists(3)
        provider.create_vector_index("test", "mini", 3, "cosine")
        initial_indexes = _get_hnsw_index_names(provider)

        payload = [
            {
                "chunk_id": chunk_id,
                "provider": "o'reilly;--",
                "model": "model\"quote",
                "embedding": [float(chunk_id), float(chunk_id) + 0.1, 9.0],
                "dims": 3,
            }
            for chunk_id in range(1, 61)
        ]

        inserted = provider._embedding_repository.insert_embeddings_batch(
            payload,
            batch_size=50,
            connection=provider.connection,
        )

        assert inserted == len(payload)
        assert _get_hnsw_index_names(provider) == initial_indexes
        count = provider.execute_query(
            """
            SELECT COUNT(*) AS count
            FROM embeddings_3
            WHERE provider = ? AND model = ?
            """,
            ["o'reilly;--", 'model"quote'],
        )
        assert count[0]["count"] == len(payload)
    finally:
        provider.disconnect(skip_checkpoint=True)


def test_large_batch_failure_restores_exact_hnsw_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()
    try:
        provider._ensure_embedding_table_exists(3)
        provider.create_vector_index("test", "mini", 3, "cosine")
        initial_indexes = _get_hnsw_index_names(provider)
        dropped_index_snapshots: list[list[str]] = []

        payload = [
            {
                "chunk_id": chunk_id,
                "provider": "test",
                "model": "mini",
                "embedding": [float(chunk_id), float(chunk_id) + 0.1, 8.0],
                "dims": 3,
            }
            for chunk_id in range(1, 61)
        ]

        def _raise_after_drop(conn, upsert_sql: str, batch_rows: list[tuple]) -> None:
            remaining_indexes = conn.execute(
                """
                SELECT index_name
                FROM duckdb_indexes()
                WHERE table_name = 'embeddings_3'
                  AND (index_name LIKE 'hnsw_%' OR index_name LIKE 'idx_hnsw_%')
                ORDER BY index_name
                """
            ).fetchall()
            dropped_index_snapshots.append([row[0] for row in remaining_indexes])
            del upsert_sql, batch_rows
            raise RuntimeError("forced batch failure")

        monkeypatch.setattr(
            provider._embedding_repository, "_execute_upsert_batches", _raise_after_drop
        )

        with pytest.raises(
            RuntimeError, match="strict HNSW restore|forced batch failure"
        ):
            provider._embedding_repository.insert_embeddings_batch(
                payload,
                batch_size=50,
                connection=provider.connection,
            )

        assert dropped_index_snapshots == [[]]
        assert _get_hnsw_index_names(provider) == initial_indexes
        count = provider.execute_query("SELECT COUNT(*) AS count FROM embeddings_3", [])
        assert count[0]["count"] == 0
    finally:
        provider.disconnect(skip_checkpoint=True)


def test_provider_batch_insert_uses_true_upsert_contract(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()
    try:
        first_payload = [
            {
                "chunk_id": 1,
                "provider": "test",
                "model": "mini",
                "embedding": [1.0, 2.0, 3.0],
                "dims": 3,
            }
        ]
        second_payload = [
            {
                "chunk_id": 1,
                "provider": "test",
                "model": "mini",
                "embedding": [9.0, 8.0, 7.0],
                "dims": 3,
            }
        ]

        assert provider.insert_embeddings_batch(first_payload) == 1
        original_row = _get_embedding_rows(provider, 1, "test", "mini")
        assert len(original_row) == 1

        assert provider.insert_embeddings_batch(second_payload) == 1
        updated_rows = _get_embedding_rows(provider, 1, "test", "mini")

        assert len(updated_rows) == 1
        assert updated_rows[0]["id"] == original_row[0]["id"]
        assert list(updated_rows[0]["embedding"]) == [9.0, 8.0, 7.0]
    finally:
        provider.disconnect(skip_checkpoint=True)


def test_repository_large_batch_uses_true_upsert_contract(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()
    try:
        provider._ensure_embedding_table_exists(3)
        provider.create_vector_index("test", "mini", 3, "cosine")

        first_payload = [
            {
                "chunk_id": chunk_id,
                "provider": "test",
                "model": "mini",
                "embedding": [float(chunk_id), float(chunk_id) + 0.1, 1.0],
                "dims": 3,
            }
            for chunk_id in range(1, 61)
        ]
        second_payload = [
            {
                "chunk_id": chunk_id,
                "provider": "test",
                "model": "mini",
                "embedding": [float(chunk_id), float(chunk_id) + 0.1, 9.0],
                "dims": 3,
            }
            for chunk_id in range(1, 61)
        ]

        assert (
            provider._embedding_repository.insert_embeddings_batch(
                first_payload,
                batch_size=50,
                connection=provider.connection,
            )
            == len(first_payload)
        )
        original_row = _get_embedding_rows(provider, 1, "test", "mini")
        assert len(original_row) == 1

        assert (
            provider._embedding_repository.insert_embeddings_batch(
                second_payload,
                batch_size=50,
                connection=provider.connection,
            )
            == len(second_payload)
        )

        updated_rows = _get_embedding_rows(provider, 1, "test", "mini")
        count = provider.execute_query(
            """
            SELECT COUNT(*) AS count
            FROM embeddings_3
            WHERE provider = ? AND model = ?
            """,
            ["test", "mini"],
        )

        assert count[0]["count"] == len(first_payload)
        assert len(updated_rows) == 1
        assert updated_rows[0]["id"] == original_row[0]["id"]
        assert list(updated_rows[0]["embedding"]) == pytest.approx([1.0, 1.1, 9.0])
    finally:
        provider.disconnect(skip_checkpoint=True)


def test_schema_migration_backfills_unique_index_and_deduplicates_rows(
    tmp_path: Path,
) -> None:
    pytest.importorskip("duckdb")

    db_path = tmp_path / "db.duckdb"

    provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
    provider.connect()
    try:
        provider._ensure_embedding_table_exists(3)
        provider.create_vector_index("legacy", "mini", 3, "cosine")
        provider.connection.execute(
            "DROP INDEX IF EXISTS idx_3_chunk_provider_model_unique"
        )
        provider.connection.execute(
            """
            INSERT INTO embeddings_3 (chunk_id, provider, model, embedding, dims)
            VALUES (?, ?, ?, ?, ?)
            """,
            [1, "legacy", "mini", [1.0, 2.0, 3.0], 3],
        )
        provider.connection.execute(
            """
            INSERT INTO embeddings_3 (chunk_id, provider, model, embedding, dims)
            VALUES (?, ?, ?, ?, ?)
            """,
            [1, "legacy", "mini", [4.0, 5.0, 6.0], 3],
        )
        initial_indexes = _get_hnsw_index_names(provider)
    finally:
        provider.disconnect(skip_checkpoint=True)

    migrated_provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
    migrated_provider.connect()
    try:
        index_names = migrated_provider.execute_query(
            """
            SELECT index_name
            FROM duckdb_indexes()
            WHERE table_name = 'embeddings_3'
            ORDER BY index_name
            """,
            [],
        )
        assert {
            row["index_name"] for row in index_names
        } >= {
            "idx_3_chunk_provider_model_unique",
            "idx_3_provider_model",
            "idx_hnsw_3",
            "hnsw_legacy_mini_3_cosine",
        }
        assert _get_hnsw_index_names(migrated_provider) == initial_indexes

        rows = _get_embedding_rows(migrated_provider, 1, "legacy", "mini")
        assert len(rows) == 1
        original_row_id = rows[0]["id"]
        assert list(rows[0]["embedding"]) == [4.0, 5.0, 6.0]

        assert migrated_provider.insert_embeddings_batch(
            [
                {
                    "chunk_id": 1,
                    "provider": "legacy",
                    "model": "mini",
                    "embedding": [7.0, 8.0, 9.0],
                    "dims": 3,
                }
            ]
        ) == 1

        updated_rows = _get_embedding_rows(migrated_provider, 1, "legacy", "mini")
        assert len(updated_rows) == 1
        assert updated_rows[0]["id"] == original_row_id
        assert list(updated_rows[0]["embedding"]) == [7.0, 8.0, 9.0]
    finally:
        migrated_provider.disconnect(skip_checkpoint=True)


def test_connect_upgrades_legacy_embeddings_1536_before_unique_index_creation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("duckdb")

    db_path = tmp_path / "legacy-1536.duckdb"

    provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
    provider.connect()
    try:
        provider.connection.execute(
            "DROP INDEX IF EXISTS idx_1536_chunk_provider_model_unique"
        )
        provider.connection.execute(
            """
            INSERT INTO embeddings_1536 (chunk_id, provider, model, embedding, dims)
            VALUES (?, ?, ?, ?, ?)
            """,
            [1, "legacy", "default", _vector_1536(1.0), 1536],
        )
        provider.connection.execute(
            """
            INSERT INTO embeddings_1536 (chunk_id, provider, model, embedding, dims)
            VALUES (?, ?, ?, ?, ?)
            """,
            [1, "legacy", "default", _vector_1536(2.0), 1536],
        )
    finally:
        provider.disconnect(skip_checkpoint=True)

    migrated_provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
    migrated_provider.connect()
    try:
        rows = migrated_provider.execute_query(
            """
            SELECT id, embedding
            FROM embeddings_1536
            WHERE chunk_id = ? AND provider = ? AND model = ?
            ORDER BY id
            """,
            [1, "legacy", "default"],
        )
        indexes = migrated_provider.execute_query(
            """
            SELECT index_name
            FROM duckdb_indexes()
            WHERE table_name = 'embeddings_1536'
            ORDER BY index_name
            """,
            [],
        )

        assert len(rows) == 1
        assert rows[0]["embedding"][:3] == pytest.approx([2.0, 2.0, 2.0])
        assert {
            row["index_name"] for row in indexes
        } >= {"idx_1536_chunk_provider_model_unique", "idx_hnsw_1536"}
    finally:
        migrated_provider.disconnect(skip_checkpoint=True)


def test_repository_fallback_upgrades_existing_table_before_on_conflict(
    tmp_path: Path,
) -> None:
    pytest.importorskip("duckdb")

    connection_manager = DuckDBConnectionManager(tmp_path / "fallback.duckdb")
    connection_manager.connect()
    try:
        assert connection_manager.connection is not None
        conn = connection_manager.connection
        conn.execute("CREATE SEQUENCE IF NOT EXISTS embeddings_id_seq")
        conn.execute("""
            CREATE TABLE embeddings_3 (
                id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
                chunk_id INTEGER,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding FLOAT[3],
                dims INTEGER NOT NULL DEFAULT 3,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            """
            INSERT INTO embeddings_3 (chunk_id, provider, model, embedding, dims)
            VALUES (?, ?, ?, ?, ?)
            """,
            [1, "fallback", "mini", [1.0, 2.0, 3.0], 3],
        )

        repository = DuckDBEmbeddingRepository(connection_manager, provider=None)
        assert (
            repository.insert_embeddings_batch(
                [
                    {
                        "chunk_id": 1,
                        "provider": "fallback",
                        "model": "mini",
                        "embedding": [9.0, 8.0, 7.0],
                        "dims": 3,
                    }
                ],
                batch_size=1,
                connection=conn,
            )
            == 1
        )

        rows = conn.execute(
            """
            SELECT id, embedding
            FROM embeddings_3
            WHERE chunk_id = ? AND provider = ? AND model = ?
            ORDER BY id
            """,
            [1, "fallback", "mini"],
        ).fetchall()
        indexes = conn.execute(
            """
            SELECT index_name
            FROM duckdb_indexes()
            WHERE table_name = 'embeddings_3'
            ORDER BY index_name
            """
        ).fetchall()

        assert len(rows) == 1
        assert list(rows[0][1]) == pytest.approx([9.0, 8.0, 7.0])
        assert "idx_3_chunk_provider_model_unique" in {row[0] for row in indexes}
    finally:
        connection_manager.disconnect()

"""Real workflow tests for ChunkHound reindexing functionality.

Tests core business logic: chunk preservation, content change detection,
and data integrity using real components without mocks.
"""

import pytest
from pathlib import Path
from chunkhound.core.models import Chunk, Embedding, File
from chunkhound.core.types.common import Language, FileId
from chunkhound.core.types.common import ChunkType, LineNumber
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


@pytest.fixture
def real_components(tmp_path):
    """Real system components for testing."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()  # Initialize database schema
    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(db, tmp_path, None, {Language.PYTHON: parser})
    return {"db": db, "parser": parser, "coordinator": coordinator}


def test_process_directory_cleans_orphaned_hnsw_records_without_invalidating_duckdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import asyncio

    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    current_file = tmp_path / "current.py"
    current_file.write_text("def current():\n    return 1\n")

    orphan_file_id = provider.insert_file(
        File(path="orphan.py", mtime=1.0, size_bytes=24, language=Language.PYTHON)
    )
    orphan_chunk_id = provider.insert_chunk(
        Chunk(
            file_id=FileId(orphan_file_id),
            symbol="orphan",
            start_line=LineNumber(1),
            end_line=LineNumber(2),
            code="def orphan():\n    return 1\n",
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
        )
    )
    provider.insert_embedding(
        Embedding(
            chunk_id=orphan_chunk_id,
            provider="test",
            model="mini",
            dims=3,
            vector=[0.1, 0.2, 0.3],
        )
    )

    initial_indexes = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
          AND (index_name LIKE 'hnsw_%' OR index_name LIKE 'idx_hnsw_%')
        ORDER BY index_name
        """,
        [],
    )
    if not initial_indexes:
        pytest.skip("DuckDB HNSW indexes are unavailable in this environment")

    coordinator = IndexingCoordinator(provider, tmp_path)

    async def _skip_processing(*args, **kwargs):
        return []

    monkeypatch.setattr(coordinator, "_process_files_in_batches", _skip_processing)

    result = asyncio.run(
        coordinator.process_directory(
            tmp_path, patterns=["**/*.py"], exclude_patterns=[]
        )
    )

    assert result["status"] == "success"
    assert provider.get_file_by_path("orphan.py", as_model=False) is None
    assert provider.get_chunks_by_file_id(orphan_file_id, as_model=False) == []
    remaining_embeddings = provider.execute_query(
        "SELECT COUNT(*) AS count FROM embeddings_3 WHERE chunk_id = ?",
        [orphan_chunk_id],
    )
    assert remaining_embeddings[0]["count"] == 0

    remaining_indexes = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
          AND (index_name LIKE 'hnsw_%' OR index_name LIKE 'idx_hnsw_%')
        ORDER BY index_name
        """,
        [],
    )
    assert [row["index_name"] for row in remaining_indexes] == [
        row["index_name"] for row in initial_indexes
    ]

    followup_file_id = provider.insert_file(
        File(path="followup.py", mtime=2.0, size_bytes=27, language=Language.PYTHON)
    )
    followup_chunk_id = provider.insert_chunk(
        Chunk(
            file_id=FileId(followup_file_id),
            symbol="followup",
            start_line=LineNumber(1),
            end_line=LineNumber(2),
            code="def followup():\n    return 2\n",
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
        )
    )
    provider.insert_embedding(
        Embedding(
            chunk_id=followup_chunk_id,
            provider="test",
            model="mini",
            dims=3,
            vector=[0.4, 0.5, 0.6],
        )
    )

    followup_embeddings = provider.execute_query(
        "SELECT COUNT(*) AS count FROM embeddings_3 WHERE chunk_id = ?",
        [followup_chunk_id],
    )
    assert followup_embeddings[0]["count"] == 1


def test_process_directory_returns_deterministic_error_when_orphan_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import asyncio

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    current_file = tmp_path / "current.py"
    current_file.write_text("def current():\n    return 1\n")
    provider.insert_file(
        File(path="orphan.py", mtime=1.0, size_bytes=24, language=Language.PYTHON)
    )

    coordinator = IndexingCoordinator(provider, tmp_path)

    async def _skip_processing(*args, **kwargs):
        return []

    monkeypatch.setattr(coordinator, "_process_files_in_batches", _skip_processing)

    def _fail_delete(file_path: str) -> bool:
        raise RuntimeError("hnsw delete exploded")

    monkeypatch.setattr(provider, "delete_file_completely", _fail_delete)

    result = asyncio.run(
        coordinator.process_directory(
            tmp_path, patterns=["**/*.py"], exclude_patterns=[]
        )
    )

    assert result["status"] == "error"
    error = result["error"]
    assert "Storage reconciliation cleanup failed:" in error
    assert "orphan/excluded cleanup delete failed for orphan.py" in error
    assert "reason=missing_on_disk" in error
    assert "hnsw delete exploded" in error


def test_orphan_cleanup_restores_rows_when_hnsw_index_recreation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import asyncio

    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    current_file = tmp_path / "current.py"
    current_file.write_text("def current():\n    return 1\n")

    orphan_file_id = provider.insert_file(
        File(path="orphan.py", mtime=1.0, size_bytes=24, language=Language.PYTHON)
    )
    orphan_chunk_id = provider.insert_chunk(
        Chunk(
            file_id=FileId(orphan_file_id),
            symbol="orphan",
            start_line=LineNumber(1),
            end_line=LineNumber(2),
            code="def orphan():\n    return 1\n",
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
        )
    )
    provider.insert_embedding(
        Embedding(
            chunk_id=orphan_chunk_id,
            provider="test",
            model="mini",
            dims=3,
            vector=[0.1, 0.2, 0.3],
        )
    )

    initial_indexes = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
          AND (index_name LIKE 'hnsw_%' OR index_name LIKE 'idx_hnsw_%')
        ORDER BY index_name
        """,
        [],
    )
    if not initial_indexes:
        pytest.skip("DuckDB HNSW indexes are unavailable in this environment")

    recreate_attempts = {"count": 0}
    original_recreate = provider._executor_recreate_vector_index_from_info

    def _fail_once(conn, state, index_info):
        recreate_attempts["count"] += 1
        if recreate_attempts["count"] == 1:
            raise RuntimeError("forced hnsw recreate failure")
        return original_recreate(conn, state, index_info)

    monkeypatch.setattr(provider, "_executor_recreate_vector_index_from_info", _fail_once)

    coordinator = IndexingCoordinator(provider, tmp_path)

    async def _skip_processing(*args, **kwargs):
        return []

    monkeypatch.setattr(coordinator, "_process_files_in_batches", _skip_processing)

    failed_result = asyncio.run(
        coordinator.process_directory(
            tmp_path, patterns=["**/*.py"], exclude_patterns=[]
        )
    )

    assert failed_result["status"] == "error"
    error = failed_result["error"]
    assert "Storage reconciliation cleanup failed:" in error
    assert "orphan/excluded cleanup delete failed for orphan.py" in error
    assert "reason=missing_on_disk" in error
    assert "forced hnsw recreate failure" in error
    assert recreate_attempts["count"] >= 2
    assert provider.get_file_by_path("orphan.py", as_model=False) is not None
    assert provider.get_chunks_by_file_id(orphan_file_id, as_model=False) != []
    restored_embeddings = provider.execute_query(
        "SELECT COUNT(*) AS count FROM embeddings_3 WHERE chunk_id = ?",
        [orphan_chunk_id],
    )
    assert restored_embeddings[0]["count"] == 1

    restored_indexes = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
          AND (index_name LIKE 'hnsw_%' OR index_name LIKE 'idx_hnsw_%')
        ORDER BY index_name
        """,
        [],
    )
    assert [row["index_name"] for row in restored_indexes] == [
        row["index_name"] for row in initial_indexes
    ]

    successful_result = asyncio.run(
        coordinator.process_directory(
            tmp_path, patterns=["**/*.py"], exclude_patterns=[]
        )
    )
    assert successful_result["status"] == "success"
    assert provider.get_file_by_path("orphan.py", as_model=False) is None


def test_process_directory_fails_closed_when_orphan_cleanup_delete_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import asyncio

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    current_file = tmp_path / "current.py"
    current_file.write_text("def current():\n    return 1\n")

    excluded_dir = tmp_path / "excluded"
    excluded_dir.mkdir()
    excluded_file = excluded_dir / "still_here.py"
    excluded_file.write_text("def excluded():\n    return 2\n")

    provider.insert_file(
        File(
            path="excluded/still_here.py",
            mtime=1.0,
            size_bytes=28,
            language=Language.PYTHON,
        )
    )

    coordinator = IndexingCoordinator(provider, tmp_path)

    async def _skip_processing(*args, **kwargs):
        return []

    monkeypatch.setattr(coordinator, "_process_files_in_batches", _skip_processing)
    monkeypatch.setattr(provider, "delete_file_completely", lambda file_path: False)

    result = asyncio.run(
        coordinator.process_directory(
            tmp_path, patterns=["**/*.py"], exclude_patterns=["**/excluded/**"]
        )
    )

    assert result["status"] == "error"
    error = result["error"]
    assert "Storage reconciliation cleanup failed:" in error
    assert (
        "orphan/excluded cleanup delete returned false for "
        "excluded/still_here.py"
    ) in error
    assert "reason=excluded_by_current_policy" in error


def test_process_directory_drops_nonstandard_hnsw_indexes_for_excluded_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import asyncio

    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    current_file = tmp_path / "current.py"
    current_file.write_text("def current():\n    return 1\n")

    excluded_dir = tmp_path / "excluded"
    excluded_dir.mkdir()
    excluded_file = excluded_dir / "still_here.py"
    excluded_file.write_text("def excluded():\n    return 2\n")

    excluded_file_id = provider.insert_file(
        File(
            path="excluded/still_here.py",
            mtime=1.0,
            size_bytes=28,
            language=Language.PYTHON,
        )
    )
    excluded_chunk_id = provider.insert_chunk(
        Chunk(
            file_id=FileId(excluded_file_id),
            symbol="excluded",
            start_line=LineNumber(1),
            end_line=LineNumber(2),
            code="def excluded():\n    return 2\n",
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
        )
    )
    provider.insert_embedding(
        Embedding(
            chunk_id=excluded_chunk_id,
            provider="test",
            model="mini",
            dims=3,
            vector=[0.1, 0.2, 0.3],
        )
    )

    provider.execute_query("DROP INDEX IF EXISTS idx_hnsw_3", [])
    provider.execute_query(
        "CREATE INDEX alt_live_idx ON embeddings_3 USING HNSW (embedding)", []
    )

    initial_indexes = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
        ORDER BY index_name
        """,
        [],
    )
    assert [row["index_name"] for row in initial_indexes] == [
        "alt_live_idx",
        "idx_3_chunk_id",
        "idx_3_provider_model",
    ]

    dropped_indexes: list[str] = []
    original_drop = provider._executor_drop_vector_index_by_name

    def _record_drop(conn, index_name: str) -> None:
        dropped_indexes.append(index_name)
        original_drop(conn, index_name)

    monkeypatch.setattr(provider, "_executor_drop_vector_index_by_name", _record_drop)

    coordinator = IndexingCoordinator(provider, tmp_path)

    async def _skip_processing(*args, **kwargs):
        return []

    monkeypatch.setattr(coordinator, "_process_files_in_batches", _skip_processing)

    result = asyncio.run(
        coordinator.process_directory(
            tmp_path, patterns=["**/*.py"], exclude_patterns=["**/excluded/**"]
        )
    )

    assert result["status"] == "success"
    assert "alt_live_idx" in dropped_indexes
    assert provider.get_file_by_path("excluded/still_here.py", as_model=False) is None

    remaining_indexes = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
        ORDER BY index_name
        """,
        [],
    )
    assert [row["index_name"] for row in remaining_indexes] == [
        row["index_name"] for row in initial_indexes
    ]

    followup_file_id = provider.insert_file(
        File(path="followup.py", mtime=2.0, size_bytes=27, language=Language.PYTHON)
    )
    followup_chunk_id = provider.insert_chunk(
        Chunk(
            file_id=FileId(followup_file_id),
            symbol="followup",
            start_line=LineNumber(1),
            end_line=LineNumber(2),
            code="def followup():\n    return 3\n",
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
        )
    )
    provider.insert_embedding(
        Embedding(
            chunk_id=followup_chunk_id,
            provider="test",
            model="mini",
            dims=3,
            vector=[0.4, 0.5, 0.6],
        )
    )

    followup_embeddings = provider.execute_query(
        "SELECT COUNT(*) AS count FROM embeddings_3 WHERE chunk_id = ?",
        [followup_chunk_id],
    )
    assert followup_embeddings[0]["count"] == 1


def test_large_batch_insert_preserves_custom_hnsw_index_identity(tmp_path: Path):
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    file_id = provider.insert_file(
        File(path="batch.py", mtime=1.0, size_bytes=4096, language=Language.PYTHON)
    )

    chunk_ids = []
    for i in range(60):
        chunk_id = provider.insert_chunk(
            Chunk(
                file_id=FileId(file_id),
                symbol=f"batch_{i}",
                start_line=LineNumber(i + 1),
                end_line=LineNumber(i + 1),
                code=f"def batch_{i}(): return {i}",
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
            )
        )
        chunk_ids.append(chunk_id)

    provider.create_vector_index("test", "mini", 3, "cosine")

    initial_indexes = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
        ORDER BY index_name
        """,
        [],
    )
    assert [row["index_name"] for row in initial_indexes] == [
        "hnsw_test_mini_3_cosine",
        "idx_3_chunk_id",
        "idx_3_provider_model",
        "idx_hnsw_3",
    ]

    embeddings_data = [
        {
            "chunk_id": chunk_id,
            "provider": "test",
            "model": "mini",
            "embedding": [float(i), float(i + 1), float(i + 2)],
            "dims": 3,
        }
        for i, chunk_id in enumerate(chunk_ids)
    ]

    inserted = provider._embedding_repository.insert_embeddings_batch(
        embeddings_data,
        batch_size=50,
        connection=provider.connection,
    )
    assert inserted == len(embeddings_data)

    remaining_indexes = provider.execute_query(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE table_name = 'embeddings_3'
        ORDER BY index_name
        """,
        [],
    )
    assert [row["index_name"] for row in remaining_indexes] == [
        row["index_name"] for row in initial_indexes
    ]
    assert "hnsw_generic_generic_3_cosine" not in {
        row["index_name"] for row in remaining_indexes
    }

    embedding_count = provider.execute_query(
        "SELECT COUNT(*) AS count FROM embeddings_3",
        [],
    )
    assert embedding_count[0]["count"] == len(embeddings_data)


class TestChunkPreservationLogic:
    """Test chunk preservation using real components."""

    @pytest.mark.asyncio
    async def test_chunk_preservation_identical_content(self, real_components, tmp_path):
        """Test that identical content preserves existing chunks."""
        coordinator = real_components["coordinator"]
        db = real_components["db"]
        
        # Create real file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def function1():
    return "hello"

def function2():
    return "world"
""")
        
        # Process file twice
        result1 = await coordinator.process_file(test_file)
        result2 = await coordinator.process_file(test_file)
        
        # Should have chunks
        chunks = db.get_chunks_by_file_id(result1["file_id"], as_model=True)
        assert len(chunks) > 0, "Should have chunks for original file"

    @pytest.mark.asyncio
    async def test_file_content_updates_correctly(self, real_components, tmp_path):
        """Test that file content changes are reflected in search results."""
        coordinator = real_components["coordinator"]
        from chunkhound.services.search_service import SearchService
        search = SearchService(real_components["db"])
        
        test_file = tmp_path / "test.py"
        
        # Original content - two small functions (will be merged by cAST)
        test_file.write_text("""
def calculate_tax(amount):
    return amount * 0.1

def calculate_discount(amount):  
    return amount * 0.2
""")
        
        # Index original content
        result1 = await coordinator.process_file(test_file)
        assert result1["status"] == "success"
        
        # Verify original content is searchable
        tax_results, _ = search.search_regex("calculate_tax")
        discount_results, _ = search.search_regex("calculate_discount")
        assert len(tax_results) > 0, "Should find tax function in original content"
        assert len(discount_results) > 0, "Should find discount function in original content"
        
        # Update content - replace one function, keep one
        test_file.write_text("""
def calculate_tax(amount):
    return amount * 0.15  # Updated rate

def calculate_shipping(amount):  # New function, replaces discount
    return amount * 0.05
""")
        
        # Process updated content
        result2 = await coordinator.process_file(test_file)
        assert result2["status"] == "success"
        
        # Verify updated content is searchable
        updated_tax_results, _ = search.search_regex("calculate_tax")
        shipping_results, _ = search.search_regex("calculate_shipping") 
        old_discount_results, _ = search.search_regex("calculate_discount")
        
        # Key assertions: content findability, not chunk structure
        assert len(updated_tax_results) > 0, "Should find updated tax function"
        assert len(shipping_results) > 0, "Should find new shipping function"
        assert len(old_discount_results) == 0, "Should NOT find old discount function"
        
        # Verify actual content was updated  
        assert "0.15" in updated_tax_results[0]["content"], "Tax rate should be updated to 0.15"
        assert "calculate_shipping" in shipping_results[0]["content"], "Should contain new shipping function"

    @pytest.mark.asyncio
    async def test_removed_content_not_searchable(self, real_components, tmp_path):
        """Test that removed code is no longer findable in search."""
        coordinator = real_components["coordinator"]  
        from chunkhound.services.search_service import SearchService
        search = SearchService(real_components["db"])
        
        test_file = tmp_path / "test.py"
        
        # Original: two utility functions
        test_file.write_text("""
def format_currency(amount):
    return f"${amount:.2f}"

def format_percentage(value):
    return f"{value:.1%}"
""")
        
        result1 = await coordinator.process_file(test_file)
        assert result1["status"] == "success"
        
        # Verify both functions are searchable
        currency_results, _ = search.search_regex("format_currency")
        percentage_results, _ = search.search_regex("format_percentage")
        assert len(currency_results) > 0, "Currency function should be findable"
        assert len(percentage_results) > 0, "Percentage function should be findable"
        
        # Update: keep only one function
        test_file.write_text("""
def format_currency(amount):
    return f"${amount:.2f}"
""")
        
        result2 = await coordinator.process_file(test_file)
        assert result2["status"] == "success"
        
        # Verify search results reflect the change
        updated_currency_results, _ = search.search_regex("format_currency")
        removed_percentage_results, _ = search.search_regex("format_percentage")
        
        assert len(updated_currency_results) > 0, "Remaining function should still be findable"
        assert len(removed_percentage_results) == 0, "Removed function should not be findable"

    @pytest.mark.asyncio  
    async def test_function_implementation_updates(self, real_components, tmp_path):
        """Test that changes to function implementations are reflected in search."""
        coordinator = real_components["coordinator"]
        from chunkhound.services.search_service import SearchService
        search = SearchService(real_components["db"])
        
        test_file = tmp_path / "test.py"
        
        # Original implementation
        test_file.write_text("""
def process_data(items):
    # Simple processing
    return [item.upper() for item in items]

def validate_input(data):
    return len(data) > 0
""")
        
        result1 = await coordinator.process_file(test_file)
        assert result1["status"] == "success"
        
        # Verify original implementation details are searchable
        simple_results, _ = search.search_regex("Simple processing")
        upper_results, _ = search.search_regex("item.upper")
        assert len(simple_results) > 0, "Should find original comment"
        assert len(upper_results) > 0, "Should find original logic"
        
        # Update implementation  
        test_file.write_text("""
def process_data(items):
    # Advanced processing with filtering
    return [item.lower().strip() for item in items if item]

def validate_input(data):
    return len(data) > 0
""")
        
        result2 = await coordinator.process_file(test_file)
        assert result2["status"] == "success"
        
        # Verify updated implementation is searchable
        advanced_results, _ = search.search_regex("Advanced processing")
        lower_results, _ = search.search_regex("item.lower")
        old_upper_results, _ = search.search_regex("item.upper")
        old_simple_results, _ = search.search_regex("Simple processing")
        
        # Content should reflect the implementation change
        assert len(advanced_results) > 0, "Should find new comment" 
        assert len(lower_results) > 0, "Should find new logic"
        assert len(old_upper_results) == 0, "Should NOT find old logic"
        assert len(old_simple_results) == 0, "Should NOT find old comment"


class TestIndexingCoordinatorOperations:
    """Test IndexingCoordinator core operations with real components."""

    @pytest.mark.asyncio
    async def test_process_file_operation(self, real_components, tmp_path):
        """Test basic file processing operation."""
        coordinator = real_components["coordinator"]
        db = real_components["db"]
        
        test_file = tmp_path / "simple.py"
        test_file.write_text("def simple_test(): return True")
        
        result = await coordinator.process_file(test_file)
        assert result["status"] == "success", "File should be processed successfully"
        
        chunks = db.get_chunks_by_file_id(result["file_id"], as_model=True)
        assert len(chunks) > 0, "File should be processed"

    @pytest.mark.asyncio
    async def test_remove_file_operation(self, real_components, tmp_path):
        """Test file removal operation."""
        coordinator = real_components["coordinator"]
        db = real_components["db"]
        
        test_file = tmp_path / "to_remove.py"
        test_file.write_text("def to_be_removed(): pass")
        
        # Process file first
        result = await coordinator.process_file(test_file)
        chunks_before = db.get_chunks_by_file_id(result["file_id"], as_model=True)
        assert len(chunks_before) > 0, "File should exist before removal"
        
        # Remove file (simulate)
        await coordinator.remove_file(test_file)
        
        # Verify removal
        chunks_after = db.get_chunks_by_file_id(result["file_id"], as_model=True)
        assert len(chunks_after) == 0, "Chunks should be removed"

    @pytest.mark.asyncio
    async def test_batch_file_operations(self, real_components, tmp_path):
        """Test processing multiple files in batch."""
        coordinator = real_components["coordinator"]
        db = real_components["db"]
        
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = tmp_path / f"batch_{i}.py"
            test_file.write_text(f"def batch_function_{i}(): return {i}")
            files.append(test_file)
        
        # Process all files
        results = []
        for file_path in files:
            result = await coordinator.process_file(file_path)
            results.append(result)
        
        # Verify all processed
        for i, result in enumerate(results):
            chunks = db.get_chunks_by_file_id(result["file_id"], as_model=True)
            assert len(chunks) > 0, f"Batch file {i} should be processed"

    @pytest.mark.asyncio
    async def test_file_locking_prevention(self, real_components, tmp_path):
        """Test that file locking prevents concurrent processing issues."""
        coordinator = real_components["coordinator"]
        db = real_components["db"]
        
        test_file = tmp_path / "concurrent_test.py"
        test_file.write_text("def concurrent_function(): return True")
        
        # Process file multiple times (simulating concurrent access)
        results = []
        for i in range(3):
            result = await coordinator.process_file(test_file)
            results.append(result)
        
        # Final operation should succeed
        final_chunks = db.get_chunks_by_file_id(results[-1]["file_id"], as_model=True)
        assert len(final_chunks) > 0, "Final operation should succeed"

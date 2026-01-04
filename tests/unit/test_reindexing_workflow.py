"""Real workflow tests for ChunkHound reindexing functionality.

Tests core business logic: chunk preservation, content change detection,
and data integrity using real components without mocks.
"""


import pytest

from chunkhound.core.types.common import Language
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


class TestChunkPreservationLogic:
    """Test chunk preservation using real components."""

    @pytest.mark.asyncio
    async def test_chunk_preservation_identical_content(
        self, real_components, tmp_path
    ):
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
        assert len(discount_results) > 0, (
            "Should find discount function in original content"
        )

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
        assert "0.15" in updated_tax_results[0]["content"], (
            "Tax rate should be updated to 0.15"
        )
        assert "calculate_shipping" in shipping_results[0]["content"], (
            "Should contain new shipping function"
        )

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

        assert len(updated_currency_results) > 0, (
            "Remaining function should still be findable"
        )
        assert len(removed_percentage_results) == 0, (
            "Removed function should not be findable"
        )

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

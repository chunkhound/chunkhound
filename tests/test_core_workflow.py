"""Core workflow tests for ChunkHound - end-to-end functionality without mocks.

Tests the basic user workflow: index Python files and search for content.
Uses real components to verify actual system behavior.
"""


import pytest

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService


@pytest.fixture
def workflow_components(tmp_path):
    """Real components for end-to-end testing."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()  # Initialize database schema
    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(db, tmp_path, None, {Language.PYTHON: parser})
    search_service = SearchService(db)
    return {
        "db": db,
        "parser": parser,
        "coordinator": coordinator,
        "search": search_service,
    }


class TestEndToEndWorkflow:
    """Test complete indexing and search workflow."""

    @pytest.mark.asyncio
    async def test_can_index_and_search_python_file(
        self, workflow_components, tmp_path
    ):
        """Test basic workflow: index a Python file, search for content."""
        coordinator = workflow_components["coordinator"]
        search = workflow_components["search"]

        # Create real Python file
        test_file = tmp_path / "calculator.py"
        test_file.write_text("""
def calculate_tax(income, rate):
    '''Calculate tax based on income and rate.'''
    if income <= 0:
        return 0
    return income * rate

class TaxCalculator:
    def __init__(self, default_rate=0.25):
        self.default_rate = default_rate
    
    def compute_annual_tax(self, salary):
        return calculate_tax(salary, self.default_rate)
        
    def compute_monthly_tax(self, monthly_salary):
        annual_salary = monthly_salary * 12
        return self.compute_annual_tax(annual_salary) / 12
""")

        # Index the file
        result = await coordinator.process_file(test_file)
        assert result["status"] == "success", "File should be indexed successfully"
        assert result["chunks"] > 0, "Should create chunks from the file"

        # Search for content using regex search
        chunks, _ = search.search_regex("calculate_tax")
        assert len(chunks) > 0, "Should find chunks containing 'calculate_tax'"

        # Verify search results contain expected content
        found_tax_function = any(
            "calculate_tax" in chunk["content"] for chunk in chunks
        )
        assert found_tax_function, (
            "Should find the calculate_tax function in search results"
        )

    @pytest.mark.asyncio
    async def test_can_search_across_multiple_files(
        self, workflow_components, tmp_path
    ):
        """Test searching across multiple indexed files."""
        coordinator = workflow_components["coordinator"]
        search = workflow_components["search"]

        # Create multiple Python files
        files = {
            "math_utils.py": """
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b
""",
            "string_utils.py": """
def format_number(num):
    return f"Number: {num}"

def validate_email(email):
    return "@" in email and "." in email
""",
            "data_processor.py": """
def process_data_batch(data_list):
    results = []
    for item in data_list:
        if item > 0:
            results.append(item * 2)
    return results
""",
        }

        # Index all files
        for filename, content in files.items():
            file_path = tmp_path / filename
            file_path.write_text(content)
            result = await coordinator.process_file(file_path)
            assert result["status"] == "success", (
                f"Should index {filename} successfully"
            )

        # Search for function names across files
        math_results = search.search_regex("add_numbers")
        assert len(math_results) > 0, "Should find math functions"

        string_results = search.search_regex("format_number")
        assert len(string_results) > 0, "Should find string functions"

        data_results = search.search_regex("process_data_batch")
        assert len(data_results) > 0, "Should find data processing functions"

    @pytest.mark.asyncio
    async def test_handles_file_updates_correctly(self, workflow_components, tmp_path):
        """Test that file updates are handled correctly in the workflow."""
        coordinator = workflow_components["coordinator"]
        search = workflow_components["search"]

        test_file = tmp_path / "updatable.py"

        # Original content
        test_file.write_text("""
def original_function():
    return "original implementation"

def helper_function():
    return "unchanged helper"
""")

        # Index original
        result1 = await coordinator.process_file(test_file)
        assert result1["status"] == "success"

        # Search for original function
        original_results = search.search_regex("original_function")
        assert len(original_results) > 0, "Should find original function"

        # Update content
        test_file.write_text("""
def updated_function():
    return "new implementation"

def helper_function():
    return "unchanged helper"

def new_function():
    return "additional functionality"
""")

        # Reindex updated file
        result2 = await coordinator.process_file(test_file)
        assert result2["status"] == "success"

        # Search for new content
        updated_results = search.search_regex("updated_function")
        assert len(updated_results) > 0, "Should find updated function"

        new_results = search.search_regex("new_function")
        assert len(new_results) > 0, "Should find newly added function"

        helper_results = search.search_regex("helper_function")
        assert len(helper_results) > 0, "Should still find unchanged helper function"

    @pytest.mark.asyncio
    async def test_handles_empty_and_malformed_files(
        self, workflow_components, tmp_path
    ):
        """Test workflow handles edge cases gracefully."""
        coordinator = workflow_components["coordinator"]

        # Empty file
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")
        result_empty = await coordinator.process_file(empty_file)
        # Should handle gracefully, not crash
        assert isinstance(result_empty, dict)

        # File with only comments
        comment_file = tmp_path / "comments_only.py"
        comment_file.write_text("""
# This file only has comments
# No actual code here
""")
        result_comments = await coordinator.process_file(comment_file)
        assert isinstance(result_comments, dict)

        # File with syntax errors
        malformed_file = tmp_path / "malformed.py"
        malformed_file.write_text("""
def broken_function(
    # Missing closing parenthesis and colon
    return "this won't parse"
""")
        result_malformed = await coordinator.process_file(malformed_file)
        # Should handle gracefully, not crash
        assert isinstance(result_malformed, dict)


class TestChunkQuality:
    """Test that chunks produced meet quality requirements."""

    @pytest.mark.asyncio
    async def test_chunks_have_meaningful_content(self, workflow_components, tmp_path):
        """Test that generated chunks contain meaningful code content."""
        coordinator = workflow_components["coordinator"]
        db = workflow_components["db"]

        test_file = tmp_path / "quality_test.py"
        test_file.write_text("""
def calculate_fibonacci(n):
    '''Calculate nth Fibonacci number using iteration.'''
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    prev, curr = 0, 1
    for i in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr

class MathOperations:
    '''A collection of mathematical operations.'''
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"Added {a} + {b} = {result}")
        return result
""")

        result = await coordinator.process_file(test_file)
        assert result["status"] == "success"

        chunks = db.get_chunks_by_file_id(result["file_id"], as_model=True)
        assert len(chunks) > 0, "Should generate chunks"

        # Verify chunks contain meaningful content
        for chunk in chunks:
            assert len(chunk.code.strip()) > 0, "Chunks should not be empty"
            assert len(chunk.symbol) > 0, "Chunks should have symbols"

        # Verify we can find expected functions in chunks
        chunk_symbols = [chunk.symbol for chunk in chunks]
        chunk_content = " ".join(chunk.code for chunk in chunks)

        assert any("fibonacci" in symbol.lower() for symbol in chunk_symbols), (
            "Should find Fibonacci function"
        )
        assert "calculate_fibonacci" in chunk_content, (
            "Should preserve function content"
        )
        assert "MathOperations" in chunk_content, "Should preserve class content"

    @pytest.mark.asyncio
    async def test_chunks_respect_size_limits(self, workflow_components, tmp_path):
        """Test that chunks don't exceed configured size limits."""
        coordinator = workflow_components["coordinator"]
        db = workflow_components["db"]

        # Create a file with a very large function
        large_function_content = (
            """
def very_large_function():
    '''This function will be intentionally large to test chunking.'''
    result = []
"""
            + "\n".join([f"    result.append('line_{i}')" for i in range(200)])
            + """
    return result
"""
        )

        test_file = tmp_path / "large_function.py"
        test_file.write_text(large_function_content)

        result = await coordinator.process_file(test_file)
        assert result["status"] == "success"

        chunks = db.get_chunks_by_file_id(result["file_id"], as_model=True)

        # Verify no chunk exceeds reasonable size limits
        MAX_CHUNK_CHARS = 2000  # Reasonable limit for testing
        for i, chunk in enumerate(chunks):
            chunk_size = len(chunk.code)
            assert chunk_size <= MAX_CHUNK_CHARS, (
                f"Chunk {i} exceeds size limit: {chunk_size} chars"
            )

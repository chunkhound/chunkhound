"""Tests for Python docstring indexing.

This module tests the fix for module-level docstrings and ensures
all types of docstrings (module, function, class) are properly indexed.
"""

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_services(temp_db_dir):
    """Create test database services."""
    db_path = temp_db_dir / "test.db"
    fake_args = SimpleNamespace(path=temp_db_dir)
    config = Config(
        args=fake_args,
        database={"path": str(db_path), "provider": "duckdb"},
        embedding=None,
        indexing={"include": ["*.py"], "exclude": []},
    )

    services = create_services(db_path, config, embedding_manager=None)
    services.provider.connect()

    yield services

    try:
        services.provider.disconnect()
    except Exception:
        pass


async def test_module_docstring_indexed(test_services, temp_db_dir):
    """Test that module-level docstrings are captured and indexed."""
    # Create a Python file with a module docstring
    test_file = temp_db_dir / "test_module.py"
    test_file.write_text('''"""This is a module docstring.

UNIQUE_MODULE_MARKER_ABC123

This module contains important documentation.
"""

def example_function():
    """Function docstring."""
    pass
''')

    # Index the file
    await test_services.indexing_coordinator.process_file(test_file)

    # Search for the unique marker in the module docstring
    results = test_services.provider.search_chunks_regex("UNIQUE_MODULE_MARKER_ABC123")

    # Verify the module docstring was indexed
    assert len(results) > 0, "Module docstring should be indexed"
    assert any(
        "UNIQUE_MODULE_MARKER_ABC123" in chunk["content"] for chunk in results
    ), "Module docstring content should be searchable"


async def test_function_docstring_indexed(test_services, temp_db_dir):
    """Test that function docstrings are captured and indexed."""
    test_file = temp_db_dir / "test_function.py"
    test_file.write_text('''def my_function():
    """This is a function docstring.

    UNIQUE_FUNCTION_MARKER_DEF456
    """
    return True
''')

    # Index the file
    await test_services.indexing_coordinator.process_file(test_file)

    # Search for the unique marker
    results = test_services.provider.search_chunks_regex(
        "UNIQUE_FUNCTION_MARKER_DEF456"
    )

    # Verify function docstring was indexed
    assert len(results) > 0, "Function docstring should be indexed"


async def test_class_docstring_indexed(test_services, temp_db_dir):
    """Test that class docstrings are captured and indexed."""
    test_file = temp_db_dir / "test_class.py"
    test_file.write_text('''class MyClass:
    """This is a class docstring.

    UNIQUE_CLASS_MARKER_GHI789
    """

    def method(self):
        """Method docstring."""
        pass
''')

    # Index the file
    await test_services.indexing_coordinator.process_file(test_file)

    # Search for the unique marker
    results = test_services.provider.search_chunks_regex("UNIQUE_CLASS_MARKER_GHI789")

    # Verify class docstring was indexed
    assert len(results) > 0, "Class docstring should be indexed"


async def test_async_function_docstring_indexed(test_services, temp_db_dir):
    """Test that async function docstrings are captured and indexed."""
    test_file = temp_db_dir / "test_async.py"
    test_file.write_text('''async def async_function():
    """This is an async function docstring.

    UNIQUE_ASYNC_MARKER_JKL012
    """
    return True
''')

    # Index the file
    await test_services.indexing_coordinator.process_file(test_file)

    # Search for the unique marker
    results = test_services.provider.search_chunks_regex("UNIQUE_ASYNC_MARKER_JKL012")

    # Verify async function docstring was indexed
    assert len(results) > 0, "Async function docstring should be indexed"


async def test_multiline_docstring_indexed(test_services, temp_db_dir):
    """Test that multi-line docstrings are fully captured."""
    test_file = temp_db_dir / "test_multiline.py"
    test_file.write_text('''"""Multi-line module docstring.

This docstring spans multiple lines.
It contains various sections.

UNIQUE_MULTILINE_MARKER_MNO345

And more content after the marker.
"""

def function():
    pass
''')

    # Index the file
    await test_services.indexing_coordinator.process_file(test_file)

    # Search for the marker
    results = test_services.provider.search_chunks_regex(
        "UNIQUE_MULTILINE_MARKER_MNO345"
    )

    # Verify the entire docstring was captured
    assert len(results) > 0, "Multi-line docstring should be indexed"
    found_result = next(
        (r for r in results if "UNIQUE_MULTILINE_MARKER_MNO345" in r["content"]), None
    )
    assert found_result is not None
    assert "Multi-line module docstring" in found_result["content"], (
        "Should capture beginning of docstring"
    )
    assert "more content after" in found_result["content"], (
        "Should capture end of docstring"
    )


async def test_original_qa_test_case(test_services, temp_db_dir):
    """Test the original QA test case that exposed the bug."""
    test_file = temp_db_dir / "qa_test_original.py"
    test_file.write_text('''"""QA Test File - Python
Unique marker: PYTHON_QA_TEST_MARKER_8472ABC
"""

def quantum_entanglement_processor():
    """Process quantum entangled particles with unique_quantum_signature_xyz789"""
    return "quantum_state_superposition_active"

class NeuralNetworkOptimizer:
    """Advanced neural network optimization with gradient_descent_momentum"""

    def backpropagate_errors(self):
        """Backpropagation algorithm for neural networks"""
        return "backpropagation_complete_xyz123"
''')

    # Index the file
    await test_services.indexing_coordinator.process_file(test_file)

    # This is the critical test - module docstring should now be indexed
    results = test_services.provider.search_chunks_regex(
        "PYTHON_QA_TEST_MARKER_8472ABC"
    )

    assert len(results) > 0, "Module docstring with QA marker should be indexed"
    assert any(
        "PYTHON_QA_TEST_MARKER_8472ABC" in result["content"] for result in results
    ), "QA marker should be searchable in module docstring"

    # Verify function docstrings still work
    func_results = test_services.provider.search_chunks_regex(
        "unique_quantum_signature_xyz789"
    )
    assert len(func_results) > 0, "Function docstrings should still be indexed"


async def test_mixed_comments_and_docstrings(test_services, temp_db_dir):
    """Test that both comments and docstrings coexist."""
    test_file = temp_db_dir / "test_mixed.py"
    test_file.write_text('''"""Module docstring.

DOCSTRING_MARKER_PQR678
"""

# Regular comment
# COMMENT_MARKER_STU901

def function():
    """Function docstring.

    FUNCTION_DOC_MARKER_VWX234
    """
    # Function comment
    pass
''')

    # Index the file
    await test_services.indexing_coordinator.process_file(test_file)

    # Both docstrings and comments should be indexed
    doc_results = test_services.provider.search_chunks_regex("DOCSTRING_MARKER_PQR678")
    comment_results = test_services.provider.search_chunks_regex(
        "COMMENT_MARKER_STU901"
    )
    func_doc_results = test_services.provider.search_chunks_regex(
        "FUNCTION_DOC_MARKER_VWX234"
    )

    assert len(doc_results) > 0, "Module docstring should be indexed"
    assert len(comment_results) > 0, "Regular comments should still be indexed"
    assert len(func_doc_results) > 0, "Function docstring should be indexed"


async def test_triple_quote_variations(test_services, temp_db_dir):
    """Test different triple-quote styles."""
    test_file = temp_db_dir / "test_quotes.py"
    test_file.write_text("""'''Module with single-quote docstring.

SINGLE_QUOTE_MARKER_YZA567
'''

def function():
    \"\"\"Function with double-quote docstring.

    DOUBLE_QUOTE_MARKER_BCD890
    \"\"\"
    pass
""")

    # Index the file
    await test_services.indexing_coordinator.process_file(test_file)

    # Both styles should be indexed
    single_results = test_services.provider.search_chunks_regex(
        "SINGLE_QUOTE_MARKER_YZA567"
    )
    double_results = test_services.provider.search_chunks_regex(
        "DOUBLE_QUOTE_MARKER_BCD890"
    )

    assert len(single_results) > 0, "Single-quote docstrings should be indexed"
    assert len(double_results) > 0, "Double-quote docstrings should be indexed"


async def test_empty_file_no_docstring(test_services, temp_db_dir):
    """Test file without docstring doesn't cause errors."""
    test_file = temp_db_dir / "test_empty.py"
    test_file.write_text("""# Just a comment

def function():
    pass
""")

    # Should index without errors
    await test_services.indexing_coordinator.process_file(test_file)

    # Search should return results for the comment but not crash
    results = test_services.provider.search_chunks_regex("Just a comment")
    assert len(results) > 0, "Comments should be indexed even without docstrings"

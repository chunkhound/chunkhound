"""Unit tests for boundary expansion in v2 Coverage Synthesis Engine.

Tests the _expand_boundaries() method which expands partial file snippets
to complete functions/classes using FileReader.expand_to_natural_boundaries().
"""


import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.coverage_synthesis import CoverageSynthesisEngine
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for token estimation."""
    return FakeLLMProvider(
        responses={
            "compress": "Compressed summary",
            "synthesis": "Final synthesis",
        }
    )


@pytest.fixture
def fake_embedding_provider():
    """Create fake embedding provider."""
    return FakeEmbeddingProvider(dims=1536)


@pytest.fixture
def llm_manager(fake_llm_provider, monkeypatch):
    """Create LLM manager with fake provider."""

    def mock_create_provider(self, config):
        return fake_llm_provider

    monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}
    return LLMManager(utility_config, synthesis_config)


@pytest.fixture
def embedding_manager(fake_embedding_provider, monkeypatch):
    """Create embedding manager with fake provider."""

    class MockEmbeddingManager:
        def get_provider(self):
            return fake_embedding_provider

    return MockEmbeddingManager()


@pytest.fixture
def db_services(tmp_path):
    """Create mock database services with temporary directory."""

    class MockProvider:
        def get_base_directory(self):
            return tmp_path

    class MockDatabaseServices:
        provider = MockProvider()

    return MockDatabaseServices()


@pytest.fixture
def research_config():
    """Create research configuration for testing."""
    return ResearchConfig(
        target_tokens=10000,
        max_chunks_per_file_repr=5,
        max_tokens_per_file_repr=2000,
        max_boundary_expansion_lines=300,
        max_compression_iterations=3,
        min_cluster_size=2,
        shard_budget=20_000,
        gap_similarity_threshold=0.3,
        min_gaps=1,
        max_gaps=5,
        max_symbols=10,
        query_expansion_enabled=True,
    )


@pytest.fixture
def synthesis_engine(llm_manager, embedding_manager, db_services, research_config):
    """Create coverage synthesis engine with mocked dependencies."""
    return CoverageSynthesisEngine(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config,
    )


@pytest.fixture
def sample_python_file(tmp_path):
    """Create a sample Python file with functions for testing boundary expansion."""
    file_path = tmp_path / "sample.py"
    content = '''"""Sample module for testing."""

import os


def helper_function():
    """Helper function before main."""
    return "helper"


def main_function(arg1, arg2):
    """Main function to test boundary expansion.

    This is a multi-line docstring
    that should be included.
    """
    result = arg1 + arg2

    # Some logic
    if result > 10:
        print("Large result")
    else:
        print("Small result")

    return result


def another_function():
    """Another function after main."""
    pass


class SampleClass:
    """A sample class."""

    def method_one(self):
        """First method."""
        return 1

    def method_two(self):
        """Second method."""
        return 2
'''
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_javascript_file(tmp_path):
    """Create a sample JavaScript file with functions for testing boundary expansion."""
    file_path = tmp_path / "sample.js"
    content = '''// Sample JavaScript module

function helperFunction() {
  return "helper";
}

/**
 * Main function to test boundary expansion.
 * This is a multi-line comment.
 */
function mainFunction(arg1, arg2) {
  const result = arg1 + arg2;

  if (result > 10) {
    console.log("Large result");
  } else {
    console.log("Small result");
  }

  return result;
}

function anotherFunction() {
  return "another";
}

class SampleClass {
  methodOne() {
    return 1;
  }

  methodTwo() {
    return 2;
  }
}
'''
    file_path.write_text(content)
    return file_path


@pytest.mark.asyncio
async def test_expand_boundaries_full_file(synthesis_engine, sample_python_file):
    """Test that full files are not expanded (pass-through)."""
    # Full file content (no "..." separator)
    full_content = sample_python_file.read_text()
    budgeted_files = {"sample.py": full_content}

    # Expand boundaries
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    # Should return unchanged
    assert expanded == budgeted_files
    assert expanded["sample.py"] == full_content


@pytest.mark.asyncio
async def test_expand_boundaries_partial_file_python(
    synthesis_engine, sample_python_file, tmp_path
):
    """Test boundary expansion for partial Python file with snippets."""
    # Create partial content with just the middle of main_function (lines 13-18)
    # Note: Must include single snippet to avoid being treated as full file
    partial_content = '''# Lines 13-18
def main_function(arg1, arg2):
    """Main function to test boundary expansion.

    This is a multi-line docstring
    that should be included.
    """

...

# Lines 28-30
def another_function():
    """Another function after main."""
    pass'''

    # Update db_services to point to tmp_path
    synthesis_engine._db_services.provider.get_base_directory = lambda: tmp_path

    budgeted_files = {"sample.py": partial_content}

    # Expand boundaries
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    # Should expand to include full function
    expanded_content = expanded["sample.py"]

    # Verify expansion occurred
    assert "# Lines" in expanded_content
    assert "def main_function" in expanded_content
    assert "return result" in expanded_content  # End of function should be included

    # Should NOT include helper_function
    assert "helper_function" not in expanded_content


@pytest.mark.asyncio
async def test_expand_boundaries_partial_file_javascript(
    synthesis_engine, sample_javascript_file, tmp_path
):
    """Test boundary expansion for partial JavaScript file with snippets."""
    # Create partial content with just the middle of mainFunction (lines 10-15)
    partial_content = '''# Lines 10-15
function mainFunction(arg1, arg2) {
  const result = arg1 + arg2;

  if (result > 10) {
    console.log("Large result");
  }

...

# Lines 23-25
function anotherFunction() {
  return "another";
}'''

    # Update db_services to point to tmp_path
    synthesis_engine._db_services.provider.get_base_directory = lambda: tmp_path

    budgeted_files = {"sample.js": partial_content}

    # Expand boundaries
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    # Should expand to include full function with closing brace
    expanded_content = expanded["sample.js"]

    # Verify expansion occurred
    assert "# Lines" in expanded_content
    assert "function mainFunction" in expanded_content
    assert "return result;" in expanded_content  # End of function should be included

    # Should NOT include helperFunction
    assert "helperFunction" not in expanded_content


@pytest.mark.asyncio
async def test_expand_boundaries_multiple_snippets(
    synthesis_engine, sample_python_file, tmp_path
):
    """Test boundary expansion for multiple snippets in same file."""
    # Create partial content with two separate snippets
    partial_content = '''# Lines 7-9
def helper_function():
    """Helper function before main."""
    return "helper"

...

# Lines 28-30
def another_function():
    """Another function after main."""
    pass'''

    # Update db_services to point to tmp_path
    synthesis_engine._db_services.provider.get_base_directory = lambda: tmp_path

    budgeted_files = {"sample.py": partial_content}

    # Expand boundaries
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    expanded_content = expanded["sample.py"]

    # Should have two sections separated by "..."
    sections = expanded_content.split("\n\n...\n\n")
    assert len(sections) == 2

    # First section should contain complete helper_function
    assert "def helper_function" in sections[0]
    assert "return" in sections[0]

    # Second section should contain another_function
    assert "def another_function" in sections[1]
    assert "pass" in sections[1]


@pytest.mark.asyncio
async def test_expand_boundaries_respects_max_lines(
    synthesis_engine, sample_python_file, tmp_path, research_config
):
    """Test that expansion respects max_boundary_expansion_lines limit."""
    # Set very small limit
    research_config.max_boundary_expansion_lines = 10

    # Create new engine with restricted config
    engine = CoverageSynthesisEngine(
        llm_manager=synthesis_engine._llm_manager,
        embedding_manager=synthesis_engine._embedding_manager,
        db_services=synthesis_engine._db_services,
        config=research_config,
    )

    # Update db_services to point to tmp_path
    engine._db_services.provider.get_base_directory = lambda: tmp_path

    # Create partial content
    partial_content = """# Lines 13-18
def main_function(arg1, arg2):
    """

    budgeted_files = {"sample.py": partial_content}

    # Expand boundaries
    expanded = await engine._expand_boundaries(budgeted_files)

    expanded_content = expanded["sample.py"]

    # Parse expanded line range
    header = expanded_content.split("\n")[0]
    range_str = header.split("Lines ", 1)[1]
    start, end = map(int, range_str.split("-"))

    # Should not exceed limit
    assert end - start + 1 <= research_config.max_boundary_expansion_lines


@pytest.mark.asyncio
async def test_expand_boundaries_missing_file(synthesis_engine, tmp_path):
    """Test handling of missing files during boundary expansion."""
    # Update db_services to point to tmp_path
    synthesis_engine._db_services.provider.get_base_directory = lambda: tmp_path

    # Create partial content for non-existent file
    partial_content = """# Lines 10-15
def some_function():
    pass"""

    budgeted_files = {"nonexistent.py": partial_content}

    # Expand boundaries (should not crash)
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    # Should return original content unchanged
    assert expanded["nonexistent.py"] == partial_content


@pytest.mark.asyncio
async def test_expand_boundaries_malformed_header(
    synthesis_engine, sample_python_file, tmp_path
):
    """Test handling of malformed line range headers."""
    # Update db_services to point to tmp_path
    synthesis_engine._db_services.provider.get_base_directory = lambda: tmp_path

    # Create partial content with malformed header
    partial_content = """# Invalid header format
def main_function(arg1, arg2):
    pass"""

    budgeted_files = {"sample.py": partial_content}

    # Expand boundaries (should not crash)
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    # Should return original content unchanged
    assert expanded["sample.py"] == partial_content


@pytest.mark.asyncio
async def test_expand_boundaries_no_header(
    synthesis_engine, sample_python_file, tmp_path
):
    """Test handling of snippets without line range headers."""
    # Update db_services to point to tmp_path
    synthesis_engine._db_services.provider.get_base_directory = lambda: tmp_path

    # Create partial content without header
    partial_content = """def main_function(arg1, arg2):
    pass

...

def another_function():
    pass"""

    budgeted_files = {"sample.py": partial_content}

    # Expand boundaries (should not crash)
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    # Should return original content unchanged
    assert expanded["sample.py"] == partial_content


@pytest.mark.asyncio
async def test_expand_boundaries_empty_budgeted_files(synthesis_engine):
    """Test handling of empty budgeted_files dict."""
    budgeted_files = {}

    # Expand boundaries
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    # Should return empty dict
    assert expanded == {}


@pytest.mark.asyncio
async def test_expand_boundaries_tracks_stats(
    synthesis_engine, sample_python_file, tmp_path
):
    """Test that expansion statistics are logged correctly."""
    # Update db_services to point to tmp_path
    synthesis_engine._db_services.provider.get_base_directory = lambda: tmp_path

    # Create mix of full and partial files
    full_content = sample_python_file.read_text()
    partial_content = """# Lines 13-18
def main_function(arg1, arg2):
    pass

...

# Lines 28-30
def another_function():
    pass"""

    budgeted_files = {
        "sample_full.py": full_content,
        "sample.py": partial_content,
    }

    # Copy sample file for full version
    (tmp_path / "sample_full.py").write_text(full_content)

    # Expand boundaries
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    # Should have both files
    assert len(expanded) == 2
    assert "sample_full.py" in expanded
    assert "sample.py" in expanded

    # Full file should be unchanged
    assert expanded["sample_full.py"] == full_content

    # Partial file should be expanded
    assert len(expanded["sample.py"]) > len(partial_content)


@pytest.mark.asyncio
async def test_expand_boundaries_preserves_separator(
    synthesis_engine, sample_python_file, tmp_path
):
    """Test that snippet separator is preserved in expanded output."""
    # Update db_services to point to tmp_path
    synthesis_engine._db_services.provider.get_base_directory = lambda: tmp_path

    # Create partial content with multiple snippets
    partial_content = """# Lines 7-9
def helper_function():
    pass

...

# Lines 22-24
def another_function():
    pass"""

    budgeted_files = {"sample.py": partial_content}

    # Expand boundaries
    expanded = await synthesis_engine._expand_boundaries(budgeted_files)

    # Should preserve "..." separator
    assert "\n\n...\n\n" in expanded["sample.py"]

    # Should have exactly one separator (2 snippets)
    assert expanded["sample.py"].count("\n\n...\n\n") == 1

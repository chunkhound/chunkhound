"""In-process contract tests for search functionality.

Replaces the flaky subprocess-based TestSearchCLI in tests/test_search_cli.py.
Uses real DuckDB in-memory, real tree-sitter parsing, and real regex search —
no subprocesses, no network calls, no mocks.
"""

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.database_factory import create_services
from chunkhound.mcp_server.tools import search_impl
from chunkhound.services.directory_indexing_service import DirectoryIndexingService

# --- Shared test project content (matches old TestSearchCLI fixture) ---

CALCULATOR_PY = '''
def calculate_tax(income, rate):
    """Calculate tax based on income and rate."""
    if income <= 0:
        return 0
    return income * rate

class TaxCalculator:
    def __init__(self, default_rate=0.25):
        self.default_rate = default_rate

    def compute_annual_tax(self, salary):
        return calculate_tax(salary, self.default_rate)
'''

STRING_UTILS_PY = '''
def format_number(num):
    """Format a number for display."""
    return f"Number: {num}"

def validate_email(email):
    """Basic email validation."""
    return "@" in email and "." in email

def process_text(text):
    """Process text by trimming and lowercasing."""
    return text.strip().lower()
'''

MAIN_PY = '''
from calculator import TaxCalculator

def main():
    """Main application entry point."""
    calc = TaxCalculator()
    result = calc.compute_annual_tax(50000)
    print(f"Annual tax: {result}")

if __name__ == "__main__":
    main()
'''


@pytest.fixture
async def indexed_project(tmp_path):
    """Create a temporary project, index it in-memory, yield (tmp_path, services)."""
    (tmp_path / "calculator.py").write_text(CALCULATOR_PY)
    utils_dir = tmp_path / "utils"
    utils_dir.mkdir()
    (utils_dir / "string_utils.py").write_text(STRING_UTILS_PY)
    (tmp_path / "main.py").write_text(MAIN_PY)

    config = Config(
        database=DatabaseConfig(path=":memory:", provider="duckdb"),
        indexing=IndexingConfig(include=["*.py"]),
        target_dir=tmp_path,
    )
    services = create_services(":memory:", config)

    indexing_service = DirectoryIndexingService(
        indexing_coordinator=services.indexing_coordinator,
        config=config,
    )
    await indexing_service.process_directory(tmp_path, no_embeddings=True)

    yield tmp_path, services


async def _regex_search(services, query, **kwargs):
    return await search_impl(
        services=services,
        embedding_manager=None,
        type="regex",
        query=query,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_regex_finds_function(indexed_project):
    tmp_path, services = indexed_project
    resp = await _regex_search(services, "calculate_tax")
    results = resp["results"]
    assert len(results) >= 1
    matched = [r for r in results if "calculate_tax" in r.get("content", "")]
    assert len(matched) >= 1
    file_paths = " ".join(r.get("file_path", "") for r in matched)
    assert "calculator.py" in file_paths


@pytest.mark.asyncio
async def test_path_filter_scopes_results(indexed_project):
    tmp_path, services = indexed_project
    resp = await _regex_search(services, "def", path="utils/")
    results = resp["results"]
    assert len(results) >= 1
    for r in results:
        assert "utils" in r.get("file_path", "")


@pytest.mark.asyncio
async def test_pagination_page_size(indexed_project):
    tmp_path, services = indexed_project
    resp = await _regex_search(services, "def", page_size=1)
    assert len(resp["results"]) == 1
    pagination = resp["pagination"]
    assert pagination["has_more"] is True or pagination.get("total", 0) > 1


@pytest.mark.asyncio
async def test_pagination_offset_different_results(indexed_project):
    tmp_path, services = indexed_project
    resp0 = await _regex_search(services, "def", page_size=1, offset=0)
    resp1 = await _regex_search(services, "def", page_size=1, offset=1)
    r0 = resp0["results"][0]
    r1 = resp1["results"][0]
    assert r0.get("content") != r1.get("content") or r0.get("start_line") != r1.get(
        "start_line"
    )


@pytest.mark.asyncio
async def test_regex_empty_results(indexed_project):
    tmp_path, services = indexed_project
    resp = await _regex_search(services, "nonexistent_xyz_abc")
    assert resp["results"] == []


@pytest.mark.asyncio
async def test_result_structure(indexed_project):
    tmp_path, services = indexed_project
    resp = await _regex_search(services, "def")
    assert len(resp["results"]) >= 1
    required_keys = {"file_path", "content", "start_line", "end_line"}
    for r in resp["results"]:
        assert required_keys.issubset(r.keys()), (
            f"Missing keys: {required_keys - r.keys()}"
        )

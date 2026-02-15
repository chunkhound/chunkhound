#!/usr/bin/env python3
"""
End-to-end tests for the `chunkhound search` CLI command.

These tests execute the actual CLI command without mocks to verify:
1. CLI argument parsing and validation
2. Search functionality works end-to-end via CLI
3. Output formatting and pagination
4. Error handling scenarios
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.test_utils import get_api_key_for_tests
from tests.utils.windows_subprocess import get_safe_subprocess_env


class TestSearchCLI:
    """Test the chunkhound search CLI command end-to-end."""

    @pytest.fixture
    def cli_project_setup(self, clean_environment):
        """Set up a temporary project with indexed content for CLI testing."""
        temp_dir = Path(tempfile.mkdtemp()).resolve()
        project_dir = (temp_dir / "project").resolve()
        project_dir.mkdir()

        # Create test Python files with known content
        test_files = {
            "calculator.py": '''
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
''',
            "utils/string_utils.py": '''
def format_number(num):
    """Format a number for display."""
    return f"Number: {num}"

def validate_email(email):
    """Basic email validation."""
    return "@" in email and "." in email

def process_text(text):
    """Process text by trimming and lowercasing."""
    return text.strip().lower()
''',
            "main.py": '''
from calculator import TaxCalculator

def main():
    """Main application entry point."""
    calc = TaxCalculator()
    result = calc.compute_annual_tax(50000)
    print(f"Annual tax: {result}")

if __name__ == "__main__":
    main()
''',
        }

        # Create the test files
        for file_path, content in test_files.items():
            full_path = project_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # Standard API key discovery
        api_key, provider = get_api_key_for_tests()

        # Create chunkhound config with absolute database path
        # Use absolute, pre-resolved path to ensure consistent resolution across subprocesses
        # This prevents Ubuntu CI symlink issues where relative paths resolve differently
        db_path = (project_dir / ".chunkhound" / "test.db").resolve()
        config = {
            "database": {
                "path": str(db_path),
                "provider": "duckdb",
            },
            "indexing": {"include": ["*.py"]},
        }

        # Add embedding config only if API key available
        if api_key and provider:
            model = "text-embedding-3-small" if provider == "openai" else "voyage-3.5"
            config["embedding"] = {
                "provider": provider,
                "api_key": api_key,
                "model": model,
            }

        config_path = project_dir / ".chunkhound.json"
        config_path.write_text(json.dumps(config))

        # Index the project files
        index_cmd = ["uv", "run", "chunkhound", "index", "."]
        if not (api_key and provider):
            index_cmd.append("--no-embeddings")

        index_result = subprocess.run(
            index_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=get_safe_subprocess_env(),
            cwd=project_dir,
        )

        if index_result.returncode != 0:
            # Cleanup on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            pytest.fail(f"Failed to index test project: {index_result.stderr}")

        yield project_dir

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_search_regex_basic(self, cli_project_setup):
        """Test basic regex search via CLI."""
        project_dir = cli_project_setup

        result = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "search",
                "calculate_tax",
                ".",
                "--regex",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=get_safe_subprocess_env(),
            cwd=project_dir,
        )

        assert result.returncode == 0, f"Search failed: {result.stderr}"
        assert "calculate_tax" in result.stdout, "Should find calculate_tax function"
        assert "calculator.py" in result.stdout, "Should show file path in results"

    def test_search_regex_with_pagination(self, cli_project_setup):
        """Test regex search with pagination options."""
        project_dir = cli_project_setup

        # Test with small page size
        result = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "search",
                "def",
                ".",
                "--regex",
                "--page-size",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=get_safe_subprocess_env(),
            cwd=project_dir,
        )

        assert result.returncode == 0, f"Search failed: {result.stderr}"
        assert "def" in result.stdout, "Should find function definitions"

    def test_search_regex_with_path_filter(self, cli_project_setup):
        """Test regex search with path filtering."""
        project_dir = cli_project_setup

        # Search only in utils directory
        result = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "search",
                "format_number",
                ".",
                "--regex",
                "--path-filter",
                "utils/",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=get_safe_subprocess_env(),
            cwd=project_dir,
        )

        assert result.returncode == 0, f"Search failed: {result.stderr}"
        assert "format_number" in result.stdout, "Should find function in utils"
        assert "string_utils.py" in result.stdout, "Should show utils file"

    def test_search_regex_empty_results(self, cli_project_setup):
        """Test regex search with no matches."""
        project_dir = cli_project_setup

        result = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "search",
                "nonexistent_function",
                ".",
                "--regex",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=get_safe_subprocess_env(),
            cwd=project_dir,
        )

        assert result.returncode == 0, f"Search failed: {result.stderr}"
        # Should handle empty results gracefully - should show "No results found"
        assert "No results found" in result.stdout or result.stdout.strip() == ""

    def test_search_no_database_error(self, clean_environment):
        """Test search command when database doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / "empty_project"
            empty_dir.mkdir()

            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "chunkhound",
                    "search",
                    "anything",
                    str(empty_dir),
                    "--regex",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                env=get_safe_subprocess_env(),
            )

            # Should exit with error when no database exists
            assert result.returncode != 0, "Should fail when no database exists"
            # Error message may be in stdout or stderr
            error_output = result.stderr + result.stdout
            assert error_output, "Should have error message"

    def test_search_output_format_structure(self, cli_project_setup):
        """Test that search output has expected structure."""
        project_dir = cli_project_setup

        result = subprocess.run(
            ["uv", "run", "chunkhound", "search", "def", ".", "--regex"],
            capture_output=True,
            text=True,
            timeout=10,
            env=get_safe_subprocess_env(),
            cwd=project_dir,
        )

        assert result.returncode == 0, f"Search failed: {result.stderr}"

        # Check that output contains expected elements
        assert result.stdout, "Should have output"

        # Should contain file paths (basic format check)
        assert ".py" in result.stdout, "Should show Python file extensions"

    def test_search_with_offset(self, cli_project_setup):
        """Test search with offset parameter."""
        project_dir = cli_project_setup

        # Get first page
        result1 = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "search",
                "def",
                ".",
                "--regex",
                "--page-size",
                "1",
                "--offset",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=get_safe_subprocess_env(),
            cwd=project_dir,
        )

        # Get second page
        result2 = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "search",
                "def",
                ".",
                "--regex",
                "--page-size",
                "1",
                "--offset",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=get_safe_subprocess_env(),
            cwd=project_dir,
        )

        assert result1.returncode == 0, f"First search failed: {result1.stderr}"
        assert result2.returncode == 0, f"Second search failed: {result2.stderr}"

        # Results should be different (pagination working)
        if result1.stdout and result2.stdout:
            assert result1.stdout != result2.stdout, (
                "Pagination should return different results"
            )


class TestSearchCLIArguments:
    """Test CLI argument parsing and validation."""

    def test_search_help(self):
        """Test that search help command works."""
        result = subprocess.run(
            ["uv", "run", "chunkhound", "search", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            env=get_safe_subprocess_env(),
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "search" in result.stdout.lower(), "Help should mention search"
        assert "--regex" in result.stdout, "Help should show --regex option"
        assert "--semantic" in result.stdout, "Help should show --semantic option"
        assert "--page-size" in result.stdout, "Help should show --page-size option"

    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        # Test invalid flag
        result = subprocess.run(
            ["uv", "run", "chunkhound", "search", "query", "--invalid-flag"],
            capture_output=True,
            text=True,
            timeout=5,
            env=get_safe_subprocess_env(),
        )

        assert result.returncode != 0, "Should fail with invalid argument"

    def test_missing_query(self):
        """Test handling when query is missing."""
        result = subprocess.run(
            ["uv", "run", "chunkhound", "search", "--regex"],
            capture_output=True,
            text=True,
            timeout=5,
            env=get_safe_subprocess_env(),
        )

        assert result.returncode != 0, "Should fail when query is missing"

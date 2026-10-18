#!/usr/bin/env python3
"""Tests for the `chunkhound search` CLI argument parsing and validation."""

from tests.helpers.cli import run_cli


class TestSearchCLIArguments:
    """Test CLI argument parsing and validation."""

    def test_search_help(self):
        """Test that search help command works."""
        result = run_cli(["search", "--help"], timeout=5)

        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "search" in result.stdout.lower(), "Help should mention search"
        assert "--regex" in result.stdout, "Help should show --regex option"
        assert "--semantic" in result.stdout, "Help should show --semantic option"
        assert "--page-size" in result.stdout, "Help should show --page-size option"

    def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        result = run_cli(["search", "query", "--invalid-flag"], timeout=5)

        assert result.returncode != 0, "Should fail with invalid argument"

    def test_missing_query(self):
        """Test handling when query is missing."""
        result = run_cli(["search", "--regex"], timeout=5)

        assert result.returncode != 0, "Should fail when query is missing"

    def test_regex_search_smoke(self, tmp_path):
        """Test end-to-end regex search through the real CLI."""
        (tmp_path / "calculator.py").write_text(
            "def calculate_tax(income, rate):\n    return income * rate\n",
            encoding="utf-8",
        )
        (tmp_path / "main.py").write_text(
            "from calculator import calculate_tax\nprint(calculate_tax(1, 2))\n",
            encoding="utf-8",
        )

        index_result = run_cli(
            ["index", str(tmp_path), "--no-embeddings"],
            cwd=tmp_path,
            timeout=30,
            env={"CHUNKHOUND_NO_PROMPTS": "1"},
        )
        assert index_result.returncode == 0, index_result.stderr

        search_result = run_cli(
            ["search", "calculate_tax", str(tmp_path), "--regex"],
            cwd=tmp_path,
            timeout=15,
            env={"CHUNKHOUND_NO_PROMPTS": "1"},
        )

        assert search_result.returncode == 0, search_result.stderr
        assert "calculator.py" in search_result.stdout
        assert "calculate_tax" in search_result.stdout

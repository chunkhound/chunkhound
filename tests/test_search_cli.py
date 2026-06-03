#!/usr/bin/env python3
"""Tests for the `chunkhound search` CLI argument parsing and validation."""

import os
import subprocess

from tests.utils.windows_subprocess import get_safe_subprocess_env


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

        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CHUNKHOUND_")
        }
        env.update(
            {
                "CHUNKHOUND_NO_PROMPTS": "1",
                "CHUNKHOUND_NO_RICH": "1",
                "UV_CACHE_DIR": str(tmp_path / ".uv-cache"),
            }
        )
        env = get_safe_subprocess_env(env)

        index_result = subprocess.run(
            ["uv", "run", "chunkhound", "index", str(tmp_path), "--no-embeddings"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=tmp_path,
        )
        assert index_result.returncode == 0, index_result.stderr

        search_result = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "search",
                "calculate_tax",
                str(tmp_path),
                "--regex",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
            cwd=tmp_path,
        )

        assert search_result.returncode == 0, search_result.stderr
        assert "calculator.py" in search_result.stdout
        assert "calculate_tax" in search_result.stdout

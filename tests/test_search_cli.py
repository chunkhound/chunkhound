#!/usr/bin/env python3
"""Tests for the `chunkhound search` CLI argument parsing and validation."""

import subprocess

import pytest

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

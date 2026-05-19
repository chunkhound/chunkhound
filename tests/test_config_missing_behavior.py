"""Test behavior when config file is missing."""

import os
import subprocess
import tempfile

class TestConfigMissingBehavior:
    """Test behavior when config file is missing or incomplete."""

    @staticmethod
    def _clean_env() -> dict[str, str]:
        """Return a subprocess environment without ambient ChunkHound config."""
        return {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CHUNKHOUND_")
        }

    def test_index_without_config_shows_helpful_error(self):
        """Test that running index without config shows a helpful error message."""
        with tempfile.TemporaryDirectory() as test_dir:
            result = subprocess.run(
                ["uv", "run", "chunkhound", "index", test_dir],
                capture_output=True,
                text=True,
                env=self._clean_env(),
                timeout=30,
            )

            # Should exit with error
            assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"

            # Helpful guidance should be emitted to stderr for non-interactive subprocesses.
            assert "No embedding provider configured" in result.stderr, f"Error message not found in stderr: {result.stderr}"
            assert "https://chunkhound.ai" in result.stderr, f"Website suggestion not found in stderr: {result.stderr}"
            assert ".chunkhound.json" in result.stderr, f"Config file suggestion not found in stderr: {result.stderr}"
            assert result.stdout == "", f"Did not expect stdout banner in non-TTY mode: {result.stdout}"

    def test_no_attribute_error_crash(self):
        """Test that we don't get AttributeError when no config exists (the main bug fix)."""
        with tempfile.TemporaryDirectory() as test_dir:
            result = subprocess.run(
                ["uv", "run", "chunkhound", "index", test_dir],
                capture_output=True,
                text=True,
                env=self._clean_env(),
                timeout=30,
            )

            # The critical fix: should not crash with AttributeError
            assert "'NoneType' object has no attribute 'provider'" not in result.stderr, f"AttributeError crash found: {result.stderr}"
            assert "AttributeError: 'Namespace' object has no attribute 'provider'" not in result.stderr, f"CLI AttributeError found: {result.stderr}"

"""Test behavior when config file is missing."""

import os
import subprocess
import tempfile


class TestConfigMissingBehavior:
    """Test behavior when config file is missing or incomplete."""

    def test_index_without_config_shows_helpful_error(self):
        """Test that running index without config shows a helpful error message."""
        with tempfile.TemporaryDirectory() as test_dir:
            # Run chunkhound index without any config
            # Copy env and clear provider
            env = os.environ.copy()
            env.pop("CHUNKHOUND_EMBEDDING_PROVIDER", None)
            env.pop("CHUNKHOUND_EMBEDDING__PROVIDER", None)

            result = subprocess.run(
                ["uv", "run", "chunkhound", "index", test_dir],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )

            # Should exit with error
            assert result.returncode == 1, (
                f"Expected exit code 1, got {result.returncode}"
            )

            # Should show helpful error message
            assert "No embedding provider configured" in result.stderr, (
                f"Error message not found in stderr: {result.stderr}"
            )
            assert ".chunkhound.json config file" in result.stdout, (
                f"Config file suggestion not found in stdout: {result.stdout}"
            )
            assert "--no-embeddings" in result.stdout, (
                f"--no-embeddings option not found in stdout: {result.stdout}"
            )

    def test_no_attribute_error_crash(self):
        """Test that we don't get AttributeError when no config exists (the main bug fix)."""
        with tempfile.TemporaryDirectory() as test_dir:
            # Clear all embedding environment variables
            env = os.environ.copy()
            env.pop("CHUNKHOUND_EMBEDDING_PROVIDER", None)
            env.pop("CHUNKHOUND_EMBEDDING__PROVIDER", None)
            env.pop("CHUNKHOUND_EMBEDDING_API_KEY", None)
            env.pop("CHUNKHOUND_EMBEDDING__API_KEY", None)

            result = subprocess.run(
                ["uv", "run", "chunkhound", "index", test_dir],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )

            # The critical fix: should not crash with AttributeError
            assert (
                "'NoneType' object has no attribute 'provider'" not in result.stderr
            ), f"AttributeError crash found: {result.stderr}"
            assert (
                "AttributeError: 'Namespace' object has no attribute 'provider'"
                not in result.stderr
            ), f"CLI AttributeError found: {result.stderr}"

"""Tests for database retry logic and timeout handling."""

import asyncio
import time
from unittest.mock import MagicMock, patch, call
import pytest

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.providers.database.serial_executor import SerialDatabaseExecutor


class TestDatabaseRetryLogic:
    """Test retry logic in SerialDatabaseExecutor."""

    def test_retry_on_timeout_success_after_retry(self):
        """Test that TimeoutError triggers retry and succeeds on second attempt."""
        config = DatabaseConfig(retry_on_timeout=True, max_retries=2, retry_backoff_seconds=0.01)
        executor = SerialDatabaseExecutor(config)

        # Mock operation that fails once then succeeds
        call_count = 0
        def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Database timeout")
            return "success"

        result = executor._execute_with_retry(mock_operation)
        assert result == "success"
        assert call_count == 2

    def test_retry_on_timeout_exhausted_raises_final_exception(self):
        """Test that exhausted retries raise the final TimeoutError."""
        config = DatabaseConfig(retry_on_timeout=True, max_retries=2, retry_backoff_seconds=0.01)
        executor = SerialDatabaseExecutor(config)

        # Mock operation that always fails
        call_count = 0
        def mock_operation():
            nonlocal call_count
            call_count += 1
            raise TimeoutError(f"Database timeout #{call_count}")

        with pytest.raises(TimeoutError, match="Database timeout #3"):
            executor._execute_with_retry(mock_operation)
        assert call_count == 3  # initial + 2 retries

    def test_retry_disabled_does_not_retry(self):
        """Test that retry_on_timeout=False prevents retries."""
        config = DatabaseConfig(retry_on_timeout=False, max_retries=3)
        executor = SerialDatabaseExecutor(config)

        call_count = 0
        def mock_operation():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Database timeout")

        with pytest.raises(TimeoutError, match="Database timeout"):
            executor._execute_with_retry(mock_operation)
        assert call_count == 1  # only initial call

    def test_retry_backoff_timing(self):
        """Test that exponential backoff timing is respected."""
        config = DatabaseConfig(retry_on_timeout=True, max_retries=2, retry_backoff_seconds=0.1)
        executor = SerialDatabaseExecutor(config)

        call_count = 0
        start_time = time.time()
        def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Database timeout")
            return "success"

        result = executor._execute_with_retry(mock_operation)
        elapsed = time.time() - start_time

        assert result == "success"
        # Should have waited: 0.1 + 0.2 = 0.3 seconds minimum
        assert elapsed >= 0.25  # Allow some tolerance

    def test_non_timeout_exceptions_not_retried(self):
        """Test that non-TimeoutError exceptions are not retried."""
        config = DatabaseConfig(retry_on_timeout=True, max_retries=3)
        executor = SerialDatabaseExecutor(config)

        call_count = 0
        def mock_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not a timeout error")

        with pytest.raises(ValueError, match="Not a timeout error"):
            executor._execute_with_retry(mock_operation)
        assert call_count == 1  # only initial call

    def test_max_retries_zero_disables_retries(self):
        """Test that max_retries=0 disables retries."""
        config = DatabaseConfig(retry_on_timeout=True, max_retries=0)
        executor = SerialDatabaseExecutor(config)

        call_count = 0
        def mock_operation():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Database timeout")

        with pytest.raises(TimeoutError, match="Database timeout"):
            executor._execute_with_retry(mock_operation)
        assert call_count == 1

    def test_invalid_config_values_use_defaults(self):
        """Test that invalid config values fall back to defaults."""
        # Test with None config
        executor = SerialDatabaseExecutor(None)
        assert executor._retry_on_timeout is True
        assert executor._max_retries == 3
        assert executor._retry_backoff == 1.0

        # Test with partial config
        config = DatabaseConfig()
        executor = SerialDatabaseExecutor(config)
        assert executor._retry_on_timeout is True
        assert executor._max_retries == 3
        assert executor._retry_backoff == 1.0

    @pytest.mark.asyncio
    async def test_async_retry_logic(self):
        """Test that async retry logic works correctly."""
        config = DatabaseConfig(retry_on_timeout=True, max_retries=1, retry_backoff_seconds=0.01)
        executor = SerialDatabaseExecutor(config)

        call_count = 0
        async def mock_async_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Async database timeout")
            return "async success"

        result = await executor._execute_with_retry_async(mock_async_operation)
        assert result == "async success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_retry_backoff_timing(self):
        """Test that async retry backoff timing works."""
        config = DatabaseConfig(retry_on_timeout=True, max_retries=1, retry_backoff_seconds=0.1)
        executor = SerialDatabaseExecutor(config)

        call_count = 0
        start_time = time.time()
        async def mock_async_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Async database timeout")
            return "async success"

        result = await executor._execute_with_retry_async(mock_async_operation)
        elapsed = time.time() - start_time

        assert result == "async success"
        assert call_count == 2  # initial + 1 retry
        assert elapsed >= 0.1  # at least the backoff time


class TestDatabaseConfigRetrySettings:
    """Test database config retry settings loading."""

    def test_retry_config_defaults(self):
        """Test default retry configuration values."""
        config = DatabaseConfig()
        assert config.retry_on_timeout is True
        assert config.max_retries == 3
        assert config.retry_backoff_seconds == 1.0

    def test_retry_config_custom_values(self):
        """Test custom retry configuration values."""
        config = DatabaseConfig(
            retry_on_timeout=False,
            max_retries=5,
            retry_backoff_seconds=2.5
        )
        assert config.retry_on_timeout is False
        assert config.max_retries == 5
        assert config.retry_backoff_seconds == 2.5

    @patch.dict('os.environ', {
        'CHUNKHOUND_DATABASE__RETRY_ON_TIMEOUT': 'false',
        'CHUNKHOUND_DATABASE__MAX_RETRIES': '10',
        'CHUNKHOUND_DATABASE__RETRY_BACKOFF_SECONDS': '0.5'
    })
    def test_retry_config_env_vars(self):
        """Test retry configuration from environment variables."""
        env_config = DatabaseConfig.load_from_env()
        config = DatabaseConfig(**env_config)
        assert config.retry_on_timeout is False
        assert config.max_retries == 10
        assert config.retry_backoff_seconds == 0.5

    @patch.dict('os.environ', {
        'CHUNKHOUND_DATABASE__RETRY_ON_TIMEOUT': 'invalid',
        'CHUNKHOUND_DATABASE__MAX_RETRIES': 'not-a-number',
        'CHUNKHOUND_DATABASE__RETRY_BACKOFF_SECONDS': 'also-not-a-number'
    })
    def test_retry_config_invalid_env_vars_ignored(self):
        """Test that invalid environment variables are silently ignored."""
        env_config = DatabaseConfig.load_from_env()
        config = DatabaseConfig(**env_config)
        # Should use defaults since env vars are invalid
        assert config.retry_on_timeout is True  # default
        assert config.max_retries == 3  # default
        assert config.retry_backoff_seconds == 1.0  # default

    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        # Valid configurations should work
        config = DatabaseConfig(max_retries=0)
        assert config.max_retries == 0

        config = DatabaseConfig(max_retries=100)
        assert config.max_retries == 100

        config = DatabaseConfig(retry_backoff_seconds=0.0)
        assert config.retry_backoff_seconds == 0.0

        # Negative values should be rejected by pydantic
        with pytest.raises(ValueError):
            DatabaseConfig(max_retries=-1)

        with pytest.raises(ValueError):
            DatabaseConfig(retry_backoff_seconds=-1.0)
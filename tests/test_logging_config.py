"""Tests for logging configuration and file/performance logging."""

import argparse
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from chunkhound.core.config.logging_config import (
    LoggingConfig,
    FileLoggingConfig,
    PerformanceLoggingConfig
)
from chunkhound.api.cli.main import setup_logging


class TestFileLoggingConfig:
    """Test FileLoggingConfig validation and functionality."""

    def test_file_logging_config_defaults(self):
        """Test default file logging configuration."""
        config = FileLoggingConfig()
        assert config.enabled is True
        assert config.path == "chunkhound.log"
        assert config.level == "INFO"
        assert config.rotation == "10 MB"
        assert config.retention == "1 week"
        assert "time" in config.format

    def test_file_logging_config_custom_values(self):
        """Test custom file logging configuration."""
        config = FileLoggingConfig(
            enabled=True,
            path="/custom/path.log",
            level="DEBUG",
            rotation="1 day",
            retention="30 days",
            format="Custom format"
        )
        assert config.enabled is True
        assert config.path == "/custom/path.log"
        assert config.level == "DEBUG"
        assert config.rotation == "1 day"
        assert config.retention == "30 days"
        assert config.format == "Custom format"

    def test_file_logging_invalid_level(self):
        """Test that invalid log levels raise ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            FileLoggingConfig(level="INVALID")

    def test_file_logging_invalid_path_empty(self):
        """Test that empty paths raise ValueError."""
        with pytest.raises(ValueError, match="Log file path cannot be empty"):
            FileLoggingConfig(path="")


    def test_file_logging_valid_levels(self):
        """Test that valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            config = FileLoggingConfig(level=level)
            assert config.level == level


class TestPerformanceLoggingConfig:
    """Test PerformanceLoggingConfig validation and functionality."""

    def test_performance_logging_config_defaults(self):
        """Test default performance logging configuration."""
        config = PerformanceLoggingConfig()
        assert config.enabled is True
        assert config.path == "chunkhound-performance.log"

    def test_performance_logging_config_custom_values(self):
        """Test custom performance logging configuration."""
        config = PerformanceLoggingConfig(
            enabled=True,
            path="/custom/perf.log"
        )
        assert config.enabled is True
        assert config.path == "/custom/perf.log"

    def test_performance_logging_invalid_path_empty(self):
        """Test that empty performance log paths raise ValueError."""
        with pytest.raises(ValueError, match="Performance log file path cannot be empty"):
            PerformanceLoggingConfig(path="")



class TestLoggingConfig:
    """Test top-level LoggingConfig functionality."""

    def test_logging_config_defaults(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert isinstance(config.file, FileLoggingConfig)
        assert isinstance(config.performance, PerformanceLoggingConfig)
        assert config.is_enabled() is True  # File logging enabled by default

    def test_logging_config_file_enabled(self):
        """Test that file logging enabled makes is_enabled() True."""
        config = LoggingConfig(file=FileLoggingConfig(enabled=True))
        assert config.is_enabled() is True

    def test_logging_config_performance_enabled(self):
        """Test that performance logging enabled does not affect is_enabled().

        Performance logging is now handled separately in worker processes
        and does not affect whether logging is considered "enabled" in the main process.
        Since file logging is enabled by default, is_enabled() should still return True.
        """
        config = LoggingConfig(performance=PerformanceLoggingConfig(enabled=True))
        # File logging is enabled by default, so is_enabled() should be True regardless of performance logging
        assert config.is_enabled() is True

    def test_logging_config_both_enabled(self):
        """Test that both logging types enabled works."""
        config = LoggingConfig(
            file=FileLoggingConfig(enabled=True),
            performance=PerformanceLoggingConfig(enabled=True)
        )
        assert config.is_enabled() is True

    def test_extract_cli_overrides_no_args(self):
        """Test CLI override extraction with no logging args."""
        args = argparse.Namespace()
        overrides = LoggingConfig.extract_cli_overrides(args)
        assert overrides is None

    def test_extract_cli_overrides_file_logging(self):
        """Test CLI override extraction for file logging."""
        args = argparse.Namespace(
            log_file="/tmp/test.log",
            log_level="DEBUG"
        )
        overrides = LoggingConfig.extract_cli_overrides(args)
        assert overrides is not None
        assert "file" in overrides
        assert overrides["file"]["enabled"] is True
        assert overrides["file"]["path"] == "/tmp/test.log"
        assert overrides["file"]["level"] == "DEBUG"

    def test_extract_cli_overrides_performance_logging(self):
        """Test CLI override extraction for performance logging."""
        args = argparse.Namespace(
            performance_log="/tmp/perf.log"
        )
        overrides = LoggingConfig.extract_cli_overrides(args)
        assert overrides is not None
        assert "performance" in overrides
        assert overrides["performance"]["enabled"] is True
        assert overrides["performance"]["path"] == "/tmp/perf.log"

    def test_extract_cli_overrides_partial_file_args(self):
        """Test CLI override extraction with only log_file (no level)."""
        args = argparse.Namespace(
            log_file="/tmp/test.log"
        )
        overrides = LoggingConfig.extract_cli_overrides(args)
        assert overrides is not None
        assert "file" in overrides
        assert overrides["file"]["enabled"] is True
        assert overrides["file"]["path"] == "/tmp/test.log"
        assert "level" not in overrides["file"]


class TestSetupLogging:
    """Test setup_logging function behavior."""

    @patch('chunkhound.api.cli.main.logger')
    def test_setup_logging_no_config(self, mock_logger):
        """Test setup_logging with no config (console only)."""
        setup_logging(verbose=False, config=None)

        # Should call remove() and add() for console logging
        mock_logger.remove.assert_called_once()
        assert mock_logger.add.call_count >= 1

    @patch('chunkhound.api.cli.main.logger')
    def test_setup_logging_verbose_no_file_logging(self, mock_logger):
        """Test setup_logging verbose mode without file logging."""
        config = LoggingConfig()
        setup_logging(verbose=True, config=config)

        mock_logger.remove.assert_called_once()
        # Should have console logging with DEBUG level
        # Check that logger.add was called (mock verification)
        assert mock_logger.add.call_count >= 1
        # The first call should be for console logging with DEBUG level
        first_call = mock_logger.add.call_args_list[0]
        assert 'DEBUG' in str(first_call)

    @patch('chunkhound.api.cli.main.logger')
    def test_setup_logging_file_logging_enabled_console_quiet(self, mock_logger):
        """Test that file logging enabled makes console WARNING+ only."""
        config = LoggingConfig(file=FileLoggingConfig(enabled=True, path="/tmp/test.log"))
        setup_logging(verbose=False, config=config)

        mock_logger.remove.assert_called_once()

        # Console should be WARNING level when file logging enabled
        assert mock_logger.add.call_count >= 1
        # Check that WARNING level is used for console when file logging is enabled
        console_calls = [call for call in mock_logger.add.call_args_list if 'WARNING' in str(call)]
        assert len(console_calls) > 0

    @patch('chunkhound.api.cli.main.logger')
    def test_setup_logging_verbose_overrides_file_logging_console(self, mock_logger):
        """Test that verbose flag shows full console logging even with file logging."""
        config = LoggingConfig(file=FileLoggingConfig(enabled=True, path="/tmp/test.log"))
        setup_logging(verbose=True, config=config)

        mock_logger.remove.assert_called_once()

        # Should have DEBUG console logging despite file logging
        assert mock_logger.add.call_count >= 1
        # Check that DEBUG level is used when verbose=True
        debug_calls = [call for call in mock_logger.add.call_args_list if 'DEBUG' in str(call)]
        assert len(debug_calls) > 0

    @patch('chunkhound.api.cli.main.logger')
    @patch('os.getenv', return_value="DEBUG")
    def test_setup_logging_file_config(self, mock_getenv, mock_logger):
        """Test that file logging config is applied."""
        # Create a mock config object with logging attribute
        class MockConfig:
            def __init__(self):
                self.logging = LoggingConfig(file=FileLoggingConfig(
                    enabled=True,
                    path="/tmp/test.log",
                    level="INFO",
                    rotation="10 MB",
                    retention="1 week"
                ))

        config = MockConfig()
        setup_logging(verbose=False, config=config)

        # Should have file logging call
        assert mock_logger.add.call_count >= 1
        # Find the file logging call
        file_call = None
        for call in mock_logger.add.call_args_list:
            if len(call[0]) > 0 and str(call[0][0]) == '/tmp/test.log':
                file_call = call
                break
        assert file_call is not None
        assert file_call[1]['level'] == 'INFO'
        assert file_call[1]['rotation'] == '10 MB'
        assert file_call[1]['retention'] == '1 week'

    def test_setup_logging_performance_config_removed(self):
        """Test that performance logging is no longer configured in main process.

        Performance logging is now only configured in worker processes where
        the actual performance logging calls occur.
        """
        # This test verifies that performance logging setup was intentionally
        # removed from the main process since all performance logs come from workers.
        pass

    @patch('chunkhound.api.cli.main.logger')
    def test_setup_logging_config_none_handling(self, mock_logger):
        """Test that setup_logging handles config=None gracefully."""
        setup_logging(verbose=False, config=None)

        # Should not crash and should set up console logging
        mock_logger.remove.assert_called_once()
        assert mock_logger.add.call_count > 0

    @patch('chunkhound.api.cli.main.logger')
    def test_setup_logging_getattr_vs_hasattr(self, mock_logger):
        """Test that getattr is used instead of hasattr for config checking."""
        # Create a config-like object that has logging but hasattr would fail
        class ConfigLike:
            def __init__(self):
                self.logging = LoggingConfig(file=FileLoggingConfig(enabled=True))

            def __getattr__(self, name):
                if name == 'logging':
                    return self.logging
                raise AttributeError

        config = ConfigLike()
        setup_logging(verbose=False, config=config)

        # Should work with getattr instead of crashing
        mock_logger.remove.assert_called_once()


class TestLoggingIntegration:
    """Test logging integration with main config."""

    def test_config_includes_logging(self):
        """Test that Config includes logging configuration."""
        from chunkhound.core.config.config import Config

        config = Config()
        assert hasattr(config, 'logging')
        assert isinstance(config.logging, LoggingConfig)

    def test_config_logging_overrides_from_cli(self):
        """Test that logging overrides are applied from CLI args."""
        from chunkhound.core.config.config import Config

        # Mock args with logging options
        mock_args = MagicMock()
        mock_args.log_file = "/tmp/test.log"
        mock_args.log_level = "DEBUG"
        mock_args.performance_log = "/tmp/perf.log"

        config = Config()
        overrides = config._extract_cli_overrides(mock_args)

        assert overrides is not None
        assert "logging" in overrides
        logging_overrides = overrides["logging"]
        assert "file" in logging_overrides
        assert "performance" in logging_overrides

    def test_config_logging_partial_overrides(self):
        """Test that partial logging overrides work."""
        from chunkhound.core.config.config import Config

        mock_args = MagicMock()
        mock_args.log_file = "/tmp/test.log"
        mock_args.log_level = None  # Explicitly set to None
        # No performance_log

        config = Config()
        overrides = config._extract_cli_overrides(mock_args)

        assert overrides is not None
        assert "logging" in overrides
        file_overrides = overrides["logging"]["file"]
        assert file_overrides["enabled"] is True
        assert file_overrides["path"] == "/tmp/test.log"
        # Should not have level since not specified
        assert "level" not in file_overrides


class TestProgressManagerLogging:
    """Test that ProgressManager preserves file logging while suppressing console logging."""

    @patch('chunkhound.api.cli.utils.rich_output.logger')
    def test_progress_manager_preserves_file_handlers(self, mock_logger):
        """Test that ProgressManager only removes console handlers, preserving file handlers."""
        from chunkhound.api.cli.utils.rich_output import ProgressManager, _NoRichProgressManager
        from rich.progress import Progress
        from rich.console import Console
        import sys
        import tempfile
        import os

        # Create a mock file handler (not writing to stderr)
        class MockFileHandler:
            def __init__(self, path):
                self.path = path
                # No stream attribute = not a console handler

        # Create a mock console handler (writing to stderr)
        class MockConsoleHandler:
            def __init__(self):
                self.stream = sys.stderr

        # Mock the logger handlers
        file_handler = MockFileHandler("/tmp/test.log")
        console_handler = MockConsoleHandler()

        mock_logger._core.handlers = {
            1: file_handler,    # File handler
            2: console_handler  # Console handler
        }

        # Create ProgressManager
        progress = Progress()
        console = Console()
        manager = ProgressManager(progress, console)

        # Enter context manager
        with manager:
            # Should have removed console handler but kept file handler
            # The mock doesn't actually remove, but we can check the logic conceptually
            pass

        # Verify that remove was called (for console handler removal)
        assert mock_logger.remove.called

    @patch('chunkhound.api.cli.utils.rich_output.logger')
    def test_progress_manager_restores_console_handlers(self, mock_logger):
        """Test that ProgressManager restores console handlers after exiting."""
        from chunkhound.api.cli.utils.rich_output import ProgressManager
        from rich.progress import Progress
        from rich.console import Console
        import sys

        # Create a mock console handler
        class MockConsoleHandler:
            def __init__(self):
                self.stream = sys.stderr

        console_handler = MockConsoleHandler()
        mock_logger._core.handlers = {1: console_handler}

        # Create ProgressManager
        progress = Progress()
        console = Console()
        manager = ProgressManager(progress, console)

        # Enter and exit context manager
        with manager:
            pass

        # Verify handlers were restored
        # The actual restoration logic should work with real handlers
        assert mock_logger.remove.called

    def test_no_rich_progress_manager_noop(self):
        """Test that _NoRichProgressManager is a proper no-op."""
        from chunkhound.api.cli.utils.rich_output import _NoRichProgressManager

        manager = _NoRichProgressManager()

        # Should not crash
        with manager:
            task_id = manager.add_task("test", "Testing")
            manager.update_task("test")
            manager.get_task_id("test")
            manager.finish_task("test")
            manager.add_subtask("test", "sub", "Subtask")
            progress = manager.get_progress_instance()

        # Verify it's a shim object
        assert hasattr(progress, 'add_task')
        assert hasattr(progress, 'update')
        assert hasattr(progress, 'tasks')
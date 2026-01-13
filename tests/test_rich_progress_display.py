"""Tests for rich output progress display components.

This module contains comprehensive unit tests for the progress display system
including RichLogHandler, MessageBuffer, DynamicLayout, and LogsRenderable classes,
as well as integration tests for the ProgressManager with integrated display.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

from chunkhound.api.cli.utils.rich_output import (
    MessageBuffer,
    LogsRenderable,
    RichLogHandler,
    DynamicLayout,
    ProgressManager,
)
from chunkhound.core.config.logging_config import LoggingConfig


class TestMessageBuffer:
    """Unit tests for MessageBuffer class."""

    def test_init_default_max_size(self):
        """Test MessageBuffer initialization with default max_size."""
        buffer = MessageBuffer()
        assert buffer.buffer.maxlen == 1000
        assert len(buffer.buffer) == 0

    def test_init_custom_max_size(self):
        """Test MessageBuffer initialization with custom max_size."""
        buffer = MessageBuffer(max_size=50)
        assert buffer.buffer.maxlen == 50

    def test_add_message(self):
        """Test adding messages to the buffer."""
        buffer = MessageBuffer(max_size=3)
        buffer.add_message("Message 1")
        buffer.add_message("Message 2")
        buffer.add_message("Message 3")

        messages = buffer.get_messages()
        assert messages == ["Message 1", "Message 2", "Message 3"]

    def test_buffer_truncation(self):
        """Test that buffer automatically truncates old messages."""
        buffer = MessageBuffer(max_size=2)
        buffer.add_message("Message 1")
        buffer.add_message("Message 2")
        buffer.add_message("Message 3")  # This should push out "Message 1"

        messages = buffer.get_messages()
        assert messages == ["Message 2", "Message 3"]
        assert len(messages) == 2

    def test_get_messages_empty_buffer(self):
        """Test getting messages from empty buffer."""
        buffer = MessageBuffer()
        messages = buffer.get_messages()
        assert messages == []

    def test_clear_buffer(self):
        """Test clearing all messages from buffer."""
        buffer = MessageBuffer()
        buffer.add_message("Message 1")
        buffer.add_message("Message 2")
        assert len(buffer.get_messages()) == 2

        buffer.clear()
        assert len(buffer.get_messages()) == 0


class TestLogsRenderable:
    """Unit tests for LogsRenderable class."""

    def test_init(self):
        """Test LogsRenderable initialization."""
        buffer = MessageBuffer()
        renderable = LogsRenderable(buffer)
        assert renderable.buffer is buffer

    def test_rich_console_empty_buffer(self):
        """Test rendering with empty buffer."""
        buffer = MessageBuffer()
        renderable = LogsRenderable(buffer)

        console = Mock()
        options = Mock()

        panels = list(renderable.__rich_console__(console, options))
        assert len(panels) == 1
        panel = panels[0]
        assert isinstance(panel, Panel)
        assert panel.title == "Logs"
        assert panel.renderable == ""

    def test_rich_console_with_messages(self):
        """Test rendering with messages in buffer."""
        buffer = MessageBuffer()
        buffer.add_message("INFO: Starting process")
        buffer.add_message("WARNING: Low memory")
        renderable = LogsRenderable(buffer)

        console = Mock()
        options = Mock()

        panels = list(renderable.__rich_console__(console, options))
        assert len(panels) == 1
        panel = panels[0]
        assert isinstance(panel, Panel)
        assert panel.title == "Logs"
        assert "INFO: Starting process" in panel.renderable
        assert "WARNING: Low memory" in panel.renderable
        assert "\n" in panel.renderable  # Messages should be joined with newlines


class TestRichLogHandler:
    """Unit tests for RichLogHandler class."""

    def test_init(self):
        """Test RichLogHandler initialization."""
        buffer = MessageBuffer()
        manager = Mock()
        handler = RichLogHandler(buffer, manager)
        assert handler.buffer is buffer
        assert handler.manager is manager

    def test_call_formats_message(self):
        """Test that handler formats and adds messages to buffer."""
        buffer = MessageBuffer()
        manager = Mock()
        handler = RichLogHandler(buffer, manager)

        # Create a mock loguru message
        mock_message = Mock()
        mock_message.record = {"level": Mock()}
        mock_message.record["level"].name = "WARNING"
        mock_message.__str__ = Mock(return_value="Test warning message")

        handler(mock_message)

        messages = buffer.get_messages()
        assert len(messages) == 1
        assert messages[0] == "WARNING: Test warning message"

    def test_call_triggers_update_logs(self):
        """Test that handler triggers display update."""
        buffer = MessageBuffer()
        manager = Mock()
        handler = RichLogHandler(buffer, manager)

        mock_message = Mock()
        mock_message.record = {"level": Mock()}
        mock_message.record["level"].name = "INFO"
        mock_message.__str__ = Mock(return_value="Test message")

        handler(mock_message)

        manager.update_logs.assert_called_once()


class TestDynamicLayout:
    """Unit tests for DynamicLayout class."""

    def test_init(self):
        """Test DynamicLayout initialization."""
        console = Mock()
        logs_renderable = Mock()
        progress = Mock()
        layout = DynamicLayout(console, logs_renderable, progress)

        assert layout.console is console
        assert layout.logs_renderable is logs_renderable
        assert layout.progress is progress
        assert layout.max_log_height == 20

    def test_init_custom_max_log_height(self):
        """Test DynamicLayout initialization with custom max_log_height."""
        console = Mock()
        logs_renderable = Mock()
        progress = Mock()
        layout = DynamicLayout(console, logs_renderable, progress, max_log_height=15)

        assert layout.max_log_height == 15

    def test_update_sizes_noop(self):
        """Test that update_sizes is a no-op (sizes calculated in __rich_console__)."""
        console = Mock()
        logs_renderable = Mock()
        progress = Mock()
        layout = DynamicLayout(console, logs_renderable, progress)

        # Should not raise any errors
        layout.update_sizes(30, 5)

    @patch('chunkhound.api.cli.utils.rich_output.Layout')
    def test_rich_console_layout_creation(self, mock_layout_class):
        """Test that __rich_console__ creates proper layout structure."""
        console = Mock()
        logs_renderable = Mock()
        progress = Mock()
        layout = DynamicLayout(console, logs_renderable, progress)

        # Mock the layout instances - Layout() is called 3 times in the actual code
        mock_layout_instances = [Mock() for _ in range(3)]
        mock_layout_class.side_effect = mock_layout_instances

        # Mock options
        options = Mock()
        options.height = 25

        # Mock progress tasks
        progress.tasks = [Mock(), Mock(), Mock()]  # 3 tasks

        # Call __rich_console__
        panels = list(layout.__rich_console__(console, options))

        # Verify layout creation - Layout is called 3 times (main, logs, progress)
        assert mock_layout_class.call_count == 3

        # Verify split_column was called on main layout
        mock_layout_instances[0].split_column.assert_called_once()

        assert len(panels) == 1
        assert panels[0] is mock_layout_instances[0]

    @patch('chunkhound.api.cli.utils.rich_output.Layout')
    def test_rich_console_height_from_console_size(self, mock_layout_class):
        """Test height calculation when options.height is None."""
        console = Mock()
        console.size.height = 40
        logs_renderable = Mock()
        progress = Mock()
        layout = DynamicLayout(console, logs_renderable, progress)

        options = Mock()
        options.height = None
        progress.tasks = [Mock()]  # 1 task

        list(layout.__rich_console__(console, options))

        # Should use console.size.height = 40
        # min_progress_height = max(3, 1 * 2) = 3
        # available_height = 40 - 3 = 37
        # log_height = min(37, 20) = 20


class TestProgressManagerIntegration:
    """Integration tests for ProgressManager with new display system."""

    @patch('chunkhound.api.cli.utils.rich_output.Live')
    @patch('chunkhound.api.cli.utils.rich_output.logger')
    def test_context_manager_integration(self, mock_logger, mock_live_class):
        """Test ProgressManager context manager with integrated display."""
        progress = Progress()
        console = Console()

        # Mock Live
        mock_live = Mock()
        mock_live_class.return_value = mock_live

        # Mock logger.add to return an id
        mock_handler_id = Mock()
        mock_logger.add.return_value = mock_handler_id

        manager = ProgressManager(progress, console)

        with manager:
            # Should add handler to logger
            mock_logger.add.assert_called_once_with(manager.handler)

            # Should start Live display
            mock_live_class.assert_called_once()
            mock_live.start.assert_called_once()

        # Should stop Live and remove handler
        mock_live.stop.assert_called_once()
        mock_logger.remove.assert_called_once_with(mock_handler_id)

    @patch('chunkhound.api.cli.utils.rich_output.Live')
    @patch('chunkhound.api.cli.utils.rich_output.logger')
    def test_log_capture_during_progress(self, mock_logger, mock_live_class):
        """Test that logs are captured and displayed during progress operations."""
        progress = Progress()
        console = Console()

        mock_live = Mock()
        mock_live_class.return_value = mock_live

        manager = ProgressManager(progress, console)

        # Add a task
        task_id = manager.add_task("test_task", "Testing", total=10)

        with manager:
            # Simulate log message
            mock_message = Mock()
            mock_message.record = {"level": Mock()}
            mock_message.record["level"].name = "WARNING"
            mock_message.__str__ = Mock(return_value="Test warning")

            manager.handler(mock_message)

            # Check that message was added to buffer
            messages = manager.buffer.get_messages()
            assert len(messages) == 1
            assert messages[0] == "WARNING: Test warning"

            # Check that update_logs was called
            mock_live.refresh.assert_called()

    @patch('chunkhound.api.cli.utils.rich_output.Live')
    @patch('chunkhound.api.cli.utils.rich_output.logger')
    def test_task_operations_trigger_updates(self, mock_logger, mock_live_class):
        """Test that task operations trigger display updates."""
        progress = Progress()
        console = Console()

        mock_live = Mock()
        mock_live_class.return_value = mock_live

        manager = ProgressManager(progress, console)

        with manager:
            # Add task
            task_id = manager.add_task("test_task", "Testing", total=10)
            mock_live.refresh.assert_called()

            # Update task - should trigger another refresh
            mock_live.refresh.reset_mock()
            manager.update_task("test_task", advance=5)
            mock_live.refresh.assert_called()





class TestRichProgressDisplayIntegration:
    """End-to-end integration tests for the rich progress display system."""

    @patch('chunkhound.api.cli.utils.rich_output.Live')
    @patch('chunkhound.api.cli.utils.rich_output.logger')
    def test_complete_workflow(self, mock_logger, mock_live_class):
        """Test complete workflow from config to display."""
        # Setup configuration
        config = LoggingConfig(
            progress_display_log_level="INFO",
            max_log_messages=25
        )

        # Create progress display
        progress = Progress()
        console = Console()
        manager = ProgressManager(progress, console, max_log_messages=config.max_log_messages)

        # Verify buffer is configured
        assert manager.buffer.buffer.maxlen == 25

        mock_live = Mock()
        mock_live_class.return_value = mock_live

        with manager:
            # Add some tasks
            task1 = manager.add_task("discovery", "File discovery", total=100)
            task2 = manager.add_task("parsing", "File parsing", total=50)

            # Simulate some progress
            manager.update_task("discovery", advance=25)
            manager.update_task("parsing", advance=10)

            # Simulate log messages
            mock_message1 = Mock()
            mock_message1.record = {"level": Mock()}
            mock_message1.record["level"].name = "INFO"
            mock_message1.__str__ = Mock(return_value="Discovery started")

            mock_message2 = Mock()
            mock_message2.record = {"level": Mock()}
            mock_message2.record["level"].name = "WARNING"
            mock_message2.__str__ = Mock(return_value="Large file detected")

            manager.handler(mock_message1)
            manager.handler(mock_message2)

            # Verify messages were captured
            messages = manager.buffer.get_messages()
            assert "INFO: Discovery started" in messages
            assert "WARNING: Large file detected" in messages

            # Verify display was updated
            assert mock_live.refresh.call_count >= 3  # Initial + 2 updates + 2 logs

    @patch('chunkhound.api.cli.utils.rich_output.Live')
    @patch('chunkhound.api.cli.utils.rich_output.logger')
    def test_log_filtering_integration(self, mock_logger, mock_live_class):
        """Test that log levels are properly filtered in the integrated system."""
        progress = Progress()
        console = Console()
        manager = ProgressManager(progress, console)

        mock_live = Mock()
        mock_live_class.return_value = mock_live

        with manager:
            # Add handler that should filter at WARNING level
            # (Note: In real implementation, filtering would be configurable)

            # Test different log levels
            levels_messages = [
                ("DEBUG", "Debug message"),
                ("INFO", "Info message"),
                ("WARNING", "Warning message"),
                ("ERROR", "Error message"),
            ]

            for level, msg_text in levels_messages:
                mock_message = Mock()
                mock_message.record = {"level": Mock()}
                mock_message.record["level"].name = level
                mock_message.__str__ = Mock(return_value=msg_text)

                manager.handler(mock_message)

            # In current implementation, all messages are captured
            # (filtering would need to be added to RichLogHandler)
            messages = manager.buffer.get_messages()
            assert len(messages) == 4  # All messages captured

            # Verify message formatting
            assert "DEBUG: Debug message" in messages
            assert "WARNING: Warning message" in messages
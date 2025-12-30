"""Modern Rich-based output formatting utilities for ChunkHound CLI commands."""

import sys
from collections import deque
from typing import Any, Literal

import rich.box
from loguru import logger
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


# Constants for fallback message prefixes
class MessagePrefixes:
    """Constants for consistent message prefixes in fallback mode."""

    INFO = "[INFO]"
    SUCCESS = "[SUCCESS]"
    WARN = "[WARN]"
    ERROR = "[ERROR]"
    DEBUG = "[DEBUG]"
    PROGRESS = "[PROGRESS]"


class RichOutputFormatter:
    """Modern terminal UI formatter using Rich library."""

    def __init__(self, verbose: bool = False):
        """Initialize Rich output formatter.

        Args:
            verbose: Whether to enable verbose output
        """
        self.verbose = verbose
        self._terminal_compatible = self._check_terminal_compatibility()
        self.console = Console() if self._terminal_compatible else None
        self._progress: Progress | None = None
        self._live: Live | None = None

    def _check_terminal_compatibility(self) -> bool:
        """Check if terminal supports Rich formatting."""
        import os
        import sys

        # Check for explicit Rich disable
        if os.environ.get("CHUNKHOUND_NO_RICH"):
            return False

        # Basic checks for terminal compatibility
        try:
            # Check if stdout is a TTY
            if not sys.stdout.isatty():
                return False

            # Check for common terminal variables
            term = os.environ.get("TERM", "")

            # Skip Rich for very basic terminals
            if term in ["dumb", "unknown"]:
                return False

            # Test Rich console creation
            try:
                test_console = Console()
                # Simple test - try to create a text object
                from rich.text import Text

                Text("test")
                return True
            except Exception:
                return False

        except Exception:
            return False

    def _safe_print(self, message: str, fallback_prefix: str = "") -> None:
        """Safely print with Rich or fallback to plain text."""
        if self._terminal_compatible and self.console is not None:
            try:
                self.console.print(message)
                return
            except Exception:
                # Rich failed, fall through to plain print
                pass

        # Fallback to plain print
        # Note: message is already escaped when passed to _safe_print from other methods
        if fallback_prefix:
            print(f"{fallback_prefix} {message}")
        else:
            print(message)

    def info(self, message: str) -> None:
        """Print an info message."""
        self._safe_print(f"[blue][INFO][/blue] {escape(message)}", MessagePrefixes.INFO)

    def success(self, message: str) -> None:
        """Print a success message."""
        self._safe_print(
            f"[green][SUCCESS][/green] {escape(message)}", MessagePrefixes.SUCCESS
        )

    def warning(self, message: str) -> None:
        """Print a warning message."""
        self._safe_print(
            f"[yellow][WARN][/yellow] {escape(message)}", MessagePrefixes.WARN
        )

    def error(self, message: str) -> None:
        """Print an error message."""
        import os

        # Skip stderr output in MCP mode to avoid JSON-RPC interference
        if not os.environ.get("CHUNKHOUND_MCP_MODE"):
            self._safe_print(
                f"[red][ERROR][/red] {escape(message)}", MessagePrefixes.ERROR
            )

    def verbose_info(self, message: str) -> None:
        """Print a verbose info message if verbose mode is enabled."""
        if self.verbose:
            self._safe_print(
                f"[cyan][DEBUG][/cyan] {escape(message)}", MessagePrefixes.DEBUG
            )

    def progress_indicator(self, message: str) -> None:
        """Print a progress indicator message."""
        self._safe_print(
            f"[cyan][PROGRESS][/cyan] {escape(message)}", MessagePrefixes.PROGRESS
        )

    def safe_progress_indicator(self, message: str) -> None:
        """
        Display progress indicator with comprehensive error handling.

        This method provides an extra layer of safety beyond the standard
        progress_indicator() method for cases where Rich formatting might
        fail in unexpected ways.

        Args:
            message: Progress message to display
        """
        try:
            self.progress_indicator(message)
        except Exception:
            # Ultimate fallback - bypass all Rich formatting
            # Note: escape() not needed here since this is typically internal progress messages
            print(f"{MessagePrefixes.PROGRESS} {message}")

    def section_header(self, title: str) -> None:
        """Print a section header with consistent formatting."""
        if self._terminal_compatible and self.console is not None:
            try:
                from rich.panel import Panel

                self.console.print(Panel(title, style="bold cyan", padding=(0, 1)))
                return
            except Exception:
                pass

        # Fallback to simple header
        print(f"\n=== {title} ===\n")

    def bullet_list(self, items: list[str], indent: int = 2) -> None:
        """Print a clean bullet list."""
        for item in items:
            self._safe_print(f"{'  ' * indent}- {item}", f"{'  ' * indent}-")

    def json_output(self, data: dict[str, Any]) -> None:
        """Print data as formatted JSON."""
        import json

        from rich.syntax import Syntax

        json_str = json.dumps(data, indent=2, default=str)
        syntax = Syntax(json_str, "json", theme="monokai")
        if self.console is not None:
            self.console.print(syntax)
        else:
            print(json_str)

    def box_section(
        self, title: str, content: list[tuple[str, str]], width: int = 50
    ) -> None:
        """Print a bordered section with key-value pairs."""
        from rich.table import Table

        table = Table(title=title, show_header=False, box=rich.box.ROUNDED)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        for key, value in content:
            # Truncate long values
            if len(value) > width - len(key) - 5:
                value = value[: width - len(key) - 8] + "..."
            table.add_row(key, value)

        if self.console is not None:
            self.console.print(table)
        else:
            print(f"\n{title}")
            for key, value in content:
                print(f"  {key}: {value}")

    def table_header(self, headers: list[str], widths: list[int] | None = None) -> None:
        """Print a table header - use Rich Table instead."""
        # This is a legacy method - recommend using Rich Table directly
        from rich.table import Table

        table = Table(show_header=True)
        for header in headers:
            table.add_column(header)
        # Note: This just prints the header structure
        # Actual data rows should be added separately
        if self.console is not None:
            self.console.print(table)
        else:
            print(" | ".join(headers))

    def startup_info(
        self, version: str, directory: str, database: str, config: dict[str, Any]
    ) -> None:
        """Display startup information in a styled panel."""
        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style="cyan")
        info_table.add_column()

        info_table.add_row("Version:", f"[green]{version}[/green]")
        info_table.add_row("Directory:", f"[blue]{directory}[/blue]")
        info_table.add_row("Database:", f"[magenta]{database}[/magenta]")

        # Add provider info if available
        if hasattr(config, "embedding") and config.embedding:
            provider = config.embedding.provider
            model = getattr(config.embedding, "model", "default")
            info_table.add_row("Provider:", f"[yellow]{provider}[/yellow] ({model})")

        panel = Panel(
            info_table,
            title="[bold cyan]ChunkHound Indexing[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
        if self.console is not None:
            self.console.print(panel)
        else:
            print("ChunkHound Indexing")
            print(f"Version: {version}")
            print(f"Directory: {directory}")
            print(f"Database: {database}")
            if hasattr(config, "embedding") and config.embedding:
                provider = config.embedding.provider
                model = getattr(config.embedding, "model", "default")
                print(f"Provider: {provider} ({model})")

    def create_progress_display(self, max_log_messages: int | None = None) -> "ProgressManager":
        """Create a modern progress display with multiple bars.

        Args:
            max_log_messages: Maximum number of log messages to buffer. If None, uses default.
        """

        # Fallback: no Rich/TTY support → return a no-op manager
        if not self._terminal_compatible or self.console is None:
            return _NoRichProgressManager()

        # Create custom text columns that handle missing fields gracefully
        def render_field(
            task, field_name: str, default: str = "", style: str = ""
        ) -> str:
            try:
                return task.fields.get(field_name, default)
            except (AttributeError, KeyError):
                return default

        class SafeTextColumn(TextColumn):
            def __init__(
                self,
                field_name: str,
                default: str = "",
                style: str = "",
                justify: Literal["default", "left", "center", "right", "full"] = "left",
                max_width: int | None = None,
            ) -> None:
                self.field_name = field_name
                self.default = default
                self.max_width = max_width
                # Use a simple format string that we'll handle ourselves
                super().__init__("", style=style, justify=justify)

            def render(self, task) -> Text:
                value = (
                    task.fields.get(self.field_name, self.default)
                    if hasattr(task, "fields")
                    else self.default
                )

                # Truncate value if max_width is specified and exceeded
                if self.max_width and len(value) > self.max_width:
                    value = value[:self.max_width - 3] + "..."

                return Text(value, style=self.style)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            SafeTextColumn("speed", "", style="green", max_width=20),
            SafeTextColumn("info", "", style="dim", max_width=40),
            console=self.console,
            expand=False,
            transient=False,  # Don't make progress disappear when complete
        )

        return ProgressManager(progress, self.console or Console(), max_log_messages=max_log_messages)

    def completion_summary(self, stats: dict[str, Any], processing_time: float) -> None:
        """Display completion summary in a styled panel."""
        # Create summary table
        summary_table = Table.grid(padding=(0, 2))
        summary_table.add_column(style="cyan")
        summary_table.add_column()

        summary_table.add_row(
            "Processed:", f"[green]{stats.get('files_processed', 0)}[/green] files"
        )
        summary_table.add_row(
            "Skipped:", f"[yellow]{stats.get('files_skipped', 0)}[/yellow] files"
        )
        if stats.get("skipped_unchanged", 0) > 0:
            summary_table.add_row(
                "  └─ Unchanged:", f"[yellow]{stats.get('skipped_unchanged', 0)}[/yellow] files"
            )
        if stats.get("skipped_filtered", 0) > 0:
            summary_table.add_row(
                "  └─ Filtered:", f"[yellow]{stats.get('skipped_filtered', 0)}[/yellow] files"
            )
        summary_table.add_row(
            "Errors:", f"[red]{stats.get('files_errors', 0)}[/red] files"
        )
        summary_table.add_row(
            "Total chunks:", f"[blue]{stats.get('chunks_created', 0)}[/blue]"
        )

        if "embeddings_generated" in stats:
            summary_table.add_row(
                "Embeddings:", f"[magenta]{stats['embeddings_generated']}[/magenta]"
            )

        summary_table.add_row("Time:", f"[cyan]{processing_time:.2f}s[/cyan]")

        # Add cleanup stats if any
        if stats.get("cleanup_deleted_files", 0) > 0:
            summary_table.add_row(
                "Cleaned files:", f"[yellow]{stats['cleanup_deleted_files']}[/yellow]"
            )
        if stats.get("cleanup_deleted_chunks", 0) > 0:
            summary_table.add_row(
                "Cleaned chunks:", f"[yellow]{stats['cleanup_deleted_chunks']}[/yellow]"
            )

        panel = Panel(
            summary_table,
            title="[bold green]Processing Complete[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
        # Render main panel
        if self.console is not None:
            self.console.print(panel)
        else:
            print("Processing Complete")
            print(f"Processed: {stats.get('files_processed', 0)} files")
            print(f"Skipped: {stats.get('files_skipped', 0)} files")
            print(f"Errors: {stats.get('files_errors', 0)} files")
            print(f"Total chunks: {stats.get('chunks_created', 0)}")
            if "embeddings_generated" in stats:
                print(f"Embeddings: {stats['embeddings_generated']}")
            print(f"Time: {processing_time:.2f}s")

        # If we have a list of files skipped due to timeout, display them
        skipped_timeouts = stats.get("skipped_due_to_timeout", [])
        if skipped_timeouts:
            if self.console is not None:
                timeout_table = Table.grid(padding=(0, 1))
                timeout_table.add_column(style="yellow")
                for fp in skipped_timeouts:
                    timeout_table.add_row(fp)
                timeout_panel = Panel(
                    timeout_table,
                    title=f"[bold yellow]Skipped Due to Timeout ({len(skipped_timeouts)})[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 2),
                )
                self.console.print(timeout_panel)
            else:
                print(
                    f"Skipped Due to Timeout ({len(skipped_timeouts)}):\n  "
                    + "\n  ".join(str(p) for p in skipped_timeouts)
                )

    def initial_stats_panel(self, stats: dict[str, Any]) -> None:
        """Display initial database statistics."""
        stats_table = Table.grid(padding=(0, 1))
        stats_table.add_column(style="dim")
        stats_table.add_column()

        stats_table.add_row("Files:", f"{stats.get('files', 0)}")
        stats_table.add_row("Chunks:", f"{stats.get('chunks', 0)}")
        stats_table.add_row("Embeddings:", f"{stats.get('embeddings', 0)}")

        if self.console is not None:
            self.console.print(
                f"[dim]Initial stats: {stats.get('files', 0)} files, {stats.get('chunks', 0)} chunks, {stats.get('embeddings', 0)} embeddings[/dim]"
            )
        else:
            print(
                f"Initial stats: {stats.get('files', 0)} files, {stats.get('chunks', 0)} chunks, {stats.get('embeddings', 0)} embeddings"
            )


class ProgressManager:
    """Manages multiple progress bars with Rich."""

    def __init__(self, progress: Progress, console: Console, max_log_messages: int = 1000):
        self.progress = progress
        self.console = console
        self._tasks: dict[str, TaskID] = {}
        self._live: Live | None = None
        self.buffer = MessageBuffer(max_size=max_log_messages)
        self.logs_renderable = LogsRenderable(self.buffer)
        self.layout = DynamicLayout(console, self.logs_renderable, self.progress)
        self.handler = RichLogHandler(self.buffer, self)
        self._handler_id: int | None = None

    def __enter__(self) -> "ProgressManager":
        self._handler_id = logger.add(self.handler)
        self._live = Live(self.layout, console=self.console, refresh_per_second=10)
        self._live.start()
        self.update_logs()  # Initial update
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if self._live:
                self._live.stop()
        finally:
            if self._handler_id is not None:
                logger.remove(self._handler_id)

    def update_logs(self):
        if self._live:
            self._live.refresh()

    def add_task(
        self,
        name: str,
        description: str,
        total: int | None = None,
        speed: str = "",
        info: str = "",
    ) -> TaskID:
        """Add a new progress task."""
        task_id = self.progress.add_task(
            description, total=total, speed=speed, info=info
        )
        self._tasks[name] = task_id
        return task_id

    def update_task(
        self,
        name: str,
        advance: int = 1,
        description: str | None = None,
        total: int | None = None,
        speed: str = "",
        **fields: Any,
    ) -> None:
        """Update a progress task."""
        if name in self._tasks:
            update_kwargs = {"advance": advance}
            if description:
                update_kwargs["description"] = description
            if total is not None:
                update_kwargs["total"] = total
            if speed:
                fields["speed"] = speed
            if fields:
                update_kwargs.update(fields)

            self.progress.update(self._tasks[name], **update_kwargs)
            self.update_logs()

    def get_task_id(self, name: str) -> TaskID | None:
        """Get task ID by name."""
        return self._tasks.get(name)

    def finish_task(self, name: str) -> None:
        """Mark a task as finished."""
        if name in self._tasks:
            task_id = self._tasks[name]
            # Get current completed/total to set to 100%
            task = self.progress.tasks[task_id]
            if task.total:
                self.progress.update(task_id, completed=task.total)

    def add_subtask(
        self,
        parent_name: str,
        name: str,
        description: str,
        total: int | None = None,
        indent_level: int = 1,
    ) -> TaskID:
        """Add a subtask under a parent task with visual hierarchy."""
        # Create indented description for visual hierarchy
        indent = "  " + "└─ " if indent_level == 1 else "    " * indent_level + "└─ "
        task_id = self.progress.add_task(
            f"{indent}{description}", total=total, speed="", info=""
        )
        self._tasks[name] = task_id
        return task_id

    def get_progress_instance(self) -> Progress:
        """Get the underlying Progress instance for service layer use."""
        return self.progress


class _NoRichProgressManager:
    """No-op progress manager for terminals that don't support Rich/output encoding.

    Provides the same methods but avoids creating any Rich objects. Used on
    Windows consoles with non-UTF8 code pages and non-TTY environments to
    prevent Unicode/encoding failures (e.g., bullets, box characters).
    """

    def __enter__(self) -> "_NoRichProgressManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def add_task(self, *args, **kwargs):  # noqa: ANN001
        return 0

    def update_task(self, *args, **kwargs) -> None:  # noqa: ANN001
        return None

    def get_task_id(self, *args, **kwargs):  # noqa: ANN001
        return None

    def finish_task(self, *args, **kwargs) -> None:  # noqa: ANN001
        return None

    def add_subtask(self, *args, **kwargs):  # noqa: ANN001
        return 0

    def get_progress_instance(self):  # noqa: ANN001
        # Return a minimal shim object that mimics the Rich Progress API
        # used by service layers (add_task, advance, update, tasks mapping).
        class _Shim:
            def __init__(self) -> None:
                self._next_id = 1
                # Minimal task object with .total and .completed attributes
                class _Task:
                    def __init__(self, total: int | None = None) -> None:
                        self.total = total
                        self.completed = 0

                self._Task = _Task
                self.tasks: dict[int, _Task] = {}

            def add_task(  # noqa: ANN001
                self, description: str, total: int | None = None, **_: Any
            ) -> int:
                task_id = self._next_id
                self._next_id += 1
                self.tasks[task_id] = self._Task(total)
                return task_id

            def update(self, task_id: int, **kwargs: Any) -> None:  # noqa: ANN001, D401
                # Support patterns used in codebase: advance, total, completed
                task = self.tasks.get(task_id)
                if not task:
                    return
                advance = int(kwargs.pop("advance", 0)) if "advance" in kwargs else 0
                if advance:
                    task.completed += advance
                if "total" in kwargs and kwargs["total"] is not None:
                    task.total = int(kwargs["total"])
                if "completed" in kwargs and kwargs["completed"] is not None:
                    task.completed = int(kwargs["completed"])
                # Ignore any other Rich-specific fields (description, speed, info)
                return None

            def advance(self, task_id: int, step: int = 1) -> None:  # noqa: ANN001
                task = self.tasks.get(task_id)
                if not task:
                    return
                task.completed += int(step)
                return None

        return _Shim()


class MessageBuffer:
    """Manages log message storage with automatic truncation using deque."""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)

    def add_message(self, message: str) -> None:
        """Add a message to the buffer."""
        self.buffer.append(message)

    def get_messages(self) -> list[str]:
        """Get all messages in the buffer."""
        return list(self.buffer)

    def clear(self) -> None:
        """Clear all messages from the buffer."""
        self.buffer.clear()


class LogsRenderable:
    """Renderable that displays the current log messages from the buffer."""

    def __init__(self, buffer: MessageBuffer):
        self.buffer = buffer

    def __rich_console__(self, console, options):
        messages = self.buffer.get_messages()
        content = "\n".join(messages) if messages else ""
        yield Panel(content, title="Logs")


class RichLogHandler:
    """Custom loguru handler that captures messages and triggers display updates."""

    def __init__(self, buffer: MessageBuffer, manager: "ProgressManager"):
        self.buffer = buffer
        self.manager = manager

    def __call__(self, message):
        """Handle a loguru message by adding it to the buffer and updating display."""
        # Format the message similar to loguru's default
        formatted = f"{message.record['level'].name}: {str(message)}"
        self.buffer.add_message(formatted)
        self.manager.update_logs()


class DynamicLayout:
    """Manages a Rich Layout with vertical split for logs and progress bars.

    Provides dynamic sizing logic that calculates panel heights based on terminal
    dimensions, number of progress tasks, and available space.
    """

    def __init__(self, console: Console, logs_renderable, progress, max_log_height: int = 20):
        """Initialize the dynamic layout.

        Args:
            console: Rich console instance
            logs_renderable: Renderable for logs
            progress: Progress instance
            max_log_height: Maximum height for the log panel
        """
        self.console = console
        self.logs_renderable = logs_renderable
        self.progress = progress
        self.max_log_height = max_log_height

    def update_sizes(self, terminal_height: int, progress_tasks_count: int) -> None:
        """Update panel sizes based on terminal dimensions and progress tasks.

        Args:
            terminal_height: Current terminal height in lines
            progress_tasks_count: Number of active progress tasks
        """
        # Not needed, sizes are calculated in __rich_console__
        pass

    def __rich_console__(self, console, options):
        """Render the layout with logs at top and progress at bottom."""
        layout = Layout()
        logs_layout = Layout(renderable=self.logs_renderable)
        progress_layout = Layout(renderable=self.progress)
        layout.split_column(logs_layout, progress_layout)

        height = options.height or console.size.height
        progress_tasks_count = len(self.progress.tasks)
        min_progress_height = max(3, progress_tasks_count * 2)
        available_height = height - min_progress_height
        log_height = max(0, min(available_height, self.max_log_height))
        logs_layout.size = log_height

        yield layout


def format_stats(stats: Any) -> str:
    """Format database statistics for display.

    Args:
        stats: Statistics dictionary from database

    Returns:
        Formatted statistics string
    """
    if hasattr(stats, "__dict__"):
        stats_dict = stats.__dict__
    else:
        stats_dict = stats if isinstance(stats, dict) else {}

    files = stats_dict.get("files", 0)
    chunks = stats_dict.get("chunks", 0)
    embeddings = stats_dict.get("embeddings", 0)

    return f"{files} files, {chunks} chunks, {embeddings} embeddings"


def format_health_status(status: dict[str, Any]) -> str:
    """Format health status for display.

    Args:
        status: Health status dictionary

    Returns:
        Formatted status string
    """
    if status.get("healthy", False):
        response_time = status.get("response_time_ms", 0)
        return f"Healthy ({response_time}ms)"
    else:
        error = status.get("error", "Unknown error")
        return f"Unhealthy: {error}"

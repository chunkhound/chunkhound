"""Tree-based progress display for deep research operations.

Provides hierarchical, streaming progress visualization with relative timestamps.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any, TextIO

from chunkhound.utils.tree_formatter import build_tree_prefix


@dataclass
class ProgressEvent:
    """Structured progress event for tree display."""

    type: str  # Event type: node_start, search_semantic, llm_call, node_complete, etc.
    timestamp: float  # Relative seconds from start
    node_id: int | None  # BFS node identifier
    depth: int | None  # BFS depth level
    message: str  # Human-readable description
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional data


class TreeProgressDisplay:
    """Streaming tree-based progress display for concurrent operations.

    Features:
    - Hierarchical tree structure with box-drawing characters
    - Relative timestamps (+0.5s format)
    - Append-only streaming output
    - Thread-safe event queue
    - Sequential logging of concurrent operations
    """

    def __init__(self, output: TextIO = sys.stdout):
        """Initialize tree progress display.

        Args:
            output: Output stream for progress display
        """
        self.output = output
        self.start_time: float | None = None
        self._event_queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._node_depth_map: dict[int, int] = {}  # node_id -> depth
        self._node_parent_map: dict[int, int | None] = {}  # node_id -> parent_id
        self._running = False

    def start(self) -> None:
        """Start progress display and initialize timestamp baseline."""
        self.start_time = time.time()
        self._running = True

    def stop(self) -> None:
        """Stop progress display."""
        self._running = False

    def _format_timestamp(self, timestamp: float) -> str:
        """Format relative timestamp with fixed width for proper alignment.

        Args:
            timestamp: Relative seconds from start

        Returns:
            Fixed-width formatted timestamp string (right-aligned)
        """
        # Fixed width ensures alignment even for hour-long operations
        # Format: "+3600.0s" (9 chars) vs "+0.0s" (5 chars)
        # Right-align within 9-char width for column alignment
        return f"+{timestamp:7.1f}s"


    def _get_event_symbol(self, event_type: str) -> str:
        """Get visual symbol for event type.

        Args:
            event_type: Event type identifier

        Returns:
            Symbol string
        """
        symbols = {
            "main_start": "🔍",
            "main_info": "ℹ️",
            "depth_start": "📊",
            "poi_start": "📌",
            "poi_complete": "📍",
            "poi_failed": "⚠️",
            "node_start": "🔹",
            "query_expand": "🔄",
            "query_expand_complete": "✨",
            "search_semantic": "🔎",
            "search_regex": "📝",
            "search_regex_complete": "✓",
            "extract_symbols": "🏷️",
            "extract_symbols_complete": "📋",
            "read_files": "📖",
            "read_files_complete": "📄",
            "llm_followup": "🤖",
            "llm_followup_complete": "💡",
            "llm_synthesis": "🧠",
            "llm_synthesis_complete": "✅",
            "node_complete": "✅",
            "node_terminated": "⏹️",
            "synthesis_start": "🧩",
            "synthesis_validate": "🔍",
            "synthesis_complete": "✨",
            "main_complete": "🎉",
            "error": "❌",
            "evidence_ledger": "📊",
        }
        return symbols.get(event_type, "•")

    def _format_metadata(self, metadata: dict[str, Any]) -> str:
        """Format metadata for display.

        Args:
            metadata: Event metadata dictionary

        Returns:
            Formatted metadata string
        """
        if not metadata:
            return ""

        parts = []
        for key, value in metadata.items():
            if key in (
                "chunks",
                "files",
                "children",
                "tokens",
                "queries",
                "symbols",
                "constants_count",
                "facts_count",
            ):
                parts.append(f"{key}={value}")
            elif key == "max_completion_tokens":
                parts.append(f"max_completion_tokens={value}")
            elif key == "duration":
                parts.append(f"{value:.2f}s")

        return f" ({', '.join(parts)})" if parts else ""

    async def emit_event(
        self,
        event_type: str,
        message: str,
        node_id: int | None = None,
        depth: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a progress event for display.

        Args:
            event_type: Event type identifier
            message: Human-readable event description
            node_id: Optional BFS node ID
            depth: Optional BFS depth level
            metadata: Optional additional event data
        """
        if not self._running or self.start_time is None:
            return

        timestamp = time.time() - self.start_time
        event = ProgressEvent(
            type=event_type,
            timestamp=timestamp,
            node_id=node_id,
            depth=depth,
            message=message,
            metadata=metadata or {},
        )

        await self._event_queue.put(event)
        await self._render_event(event)

    async def _render_event(self, event: ProgressEvent) -> None:
        """Render a single event to output stream.

        Args:
            event: Progress event to render
        """
        async with self._lock:
            # Track node depth for tree structure
            if event.node_id is not None and event.depth is not None:
                self._node_depth_map[event.node_id] = event.depth

            # Build output line
            timestamp_str = self._format_timestamp(event.timestamp)
            symbol = self._get_event_symbol(event.type)
            metadata_str = self._format_metadata(event.metadata)

            # Determine indentation based on depth
            depth = event.depth if event.depth is not None else 0
            tree_prefix = build_tree_prefix(depth) if depth > 0 else ""

            # Format line: [timestamp] prefix symbol message (metadata)
            prefix = f"[{timestamp_str}] {tree_prefix}{symbol} "
            line = f"{prefix}{event.message}{metadata_str}\n"

            # Write to output
            self.output.write(line)
            self.output.flush()

            # Print evidence table on separate lines if present
            if event.metadata.get("evidence_table"):
                # Indent table to align with tree structure
                table_indent = " " * (len(f"[{timestamp_str}] ") + len(tree_prefix))
                for table_line in event.metadata["evidence_table"].split("\n"):
                    if table_line.strip():  # Skip empty lines
                        self.output.write(f"{table_indent}{table_line}\n")
                self.output.flush()

    def __enter__(self) -> "TreeProgressDisplay":
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.stop()

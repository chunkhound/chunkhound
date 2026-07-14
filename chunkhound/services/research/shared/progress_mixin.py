"""Mixin for emitting progress events to the terminal display."""

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay


class ProgressEmitterMixin:
    """Mixin that adds _emit_event to a service class.

    Requires the subclass to set self._progress in its __init__.
    If the subclass sets self._progress_lock (an asyncio.Lock), emits are
    serialized under that lock — needed when multiple coroutines emit concurrently.

    Subclasses that emit exclusively nested step events (e.g. depth=2) can set
    _default_depth as a class variable instead of passing depth= on every call.
    """

    _progress: "TreeProgressDisplay | None" = None
    _progress_lock: "asyncio.Lock | None" = None
    _default_depth: int | None = None  # override in subclass for nested step events

    async def _emit_event(
        self,
        event_type: str,
        message: str,
        depth: int | None = None,
        node_id: int | None = None,
        **metadata: Any,
    ) -> None:
        """Emit a progress event to the terminal display.

        Args:
            event_type: Event type identifier (e.g. "gap_step")
            message: Human-readable description
            depth: Tree indentation depth. Falls back to _default_depth if None.
            node_id: Optional BFS node ID for tree placement
            **metadata: Additional data (e.g. duration=0.62)
        """
        if not self._progress:
            return
        effective_depth = depth if depth is not None else self._default_depth
        if self._progress_lock is not None:
            async with self._progress_lock:
                await self._progress.emit_event(
                    event_type=event_type,
                    message=message,
                    node_id=node_id,
                    depth=effective_depth,
                    metadata=metadata,
                )
        else:
            await self._progress.emit_event(
                event_type=event_type,
                message=message,
                node_id=node_id,
                depth=effective_depth,
                metadata=metadata,
            )

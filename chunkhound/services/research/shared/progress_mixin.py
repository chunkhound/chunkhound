"""Mixin for emitting progress events to the terminal display."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay


class ProgressEmitterMixin:
    """Mixin that adds _emit_event to a service class.

    Requires the subclass to set self._progress in its __init__.
    """

    _progress: "TreeProgressDisplay | None" = None

    async def _emit_event(
        self,
        event_type: str,
        message: str,
        depth: int = 2,
        **metadata: Any,
    ) -> None:
        """Emit a progress event to the terminal display.

        Args:
            event_type: Event type identifier (e.g. "gap_step")
            message: Human-readable description
            depth: Tree indentation depth (2 = nested under the current phase)
            **metadata: Additional data (e.g. duration=0.62)
        """
        if not self._progress:
            return
        await self._progress.emit_event(
            event_type=event_type,
            message=message,
            depth=depth,
            metadata=metadata,
        )

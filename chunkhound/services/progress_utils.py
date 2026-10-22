"""Shared Rich progress-bar helpers used by both IndexingCoordinator's Python
path and chunkhound.services.rust_pipeline_runner's Rust path — kept in one
place so the two don't carry independently-drifting copies.
"""

from rich.progress import Progress, TaskID


def _update_speed_field(progress: Progress, task_id: TaskID, unit: str) -> None:
    """Compute and set a task's ``speed`` field from its own completed/elapsed.

    Looks the task up by TaskID directly (``progress._tasks``) rather than via
    the public ``progress.tasks`` list — that list is insertion-ordered and
    desyncs from TaskID once any earlier task is removed (as ``store_task`` is
    on the Rust pipeline path), which would silently target the wrong task.
    """
    task_obj = progress._tasks.get(task_id)
    if task_obj is None:
        return
    if task_obj.elapsed and task_obj.elapsed > 0 and task_obj.completed:
        rate = task_obj.completed / task_obj.elapsed * 60
        progress.update(task_id, speed=f"{rate:.1f} {unit}")


def _start_task_once(progress: Progress, task_id: TaskID) -> None:
    """Start a task clock only if it has not started.

    Rich's ``reset()`` always clears ``start_time``, so it must not run on an
    active build. A completion-only row still needs a start time: Rich sets
    ``finished_time`` from ``elapsed``, and ``elapsed`` is None until then, so
    an unstarted completed row keeps rendering a spinner at 100%.
    """
    task = progress._tasks.get(task_id)
    if task is not None and task.start_time is None:
        progress.reset(task_id, start=True)


def format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (e.g. '9.8GB')."""
    # GB is the terminal unit: everything above it keeps rendering as GB.
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"

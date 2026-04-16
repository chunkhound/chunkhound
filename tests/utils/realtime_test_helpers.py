"""Deterministic helpers for realtime-indexing tests."""

from pathlib import Path

from chunkhound.database_factory import DatabaseServices
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService


async def write_and_index_file(
    services: DatabaseServices,
    file_path: Path,
    content: str | bytes,
) -> dict:
    """Write a file and index it directly without filesystem monitoring."""
    if isinstance(content, bytes):
        file_path.write_bytes(content)
    else:
        file_path.write_text(content)
    return await services.indexing_coordinator.process_file(file_path)


async def remove_file_from_index(
    realtime_service: RealtimeIndexingService,
    file_path: Path,
) -> None:
    """Delete a file on disk and remove its indexed content directly."""
    realtime_service.reset_file_tracking(file_path)
    if file_path.exists():
        file_path.unlink()
    await realtime_service.remove_file(file_path)

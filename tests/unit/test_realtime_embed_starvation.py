import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.services.realtime.service import RealtimeIndexingService


@pytest.fixture
async def service():
    svc = RealtimeIndexingService(
        services=MagicMock(),
        config=MagicMock(),
    )
    yield svc
    await svc.stop()


@pytest.mark.asyncio
async def test_overdue_embed_runs_ahead_of_queued_changes(
    service: RealtimeIndexingService,
    tmp_path: Path,
) -> None:
    """A steady stream of changes must not starve embeds forever.

    Embeds rank below changes, so before this fix an embed waited until the change
    queue was empty, which never happens while files keep being edited.
    """
    embedded = tmp_path / "embedded.py"
    changed = tmp_path / "changed.py"
    embedded.write_text("def a():\n    return 1\n", encoding="utf-8")
    changed.write_text("def b():\n    return 2\n", encoding="utf-8")

    calls: list[str] = []
    done = asyncio.Event()

    async def generate_missing_embeddings() -> None:
        calls.append("embed")

    async def process_file(file_path: Path, skip_embeddings: bool = False) -> dict:
        calls.append(f"change:{file_path.name}")
        done.set()
        return {"chunks": 1, "embeddings": 0}

    service.services = SimpleNamespace(
        indexing_coordinator=SimpleNamespace(
            generate_missing_embeddings=generate_missing_embeddings,
            process_file=process_file,
        ),
        provider=SimpleNamespace(flush=AsyncMock()),
    )
    service.add_file = AsyncMock(return_value=True)

    old = (datetime.now(timezone.utc) - timedelta(minutes=5)).replace(microsecond=0)
    await service._enqueue_mutation(
        service._build_mutation("embed", embedded, first_queued_at=old.isoformat())
    )
    await service._enqueue_mutation(service._build_mutation("change", changed))

    process_task = asyncio.create_task(service._process_loop())
    service.process_task = process_task
    try:
        await asyncio.wait_for(done.wait(), timeout=2.0)
        # Let the loop drain the superseded embed entry.
        for _ in range(20):
            if service.file_queue.empty():
                break
            await asyncio.sleep(0.01)
    finally:
        process_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await process_task

    # The overdue embed ran before the change, and its queue entry did not run again.
    assert calls == ["embed", "change:changed.py"]
    assert not any(op == "embed" for op, _ in service._pending_mutations)

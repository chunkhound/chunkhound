"""Tests for run_rust_indexing_phase()'s DB release/reconnect contract.

release_for_rust_pipeline() hands write ownership of the DuckDB file to the
Rust pipeline; the coordinator must always reconnect once the Rust pipeline
finishes -- even if release_for_rust_pipeline() itself raised, or the Rust
pipeline raised -- or a long-lived server process is left permanently
disconnected. See run_rust_pipeline_runner.run_rust_indexing_phase's inline
comments for the invariant this guards.
"""

from pathlib import Path

import pytest

import chunkhound.services.rust_pipeline_runner as runner_module
from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase


class FakeDatabaseProvider:
    """Minimal DatabaseProvider stub tracking release/connect call counts."""

    def __init__(
        self, *, connected: bool = True, release_error: Exception | None = None
    ) -> None:
        self.db_path = "/tmp/fake_chunkhound/db/chunks.db"
        self._connected = connected
        self._release_error = release_error
        self.release_calls = 0
        self.connect_calls = 0
        self.rust_pipeline_active = False
        self.rust_pipeline_active_during_run: bool | None = None
        self.rust_pipeline_active_during_release: bool | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def release_for_rust_pipeline(self) -> None:
        self.release_calls += 1
        self.rust_pipeline_active_during_release = self.rust_pipeline_active
        if self._release_error is not None:
            # Mirrors the real contract: partially closed, still raises.
            raise self._release_error
        self._connected = False

    def set_rust_pipeline_in_progress(self, active: bool) -> None:
        self.rust_pipeline_active = active

    def connect(self) -> None:
        self.connect_calls += 1
        self._connected = True


async def _fake_run_rust_pipeline_success(*args, **kwargs) -> dict:
    return {
        "total_files": 1,
        "total_chunks": 2,
        "embeddings_generated": 0,
        "errors": [],
        "files_skipped_unchanged": 0,
    }


async def _fake_run_rust_pipeline_raises(*args, **kwargs) -> dict:
    raise RuntimeError("rust pipeline boom")


def _phase_kwargs(db: FakeDatabaseProvider) -> dict:
    return dict(
        db=db,
        config=None,
        embedding_provider=None,
        progress=None,
        files_to_process=[],
        directory=Path("/tmp/fake_project"),
        force_reindex=False,
        do_cleanup=True,
        diff_task=None,
        parse_task=None,
    )


@pytest.mark.asyncio
async def test_reconnects_after_successful_run(monkeypatch):
    """Happy path: release then reconnect, exactly once each."""
    monkeypatch.setattr(
        runner_module, "run_rust_pipeline", _fake_run_rust_pipeline_success
    )
    db = FakeDatabaseProvider(connected=True)

    await run_rust_indexing_phase(**_phase_kwargs(db))

    assert db.release_calls == 1
    assert db.connect_calls == 1


@pytest.mark.asyncio
async def test_reconnects_even_when_release_for_rust_pipeline_raises(monkeypatch):
    """release_for_rust_pipeline() raising must not skip the reconnect.

    Regression test for the bug fixed by "always attempt DB reconnect even
    if release-for-rust itself fails": release_for_rust_pipeline() can raise
    after only partially closing its connections, and the Rust pipeline must
    never be handed write ownership on that ambiguous state -- but the
    process must still end up reconnected afterward.
    """
    rust_pipeline_called = False

    async def fake_run_rust_pipeline(*args, **kwargs):
        nonlocal rust_pipeline_called
        rust_pipeline_called = True
        return {}

    monkeypatch.setattr(runner_module, "run_rust_pipeline", fake_run_rust_pipeline)
    release_error = RuntimeError("partial close failure")
    db = FakeDatabaseProvider(connected=True, release_error=release_error)

    with pytest.raises(RuntimeError, match="partial close failure"):
        await run_rust_indexing_phase(**_phase_kwargs(db))

    assert rust_pipeline_called is False, (
        "must not hand write ownership to Rust on an ambiguously-closed connection"
    )
    assert db.release_calls == 1
    assert db.connect_calls == 1


@pytest.mark.asyncio
async def test_reconnects_even_when_rust_pipeline_raises(monkeypatch):
    """run_rust_pipeline() raising must not leave the DB disconnected."""
    monkeypatch.setattr(
        runner_module, "run_rust_pipeline", _fake_run_rust_pipeline_raises
    )
    db = FakeDatabaseProvider(connected=True)

    with pytest.raises(RuntimeError, match="rust pipeline boom"):
        await run_rust_indexing_phase(**_phase_kwargs(db))

    assert db.release_calls == 1
    assert db.connect_calls == 1


@pytest.mark.asyncio
async def test_release_and_reconnect_still_happen_when_db_starts_disconnected(
    monkeypatch,
):
    """Regression test: the release/ownership-flag/reconnect sequence must not
    be skipped just because `db.is_connected` is already False going in.

    `is_connected` only reflects the connection manager's connection, not the
    executor's separate thread-local one -- a prior run that failed mid-release
    can leave the two out of sync. Gating this block on `db.is_connected`
    meant a `db` left disconnected by an earlier failure skipped release and
    the ownership flag entirely while Rust still ran below, producing a
    second native writer on the same file with no guard at all.
    release_for_rust_pipeline() tolerates already-closed connections, so it's
    always safe to call.
    """
    monkeypatch.setattr(
        runner_module, "run_rust_pipeline", _fake_run_rust_pipeline_success
    )
    db = FakeDatabaseProvider(connected=False)

    await run_rust_indexing_phase(**_phase_kwargs(db))

    assert db.release_calls == 1
    assert db.connect_calls == 1
    assert db.rust_pipeline_active_during_release is True


@pytest.mark.asyncio
async def test_rust_pipeline_flag_set_during_run_and_cleared_after(monkeypatch):
    """The Rust-pipeline-in-progress flag is True for the duration of the
    run_rust_pipeline() call, and False again once run_rust_indexing_phase
    returns -- other callers (e.g. MCP search) must see it cleared by then.
    """
    db = FakeDatabaseProvider(connected=True)

    async def fake_run_rust_pipeline(*args, **kwargs):
        db.rust_pipeline_active_during_run = db.rust_pipeline_active
        return {
            "total_files": 1,
            "total_chunks": 2,
            "embeddings_generated": 0,
            "errors": [],
            "files_skipped_unchanged": 0,
        }

    monkeypatch.setattr(runner_module, "run_rust_pipeline", fake_run_rust_pipeline)

    await run_rust_indexing_phase(**_phase_kwargs(db))

    assert db.rust_pipeline_active_during_run is True
    assert db.rust_pipeline_active is False


@pytest.mark.asyncio
async def test_rust_pipeline_flag_set_before_release_starts(monkeypatch):
    """The flag must be published *before* release_for_rust_pipeline() runs,
    not after it returns -- release runs a CHECKPOINT that can take seconds,
    and a caller submitting work during that window must see the flag
    already set so it fast-fails instead of queueing behind the disconnect
    and reopening a connection just as Rust starts.
    """
    monkeypatch.setattr(
        runner_module, "run_rust_pipeline", _fake_run_rust_pipeline_success
    )
    db = FakeDatabaseProvider(connected=True)

    await run_rust_indexing_phase(**_phase_kwargs(db))

    assert db.rust_pipeline_active_during_release is True


@pytest.mark.asyncio
async def test_rust_pipeline_flag_never_left_set_when_release_raises(monkeypatch):
    """A failed release_for_rust_pipeline() must not leave "Rust owns the
    file" published -- Rust never actually starts in that case, even though
    the flag is briefly set before the release attempt.
    """
    rust_pipeline_called = False

    async def fake_run_rust_pipeline(*args, **kwargs):
        nonlocal rust_pipeline_called
        rust_pipeline_called = True
        return {}

    monkeypatch.setattr(runner_module, "run_rust_pipeline", fake_run_rust_pipeline)
    release_error = RuntimeError("partial close failure")
    db = FakeDatabaseProvider(connected=True, release_error=release_error)

    with pytest.raises(RuntimeError, match="partial close failure"):
        await run_rust_indexing_phase(**_phase_kwargs(db))

    assert rust_pipeline_called is False
    assert db.rust_pipeline_active is False


@pytest.mark.asyncio
async def test_rust_pipeline_flag_cleared_even_when_rust_pipeline_raises(monkeypatch):
    """run_rust_pipeline() raising must still clear the flag on the way out."""
    monkeypatch.setattr(
        runner_module, "run_rust_pipeline", _fake_run_rust_pipeline_raises
    )
    db = FakeDatabaseProvider(connected=True)

    with pytest.raises(RuntimeError, match="rust pipeline boom"):
        await run_rust_indexing_phase(**_phase_kwargs(db))

    assert db.rust_pipeline_active is False

"""Runs the Rust indexing pipeline for one process_directory() call and bridges
its progress_callback(phase, current, total, chunks=0) protocol to Rich bars.

Extracted from IndexingCoordinator.process_directory()'s `if _use_rust:`
branch and the module-scope progress-bridge class it built inline — both are
fully self-contained given plain values (Progress/TaskID/Path/bool/dict), so
neither needs the coordinator instance itself. Kept in chunkhound/services/
(not chunkhound/) because run_rust_indexing_phase() produces
process_directory()-shaped output, matching chunkhound/services/
batch_processor.py's precedent for a DB-free, plain function/class service
module.

Used by:
- IndexingCoordinator.process_directory() — calls run_rust_indexing_phase()
  when the Rust pipeline decision (_use_rust) is True.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import Progress, TaskID

from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import APIEmbeddingProvider
from chunkhound.pipeline_bridge import (
    resolve_duckdb_hnsw_enabled,
    run_rust_pipeline,
)
from chunkhound.services.progress_utils import (
    _start_task_once,
    _update_speed_field,
    format_bytes,
)


class RustProgressBridge:
    """Bridges the Rust pipeline's `progress_callback(phase, current, total,
    chunks=0)` calls to Rich progress bars, and exposes the handful of
    values `run_rust_indexing_phase` needs to read back after the run
    completes.

    Constructed once per `run_rust_pipeline()` call, only when both
    `_use_rust` and `progress` are truthy -- callers pass `None`
    instead of an instance otherwise.
    """

    def __init__(
        self,
        progress: Progress,
        parse_task: TaskID,
        data_task: TaskID,
        index_task: TaskID,
        compact_task: TaskID,
        diff_task: TaskID | None,
        embed_task: TaskID | None,
        compact_db_file: str,
        hnsw_enabled: bool,
    ) -> None:
        self._pr = progress
        self._pt = parse_task
        self._data_task = data_task
        self._index_task = index_task
        self._compact_task = compact_task
        self._diff_task = diff_task
        self._embed_task = embed_task
        self._compact_db_file = compact_db_file

        # Closure-internal bookkeeping (previously `nonlocal` locals).
        self._embed_start = 0.0
        self._embed_reset_done = False
        self._data_task_total = 1
        self._data_start = 0.0
        self._data_reset_done = False
        self._diff_start = 0.0
        self._diff_reset_done = False
        self._parse_reset_done = False
        self._compact_info: str | None = None
        self._hnsw_enabled = hnsw_enabled
        # `write-done` finalizes the index row only while a real build runs;
        # every other outcome is already complete.
        self._index_building = False

        # Read by run_rust_indexing_phase after run_rust_pipeline() returns.
        self.diff_elapsed = 0.0
        self.compact_ran = False
        self.compact_size_before: int | None = None
        self.compact_size_after: int | None = None
        self.compact_reduction_pct: float | None = None

    def _mark_data_done(self) -> None:
        self._pr.update(self._data_task, completed=self._data_task_total, info="done")

    def _mark_index(
        self, info: str, *, running: bool = False, visible: bool = True
    ) -> None:
        # Hide the row once the phase proves no index work can happen.
        # Availability is known only after Rust opens the DB, so a VSS-less
        # run shows the row until its write phase.
        self._pr.update(self._index_task, visible=visible)
        if running:
            self._pr.reset(self._index_task, start=True)
        else:
            _start_task_once(self._pr, self._index_task)
            self._pr.update(self._index_task, completed=1)
        self._pr.update(self._index_task, info=info)
        self._index_building = running

    def _mark_compact(self, info: str, *, running: bool = False) -> None:
        if running:
            self._pr.reset(self._compact_task, start=True)
        else:
            _start_task_once(self._pr, self._compact_task)
            self._pr.update(self._compact_task, completed=1)
        self._pr.update(self._compact_task, info=info)
        self._compact_info = info

    def _snapshot_compact_size(self) -> None:
        try:
            self.compact_size_before = os.path.getsize(self._compact_db_file)
        except OSError:
            self.compact_size_before = None

    def __call__(self, phase: str, current: int, total: int, chunks: int = 0) -> None:
        # `chunks` is the cumulative chunk count, sent only by the
        # write-data phase (4-arg call); other phases use 3 args and
        # leave it at 0.
        _pr = self._pr
        _pt = self._pt
        if phase == "diff":
            if self._diff_task is not None:
                # First-call-based reset, mirroring the embed phase below:
                # current may already be > 0 on the first callback.
                if not self._diff_reset_done:
                    _pr.reset(self._diff_task, total=max(total, 1), start=True)
                    self._diff_start = time.time()
                    self._diff_reset_done = True
                _pr.update(
                    self._diff_task,
                    completed=current,
                    total=max(total, 1),
                    info=f"{current}/{total} checked",
                )
                if current >= total:
                    self.diff_elapsed = time.time() - self._diff_start
                    _pr.update(self._diff_task, info="done")
        elif phase == "parse":
            # First-call-based reset: _pt's clock started ticking at
            # add_task() time, before the diff phase even ran, so its raw
            # elapsed would understate the parse-phase rate.
            if not self._parse_reset_done:
                _pr.reset(_pt, total=max(total, 1), start=True)
                self._parse_reset_done = True
            _pr.update(_pt, completed=current, info=f"{current}/{total} parsed")
            _update_speed_field(_pr, _pt, "files/min")
        elif phase == "embed":
            if self._embed_task is not None:
                # First-call-based reset (not current==0-based): the
                # streaming pipeline reports a live, per-batch-refined
                # total, so its first call may already have current > 0.
                if not self._embed_reset_done:
                    _pr.reset(self._embed_task, total=max(total, 1), start=True)
                    self._embed_start = time.time()
                    self._embed_reset_done = True
                elapsed = time.time() - self._embed_start
                # Require a minimum sample window before trusting the
                # division — a near-zero elapsed (e.g. right at the reset
                # above) against an already-nonzero current would
                # otherwise produce a nonsensical speed.
                speed = current / elapsed if elapsed > 0.05 else 0
                _pr.update(
                    self._embed_task,
                    completed=current,
                    total=max(total, 1),
                    speed=f"{speed:.1f} chunks/s",
                    info=f"{current}/{total} embedded",
                )
        elif phase == "write":
            # Backward compat: old Rust extensions emit one "write"
            # callback with no sub-phase breakdown.
            self._mark_data_done()
            # Old extensions do not emit the disabled/unavailable phases, so
            # the config flag is the only signal available here.
            self._mark_index("done", visible=self._hnsw_enabled)
            self._mark_compact("done")
        elif phase == "write-prepare":
            # Python's DuckDB connection is already closed by now
            # (disconnected before run_rust_pipeline() was even called) —
            # this is purely a progress bar update. Prepare is sub-second
            # (create tables); roll it into the write-data bar as an
            # opening tick.
            self._data_task_total = max(total, 1)
            _pr.reset(self._data_task, total=self._data_task_total, start=True)
            _pr.update(self._data_task, info="preparing...")
        elif phase == "write-data":
            # Batch-keyed in the streaming pipeline (total > 1 — the
            # exact, upfront-known batch count); single-shot in the
            # sequential path (total == 1).
            if total > 1:
                # Speed = true throughput in chunks/s (not cumulative
                # batches/min, which decays from the hot start and sags as
                # files get heavier). Mirrors the embed bar. `chunks` is
                # cumulative; timer starts on the first write-data callback.
                if not self._data_reset_done:
                    self._data_start = time.time()
                    self._data_reset_done = True
                _elapsed = time.time() - self._data_start
                _cps = chunks / _elapsed if _elapsed > 0.05 else 0
                _pr.update(
                    self._data_task,
                    completed=current,
                    speed=f"{_cps:.1f} chunks/s",
                    info=f"{current}/{total} batches written",
                )
            else:
                _pr.update(self._data_task, info="writing...")
        elif phase == "write-index":
            # Data write done; compaction wasn't needed. (0, 1) means a real
            # build runs; (1, 1) means the run changed nothing and the
            # indexes were left in place. Only the build path starts the
            # index clock.
            self._mark_data_done()
            if current >= total:
                self._mark_index("up to date (no changes)")
            else:
                self._mark_index("building...", running=True)
            self._mark_compact("not needed")
        elif phase == "write-index-disabled":
            self._mark_data_done()
            self._mark_index("disabled (not built)", visible=False)
            self._mark_compact("not needed")
        elif phase == "write-index-unavailable":
            self._mark_data_done()
            self._mark_index("skipped (VSS unavailable)", visible=False)
            self._mark_compact("not needed")
        elif phase == "write-compact":
            # Compaction rebuilds the HNSW index through reopen(), so the
            # index bar must not run on its own here.
            self._mark_data_done()
            self._mark_index("maintained during compaction")
            self._mark_compact("compacting (with index)...", running=True)
            self.compact_ran = True
            self._snapshot_compact_size()
        elif phase == "write-compact-disabled":
            self._mark_data_done()
            self._mark_index("not built (HNSW disabled)", visible=False)
            self._mark_compact("compacting (no index)...", running=True)
            self.compact_ran = True
            self._snapshot_compact_size()
        elif phase == "write-compact-unavailable":
            self._mark_data_done()
            self._mark_index("not built (VSS unavailable)", visible=False)
            self._mark_compact("compacting (no index)...", running=True)
            self.compact_ran = True
            self._snapshot_compact_size()
        elif phase == "write-done":
            # Final wrap-up. The index phase already resolved the index bar;
            # only a real build needs its final "done" tick. The compact
            # bar's final size/ratio text is filled in here after compaction
            # actually completes.
            self._mark_data_done()
            if self.compact_ran:
                if self.compact_size_before is not None:
                    try:
                        self.compact_size_after = os.path.getsize(self._compact_db_file)
                        pct = (
                            (self.compact_size_before - self.compact_size_after)
                            / self.compact_size_before
                            * 100
                            if self.compact_size_before
                            else 0.0
                        )
                        direction = "smaller" if pct >= 0 else "larger"
                        self._compact_info = (
                            f"{format_bytes(self.compact_size_before)} → "
                            f"{format_bytes(self.compact_size_after)} "
                            f"({abs(pct):.0f}% {direction})"
                        )
                        self.compact_reduction_pct = pct
                    except OSError:
                        self._compact_info = "done"
                else:
                    self._compact_info = "done"
                _pr.update(self._compact_task, completed=1, info=self._compact_info)
            elif self._index_building:
                self._mark_index("done")
        elif phase == "done":
            # Ensure the write bars are at 100%. The parse and embed bars
            # don't need re-finalizing here — both phases always receive
            # an exact final (total, total) call of their own before
            # "done" fires, in both the sequential and streaming pipeline
            # paths. _index_task/_compact_task are already resolved by
            # write-index*/write-compact*/write-done above — only re-sync
            # _data_task here.
            self._mark_data_done()


@dataclass
class RustPhaseResult:
    """Coordinator-facing outcome of one Rust-pipeline run, merged into
    process_directory()'s stats dict by the caller."""

    total_files: int
    total_chunks: int
    embeddings_generated: int
    errors: list[dict[str, Any]]
    files_skipped_unchanged: int
    skipped_paths: list[tuple[str, str]]
    diff_elapsed: float
    compact_ran: bool
    compact_size_before: int | None
    compact_size_after: int | None
    compact_reduction_pct: float | None


async def run_rust_indexing_phase(
    *,
    db: DatabaseProvider,
    config: Any | None,
    embedding_provider: APIEmbeddingProvider | None,
    progress: Progress | None,
    files_to_process: list[tuple[Path, str | None]],
    directory: Path,
    force_reindex: bool,
    do_cleanup: bool,
    diff_task: TaskID | None,
    parse_task: TaskID | None,
) -> RustPhaseResult:
    """Runs the Rust pipeline for one process_directory() call.

    Builds the Rich progress bars for Rust's write sub-phases, hands the DB
    connection to Rust for the duration of the run, calls
    chunkhound.pipeline_bridge.run_rust_pipeline(), and returns the
    aggregated stats plus the progress-bridge readback fields.

    Exceptions from run_rust_pipeline() (RustPipelineError,
    DiskUsageLimitExceededError, etc.) propagate uncaught — the caller's own
    exception handling is unchanged by this extraction.
    """
    db_path = Path(str(db.db_path)).parent
    embeddings_disabled_by_config = (
        config.embeddings_disabled
        if config and hasattr(config, "embeddings_disabled")
        else False
    )
    skip_embeddings = embeddings_disabled_by_config or embedding_provider is None
    # Same resolver the config dict uses, so the bar cannot disagree with
    # the Rust writer about whether HNSW is enabled.
    hnsw_enabled = resolve_duckdb_hnsw_enabled(config)

    # ── Progress callback for Rust pipeline ─────────────────
    # Maps Rust phases → Rich progress bars.
    # All bars are pre-created so that Rich's Live display picks
    # them up immediately — dynamically added tasks via
    # add_task() inside a PyO3 callback are not reliably
    # rendered by Live until a major refresh.
    progress_bridge: RustProgressBridge | None = None
    if progress:
        # parse_task is guaranteed non-None here (progress is truthy in
        # this branch) — mypy can't see that guarantee across the
        # caller/callee boundary.
        pt: TaskID = parse_task  # type: ignore[assignment]

        # Pre-create embed task (total set to 1 placeholder; the
        # first embed callback will reset it to the real count).
        # start=False so the clock doesn't tick until the phase begins.
        embed_task: TaskID | None = None
        if not skip_embeddings:
            embed_task = progress.add_task(
                "  └─ Embedding", total=1, speed="", info="", start=False
            )

        # Pre-create all write sub-phase bars (no need for a
        # separate "prepare" bar — prepare is sub-second).
        # start=False — clocks are restarted via reset(start=True)
        # when each phase actually begins.
        data_task: TaskID = progress.add_task(
            "  └─ Writing data", total=1, speed="", info="", start=False
        )
        index_task: TaskID = progress.add_task(
            "  └─ Vector index (HNSW)",
            total=1,
            speed="",
            info="",
            start=False,
            visible=hnsw_enabled,
        )
        compact_task: TaskID = progress.add_task(
            "  └─ Compacting", total=1, speed="", info="", start=False
        )
        # Captured now (before db.release_for_rust_pipeline() below) so
        # the compaction phase handlers can stat the file for a
        # before/after size comparison without touching db.
        compact_db_file = str(db.db_path)

        progress_bridge = RustProgressBridge(
            progress=progress,
            parse_task=pt,
            data_task=data_task,
            index_task=index_task,
            compact_task=compact_task,
            diff_task=diff_task,
            embed_task=embed_task,
            compact_db_file=compact_db_file,
            hnsw_enabled=hnsw_enabled,
        )
    progress_cb = progress_bridge

    # Close Python-side DuckDB BEFORE the Rust pipeline starts.
    #
    # pipeline.run() opens its own independent DuckDB connection
    # as its very first step — compute_diff_blocking() (the
    # incremental-diff query) runs before parsing even begins.
    # A live Python connection at that point conflicts with
    # Rust's connection (DuckDB enforces a single writer),
    # producing lock errors or, worse, silent on-disk corruption
    # that only surfaces later as a deserialization error when
    # some other process reopens the DB.
    #
    # Safe to close here: the parse_batch_callback's
    # ProcessPoolExecutor (whose fork'd children inherit
    # whatever DuckDB state is live at fork time) is a lazy,
    # process-wide singleton (see pipeline_bridge._get_parse_pool)
    # — it's created only once, on the first call from Rust's
    # parse phase, which starts only after this call returns on
    # this very first run. Every later run reuses the
    # already-created workers, so no new fork ever happens after
    # this point — there's nothing later to inherit a live
    # connection. Must use the provider's full disconnect(),
    # not just closing _connection_manager.connection: the
    # SerialExecutor holds its own separate thread-local DuckDB
    # connection (created lazily on its worker thread), which a
    # bare connection_manager close would leave dangling.
    # `released_for_rust` tracks whether we *attempted* release, not
    # whether it succeeded — release_for_rust_pipeline() can raise after
    # only partially closing its two connections (see its docstring), and
    # the reconnect in `finally` below must still run in that case, or a
    # long-lived server process is left permanently disconnected.
    #
    # Deliberately NOT gated on `db.is_connected`: that property only
    # reflects the connection manager's connection, not the executor's
    # separate thread-local one, and a prior run that failed mid-release
    # can leave the two out of sync. Gating on it meant a `db` left
    # disconnected by an earlier failure skipped this whole block —
    # publishing no ownership flag and never calling release — while Rust
    # still ran below, producing a second native writer on the same file
    # with no guard at all. release_for_rust_pipeline() is idempotent (it
    # tolerates already-closed connections), so it's always safe to call.
    released_for_rust = False
    release_error: Exception | None = None
    if db is not None:
        released_for_rust = True
        # Publish "Rust owns the file" *before* release starts, not after it
        # returns: release_for_rust_pipeline() runs a CHECKPOINT that can take
        # seconds, and the fast-fail guard on new submissions only helps if
        # it's live for that whole window. Setting the flag only after release
        # completes left a gap where a concurrent op could pass the stale
        # guard, queue behind the disconnect, and open a fresh connection just
        # as Rust starts — a single-writer lock error.
        db.set_rust_pipeline_in_progress(True)
        try:
            db.release_for_rust_pipeline()
        except Exception as e:
            release_error = e
            # Release failed, so Rust never takes ownership (see the `raise
            # release_error` below) — don't leave other callers fast-failing
            # against a database Python still owns.
            db.set_rust_pipeline_in_progress(False)

    try:
        if release_error is not None:
            # Don't hand write ownership to Rust on an ambiguously-closed
            # connection — release_for_rust_pipeline()'s docstring requires
            # this. Still falls through to the reconnect attempt below.
            raise release_error
        rust_stats = await run_rust_pipeline(
            files_to_process,
            db_path=db_path,
            project_root=directory,
            force_reindex=force_reindex,
            skip_embeddings=skip_embeddings,
            do_cleanup=do_cleanup,
            config=config,
            progress_callback=progress_cb,
        )
    finally:
        # Reopen the Python-side DuckDB connection whether or not
        # the Rust pipeline succeeded — release_for_rust_pipeline()
        # kept the executor alive; connect() creates a fresh
        # thread-local connection inside that executor. Without
        # this in a finally, a raised exception (embed/DB-write
        # failure, Rust panic surfaced as PyErr, release_for_rust_pipeline()
        # itself failing, etc.) leaves db permanently disconnected for the
        # rest of the process's life.
        if released_for_rust and db is not None:
            # Clear before connect(): connect()'s own defense-in-depth
            # check would otherwise reject this trusted reconnect.
            db.set_rust_pipeline_in_progress(False)
            db.connect()

    diff_elapsed = 0.0
    compact_ran = False
    compact_size_before: int | None = None
    compact_size_after: int | None = None
    compact_reduction_pct: float | None = None
    if progress_bridge is not None:
        diff_elapsed = progress_bridge.diff_elapsed
        compact_ran = progress_bridge.compact_ran
        compact_size_before = progress_bridge.compact_size_before
        compact_size_after = progress_bridge.compact_size_after
        compact_reduction_pct = progress_bridge.compact_reduction_pct

    return RustPhaseResult(
        total_files=int(rust_stats.get("total_files", 0)),
        total_chunks=int(rust_stats.get("total_chunks", 0)),
        embeddings_generated=int(rust_stats.get("embeddings_generated", 0)),
        errors=list(rust_stats.get("errors", [])),
        files_skipped_unchanged=int(rust_stats.get("files_skipped_unchanged", 0)),
        skipped_paths=[
            (str(p), str(r)) for p, r in rust_stats.get("skipped_paths", [])
        ],
        diff_elapsed=diff_elapsed,
        compact_ran=compact_ran,
        compact_size_before=compact_size_before,
        compact_size_after=compact_size_after,
        compact_reduction_pct=compact_reduction_pct,
    )

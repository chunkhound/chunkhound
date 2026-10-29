"""Worker drainage keeps database ownership until the actual work finishes."""

import asyncio
import concurrent.futures
import contextvars
import inspect
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

import chunkhound_native as native
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.serial_executor import (
    DatabaseRustPipelineInProgressError,
    SerialDatabaseExecutor,
)
from chunkhound.utils.async_worker import await_worker_completion


def test_admitted_connect_cannot_reopen_python_handles_during_native_handoff(tmp_path):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    admitted, resume = threading.Event(), threading.Event()
    prepare = SerialDatabaseExecutor.prepare_connection
    source, start = inspect.getsourcelines(prepare)
    submit_line = next(
        start + index
        for index, line in enumerate(source)
        if "future = self._submit" in line
    )

    def pause_after_admission(frame, event, arg):
        # Schedule a real public connect at the caller's admission/submission
        # boundary, without replacing any executor or backend operation.
        if (
            event == "line"
            and frame.f_code is prepare.__code__
            and frame.f_lineno == submit_line
        ):
            admitted.set()
            assert resume.wait(20)
        return pause_after_admission

    def connect():
        sys.settrace(pause_after_admission)
        try:
            provider.connect()
        finally:
            sys.settrace(None)

    try:
        with concurrent.futures.ThreadPoolExecutor(1) as callers:
            pending = callers.submit(connect)
            try:
                assert admitted.wait(10)
                provider.set_rust_pipeline_in_progress(True)
                provider.release_for_rust_pipeline()
                assert provider.connection is None
                resume.set()
                with pytest.raises(DatabaseRustPipelineInProgressError):
                    pending.result(timeout=20)
                assert provider.is_rust_pipeline_in_progress()
                assert provider.connection is None, "stale connect reopened manager"
                assert provider._executor.handles_closed
                with pytest.raises(native.DatabaseBusyError):
                    native.acquire_install_lease(str(path), "duckdb")
                native.IndexingPipeline(
                    {"db_path": str(tmp_path), "skip_embeddings": True}
                ).run([], lambda paths, config: [])
                assert provider.connection is None
            finally:
                resume.set()
        provider.set_rust_pipeline_in_progress(False)
        provider.connect()
        assert provider.execute_query("SELECT 42 AS value") == [{"value": 42}]
    finally:
        resume.set()
        provider.set_rust_pipeline_in_progress(False)
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancelled_serial_work_drains_before_close(tmp_path, worker_fails):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work(value: int) -> int:
        started.set()
        try:
            assert release.wait(20)
            if worker_fails:
                raise OSError("injected worker failure")
            return value
        finally:
            finished.set()

    provider.connection.create_function("lifetime_blocked", work, side_effects=True)
    task = asyncio.create_task(
        provider._execute_in_db_thread(
            "execute_query", "SELECT lifetime_blocked(42)", None
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 10)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


def test_timeout_does_not_release_running_work(tmp_path):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider._executor._execute_timeout_seconds = 0.05
    release = threading.Event()

    def blocked(value: int) -> int:
        assert release.wait(20)
        return value

    provider.connection.create_function("blocked", blocked, side_effects=True)
    with concurrent.futures.ThreadPoolExecutor(1) as caller:
        try:
            with pytest.raises(TimeoutError):
                provider.execute_query("SELECT blocked(42)")
            closing = caller.submit(provider.close)
            time.sleep(0.1)
            assert not closing.done()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        finally:
            release.set()
    provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
@pytest.mark.parametrize("paused_stage", ["parse", "cleanup"])
async def test_native_cancellation_drains_before_reconnect(
    tmp_path, monkeypatch, paused_stage
):
    import chunkhound.pipeline_bridge as bridge
    from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase

    path, source = tmp_path / "chunks.db", tmp_path / "example.py"
    source.write_text("def example():\n    return 42\n")
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    started, release = threading.Event(), threading.Event()
    callback_name = (
        "parse_batch_callback"
        if paused_stage == "parse"
        else "_shutdown_embed_thread_resources"
    )
    actual = getattr(bridge, callback_name)

    def paused(*args, **kwargs):
        started.set()
        assert release.wait(30)
        return actual(*args, **kwargs)

    monkeypatch.setattr(bridge, callback_name, paused)
    task = asyncio.create_task(
        run_rust_indexing_phase(
            db=provider,
            config=None,
            embedding_provider=None,
            progress=None,
            files_to_process=[(source, None)],
            directory=tmp_path,
            force_reindex=True,
            do_cleanup=True,
            diff_task=None,
            parse_task=None,
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 20)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and provider.is_rust_pipeline_in_progress()
            assert not provider.is_connected
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.is_connected and not provider.is_rust_pipeline_in_progress()
        assert provider.get_stats()["files"] == (0 if paused_stage == "parse" else 1)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        provider.close()


@pytest.mark.parametrize("paused_stage", ["parse", "cleanup"])
def test_asyncio_shutdown_drains_native_handoff_before_reconnect(
    tmp_path, monkeypatch, paused_stage
):
    """asyncio.run cancels every Task, not just the indexing owner."""
    import chunkhound.pipeline_bridge as bridge
    from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase

    path, source = tmp_path / "chunks.db", tmp_path / "example.py"
    source.write_text("def example():\n    return 42\n")
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    started, release = threading.Event(), threading.Event()
    parsed, cleaned = threading.Event(), threading.Event()
    resource_loop = asyncio.new_event_loop()
    failures, handoff_states, cleanup_states = [], [], []
    actual_parse = bridge.parse_batch_callback
    actual_cleanup = bridge._shutdown_embed_thread_resources

    def parse(*args, **kwargs):
        if paused_stage == "parse":
            started.set()
            assert release.wait(20)
        try:
            return actual_parse(*args, **kwargs)
        finally:
            parsed.set()

    def cleanup(cache):
        cleanup_states.append(parsed.is_set())
        # A real per-run event loop must remain open until cleanup completes.
        cache.loops[threading.get_ident()] = resource_loop
        if paused_stage == "cleanup":
            started.set()
            assert release.wait(20)
        try:
            actual_cleanup(cache)
        finally:
            cleaned.set()

    monkeypatch.setattr(bridge, "parse_batch_callback", parse)
    monkeypatch.setattr(bridge, "_shutdown_embed_thread_resources", cleanup)

    async def owner():
        try:
            await run_rust_indexing_phase(
                db=provider,
                config=None,
                embedding_provider=None,
                progress=None,
                files_to_process=[(source, None)],
                directory=tmp_path,
                force_reindex=True,
                do_cleanup=True,
                diff_task=None,
                parse_task=None,
            )
        finally:
            handoff_states.append(
                (parsed.is_set(), cleaned.is_set(), resource_loop.is_closed())
            )

    def probe(loop, task, remaining):
        try:
            assert not task.done(), "owner returned while its worker was blocked"
            assert provider.is_rust_pipeline_in_progress()
            assert not provider.is_connected
            assert not cleaned.is_set() and not resource_loop.is_closed()
            if paused_stage == "parse":
                assert not cleanup_states, "cleanup overlapped the native run"
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
            if remaining:
                # Repeat shutdown-style cancellation, including any worker Tasks.
                for pending in asyncio.all_tasks(loop):
                    pending.cancel()
                loop.call_later(0.02, probe, loop, task, remaining - 1)
                return
        except BaseException as error:
            failures.append(error)
        release.set()

    async def main():
        task = asyncio.create_task(owner())
        assert await asyncio.to_thread(started.wait, 15)
        loop = asyncio.get_running_loop()
        loop.call_later(0.02, probe, loop, task, 2)
        # Returning triggers real asyncio.run all-Tasks cancellation.
        return task

    try:
        task = asyncio.run(main())
        if failures:
            raise failures[0]
        assert task.cancelled()
        assert cleanup_states == [True]
        assert handoff_states == [(True, True, True)]
        assert provider.is_connected and not provider.is_rust_pipeline_in_progress()
        assert provider.get_stats()["files"] == (0 if paused_stage == "parse" else 1)
    finally:
        release.set()
        resource_loop.close()
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
async def test_native_progress_and_resource_shutdown_keep_caller_context(
    tmp_path, monkeypatch
):
    import chunkhound.pipeline_bridge as bridge

    source = tmp_path / "example.py"
    source.write_text("def example():\n    return 42\n")
    request = contextvars.ContextVar("indexing_request", default=None)
    observed = {}
    actual_cleanup = bridge._shutdown_embed_thread_resources

    class Client:
        async def shutdown(self):
            observed["shutdown"] = request.get()

    def cleanup(cache):
        # Stand in only for the external HTTP client; cleanup and loop are real.
        tid = threading.get_ident()
        cache.providers[tid] = Client()
        cache.loops[tid] = asyncio.new_event_loop()
        actual_cleanup(cache)

    def progress(phase, *_):
        if phase == "done":  # Native calls this on the submitting worker thread.
            observed["progress"] = request.get()

    monkeypatch.setattr(bridge, "_shutdown_embed_thread_resources", cleanup)
    token = request.set("caller-request")
    try:
        result = await bridge.run_rust_pipeline(
            [(source, None)],
            db_path=tmp_path,
            project_root=tmp_path,
            skip_embeddings=True,
            progress_callback=progress,
        )
    finally:
        request.reset(token)
    assert result["total_files"] == 1
    assert observed == {"progress": "caller-request", "shutdown": "caller-request"}


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,fail", [
    ("release", False), ("reconnect", False), ("reconnect", True),
])
async def test_handoff_reconnect_are_responsive_guarded_and_drained(
    tmp_path, monkeypatch, stage, fail
):
    from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase

    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider.ensure_indexed_root_identity(
        requested_root=tmp_path, allow_claim_if_missing=True
    )
    source = tmp_path / "example.py"
    source.write_text("def example():\n    return 42\n")
    manager = provider.connection
    sidecar = Path(str(path) + ".root.json")
    started, release, watchdog_fired = (
        threading.Event(), threading.Event(), threading.Event()
    )
    real_close = duckdb.DuckDBPyConnection.close
    real_read_text = Path.read_text

    def pause():
        started.set()
        assert release.wait(20)

    def paused_close(connection):
        if stage == "release" and connection is manager and not started.is_set():
            pause()
        return real_close(connection)

    def paused_read_text(file, *args, **kwargs):
        if stage == "reconnect" and file == sidecar and not started.is_set():
            pause()
            if fail:
                raise OSError("reconnect filesystem unavailable")
        return real_read_text(file, *args, **kwargs)

    def unblock_baseline():
        watchdog_fired.set()
        release.set()

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", paused_close)
    monkeypatch.setattr(Path, "read_text", paused_read_text)
    # A real thread watchdog unblocks the synchronous baseline even if its
    # event loop cannot service an asyncio timeout. Not a latency SLA.
    watchdog = threading.Timer(15, unblock_baseline)
    watchdog.start()
    task = asyncio.create_task(run_rust_indexing_phase(
        db=provider, config=None, embedding_provider=None, progress=None,
        files_to_process=[(source, None)], directory=tmp_path,
        force_reindex=True, do_cleanup=True, diff_task=None, parse_task=None,
    ))
    heartbeats = 0
    try:
        assert await asyncio.to_thread(started.wait, 20)
        assert not watchdog_fired.is_set(), "blocking lifecycle starved event loop"
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
            heartbeats += 1
            assert not release.is_set() and not task.done()
            assert provider.is_rust_pipeline_in_progress()
            with pytest.raises(DatabaseRustPipelineInProgressError):
                provider.execute_query("SELECT 1")
            with pytest.raises(DatabaseRustPipelineInProgressError):
                provider.connect()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        assert heartbeats == 3 and not watchdog_fired.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.is_rust_pipeline_in_progress() is fail
        if fail:
            assert provider._rust_pipeline_cleanup is not None
            with pytest.raises(DatabaseRustPipelineInProgressError):
                provider.execute_query("SELECT 1")
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        else:
            assert provider.execute_query("SELECT 42 AS value") == [{"value": 42}]
            assert provider.get_stats()["files"] == (0 if stage == "release" else 1)
    finally:
        release.set()
        watchdog.cancel()
        watchdog.join(timeout=5)
        await asyncio.gather(task, return_exceptions=True)
        provider.set_rust_pipeline_in_progress(False)  # explicit teardown after joins
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
async def test_failed_client_cleanup_retains_unavailable_provider_after_cancel(
    tmp_path, monkeypatch
):
    import chunkhound.pipeline_bridge as bridge
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
    from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase

    path = tmp_path / "chunks.db"
    provider = None
    tasks, observed, exception_resources = [], [], []
    entered, release, watchdog_fired = (
        threading.Event(), threading.Event(), threading.Event()
    )
    watchdog = None
    fail = True

    class EmbeddingConfig:
        provider = "contract-callback"
        model = "fixture"
        batch_size = 100
        max_concurrent_batches = 1

        def model_copy(self, *, update):
            return self

    class Client:
        closed = False

        async def embed(self, texts):
            observed.append((self, asyncio.get_running_loop()))
            return [[1.0, 2.0, 3.0] for _ in texts]

        async def shutdown(self):
            assert (self, asyncio.get_running_loop()) in observed
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            if fail:
                raise OSError("external client close failed")
            self.closed = True

    def unblock():
        watchdog_fired.set()
        release.set()

    async def own(**kwargs):
        try:
            return await run_rust_indexing_phase(**kwargs)
        except BaseException as error:
            # Observe Task 2's direct-bridge attachment before Task cancellation
            # wrappers can discard it; never inject a replacement resource owner.
            resources = getattr(error, "resources", None)
            if resources is not None:
                exception_resources.append(resources)
            raise

    try:
        provider = DuckDBProvider(path, base_directory=tmp_path)
        provider.connect()
        source = tmp_path / "example.py"
        source.write_text("def example():\n    return 1\n")
        monkeypatch.setattr(
            EmbeddingProviderFactory, "create_provider",
            staticmethod(lambda config: Client()),
        )
        config = SimpleNamespace(
            embedding=EmbeddingConfig(), embeddings_disabled=False,
            indexing=SimpleNamespace(max_concurrent=1),
            database=SimpleNamespace(fragmentation_threshold_pct=None),
        )
        watchdog = threading.Timer(40, unblock)
        watchdog.start()
        task = asyncio.create_task(own(
            db=provider, config=config, embedding_provider=Client(), progress=None,
            files_to_process=[(source, None)], directory=tmp_path,
            force_reindex=True, do_cleanup=True, diff_task=None, parse_task=None,
        ))
        tasks.append(task)
        assert await asyncio.to_thread(entered.wait, 20)
        assert not watchdog_fired.is_set(), "watchdog escape is not behavioral RED"
        assert len(set(observed)) == 1
        client, owner_loop = observed[0]
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
            assert not release.is_set() and not task.done()
            assert provider.is_rust_pipeline_in_progress()
            assert not client.closed and not owner_loop.is_closed()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not watchdog_fired.is_set()
        # Public behavior MUST precede optional Task 3 resource-slot assertions:
        # the retrospective Task 2 baseline falsely reconnects here.
        assert provider.is_rust_pipeline_in_progress() and not provider.is_connected
        with pytest.raises(DatabaseRustPipelineInProgressError):
            provider.execute_query("SELECT 1")
        with pytest.raises(DatabaseRustPipelineInProgressError):
            provider.connect()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        assert not client.closed and not owner_loop.is_closed()
        resources = getattr(provider, "_rust_pipeline_cleanup", None)
        assert resources is not None
        assert client in resources.cache.providers.values()
        assert owner_loop in resources.cache.loops.values()
        assert not resources.cache_closed
        assert resources.native_future.done() and resources.token.cleanup_complete
        assert resources.cleanup_future.done()
    finally:
        release.set()
        if watchdog is not None:
            watchdog.cancel()
            if watchdog.ident is not None:
                watchdog.join(timeout=5)
        await asyncio.gather(*tasks, return_exceptions=True)
        fail = False  # Explicit recovery only after every owner task has drained.
        resources = getattr(provider, "_rust_pipeline_cleanup", None)
        if resources is None and exception_resources:
            resources = exception_resources[0]
        recovered = False
        try:
            if resources is not None:
                for future in (resources.native_future, resources.cleanup_future):
                    assert future is None or future.done()
                assert not resources.native_started or resources.token.cleanup_complete
                await asyncio.to_thread(
                    bridge._shutdown_embed_thread_resources, resources.cache
                )
                assert not resources.cache.providers and not resources.cache.loops
            else:
                # Setup-only fallback: never close a loop with active work.
                shutdown_entered = entered.is_set()

                def recover_observed():
                    for client, loop in set(observed):
                        if not loop.is_closed():
                            assert not loop.is_running()
                            loop.run_until_complete(client.shutdown())
                            loop.close()

                await asyncio.to_thread(recover_observed)
                assert not shutdown_entered, "failed shutdown lost its resource owner"
            assert all(c.closed and loop.is_closed() for c, loop in observed)
            recovered = True
        finally:
            if recovered and provider is not None:
                if hasattr(provider, "_rust_pipeline_cleanup"):
                    provider._rust_pipeline_cleanup = None
                provider.set_rust_pipeline_in_progress(False)
                provider.close()
                native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
async def test_duplicate_indexing_scope_cannot_take_active_owner(tmp_path, monkeypatch):
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
    from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase

    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    closing, resume_close, closed = (
        threading.Event(), threading.Event(), threading.Event()
    )
    watchdog_fired = threading.Event()
    owner_loops = []
    tasks = []

    class EmbeddingConfig:
        provider = "contract-callback"
        model = "fixture"
        batch_size = 100
        max_concurrent_batches = 1

        def model_copy(self, *, update):
            return self

    class Client:
        async def embed(self, texts):
            owner_loops.append(asyncio.get_running_loop())
            return [[1.0, 2.0, 3.0] for _ in texts]

        async def shutdown(self):
            assert asyncio.get_running_loop() is owner_loops[0]
            closing.set()
            while not resume_close.is_set():
                await asyncio.sleep(0.01)
            closed.set()

    def unblock_baseline():
        watchdog_fired.set()
        resume_close.set()

    watchdog = threading.Timer(40, unblock_baseline)
    try:
        provider.connect()
        source = tmp_path / "example.py"
        source.write_text("def example():\n    return 42\n")
        client = Client()
        monkeypatch.setattr(
            EmbeddingProviderFactory,
            "create_provider",
            staticmethod(lambda config: client),
        )
        config = SimpleNamespace(
            embedding=EmbeddingConfig(), embeddings_disabled=False,
            indexing=SimpleNamespace(max_concurrent=1),
            database=SimpleNamespace(fragmentation_threshold_pct=None),
        )
        watchdog.start()
        first = asyncio.create_task(run_rust_indexing_phase(
            db=provider, config=config, embedding_provider=client, progress=None,
            files_to_process=[(source, None)], directory=tmp_path,
            force_reindex=True, do_cleanup=True, diff_task=None, parse_task=None,
        ))
        tasks.append(first)
        assert await asyncio.to_thread(closing.wait, 20)
        assert not watchdog_fired.is_set()
        assert not first.done() and provider.is_rust_pipeline_in_progress()
        assert not closed.is_set() and not owner_loops[0].is_closed()
        # Shutdown begins only after the actual native worker has joined.
        # The baseline has no provider resource slot; admission must fail first.
        resources = getattr(provider, "_rust_pipeline_cleanup", None)
        second = asyncio.create_task(run_rust_indexing_phase(
            db=provider, config=config, embedding_provider=None, progress=None,
            files_to_process=[], directory=tmp_path,
            force_reindex=True, do_cleanup=False, diff_task=None, parse_task=None,
        ))
        tasks.append(second)
        done, _ = await asyncio.wait({second}, timeout=10)
        assert not watchdog_fired.is_set(), "watchdog release is not admission RED"
        assert second in done, "duplicate scope did not finish bounded observation"
        with pytest.raises(DatabaseRustPipelineInProgressError):
            second.result()
        assert not first.done() and provider.is_rust_pipeline_in_progress()
        assert not closed.is_set() and not owner_loops[0].is_closed()
        if hasattr(provider, "_rust_pipeline_cleanup"):
            assert resources is not None
            assert provider._rust_pipeline_cleanup is resources
            assert resources.native_future.done() and resources.token.cleanup_complete
            assert resources.cache.providers and not resources.cache_closed
        with pytest.raises(DatabaseRustPipelineInProgressError):
            provider.execute_query("SELECT 1")
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        assert not watchdog_fired.is_set()
    finally:
        resume_close.set()
        watchdog.cancel()
        if watchdog.ident is not None:
            watchdog.join(timeout=5)
        await asyncio.gather(*tasks, return_exceptions=True)
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
async def test_duplicate_indexing_refuses_while_reconnect_reads_sidecar(
    tmp_path, monkeypatch
):
    from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase

    path = tmp_path / "chunks.db"
    sidecar = Path(str(path) + ".root.json")
    provider = DuckDBProvider(path, base_directory=tmp_path)
    entered, release, watchdog_fired = (
        threading.Event(), threading.Event(), threading.Event()
    )
    original_read_text = Path.read_text
    tasks = []
    config = SimpleNamespace(
        embedding=None, embeddings_disabled=True,
        indexing=SimpleNamespace(max_concurrent=1),
        database=SimpleNamespace(fragmentation_threshold_pct=None),
    )

    def paused_read_text(self, *args, **kwargs):
        if (
            self == sidecar
            and provider.is_rust_pipeline_in_progress()
            and not provider.is_connected
        ):
            # This is trusted reconnect, not initial connection/preparation:
            # native execution and client cleanup have already completed.
            resources = provider._rust_pipeline_cleanup
            assert resources.native_started and resources.native_future.done()
            assert resources.token.cleanup_complete and resources.safe_to_reconnect
            entered.set()
            assert release.wait(20), "sidecar read was not released"
        return original_read_text(self, *args, **kwargs)

    def unblock_baseline():
        watchdog_fired.set()
        release.set()

    watchdog = threading.Timer(10, unblock_baseline)

    def start_run():
        task = asyncio.create_task(run_rust_indexing_phase(
            db=provider, config=config, embedding_provider=None, progress=None,
            files_to_process=[], directory=tmp_path, force_reindex=True,
            do_cleanup=False, diff_task=None, parse_task=None,
        ))
        tasks.append(task)
        return task

    try:
        provider.connect()
        # Initial connection defers legacy migration; claim through the real
        # public filesystem contract before installing the read boundary.
        provider.ensure_indexed_root_identity(
            requested_root=tmp_path, allow_claim_if_missing=True
        )
        assert sidecar.is_file()
        monkeypatch.setattr(Path, "read_text", paused_read_text)
        first = start_run()
        assert await asyncio.to_thread(entered.wait, 10)
        assert not first.done() and provider.is_rust_pipeline_in_progress()
        resources = provider._rust_pipeline_cleanup
        acquired = provider._ownership_lock.acquire(blocking=False)
        if acquired:
            provider._ownership_lock.release()
        assert not acquired, "reconnect sidecar read must hold the ownership mutex"
        watchdog.start()
        second = start_run()
        done, _ = await asyncio.wait({second}, timeout=5)
        assert not watchdog_fired.is_set(), (
            "watchdog release is not loop responsiveness"
        )
        assert second in done
        with pytest.raises(DatabaseRustPipelineInProgressError):
            second.result()
        await asyncio.sleep(0)
        assert not first.done() and provider.is_rust_pipeline_in_progress()
        assert provider._rust_pipeline_cleanup is resources
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        release.set()
        await first
        assert provider.is_connected and not provider.is_rust_pipeline_in_progress()
    finally:
        release.set()
        watchdog.cancel()
        if watchdog.ident is not None:
            watchdog.join(timeout=5)
        await asyncio.gather(*tasks, return_exceptions=True)
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
async def test_direct_cleanup_failure_preserves_cancellation_and_resources(tmp_path, monkeypatch):
    import chunkhound.pipeline_bridge as bridge
    entered, release = threading.Event(), threading.Event()
    resources = bridge.RustRunResources()
    loop = asyncio.new_event_loop()
    class Client:
        async def shutdown(self):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            raise OSError("external close failure")
    resources.cache.providers[1] = Client()
    resources.cache.loops[1] = loop
    # Supply the pre-seeded per-run resource only at construction; exercise
    # the public direct caller path (owned=True), including error attachment.
    monkeypatch.setattr(bridge, "RustRunResources", lambda: resources)
    task = asyncio.create_task(bridge.run_rust_pipeline(
        [], db_path=tmp_path, project_root=tmp_path, skip_embeddings=True,
    ))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.resources is resources
        assert not loop.is_closed() and resources.cache.providers
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        loop.close()


@pytest.mark.asyncio
async def test_stop_callback_failure_cannot_abandon_running_worker(tmp_path):
    from chunkhound.utils.async_worker import WorkerDrain

    path = tmp_path / "chunks.db"
    lease = native.acquire_database_lease(str(path))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        entered.set()
        try:
            assert release.wait(20)
            return 42
        finally:
            finished.set()

    def broken_stop():
        raise OSError("stop notification failed")

    async def own():
        drain = WorkerDrain(broken_stop)
        future = asyncio.get_running_loop().run_in_executor(None, work)
        try:
            result = await drain.wait(future)
        except BaseException as error:
            drain.finish(error)
            raise
        drain.finish()
        return result

    task = asyncio.create_task(own())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done() and not finished.is_set()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        lease.close()


@pytest.mark.asyncio
async def test_actual_native_callback_cancel_keeps_owner_through_client_close(
    tmp_path, monkeypatch
):
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
    from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase

    path = tmp_path / "chunks.db"
    source = tmp_path / "example.py"
    source.write_text("def example():\n    return 42\n")
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    entered, resume_embed = threading.Event(), threading.Event()
    closing, resume_close = threading.Event(), threading.Event()
    closed = threading.Event()
    owner_loops = []

    class EmbeddingConfig:
        provider = "contract-callback"
        model = "fixture"
        batch_size = 100
        max_concurrent_batches = 1

        def model_copy(self, *, update):
            return self

    class Client:
        async def embed(self, texts):
            owner_loops.append(asyncio.get_running_loop())
            entered.set()
            while not resume_embed.is_set():
                await asyncio.sleep(0.01)
            return [[1.0, 2.0, 3.0] for _ in texts]

        async def shutdown(self):
            assert asyncio.get_running_loop() is owner_loops[0]
            closing.set()
            while not resume_close.is_set():
                await asyncio.sleep(0.01)
            closed.set()

    client = Client()
    monkeypatch.setattr(
        EmbeddingProviderFactory, "create_provider", staticmethod(lambda config: client)
    )
    config = SimpleNamespace(
        embedding=EmbeddingConfig(), embeddings_disabled=False,
        indexing=SimpleNamespace(max_concurrent=1),
        database=SimpleNamespace(fragmentation_threshold_pct=None),
    )
    task = asyncio.create_task(run_rust_indexing_phase(
        db=provider, config=config, embedding_provider=client, progress=None,
        files_to_process=[(source, None)], directory=tmp_path,
        force_reindex=True, do_cleanup=True, diff_task=None, parse_task=None,
    ))
    beats = 0
    try:
        assert await asyncio.to_thread(entered.wait, 20)
        assert provider.is_rust_pipeline_in_progress() and not task.done()
        for phase in ("native-callback", "client-close"):
            if phase == "client-close":
                resume_embed.set()
                assert await asyncio.to_thread(closing.wait, 20)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.01)
                beats += 1
                assert not task.done() and not closed.is_set()
                assert provider.is_rust_pipeline_in_progress()
                assert not provider.is_connected
                assert owner_loops and not owner_loops[0].is_closed()
                with pytest.raises(DatabaseRustPipelineInProgressError):
                    provider.execute_query("SELECT 1")
                with pytest.raises(native.DatabaseBusyError):
                    native.acquire_install_lease(str(path), "duckdb")
        assert beats == 6
        resume_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set() and owner_loops[0].is_closed()
        assert provider.is_connected and not provider.is_rust_pipeline_in_progress()
        assert provider.get_stats()["files"] == 0
        assert provider.execute_query("SELECT 42 AS value") == [{"value": 42}]
    finally:
        resume_embed.set()
        resume_close.set()
        await asyncio.gather(task, return_exceptions=True)
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL owner/child contract")
def test_parser_worker_does_not_inherit_owner_lease(tmp_path):
    script = tmp_path / "owner.py"
    script.write_text(
        textwrap.dedent("""
        import os, sys, time
        from pathlib import Path
        import chunkhound_native as native
        from chunkhound.pipeline_bridge import _get_parse_pool

        def child(pid_path, stop_path):
            Path(pid_path).write_text(str(os.getpid()))
            deadline = time.monotonic() + 30
            while not Path(stop_path).exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            os._exit(0)

        if __name__ == '__main__':
            lease = native.acquire_database_lease(sys.argv[1])
            Path(sys.argv[4]).write_text(str(os.getpid()))
            _get_parse_pool(1).submit(child, sys.argv[2], sys.argv[3]).result()
    """)
    )
    target = tmp_path / "chunks.db"
    child_file, owner_file = tmp_path / "child.pid", tmp_path / "owner.pid"
    stop = tmp_path / "stop"
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "python",
            str(script),
            str(target),
            str(child_file),
            str(stop),
            str(owner_file),
        ],
        env={**os.environ, "UV_NO_SYNC": "1"},
    )
    try:
        deadline = time.monotonic() + 20
        while not child_file.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        assert child_file.exists()
        os.kill(int(owner_file.read_text()), signal.SIGKILL)
        process.wait(timeout=10)
        os.kill(int(child_file.read_text()), 0)
        native.acquire_install_lease(str(target), "duckdb").close()
    finally:
        stop.touch()
        if process.poll() is None and owner_file.exists():
            try:
                os.kill(int(owner_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=35)


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_fails", [False, True])
async def test_worker_completion_preserves_result_or_error(tmp_path, worker_fails):
    path = tmp_path / "chunks.db"
    lease = native.acquire_database_lease(str(path))
    failure = OSError("injected worker failure")

    def work():
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        if worker_fails:
            raise failure
        return 42

    try:
        worker = asyncio.get_running_loop().run_in_executor(None, work)
        if worker_fails:
            with pytest.raises(OSError) as caught:
                await await_worker_completion(worker)
            assert caught.value is failure
        else:
            assert await await_worker_completion(worker) == 42
    finally:
        lease.close()
    native.acquire_install_lease(str(path), "duckdb").close()

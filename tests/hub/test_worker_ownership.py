"""Worker drainage keeps database ownership until the actual work finishes."""

import asyncio
import concurrent.futures
import contextvars
import os
import signal
import subprocess
import textwrap
import threading
import time

import pytest

import chunkhound_native as native
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.utils.async_worker import await_worker_completion


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
        assert provider.get_stats()["files"] == 1
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
            assert task.cancelling(), "asyncio.run must have begun shutdown"
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
        assert provider.get_stats()["files"] == 1
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

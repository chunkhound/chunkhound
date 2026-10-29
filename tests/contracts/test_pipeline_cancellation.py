import concurrent.futures
import threading
from pathlib import Path
import pytest
import chunkhound_native as native
from chunkhound.providers.database.duckdb_provider import DuckDBProvider


def parse_fixture(paths, config):
    return [("python", [{"chunk_type": "function", "symbol": Path(p).stem,
        "code": Path(p).read_text(), "start_line": 1, "end_line": 2,
        "language": "python"}], None, None) for p in paths]


def pipeline(root, **overrides):
    return native.IndexingPipeline({"db_path": str(root), "skip_embeddings": True,
        "parse_batch_size": 1, "embed_batch_size": 100,
        "embed_thread_pool_size": 1, "compaction_threshold": None, **overrides})


def test_pre_cancel_is_independent_single_use_and_creates_no_database(tmp_path):
    token = native.PipelineCancellationToken()
    assert not token.is_cancelled() and not token.cleanup_complete
    worker = threading.Thread(target=lambda: (token.cancel(), token.cancel()))
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and token.is_cancelled()
    with pytest.raises(native.PipelineCancelledError):
        pipeline(tmp_path).run([], parse_fixture, cancellation_token=token)
    assert token.cleanup_complete and not (tmp_path / "chunks.db").exists()
    with pytest.raises(ValueError, match="single-use"):
        pipeline(tmp_path).run([], parse_fixture, cancellation_token=token)


def test_cancelled_diff_never_deletes_unvisited_files(tmp_path):
    files = [tmp_path / f"f{i}.py" for i in range(205)]
    for source in files:
        source.write_text("def f():\n    return 1\n")
    entries = [(str(p), p.name) for p in files]
    pipeline(tmp_path, parse_batch_size=200).run(entries, parse_fixture)
    token = native.PipelineCancellationToken()
    def progress(phase, current, total, *args):
        if phase == "diff" and current == 200:
            token.cancel()
    with pytest.raises(native.PipelineCancelledError):
        pipeline(tmp_path).run(entries, parse_fixture, incremental=True,
            progress_callback=progress, cancellation_token=token)
    provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
    try:
        provider.connect()
        assert provider.get_stats()["files"] == len(files)
    finally:
        provider.close()
    assert token.cleanup_complete


@pytest.mark.parametrize("blocked", ["embed", "store"])
def test_cancellation_drains_callback_and_both_backpressure_channels(tmp_path, blocked):
    files = [tmp_path / f"f{i}.py" for i in range(40)]
    for source in files:
        source.write_text("def f():\n    return 1\n")
    entries = [(str(p), p.name) for p in files]
    started, release, filled = threading.Event(), threading.Event(), threading.Event()
    token = native.PipelineCancellationToken()
    parsed = []
    def parse(paths, config):
        parsed.extend(paths)
        return parse_fixture(paths, config)
    def embed(texts):
        started.set()
        assert release.wait(20)
        return [[1.0, 2.0, 3.0] for _ in texts]
    def progress(phase, current, total, *args):
        if phase == "parse" and current >= (4 if blocked == "embed" else 12):
            filled.set()
        if blocked == "store" and phase == "write-data" and not started.is_set():
            started.set()
            assert release.wait(20)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        work = pool.submit(pipeline(tmp_path, skip_embeddings=blocked == "store").run,
            entries, parse, embed if blocked == "embed" else None,
            progress_callback=progress, cancellation_token=token)
        try:
            assert started.wait(10) and filled.wait(10)
            token.cancel()
            assert not work.done()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(tmp_path / "chunks.db"), "duckdb")
        finally:
            token.cancel()
            release.set()
        with pytest.raises(native.PipelineCancelledError):
            work.result(timeout=20)
    assert len(parsed) < len(files) and token.cleanup_complete
    native.acquire_install_lease(str(tmp_path / "chunks.db"), "duckdb").close()


def test_cancel_after_preparation_commits_replacement_and_restores_indexes(tmp_path):
    source = tmp_path / "example.py"
    source.write_text("def example():\n    return 1\n")
    entries = [(str(source), source.name)]
    def configured():
        return pipeline(tmp_path, skip_embeddings=False,
                        embedding_provider="callback", embedding_model="fixture")
    def embed(texts):
        return [[1.0, 2.0, 3.0] for _ in texts]
    configured().run(entries, parse_fixture, embed)
    source.write_text("def example():\n    return 222\n")
    token = native.PipelineCancellationToken()
    prepared = []
    def progress(phase, current, total, *args):
        if phase == "write-data" and not prepared:
            prepared.append(True)
            token.cancel()  # existing post-prepare/pre-insert progress point
    with pytest.raises(native.PipelineCancelledError):
        configured().run(entries, parse_fixture, embed,
            progress_callback=progress, cancellation_token=token)
    assert prepared and token.cleanup_complete
    provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
    try:
        provider.connect()
        rows = provider.execute_query("SELECT code FROM chunks")
        assert len(rows) == 1 and "222" in rows[0]["code"]
        assert provider.execute_query(
            "SELECT index_name FROM duckdb_indexes() WHERE sql ILIKE '%HNSW%'"
        )
    finally:
        provider.close()
    native.acquire_install_lease(str(tmp_path / "chunks.db"), "duckdb").close()


import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

@pytest.mark.parametrize("vendor,model", [("openai", "text-embedding-3-small"),
                                         ("voyageai", "voyage-code-3")])
def test_native_stop_prevents_retry_and_missing_vector_writes(tmp_path, vendor, model):
    seen = threading.Event()
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(429)
            self.send_header("Retry-After", "120")
            self.send_header("Content-Length", "0")
            self.end_headers()
            seen.set()
        def log_message(self, *args):
            return
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    try:
        server_thread.start()
        source = tmp_path / "example.py"
        source.write_text("def example():\n    return 42\n")
        token = native.PipelineCancellationToken()
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(pipeline(tmp_path, skip_embeddings=False,
                    embedding_provider=vendor, embedding_model=model,
                    embedding_api_key="test-secret",
                    embedding_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                ).run, [(str(source), source.name)], parse_fixture,
                    cancellation_token=token)
                try:
                    assert seen.wait(10)
                finally:
                    token.cancel()
                with pytest.raises(native.PipelineCancelledError):
                    future.result(timeout=10)
            assert len(requests) == 1 and token.cleanup_complete
            provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
            try:
                provider.connect()
                assert provider.get_stats()["chunks"] == 0
            finally:
                provider.close()
        finally:
            token.cancel()
    finally:
        # Server ownership starts before fixture/token setup and never depends
        # on a token existing (including the missing-API RED case).
        try:
            if server_thread.ident is not None:
                server.shutdown()
        finally:
            try:
                server.server_close()
            finally:
                if server_thread.ident is not None:
                    server_thread.join(timeout=5)
                    assert not server_thread.is_alive()

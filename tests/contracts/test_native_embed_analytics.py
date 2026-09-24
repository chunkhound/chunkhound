"""End-to-end contract: the native Rust OpenAI/VoyageAI embedding path
records analytics directly in Rust (src/embed/openai.rs, voyageai.rs),
with zero Python involvement in the embedding HTTP call itself -- this is
the Phase 4 native-path instrumentation from the ChunkHound Per-User
Analytics design. Uses a real local HTTP server (no mocking) and the real
chunkhound_native.AnalyticsRecorder, exactly like
test_native_embed_provider.py does for the embedding path itself.
"""

from __future__ import annotations

import glob
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from chunkhound.pipeline_bridge import parse_batch_callback
from chunkhound_native import AnalyticsRecorder, IndexingPipeline


class _EmbeddingServer:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.requests: list[dict[str, Any]] = []
        self._served = 0
        self._fail_first = fail_first
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                fixture.requests.append(body)
                fixture._served += 1
                if fixture._fail_first and fixture._served == 1:
                    self.send_response(429)
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                    return
                texts = body["input"]
                payload = {
                    "data": [
                        {"index": i, "embedding": [1.0, 2.0, 3.0]}
                        for i in range(len(texts))
                    ],
                    "usage": {"total_tokens": 7 * len(texts)},
                }
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> _EmbeddingServer:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _read_events(buffer_dir: Path) -> list[dict]:
    events = []
    for path in glob.glob(str(buffer_dir / "buffer-*.jsonl")):
        for line in Path(path).read_text().splitlines():
            events.append(json.loads(line))
    return events


def _run_pipeline(tmp_path: Path, server: _EmbeddingServer, recorder, handle) -> Any:
    file_path = tmp_path / "sample.py"
    file_path.write_text("def one():\n    return 1\n\ndef two():\n    return 2\n")
    pipeline = IndexingPipeline(
        {
            "db_path": str(tmp_path / "db"),
            "db_batch_size": 100,
            "compaction_threshold": None,
            "compaction_min_size_mb": 10,
            "disk_usage_limit_mb": None,
            "parse_batch_size": 200,
            "parse_thread_pool_size": 1,
            "embed_thread_pool_size": 1,
            "embed_batch_size": 100,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_api_key": "secret-test-key",
            "embedding_base_url": server.base_url,
            "embed_max_tokens_per_batch": 8191,
            "skip_embeddings": False,
        }
    )
    return pipeline.run(
        files=[(str(file_path), "sample.py")],
        parse_batch_callback=parse_batch_callback,
        embed_batch_callback=None,
        progress_callback=None,
        incremental=False,
        analytics_recorder=recorder,
        analytics_handle=handle,
    )


def test_native_openai_path_records_analytics_with_zero_python_round_trips(
    tmp_path: Path,
) -> None:
    buffer_dir = tmp_path / "analytics"
    recorder = AnalyticsRecorder(
        {
            "enabled": True,
            "buffer_dir": str(buffer_dir),
            "salt_path": str(buffer_dir / "salt"),
            "repository_dir": str(tmp_path),
            "os_username": "test-user",
            "chunkhound_version": "test",
            "flush_interval_seconds": 999999,
            "flush_batch_size": 999999,
        }
    )
    handle = recorder.start_command("index", "cli", json.dumps({"mode": "initial"}))

    with _EmbeddingServer() as server:
        report = _run_pipeline(tmp_path, server, recorder, handle)

    recorder.end_command(handle, True)

    assert report.embeddings_generated > 0
    events = _read_events(buffer_dir)
    assert len(events) == 1
    embedding = events[0]["providers"]["embedding"][0]
    assert embedding["provider"] == "openai"
    assert embedding["model"] == "text-embedding-3-small"
    assert embedding["calls"] == 1
    assert embedding["fails"] == 0
    assert embedding["input_tokens"] > 0


def test_native_openai_path_records_retry_then_success(tmp_path: Path) -> None:
    buffer_dir = tmp_path / "analytics"
    recorder = AnalyticsRecorder(
        {
            "enabled": True,
            "buffer_dir": str(buffer_dir),
            "salt_path": str(buffer_dir / "salt"),
            "repository_dir": str(tmp_path),
            "os_username": "test-user",
            "chunkhound_version": "test",
            "flush_interval_seconds": 999999,
            "flush_batch_size": 999999,
        }
    )
    handle = recorder.start_command("index", "cli", json.dumps({"mode": "initial"}))

    with _EmbeddingServer(fail_first=True) as server:
        report = _run_pipeline(tmp_path, server, recorder, handle)

    recorder.end_command(handle, True)

    assert report.embeddings_generated > 0
    events = _read_events(buffer_dir)
    embedding = events[0]["providers"]["embedding"][0]
    assert embedding["calls"] == 2
    assert embedding["fails"] == 1
    assert embedding["error_types"] == {"RateLimited": 1}

"""End-to-end contract for native OpenAI-compatible embedding requests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from chunkhound.pipeline_bridge import parse_batch_callback
from chunkhound_native import IndexingPipeline


class _EmbeddingServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                fixture.requests.append(body)
                texts = body["input"]
                payload = {
                    "data": [
                        {"index": index, "embedding": [float(index + 1), 2.0, 3.0]}
                        for index in reversed(range(len(texts)))
                    ]
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


def test_native_openai_path_orders_responses_and_skips_python_callback(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        "def first():\n    return 'one'\n\ndef second():\n    return 'two'\n"
    )
    db_dir = tmp_path / "db"

    def unexpected_callback(texts: list[str]) -> list[list[float]]:
        raise AssertionError("native provider should bypass the Python callback")

    with _EmbeddingServer() as server:
        pipeline = IndexingPipeline(
            {
                "db_path": str(db_dir),
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
        report = pipeline.run(
            files=[(str(file_path), "sample.py")],
            parse_batch_callback=parse_batch_callback,
            embed_batch_callback=unexpected_callback,
            progress_callback=None,
            incremental=False,
        )

    assert report.chunks_written > 0
    assert report.embeddings_generated == report.chunks_written
    assert len(server.requests) == 1
    assert server.requests[0]["model"] == "text-embedding-3-small"

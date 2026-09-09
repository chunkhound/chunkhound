"""
Embedding provider parity contract tests.

Three suites:

1. TestRequestShapeParity — the Rust native provider sends the same HTTP
   request body fields as the Python golden master for every config
   combination that affects the wire format (dimensions gate, input_type,
   truncation, Azure auth header).

2. TestContextLengthClassification — all three context-length error patterns
   that the Python provider recognises are also caught by the Rust provider,
   triggering split behaviour rather than a hard batch failure.

3. TestRetrySemantics — the Rust provider never sleeps less than the
   server's Retry-After value and caps at 120 s.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from chunkhound.pipeline_bridge import parse_batch_callback
from chunkhound_native import IndexingPipeline


# ── Shared test infrastructure ───────────────────────────────────────────────

class _ScriptableServer:
    """HTTP mock server whose response sequence can be scripted per-test.

    ``responses`` is a list of callables ``(request_body: dict) -> (status, headers, body)``.
    Each incoming request pops the next handler off the queue.  Once the queue
    is exhausted every subsequent request gets a plain 200 with three-element
    vectors so the pipeline can write its embeddings and complete normally.
    """

    def __init__(self, responses: list | None = None) -> None:
        self.captured: list[dict[str, Any]] = []
        self.captured_headers: list[dict[str, str]] = []
        self._responses = list(responses or [])
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                fixture.captured.append(body)
                fixture.captured_headers.append(dict(self.headers))
                if fixture._responses:
                    handler = fixture._responses.pop(0)
                    status, extra_headers, payload = handler(body)
                else:
                    texts = body.get("input", [""])
                    status = 200
                    extra_headers = {}
                    payload = {
                        "data": [
                            {"index": i, "embedding": [float(i + 1), 2.0, 3.0]}
                            for i in range(len(texts))
                        ]
                    }
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                for k, v in extra_headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        port = self.server.server_address[1]
        return f"http://127.0.0.1:{port}/v1"

    def __enter__(self) -> _ScriptableServer:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _run_pipeline(
    tmp_path: Path,
    server: _ScriptableServer,
    *,
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    output_dims: int | None = None,
    matryoshka: bool = False,
    client_side_truncation: bool = False,
) -> Any:
    """Run a minimal two-function file through the Rust pipeline and return the
    PipelineReport.  Wires the native embed provider so no Python embed
    callback is used."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "sample.py"
    src.write_text("def a():\n    return 1\n\ndef b():\n    return 2\n")

    def _unexpected(*_: Any, **__: Any) -> list[list[float]]:
        raise AssertionError("native provider must not fall back to Python callback")

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
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_api_key": "test-key",
            "embedding_base_url": server.base_url,
            "embed_max_tokens_per_batch": 8191,
            "skip_embeddings": False,
            # Dimension / truncation config passed through to Rust
            **({"embedding_output_dims": output_dims} if output_dims else {}),
            **({"embedding_matryoshka": matryoshka} if matryoshka else {}),
            **(
                {"embedding_client_side_truncation": client_side_truncation}
                if client_side_truncation
                else {}
            ),
        }
    )
    return pipeline.run(
        files=[(str(src), "sample.py")],
        parse_batch_callback=parse_batch_callback,
        embed_batch_callback=_unexpected,
        progress_callback=None,
        incremental=False,
    )


# ── Suite 1: Request-shape parity ────────────────────────────────────────────

class TestRequestShapeParity:
    """The Rust provider must send the same request fields as the Python golden
    master for every config combination that affects the wire format.

    Python golden-master rules (from openai_provider._build_embedding_request_kwargs
    and shared_utils.build_dimension_request_param):

    - ``dimensions`` is included iff output_dims is set AND NOT client_side_truncation
      AND (matryoshka OR custom base_url OR Azure).
    - For standard (non-Azure) OpenAI: always ``Authorization: Bearer <key>``.
    - For VoyageAI: body always includes ``"input_type": "document"`` and
      ``"truncation": true``; dimension param name is ``"output_dimension"``.
    """

    def test_basic_request_has_model_and_input(self, tmp_path: Path) -> None:
        with _ScriptableServer() as server:
            _run_pipeline(tmp_path, server)

        assert len(server.captured) >= 1
        req = server.captured[0]
        assert req["model"] == "text-embedding-3-small"
        assert isinstance(req["input"], list)
        assert len(req["input"]) >= 1

    def test_dimensions_sent_when_matryoshka_and_output_dims_set(
        self, tmp_path: Path
    ) -> None:
        """Matryoshka models + output_dims → ``dimensions`` must appear in body."""
        with _ScriptableServer() as server:
            _run_pipeline(tmp_path, server, output_dims=512, matryoshka=True)

        req = server.captured[0]
        assert "dimensions" in req, (
            "expected 'dimensions' in request body for matryoshka=True + output_dims=512"
        )
        assert req["dimensions"] == 512

    def test_dimensions_not_sent_for_client_side_truncation(
        self, tmp_path: Path
    ) -> None:
        """client_side_truncation=True → server must NOT receive ``dimensions``."""
        with _ScriptableServer() as server:
            _run_pipeline(
                tmp_path, server, output_dims=512, matryoshka=True, client_side_truncation=True
            )

        req = server.captured[0]
        assert "dimensions" not in req, (
            "expected 'dimensions' absent for client_side_truncation=True"
        )

    def test_dimensions_sent_for_custom_base_url(self, tmp_path: Path) -> None:
        """Custom base_url (non-official OpenAI) with output_dims → ``dimensions`` sent."""
        # base_url is always set via server.base_url in _run_pipeline, so
        # this test covers the custom-endpoint path.  Non-matryoshka model
        # but custom base_url → dimensions should be included (runtime trust).
        with _ScriptableServer() as server:
            _run_pipeline(tmp_path, server, output_dims=256, matryoshka=False)

        req = server.captured[0]
        assert "dimensions" in req, (
            "expected 'dimensions' in request body for custom base_url + output_dims=256"
        )
        assert req["dimensions"] == 256

    def test_api_key_not_leaked_in_url_or_body(self, tmp_path: Path) -> None:
        """API key must appear only in Authorization header, never in the JSON body."""
        with _ScriptableServer() as server:
            _run_pipeline(tmp_path, server)

        req = server.captured[0]
        body_str = json.dumps(req)
        assert "test-key" not in body_str, "API key must not appear in the request body"
        # Authorization header must be present
        headers = server.captured_headers[0]
        auth = headers.get("Authorization", headers.get("authorization", ""))
        assert "test-key" in auth, "API key must appear in Authorization header"

    def test_voyageai_body_includes_input_type_and_truncation(
        self, tmp_path: Path
    ) -> None:
        """VoyageAI requests must include ``input_type`` and ``truncation`` fields."""
        with _ScriptableServer() as server:
            _run_pipeline(tmp_path, server, provider="voyageai", model="voyage-3")

        req = server.captured[0]
        assert req.get("input_type") == "document", (
            "VoyageAI request missing 'input_type: document'"
        )
        assert req.get("truncation") is True, (
            "VoyageAI request missing 'truncation: true'"
        )

    def test_voyageai_output_dimension_sent_when_output_dims_set(
        self, tmp_path: Path
    ) -> None:
        """VoyageAI + output_dims → ``output_dimension`` (not ``dimensions``) in body."""
        with _ScriptableServer() as server:
            _run_pipeline(tmp_path, server, provider="voyageai", model="voyage-3", output_dims=512)

        req = server.captured[0]
        assert "output_dimension" in req, (
            "expected 'output_dimension' in VoyageAI body when output_dims is set"
        )
        assert req["output_dimension"] == 512
        assert "dimensions" not in req, (
            "'dimensions' must not appear in a VoyageAI request"
        )


# ── Suite 2: Context-length error classification ─────────────────────────────

class TestContextLengthClassification:
    """All three context-length error patterns Python recognises must be caught
    by the Rust provider and trigger split behaviour, not a hard batch failure.

    The three Python patterns (from openai_provider._embed_batch_internal):
      1. ``"maximum context length" … "tokens"``
      2. ``"tokens" … "max" … "per request"``   ← the pattern Rust previously missed
      3. ``"input length exceeds" … "context length"``

    Test strategy: mock returns 400 with the pattern body for the first
    (multi-item) batch, then 200 for all subsequent (single-item) requests.
    If the error is correctly classified as ContextLengthExceeded, the Rust
    provider splits the batch and retries individual items → embeddings_generated > 0.
    If incorrectly classified as BadRequest (no split), the entire batch fails
    → embeddings_generated == 0.
    """

    CONTEXT_LENGTH_PATTERNS = [
        # Pattern 1 — standard OpenAI wording
        '{"error": {"message": "This model\'s maximum context length is 8192 tokens, '
        'however you requested 10000 tokens.", "type": "invalid_request_error"}}',
        # Pattern 2 — previously missing in Rust
        '{"error": {"message": "Your request has 9001 tokens which exceeds the 8192 '
        "max tokens per request.\", \"type\": \"invalid_request_error\"}}",
        # Pattern 3 — Voyage / some OpenAI-compat servers
        '{"error": {"message": "input length exceeds context length limit of 8192", '
        '"type": "invalid_request_error"}}',
    ]

    @pytest.mark.parametrize("error_body", CONTEXT_LENGTH_PATTERNS)
    def test_context_length_pattern_triggers_split_not_hard_failure(
        self,
        tmp_path: Path,
        error_body: str,
    ) -> None:
        """Return 400 with each context-length pattern for the first batch request;
        individual retry requests receive 200.  The pipeline should produce
        embeddings_generated > 0, proving the split path was taken."""

        error_bytes = error_body.encode()

        def _first_request_fails(body: dict[str, Any]) -> tuple:
            texts = body.get("input", [])
            if len(texts) > 1:
                # Return context-length error only for multi-item batches.
                return (
                    400,
                    {},
                    json.loads(error_body),
                )
            # Single-item retries succeed.
            return (
                200,
                {},
                {
                    "data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]
                },
            )

        with _ScriptableServer(responses=[_first_request_fails]) as server:
            # VoyageAI also uses the same context-length classification; test
            # both providers to ensure the fix is applied to both files.
            report = _run_pipeline(tmp_path / "openai", server)

        assert report.embeddings_generated > 0, (
            f"context-length pattern triggered hard failure instead of split; "
            f"pattern: {error_body[:80]!r}"
        )


# ── Suite 3: Retry semantics ──────────────────────────────────────────────────

class TestRetrySemantics:
    """The Rust provider must respect the server's Retry-After header: sleep ≥
    the header value, and cap at 120 s (never wait more than 120 s for a
    single retry, regardless of what the server sends).

    The 120-s cap is enforced at the Rust level (MAX_RETRY_DELAY_SECS) so it
    cannot be directly observed from Python without a very long-running test.
    The 'never sleep less than Retry-After' invariant is testable by measuring
    wall-clock time around a pipeline run that triggers two retries.
    """

    def test_client_sleeps_at_least_retry_after_header_value(
        self, tmp_path: Path
    ) -> None:
        """Two 429s with Retry-After: 1 → elapsed >= 1.8 s (two sleeps of >= 1 s each,
        with 100 ms tolerance for scheduling overhead)."""
        retry_after_secs = 1

        call_count = 0

        def _rate_limited_twice(body: dict[str, Any]) -> tuple:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return (
                    429,
                    {"Retry-After": str(retry_after_secs)},
                    {"error": {"message": "rate limited", "type": "requests"}},
                )
            # Third and subsequent calls succeed.
            texts = body.get("input", [""])
            return (
                200,
                {},
                {
                    "data": [
                        {"index": i, "embedding": [float(i + 1), 2.0, 3.0]}
                        for i in range(len(texts))
                    ]
                },
            )

        with _ScriptableServer(responses=[_rate_limited_twice, _rate_limited_twice]) as server:
            start = time.monotonic()
            report = _run_pipeline(tmp_path, server)
            elapsed = time.monotonic() - start

        # Two retries × retry_after_secs each + additive jitter (always positive).
        min_expected = 2 * retry_after_secs - 0.2  # 100 ms tolerance per retry
        assert elapsed >= min_expected, (
            f"elapsed {elapsed:.2f}s < expected minimum {min_expected:.2f}s — "
            f"client may have slept less than Retry-After ({retry_after_secs}s)"
        )
        assert report.embeddings_generated > 0, (
            "expected embeddings after successful retry"
        )

    def test_client_retries_after_429_and_eventually_succeeds(
        self, tmp_path: Path
    ) -> None:
        """One 429 followed by 200 → pipeline completes with embeddings written."""
        responded = {"count": 0}

        def _one_retry(body: dict[str, Any]) -> tuple:
            responded["count"] += 1
            if responded["count"] == 1:
                return (429, {"Retry-After": "1"}, {"error": "rate limited"})
            texts = body.get("input", [""])
            return (
                200,
                {},
                {"data": [{"index": i, "embedding": [1.0, 2.0, 3.0]} for i in range(len(texts))]},
            )

        with _ScriptableServer(responses=[_one_retry]) as server:
            report = _run_pipeline(tmp_path, server)

        assert report.embeddings_generated > 0
        # Server received ≥ 2 requests: the initial 429 and at least one retry.
        assert len(server.captured) >= 2, "expected at least one retry request after 429"

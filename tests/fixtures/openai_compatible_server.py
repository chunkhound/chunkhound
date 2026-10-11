"""Loopback-only scripted OpenAI Chat Completions test server.

The fixture exercises the real OpenAI SDK HTTP serialization without allowing
requests to leave the local machine. Scripts are selected by request content,
not arrival order, so concurrent callers remain deterministic.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

RequestPredicate = Callable[[Mapping[str, Any], str], bool]


@dataclass(frozen=True)
class ChatCompletionScript:
    """A content-matched response expected a fixed number of times."""

    name: str
    content: str | None = "ok"
    finish_reason: str = "stop"
    prompt_tokens: int = 1
    completion_tokens: int = 1
    marker: str | None = None
    predicate: RequestPredicate | None = None
    stage: str | None = None
    expected_calls: int = 1
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)

    def matches(self, body: Mapping[str, Any], flattened_messages: str) -> bool:
        """Match by a unique marker and/or a caller-supplied stable predicate."""
        marker_matches = self.marker is None or self.marker in flattened_messages
        predicate_matches = self.predicate is None or self.predicate(
            body, flattened_messages
        )
        return marker_matches and predicate_matches


class OpenAICompatibleTestServer:
    """Context-managed, concurrent-safe Chat Completions loopback server."""

    def __init__(self, scripts: list[ChatCompletionScript]):
        if not scripts:
            raise ValueError("At least one response script is required")
        if any(
            script.marker is None and script.predicate is None for script in scripts
        ):
            raise ValueError("Every script needs a marker or predicate")
        markers = [script.marker for script in scripts if script.marker is not None]
        if len(markers) != len(set(markers)):
            raise ValueError("Script markers must be unique")
        if any(script.expected_calls < 1 for script in scripts):
            raise ValueError("expected_calls must be positive")

        self._scripts = tuple(scripts)
        self._remaining = {script.name: script.expected_calls for script in scripts}
        if len(self._remaining) != len(scripts):
            raise ValueError("Script names must be unique")
        self._lock = threading.Lock()
        self._sequence = 0
        self.requests: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        """Return the SDK base URL, refusing anything except IPv4 loopback."""
        if self._server is None:
            raise RuntimeError("Server is not running")
        host, port = self._server.server_address[:2]
        url = f"http://{host}:{port}/v1"
        self.assert_loopback_url(url)
        return url

    @staticmethod
    def assert_loopback_url(url: str) -> None:
        """Fail closed if a test attempts to configure a non-loopback endpoint."""
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise AssertionError(f"Refusing non-loopback OpenAI test URL: {url}")

    def start(self) -> OpenAICompatibleTestServer:
        if self._server is not None:
            raise RuntimeError("Server is already running")
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                fixture._handle_post(self)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="fake-openai-compatible-server",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> OpenAICompatibleTestServer:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _unmatched_requests(self) -> list[dict[str, Any]]:
        """Requests that matched no script — the server answered them with 400."""
        with self._lock:
            return [
                request
                for request in self.requests
                if request["matched_script"] is None
            ]

    def assert_all_scripts_consumed(self) -> None:
        """Assert expected calls were consumed and no request was unmatched."""
        with self._lock:
            remaining = {
                name: count for name, count in self._remaining.items() if count
            }
        assert not remaining, f"Unconsumed OpenAI response scripts: {remaining}"
        unmatched = self._unmatched_requests()
        assert not unmatched, f"Unexpected OpenAI requests: {unmatched}"

    def assert_no_unmatched_requests(self) -> None:
        """Assert every recorded request matched a scripted response.

        A call beyond the scripted budget gets a 400 and is recorded as
        unmatched, so callers turn an over-budget flake into a hard failure.
        """
        unmatched = self._unmatched_requests()
        assert not unmatched, f"Unexpected OpenAI requests: {unmatched}"

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path != "/v1/chat/completions":
            self._send_json(handler, 404, {"error": {"message": "Not found"}})
            return

        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            body = json.loads(handler.rfile.read(content_length))
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(handler, 400, {"error": {"message": str(error)}})
            return

        flattened = _flatten_messages(body.get("messages"))
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            script = next(
                (
                    candidate
                    for candidate in self._scripts
                    if self._remaining[candidate.name] > 0
                    and candidate.matches(body, flattened)
                ),
                None,
            )
            if script is not None:
                self._remaining[script.name] -= 1
            usage = _usage(script)
            self.requests.append(
                {
                    "sequence": sequence,
                    "time_ns": time.monotonic_ns(),
                    "json": body,
                    "flattened_messages": flattened,
                    "prompt_hash": hashlib.sha256(flattened.encode()).hexdigest(),
                    "stage": (script.stage or script.name) if script else "unmatched",
                    "model": body.get("model"),
                    "max_tokens": body.get("max_tokens"),
                    "max_completion_tokens": body.get("max_completion_tokens"),
                    "matched_script": script.name if script else None,
                    "finish_reason": (
                        script.finish_reason
                        if script is not None and script.status_code == 200
                        else None
                    ),
                    "usage": usage,
                }
            )

        if script is None:
            self._send_json(
                handler,
                400,
                {"error": {"message": "No scripted response matched this request"}},
            )
            return
        if script.status_code != 200:
            self._send_json(
                handler,
                script.status_code,
                {"error": {"message": script.content or script.name}},
                script.headers,
            )
            return

        if body.get("stream"):
            self._send_sse(handler, body, script, sequence)
            return

        response = {
            "id": f"chatcmpl-loopback-{sequence}",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model", "loopback-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": script.content},
                    "finish_reason": script.finish_reason,
                }
            ],
            "usage": usage,
        }
        self._send_json(handler, 200, response, script.headers)

    def _send_sse(
        self,
        handler: BaseHTTPRequestHandler,
        body: Mapping[str, Any],
        script: ChatCompletionScript,
        sequence: int,
    ) -> None:
        """Answer a streaming request as OpenAI `chat.completion.chunk` SSE.

        CLI subprocess providers (opencode) request `stream=true`; a buffered
        JSON body is not a valid stream, so it must be chunked. The whole
        payload is available at once, so it is sent with Content-Length while
        still carrying `text/event-stream` framing the client parses line-wise.
        """
        encoded = _sse_payload(body, script, sequence)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Content-Length", str(len(encoded)))
        for name, value in script.headers.items():
            handler.send_header(name, value)
        handler.end_headers()
        handler.wfile.write(encoded)

    @staticmethod
    def _send_json(
        handler: BaseHTTPRequestHandler,
        status: int,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(encoded)))
        for name, value in (headers or {}).items():
            handler.send_header(name, value)
        handler.end_headers()
        handler.wfile.write(encoded)


def _sse_payload(
    body: Mapping[str, Any],
    script: ChatCompletionScript,
    sequence: int,
) -> bytes:
    """Encode a scripted completion as OpenAI `chat.completion.chunk` SSE."""
    base = {
        "id": f"chatcmpl-loopback-{sequence}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": body.get("model", "loopback-model"),
    }

    def chunk(delta: dict[str, Any], finish: str | None) -> dict[str, Any]:
        choice = {"index": 0, "delta": delta, "finish_reason": finish}
        event = {**base, "choices": [choice]}
        if finish is not None:
            event["usage"] = _usage(script)
        return event

    chunks = [
        chunk({"role": "assistant"}, None),
        chunk({"content": script.content or ""}, None),
        chunk({}, script.finish_reason),
    ]
    events = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (events + "data: [DONE]\n\n").encode()


def _usage(script: ChatCompletionScript | None) -> dict[str, int] | None:
    """Token usage for a matched 200 script, else None."""
    if script is None or script.status_code != 200:
        return None
    return {
        "prompt_tokens": script.prompt_tokens,
        "completion_tokens": script.completion_tokens,
        "total_tokens": script.prompt_tokens + script.completion_tokens,
    }


def _flatten_messages(messages: object) -> str:
    """Create a stable, searchable representation of Chat Completions messages."""
    if not isinstance(messages, list):
        return ""
    flattened: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            flattened.append(json.dumps(message, sort_keys=True))
            continue
        role = message.get("role", "")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True, separators=(",", ":"))
        flattened.append(f"{role}:{content}")
    return "\n".join(flattened)

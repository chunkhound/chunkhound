"""Loopback OpenAI-compatible embedding server for deterministic integration tests.

Vectors are selected by content markers, never by request order or batch
position. The same text therefore yields the same vector after batching,
retries, or realtime re-indexing, so semantic assertions survive refactors.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True)
class EmbeddingConcept:
    """A labelled vector shared by every text that carries one of its markers."""

    name: str
    markers: tuple[str, ...]
    vector: tuple[float, ...]


def unit_vector(axis: int, dimensions: int) -> tuple[float, ...]:
    """Return a unit basis vector so distinct concepts are orthogonal."""
    if not 0 <= axis < dimensions:
        raise ValueError(f"axis must be in [0, {dimensions}), got {axis}")
    return tuple(1.0 if index == axis else 0.0 for index in range(dimensions))


class MockEmbeddingServer:
    """Deterministic embedding HTTP server with explicit concept responses."""

    def __init__(
        self,
        concepts: tuple[EmbeddingConcept, ...],
        *,
        dimensions: int = 8,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        for concept in concepts:
            if len(concept.vector) != dimensions:
                raise ValueError(
                    f"Concept '{concept.name}' has {len(concept.vector)} dimensions, "
                    f"expected {dimensions}"
                )
            if not concept.markers:
                raise ValueError(f"Concept '{concept.name}' needs at least one marker")

        self._concepts = concepts
        self.dimensions = dimensions
        self.requests: list[dict[str, Any]] = []
        self.matched_concepts: Counter[str] = Counter()
        self.fallback_texts: list[str] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        """Return the SDK base URL, refusing anything except IPv4 loopback."""
        if self._server is None:
            raise RuntimeError("Server is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def vector_for(self, text: str) -> list[float]:
        """Return a stable vector for one input text.

        Each concept is scored by its longest marker that actually occurs, so
        an absent long marker cannot outrank a present specific one. Ties keep
        declaration order because ``max`` returns the first maximum.
        """
        winner: EmbeddingConcept | None = None
        winner_length = 0
        for concept in self._concepts:
            present = max(
                (len(marker) for marker in concept.markers if marker in text),
                default=0,
            )
            if present > winner_length:
                winner, winner_length = concept, present

        if winner is not None:
            self.matched_concepts[winner.name] += 1
            return list(winner.vector)

        # Unmarked content still needs a stable vector for the rest of the
        # suite. Hash it instead of using request position, which would drift
        # when batching or concurrency changes.
        self.fallback_texts.append(text)
        digest = hashlib.blake2b(
            text.encode(), digest_size=self.dimensions * 2
        ).digest()
        raw = [
            int.from_bytes(digest[index : index + 2], "big") / 65535.0 * 2.0 - 1.0
            for index in range(0, len(digest), 2)
        ]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]

    def start(self) -> MockEmbeddingServer:
        if self._server is not None:
            raise RuntimeError("Server is already running")
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                fixture._handle_embeddings(self)

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                if self.path != "/health":
                    fixture._send_json(self, 404, {"error": "Not found"})
                    return
                fixture._send_json(
                    self,
                    200,
                    {
                        "healthy": True,
                        "service": "mock-embedding-server",
                        "dimensions": fixture.dimensions,
                        "concept_count": len(fixture._concepts),
                    },
                )

            def log_message(self, *_: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="mock-embedding-server",
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

    def __enter__(self) -> MockEmbeddingServer:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _handle_embeddings(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path != "/v1/embeddings":
            self._send_json(handler, 404, {"error": "Not found"})
            return

        try:
            length = int(handler.headers.get("Content-Length", "0"))
            body = json.loads(handler.rfile.read(length))
            texts = body["input"]
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, list) or not all(
                isinstance(text, str) for text in texts
            ):
                raise ValueError("input must be a string or list of strings")
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(handler, 400, {"error": str(error)})
            return

        self.requests.append(body)
        self._send_json(
            handler,
            200,
            {
                "object": "list",
                "model": body.get("model", "mock-embedding"),
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": self.vector_for(text),
                    }
                    for index, text in enumerate(texts)
                ],
                "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
            },
        )

    @staticmethod
    def _send_json(
        handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
    ) -> None:
        encoded = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(encoded)))
        handler.end_headers()
        handler.wfile.write(encoded)

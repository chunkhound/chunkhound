"""Contract tests for separate query and document prompts (OpenAI-compatible).

Models such as Qwen3-Embedding and EmbeddingGemma expect different input text
for search queries and indexed documents. ``query_prefix`` and
``document_template`` configure that, and must leave requests untouched when
unset.

The provider is exercised against a real local HTTP server that speaks the
OpenAI ``/v1/embeddings`` protocol and records every request body.
"""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.utils.embedding_utils import (
    format_document_for_provider,
    render_document_text,
)

DIMS = 4


class _FakeEmbeddingServer:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                texts = body["input"]
                outer.inputs.append(list(texts))
                payload = json.dumps(
                    {
                        "object": "list",
                        "model": body["model"],
                        "data": [
                            {
                                "object": "embedding",
                                "index": i,
                                "embedding": [0.1] * DIMS,
                            }
                            for i in range(len(texts))
                        ],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def sent(self) -> list[str]:
        return [text for batch in self.inputs for text in batch]


@pytest.fixture
def server() -> Iterator[_FakeEmbeddingServer]:
    fake = _FakeEmbeddingServer()
    fake.start()
    try:
        yield fake
    finally:
        fake.stop()


def _provider(server: _FakeEmbeddingServer, **settings: str):
    config = EmbeddingConfig(
        provider="openai",
        model="embeddinggemma-2",
        base_url=server.base_url,
        **settings,
    )
    return EmbeddingProviderFactory.create_provider(config)


class TestRequestPayloads:
    @pytest.mark.asyncio
    async def test_defaults_send_text_unchanged(self, server):
        provider = _provider(server)

        await provider.embed_query(["find retries"])
        await provider.embed(["def retry(): pass"])

        assert server.sent == ["find retries", "def retry(): pass"]

    @pytest.mark.asyncio
    async def test_query_prefix_applies_to_queries_only(self, server):
        provider = _provider(server, query_prefix="task: code retrieval | query: ")

        await provider.embed_query(["find retries"])
        await provider.embed(["def retry(): pass"])

        assert server.sent == [
            "task: code retrieval | query: find retries",
            "def retry(): pass",
        ]

    @pytest.mark.asyncio
    async def test_document_template_is_exposed_for_indexing(self, server):
        provider = _provider(
            server, document_template="title: {file_path} | text: {text}"
        )

        assert (
            format_document_for_provider(
                provider, "def retry(): pass", "src/network/retry.ts"
            )
            == "title: src/network/retry.ts | text: def retry(): pass"
        )
        # Queries never go through the document template.
        await provider.embed_query(["find retries"])
        assert server.sent == ["find retries"]


class TestDocumentTemplateRendering:
    def test_identity_template_returns_text(self):
        assert render_document_text("{text}", "x = 1", "a.py") == "x = 1"

    def test_file_path_and_filename(self):
        template = "title: {filename} ({file_path}) | text: {text}"

        assert (
            render_document_text(template, "x", "src\\network\\retry.ts")
            == "title: retry.ts (src/network/retry.ts) | text: x"
        )

    def test_missing_file_metadata_renders_none(self):
        assert (
            render_document_text("title: {file_path} | text: {text}", "x")
            == "title: none | text: x"
        )
        assert render_document_text("{filename}: {text}", "x") == "none: x"

    def test_braces_in_source_code_stay_literal(self):
        code = "const o = {text}; fmt('{file_path}', {})"

        assert (
            render_document_text("title: {file_path} | text: {text}", code, "a.ts")
            == f"title: a.ts | text: {code}"
        )

    def test_providers_without_a_template_keep_text(self):
        class NoTemplate:
            pass

        assert format_document_for_provider(NoTemplate(), "x", "a.py") == "x"


class TestConfigValidation:
    def test_defaults_do_not_reach_provider_config(self):
        config = EmbeddingConfig(model="m", base_url="http://localhost:1")

        assert config.query_prefix == ""
        assert config.document_template == "{text}"
        provider_config = config.get_provider_config()
        assert "query_prefix" not in provider_config
        assert "document_template" not in provider_config

    def test_settings_reach_provider_config(self):
        config = EmbeddingConfig(
            model="m",
            base_url="http://localhost:1",
            query_prefix="query: ",
            document_template="passage: {text}",
        )

        provider_config = config.get_provider_config()
        assert provider_config["query_prefix"] == "query: "
        assert provider_config["document_template"] == "passage: {text}"

    def test_unknown_placeholder_is_rejected_by_name(self):
        with pytest.raises(ValueError, match=r"unknown placeholder \{title\}"):
            EmbeddingConfig(document_template="{title} | {text}")

    def test_template_without_text_is_rejected(self):
        with pytest.raises(ValueError, match=r"\{text\}"):
            EmbeddingConfig(document_template="title: {file_path}")

    def test_voyageai_rejects_prompt_settings(self):
        with pytest.raises(ValueError, match="only supported with provider 'openai'"):
            EmbeddingConfig(provider="voyageai", query_prefix="query: ")

    def test_load_from_env(self, monkeypatch):
        monkeypatch.setenv("CHUNKHOUND_EMBEDDING__QUERY_PREFIX", "query: ")
        monkeypatch.setenv("CHUNKHOUND_EMBEDDING__DOCUMENT_TEMPLATE", "passage: {text}")

        env = EmbeddingConfig.load_from_env()

        assert env["query_prefix"] == "query: "
        assert env["document_template"] == "passage: {text}"

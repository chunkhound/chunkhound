"""Contract test: ``document_template`` shapes the text embedded at index time.

Both indexing routes must hand the embedder the rendered template (with the
chunk's repository-relative path), and must leave the text untouched when no
template is configured:

- the default Rust pipeline, which embeds ``embed_text`` set by the parse step;
- the Python fallback pipeline (``CHUNKHOUND_USE_RUST=0``);
- ``EmbeddingService.generate_missing_embeddings``.
"""

import functools
import tempfile
from pathlib import Path

import pytest

from chunkhound.core.utils.path_utils import get_relative_path_safe
from tests.contracts.mock_embed import MockEmbeddingProvider
from tests.contracts.pipeline_harness import (
    default_rust_config,
    index_with_python,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"
TEMPLATE = "title: {file_path} | text: {text}"


def _index_with_rust_recording(template: str | None) -> tuple[list[str], list[str]]:
    """Index the fixture dir; return (embedded texts, chunk code values)."""
    from chunkhound.pipeline_bridge import parse_batch_callback
    from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
    from tests.contracts.mock_embed import embed_texts

    recorded: list[str] = []

    def recording_embed(texts: list[str]) -> list[list[float]]:
        recorded.extend(texts)
        return embed_texts(texts)

    base = FIXTURE_DIR.resolve()
    file_entries = [
        (str(f), get_relative_path_safe(f, base).as_posix())
        for f in sorted(base.glob("*"))
        if f.is_file()
    ]
    parse = parse_batch_callback
    if template is not None:
        parse = functools.partial(
            parse_batch_callback,
            document_template=template,
            rel_keys=dict(file_entries),
        )

    with tempfile.TemporaryDirectory() as tmp_db:
        db_dir = Path(tmp_db) / "db"
        db_dir.mkdir(parents=True)
        pipeline = IndexingPipeline(
            default_rust_config(
                db_dir,
                skip_embeddings=False,
                embedding_provider="mock-contract",
                embedding_model="mock-v1",
            )
        )
        report = pipeline.run(
            files=file_entries,
            parse_batch_callback=parse,
            embed_batch_callback=recording_embed,
            progress_callback=None,
            incremental=False,
        )
    assert report.chunks_written > 0
    assert not report.errors
    return recorded, [Path(rel).name for _, rel in file_entries]


class TestRustPipeline:
    def test_template_text_and_relative_path_are_embedded(self):
        recorded, filenames = _index_with_rust_recording(TEMPLATE)

        assert recorded
        for text in recorded:
            assert text.startswith("title: ")
            title, _, body = text.removeprefix("title: ").partition(" | text: ")
            assert title in filenames  # fixtures are flat: path == filename
            assert body.strip()

    def test_no_template_embeds_plain_chunk_code(self):
        recorded, _ = _index_with_rust_recording(None)

        assert recorded
        assert not any(text.startswith("title: ") for text in recorded)


class _TemplatedProvider(MockEmbeddingProvider):
    document_template = TEMPLATE

    def __init__(self) -> None:
        self.embedded: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return await super().embed(texts)

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        self.embedded.extend(texts)
        return await super().embed_batch(texts, batch_size)


class TestPythonPipeline:
    @pytest.mark.asyncio
    async def test_template_text_and_relative_path_are_embedded(self):
        provider = _TemplatedProvider()
        filenames = {f.name for f in FIXTURE_DIR.glob("*") if f.is_file()}

        with tempfile.TemporaryDirectory() as tmp_db:
            result = await index_with_python(
                FIXTURE_DIR,
                Path(tmp_db) / "db",
                skip_embeddings=False,
                embedding_provider=provider,
            )

        assert result.embeddings_generated > 0
        assert provider.embedded
        for text in provider.embedded:
            title, _, body = text.removeprefix("title: ").partition(" | text: ")
            assert text.startswith("title: ")
            assert title in filenames
            assert body.strip()


class TestMissingEmbeddings:
    @pytest.mark.asyncio
    async def test_template_text_and_relative_path_are_embedded(self, tmp_path: Path):
        pytest.importorskip("duckdb")
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.embedding_service import EmbeddingService

        db = DuckDBProvider(db_path=tmp_path / "test.duckdb", base_directory=tmp_path)
        db.connect()
        try:
            db.connection.execute(
                "INSERT INTO files (id, path, name, content_hash) "
                "VALUES (1, 'src/network/retry.ts', 'retry.ts', 'aaa')"
            )
            db.connection.execute(
                "INSERT INTO chunks "
                "(id, file_id, code, symbol, chunk_type, start_line, end_line) "
                "VALUES (1, 1, 'function retry() {}', 'retry', 'function', 1, 1)"
            )
            provider = _TemplatedProvider()
            service = EmbeddingService(
                database_provider=db, embedding_provider=provider
            )

            result = await service.generate_missing_embeddings()

            assert result["status"] == "success", result
            assert provider.embedded == [
                "title: src/network/retry.ts | text: function retry() {}"
            ]
        finally:
            db.disconnect(skip_checkpoint=True)

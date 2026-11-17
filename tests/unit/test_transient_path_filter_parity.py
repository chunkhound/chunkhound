"""Contract: transient (diff/web) search filters paths exactly like the database.

``vector_source="both"`` fuses diff results with indexed results, so a
``path_filter`` must select the same files from both sources.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService
from chunkhound.services.transient_search_service import TransientSearchService
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from tests.unit.transient_search_helpers import make_chunk

_MARKER = "path_filter_parity_marker"

_FILES = [
    "src/foo.py",
    "src/foo.py.bak.py",
    "src/module.py",
    "lib/module.py.bak.py",
    "chunkhound/services/search.py",
    "services/top.py",
    "other/servicesx/nope.py",
    ".github/workflows/ci.py",
]

_FILTERS = [
    "src/foo.py",
    "module.py",
    "services",
    "services/",
    "chunkhound/services",
    ".github",
    "/src",
    "src\\foo.py",
]


def _stream(chunks: list) -> object:
    async def stream() -> AsyncIterator[list]:
        await asyncio.sleep(0)
        yield chunks

    return stream


@pytest.mark.asyncio
@pytest.mark.parametrize("path_filter", _FILTERS)
async def test_transient_regex_path_filter_matches_database(
    tmp_path: Path, path_filter: str
) -> None:
    for relative in _FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"def {_MARKER}():\n    return 1\n", encoding="utf-8")

    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()
    try:
        embedding_provider = FakeEmbeddingProvider()
        parser = create_parser_for_language(Language.PYTHON)
        coordinator = IndexingCoordinator(
            db, tmp_path, embedding_provider, {Language.PYTHON: parser}
        )
        for relative in _FILES:
            await coordinator.process_file(tmp_path / relative)
        database = SearchService(db, embedding_provider)

        transient = TransientSearchService(
            database,
            _stream(
                [
                    make_chunk(_MARKER, file_path=relative, code=f"def {_MARKER}()")
                    for relative in _FILES
                ]
            ),
            "diff",
            MagicMock(),
        )

        db_results, _ = await database.search_regex_async(
            _MARKER, page_size=50, path_filter=path_filter
        )
        diff_results, _ = await transient.search_regex_async(
            _MARKER, page_size=50, path_filter=path_filter
        )
    finally:
        db.disconnect()

    db_paths = {str(r["file_path"]).replace("\\", "/") for r in db_results}
    diff_paths = {str(r["file_path"]) for r in diff_results}
    assert db_paths, f"database matched nothing for {path_filter!r}"
    assert diff_paths == db_paths


@pytest.mark.asyncio
async def test_transient_path_filter_rejects_traversal() -> None:
    transient = TransientSearchService(
        MagicMock(), _stream([make_chunk("x")]), "diff", MagicMock()
    )
    with pytest.raises(ValueError, match="forbidden pattern"):
        await transient.search_regex_async("x", path_filter="../etc")

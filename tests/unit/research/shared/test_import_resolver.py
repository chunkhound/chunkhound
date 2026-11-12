"""Import resolution contracts for database-relative file paths.

Deep research reads chunks whose ``file_path`` is project-relative. Dotted
imports must still resolve under the indexed root, and a target outside that
root must be skipped loudly rather than silently dropped.
"""

import io
from pathlib import Path

import pytest
from loguru import logger

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.services.research.shared.import_resolution_helper import (
    resolve_and_fetch_imports,
)
from chunkhound.services.research.shared.import_resolver import ImportResolverService


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Indexed project root holding a package with a relative import."""
    pkg = tmp_path / "root" / "chunkhound"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from .version import __version__\n", encoding="utf-8"
    )
    (pkg / "version.py").write_text('__version__ = "6.1.0"\n', encoding="utf-8")
    return tmp_path / "root"


@pytest.fixture
def resolver() -> ImportResolverService:
    return ImportResolverService(ParserFactory())


class FakeProvider:
    """Serves chunks for the files a test declares as indexed."""

    def __init__(self, base_dir: Path, indexed: dict[str, str]):
        self._base_dir = base_dir
        self._indexed = indexed

    def get_base_directory(self) -> Path:
        return self._base_dir

    def get_file_by_path(self, relative_path: str, as_model: bool = False):
        if relative_path not in self._indexed:
            return None
        return {"id": relative_path}

    def get_chunks_by_file_id(self, file_id: str, as_model: bool = False):
        return [{"file_path": file_id, "content": self._indexed[file_id]}]


class FakeDatabaseServices:
    def __init__(self, base_dir: Path, indexed: dict[str, str]):
        self.provider = FakeProvider(base_dir, indexed)


@pytest.fixture
def db_services(root: Path) -> FakeDatabaseServices:
    return FakeDatabaseServices(
        root, {"chunkhound/version.py": '__version__ = "6.1.0"'}
    )


async def test_dotted_import_from_database_path_resolves_under_root(root, resolver):
    """A project-relative source file resolves its dotted import inside the root."""
    content = (root / "chunkhound" / "__init__.py").read_text(encoding="utf-8")

    resolved = await resolver.resolve_imports("chunkhound/__init__.py", content, root)

    assert resolved == [root / "chunkhound" / "version.py"]


async def test_dotted_import_from_absolute_path_still_resolves(root, resolver):
    """Anchoring must not change behaviour for callers passing absolute paths."""
    source = root / "chunkhound" / "__init__.py"

    resolved = await resolver.resolve_imports(
        str(source), source.read_text(encoding="utf-8"), root
    )

    assert resolved == [source.parent / "version.py"]


async def test_dotted_import_chunks_are_fetched(root, resolver, db_services):
    """Research gains the imported module's chunks, not an empty expansion."""
    chunks = [{"file_path": "chunkhound/__init__.py", "content": "from .version"}]

    result = await resolve_and_fetch_imports(
        chunks, resolver, db_services, ResearchConfig()
    )

    assert [c["file_path"] for c in result] == ["chunkhound/version.py"]
    assert result[0]["rerank_score"] == 0.3


async def test_import_escaping_root_is_skipped_with_warning(
    tmp_path, root, resolver, db_services
):
    """A dotted import reaching outside the root is dropped and named in a warning."""
    (tmp_path / "outside.py").write_text("thing = 1\n", encoding="utf-8")
    (root / "chunkhound" / "escape.py").write_text(
        "from ...outside import thing\n", encoding="utf-8"
    )
    chunks = [{"file_path": "chunkhound/escape.py", "content": "from ...outside"}]

    buf = io.StringIO()
    sink = logger.add(buf, level="WARNING")
    try:
        result = await resolve_and_fetch_imports(
            chunks, resolver, db_services, ResearchConfig(), path_filter="chunkhound/"
        )
    finally:
        logger.remove(sink)

    assert result == []
    logged = buf.getvalue()
    assert len([line for line in logged.splitlines() if "outside.py" in line]) == 1


async def test_import_escaping_root_never_fetches_chunks(tmp_path, root, resolver):
    """Without a path filter the escaping target still cannot reach the database."""
    (tmp_path / "outside.py").write_text("thing = 1\n", encoding="utf-8")
    (root / "chunkhound" / "escape.py").write_text(
        "from ...outside import thing\n", encoding="utf-8"
    )
    db_services = FakeDatabaseServices(root, {"../outside.py": "thing = 1"})
    chunks = [{"file_path": "chunkhound/escape.py", "content": "from ...outside"}]

    buf = io.StringIO()
    sink = logger.add(buf, level="WARNING")
    try:
        result = await resolve_and_fetch_imports(
            chunks, resolver, db_services, ResearchConfig()
        )
    finally:
        logger.remove(sink)

    assert result == []
    assert "outside.py" in buf.getvalue()

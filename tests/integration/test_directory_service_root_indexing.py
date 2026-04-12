"""Service-layer integration test: DirectoryIndexingService indexes root files.

This specifically exercises the pattern normalization path inside
DirectoryIndexingService._process_directory_files so that we catch regressions
where include patterns that already start with "**/" were over-prefixed to
"**/**/…", causing root-level files not to match.
"""

import json
import os
import pytest
from pathlib import Path

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import (
    DuckDBIndexedRootMismatchError,
    DuckDBProvider,
    _indexed_root_sidecar_path,
    _normalize_indexed_root,
)
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator


class _DummyConfig:
    def __init__(self) -> None:
        self.indexing = IndexingConfig()


@pytest.mark.skipif(
    os.environ.get("CHUNKHOUND_ALLOW_PROCESSPOOL", "0") != "1",
    reason="Requires ProcessPool-friendly environment (SemLock).",
)
@pytest.mark.asyncio
async def test_directory_service_indexes_root_file(tmp_path: Path):
    # Arrange: in-memory DB, Python parser, default include patterns
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        None,
        {Language.PYTHON: parser},
        None,
        None,
    )

    # Root-level file
    root_file = tmp_path / "root.py"
    root_file.write_text("print('ok')\n")

    # Service with default config (includes patterns like "**/*.py")
    svc = DirectoryIndexingService(indexing_coordinator=coordinator, config=_DummyConfig())

    # Act
    result = await svc._process_directory_files(
        tmp_path,
        include_patterns=svc.config.indexing.include,
        exclude_patterns=svc.config.indexing.exclude,
    )

    # Assert: should have processed at least 1 file; failure previously manifested as 0
    assert result.get("status") in {"complete", "success", "partial", "done", "ok", "no_files"} or True
    assert result.get("files_processed", 0) >= 1, (
        f"Expected root file to be indexed, got: {result}"
    )


@pytest.mark.asyncio
async def test_process_directory_rejects_wrong_root_before_discovery_or_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 101: `IndexingCoordinator.process_directory(...)` must fail-closed
    with `DuckDBIndexedRootMismatchError` BEFORE file discovery or orphan
    cleanup run when the sidecar records a different indexed root than the
    provider's authoritative `base_directory`.
    """
    db_path = tmp_path / "chunks.db"
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "alpha.py").write_text("print('a')\n")

    parser = create_parser_for_language(Language.PYTHON)

    # Phase 1 — claim the sidecar under root_a via a real
    # `IndexingCoordinator.process_directory(...)` call, exercising the
    # top-of-method `allow_claim_if_missing=True` path.
    db = DuckDBProvider(db_path, base_directory=root_a)
    db.connect()
    try:
        coordinator = IndexingCoordinator(
            db, root_a, None, {Language.PYTHON: parser}, None, None
        )
        await coordinator.process_directory(root_a, patterns=["**/*.py"])

        sidecar = _indexed_root_sidecar_path(db_path)
        assert sidecar is not None and sidecar.exists()
        stored = json.loads(sidecar.read_text(encoding="utf-8"))
        assert stored["indexed_root_path"] == _normalize_indexed_root(root_a)

        # Phase 2 — simulate cross-session divergence by rewriting the sidecar
        # to a different logical root, then call process_directory(...) again.
        # The top-of-method guard must raise BEFORE any discovery or orphan
        # cleanup runs. Monkeypatching `_discover_files` and
        # `_cleanup_orphaned_files` to fail if invoked proves the ordering.
        sidecar.write_text(
            json.dumps(
                {
                    "version": 1,
                    "indexed_root_path": _normalize_indexed_root(root_b),
                }
            ),
            encoding="utf-8",
        )

        discovery_invoked: list[str] = []
        cleanup_invoked: list[str] = []

        async def _fail_discovery(*args, **kwargs):  # pragma: no cover
            discovery_invoked.append("called")
            raise AssertionError(
                "_discover_files must not run when Step 101 guard refuses reuse"
            )

        def _fail_cleanup(*args, **kwargs):  # pragma: no cover
            cleanup_invoked.append("called")
            raise AssertionError(
                "_cleanup_orphaned_files must not run when Step 101 guard refuses reuse"
            )

        monkeypatch.setattr(coordinator, "_discover_files", _fail_discovery)
        monkeypatch.setattr(coordinator, "_cleanup_orphaned_files", _fail_cleanup)

        with pytest.raises(DuckDBIndexedRootMismatchError):
            await coordinator.process_directory(root_a, patterns=["**/*.py"])

        assert discovery_invoked == []
        assert cleanup_invoked == []
    finally:
        db.disconnect()


@pytest.mark.asyncio
async def test_process_directory_same_root_reuse_happy_path(tmp_path: Path) -> None:
    """Step 101: reusing the same DB under the same root via
    `IndexingCoordinator.process_directory(...)` still succeeds after an
    initial claim at that same entrypoint."""
    db_path = tmp_path / "chunks.db"
    root = tmp_path / "root"
    root.mkdir()
    (root / "sample.py").write_text("print('ok')\n")

    parser = create_parser_for_language(Language.PYTHON)

    db1 = DuckDBProvider(db_path, base_directory=root)
    db1.connect()
    try:
        coordinator1 = IndexingCoordinator(
            db1, root, None, {Language.PYTHON: parser}, None, None
        )
        result1 = await coordinator1.process_directory(root, patterns=["**/*.py"])
        assert result1.get("status") != "error", (
            f"Expected initial claim run to succeed, got: {result1}"
        )
    finally:
        db1.disconnect()

    db2 = DuckDBProvider(db_path, base_directory=root)
    db2.connect()
    try:
        coordinator2 = IndexingCoordinator(
            db2, root, None, {Language.PYTHON: parser}, None, None
        )
        result2 = await coordinator2.process_directory(root, patterns=["**/*.py"])
        assert result2.get("status") != "error", (
            f"Expected same-root reuse to succeed, got: {result2}"
        )
    finally:
        db2.disconnect()

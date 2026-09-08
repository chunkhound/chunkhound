"""IndexingCoordinator must not misclassify a genuine Rust-path parse/embed
error as a harmless timeout skip just because its message happens to
*contain* the timeout phrase somewhere in free text.

Regression test for the fix that changed the classification check from a bare
substring search (`"parse timed out after" in err["error"]`) to
`err["error"].startswith("parse timed out after")` — `_split_rust_error()`
already strips the "{path}: " prefix Rust adds, so a genuine timeout's
`error` field is *exactly* `_parse_with_timeout()`'s returned message with
nothing else concatenated onto it; anything else must be a real error.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import chunkhound.services.rust_pipeline_runner as rust_pipeline_runner_module
from chunkhound.core.config.config import Config
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.rust_pipeline_runner import RustPhaseResult
from tests.contracts.pipeline_harness import disconnect_registry_db


def _build_config(root: Path, db_dir: Path) -> Config:
    return Config(
        target_dir=root,
        database={"provider": "duckdb", "path": str(db_dir)},
        indexing={"include": ["**/*"], "exclude": []},
        embeddings_disabled=True,
    )


def _fake_rust_result(error_message: str) -> RustPhaseResult:
    return RustPhaseResult(
        total_files=1,
        total_chunks=0,
        embeddings_generated=0,
        errors=[{"file": "bad.py", "error": error_message}],
        files_skipped_unchanged=0,
        skipped_paths=[],
        diff_elapsed=0.0,
        compact_ran=False,
        compact_size_before=None,
        compact_size_after=None,
        compact_reduction_pct=None,
    )


def _run_with_fake_rust_result(tmp_path, monkeypatch, error_message: str):
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    root = tmp_path / "repo"
    root.mkdir()
    (root / "real_code.py").write_text("def real_function():\n    return 42\n")

    async def fake_run_rust_indexing_phase(**kwargs):
        return _fake_rust_result(error_message)

    monkeypatch.setattr(
        rust_pipeline_runner_module,
        "run_rust_indexing_phase",
        fake_run_rust_indexing_phase,
    )

    db_dir = tmp_path / "db"
    config = _build_config(root, db_dir)
    configure_registry(config)
    coordinator = create_indexing_coordinator()
    service = DirectoryIndexingService(indexing_coordinator=coordinator, config=config)
    try:
        return asyncio.run(service.process_directory(root, no_embeddings=True))
    finally:
        disconnect_registry_db()


def test_error_message_only_mentioning_timeout_is_not_misclassified(
    tmp_path, monkeypatch
):
    """A genuine error whose free-text message happens to contain the
    timeout phrase (but doesn't consist of it) must stay a real error."""
    stats = _run_with_fake_rust_result(
        tmp_path,
        monkeypatch,
        "provider request failed, note: server said parse timed out after retry",
    )

    assert stats.skipped_due_to_timeout == [], (
        "a real error must not be reclassified as a timeout skip just "
        "because its message contains the timeout phrase somewhere in it"
    )
    assert stats.files_errors == 1


def test_genuine_parse_timeout_is_still_classified_as_skipped(tmp_path, monkeypatch):
    """The intended case must still work: an error message that IS the
    timeout message (no other prefix) is a timeout skip, not a hard error."""
    stats = _run_with_fake_rust_result(
        tmp_path, monkeypatch, "parse timed out after 5.0s"
    )

    assert stats.skipped_due_to_timeout == ["bad.py"]
    assert stats.files_errors == 0

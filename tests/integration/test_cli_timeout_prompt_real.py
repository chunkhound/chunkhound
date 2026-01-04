import json
from argparse import Namespace
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")


@pytest.mark.asyncio
async def test_cli_timeout_prompt_real(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end: low timeout forces a skip, and the prompt adds it to exclude."""
    from chunkhound.api.cli.commands import run as run_mod
    from chunkhound.core.config.config import Config as CoreConfig
    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.core.types.common import Language
    from chunkhound.services.batch_processor import ParsedFileResult
    from chunkhound.services.indexing_coordinator import IndexingCoordinator

    # Create a relatively larger .py file to exercise timeout reliably
    f = tmp_path / "slow.py"
    with f.open("w") as fh:
        fh.write("# slow file\n")
        # ~2MB of content
        fh.write("x" * (2 * 1024 * 1024))

    # Ensure prompt is shown and accepted
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.delenv("CHUNKHOUND_NO_PROMPTS", raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "y")

    # Force timeout path, apply to all sizes, and reindex regardless of unchanged
    monkeypatch.setenv("CHUNKHOUND_INDEXING__PER_FILE_TIMEOUT_SECONDS", "0.000001")
    monkeypatch.setenv("CHUNKHOUND_INDEXING__PER_FILE_TIMEOUT_MIN_SIZE_KB", "0")
    monkeypatch.setenv("CHUNKHOUND_INDEXING__FORCE_REINDEX", "true")

    # Deterministically simulate a timeout at the coordinator seam to avoid ProcessPool in CI

    async def _stub_batches(
        self, files, config_file_size_threshold_kb=20, parse_task=None, on_batch=None
    ):
        results = []
        for p in files:
            st = p.stat()
            results.append(
                ParsedFileResult(
                    file_path=p,
                    chunks=[],
                    language=Language.PYTHON,
                    file_size=st.st_size,
                    file_mtime=st.st_mtime,
                    status="skipped",
                    error="timeout",
                )
            )
        if on_batch:
            await on_batch(results)
        return results

    monkeypatch.setattr(IndexingCoordinator, "_process_files_in_batches", _stub_batches)

    # Minimal .chunkhound.json in target dir
    config_path = tmp_path / ".chunkhound.json"
    config_path.write_text("{}\n")

    # Build args and config
    args = Namespace(
        path=tmp_path,
        verbose=False,
        no_embeddings=True,
        include=["*.py"],
        exclude=None,
        force_reindex=True,
        db=None,
    )

    db_cfg = DatabaseConfig(provider="duckdb", path=tmp_path / ".chunkhound" / "db")
    cfg = CoreConfig(target_dir=tmp_path)
    cfg.database = db_cfg
    # Ensure discovery includes our file pattern via config
    cfg.indexing.include = ["*.py"]
    cfg.indexing.force_reindex = True

    await run_mod.run_command(args, cfg)

    # Verify exclusion written
    data = json.loads(config_path.read_text())
    assert "indexing" in data and "exclude" in data["indexing"]
    # Path is relative in prompt path; we expect just the filename for a file in project root
    assert "slow.py" in data["indexing"]["exclude"]

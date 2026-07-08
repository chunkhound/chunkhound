"""Tests for index_unknown_files feature (issue #277).

Before-fix tests: regression guards that pass today and must stay passing.
After-fix tests: written first, fail until implementation is done.
"""

import argparse
from pathlib import Path

import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.services.batch_processor import process_file_batch


def _cfg(**overrides) -> dict:
    """Minimal config_dict for process_file_batch with timeout disabled."""
    base = {"per_file_timeout_seconds": 0.0}
    base.update(overrides)
    return base


def _write_text_unknown(tmp_path: Path, name: str = "myfile.xyzunknown") -> Path:
    """Create a text file with an extension unknown to ChunkHound."""
    path = tmp_path / name
    # 20 lines so TextMapping has enough content to produce chunks
    path.write_text("\n".join(f"line {i}: some content here" for i in range(20)) + "\n")
    return path


def _write_binary_unknown(tmp_path: Path) -> Path:
    path = tmp_path / "model.xyzunknown"
    path.write_bytes(b"\x00\x01\x02" * 1000)
    return path


# ════════════════════════════════════════════════════════════════════════════
# BEFORE FIX — regression guard
# These pass right now. They must continue to pass after implementation.
# ════════════════════════════════════════════════════════════════════════════


class TestDefaultBehaviourPreserved:
    """Existing behaviour: unknown files skipped, no wildcard injected by default."""

    def test_unknown_file_skipped_by_default(self, tmp_path: Path):
        unknown = _write_text_unknown(tmp_path)
        results = process_file_batch([unknown], _cfg())
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert results[0].error == "Unknown file type"

    def test_wildcard_not_in_default_include(self):
        config = IndexingConfig()
        assert "**/*" not in config.include

    def test_known_python_file_not_skipped_as_unknown(self, tmp_path: Path):
        pyfile = tmp_path / "hello.py"
        pyfile.write_text("x = 1\n")
        results = process_file_batch([pyfile], _cfg())
        assert len(results) == 1
        assert not (
            results[0].status == "skipped" and results[0].error == "Unknown file type"
        )


# ════════════════════════════════════════════════════════════════════════════
# AFTER FIX — new behaviour
# Written first. Fail until implementation lands.
# ════════════════════════════════════════════════════════════════════════════


class TestIndexUnknownFilesFlag:
    """Flag-on: text unknown files indexed; binary files still skipped."""

    def test_unknown_text_file_indexed_when_flag_on(self, tmp_path: Path):
        unknown = _write_text_unknown(tmp_path)
        results = process_file_batch([unknown], _cfg(index_unknown_files=True))
        assert len(results) == 1
        assert results[0].status == "success"
        assert len(results[0].chunks) > 0

    def test_binary_unknown_file_skipped_when_flag_on(self, tmp_path: Path):
        binfile = _write_binary_unknown(tmp_path)
        results = process_file_batch([binfile], _cfg(index_unknown_files=True))
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert results[0].error == "binary_file"

    def test_wildcard_injected_when_flag_on(self):
        config = IndexingConfig(index_unknown_files=True)
        assert "**/*" in config.include

    def test_wildcard_appended_to_custom_include(self):
        config = IndexingConfig(index_unknown_files=True, include=["**/*.py"])
        assert "**/*" in config.include
        assert "**/*.py" in config.include

    def test_cli_flag_sets_index_unknown_files(self):
        parser = argparse.ArgumentParser()
        IndexingConfig.add_cli_arguments(parser)
        args = parser.parse_args(["--index-unknown-files"])
        assert args.index_unknown_files is True

    def test_env_var_sets_index_unknown_files(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHUNKHOUND_INDEXING__INDEX_UNKNOWN_FILES", "true")
        overrides = IndexingConfig.load_from_env()
        assert overrides.get("index_unknown_files") is True

    def test_json_config_sets_index_unknown_files(self):
        config = IndexingConfig.model_validate({"index_unknown_files": True})
        assert config.index_unknown_files is True

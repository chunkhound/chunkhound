from pathlib import Path

from chunkhound.core.config.config import Config


def test_default_snapshot_dir_is_derived_and_not_explicit(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("CHUNKHOUND_RESEARCH_CHUNK_SYSTEMS_SNAPSHOT_DIR", raising=False)
    monkeypatch.delenv("CHUNKHOUND_CONFIG_FILE", raising=False)

    db_dir = tmp_path / "db"
    cfg = Config(target_dir=tmp_path, database={"path": db_dir})

    assert cfg.research.chunk_systems_snapshot_dir == cfg.database.path / "chunk_systems_snapshot"
    assert "chunk_systems_snapshot_dir" not in cfg.research.model_fields_set


def test_explicit_snapshot_dir_is_preserved(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CHUNKHOUND_RESEARCH_CHUNK_SYSTEMS_SNAPSHOT_DIR", raising=False)
    monkeypatch.delenv("CHUNKHOUND_CONFIG_FILE", raising=False)

    db_dir = tmp_path / "db"
    custom = tmp_path / "custom"
    cfg = Config(
        target_dir=tmp_path,
        database={"path": db_dir},
        research={"chunk_systems_snapshot_dir": custom},
    )

    assert cfg.research.chunk_systems_snapshot_dir == custom
    assert "chunk_systems_snapshot_dir" in cfg.research.model_fields_set


def test_memory_db_does_not_set_default_snapshot_dir(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CHUNKHOUND_RESEARCH_CHUNK_SYSTEMS_SNAPSHOT_DIR", raising=False)
    monkeypatch.delenv("CHUNKHOUND_CONFIG_FILE", raising=False)

    cfg = Config(target_dir=tmp_path, database={"path": ":memory:"})

    assert str(cfg.database.path) == ":memory:"
    assert cfg.research.chunk_systems_snapshot_dir is None


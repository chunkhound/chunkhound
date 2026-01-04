from __future__ import annotations

import asyncio
from pathlib import Path

from chunkhound.core.config.config import Config
from chunkhound.services.realtime_indexing_service import SimpleEventHandler


def _handler(tmp_path: Path) -> SimpleEventHandler:
    cfg = Config(target_dir=tmp_path)
    # Enable .gitignore mode explicitly
    cfg.indexing.exclude_sentinel = ".gitignore"
    q: asyncio.Queue = asyncio.Queue()
    return SimpleEventHandler(q, config=cfg)


def test_realtime_should_index_respects_repo_boundary(tmp_path: Path) -> None:
    root = tmp_path
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("*.txt\n")
    (root / "subrepo").mkdir()
    (root / "subrepo" / ".git").mkdir()
    (root / "subrepo" / "keep.txt").write_text("x")

    h = _handler(root)

    # Parent ignored
    assert h._should_index(root / "a.txt") is False
    # Inside subrepo should be allowed
    assert h._should_index(root / "subrepo" / "keep.txt") is True


def test_realtime_should_index_prunes_excluded_subtrees(tmp_path: Path) -> None:
    root = tmp_path
    (root / "node_modules").mkdir()
    (root / "node_modules" / "mod.txt").write_text("x")
    (root / "README.md").write_text("x")

    h = _handler(root)
    # node_modules content should be excluded by default config excludes
    assert h._should_index(root / "node_modules" / "mod.txt") is False
    # README should be checked against include patterns (md is supported)
    assert h._should_index(root / "README.md") is True

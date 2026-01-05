from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_discover_files_repo_boundary(tmp_path: Path) -> None:
    root = tmp_path
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("*.txt\n")
    (root / ".chunkhound.json").write_text('{"indexing": {"exclude": ".gitignore"}}\n')
    (root / "subrepo").mkdir()
    (root / "subrepo" / ".git").mkdir()
    (root / "subrepo" / "keep.txt").write_text("x")

    # Build a minimal Config and coordinator
    from chunkhound.core.config.config import Config

    cfg = Config(target_dir=root)

    from chunkhound.registry import configure_registry, create_indexing_coordinator

    # Ensure DB dir exists
    db_dir = cfg.database.path.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    configure_registry(cfg)
    coord = create_indexing_coordinator()

    files = await coord._discover_files(
        root, patterns=["**/*.txt"], exclude_patterns=[], parallel_discovery=False
    )  # type: ignore[attr-defined]
    rels = sorted([p.resolve().relative_to(root).as_posix() for p in files])
    assert "subrepo/keep.txt" in rels

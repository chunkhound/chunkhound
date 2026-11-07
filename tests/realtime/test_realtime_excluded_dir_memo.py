"""A burst of events under an excluded directory matches each directory once.

Every file event under node_modules/ used to run the full ignore match for the
file itself, repeating the match that already excludes its directory: about
0.43 ms per event, 20 s for one junction to a real node_modules tree.
"""

from __future__ import annotations

from pathlib import Path

from chunkhound.core.config.config import Config
from chunkhound.services.realtime_path_filter import RealtimePathFilter


class _CountingEngine:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0

    def matches(self, path: Path, is_dir: bool):
        self.calls += 1
        return self._inner.matches(path, is_dir)


def test_files_under_an_excluded_directory_skip_per_file_matching(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    cfg = Config(
        database={"provider": "duckdb", "path": str(tmp_path / "db.duckdb")},
        indexing={"include": ["**/*.js"], "exclude": ["**/node_modules/"]},
        target_dir=root,
    )
    path_filter = RealtimePathFilter(config=cfg, root_path=root)
    assert path_filter._ensure_engine()
    engine = _CountingEngine(path_filter._engine)
    path_filter._engine = engine

    burst = [
        root / "node_modules" / f"pkg{i % 5}" / "lib" / f"f{i}.js" for i in range(200)
    ]
    assert not any(path_filter.should_index(p) for p in burst)
    # node_modules is decided once; nothing below it is matched at all.
    assert engine.calls <= 2, engine.calls

    assert path_filter.should_index(root / "src" / "app.js")

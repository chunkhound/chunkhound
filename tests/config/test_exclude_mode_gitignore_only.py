from __future__ import annotations

from chunkhound.core.config.config import Config


def test_exclude_list_gitignore_only_mode() -> None:
    cfg = Config(
        **{"indexing": {"exclude": ["**/dist/**"], "exclude_mode": "gitignore_only"}}
    )
    sources = cfg.indexing.resolve_ignore_sources()  # type: ignore[attr-defined]
    assert sources == ["gitignore"]

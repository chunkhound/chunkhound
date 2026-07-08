from __future__ import annotations

import json
from pathlib import Path

from chunkhound.core.config.config import Config


def test_local_config_exclude_list_enables_overlay(tmp_path: Path) -> None:
    cfg_path = tmp_path / ".chunkhound.json"
    cfg_path.write_text(
        json.dumps(
            {
                "indexing": {
                    "exclude": ["**/foo/bar/baz.txt"],
                }
            }
        )
    )
    cfg = Config(target_dir=tmp_path)
    # When exclude is a list in local config, we default to combined overlay
    assert cfg.indexing.resolve_ignore_sources() == ["gitignore", "config"]

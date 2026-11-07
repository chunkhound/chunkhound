"""Watchman junction discovery skips junctions under excluded directories.

Every junction under the watched tree whose target lies outside it became its
own Watchman watch root and subscription, including ones under node_modules/
(npm links a local file: package as a junction). Every change in such a
package was sent to ChunkHound and then dropped by the event filter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime.service import RealtimeIndexingService


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junctions")
def test_junction_under_excluded_directory_gets_no_watch_scope(
    tmp_path: Path,
) -> None:
    import _winapi

    root = tmp_path / "repo"
    (root / "node_modules" / "@strasz").mkdir(parents=True)
    package = tmp_path / "kit"
    shared = tmp_path / "shared"
    package.mkdir()
    shared.mkdir()
    excluded_link = root / "node_modules" / "@strasz" / "kit"
    included_link = root / "shared"
    _winapi.CreateJunction(str(package), str(excluded_link))
    _winapi.CreateJunction(str(shared), str(included_link))

    db_path = tmp_path / "db.duckdb"
    config = Config(
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={"exclude": ["**/node_modules/**"], "realtime_backend": "watchman"},
        target_dir=root,
    )
    services = create_services(db_path, config)
    try:
        service = RealtimeIndexingService(services, config)
        scopes = service._discover_nested_windows_junction_scopes(root)
        assert [scope.requested_path.name for scope in scopes] == ["shared"]
    finally:
        os.rmdir(excluded_link)  # removes the junction, never its target
        os.rmdir(included_link)

"""A database held open by another process is in use, not WAL-corrupt.

On Windows, ATTACHing a file another process holds fails with "The process
cannot access the file because it is being used by another process. File is
already open in ...". Treating that as corruption copied and removed the live
WAL of a running daemon.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chunkhound.providers.database.duckdb import connection_manager
from chunkhound.providers.database.duckdb.connection_manager import (
    DuckDBConnectionManager,
)


def test_wal_of_db_held_by_another_process_is_left_in_place(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "chunks.db"
    db_path.write_bytes(b"db")
    wal_path = tmp_path / "chunks.db.wal"
    wal_path.write_bytes(b"live wal")

    held = duckdb.IOException(
        f'IO Error: Cannot open file "{db_path}": The process cannot access '
        "the file because it is being used by another process.\n\n"
        "File is already open in \nC:\\Python\\python.exe (PID 29808)"
    )

    def execute(sql: str, *args: object) -> MagicMock:
        if sql.startswith("ATTACH"):
            raise held
        return MagicMock()

    fake_conn = MagicMock()
    fake_conn.execute.side_effect = execute

    with patch.object(connection_manager.duckdb, "connect", return_value=fake_conn):
        DuckDBConnectionManager(db_path, None)._preemptive_wal_cleanup()

    assert wal_path.read_bytes() == b"live wal"
    assert not (tmp_path / "chunks.db.wal.corrupt").exists()

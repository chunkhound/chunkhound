"""WAL validation must not leave secondary indexes missing replayed rows.

duckdb/duckdb#26106: after a WAL replay, the implicit checkpoint (close /
DETACH) persists secondary ART indexes without the replayed rows; deleting
those rows later fails "Failed to delete all rows from index".
"""

import subprocess
import sys
from pathlib import Path

import pytest

from chunkhound.providers.database.duckdb.connection_manager import (
    DuckDBConnectionManager,
)

_WRITE_AND_DIE = """
import os, sys, duckdb
con = duckdb.connect(sys.argv[1])
con.execute("PRAGMA wal_autocheckpoint = '1GB'")
con.execute("INSERT INTO t SELECT range, 7 FROM range(300)")
os._exit(0)  # no close, no checkpoint: the rows live only in the WAL
"""


def test_wal_validation_keeps_replayed_rows_in_secondary_indexes(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "chunks.db"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE t (id INTEGER, k INTEGER)")
    con.execute("CREATE INDEX idx_t_k ON t(k)")
    con.close()

    subprocess.run([sys.executable, "-c", _WRITE_AND_DIE, str(db_path)], check=True)
    assert db_path.with_suffix(".db.wal").exists()

    DuckDBConnectionManager(db_path, None)._preemptive_wal_cleanup()

    con = duckdb.connect(str(db_path))
    try:
        con.execute("DELETE FROM t WHERE k = 7")
        assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 0
    finally:
        con.close()

"""Tests for DuckDBConnectionManager compaction crash recovery.

Verifies that connection startup correctly handles leftover artifacts
from interrupted compaction operations: lock files, stale temp databases,
and mid-swap states.
"""

import os
import shutil
import time
from pathlib import Path

import duckdb
import pytest
from loguru import logger

from chunkhound.core.exceptions import CompactionError
from chunkhound.providers.database.duckdb.connection_manager import (
    DuckDBConnectionManager,
    _REQUIRED_TABLES,
    get_compaction_lock_path,
)


def _create_valid_duckdb(path: Path) -> None:
    """Create a minimal valid DuckDB database with required tables.

    Uses ``_REQUIRED_TABLES`` from connection_manager so the integrity
    probe (``_probe_db_valid``) recognises the file as a valid ChunkHound DB.
    """
    conn = duckdb.connect(str(path))
    for table in _REQUIRED_TABLES:
        conn.execute(f"CREATE TABLE {table} (id INTEGER)")
    conn.close()


class TestCrashMidSwap:
    """Test recovery when process crashed between rename operations."""

    def test_restores_from_old_when_original_missing(self, tmp_path: Path):
        """If db_path is gone but .duckdb.old exists, restore the old file."""
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")

        # Simulate crash: original renamed to .old, new never moved into place
        _create_valid_duckdb(old_path)
        assert not db_path.exists()
        assert old_path.exists()

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists(), "Original should be restored from .old"
            assert not old_path.exists(), ".old should be cleaned up"
        finally:
            mgr.disconnect()

    def test_cleans_stale_old_when_both_exist(self, tmp_path: Path):
        """If both db_path and .duckdb.old exist and db_path is valid, remove the .old artifact."""
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")

        _create_valid_duckdb(db_path)
        _create_valid_duckdb(old_path)

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists()
            assert not old_path.exists(), "Stale .old should be cleaned up"
        finally:
            mgr.disconnect()

    def test_restores_backup_when_db_corrupted(self, tmp_path: Path):
        """If db_path is corrupt but .duckdb.old is valid, restore from backup.

        On Windows the two-step os.replace() swap is not atomic — a crash
        between the two renames can leave a truncated/corrupt db_path
        alongside the valid pre-compaction backup (.duckdb.old).  Recovery
        must detect the corruption and prefer the backup.
        """
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")

        # Create a valid backup and a corrupted primary
        _create_valid_duckdb(old_path)
        db_path.write_bytes(b"not a valid duckdb file")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists(), "db_path should be restored from backup"
            assert not old_path.exists(), ".old should be cleaned up after restore"
            # Verify the restored file is actually usable
            result = mgr.connection.execute(
                "SELECT count(*) FROM files"
            ).fetchone()
            assert result == (0,), "Restored DB should have files table"
        finally:
            mgr.disconnect()

    def test_connect_fails_when_both_db_and_backup_corrupt(self, tmp_path: Path):
        """If both db_path and .duckdb.old are corrupt, connect raises."""
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")

        db_path.write_bytes(b"corrupt primary")
        old_path.write_bytes(b"corrupt backup")

        mgr = DuckDBConnectionManager(db_path)
        with pytest.raises(Exception):
            mgr.connect()


class TestStaleArtifacts:
    """Test cleanup of leftover compaction artifacts."""

    def test_cleans_compact_db_artifact(self, tmp_path: Path):
        """Leftover .compact.duckdb is cleaned up on connect."""
        db_path = tmp_path / "chunks.duckdb"
        compact_path = db_path.with_suffix(".compact.duckdb")

        _create_valid_duckdb(db_path)
        compact_path.write_bytes(b"stale")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert not compact_path.exists(), ".compact.duckdb should be cleaned up"
        finally:
            mgr.disconnect()

    def test_cleans_export_directory(self, tmp_path: Path):
        """Leftover export directory is cleaned up on connect."""
        db_path = tmp_path / "chunks.duckdb"
        export_dir = tmp_path / ".chunkhound_compaction_export"

        _create_valid_duckdb(db_path)
        export_dir.mkdir()
        (export_dir / "leftover.parquet").write_bytes(b"data")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert not export_dir.exists(), "Export dir should be cleaned up"
        finally:
            mgr.disconnect()


class TestLockFileRecovery:
    """Test PID-aware lock file handling on startup."""

    def test_removes_stale_lock_dead_pid(self, tmp_path: Path):
        """Lock file with dead PID is removed on connect."""
        db_path = tmp_path / "chunks.duckdb"
        lock_path = get_compaction_lock_path(db_path)

        _create_valid_duckdb(db_path)
        # Use a PID that almost certainly doesn't exist (new format with timestamp)
        lock_path.write_text(f"999999999:{time.time():.0f}")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert not lock_path.exists(), "Stale lock (dead PID) should be removed"
        finally:
            mgr.disconnect()

    def test_preserves_lock_live_pid(self, tmp_path: Path):
        """Lock file with live PID is NOT removed on connect."""
        db_path = tmp_path / "chunks.duckdb"
        lock_path = get_compaction_lock_path(db_path)

        _create_valid_duckdb(db_path)
        # Write current process PID with recent timestamp (guaranteed alive)
        lock_path.write_text(f"{os.getpid()}:{time.time():.0f}")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert lock_path.exists(), "Lock held by live process should be preserved"
        finally:
            mgr.disconnect()
            # Clean up lock so it doesn't interfere with other tests
            lock_path.unlink(missing_ok=True)

    def test_preserves_lock_old_timestamp_live_pid(self, tmp_path: Path):
        """Lock with live PID is preserved even with old timestamp (long compaction).

        Invariant: a live process's compaction lock must NEVER be removed based
        on age alone.  Compaction on a large database may legitimately run for
        hours.  Removing the lock while the process is alive would allow a
        second compaction to start concurrently, risking data corruption.

        The timestamp is anchored to boot_time (not wall-clock) so the test
        is valid in CI containers where uptime can be shorter than the
        simulated lock age.
        """
        import psutil

        db_path = tmp_path / "chunks.duckdb"
        lock_path = get_compaction_lock_path(db_path)

        _create_valid_duckdb(db_path)
        # Shortly after boot — old enough to be non-trivial, but guaranteed
        # post-boot so the "PID reused after reboot" path is not triggered.
        old_timestamp = psutil.boot_time() + 60
        lock_path.write_text(f"{os.getpid()}:{old_timestamp:.0f}")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert lock_path.exists(), (
                "Lock with live PID should be preserved even with old timestamp"
            )
        finally:
            mgr.disconnect()
            lock_path.unlink(missing_ok=True)

    def test_removes_lock_from_before_reboot(self, tmp_path: Path):
        """Lock created before last boot is removed even if PID is alive (reuse)."""
        import psutil

        db_path = tmp_path / "chunks.duckdb"
        lock_path = get_compaction_lock_path(db_path)

        _create_valid_duckdb(db_path)
        # Timestamp before boot = guaranteed PID reuse
        pre_boot = psutil.boot_time() - 3600
        lock_path.write_text(f"{os.getpid()}:{pre_boot:.0f}")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert not lock_path.exists(), (
                "Lock from before reboot should be removed (PID reuse)"
            )
        finally:
            mgr.disconnect()

    def test_preserves_lock_legacy_pid_only(self, tmp_path: Path):
        """Legacy lock file with only PID (no timestamp) and live PID is preserved."""
        db_path = tmp_path / "chunks.duckdb"
        lock_path = get_compaction_lock_path(db_path)

        _create_valid_duckdb(db_path)
        # Legacy format: PID only, no timestamp
        lock_path.write_text(str(os.getpid()))

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert lock_path.exists(), (
                "Legacy lock with live PID should be preserved"
            )
        finally:
            mgr.disconnect()
            lock_path.unlink(missing_ok=True)

    def test_skips_recovery_when_foreign_lock_alive(self, tmp_path: Path):
        """When a live process holds the compaction lock, recovery and cleanup are skipped."""
        db_path = tmp_path / "chunks.duckdb"
        lock_path = get_compaction_lock_path(db_path)
        intent_path = Path(str(db_path) + ".swap_intent")
        compact_path = db_path.with_suffix(".compact.duckdb")
        export_dir = db_path.parent / ".chunkhound_compaction_export"

        _create_valid_duckdb(db_path)
        # Write a lock held by current process (guaranteed alive)
        lock_path.write_text(f"{os.getpid()}:{time.time():.0f}")
        # Create artifacts that a foreign compaction would be using
        intent_path.write_text("phase2")
        compact_path.write_bytes(b"compacting")
        export_dir.mkdir()
        (export_dir / "data.parquet").write_bytes(b"export data")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert lock_path.exists(), "Lock should be preserved"
            assert intent_path.exists(), "Intent file should NOT be cleaned up"
            assert compact_path.exists(), "Compact DB should NOT be deleted"
            assert export_dir.exists(), "Export dir should NOT be deleted"
        finally:
            mgr.disconnect()
            lock_path.unlink(missing_ok=True)
            intent_path.unlink(missing_ok=True)
            compact_path.unlink(missing_ok=True)
            shutil.rmtree(export_dir, ignore_errors=True)

    def test_removes_empty_legacy_lock(self, tmp_path: Path):
        """Empty lock file (legacy format) is treated as stale."""
        db_path = tmp_path / "chunks.duckdb"
        lock_path = get_compaction_lock_path(db_path)

        _create_valid_duckdb(db_path)
        lock_path.write_text("")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert not lock_path.exists(), "Empty (legacy) lock should be removed"
        finally:
            mgr.disconnect()


class TestProbeDbValid:
    """Test integrity probe rejects databases missing required tables."""

    def test_probe_rejects_missing_tables(self, tmp_path: Path):
        """Database without 'files' and 'chunks' tables fails probe."""
        db_path = tmp_path / "incomplete.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE other_table (id INTEGER)")
        conn.close()

        mgr = DuckDBConnectionManager(db_path)
        assert not mgr._probe_db_valid(db_path)

    def test_probe_accepts_valid_schema(self, tmp_path: Path):
        """Database with 'files' and 'chunks' tables passes probe."""
        db_path = tmp_path / "valid.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE files (id INTEGER)")
        conn.execute("CREATE TABLE chunks (id INTEGER)")
        conn.close()

        mgr = DuckDBConnectionManager(db_path)
        assert mgr._probe_db_valid(db_path)

    def test_probe_rejects_corrupt_file(self, tmp_path: Path):
        """Corrupt file fails probe gracefully."""
        db_path = tmp_path / "corrupt.duckdb"
        db_path.write_bytes(b"not a database")

        mgr = DuckDBConnectionManager(db_path)
        assert not mgr._probe_db_valid(db_path)


class TestIntentBasedRecovery:
    """Test intent-file-based crash recovery."""

    def test_phase1_restores_old_discards_compact(self, tmp_path: Path):
        """phase1 crash: db missing, old exists, compact exists -> restore old, discard compact."""
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")
        compact_path = db_path.with_suffix(".compact.duckdb")
        intent_path = Path(str(db_path) + ".swap_intent")

        _create_valid_duckdb(old_path)
        compact_path.write_bytes(b"incomplete compact")
        intent_path.write_text("phase1")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists(), "Should be restored from old"
            assert not old_path.exists(), "old should be consumed"
            assert not compact_path.exists(), "compact should be discarded"
            assert not intent_path.exists(), "intent should be cleaned up"
        finally:
            mgr.disconnect()

    def test_phase2_completes_swap_with_compact_db(self, tmp_path: Path):
        """phase2 crash: db missing, compact exists -> complete swap with compact (THE key fix)."""
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")
        compact_path = db_path.with_suffix(".compact.duckdb")
        intent_path = Path(str(db_path) + ".swap_intent")

        _create_valid_duckdb(old_path)
        _create_valid_duckdb(compact_path)
        intent_path.write_text("phase2")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists(), "Should be completed from compact"
            assert not compact_path.exists(), "compact should be consumed/cleaned"
            assert not old_path.exists(), "old should be cleaned after successful swap"
            assert not intent_path.exists(), "intent should be cleaned up"
        finally:
            mgr.disconnect()

    def test_phase2_falls_back_to_old_when_compact_missing(self, tmp_path: Path):
        """phase2 crash: db missing, compact missing, old exists -> restore old."""
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")
        intent_path = Path(str(db_path) + ".swap_intent")

        _create_valid_duckdb(old_path)
        intent_path.write_text("phase2")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists(), "Should be restored from old"
            assert not old_path.exists(), "old should be consumed"
            assert not intent_path.exists(), "intent should be cleaned up"
        finally:
            mgr.disconnect()

    def test_intent_cleaned_after_recovery(self, tmp_path: Path):
        """Intent file is removed even when no crash artifacts exist."""
        db_path = tmp_path / "chunks.duckdb"
        intent_path = Path(str(db_path) + ".swap_intent")

        _create_valid_duckdb(db_path)
        intent_path.write_text("phase2")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists()
            assert not intent_path.exists(), "intent should always be cleaned up"
        finally:
            mgr.disconnect()

    def test_phase2_falls_back_when_compact_corrupt(self, tmp_path: Path):
        """phase2 crash: compact_db corrupt + old_db valid -> restore old_db."""
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")
        compact_path = db_path.with_suffix(".compact.duckdb")
        intent_path = Path(str(db_path) + ".swap_intent")

        _create_valid_duckdb(old_path)
        compact_path.write_bytes(b"corrupt compact db")
        intent_path.write_text("phase2")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists(), "Should be restored from old_db"
            assert not old_path.exists(), "old should be consumed"
            assert not compact_path.exists(), "corrupt compact should be cleaned up"
            assert not intent_path.exists(), "intent should be cleaned up"
            # Verify the restored file is usable
            result = mgr.connection.execute(
                "SELECT count(*) FROM files"
            ).fetchone()
            assert result == (0,), "Restored DB should have files table"
        finally:
            mgr.disconnect()

    def test_phase2_completes_swap_without_backup(self, tmp_path: Path):
        """phase2 crash: db missing, compact valid, no old_db -> install compact."""
        db_path = tmp_path / "chunks.duckdb"
        compact_path = db_path.with_suffix(".compact.duckdb")
        intent_path = Path(str(db_path) + ".swap_intent")

        _create_valid_duckdb(compact_path)
        intent_path.write_text("phase2")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists(), "Should be installed from compact"
            assert not compact_path.exists(), "compact should be consumed"
            assert not intent_path.exists(), "intent should be cleaned up"
            result = mgr.connection.execute(
                "SELECT count(*) FROM files"
            ).fetchone()
            assert result == (0,), "Installed DB should have files table"
        finally:
            mgr.disconnect()

    def test_phase2_raises_when_both_unavailable(self, tmp_path: Path):
        """phase2 crash: compact_db corrupt + no old_db -> raise CompactionError."""
        db_path = tmp_path / "chunks.duckdb"
        compact_path = db_path.with_suffix(".compact.duckdb")
        intent_path = Path(str(db_path) + ".swap_intent")

        compact_path.write_bytes(b"corrupt compact db")
        intent_path.write_text("phase2")

        mgr = DuckDBConnectionManager(db_path)
        with pytest.raises(CompactionError, match="Unrecoverable"):
            mgr.connect()

    def test_pre_swap_recovery_discards_compact(self, tmp_path: Path):
        """pre_swap crash: db_path exists, compact/old artifacts -> discard both."""
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")
        compact_path = db_path.with_suffix(".compact.duckdb")
        intent_path = Path(str(db_path) + ".swap_intent")

        _create_valid_duckdb(db_path)
        old_path.write_bytes(b"stale old db")
        compact_path.write_bytes(b"incomplete compact")
        intent_path.write_text("pre_swap")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists(), "db_path should remain intact"
            assert not compact_path.exists(), "compact should be discarded"
            assert not old_path.exists(), "stale old_db should be cleaned up"
            assert not intent_path.exists(), "intent should be cleaned up"
        finally:
            mgr.disconnect()

    def test_unknown_phase_falls_back_to_legacy(self, tmp_path: Path):
        """Unknown intent content triggers legacy recovery path."""
        db_path = tmp_path / "chunks.duckdb"
        old_path = db_path.with_suffix(".duckdb.old")
        intent_path = Path(str(db_path) + ".swap_intent")

        # Simulate unknown phase with db missing and old present
        _create_valid_duckdb(old_path)
        intent_path.write_text("unknown_phase")

        mgr = DuckDBConnectionManager(db_path)
        mgr.connect()
        try:
            assert db_path.exists(), "Legacy recovery should restore from old"
            assert not old_path.exists(), "old should be consumed"
            assert not intent_path.exists(), "intent should be cleaned up"
        finally:
            mgr.disconnect()

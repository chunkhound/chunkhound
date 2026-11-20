"""Real native lease contracts for cooperative database installation exclusion."""

import gc
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import chunkhound_native as native

# Run the entire contract in a fresh interpreter: pytest/plugins may have
# threads whose inherited runtime locks make Python/PyO3 after fork unsafe.
_FORK_LEASE_PROBE = """
import os
import select
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import chunkhound_native as native


@contextmanager
def fork_holding_inherited_lease(lease, *, close_in_child=False):
    assert threading.active_count() == 1, "fork probe must be single-threaded"
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    try:
        pid = os.fork()
    except BaseException:
        for fd in (ready_read, ready_write, release_read, release_write):
            os.close(fd)
        raise
    if pid == 0:
        try:
            os.close(ready_read)
            os.close(release_write)
            if close_in_child:
                lease.close()
            os.write(ready_write, b"ready")
            os.read(release_read, 1)
        finally:
            os._exit(0)
    try:
        os.close(ready_write)
        os.close(release_read)
        readable, _, _ = select.select([ready_read], [], [], 20)
        assert readable, "fork child did not acknowledge inherited lease"
        assert os.read(ready_read, 5) == b"ready"
        yield
    finally:
        # Reap even when the assertion fails; never leave an inherited owner alive.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            os.waitpid(pid, 0)
            os.close(ready_read)
            os.close(release_write)


target = sys.argv[1]
"""


def _run_fork_probe(tmp_path, code):
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            _FORK_LEASE_PROBE + textwrap.dedent(code),
            str(tmp_path / "chunks.db"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="requires descriptor inheritance via fork"
)
@pytest.mark.parametrize("interrupted", [False, True])
def test_parent_close_releases_ownership_while_fork_child_holds_descriptor(
    tmp_path, interrupted
):
    _run_fork_probe(
        tmp_path,
        f"interrupted = {interrupted!r}\n" + textwrap.dedent("""
        lease = native.acquire_install_lease(target, "duckdb")
        try:
            if interrupted:
                marker = Path(target + ".chunkhound-install.json")
                marker.write_bytes(b'{"version":1}\\n')
            with fork_holding_inherited_lease(lease):
                lease.close()
                if interrupted:
                    try:
                        native.acquire_database_lease(target).close()
                    except native.DatabaseInterruptedError:
                        pass
                    else:
                        raise AssertionError("marker must refuse shared lease")
                else:
                    native.acquire_database_lease(target).close()
        finally:
            lease.close()
        """),
    )


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="requires descriptor inheritance via fork"
)
def test_fork_child_close_does_not_release_live_parent_ownership(tmp_path):
    _run_fork_probe(
        tmp_path,
        """
        lease = native.acquire_install_lease(target, "duckdb")
        try:
            with fork_holding_inherited_lease(lease, close_in_child=True):
                try:
                    native.acquire_database_lease(target).close()
                except native.DatabaseBusyError:
                    pass
                else:
                    raise AssertionError("child close allowed shared competitor")
                try:
                    native.acquire_install_lease(target, "duckdb").close()
                except native.DatabaseBusyError:
                    pass
                else:
                    raise AssertionError("child close allowed install competitor")
        finally:
            lease.close()
        native.acquire_install_lease(target, "duckdb").close()
        """,
    )


def test_each_shared_owner_excludes_install_until_its_own_close(tmp_path):
    target = tmp_path / "chunks.db"
    first = native.acquire_database_lease(str(target))
    second = native.acquire_database_lease(str(target))
    try:
        first.close()
        first.close()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
        second.assert_covers(str(target))
    finally:
        first.close()
        second.close()
    exclusive = native.acquire_install_lease(str(target), "duckdb")
    try:
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
        exclusive.assert_covers(str(target))
        assert not exclusive.interrupted
    finally:
        exclusive.close()
    assert Path(str(target) + ".chunkhound.lock").is_file()


@pytest.mark.parametrize("contents", [b"", b"{", b'{"version":1}\n'])
def test_surviving_marker_refuses_shared_but_allows_exclusive(tmp_path, contents):
    target = tmp_path / "chunks.db"
    Path(str(target) + ".chunkhound-install.json").write_bytes(contents)
    with pytest.raises(native.DatabaseInterruptedError, match="chunkhound pull"):
        native.acquire_database_lease(str(target))
    exclusive = native.acquire_install_lease(str(target), "duckdb")
    try:
        assert exclusive.interrupted
        exclusive.assert_covers(str(target))  # Identity only, no nested shared lock.
        with pytest.raises(native.DatabaseLayoutError):
            exclusive.assert_covers(str(tmp_path / "other.db"))
    finally:
        exclusive.close()
    with pytest.raises(native.DatabaseLayoutError):
        exclusive.assert_covers(str(target))


def test_relative_and_symlink_parent_aliases_share_absent_identity(
    tmp_path, monkeypatch
):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    monkeypatch.chdir(tmp_path)
    lease = native.acquire_database_lease("alias/new/deeper/chunks.db")
    try:
        lease.assert_covers("real/new/deeper/../deeper/chunks.db")
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease("real/new/deeper/chunks.db", "duckdb")
    finally:
        lease.close()


def test_process_exit_releases_ownership_without_deleting_lock(tmp_path):
    target = tmp_path / "chunks.db"
    ready = tmp_path / "ready"
    code = (
        "import pathlib,sys,time,chunkhound_native as n; "
        "lease=n.acquire_database_lease(sys.argv[1]); "
        "pathlib.Path(sys.argv[2]).write_text('ready'); time.sleep(120)"
    )
    process = subprocess.Popen([sys.executable, "-c", code, str(target), str(ready)])
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        assert ready.exists()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
    finally:
        process.kill()
        process.wait(timeout=20)
    exclusive = native.acquire_install_lease(str(target), "duckdb")
    exclusive.close()
    assert Path(str(target) + ".chunkhound.lock").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission-bit contract")
def test_read_only_existing_coordination_is_usable_but_no_unlocked_fallback(tmp_path):
    target = tmp_path / "chunks.db"
    target.write_bytes(b"owned by backend")
    seed = native.acquire_database_lease(str(target))
    seed.close()
    lock = Path(str(target) + ".chunkhound.lock")
    lock.chmod(0o444)
    tmp_path.chmod(0o555)
    try:
        lease = native.acquire_database_lease(str(target))
        lease.close()
        if os.access(tmp_path, os.W_OK):
            pytest.skip("privileged process bypasses directory permissions")
        with pytest.raises(native.DatabaseFilesystemError):
            native.acquire_database_lease(str(tmp_path / "uncoordinated.db"))
    finally:
        tmp_path.chmod(0o755)
        lock.chmod(0o644)


def test_install_refuses_symlink_leaf_but_normal_users_coordinate_referent(tmp_path):
    real = tmp_path / "real" / "chunks.db"
    real.parent.mkdir()
    real.write_bytes(b"existing")
    alias = tmp_path / "alias" / "chunks.db"
    alias.parent.mkdir()
    try:
        alias.symlink_to(real)
    except OSError:
        pytest.skip("file symlink creation is unavailable")
    holder = native.acquire_database_lease(str(alias))
    try:
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(real), "duckdb")
        with pytest.raises(native.DatabaseLayoutError):
            native.acquire_install_lease(str(alias), "duckdb")
    finally:
        holder.close()
    assert alias.is_symlink() and real.read_bytes() == b"existing"


@pytest.mark.parametrize("suffix", ["", "/", "/."])
def test_install_refuses_lancedb_symlink_leaf_with_directory_suffix(tmp_path, suffix):
    real = tmp_path / "real" / "lancedb.lancedb"
    real.mkdir(parents=True)
    alias = tmp_path / "alias" / "lancedb.lancedb"
    alias.parent.mkdir()
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    with pytest.raises(native.DatabaseLayoutError):
        native.acquire_install_lease(str(alias) + suffix, "lancedb").close()
    assert alias.is_symlink() and real.is_dir()
    assert not Path(str(real) + ".chunkhound.lock").exists()


@pytest.mark.parametrize("suffix", ["", "/", "/."])
def test_install_accepts_directory_suffix_after_symlink_parent_traversal(
    tmp_path, suffix
):
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    target = real / "lancedb.lancedb"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real / "nested", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    spelling = str(alias / ".." / "lancedb.lancedb") + suffix
    exclusive = native.acquire_install_lease(spelling, "lancedb")
    try:
        exclusive.assert_covers(str(target))
        with pytest.raises(native.DatabaseLayoutError):
            exclusive.assert_covers(str(tmp_path / "lancedb.lancedb"))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
    finally:
        exclusive.close()
    assert Path(str(target) + ".chunkhound.lock").is_file()


@pytest.mark.parametrize("suffix", ["", "/", "/."])
def test_install_refuses_symlink_leaf_after_absent_parent_traversal(tmp_path, suffix):
    real = tmp_path / "real" / "lancedb.lancedb"
    real.mkdir(parents=True)
    alias = tmp_path / "lancedb.lancedb"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    spelling = str(tmp_path / "missing" / ".." / alias.name) + suffix
    with pytest.raises(native.DatabaseLayoutError):
        native.acquire_install_lease(spelling, "lancedb").close()
    assert alias.is_symlink() and real.is_dir()
    assert not (tmp_path / "missing").exists()
    assert not list(tmp_path.rglob("*.chunkhound.lock"))


@pytest.mark.parametrize(
    "spelling",
    ["lancedb.lancedb", "alias/../missing/../new/deeper/lancedb.lancedb"],
)
def test_install_accepts_absent_target_parents(tmp_path, monkeypatch, spelling):
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real / "nested", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    monkeypatch.chdir(tmp_path)
    target = (
        tmp_path / spelling
        if spelling == "lancedb.lancedb"
        else real / "new" / "deeper" / "lancedb.lancedb"
    )
    assert not target.exists()
    exclusive = native.acquire_install_lease(spelling, "lancedb")
    try:
        exclusive.assert_covers(str(target))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
    finally:
        exclusive.close()
    assert not target.exists()
    assert not (real / "missing").exists()
    assert Path(str(target) + ".chunkhound.lock").is_file()


def test_parent_traversal_resolves_existing_symlink_before_dotdot(tmp_path):
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real / "nested", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    holder = native.acquire_database_lease(str(alias / ".." / "chunks.db"))
    try:
        holder.assert_covers(str(real / "chunks.db"))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(real / "chunks.db"), "duckdb")
        with pytest.raises(native.DatabaseLayoutError):
            holder.assert_covers(str(tmp_path / "chunks.db"))
    finally:
        holder.close()
    with pytest.raises(native.DatabaseLayoutError):
        holder.assert_covers(str(real / "chunks.db"))


def test_lancedb_directory_is_excluded_until_last_handle_is_destroyed(tmp_path):
    target = tmp_path / "lancedb.lancedb"
    target.mkdir()
    holder = native.acquire_database_lease(str(target))
    with pytest.raises(native.DatabaseBusyError):
        native.acquire_install_lease(str(target), "lancedb")
    del holder
    gc.collect()
    exclusive = native.acquire_install_lease(str(target), "lancedb")
    try:
        exclusive.assert_covers(str(target))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
    finally:
        exclusive.close()
        exclusive.close()
    with pytest.raises(native.DatabaseLayoutError):
        _ = exclusive.interrupted


def test_legacy_and_memory_ownership_remain_usable_without_installation(tmp_path):
    for path in (str(tmp_path / "legacy.db"), ":memory:"):
        holder = native.acquire_database_lease(path)
        try:
            holder.assert_covers(path)
            with pytest.raises(native.DatabaseLayoutError):
                native.acquire_install_lease(path, "duckdb")
        finally:
            holder.close()


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_install_rejects_wrong_backend_shape_without_touching_target(tmp_path, backend):
    if backend == "duckdb":
        target = tmp_path / "chunks.db"
        target.mkdir()
    else:
        target = tmp_path / "lancedb.lancedb"
        target.write_bytes(b"not a directory")
    with pytest.raises(native.DatabaseLayoutError):
        native.acquire_install_lease(str(target), backend)
    assert target.exists()
    assert not Path(str(target) + ".chunkhound.lock").exists()


def test_unknown_backend_and_empty_target_are_explicit_layout_errors(tmp_path):
    with pytest.raises(native.DatabaseLayoutError):
        native.acquire_install_lease(str(tmp_path / "chunks.db"), "unknown")
    with pytest.raises(native.DatabaseLayoutError):
        native.acquire_database_lease("")

"""Contract tests -- install_native.py's user-facing contracts.

The rationale for the install flow lives in scripts/install_native.py
(see _select_files). The contracts tested here:

1. dist-info of already-installed packages is never overwritten -- uv's
   auto-sync would otherwise reinstall the locked PyPI wheel, reverting
   the local build.
2. The source tree's copies are real, hand-editable, git-tracked files --
   uncommitted local edits that differ from the wheel are never clobbered
   (data-loss guard, including when git can't prove cleanliness); clean
   ones are.
3. find_wheels and extraction fail closed with actionable messages for
   the `make dev` loop.
"""

import importlib.util
import pathlib
import shutil
import subprocess
import zipfile

import pytest

SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "install_native.py"
)

WHEEL_INIT_CONTENT = b"fake content for chunkhound_native/__init__.py"
REPO_ROOT = SCRIPT_PATH.parent.parent


def _dry_run_make(target: str) -> str:
    """Return the commands a public Makefile target would run."""
    result = subprocess.run(
        ["make", "-n", target],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _load_install_native():
    """Import install_native.py as a module so we can call its functions
    directly with temp-directory paths (no real maturin build needed)."""
    spec = importlib.util.spec_from_file_location("install_native", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_wheel(wheel_path: pathlib.Path, dist_info_version: str = "0.1.0") -> None:
    """Create a minimal synthetic wheel with a compiled extension, __init__.py,
    a bundled .libs/ file, AND a dist-info entry at a stale version.

    The stale dist-info mirrors what maturin produces at the repo root: the
    wheel's name comes from pyproject.toml (chunkhound) and its version from
    Cargo.toml (0.1.0), neither matching the installed/locked versions.
    """
    names = [
        "chunkhound_native/__init__.py",
        "chunkhound_native/chunkhound_native.abi3.so",
        "chunkhound_native/chunkhound_native.pyd",
        "chunkhound_native.libs/libduckdb-fake.so",
        f"chunkhound-{dist_info_version}.dist-info/METADATA",
        f"chunkhound-{dist_info_version}.dist-info/RECORD",
        f"chunkhound-{dist_info_version}.data/purelib/README.txt",
    ]
    with zipfile.ZipFile(wheel_path, "w") as z:
        for name in names:
            z.writestr(name, b"fake content for " + name.encode())


@pytest.fixture
def fake_wheel_setup(tmp_path):
    """Create a temp directory with a synthetic wheel and fake site-packages."""
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    wheel_path = wheels_dir / "chunkhound-0.1.0-py3-none-any.whl"
    _make_wheel(wheel_path, dist_info_version="0.1.0")

    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # Simulate the PyPI-installed chunkhound-native>=5.2.0 dist-info already
    # present in site-packages (the one we must NOT overwrite).
    dist_info = site_dir / "chunkhound_native-5.2.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Version: 5.2.0\n")

    return wheel_path, site_dir, repo_root, tmp_path


def _git_init_with_commit(repo_root: pathlib.Path, init_content: bytes) -> pathlib.Path:
    """Turn `repo_root` into a real git repo with `chunkhound_native/__init__.py`
    committed at `init_content`, then dirty the working copy. Returns the file."""
    init_file = repo_root / "chunkhound_native" / "__init__.py"
    init_file.parent.mkdir(parents=True)
    init_file.write_bytes(init_content)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "chunkhound_native/__init__.py")
    git("commit", "-m", "baseline")

    # Dirty the working copy with content that differs from both the commit
    # and the wheel's content.
    init_file.write_bytes(b"uncommitted local edit")
    return init_file


class TestInstallNativePreservesDistInfo:
    """Extraction is a strict whitelist and never touches dist-info, so uv
    sees the already-installed PyPI version as satisfied and won't auto-
    reinstall over the local build."""

    def test_extracts_only_whitelisted_files_and_preserves_dist_info(
        self, fake_wheel_setup
    ):
        wheel_path, site_dir, repo_root, _ = fake_wheel_setup
        mod = _load_install_native()

        mod.extract_from_wheel(wheel_path, [repo_root, site_dir], repo_root)

        for extracted in (
            "chunkhound_native/chunkhound_native.abi3.so",
            "chunkhound_native/chunkhound_native.pyd",
            "chunkhound_native/__init__.py",
            "chunkhound_native.libs/libduckdb-fake.so",
        ):
            assert (site_dir / extracted).exists()

        # The whitelist is exclusive: dist-info/.data entries from the wheel
        # must not land anywhere -- a stale 0.1.0 dist-info would trigger
        # uv run's auto-sync to reinstall the locked PyPI packages.
        for skipped in ("chunkhound-0.1.0.dist-info", "chunkhound-0.1.0.data"):
            assert not (site_dir / skipped).exists()

        # The pre-existing PyPI dist-info must survive untouched
        pypi_dist = site_dir / "chunkhound_native-5.2.0.dist-info"
        assert pypi_dist.joinpath("METADATA").read_text() == "Version: 5.2.0\n"


class TestRejectsUnsafeWheelMembers:
    """Selectable wheel entries must not escape an extraction root."""

    @pytest.mark.parametrize(
        "unsafe_member",
        [
            "chunkhound_native/../../escaped.so",
            "chunkhound_native.libs/../../escaped.so",
            "chunkhound_native/..\\escaped.pyd",
            "/chunkhound_native/escaped.so",
            "C:chunkhound_native/escaped.pyd",
        ],
        ids=[
            "parent-traversal",
            "libs-parent-traversal",
            "windows-separator",
            "absolute-path",
            "windows-drive",
        ],
    )
    def test_rejects_unsafe_member_before_writing_any_files(
        self, fake_wheel_setup, unsafe_member
    ):
        wheel_path, site_dir, repo_root, tmp_path = fake_wheel_setup
        mod = _load_install_native()
        with zipfile.ZipFile(wheel_path, "a") as z:
            z.writestr(unsafe_member, b"malicious content")

        with pytest.raises(SystemExit, match="Unsafe wheel member"):
            mod.extract_from_wheel(wheel_path, [repo_root, site_dir], repo_root)

        # Validation must run before ANY extraction: nothing was written anywhere
        assert not (site_dir / "chunkhound_native").exists()
        assert not (repo_root / "chunkhound_native").exists()
        assert list(tmp_path.glob("escaped*")) == []


class TestPreservesUncommittedEdits:
    """The data-loss guard: the source tree's __init__.py is a real,
    hand-editable, git-tracked file -- uncommitted edits that differ from
    the wheel must never be clobbered; clean (committed) files must be."""

    def test_refuses_to_clobber_uncommitted_edits(self, fake_wheel_setup):
        wheel_path, site_dir, repo_root, _ = fake_wheel_setup
        mod = _load_install_native()
        init_file = _git_init_with_commit(repo_root, WHEEL_INIT_CONTENT)

        with pytest.raises(SystemExit, match="uncommitted changes"):
            mod.extract_from_wheel(wheel_path, [repo_root, site_dir], repo_root)

        assert init_file.read_bytes() == b"uncommitted local edit", (
            "the data-loss guard must leave the uncommitted edit untouched"
        )

    def test_refuses_to_clobber_untracked_file_in_a_git_repo(self, fake_wheel_setup):
        """An untracked source-tree destination is not safe to overwrite."""
        wheel_path, site_dir, repo_root, _ = fake_wheel_setup
        mod = _load_install_native()
        init_file = _git_init_with_commit(repo_root, WHEEL_INIT_CONTENT)
        subprocess.run(
            ["git", "rm", "--cached", "chunkhound_native/__init__.py"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "untrack init"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        init_file.write_bytes(b"untracked local file")

        with pytest.raises(SystemExit, match="uncommitted changes"):
            mod.extract_from_wheel(wheel_path, [repo_root, site_dir], repo_root)

        assert init_file.read_bytes() == b"untracked local file"

    @pytest.mark.parametrize(
        "artifact",
        [
            "chunkhound_native/chunkhound_native.abi3.so",
            "chunkhound_native.libs/libduckdb-fake.so",
        ],
        ids=["extension", "runtime-library"],
    )
    def test_replaces_untracked_generated_artifacts(self, fake_wheel_setup, artifact):
        """Generated wheel artifacts must be replaceable on every rebuild."""
        wheel_path, site_dir, repo_root, _ = fake_wheel_setup
        mod = _load_install_native()
        init_file = _git_init_with_commit(repo_root, WHEEL_INIT_CONTENT)
        init_file.write_bytes(WHEEL_INIT_CONTENT)
        artifact_file = repo_root / artifact
        artifact_file.parent.mkdir(parents=True, exist_ok=True)
        artifact_file.write_bytes(b"previous generated artifact")

        mod.extract_from_wheel(wheel_path, [repo_root, site_dir], repo_root)

        assert artifact_file.read_bytes() == b"fake content for " + artifact.encode()

    def test_refuses_to_clobber_outside_a_git_repo(self, fake_wheel_setup):
        """Fail closed even when git cannot prove the file is clean (rc=128
        outside a git repo): a differing existing file is never overwritten."""
        wheel_path, site_dir, repo_root, _ = fake_wheel_setup
        mod = _load_install_native()
        init_file = repo_root / "chunkhound_native" / "__init__.py"
        init_file.parent.mkdir(parents=True)
        init_file.write_bytes(b"uncommitted local edit")

        with pytest.raises(SystemExit, match="uncommitted changes"):
            mod.extract_from_wheel(wheel_path, [repo_root, site_dir], repo_root)

        assert init_file.read_bytes() == b"uncommitted local edit"

    def test_overwrites_clean_committed_files(self, fake_wheel_setup):
        wheel_path, site_dir, repo_root, _ = fake_wheel_setup
        mod = _load_install_native()
        # Committed content differs from the wheel's content but is clean
        init_file = _git_init_with_commit(repo_root, b"committed baseline")
        # Restore the committed content so the working copy is not dirty
        init_file.write_bytes(b"committed baseline")

        mod.extract_from_wheel(wheel_path, [repo_root, site_dir], repo_root)

        assert init_file.read_bytes() == WHEEL_INIT_CONTENT


class TestFindWheelsErrorPaths:
    """find_wheels must fail explicitly with actionable messages when the
    wheel directory is unusable -- a broken `make dev` should say why."""

    def test_no_wheel_fails_with_build_hint(self, tmp_path):
        mod = _load_install_native()

        with pytest.raises(SystemExit, match="No wheel found.*maturin build"):
            mod.find_wheels(tmp_path)

    def test_multiple_wheels_fail_with_cleanup_hint(self, tmp_path):
        mod = _load_install_native()
        _make_wheel(tmp_path / "chunkhound-0.1.0-cp39.whl")
        _make_wheel(tmp_path / "chunkhound-0.1.0-cp312.whl")

        with pytest.raises(SystemExit, match="Multiple wheels found"):
            mod.find_wheels(tmp_path)


class TestFailsClosedOnEmptyWheel:
    """A wheel with no extension files must fail explicitly -- silently
    installing nothing would leave `make dev` with a stale/broken native
    extension (fail-closed, matching the repo's philosophy)."""

    @pytest.mark.parametrize(
        "wheel_contents",
        [
            # A stale or foreign metadata-only chunkhound*.whl
            {"chunkhound-0.1.0.dist-info/METADATA": b"stale metadata only"},
            # Selectable support files cannot make a wheel valid without a
            # compiled extension
            {
                "chunkhound_native/__init__.py": b"package marker",
                "chunkhound_native.libs/libduckdb.so": b"runtime",
            },
        ],
        ids=["metadata-only", "support-files-only"],
    )
    def test_wheel_without_extension_fails_with_actionable_message(
        self, tmp_path, wheel_contents
    ):
        mod = _load_install_native()
        wheel_path = tmp_path / "chunkhound-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel_path, "w") as z:
            for name, content in wheel_contents.items():
                z.writestr(name, content)

        with pytest.raises(SystemExit, match="No extension files"):
            mod.extract_from_wheel(
                wheel_path, [tmp_path / "site", tmp_path / "repo"], tmp_path / "repo"
            )
        assert not (tmp_path / "site" / "chunkhound_native").exists()


class TestIsSafeMember:
    """Direct unit tests for _is_safe_member's edge-case branches.

    Integration tests (TestRejectsUnsafeWheelMembers) exercise _validate_members
    through extract_from_wheel, but zipfile.writestr silently truncates names at
    \x00, so the null-byte branch is only reachable via a direct call.
    """

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("", False),  # empty name
            ("chunkhound_native/x.so", True),  # happy path
            ("chunkhound_native/../x.so", False),  # parent traversal
            ("chunkhound_native/./x.so", False),  # dot segment
            ("chunkhound_native//x.so", False),  # empty segment
            ("chunkhound_native/\x00x.so", False),  # null byte (zipfile truncates
            # names at \x00, so this shape is only reachable via a direct call)
        ],
        ids=[
            "empty",
            "valid",
            "parent-traversal",
            "dot-segment",
            "empty-segment",
            "null-byte",
        ],
    )
    def test_rejects_unsafe_members(self, name, expected):
        mod = _load_install_native()
        assert mod._is_safe_member(name) is expected


requires_make = pytest.mark.skipif(
    shutil.which("make") is None,
    reason="GNU make is not on PATH (e.g. GitHub Actions windows-latest runners)",
)


@requires_make
def test_dev_target_uses_the_shared_native_install_workflow():
    """`make dev` and `make dev-release` must share the no-sync install contract."""
    for target in ("dev", "dev-release"):
        commands = _dry_run_make(target)
        for required in (
            "uv run --no-sync python scripts/install_native.py",
            "uv run --no-sync python scripts/copy_duckdb_runtime.py",
        ):
            assert required in commands, f"{target} missing {required!r}"
        # Both must use maturin build (not develop) to avoid 0.1.0 dist-info conflict
        assert "maturin build" in commands
        assert "maturin develop" not in commands


@requires_make
def test_dev_is_fast_debug_while_dev_release_is_optimized():
    """`dev` is debug (no --release) for fast iteration; `dev-release` is --release."""
    dev = _dry_run_make("dev")
    rel = _dry_run_make("dev-release")
    assert "maturin build --out target/wheels/" in dev
    assert "maturin build --release --out target/wheels/" in rel
    # cargo clean must match build profile
    assert "cargo clean -p chunkhound_native --release" in rel
    assert "cargo clean -p chunkhound_native --release" not in dev

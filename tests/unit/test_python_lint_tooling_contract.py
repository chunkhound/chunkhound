"""Contract tests for the ChunkHound pre-commit lint tooling.

Tests the external invariants of:
  - Script subcommands: install, run-files, run-staged, run-changed, run-ruff
  - Git diff range resolution for CI events (pull_request, merge_group, push)
  - Ruff rewrite detection via SHA256 digests
  - Real uv+ruff integration (guarded behind CHUNKHOUND_RUN_RUFF_INTEGRATION)
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]
from tests.utils import SUBPROCESS_ENV_ALLOWLIST as _SUBPROCESS_ENV_ALLOWLIST
from tests.utils.git_repo import (
    commit_all as _commit_all,
)
from tests.utils.git_repo import (
    create_repo as _create_repo,
)
from tests.utils.git_repo import (
    run as _run,
)
from tests.utils.windows_subprocess import get_safe_subprocess_env

ROOT = Path(__file__).resolve().parents[2]
PRE_COMMIT_SCRIPT = ROOT / "scripts" / "pre_commit.py"


def _load_pre_commit_module() -> Any:
    spec = importlib.util.spec_from_file_location("_pre_commit_mod", PRE_COMMIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PRE_COMMIT_MODULE = _load_pre_commit_module()
PRE_COMMIT_PACKAGE = _PRE_COMMIT_MODULE.PRE_COMMIT_PACKAGE
RUFF_PACKAGE = _PRE_COMMIT_MODULE.RUFF_PACKAGE
STAGED_SNAPSHOT_PREFIX = _PRE_COMMIT_MODULE.STAGED_SNAPSHOT_PREFIX
HOOK_MARKER = "chunkhound-managed-pre-commit-hook"


def _run_capture(
    command: list[str], cwd: Path, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _current_branch(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fake_uv_dir(
    tmp_path: Path,
    *,
    exit_codes: tuple[int, ...] = (0,),
    file_updates: tuple[tuple[int, str, str], ...] = (),
) -> tuple[Path, Path]:
    """Create a fake uv executable that logs invocations and optionally modifies files.

    Protocols:
      - ``<<<UV-CALL>>>`` separator: delimits multiple invocations in the log file.
        ``_uv_log_calls`` splits on this to recover per-call argument lists.
      - ``file_updates`` paths are resolved relative to the fake uv process cwd.
      - ``BASENAME:`` prefix in ``file_updates``: at runtime the fake resolves
        ``BASENAME:tracked.py`` to the actual snapshot path whose basename matches.
        This lets tests specify file updates without knowing the temp directory.
    """
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log_path = tmp_path / "uv.log"
    count_path = tmp_path / "uv.count"
    python = sys.executable.replace("\\", "\\\\")
    script = "\n".join(
        [
            "import pathlib, sys",
            f"log_path = pathlib.Path({str(log_path)!r})",
            f"count_path = pathlib.Path({str(count_path)!r})",
            f"exit_codes = {list(exit_codes)!r}",
            f"file_updates = {list(file_updates)!r}",
            "def _normalized_args(argv):",
            "    normalized = []",
            "    for arg in argv:",
            "        if (",
            f"            '{STAGED_SNAPSHOT_PREFIX}' in arg",
            "            and arg.endswith(('.py', '.pyi'))",
            "        ):",
            "            normalized.append(pathlib.Path(arg).name)",
            "        else:",
            "            normalized.append(arg)",
            "    return normalized",
            "def _resolve_update_path(relative_path):",
            "    if relative_path.startswith('BASENAME:'):",
            "        target = relative_path.removeprefix('BASENAME:')",
            "        for arg in sys.argv[1:]:",
            "            if pathlib.Path(arg).name == target:",
            "                return pathlib.Path(arg)",
            "    return pathlib.Path(relative_path)",
            "count = (",
            "    int(count_path.read_text(encoding='utf-8'))",
            "    if count_path.exists()",
            "    else 0",
            ")",
            "with log_path.open('a', encoding='utf-8') as handle:",
            "    if count:",
            "        handle.write('<<<UV-CALL>>>\\n')",
            "    handle.write('\\n'.join(_normalized_args(sys.argv[1:])) + '\\n')",
            "for update_count, relative_path, contents in file_updates:",
            "    if update_count == count:",
            "        path = _resolve_update_path(relative_path)",
            "        path.write_text(contents, encoding='utf-8')",
            "count_path.write_text(str(count + 1), encoding='utf-8')",
            "exit_code = exit_codes[count] if count < len(exit_codes) else (",
            "    exit_codes[-1]",
            ")",
            "raise SystemExit(exit_code)",
            "",
        ]
    )

    if os.name == "nt":
        # Write the implementation as a standalone Python script referenced
        # by the batch file — avoids multi-line quoting issues in CMD.
        impl_path = bin_dir / "_uv_fake.py"
        impl_path.write_text(script, encoding="utf-8")
        (bin_dir / "uv.cmd").write_text(
            f'@"{sys.executable}" "%~dp0_uv_fake.py" %*\n',
            encoding="utf-8",
        )
    else:
        uv_path = bin_dir / "uv"
        uv_path.write_text(f"#!{python}\n{script}", encoding="utf-8")
        uv_path.chmod(0o755)

    return bin_dir, log_path


def _uv_log_calls(log_path: Path) -> list[list[str]]:
    contents = log_path.read_text(encoding="utf-8")
    return [
        call.strip().splitlines()
        for call in contents.split("<<<UV-CALL>>>")
        if call.strip()
    ]


def _script_env(bin_dir: Path) -> dict[str, str]:
    base_env = {
        key: os.environ[key]
        for key in (*_SUBPROCESS_ENV_ALLOWLIST, "SHELL")
        if key in os.environ
    }
    path = str(bin_dir)
    if "PATH" in base_env:
        path = f"{path}{os.pathsep}{base_env['PATH']}"
    base_env["PATH"] = path
    return get_safe_subprocess_env(base_env)


def _run_pre_commit_script(
    repo_dir: Path, *args: str, env: dict[str, str], timeout: int = 10
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PRE_COMMIT_SCRIPT), *args],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _repo_env(tmp_path: Path) -> dict[str, str]:
    base_env = get_safe_subprocess_env(dict(os.environ))
    base_env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    base_env["UV_TOOL_DIR"] = str(tmp_path / "uv-tools")
    base_env["UV_PYTHON_INSTALL_DIR"] = str(tmp_path / "uv-python")
    base_env["HOME"] = str(tmp_path / "home")
    for key in ("UV_CACHE_DIR", "UV_TOOL_DIR", "UV_PYTHON_INSTALL_DIR", "HOME"):
        Path(base_env[key]).mkdir(parents=True, exist_ok=True)
    return base_env


def _copy_pre_commit_config(repo_dir: Path) -> None:
    shutil.copy2(ROOT / ".pre-commit-config.yaml", repo_dir / ".pre-commit-config.yaml")


def _copy_pre_commit_script(repo_dir: Path) -> None:
    script_path = repo_dir / "scripts" / "pre_commit.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRE_COMMIT_SCRIPT, script_path)


def _create_changed_python_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _create_repo(repo_dir)
    (repo_dir / "kept.py").write_text("print('old')\n", encoding="utf-8")
    (repo_dir / "deleted.py").write_text("print('delete')\n", encoding="utf-8")
    (repo_dir / "notes.md").write_text("old\n", encoding="utf-8")
    base = _commit_all(repo_dir, "base")

    (repo_dir / "kept.py").write_text("print('new')\n", encoding="utf-8")
    (repo_dir / "added.pyi").write_text("value: int\n", encoding="utf-8")
    (repo_dir / "notes.md").write_text("new\n", encoding="utf-8")
    (repo_dir / "deleted.py").unlink()
    head = _commit_all(repo_dir, "head")
    return repo_dir, base, head


def _create_changed_python_repo_with_subdir(
    tmp_path: Path,
) -> tuple[Path, str, str]:
    """Like _create_changed_python_repo but with files in subdirectories.

    Verifies that git diff pathspec *.py matches across directory
    boundaries (not just repo root).
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _create_repo(repo_dir)
    (repo_dir / "src").mkdir()
    (repo_dir / "tests").mkdir()
    (repo_dir / "kept.py").write_text("print('old')\n", encoding="utf-8")
    (repo_dir / "deleted.py").write_text("print('delete')\n", encoding="utf-8")
    (repo_dir / "src" / "utils.py").write_text("print('old')\n", encoding="utf-8")
    (repo_dir / "tests" / "test_app.py").write_text("print('old')\n", encoding="utf-8")
    (repo_dir / "notes.md").write_text("old\n", encoding="utf-8")
    base = _commit_all(repo_dir, "base")

    (repo_dir / "kept.py").write_text("print('new')\n", encoding="utf-8")
    (repo_dir / "added.pyi").write_text("value: int\n", encoding="utf-8")
    (repo_dir / "src" / "utils.py").write_text("print('new')\n", encoding="utf-8")
    (repo_dir / "tests" / "test_app.py").write_text(
        "print('updated')\n", encoding="utf-8"
    )
    (repo_dir / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repo_dir / "tests" / "test_api.pyi").write_text("value: int\n", encoding="utf-8")
    (repo_dir / "notes.md").write_text("new\n", encoding="utf-8")
    (repo_dir / "deleted.py").unlink()
    head = _commit_all(repo_dir, "head")
    return repo_dir, base, head


def _create_non_python_change_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _create_repo(repo_dir)
    (repo_dir / "notes.md").write_text("old\n", encoding="utf-8")
    base = _commit_all(repo_dir, "base")

    (repo_dir / "notes.md").write_text("new\n", encoding="utf-8")
    (repo_dir / "data.json").write_text("{}", encoding="utf-8")
    head = _commit_all(repo_dir, "head")
    return repo_dir, base, head


def _run_diff_command(
    repo_dir: Path,
    command: str,
    *,
    env: dict[str, str],
    base: str | None = None,
    head: str | None = None,
    github_actions: bool = False,
    timeout: int = 10,
) -> subprocess.CompletedProcess[str]:
    args = [command]
    if github_actions:
        args.append("--github-actions")
    elif base is not None and head is not None:
        args.extend(["--event-name", "push", "--base", base, "--head", head])
    return _run_pre_commit_script(repo_dir, *args, env=env, timeout=timeout)


def test_pre_commit_ruff_hook_contract() -> None:
    config = cast(
        dict[str, Any],
        yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")),
    )

    repos = cast(list[dict[str, Any]], config["repos"])
    assert len(repos) == 1
    repo = repos[0]
    assert repo["repo"] == "https://github.com/astral-sh/ruff-pre-commit"
    assert repo["rev"] == f"v{RUFF_PACKAGE.removeprefix('ruff==')}"

    hooks = cast(list[dict[str, Any]], repo["hooks"])
    assert [hook["id"] for hook in hooks] == ["ruff-check", "ruff-format"]
    assert hooks[0]["args"] == ["--fix"]
    assert hooks[0]["types_or"] == ["python", "pyi"]
    assert hooks[1]["types_or"] == ["python", "pyi"]


def test_install_hooks_make_target_contract() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "install-hooks:\n\tuv run python scripts/pre_commit.py install\n" in makefile


@pytest.mark.skipif(
    os.name != "nt",
    reason="Windows PATHEXT resolution only applies on Windows",
)
def test_run_files_finds_uv_cmd_via_pathext_on_windows(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _create_repo(repo_dir)
    bin_dir, log_path = _fake_uv_dir(tmp_path)

    result = _run_pre_commit_script(
        repo_dir,
        "run-files",
        "chunkhound/app.py",
        env=_script_env(bin_dir),
    )

    assert result.returncode == 0
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "tool",
        "run",
        "--from",
        PRE_COMMIT_PACKAGE,
        "pre-commit",
        "run",
        "--files",
        "chunkhound/app.py",
    ]


class TestPreCommitScript:
    def test_install_writes_managed_hook(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        bin_dir, _ = _fake_uv_dir(tmp_path)

        result = _run_pre_commit_script(repo_dir, "install", env=_script_env(bin_dir))

        hook_path = repo_dir / ".git" / "hooks" / "pre-commit"
        assert result.returncode == 0
        assert hook_path.exists()
        assert HOOK_MARKER in hook_path.read_text(encoding="utf-8")
        assert "run-staged" in hook_path.read_text(encoding="utf-8")

    def test_install_respects_configured_hookspath(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        _run(["git", "config", "core.hooksPath", ".githooks"], repo_dir)
        bin_dir, _ = _fake_uv_dir(tmp_path)

        result = _run_pre_commit_script(repo_dir, "install", env=_script_env(bin_dir))

        hook_path = repo_dir / ".githooks" / "pre-commit"
        assert result.returncode == 0
        assert hook_path.exists()
        assert HOOK_MARKER in hook_path.read_text(encoding="utf-8")

    def test_install_refuses_to_overwrite_non_managed_hook(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        hook_path = repo_dir / ".git" / "hooks" / "pre-commit"
        hook_path.write_text("#!/bin/sh\necho custom\n", encoding="utf-8")
        bin_dir, _ = _fake_uv_dir(tmp_path)

        result = _run_pre_commit_script(repo_dir, "install", env=_script_env(bin_dir))

        assert result.returncode == 1
        assert "Refusing to overwrite existing Git hook" in result.stderr
        assert "--overwrite-hook" in result.stderr
        assert HOOK_MARKER not in hook_path.read_text(encoding="utf-8")

    def test_install_can_overwrite_non_managed_hook(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        hook_path = repo_dir / ".git" / "hooks" / "pre-commit"
        hook_path.write_text("#!/bin/sh\necho custom\n", encoding="utf-8")
        bin_dir, _ = _fake_uv_dir(tmp_path)

        result = _run_pre_commit_script(
            repo_dir,
            "install",
            "--overwrite-hook",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert HOOK_MARKER in hook_path.read_text(encoding="utf-8")

    def test_run_files_forwards_paths_to_pre_commit(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        bin_dir, log_path = _fake_uv_dir(tmp_path)

        result = _run_pre_commit_script(
            repo_dir,
            "run-files",
            "chunkhound/app.py",
            "chunkhound/types.pyi",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert log_path.read_text(encoding="utf-8").splitlines() == [
            "tool",
            "run",
            "--from",
            PRE_COMMIT_PACKAGE,
            "pre-commit",
            "run",
            "--files",
            "chunkhound/app.py",
            "chunkhound/types.pyi",
        ]

    def test_run_staged_uses_only_staged_python_paths(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        _copy_pre_commit_config(repo_dir)
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "tracked.py").write_text("x=1\n", encoding="utf-8")
        (repo_dir / "added.pyi").write_text("value: int\n", encoding="utf-8")
        (repo_dir / "notes.md").write_text("new\n", encoding="utf-8")
        _run(["git", "add", "tracked.py", "added.pyi", "notes.md"], repo_dir)
        bin_dir, log_path = _fake_uv_dir(tmp_path)

        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert _uv_log_calls(log_path) == [
            [
                "tool",
                "run",
                "--from",
                RUFF_PACKAGE,
                "ruff",
                "check",
                "--fix",
                "--",
                "added.pyi",
                "tracked.py",
            ],
            [
                "tool",
                "run",
                "--from",
                RUFF_PACKAGE,
                "ruff",
                "format",
                "--",
                "added.pyi",
                "tracked.py",
            ],
        ]

    def test_run_staged_uses_pyproject_config_when_present(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        # Production hook runs from repo root where pyproject.toml exists;
        # verify that _repo_ruff_args adds --config in that case.
        shutil.copy2(ROOT / "pyproject.toml", repo_dir / "pyproject.toml")
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "tracked.py").write_text("x=1\n", encoding="utf-8")
        _run(["git", "add", "tracked.py"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(0, 0))
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        calls = _uv_log_calls(log_path)
        assert len(calls) == 2
        for call in calls:
            assert "--config" in call, f"--config missing from ruff args: {call}"
            assert any("pyproject.toml" in arg for arg in call), (
                f"pyproject.toml path missing from ruff args: {call}"
            )

    def test_run_staged_handles_subdirectory_paths(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "src").mkdir()
        (repo_dir / "src" / "util.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "src" / "util.py").write_text("x=1\n", encoding="utf-8")
        _run(["git", "add", "src/util.py"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(0, 0))
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        calls = _uv_log_calls(log_path)
        assert len(calls) == 2
        for call in calls:
            assert any("src/util.py" == arg for arg in call)

    def test_run_staged_preserves_same_basename_paths_in_different_directories(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "pkg_a").mkdir()
        (repo_dir / "pkg_b").mkdir()
        (repo_dir / "pkg_a" / "shared.py").write_text("a = 1\n", encoding="utf-8")
        (repo_dir / "pkg_b" / "shared.py").write_text("b = 2\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "pkg_a" / "shared.py").write_text("a=1\n", encoding="utf-8")
        (repo_dir / "pkg_b" / "shared.py").write_text("b=2\n", encoding="utf-8")
        _run(["git", "add", "pkg_a/shared.py", "pkg_b/shared.py"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(
            tmp_path,
            exit_codes=(0, 0),
            file_updates=(
                (0, "pkg_a/shared.py", "a = 1\n"),
                (0, "pkg_b/shared.py", "b = 2\n"),
            ),
        )
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert (repo_dir / "pkg_a" / "shared.py").read_text(encoding="utf-8") == (
            "a = 1\n"
        )
        assert (repo_dir / "pkg_b" / "shared.py").read_text(encoding="utf-8") == (
            "b = 2\n"
        )
        assert _run_capture(["git", "show", ":pkg_a/shared.py"], repo_dir).stdout == (
            "a = 1\n"
        )
        assert _run_capture(["git", "show", ":pkg_b/shared.py"], repo_dir).stdout == (
            "b = 2\n"
        )
        calls = _uv_log_calls(log_path)
        assert len(calls) == 2
        for call in calls:
            assert "pkg_a/shared.py" in call
            assert "pkg_b/shared.py" in call

    def test_run_staged_matches_repo_relative_ruff_paths(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        shutil.copy2(ROOT / "pyproject.toml", repo_dir / "pyproject.toml")
        target_dir = repo_dir / "chunkhound" / "mcp_server"
        target_dir.mkdir(parents=True)
        target = target_dir / "tools.py"
        target.write_text("# base\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        target.write_text("# " + "x" * 120 + "\n", encoding="utf-8")
        _run(["git", "add", "chunkhound/mcp_server/tools.py"], repo_dir)

        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_repo_env(tmp_path),
            timeout=120,
        )

        assert result.returncode == 0, result.stderr
        assert "Ruff rewrote" not in result.stderr
        staged_file = _run_capture(
            ["git", "show", ":chunkhound/mcp_server/tools.py"],
            repo_dir,
        )
        assert staged_file.stdout == "# " + "x" * 120 + "\n"

    def test_run_staged_returns_explicit_error_for_unmerged_file(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        base_branch = _current_branch(repo_dir)

        _run(["git", "checkout", "-b", "other"], repo_dir)
        (repo_dir / "tracked.py").write_text("x = 2\n", encoding="utf-8")
        _commit_all(repo_dir, "other change")

        _run(["git", "checkout", base_branch], repo_dir)
        (repo_dir / "tracked.py").write_text("x = 3\n", encoding="utf-8")
        _commit_all(repo_dir, "main change")

        merge = _run_capture(["git", "merge", "other"], repo_dir)
        assert merge.returncode != 0

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(0, 0))
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert (
            "run-staged does not support unmerged files. Resolve conflicts first."
            in result.stderr
        )
        assert (
            "fatal: path 'tracked.py' is in the index, but not at stage 0"
            not in result.stderr
        )
        assert not log_path.exists()

    def test_run_staged_returns_nonzero_when_ruff_check_rewrites_file(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "tracked.py").write_text("x=1\n", encoding="utf-8")
        _run(["git", "add", "tracked.py"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(
            tmp_path,
            exit_codes=(0, 0),
            file_updates=((0, "BASENAME:tracked.py", "x = 1\n"),),
        )
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert "Ruff rewrote staged Python files" in result.stderr
        assert "tracked.py" in result.stderr
        assert (repo_dir / "tracked.py").read_text(encoding="utf-8") == "x = 1\n"
        staged_file = _run_capture(["git", "show", ":tracked.py"], repo_dir)
        assert staged_file.stdout == "x = 1\n"
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_staged_returns_nonzero_when_ruff_format_rewrites_file(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "tracked.py").write_text("x=1\n", encoding="utf-8")
        _run(["git", "add", "tracked.py"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(
            tmp_path,
            exit_codes=(0, 0),
            file_updates=((1, "BASENAME:tracked.py", "x = 1\n"),),
        )
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert "Ruff rewrote staged Python files" in result.stderr
        assert "tracked.py" in result.stderr
        assert (repo_dir / "tracked.py").read_text(encoding="utf-8") == "x = 1\n"
        staged_file = _run_capture(["git", "show", ":tracked.py"], repo_dir)
        assert staged_file.stdout == "x = 1\n"
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_staged_preserves_unstaged_hunks_for_partially_staged_file(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "tracked.py").write_text("x=1\n", encoding="utf-8")
        _run(["git", "add", "tracked.py"], repo_dir)
        (repo_dir / "tracked.py").write_text(
            "x=1\nunstaged_value=2\n",
            encoding="utf-8",
        )

        bin_dir, log_path = _fake_uv_dir(
            tmp_path,
            exit_codes=(0, 0),
            file_updates=((0, "BASENAME:tracked.py", "x = 1\n"),),
        )
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert "Ruff rewrote staged Python files" in result.stderr
        assert "fixed in the index only" in result.stderr
        assert (repo_dir / "tracked.py").read_text(encoding="utf-8") == (
            "x=1\nunstaged_value=2\n"
        )
        staged_file = _run_capture(["git", "show", ":tracked.py"], repo_dir)
        assert staged_file.stdout == "x = 1\n"
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_staged_returns_zero_when_ruff_does_not_rewrite(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "tracked.py").write_text("x=1\n", encoding="utf-8")
        _run(["git", "add", "tracked.py"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(0, 0))
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert "Ruff rewrote" not in result.stderr
        assert (repo_dir / "tracked.py").read_text(encoding="utf-8") == "x=1\n"
        staged_file = _run_capture(["git", "show", ":tracked.py"], repo_dir)
        assert staged_file.stdout == "x=1\n"
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_staged_preserves_unstaged_hunks_for_multiple_partially_staged_files(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
        (repo_dir / "b.py").write_text("y = 2\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "a.py").write_text("x=1\n", encoding="utf-8")
        (repo_dir / "b.py").write_text("y=2\n", encoding="utf-8")
        _run(["git", "add", "a.py", "b.py"], repo_dir)
        (repo_dir / "a.py").write_text("x=1\nunstaged_a=1\n", encoding="utf-8")
        (repo_dir / "b.py").write_text("y=2\nunstaged_b=2\n", encoding="utf-8")

        bin_dir, log_path = _fake_uv_dir(
            tmp_path,
            exit_codes=(0, 0),
            file_updates=(
                (0, "BASENAME:a.py", "x = 1\n"),
                (0, "BASENAME:b.py", "y = 2\n"),
            ),
        )
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert "fixed in the index only" in result.stderr
        assert (repo_dir / "a.py").read_text(encoding="utf-8") == (
            "x=1\nunstaged_a=1\n"
        )
        assert (repo_dir / "b.py").read_text(encoding="utf-8") == (
            "y=2\nunstaged_b=2\n"
        )
        assert _run_capture(["git", "show", ":a.py"], repo_dir).stdout == "x = 1\n"
        assert _run_capture(["git", "show", ":b.py"], repo_dir).stdout == "y = 2\n"
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_staged_handles_mix_of_fully_and_partially_staged_files(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "full.py").write_text("a = 1\n", encoding="utf-8")
        (repo_dir / "partial.py").write_text("b = 2\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "full.py").write_text("a=1\n", encoding="utf-8")
        (repo_dir / "partial.py").write_text("b=2\n", encoding="utf-8")
        _run(["git", "add", "full.py", "partial.py"], repo_dir)
        # Only partial.py has unstaged edits
        (repo_dir / "partial.py").write_text("b=2\nunstaged=3\n", encoding="utf-8")

        bin_dir, log_path = _fake_uv_dir(
            tmp_path,
            exit_codes=(0, 0),
            file_updates=(
                (0, "BASENAME:full.py", "a = 1\n"),
                (0, "BASENAME:partial.py", "b = 2\n"),
            ),
        )
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        # full.py is fully staged → worktree synced from snapshot
        assert (repo_dir / "full.py").read_text(encoding="utf-8") == "a = 1\n"
        assert _run_capture(["git", "show", ":full.py"], repo_dir).stdout == "a = 1\n"
        # partial.py has unstaged edits → worktree preserved
        assert (repo_dir / "partial.py").read_text(encoding="utf-8") == (
            "b=2\nunstaged=3\n"
        )
        assert (
            _run_capture(["git", "show", ":partial.py"], repo_dir).stdout == "b = 2\n"
        )
        assert "fixed in the index only" in result.stderr
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_staged_returns_zero_for_partial_file_without_rewrite(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "tracked.py").write_text("x=1\n", encoding="utf-8")
        _run(["git", "add", "tracked.py"], repo_dir)
        (repo_dir / "tracked.py").write_text(
            "x=1\nunstaged_value=2\n", encoding="utf-8"
        )

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(0, 0))
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert "Ruff rewrote" not in result.stderr
        assert (repo_dir / "tracked.py").read_text(encoding="utf-8") == (
            "x=1\nunstaged_value=2\n"
        )
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_staged_skips_deleted_files(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        _run(["git", "rm", "tracked.py"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(0, 0))
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert "No staged Python files found" in result.stdout
        assert not log_path.exists()

    def test_run_staged_returns_zero_when_no_python_files_staged(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "notes.md").write_text("old\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "notes.md").write_text("new\n", encoding="utf-8")
        _run(["git", "add", "notes.md"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(tmp_path)
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert "No staged Python files found" in result.stdout
        assert not log_path.exists()

    def test_run_staged_returns_nonzero_when_ruff_check_fails(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "tracked.py").write_text("x = 1\n", encoding="utf-8")
        _commit_all(repo_dir, "base")
        (repo_dir / "tracked.py").write_text("x=1\n", encoding="utf-8")
        _run(["git", "add", "tracked.py"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(1, 0))
        result = _run_pre_commit_script(
            repo_dir,
            "run-staged",
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert len(_uv_log_calls(log_path)) == 2

    @pytest.mark.parametrize("command", ["run-changed", "run-ruff"])
    @pytest.mark.parametrize("with_subdir", [False, True], ids=["root", "subdir"])
    def test_diff_commands_use_only_changed_python_paths(
        self, tmp_path: Path, command: str, with_subdir: bool
    ) -> None:
        if with_subdir:
            repo_dir, base, head = _create_changed_python_repo_with_subdir(tmp_path)
        else:
            repo_dir, base, head = _create_changed_python_repo(tmp_path)
        bin_dir, log_path = _fake_uv_dir(tmp_path)

        result = _run_diff_command(
            repo_dir,
            command,
            base=base,
            head=head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        if command == "run-changed":
            if with_subdir:
                assert log_path.read_text(encoding="utf-8").splitlines() == [
                    "tool",
                    "run",
                    "--from",
                    PRE_COMMIT_PACKAGE,
                    "pre-commit",
                    "run",
                    "--files",
                    "added.pyi",
                    "kept.py",
                    "src/__init__.py",
                    "src/utils.py",
                    "tests/test_api.pyi",
                    "tests/test_app.py",
                ]
            else:
                assert log_path.read_text(encoding="utf-8").splitlines() == [
                    "tool",
                    "run",
                    "--from",
                    PRE_COMMIT_PACKAGE,
                    "pre-commit",
                    "run",
                    "--files",
                    "added.pyi",
                    "kept.py",
                ]
            return
        if with_subdir:
            assert _uv_log_calls(log_path) == [
                [
                    "tool",
                    "run",
                    "--from",
                    RUFF_PACKAGE,
                    "ruff",
                    "check",
                    "--fix",
                    "--",
                    "added.pyi",
                    "kept.py",
                    "src/__init__.py",
                    "src/utils.py",
                    "tests/test_api.pyi",
                    "tests/test_app.py",
                ],
                [
                    "tool",
                    "run",
                    "--from",
                    RUFF_PACKAGE,
                    "ruff",
                    "format",
                    "--check",
                    "--",
                    "added.pyi",
                    "kept.py",
                    "src/__init__.py",
                    "src/utils.py",
                    "tests/test_api.pyi",
                    "tests/test_app.py",
                ],
            ]
        else:
            assert _uv_log_calls(log_path) == [
                [
                    "tool",
                    "run",
                    "--from",
                    RUFF_PACKAGE,
                    "ruff",
                    "check",
                    "--fix",
                    "--",
                    "added.pyi",
                    "kept.py",
                ],
                [
                    "tool",
                    "run",
                    "--from",
                    RUFF_PACKAGE,
                    "ruff",
                    "format",
                    "--check",
                    "--",
                    "added.pyi",
                    "kept.py",
                ],
            ]

    @pytest.mark.parametrize("command", ["run-changed", "run-ruff"])
    def test_diff_commands_merge_group_use_base_sha_directly(
        self, tmp_path: Path, command: str
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "old.py").write_text("x = 1\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")

        (repo_dir / "old.py").write_text("x = 2\n", encoding="utf-8")
        (repo_dir / "new.py").write_text("y = 3\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        bin_dir, log_path = _fake_uv_dir(tmp_path)
        result = _run_pre_commit_script(
            repo_dir,
            command,
            "--event-name",
            "merge_group",
            "--base",
            base,
            "--head",
            head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        log_lines = (
            log_path.read_text(encoding="utf-8").splitlines()
            if command == "run-changed"
            else _uv_log_calls(log_path)[0]
        )
        assert "new.py" in log_lines
        assert "old.py" in log_lines

    @pytest.mark.parametrize("command", ["run-changed", "run-ruff"])
    def test_diff_commands_return_zero_when_no_python_files_changed(
        self, tmp_path: Path, command: str
    ) -> None:
        repo_dir, base, head = _create_non_python_change_repo(tmp_path)
        bin_dir, log_path = _fake_uv_dir(tmp_path)

        result = _run_diff_command(
            repo_dir,
            command,
            base=base,
            head=head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert "No changed Python files found" in result.stdout
        if command == "run-ruff":
            assert not log_path.exists()

    @pytest.mark.parametrize("command", ["run-changed", "run-ruff"])
    def test_diff_commands_pull_request_use_merge_base_diff(
        self, tmp_path: Path, command: str
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "shared.py").write_text("print('base')\n", encoding="utf-8")
        _commit_all(repo_dir, "base")

        main_branch = _current_branch(repo_dir)
        _run(["git", "checkout", "-b", "feature"], repo_dir)
        (repo_dir / "feature.py").write_text("print('feature')\n", encoding="utf-8")
        feature_head = _commit_all(repo_dir, "feature change")

        _run(["git", "checkout", main_branch], repo_dir)
        (repo_dir / "base_only.py").write_text("print('base only')\n", encoding="utf-8")
        base_tip = _commit_all(repo_dir, "base-only change")
        _run(["git", "checkout", "feature"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(tmp_path)
        result = _run_pre_commit_script(
            repo_dir,
            command,
            "--event-name",
            "pull_request",
            "--base",
            base_tip,
            "--head",
            feature_head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        log_lines = (
            log_path.read_text(encoding="utf-8").splitlines()
            if command == "run-changed"
            else _uv_log_calls(log_path)[0]
        )
        assert log_lines[-1] == "feature.py"
        assert "base_only.py" not in log_lines

    @pytest.mark.parametrize("command", ["run-changed", "run-ruff"])
    def test_diff_commands_use_github_actions_diff_range(
        self, tmp_path: Path, command: str
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "shared.py").write_text("print('base')\n", encoding="utf-8")
        _commit_all(repo_dir, "base")

        main_branch = _current_branch(repo_dir)
        _run(["git", "checkout", "-b", "feature"], repo_dir)
        (repo_dir / "feature.py").write_text("print('one')\n", encoding="utf-8")
        _commit_all(repo_dir, "feature one")
        (repo_dir / "feature.py").write_text("print('two')\n", encoding="utf-8")
        head = _commit_all(repo_dir, "feature two")

        default_branch_sha = subprocess.run(
            ["git", "rev-parse", main_branch],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _run(
            [
                "git",
                "update-ref",
                f"refs/remotes/origin/{main_branch}",
                default_branch_sha,
            ],
            repo_dir,
        )
        _run(["git", "checkout", "feature"], repo_dir)

        bin_dir, log_path = _fake_uv_dir(tmp_path)
        env = _script_env(bin_dir)
        env.update(
            {
                "CHUNKHOUND_GITHUB_EVENT_NAME": "workflow_dispatch",
                "CHUNKHOUND_GITHUB_HEAD_SHA": head,
                "CHUNKHOUND_GITHUB_DEFAULT_BRANCH": main_branch,
            }
        )
        result = _run_diff_command(
            repo_dir,
            command,
            github_actions=True,
            env=env,
        )

        assert result.returncode == 0
        if command == "run-changed":
            assert log_path.read_text(encoding="utf-8").splitlines()[-1] == "feature.py"
            return
        assert _uv_log_calls(log_path)[0][-1] == "feature.py"

    @pytest.mark.parametrize("command", ["run-changed", "run-ruff"])
    def test_diff_commands_surface_git_failures(
        self, tmp_path: Path, command: str
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        bin_dir, _ = _fake_uv_dir(tmp_path)

        result = _run_pre_commit_script(
            repo_dir,
            command,
            "--event-name",
            "push",
            "--base",
            "deadbeef",
            "--head",
            "cafebabe",
            env=_script_env(bin_dir),
        )

        assert result.returncode != 0
        assert "Command failed: git diff" in result.stderr
        assert "ambiguous argument" in result.stderr or "bad revision" in result.stderr

    @pytest.mark.parametrize("command", ["run-changed", "run-ruff"])
    def test_diff_commands_require_explicit_range_without_github_actions(
        self, tmp_path: Path, command: str
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        bin_dir, _ = _fake_uv_dir(tmp_path)

        result = _run_pre_commit_script(repo_dir, command, env=_script_env(bin_dir))

        assert result.returncode != 0
        assert f"{command} requires --event-name, --base, and --head" in result.stderr

    @pytest.mark.parametrize("command", ["run-changed", "run-ruff"])
    def test_diff_commands_fail_when_github_actions_missing_base_sha(
        self, tmp_path: Path, command: str
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "init.py").write_text("pass\n", encoding="utf-8")
        head = _commit_all(repo_dir, "initial")
        bin_dir, _ = _fake_uv_dir(tmp_path)

        env = _script_env(bin_dir)
        env["CHUNKHOUND_GITHUB_EVENT_NAME"] = "push"
        env["CHUNKHOUND_GITHUB_BASE_SHA"] = ""
        env["CHUNKHOUND_GITHUB_HEAD_SHA"] = head

        result = _run_diff_command(
            repo_dir,
            command,
            github_actions=True,
            env=env,
        )

        assert result.returncode != 0
        assert "CHUNKHOUND_GITHUB_BASE_SHA" in result.stderr

    @pytest.mark.parametrize("command", ["run-changed", "run-ruff"])
    def test_diff_commands_fail_when_workflow_dispatch_missing_default_branch(
        self, tmp_path: Path, command: str
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "init.py").write_text("pass\n", encoding="utf-8")
        head = _commit_all(repo_dir, "initial")
        bin_dir, _ = _fake_uv_dir(tmp_path)

        env = _script_env(bin_dir)
        env["CHUNKHOUND_GITHUB_EVENT_NAME"] = "workflow_dispatch"
        env["CHUNKHOUND_GITHUB_BASE_SHA"] = ""
        env["CHUNKHOUND_GITHUB_HEAD_SHA"] = head
        env["CHUNKHOUND_GITHUB_DEFAULT_BRANCH"] = ""

        result = _run_diff_command(
            repo_dir,
            command,
            github_actions=True,
            env=env,
        )

        assert result.returncode != 0
        assert "CHUNKHOUND_GITHUB_DEFAULT_BRANCH" in result.stderr

    def test_run_ruff_returns_nonzero_when_ruff_check_fails(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "bad.py").write_text("print('old')\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")
        (repo_dir / "bad.py").write_text("print('new')\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(1, 0))
        result = _run_diff_command(
            repo_dir,
            "run-ruff",
            base=base,
            head=head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_ruff_returns_nonzero_when_ruff_format_check_fails(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "bad.py").write_text("print('old')\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")
        (repo_dir / "bad.py").write_text("print('new')\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(0, 1))
        result = _run_diff_command(
            repo_dir,
            "run-ruff",
            base=base,
            head=head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_ruff_returns_nonzero_when_ruff_check_rewrites_files(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "bad.py").write_text("print('old')\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")
        (repo_dir / "bad.py").write_text("print('new')\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        bin_dir, log_path = _fake_uv_dir(
            tmp_path,
            exit_codes=(0, 0),
            file_updates=((0, str(repo_dir / "bad.py"), "print('rewritten')\n"),),
        )
        result = _run_diff_command(
            repo_dir,
            "run-ruff",
            base=base,
            head=head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 1
        assert "Ruff rewrote changed Python files in CI" in result.stderr
        assert "bad.py" in result.stderr
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_ruff_ignores_preexisting_dirty_file_when_ruff_does_not_rewrite_it(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "bad.py").write_text("print('old')\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")
        (repo_dir / "bad.py").write_text("print('new')\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")
        (repo_dir / "bad.py").write_text("print('already-dirty')\n", encoding="utf-8")

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(0, 0))
        result = _run_diff_command(
            repo_dir,
            "run-ruff",
            base=base,
            head=head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert "Ruff rewrote changed Python files in CI" not in result.stderr
        assert len(_uv_log_calls(log_path)) == 2

    def test_run_ruff_surfaces_runtime_failures(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "bad.py").write_text("print('old')\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")
        (repo_dir / "bad.py").write_text("print('new')\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(2,))
        result = _run_diff_command(
            repo_dir,
            "run-ruff",
            base=base,
            head=head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 2
        assert "Command failed:" in result.stderr
        assert "ruff check" in result.stderr
        assert len(_uv_log_calls(log_path)) == 1

    def test_run_ruff_surfaces_runtime_failures_from_format_check(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "bad.py").write_text("print('old')\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")
        (repo_dir / "bad.py").write_text("print('new')\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        bin_dir, log_path = _fake_uv_dir(tmp_path, exit_codes=(0, 2))
        result = _run_diff_command(
            repo_dir,
            "run-ruff",
            base=base,
            head=head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 2
        assert "Command failed:" in result.stderr
        assert "ruff format --check" in result.stderr
        assert len(_uv_log_calls(log_path)) == 2

    @pytest.mark.requires_ruff_integration
    def test_run_ruff_with_real_ruff_fails_on_formatting_violation(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "good.py").write_text("x = 1\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")
        (repo_dir / "good.py").write_text("x=1\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        result = _run_diff_command(
            repo_dir,
            "run-ruff",
            base=base,
            head=head,
            env=_repo_env(tmp_path),
            timeout=120,
        )

        assert result.returncode != 0
        assert "good.py" in f"{result.stdout}\n{result.stderr}"

    @pytest.mark.requires_ruff_integration
    def test_run_ruff_with_real_ruff_fails_when_check_auto_fixes_file(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "bad.py").write_text("x = 1\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")
        (repo_dir / "bad.py").write_text("import os\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        result = _run_diff_command(
            repo_dir,
            "run-ruff",
            base=base,
            head=head,
            env=_repo_env(tmp_path),
            timeout=120,
        )

        assert result.returncode != 0
        assert "Ruff rewrote changed Python files in CI" in result.stderr
        assert "bad.py" in result.stderr

    def test_install_hook_rewrites_file_and_blocks_commit(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        _copy_pre_commit_config(repo_dir)
        _copy_pre_commit_script(repo_dir)
        (repo_dir / "README.md").write_text("base\n", encoding="utf-8")
        _commit_all(repo_dir, "base")

        install = _run_pre_commit_script(
            repo_dir,
            "install",
            env=_repo_env(tmp_path),
            timeout=120,
        )
        assert install.returncode == 0, install.stderr

        bad_file = repo_dir / "bad.py"
        bad_file.write_text("x=1\n", encoding="utf-8")
        _run(["git", "add", "bad.py"], repo_dir)
        failed_commit = _run_capture(
            ["git", "commit", "-m", "lint me"],
            repo_dir,
            env=_repo_env(tmp_path),
        )

        assert failed_commit.returncode != 0
        assert bad_file.read_text(encoding="utf-8") == "x = 1\n"
        worktree_diff = _run_capture(["git", "diff", "--", "bad.py"], repo_dir)
        assert worktree_diff.stdout == ""
        staged_diff = _run_capture(
            ["git", "diff", "--cached", "--", "bad.py"],
            repo_dir,
        )
        assert staged_diff.stdout.strip() != ""

        successful_commit = _run_capture(
            ["git", "commit", "-m", "lint me"], repo_dir, env=_repo_env(tmp_path)
        )
        assert successful_commit.returncode == 0, successful_commit.stderr

    def test_run_changed_with_real_pre_commit_fails_on_unfixable_python_error(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        _copy_pre_commit_config(repo_dir)
        (repo_dir / "good.py").write_text("x = 1\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")
        (repo_dir / "good.py").write_text("def broken(:\n", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        result = _run_pre_commit_script(
            repo_dir,
            "run-changed",
            "--event-name",
            "push",
            "--base",
            base,
            "--head",
            head,
            env=_repo_env(tmp_path),
            timeout=120,
        )

        assert result.returncode != 0
        assert "good.py" in f"{result.stdout}\n{result.stderr}"

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from tests.utils.windows_subprocess import get_safe_subprocess_env

ROOT = Path(__file__).resolve().parents[2]
PRE_COMMIT_SCRIPT = ROOT / "scripts" / "pre_commit.py"


def _load_pre_commit_package_version() -> str:
    spec = importlib.util.spec_from_file_location("_pre_commit_mod", PRE_COMMIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PRE_COMMIT_PACKAGE  # type: ignore[attr-defined]


PRE_COMMIT_PACKAGE = _load_pre_commit_package_version()
HOOK_MARKER = "chunkhound-managed-pre-commit-hook"
_SUBPROCESS_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SystemRoot",
    "ComSpec",
    "PATHEXT",
    "APPDATA",
    "LOCALAPPDATA",
    "SHELL",
)


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


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


def _create_repo(repo_dir: Path) -> None:
    _run(["git", "init"], repo_dir)
    _run(["git", "config", "user.name", "ChunkHound Tests"], repo_dir)
    _run(["git", "config", "user.email", "tests@chunkhound.invalid"], repo_dir)


def _commit_all(repo_dir: Path, message: str) -> str:
    _run(["git", "add", "-A"], repo_dir)
    _run(["git", "commit", "-m", message], repo_dir)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _current_branch(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fake_uv_dir(tmp_path: Path, *, exit_code: int = 0) -> tuple[Path, Path]:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log_path = tmp_path / "uv.log"
    python = sys.executable.replace("\\", "\\\\")
    script = "\n".join(
        [
            "import pathlib, sys",
            (
                f"pathlib.Path({str(log_path)!r}).write_text("
                "'\\n'.join(sys.argv[1:]) + '\\n', encoding='utf-8')"
            ),
            f"raise SystemExit({exit_code})",
            "",
        ]
    )

    if os.name == "nt":
        (bin_dir / "uv.cmd").write_text(
            f'@"{sys.executable}" -c "{script}" %*\r\n',
            encoding="utf-8",
        )
    else:
        uv_path = bin_dir / "uv"
        uv_path.write_text(f"#!{python}\n{script}", encoding="utf-8")
        uv_path.chmod(0o755)

    return bin_dir, log_path


def _script_env(bin_dir: Path) -> dict[str, str]:
    base_env = {
        key: os.environ[key] for key in _SUBPROCESS_ENV_ALLOWLIST if key in os.environ
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


def test_pre_commit_ruff_hook_contract() -> None:
    config = cast(
        dict[str, Any],
        yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")),
    )

    repos = cast(list[dict[str, Any]], config["repos"])
    assert len(repos) == 1
    repo = repos[0]
    assert repo["repo"] == "https://github.com/astral-sh/ruff-pre-commit"

    hooks = cast(list[dict[str, Any]], repo["hooks"])
    assert [hook["id"] for hook in hooks] == ["ruff-check", "ruff-format"]
    assert hooks[0]["args"] == ["--fix"]
    assert hooks[0]["types_or"] == ["python", "pyi"]
    assert hooks[1]["types_or"] == ["python", "pyi"]


def test_install_hooks_make_target_contract() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "install-hooks:\n\tuv run python scripts/pre_commit.py install\n" in makefile


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
        assert log_path.read_text(encoding="utf-8").splitlines() == [
            "tool",
            "run",
            "--from",
            PRE_COMMIT_PACKAGE,
            "pre-commit",
            "run",
            "--files",
            "added.pyi",
            "tracked.py",
        ]

    def test_run_changed_uses_only_changed_python_paths(self, tmp_path: Path) -> None:
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

        bin_dir, log_path = _fake_uv_dir(tmp_path)
        result = _run_pre_commit_script(
            repo_dir,
            "run-changed",
            "--event-name",
            "push",
            "--base",
            base,
            "--head",
            head,
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
            "added.pyi",
            "kept.py",
        ]

    def test_run_changed_merge_group_uses_base_sha_directly(
        self, tmp_path: Path
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
            "run-changed",
            "--event-name",
            "merge_group",
            "--base",
            base,
            "--head",
            head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        log_lines = log_path.read_text(encoding="utf-8").splitlines()
        assert "new.py" in log_lines
        assert "old.py" in log_lines

    def test_run_changed_returns_zero_when_no_python_files_changed(
        self, tmp_path: Path
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        (repo_dir / "notes.md").write_text("old\n", encoding="utf-8")
        base = _commit_all(repo_dir, "base")

        (repo_dir / "notes.md").write_text("new\n", encoding="utf-8")
        (repo_dir / "data.json").write_text("{}", encoding="utf-8")
        head = _commit_all(repo_dir, "head")

        bin_dir, _ = _fake_uv_dir(tmp_path)
        result = _run_pre_commit_script(
            repo_dir,
            "run-changed",
            "--event-name",
            "push",
            "--base",
            base,
            "--head",
            head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert "No changed Python files found" in result.stdout

    def test_run_changed_pull_request_uses_merge_base_diff(
        self, tmp_path: Path
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
            "run-changed",
            "--event-name",
            "pull_request",
            "--base",
            base_tip,
            "--head",
            feature_head,
            env=_script_env(bin_dir),
        )

        assert result.returncode == 0
        assert log_path.read_text(encoding="utf-8").splitlines()[-1] == "feature.py"
        assert "base_only.py" not in log_path.read_text(encoding="utf-8")

    def test_workflow_dispatch_uses_default_branch_merge_base(
        self, tmp_path: Path
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
        result = _run_pre_commit_script(
            repo_dir,
            "run-changed",
            "--github-actions",
            env=env,
        )

        assert result.returncode == 0
        assert log_path.read_text(encoding="utf-8").splitlines()[-1] == "feature.py"

    def test_run_changed_surfaces_git_failures(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _create_repo(repo_dir)
        bin_dir, _ = _fake_uv_dir(tmp_path)

        result = _run_pre_commit_script(
            repo_dir,
            "run-changed",
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

    def test_run_changed_fails_when_github_actions_missing_base_sha(
        self, tmp_path: Path
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

        result = _run_pre_commit_script(
            repo_dir,
            "run-changed",
            "--github-actions",
            env=env,
        )

        assert result.returncode != 0
        assert "CHUNKHOUND_GITHUB_BASE_SHA" in result.stderr

    def test_run_changed_fails_when_workflow_dispatch_missing_default_branch(
        self, tmp_path: Path
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

        result = _run_pre_commit_script(
            repo_dir,
            "run-changed",
            "--github-actions",
            env=env,
        )

        assert result.returncode != 0
        assert "CHUNKHOUND_GITHUB_DEFAULT_BRANCH" in result.stderr

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
        diff_after_rewrite = _run_capture(["git", "diff", "--", "bad.py"], repo_dir)
        assert diff_after_rewrite.stdout.strip() != ""

        _run(["git", "add", "bad.py"], repo_dir)
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

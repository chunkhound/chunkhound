from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]

from tests.utils.windows_subprocess import get_safe_subprocess_env

ROOT = Path(__file__).resolve().parents[2]
RESOLVER = ROOT / "scripts" / "resolve_docs_version.sh"
INLINE_RESOLVE_SNIPPET = (
    "CHUNKHOUND_DOCS_VERSION=$(git describe --tags --abbrev=0 | sed 's/^v//')"
)
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


def _is_windows() -> bool:
    return os.name == "nt"


def _git_bash_from_git_path(git_path: str) -> str | None:
    git_executable = Path(git_path).resolve()
    candidates = [
        git_executable.parent / "bash.exe",
        git_executable.parent.parent / "bin" / "bash.exe",
        git_executable.parent.parent / "usr" / "bin" / "bash.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _is_trusted_windows_bash_path(bash_path: str) -> bool:
    bash_executable = Path(bash_path).resolve()
    if bash_executable.name.lower() != "bash.exe":
        return False

    normalized_parts = tuple(part.lower() for part in bash_executable.parts)
    trusted_layouts = (
        ("git", "cmd", "bash.exe"),
        ("git", "bin", "bash.exe"),
        ("git", "usr", "bin", "bash.exe"),
        ("msys64", "usr", "bin", "bash.exe"),
        ("mingw64", "bin", "bash.exe"),
    )
    return any(normalized_parts[-len(layout) :] == layout for layout in trusted_layouts)


def _resolver_command() -> list[str]:
    if not _is_windows():
        return ["bash", str(RESOLVER)]

    git_path = shutil.which("git")
    if git_path is not None:
        git_bash = _git_bash_from_git_path(git_path)
        if git_bash is not None and _is_trusted_windows_bash_path(git_bash):
            return [git_bash, str(RESOLVER)]

    bash_path = shutil.which("bash")
    if bash_path is not None and _is_trusted_windows_bash_path(bash_path):
        return [bash_path, str(RESOLVER)]

    pytest.skip(
        "Skipping docs resolver contract test on Windows: no trusted Windows "
        "bash was found."
    )


def _resolver_env() -> dict[str, str]:
    base_env = {
        key: os.environ[key] for key in _SUBPROCESS_ENV_ALLOWLIST if key in os.environ
    }
    return get_safe_subprocess_env(base_env)


def _run_resolver(
    repo_dir: Path,
    *,
    check: bool,
    env_updates: dict[str, str] | None = None,
    env_remove: set[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _resolver_env()
    if env_remove:
        for key in env_remove:
            env.pop(key, None)
    if env_updates:
        env.update(env_updates)

    return subprocess.run(
        _resolver_command(),
        cwd=repo_dir,
        check=check,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )


def _create_tagged_repo(repo_dir: Path, version_tag: str) -> None:
    _run(["git", "init"], repo_dir)
    _run(["git", "config", "user.name", "ChunkHound Tests"], repo_dir)
    _run(["git", "config", "user.email", "tests@chunkhound.invalid"], repo_dir)
    (repo_dir / "README.md").write_text("test\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo_dir)
    _run(["git", "commit", "-m", "initial"], repo_dir)
    _run(["git", "tag", version_tag], repo_dir)


def _load_workflow(path: str) -> dict[str, Any]:
    with (ROOT / path).open(encoding="utf-8") as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))


def _job_steps(path: str, job_name: str) -> list[dict[str, Any]]:
    workflow = _load_workflow(path)
    return cast(list[dict[str, Any]], workflow["jobs"][job_name]["steps"])


def _find_step_index(
    steps: list[dict], description: str, predicate: Callable[[dict], bool]
) -> int:
    for index, step in enumerate(steps):
        if predicate(step):
            return index
    raise AssertionError(f"Could not find step matching {description}")


def _assert_checkout_uses_full_history(steps: list[dict]) -> None:
    checkout_index = _find_step_index(
        steps,
        "actions/checkout step",
        lambda step: str(step.get("uses", "")).startswith("actions/checkout@"),
    )
    checkout_step = steps[checkout_index]
    assert checkout_step.get("with", {}).get("fetch-depth") == 0


class TestResolveDocsVersionScript:
    def test_exports_normalized_version_to_github_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            _create_tagged_repo(repo_dir, "v4.1.0b1")
            github_env = repo_dir / "github.env"
            github_env.touch()

            _run_resolver(
                repo_dir,
                check=True,
                env_updates={"GITHUB_ENV": str(github_env)},
            )

            exported = github_env.read_text(encoding="utf-8")

        assert exported == "CHUNKHOUND_DOCS_VERSION=4.1.0b1\n"

    def test_fails_without_git_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            _run(["git", "init"], repo_dir)
            _run(["git", "config", "user.name", "ChunkHound Tests"], repo_dir)
            _run(["git", "config", "user.email", "tests@chunkhound.invalid"], repo_dir)
            (repo_dir / "README.md").write_text("test\n", encoding="utf-8")
            _run(["git", "add", "README.md"], repo_dir)
            _run(["git", "commit", "-m", "initial"], repo_dir)
            github_env = repo_dir / "github.env"
            github_env.touch()

            result = _run_resolver(
                repo_dir,
                check=False,
                env_updates={"GITHUB_ENV": str(github_env)},
            )

        assert result.returncode != 0
        assert "Unable to resolve docs version" in result.stderr

    def test_fails_without_github_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            _create_tagged_repo(repo_dir, "v4.1.0")

            result = _run_resolver(repo_dir, check=False, env_remove={"GITHUB_ENV"})

        assert result.returncode != 0
        assert "GITHUB_ENV must be set" in result.stderr


class TestResolverCommand:
    def test_non_windows_uses_bash_directly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = sys.modules[__name__]
        monkeypatch.setattr(module, "_is_windows", lambda: False)

        assert _resolver_command() == ["bash", str(RESOLVER)]

    def test_windows_prefers_git_derived_bash_over_path_bash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        git_executable = tmp_path / "Git" / "cmd" / "git.exe"
        git_executable.parent.mkdir(parents=True)
        git_executable.touch()
        derived_bash = tmp_path / "Git" / "bin" / "bash.exe"
        derived_bash.parent.mkdir(parents=True)
        derived_bash.touch()
        path_bash = "C:/Windows/System32/bash.exe"

        module = sys.modules[__name__]
        monkeypatch.setattr(module, "_is_windows", lambda: True)
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: (
                str(git_executable)
                if name == "git"
                else path_bash
                if name == "bash"
                else None
            ),
        )

        assert _resolver_command() == [str(derived_bash), str(RESOLVER)]

    def test_windows_skips_when_git_derived_bash_is_untrusted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        git_executable = tmp_path / "PortableGit" / "cmd" / "git.exe"
        git_executable.parent.mkdir(parents=True)
        git_executable.touch()
        untrusted_git_bash = tmp_path / "PortableGit" / "embedded" / "bash.exe"
        untrusted_git_bash.parent.mkdir(parents=True)
        untrusted_git_bash.touch()

        module = sys.modules[__name__]
        monkeypatch.setattr(module, "_is_windows", lambda: True)
        monkeypatch.setattr(
            module, "_git_bash_from_git_path", lambda path: str(untrusted_git_bash)
        )
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: str(git_executable) if name == "git" else None,
        )

        with pytest.raises(pytest.skip.Exception, match="no trusted Windows bash"):
            _resolver_command()

    def test_windows_accepts_trusted_path_bash_when_git_bash_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path_bash = tmp_path / "msys64" / "usr" / "bin" / "bash.exe"
        path_bash.parent.mkdir(parents=True)
        path_bash.touch()

        module = sys.modules[__name__]
        monkeypatch.setattr(module, "_is_windows", lambda: True)
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: str(path_bash) if name == "bash" else None,
        )

        assert _resolver_command() == [str(path_bash), str(RESOLVER)]

    def test_windows_falls_back_to_trusted_path_bash_when_git_derived_bash_is_untrusted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        git_executable = tmp_path / "PortableGit" / "cmd" / "git.exe"
        git_executable.parent.mkdir(parents=True)
        git_executable.touch()
        untrusted_git_bash = tmp_path / "PortableGit" / "embedded" / "bash.exe"
        untrusted_git_bash.parent.mkdir(parents=True)
        untrusted_git_bash.touch()
        trusted_path_bash = tmp_path / "msys64" / "usr" / "bin" / "bash.exe"
        trusted_path_bash.parent.mkdir(parents=True)
        trusted_path_bash.touch()

        module = sys.modules[__name__]
        monkeypatch.setattr(module, "_is_windows", lambda: True)
        monkeypatch.setattr(
            module, "_git_bash_from_git_path", lambda path: str(untrusted_git_bash)
        )
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: (
                str(git_executable)
                if name == "git"
                else str(trusted_path_bash)
                if name == "bash"
                else None
            ),
        )

        assert _resolver_command() == [str(trusted_path_bash), str(RESOLVER)]

    def test_windows_skips_without_runnable_bash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        git_executable = tmp_path / "Git" / "cmd" / "git.exe"
        git_executable.parent.mkdir(parents=True)
        git_executable.touch()
        untrusted_bash = tmp_path / "Windows" / "System32" / "bash.exe"
        untrusted_bash.parent.mkdir(parents=True)
        untrusted_bash.touch()

        module = sys.modules[__name__]
        monkeypatch.setattr(module, "_is_windows", lambda: True)
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name: (
                str(git_executable)
                if name == "git"
                else str(untrusted_bash)
                if name == "bash"
                else None
            ),
        )

        with pytest.raises(pytest.skip.Exception, match="no trusted Windows bash"):
            _resolver_command()

    def test_resolver_env_allowlists_variables_and_uses_safe_subprocess_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = sys.modules[__name__]
        allowlisted_key = "TMPDIR"
        allowlisted_value = "C:/temp/chunkhound"
        blocked_key = "CHUNKHOUND_UNTRUSTED_ENV"
        blocked_value = "should-not-leak"
        observed: dict[str, str] = {}

        monkeypatch.setenv(allowlisted_key, allowlisted_value)
        monkeypatch.setenv(blocked_key, blocked_value)

        def fake_get_safe_subprocess_env(base_env: dict[str, str]) -> dict[str, str]:
            observed.update(base_env)
            return {**base_env, "SAFE_SUBPROCESS_ENV": "1"}

        monkeypatch.setattr(
            module, "get_safe_subprocess_env", fake_get_safe_subprocess_env
        )

        env = _resolver_env()

        assert observed[allowlisted_key] == allowlisted_value
        assert blocked_key not in observed
        assert env[allowlisted_key] == allowlisted_value
        assert blocked_key not in env
        assert env["SAFE_SUBPROCESS_ENV"] == "1"


class TestDocsVersionWorkflowContract:
    @pytest.mark.parametrize(
        ("path", "job_name"),
        [
            (".github/workflows/smoke-tests.yml", "site-build"),
            (".github/workflows/smoke-tests.yml", "tests"),
            (".github/workflows/deploy.yml", "build"),
        ],
    )
    def test_docs_build_jobs_use_shared_resolver_script(
        self, path: str, job_name: str
    ) -> None:
        steps = _job_steps(path, job_name)
        resolve_index = _find_step_index(
            steps,
            "shared docs version resolver step",
            lambda step: step.get("run") == "bash scripts/resolve_docs_version.sh",
        )
        resolve_step = steps[resolve_index]

        assert resolve_step["run"] == "bash scripts/resolve_docs_version.sh"

    @pytest.mark.parametrize(
        ("path", "job_name"),
        [
            (".github/workflows/smoke-tests.yml", "site-build"),
            (".github/workflows/smoke-tests.yml", "tests"),
            (".github/workflows/deploy.yml", "build"),
        ],
    )
    def test_docs_build_jobs_checkout_with_full_history(
        self, path: str, job_name: str
    ) -> None:
        steps = _job_steps(path, job_name)
        _assert_checkout_uses_full_history(steps)

    @pytest.mark.parametrize(
        ("path", "job_name", "consumer_description", "consumer_predicate"),
        [
            (
                ".github/workflows/smoke-tests.yml",
                "site-build",
                "site build step",
                lambda step: step.get("run") == "npm run build --prefix site",
            ),
            (
                ".github/workflows/smoke-tests.yml",
                "tests",
                "retry-backed test runner step",
                # This contract protects the real test runner, not the collect-only
                # check, so workflow refactors do not accidentally weaken it.
                lambda step: step.get("id") == "tests"
                and str(step.get("uses", "")).startswith("nick-fields/retry@"),
            ),
            (
                ".github/workflows/deploy.yml",
                "build",
                "site build step",
                lambda step: step.get("run") == "npm run build --prefix site",
            ),
        ],
    )
    def test_docs_version_is_resolved_before_consumer_steps(
        self,
        path: str,
        job_name: str,
        consumer_description: str,
        consumer_predicate: Callable[[dict], bool],
    ) -> None:
        steps = _job_steps(path, job_name)
        resolve_index = _find_step_index(
            steps,
            "shared docs version resolver step",
            lambda step: step.get("run") == "bash scripts/resolve_docs_version.sh",
        )
        consumer_index = _find_step_index(
            steps, consumer_description, consumer_predicate
        )

        assert resolve_index < consumer_index

    @pytest.mark.parametrize(
        "path",
        [
            ".github/workflows/smoke-tests.yml",
            ".github/workflows/deploy.yml",
        ],
    )
    def test_workflows_do_not_inline_docs_version_resolution(self, path: str) -> None:
        contents = (ROOT / path).read_text(encoding="utf-8")

        assert INLINE_RESOLVE_SNIPPET not in contents

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


def _job(path: str, job_name: str) -> dict[str, Any]:
    workflow = _load_workflow(path)
    return cast(dict[str, Any], workflow["jobs"][job_name])


def _job_steps(path: str, job_name: str) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _job(path, job_name)["steps"])


def _composite_action_steps(path: str) -> list[dict[str, Any]]:
    action = _load_workflow(path)
    return cast(list[dict[str, Any]], action["runs"]["steps"])


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
            (".github/workflows/ci.yml", "site-build"),
            (".github/workflows/ci.yml", "site-build-validation"),
            (".github/workflows/ci.yml", "tests"),
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
            (".github/workflows/ci.yml", "site-build"),
            (".github/workflows/ci.yml", "site-build-validation"),
            (".github/workflows/ci.yml", "tests"),
        ],
    )
    def test_docs_build_jobs_checkout_with_full_history(
        self, path: str, job_name: str
    ) -> None:
        _assert_checkout_uses_full_history(_job_steps(path, job_name))

    @pytest.mark.parametrize(
        ("job_name", "consumer_description", "consumer_predicate"),
        [
            (
                "site-build",
                "site build step",
                lambda step: step.get("run") == "npm run build --prefix site",
            ),
            (
                "site-build-validation",
                "site build step",
                lambda step: step.get("run") == "npm run build --prefix site",
            ),
            (
                "tests",
                "retry-backed test runner step",
                lambda step: step.get("id") == "tests"
                and str(step.get("uses", "")).startswith("nick-fields/retry@"),
            ),
        ],
    )
    def test_docs_version_is_resolved_before_consumer_steps(
        self,
        job_name: str,
        consumer_description: str,
        consumer_predicate: Callable[[dict], bool],
    ) -> None:
        steps = _job_steps(".github/workflows/ci.yml", job_name)
        resolve_index = _find_step_index(
            steps,
            "shared docs version resolver step",
            lambda step: step.get("run") == "bash scripts/resolve_docs_version.sh",
        )
        consumer_index = _find_step_index(
            steps, consumer_description, consumer_predicate
        )

        assert resolve_index < consumer_index

    def test_workflows_do_not_inline_docs_version_resolution(self) -> None:
        contents = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

        assert INLINE_RESOLVE_SNIPPET not in contents

    def test_site_build_uploads_site_dist_artifact_after_nojekyll(self) -> None:
        steps = _job_steps(".github/workflows/ci.yml", "site-build")
        nojekyll_index = _find_step_index(
            steps,
            ".nojekyll creation step",
            lambda step: step.get("run") == "touch site/dist/.nojekyll",
        )
        upload_index = _find_step_index(
            steps,
            "site artifact upload step",
            lambda step: str(step.get("uses", "")).startswith(
                "actions/upload-artifact@"
            ),
        )
        upload_step = steps[upload_index]

        assert nojekyll_index < upload_index
        assert upload_step["with"]["name"] == "site-dist"
        assert upload_step["with"]["path"] == "site/dist/"

    def test_tests_job_downloads_site_dist_and_uses_existing_build(self) -> None:
        job = _job(".github/workflows/ci.yml", "tests")
        steps = cast(list[dict[str, Any]], job["steps"])
        download_index = _find_step_index(
            steps,
            "site artifact download step",
            lambda step: str(step.get("uses", "")).startswith(
                "actions/download-artifact@"
            ),
        )
        download_step = steps[download_index]
        tests_index = _find_step_index(
            steps,
            "retry-backed test runner step",
            lambda step: step.get("id") == "tests"
            and str(step.get("uses", "")).startswith("nick-fields/retry@"),
        )
        tests_step = steps[tests_index]

        assert job["needs"] == "site-build"
        assert download_step["with"]["name"] == "site-dist"
        assert download_step["with"]["path"] == "site/dist"
        assert tests_step["env"]["CHUNKHOUND_USE_EXISTING_SITE_DIST"] == "1"

    def test_tests_job_matrix_uses_pytest_timeout_minutes_consistently(self) -> None:
        matrix = cast(
            list[dict[str, Any]],
            _job(".github/workflows/ci.yml", "tests")["strategy"]["matrix"]["include"],
        )

        assert all("pytest_timeout_minutes" in entry for entry in matrix)
        assert all("retry_timeout_minutes" not in entry for entry in matrix)

    def test_site_build_validation_job_contract(self) -> None:
        job = _job(".github/workflows/ci.yml", "site-build-validation")
        steps = cast(list[dict[str, Any]], job["steps"])
        matrix = cast(dict[str, Any], job["strategy"]["matrix"])

        assert matrix["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
        assert job["runs-on"] == "${{ matrix.os }}"
        assert not any(
            str(step.get("uses", "")).startswith("actions/upload-artifact@")
            for step in steps
        )

    def test_ruff_integration_validation_job_contract(self) -> None:
        job = _job(".github/workflows/ci.yml", "ruff-integration-validation")
        steps = cast(list[dict[str, Any]], job["steps"])
        test_index = _find_step_index(
            steps,
            "ruff integration pytest step",
            lambda step: step.get("id") == "tests"
            and step.get("name") == "Run real uv+ruff integration contract tests",
        )
        upload_index = _find_step_index(
            steps,
            "test results upload step",
            lambda step: str(step.get("uses", "")).startswith(
                "actions/upload-artifact@"
            ),
        )
        flaky_index = _find_step_index(
            steps,
            "flaky annotations check step",
            lambda step: step.get("run")
            == "uv run python scripts/check_flaky_annotations.py test-results.xml",
        )
        test_step = steps[test_index]
        upload_step = steps[upload_index]
        flaky_step = steps[flaky_index]

        assert job["runs-on"] == "ubuntu-latest"
        assert job["timeout-minutes"] == 20
        assert cast(dict[str, str], job["env"])["CHUNKHOUND_RUN_RUFF_INTEGRATION"] == "1"
        assert test_step["run"] == (
            "uv run pytest tests/unit/test_python_lint_tooling_contract.py "
            "-m requires_ruff_integration -v --junit-xml=test-results.xml"
        )
        assert upload_step["with"]["name"] == "test-results-ruff-integration-validation"
        assert upload_step["with"]["path"] == "test-results.xml"
        assert flaky_step["if"] == "failure() && steps.tests.outcome == 'failure'"
        assert test_index < upload_index < flaky_index

    def test_pages_artifact_job_contract(self) -> None:
        job = _job(".github/workflows/ci.yml", "pages-artifact")
        steps = cast(list[dict[str, Any]], job["steps"])
        download_index = _find_step_index(
            steps,
            "tested site artifact download step",
            lambda step: str(step.get("uses", "")).startswith(
                "actions/download-artifact@"
            ),
        )
        upload_index = _find_step_index(
            steps,
            "pages artifact upload step",
            lambda step: str(step.get("uses", "")).startswith(
                "actions/upload-pages-artifact@"
            ),
        )
        download_step = steps[download_index]
        upload_step = steps[upload_index]
        permissions = cast(dict[str, Any], job.get("permissions", {}))

        assert job["needs"] == [
            "lint-changed-ruff",
            "site-build",
            "tests",
            "site-build-validation",
            "watchman-rollout-gate",
            "ruff-integration-validation",
        ]
        assert job["if"] == "github.ref == 'refs/heads/main'"
        assert download_step["with"]["name"] == "site-dist"
        assert download_step["with"]["path"] == "site/dist"
        assert upload_step["with"]["path"] == "site/dist/"
        assert permissions["contents"] == "read"

    def test_workflow_concurrency_contract(self) -> None:
        workflow = _load_workflow(".github/workflows/ci.yml")
        concurrency = cast(dict[str, Any], workflow.get("concurrency", {}))

        expected_group = (
            "${{ github.ref == 'refs/heads/main' && "
            "'ch-test-deploy' || github.run_id }}"
        )

        assert concurrency["group"] == expected_group
        assert concurrency["cancel-in-progress"] is False

    def test_deploy_job_contract(self) -> None:
        job = _job(".github/workflows/ci.yml", "deploy")
        permissions = cast(dict[str, Any], job.get("permissions", {}))

        assert job["needs"] == "pages-artifact"
        assert job["if"] == "github.ref == 'refs/heads/main'"
        assert "concurrency" not in job
        assert permissions["contents"] == "read"
        assert permissions["pages"] == "write"
        assert permissions["id-token"] == "write"

    def test_changed_python_lint_job_contract(self) -> None:
        workflow = _load_workflow(".github/workflows/ci.yml")
        triggers = cast(dict[str, Any], workflow.get("on", workflow.get(True, {})))
        job = _job(".github/workflows/ci.yml", "lint-changed-ruff")
        steps = cast(list[dict[str, Any]], job["steps"])

        assert "pull_request" in triggers
        assert "push" in triggers
        assert "merge_group" in triggers
        assert "workflow_dispatch" in triggers
        assert job["runs-on"] == "ubuntu-latest"
        assert job["timeout-minutes"] == 10

        _assert_checkout_uses_full_history(steps)

        python_index = _find_step_index(
            steps,
            "setup-python step",
            lambda step: str(step.get("uses", "")).startswith("actions/setup-python@"),
        )
        uv_index = _find_step_index(
            steps,
            "setup-uv step",
            lambda step: str(step.get("uses", "")).startswith("astral-sh/setup-uv@"),
        )
        ruff_index = _find_step_index(
            steps,
            "changed python ruff step",
            lambda step: step.get("run")
            == "python scripts/pre_commit.py run-ruff --github-actions",
        )
        ruff_step = steps[ruff_index]
        ruff_env = cast(dict[str, str], ruff_step["env"])

        assert python_index < uv_index < ruff_index
        assert ruff_step["run"] == (
            "python scripts/pre_commit.py run-ruff --github-actions"
        )
        assert ruff_env["CHUNKHOUND_GITHUB_EVENT_NAME"] == ("${{ github.event_name }}")
        assert ruff_env["CHUNKHOUND_GITHUB_BASE_SHA"] == (
            "${{ github.event.pull_request.base.sha || "
            "github.event.merge_group.base_sha || github.event.before || '' }}"
        )
        assert ruff_env["CHUNKHOUND_GITHUB_HEAD_SHA"] == (
            "${{ github.event.pull_request.head.sha || "
            "github.event.merge_group.head_sha || github.sha }}"
        )
        assert ruff_env["CHUNKHOUND_GITHUB_DEFAULT_BRANCH"] == (
            "${{ github.event.repository.default_branch || '' }}"
        )


class TestTestTriageCompositeActionContract:
    def test_declares_explicit_flaky_annotation_gate_input(self) -> None:
        action = _load_workflow(".github/actions/test-triage/action.yml")
        flaky_gate = cast(dict[str, Any], action["inputs"]["check-flaky-annotations"])

        assert flaky_gate["required"] is False
        assert flaky_gate["default"] == "false"

    def test_uploads_results_before_optional_flaky_annotation_check(self) -> None:
        steps = _composite_action_steps(".github/actions/test-triage/action.yml")
        upload_index = _find_step_index(
            steps,
            "upload test results step",
            lambda step: str(step.get("uses", "")).startswith(
                "actions/upload-artifact@"
            ),
        )
        flaky_index = _find_step_index(
            steps,
            "flaky annotation check step",
            lambda step: step.get("name") == "Check flaky annotations",
        )
        upload_step = steps[upload_index]
        flaky_step = steps[flaky_index]

        assert upload_step["if"] == "always()"
        assert upload_step["with"]["if-no-files-found"] == "ignore"
        assert flaky_step["if"] == "${{ inputs.check-flaky-annotations == 'true' }}"
        expected_command = (
            "uv run python scripts/check_flaky_annotations.py "
            "${{ inputs.results-path }}"
        )
        assert flaky_step["run"] == expected_command
        assert upload_index < flaky_index

    def test_does_not_reach_into_caller_step_state(self) -> None:
        contents = (ROOT / ".github/actions/test-triage/action.yml").read_text(
            encoding="utf-8"
        )

        assert "steps.tests.outcome" not in contents

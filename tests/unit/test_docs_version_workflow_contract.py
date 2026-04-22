from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESOLVER = ROOT / "scripts" / "resolve_docs_version.sh"
INLINE_RESOLVE_SNIPPET = (
    "CHUNKHOUND_DOCS_VERSION=$(git describe --tags --abbrev=0 | sed 's/^v//')"
)


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _create_tagged_repo(repo_dir: Path, version_tag: str) -> None:
    _run(["git", "init"], repo_dir)
    _run(["git", "config", "user.name", "ChunkHound Tests"], repo_dir)
    _run(["git", "config", "user.email", "tests@chunkhound.invalid"], repo_dir)
    (repo_dir / "README.md").write_text("test\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo_dir)
    _run(["git", "commit", "-m", "initial"], repo_dir)
    _run(["git", "tag", version_tag], repo_dir)


def _load_workflow(path: str) -> dict:
    with (ROOT / path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _job_steps(path: str, job_name: str) -> list[dict]:
    workflow = _load_workflow(path)
    return workflow["jobs"][job_name]["steps"]


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

            with tempfile.NamedTemporaryFile() as github_env:
                subprocess.run(
                    ["bash", str(RESOLVER)],
                    cwd=repo_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "GITHUB_ENV": github_env.name},
                )

                exported = Path(github_env.name).read_text(encoding="utf-8")

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

            with tempfile.NamedTemporaryFile() as github_env:
                result = subprocess.run(
                    ["bash", str(RESOLVER)],
                    cwd=repo_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "GITHUB_ENV": github_env.name},
                )

        assert result.returncode != 0
        assert "Unable to resolve docs version" in result.stderr

    def test_fails_without_github_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            _create_tagged_repo(repo_dir, "v4.1.0")

            result = subprocess.run(
                ["bash", str(RESOLVER)],
                cwd=repo_dir,
                check=False,
                capture_output=True,
                text=True,
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key != "GITHUB_ENV"
                },
            )

        assert result.returncode != 0
        assert "GITHUB_ENV must be set" in result.stderr


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
                "first pytest consumer step",
                lambda step: "pytest" in str(step.get("run", ""))
                or "pytest" in str(step.get("with", {}).get("command", "")),
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
        consumer_index = _find_step_index(steps, consumer_description, consumer_predicate)

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

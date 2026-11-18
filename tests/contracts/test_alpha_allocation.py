"""Automatic alpha releases are immutable, per-build, and exact-source Git refs."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/create_next_alpha_tag.py"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repos(tmp_path):
    remote = tmp_path / "remote.git"
    repo = tmp_path / "source"
    git(tmp_path, "init", "--bare", str(remote))
    git(tmp_path, "init", str(repo))
    git(repo, "config", "user.name", "Contract Test")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "commit", "--allow-empty", "-m", "released stable")
    git(repo, "tag", "-a", "v6.1.0", "-m", "stable")
    git(repo, "commit", "--allow-empty", "-m", "tested source")
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "origin", "HEAD:refs/heads/main", "refs/tags/v6.1.0")
    return repo, remote, sha


def allocate(repo, sha, run_id="101"):
    output = repo / f"output-{run_id}.txt"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository",
            "chunkhound/chunkhound",
            "--run-id",
            run_id,
            "--source-sha",
            sha,
        ],
        cwd=repo,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )


def test_next_alpha_uses_remote_tags_and_exact_tested_sha(repos):
    repo, remote, sha = repos
    git(repo, "tag", "v6.2.0a1")
    git(repo, "tag", "v6.2.0a3")
    git(repo, "push", "origin", "--tags")
    git(repo, "tag", "v6.2.0a99")  # stale local-only tags are not allocations
    git(repo, "commit", "--allow-empty", "-m", "new main not tested by this run")
    newer = git(repo, "rev-parse", "HEAD")
    git(repo, "push", "origin", "HEAD:refs/heads/main")
    git(repo, "checkout", "--detach", sha)
    result = allocate(repo, sha)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v6.2.0a4"
    assert git(remote, "rev-parse", "v6.2.0a4^{commit}") == sha != newer
    assert git(remote, "cat-file", "-t", "v6.2.0a4") == "tag"
    identity = "refs/tags/alpha-build/chunkhound/chunkhound/101"
    assert git(remote, "cat-file", "-t", identity) == "blob"
    assert (
        json.loads(git(remote, "cat-file", "blob", identity))["release_tag"]
        == "v6.2.0a4"
    )
    git(repo, "fetch", "--tags", "origin")
    # Run identities must not interfere with docs/Hatch Git tag discovery.
    assert git(repo, "describe", "--tags", "--abbrev=0") == "v6.2.0a4"
    assert (
        repo / "output-101.txt"
    ).read_text() == f"release_tag=v6.2.0a4\nsource_sha={sha}\n"


def test_rerun_reuses_original_identity_even_after_base_change(repos):
    repo, remote, sha = repos
    first = allocate(repo, sha)
    assert first.returncode == 0, first.stderr
    assert first.stdout == "v6.2.0a1\n"
    rerun = allocate(repo, sha)
    assert rerun.returncode == 0, rerun.stderr
    assert rerun.stdout == "v6.2.0a1\n"
    # A new stable tag (6.2.0) moves the base to 6.3.0; the original run keeps
    # its alpha. A hotfix (6.2.1) does not move the base again.
    git(repo, "tag", "-a", "v6.2.0", "-m", "stable", sha)
    git(repo, "push", "origin", "refs/tags/v6.2.0")
    rerun = allocate(repo, sha)
    assert rerun.returncode == 0, rerun.stderr
    assert rerun.stdout == "v6.2.0a1\n"
    # A different successful build of the same commit uses the new base.
    other = allocate(repo, sha, "102")
    assert other.returncode == 0, other.stderr
    assert other.stdout == "v6.3.0a1\n"
    git(repo, "tag", "-a", "v6.2.1", "-m", "hotfix", sha)
    git(repo, "push", "origin", "refs/tags/v6.2.1")
    hotfix = allocate(repo, sha, "103")
    assert hotfix.returncode == 0, hotfix.stderr
    assert hotfix.stdout == "v6.3.0a2\n"


def test_base_is_numerically_highest_stable_next_minor(repos):
    repo, remote, sha = repos
    # Lexical order would pick 6.9.0 and give base 6.10.0; numeric gives 6.11.0.
    for version in ("v6.9.0", "v6.10.0"):
        git(repo, "tag", "-a", version, "-m", "stable", sha)
        git(repo, "push", "origin", f"refs/tags/{version}")
    result = allocate(repo, sha)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "v6.11.0a1\n"


def test_major_stable_moves_base_to_next_minor_of_new_major(repos):
    repo, remote, sha = repos
    git(repo, "tag", "-a", "v7.0.0", "-m", "stable", sha)
    git(repo, "push", "origin", "refs/tags/v7.0.0")
    result = allocate(repo, sha)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "v7.1.0a1\n"


def test_missing_stable_tag_fails_without_creating_alpha_tags(repos):
    repo, remote, sha = repos
    git(repo, "push", "origin", "--delete", "refs/tags/v6.1.0")
    result = allocate(repo, sha)
    assert result.returncode != 0
    assert "No stable" in result.stderr
    assert git(remote, "tag", "--list") == ""


@pytest.mark.parametrize("same_run", [True, False])
def test_concurrent_builds_allocate_atomic_immutable_identities(
    repos, tmp_path, same_run
):
    repo, remote, sha = repos
    copies = [tmp_path / "worker1", tmp_path / "worker2"]
    for copy in copies:
        git(tmp_path, "clone", str(remote), str(copy))
        git(copy, "checkout", "--detach", sha)
    # Force both workers to finish their first remote snapshot before either
    # pushes, exercising a real ref collision rather than scheduler luck.
    hook = remote / "hooks" / "pre-receive"
    hook.write_text(
        "#!/bin/sh\n"
        'mkdir -p "$GIT_DIR/collision-test"\n'
        'touch "$GIT_DIR/collision-test/$$"\n'
        "i=0\n"
        'while [ "$(ls "$GIT_DIR/collision-test" | wc -l)" -lt 2 ] '
        '&& [ "$i" -lt 100 ]; do\n'
        "  sleep 0.05; i=$((i + 1))\n"
        "done\n"
        "exit 0\n"
    )
    hook.chmod(0o755)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(allocate, copy, sha, "101" if same_run else str(101 + i))
            for i, copy in enumerate(copies)
        ]
        results = [future.result(timeout=30) for future in futures]
    assert all(result.returncode == 0 for result in results), [
        r.stderr for r in results
    ]
    assert {r.stdout.strip() for r in results} == (
        {"v6.2.0a1"} if same_run else {"v6.2.0a1", "v6.2.0a2"}
    )
    tags = git(remote, "tag", "--list", "v6.2.0a*").splitlines()
    assert len(tags) == (1 if same_run else 2)


def test_rerun_cannot_rebind_identity_to_another_source(repos):
    repo, remote, sha = repos
    assert allocate(repo, sha).returncode == 0
    original = git(remote, "rev-parse", "v6.2.0a1")
    git(repo, "commit", "--allow-empty", "-m", "another source")
    result = allocate(repo, git(repo, "rev-parse", "HEAD"))
    assert result.returncode != 0
    assert "Run identity does not match" in result.stderr
    assert git(remote, "rev-parse", "v6.2.0a1") == original
    assert git(remote, "tag", "--list", "v6.2.0a*") == "v6.2.0a1"


def test_wrong_checkout_never_tags_untested_source(repos):
    repo, remote, sha = repos
    git(repo, "commit", "--allow-empty", "-m", "wrong checkout")
    result = allocate(repo, sha)
    assert result.returncode != 0
    assert "Checkout HEAD must be the exact CI-tested source SHA" in result.stderr
    assert git(remote, "tag", "--list", "v*a*") == ""


def test_git_failure_reports_remote_error_details(repos):
    repo, remote, sha = repos
    missing_remote = repo / "missing-remote.git"
    git(repo, "remote", "set-url", "origin", str(missing_remote))
    result = allocate(repo, sha)
    assert result.returncode != 0
    assert "Alpha allocation failed" in result.stderr
    assert "missing-remote.git" in result.stderr
    assert git(remote, "tag", "--list", "v*a*") == ""


def test_remote_rejection_leaves_no_half_allocated_release(repos):
    repo, remote, sha = repos
    hook = remote / "hooks" / "update"
    hook.write_text('#!/bin/sh\ncase "$1" in refs/tags/alpha-build/*) exit 1;; esac\n')
    hook.chmod(0o755)
    result = allocate(repo, sha)
    assert result.returncode != 0
    assert "Atomic tag push failed" in result.stderr
    assert git(remote, "tag", "--list", "v*a*") == ""


def test_duplicate_metadata_fails_closed_without_allocating(repos):
    repo, remote, sha = repos
    assert allocate(repo, sha).returncode == 0
    metadata = {
        "schema": 1,
        "repository": "chunkhound/chunkhound",
        "run_id": "101",
        "source_sha": sha,
        "release_tag": "v6.2.0a2",
    }
    git(repo, "tag", "-a", "v6.2.0a2", "-m", json.dumps(metadata), sha)
    git(repo, "push", "origin", "refs/tags/v6.2.0a2")
    result = allocate(repo, sha)
    assert result.returncode != 0
    assert "Multiple alpha tags claim this build" in result.stderr
    assert git(remote, "tag", "--list", "v6.2.0a*").splitlines() == [
        "v6.2.0a1",
        "v6.2.0a2",
    ]


def test_ci_alpha_is_upstream_push_only_and_gated_on_existing_validations():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    prepare = jobs["prepare-alpha"]
    assert set(prepare["needs"]) == set(jobs["pages-artifact"]["needs"]) - {
        "site-build",
        "site-build-validation",
    }
    assert "pages-artifact" not in prepare["needs"] and "deploy" not in prepare["needs"]
    assert (
        prepare["if"] == "github.repository == 'chunkhound/chunkhound' "
        "&& github.event_name == 'push' && github.ref == 'refs/heads/main'"
    )
    assert prepare["permissions"] == {"contents": "write"}
    assert jobs["release-alpha"]["uses"] == "./.github/workflows/release-packages.yml"
    assert jobs["release-alpha"]["with"]["release_tag"] == (
        "${{ needs.prepare-alpha.outputs.release_tag }}"
    )
    assert "alpha" not in jobs["release-alpha"]["with"]
    assert jobs["release-alpha"]["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "issues": "write",
    }
    assert workflow["concurrency"]["group"] == "ci-${{ github.run_id }}"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert jobs["deploy"]["concurrency"]["cancel-in-progress"] is False
    deploy_steps = jobs["deploy"]["steps"]
    assert "git/ref/heads/main" in deploy_steps[0]["run"]
    assert deploy_steps[1]["if"] == "steps.current.outputs.deploy == 'true'"

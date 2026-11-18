"""Lock the CI split that runs site browser contracts outside the main tests job."""

import re
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

CI = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
SITE_CONFTEST = Path(__file__).resolve().parents[2] / "tests/site/conftest.py"
SITE_IGNORE = "--ignore=tests/site"
# A retry wrapper that expires at the per-test cap loses the failing test name
# and forces a full shard rerun; keep a real reporting window.
MIN_REPORT_MARGIN_S = 180
SETUP_RESERVE_S = 300


def _jobs() -> dict[str, Any]:
    with CI.open(encoding="utf-8") as handle:
        return cast(dict[str, Any], yaml.safe_load(handle)["jobs"])


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _pytest_commands(job: dict[str, Any]) -> list[str]:
    """Every pytest invocation a job runs, including retry-action commands."""
    commands = (
        step.get("run") or step.get("with", {}).get("command", "")
        for step in job["steps"]
    )
    return [command for command in commands if "pytest" in command]


def test_main_tests_job_excludes_site_and_does_not_wait_for_site_build() -> None:
    job = _jobs()["tests"]
    commands = _pytest_commands(job)

    assert commands
    assert all(SITE_IGNORE in command for command in commands)
    assert "site-build" not in _needs(job)


def test_site_tests_job_owns_site_contracts() -> None:
    job = _jobs()["site-tests"]
    commands = _pytest_commands(job)

    assert "site-build" in _needs(job)
    assert commands
    assert all(
        "tests/site" in command and SITE_IGNORE not in command for command in commands
    )
    setup = next(
        step
        for step in job["steps"]
        if step.get("uses") == "./.github/actions/ci-python-setup"
    )
    assert setup["with"]["site"] == "true"


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def _per_test_timeout_s() -> tuple[int, int]:
    """Per-test cap from the CI command and the pytest marker enforcing it."""
    command = _step(_jobs()["site-tests"], "Run site tests")["with"]["command"]
    from_command = int(re.search(r"--timeout=(\d+)", command).group(1))
    conftest = SITE_CONFTEST.read_text(encoding="utf-8")
    from_marker = int(re.search(r"_SITE_TEST_TIMEOUT_S = (\d+)", conftest).group(1))
    return from_command, from_marker


def test_site_tests_retry_budget_leaves_report_margin() -> None:
    job = _jobs()["site-tests"]
    from_command, from_marker = _per_test_timeout_s()
    # Drift here silently clips tests or drops the reporting margin.
    assert from_command == from_marker

    for entry in job["strategy"]["matrix"]["include"]:
        wrapper = entry["pytest_timeout_minutes"] * 60
        assert wrapper - from_command >= MIN_REPORT_MARGIN_S, entry["os"]
        assert 2 * wrapper + SETUP_RESERVE_S <= entry["job_timeout_minutes"] * 60


def test_site_tests_flaky_gate_is_the_failure_gate() -> None:
    job = _jobs()["site-tests"]
    names = [step.get("name") for step in job["steps"]]
    assert _step(job, "Run site tests")["continue-on-error"] is True
    upload_before_gate = names.index("Upload site test results") < names.index(
        "Check flaky annotations"
    )
    assert upload_before_gate
    assert _step(job, "Upload site test results")["if"] == "always()"

    gate = _step(job, "Check flaky annotations")
    assert gate["if"] == "steps.site-tests.outcome == 'failure'"
    assert "check_flaky_annotations.py site-test-results.xml" in gate["run"]


def test_deploy_gate_waits_for_site_contracts() -> None:
    jobs = _jobs()
    assert "site-tests" in _needs(jobs["pages-artifact"])
    assert _needs(jobs["deploy"]) == ["pages-artifact"]

"""Contributor receipt contract.

The homepage figure is generated, never hand-written: sync-contributors.mjs
turns GitHub's contributor payload into site/src/lib/contributors.json, and the
build renders that record. These guards keep the receipt honest across
refactors — the visible number, its accessible name, and the count-up target
must all come from the same value, and the label must describe what is counted.

The build renders a snapshot fixture (tests/site/fixtures/contributors-api.json),
never the live API; CI ships that generated record with the reused dist.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest

from tests.site.dom_helpers import dist_body
from tests.site.html_helpers import attributes
from tests.site.process_runner import run_text_process
from tests.site.tsx_runner import NODE, ROOT, isolated_subprocess_env

SCRIPT = ROOT / "site" / "scripts" / "sync-contributors.mjs"
FIXTURE = ROOT / "tests" / "site" / "fixtures" / "contributors-api.json"
API = "https://api.github.com/repos/chunkhound/chunkhound/contributors?per_page=100"
NEXT = "https://api.github.com/repos/chunkhound/chunkhound/contributors?page=2"
_NEXT_LINK = f"{{headers:{{link:'<{NEXT}>; rel=\"next\"'}}}}"
_ALICE = "{id:7,login:'a',type:'User'}"
_BOB = "{id:8,login:'b',type:'User'}"
# Fixed month table: strftime("%b") follows the host locale and would flake.
_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()

# The receipt metric followed by its label. Astro injects scope attributes, so
# the attributes are parsed, not pattern-matched positionally.
_METRIC = re.compile(
    r"<dt\b([^>]*)>(\d+)</dt>\s*<dd\b[^>]*>contributors from multiple companies"
)


def _run_sync(out: Path, snapshot: Path = FIXTURE) -> subprocess.CompletedProcess[str]:
    with isolated_subprocess_env(
        CHUNKHOUND_CONTRIBUTORS_JSON=str(snapshot),
        CHUNKHOUND_CONTRIBUTORS_OUT=str(out),
    ) as env:
        return run_text_process([NODE, str(SCRIPT)], cwd=ROOT, env=env)


def _fetch_stub(log: Path, handler: str) -> str:
    """Inline module entry: log every fetched URL, then answer via handler(n)."""
    return (
        "import { appendFileSync } from 'node:fs';"
        f"const log={json.dumps(str(log))};let n=0;"
        "globalThis.fetch=async(url)=>{appendFileSync(log,url+'\\n');n+=1;"
        f"return ({handler})(n);}};await import('{SCRIPT.as_uri()}')"
    )


def _run_stub(stub: str, out: Path) -> subprocess.CompletedProcess[str]:
    """Run a stubbed sync with fetch faked; the output path stays hermetic."""
    with isolated_subprocess_env(CHUNKHOUND_CONTRIBUTORS_OUT=str(out)) as env:
        return run_text_process(
            [NODE, "--input-type=module", "-e", stub], cwd=ROOT, env=env, timeout=60
        )


def test_sync_records_unique_human_accounts(tmp_path: Path) -> None:
    """Duplicate ids collapse, Bot type is dropped, and agent logins are denied."""
    out = tmp_path / "contributors.json"
    result = _run_sync(out)
    assert result.returncode == 0, result.stderr

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["count"] == 2
    assert record["source"].startswith("https://github.com/chunkhound/chunkhound")
    assert record["asOf"].endswith("Z")


def test_sync_retries_a_transient_error_while_quota_remains(tmp_path: Path) -> None:
    """A 503 with quota left retries; only an exhausted quota fails fast."""
    out = tmp_path / "contributors.json"
    stub = _fetch_stub(
        tmp_path / "calls.txt",
        "(n)=>n<2"
        "?new Response('',{status:503,"
        "headers:{'x-ratelimit-remaining':'4999','x-ratelimit-reset':'9e9'}})"
        f":Response.json([{_ALICE}])",
    )
    result = _run_stub(stub, out)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["count"] == 1


def test_sync_follows_link_header_pagination(tmp_path: Path) -> None:
    """The Link rel="next" URL, not array order, decides the next fetch."""
    out = tmp_path / "contributors.json"
    calls = tmp_path / "calls.txt"
    stub = _fetch_stub(
        calls,
        "(n)=>n===1"
        f"?Response.json([{_ALICE}],{_NEXT_LINK})"
        f":Response.json([{_ALICE},{_BOB}])",
    )
    result = _run_stub(stub, out)
    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [API, NEXT]
    assert json.loads(out.read_text(encoding="utf-8"))["count"] == 2


def test_sync_rejects_a_looping_next_link(tmp_path: Path) -> None:
    """A repeated next link must stop loudly instead of fetching forever."""
    out = tmp_path / "contributors.json"
    calls = tmp_path / "calls.txt"
    stub = _fetch_stub(calls, f"()=>Response.json([{_ALICE}],{_NEXT_LINK})")
    result = _run_stub(stub, out)
    assert result.returncode != 0
    assert NEXT in result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [API, NEXT]
    assert not out.exists()


@pytest.mark.parametrize(
    ("status", "headers", "attempts"),
    [
        pytest.param(
            403,
            "{'x-ratelimit-remaining':'0',"
            "'x-ratelimit-reset':String(Math.floor(Date.now()/1000)+3600)}",
            1,
            id="exhausted-quota",
        ),
        pytest.param(503, "{'retry-after':'1'}", 3, id="max-attempts"),
    ],
)
def test_sync_stops_on_quota_or_attempt_cap(
    tmp_path: Path, status: int, headers: str, attempts: int
) -> None:
    """Exhausted quota fails at once; a persistent outage stops at the cap."""
    out = tmp_path / "contributors.json"
    calls = tmp_path / "calls.txt"
    stub = _fetch_stub(
        calls, f"()=>new Response('',{{status:{status},headers:{headers}}})"
    )
    started = time.monotonic()
    result = _run_stub(stub, out)
    elapsed = time.monotonic() - started
    assert result.returncode != 0
    assert len(calls.read_text(encoding="utf-8").splitlines()) == attempts
    assert not out.exists()
    if attempts > 1:
        assert elapsed >= 1.5, "retry-after must be honored, not ignored"


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [
            {"id": 1, "login": "dependabot[bot]", "type": "Bot"},
            {"id": 2, "login": "CursorAgent", "type": "User"},
        ],
    ],
    ids=["empty-page", "all-excluded"],
)
def test_sync_refuses_a_zero_receipt(tmp_path: Path, entries: list[dict]) -> None:
    """Nothing left after filtering is a failure, never a published zero."""
    out = tmp_path / "contributors.json"
    stub = _fetch_stub(
        tmp_path / "calls.txt",
        f"()=>Response.json({json.dumps(entries)})",
    )
    result = _run_stub(stub, out)
    assert result.returncode != 0
    assert "zero" in result.stderr
    assert not out.exists()


def test_sync_fails_loudly_without_a_payload(tmp_path: Path) -> None:
    """A missing snapshot must fail the build, never publish a zero receipt."""
    result = _run_sync(tmp_path / "contributors.json", tmp_path / "missing.json")
    assert result.returncode != 0


def test_homepage_receipt_renders_one_generated_value() -> None:
    """Built markup carries the generated count in every representation."""
    body = dist_body("index.html")
    match = _METRIC.search(body)
    assert match, "homepage must render the generated contributor receipt"

    raw_attrs, text = match.groups()
    attrs = attributes(raw_attrs)
    assert attrs["aria-label"] == f"{text} contributors"
    assert attrs["data-count-to"] == text
    assert "data-suffix" not in attrs
    assert int(text) > 0
    assert "maintainers from multiple companies" not in body
    assert (
        '<a href="https://github.com/chunkhound/chunkhound/graphs/contributors"' in body
    )


def test_generated_receipt_provenance(generated_contributors: dict) -> None:
    """The rendered number, date and verify link come from the build's record."""
    body = dist_body("index.html")
    match = _METRIC.search(body)
    assert match, "homepage must render the generated contributor receipt"

    _, text = match.groups()
    assert int(text) == generated_contributors["count"]
    as_of = datetime.fromisoformat(generated_contributors["asOf"][:10])
    rendered_date = f"{_MONTHS[as_of.month - 1]} {as_of.day}, {as_of.year}"
    assert f"as of {rendered_date}" in body
    assert f'<a href="{generated_contributors["source"]}"' in body

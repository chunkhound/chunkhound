import hashlib
import json
import subprocess

import pytest

import chunkhound_native as native
from chunkhound.hub.inspect import inspect_database
from tests.hub.activation_support import ROOT, cli, cli_env, publish, write_config
from tests.hub.artifacts import build_database
from tests.hub.test_index_format import snapshot


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rel", ["latest", "-1", "-2"])
def test_explicit_relative_version_is_installed_and_searchable(
    tmp_path, hub_http, embedding_http, backend, rel
):
    project = tmp_path / "checkout"
    write_config(
        project, backend, hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    names = {"latest": "newest", "-1": "previous", "-2": "oldest"}
    versions = []
    for label, symbol in names.items():
        source = build_database(tmp_path / symbol, backend, symbol_prefix=symbol)
        manifest, _ = publish(hub_http, source, rel=label, prefix=symbol)
        versions.extend(manifest["embeddings"][0]["versions"])
    manifest["embeddings"][0]["versions"] = versions
    hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    prefix = names[rel]
    result = cli(project, "pull", project, "--version", rel)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Hub base installed; run index to overlay local changes" in (
        result.stdout + result.stderr
    )
    assert embedding_http.requests == []  # Explicit pull installs only.
    search = cli(project, "search", prefix + "_0", project, "--regex")
    assert search.returncode == 0, search.stdout + search.stderr
    assert "source.py" in search.stdout + search.stderr
    assert "No results found" not in search.stdout + search.stderr
    assert hub_http.requests == [
        ("/catalogue?token=secret", None),
        (f"/{prefix}.tar.gz?token=secret", None),
    ]
    assert "token=secret" not in result.stdout + result.stderr
    assert "/catalogue?" not in result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rust", ["0", "1"])
@pytest.mark.parametrize("initial", ["absent", "misaligned", "aligned"])
def test_auto_installs_before_drift_pinning_and_aligned_makes_no_request(
    tmp_path, hub_http, embedding_http, backend, rust, initial
):
    project = tmp_path / "checkout"
    project.mkdir()
    (project / "source.py").write_text(
        "\n".join(f"def local_{i}(): return 'local_marker_{i}'" for i in range(100)),
        encoding="utf-8",
    )
    if initial != "absent":
        old_model = "hub-test" if initial == "aligned" else "old-model"
        write_config(project, backend, None, embedding_http.url, old_model)
        old = cli(project, "index", project, rust=rust)
        assert old.returncode == 0, old.stdout + old.stderr
    write_config(
        project, backend, hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    publish(hub_http, build_database(tmp_path / "published", backend))
    embedding_http.requests.clear()
    result = cli(project, "index", project, "--verbose", rust=rust)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(hub_http.requests) == (0 if initial == "aligned" else 2)
    target = project / ".chunkhound" / "db" / (
        "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    )
    assert {g.model for g in inspect_database(target, backend).groups} == {"hub-test"}
    assert all(request["model"] == "hub-test" for request in embedding_http.requests)
    if initial != "aligned":
        assert embedding_http.requests  # Normal overlay followed base install.
    assert "token=secret" not in result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["--no-embeddings", "--simulate", "--check-ignores"])
def test_index_subtools_do_not_request_hub(tmp_path, hub_http, embedding_http, mode):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    (project / "source.py").write_text("def local_only(): return 1\n")
    result = cli(project, "index", project, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    assert hub_http.requests == []
    assert embedding_http.requests == []


def test_unknown_dimensions_do_not_probe_hub_or_embedding_service(tmp_path, hub_http):
    project = tmp_path / "checkout"
    data = write_config(
        project,
        "duckdb",
        hub_http.url + "/catalogue?token=secret",
        "http://127.0.0.1:1/v1",
        model="unknown-offline-model",
    )
    del data["embedding"]["output_dims"]
    (project / ".chunkhound.json").write_text(json.dumps(data))
    result = cli(project, "pull", project)
    assert result.returncode == 1
    assert hub_http.requests == []
    assert "known embedding dimensions" in result.stderr


@pytest.mark.parametrize("mode", ["pull", "index"])
@pytest.mark.parametrize(
    "failure", ["http", "checksum", "archive", "space", "mismatch"]
)
def test_cli_failure_taxonomy_is_real_not_a_mocked_registry_assertion(
    tmp_path, hub_http, embedding_http, mode, failure
):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    (project / "local.py").write_text("def local_fallback(): return 7\n")
    source = build_database(
        tmp_path / "published",
        model="wrong-model" if failure == "mismatch" else "hub-test",
    )
    manifest, route = publish(hub_http, source)
    if failure == "http":
        hub_http.status = 503
    elif failure == "checksum":
        manifest["embeddings"][0]["versions"][0]["artifact"]["sha256"] = "0" * 64
    elif failure == "archive":
        body = b"not an archive"
        hub_http.resources[route] = body
        manifest["embeddings"][0]["versions"][0]["artifact"].update(
            bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
        )
    hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    if failure == "space":
        script = """
import shutil
import sys
from pathlib import Path
from chunkhound.api.cli.main import main
original = shutil.disk_usage
def disk_usage(path):
    usage = original(path)
    return (
        usage._replace(free=0)
        if Path(path).name.startswith('.chunkhound-pull-') else usage
    )
shutil.disk_usage = disk_usage
sys.argv = ['chunkhound', *sys.argv[1:]]
main()
"""
        result = subprocess.run(
            ["uv", "run", "python", "-c", script, mode, str(project), "--verbose"],
            cwd=project,
            env=cli_env(project),
            capture_output=True,
            text=True,
            timeout=180,
        )
    else:
        result = cli(project, mode, project, "--verbose")
    fatal = mode == "pull" or failure == "mismatch"
    assert result.returncode == (1 if fatal else 0), result.stdout + result.stderr
    text = result.stdout + result.stderr
    assert "token=secret" not in text and "response-body-secret" not in text
    assert "Traceback" not in text
    target = project / ".chunkhound" / "db" / "chunks.db"
    assert target.exists() is (not fatal)
    if not fatal:
        search = cli(project, "search", "local_fallback", project, "--regex")
        assert "local.py" in search.stdout + search.stderr


@pytest.mark.parametrize("mode", ["pull", "index"])
def test_candidate_mismatch_exit_preserves_local_database_sidecars(
    tmp_path, hub_http, embedding_http, mode
):
    project = tmp_path / "checkout"
    write_config(project, "duckdb", None, embedding_http.url, "old-model")
    (project / "local.py").write_text("def retained_local_work(): return 7\n")
    baseline = cli(project, "index", project)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    target = project / ".chunkhound" / "db" / "chunks.db"
    before = snapshot(target.parent)
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    publish(hub_http, build_database(tmp_path / "published", model="contradiction"))
    result = cli(project, mode, project, "--verbose")
    assert result.returncode == 1
    assert snapshot(target.parent) == before
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["pull", "index"])
def test_preinstall_busy_is_explicit_failure_and_auto_soft(
    tmp_path, hub_http, embedding_http, mode
):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    (project / "local.py").write_text("def local_busy_fallback(): return 7\n")
    target = project / ".chunkhound" / "db" / "chunks.db"
    target.parent.mkdir(parents=True)
    publish(hub_http, build_database(tmp_path / "published"))
    # This owner excludes installation but owns no backend writer connection.
    held = native.acquire_database_lease(str(target))
    try:
        result = cli(project, mode, project, "--verbose")
    finally:
        held.close()
    assert result.returncode == (1 if mode == "pull" else 0)
    assert "destination is busy" in result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert target.exists() is (mode == "index")


def test_zero_is_not_a_relative_version(tmp_path):
    assert cli(tmp_path, "pull", "--version", "0").returncode == 2


@pytest.mark.parametrize("boundary,mode", [
    ("initial", "index"), ("recheck", "index"),
    ("candidate", "index"), ("candidate", "pull"),
])
def test_cli_reader_failure_has_sanitized_exit_without_fallback(
    tmp_path, hub_http, embedding_http, boundary, mode
):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret",
        embedding_http.url,
    )
    (project / "local.py").write_text("def must_not_index(): return 42\n")
    target = project / ".chunkhound" / "db" / "chunks.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    before = []
    if boundary != "recheck":
        build_database(target.parent, model="old-model")
        before.append(snapshot(target))
    competitor = build_database(tmp_path / "competitor", model="old-model")
    publish(hub_http, build_database(tmp_path / "published"))
    sidecar = target.with_name(target.name + ".root.json")
    retained_root = json.dumps({"version": 1, "indexed_root_path": project.as_posix()})
    sidecar.write_text(retained_root)
    if boundary == "recheck":
        def install_race_winner(route):
            if route.startswith("/base.tar.gz"):
                lease = native.acquire_install_lease(str(target), "duckdb")
                try:
                    lease.install(str(competitor))
                finally:
                    lease.close()
                sidecar.write_text(retained_root)
                before.append(snapshot(target))
        hub_http.before_get = install_race_winner
    report = tmp_path / "cli-reader-report.json"
    script = r'''
import json
import sys
from pathlib import Path
root, boundary, target_text, report_text, mode, project = sys.argv[1:]
sys.path.insert(0, root)
import pytest
import chunkhound_native as native
from chunkhound.api.cli.main import main
from tests.hub.test_activation_client import reader_close_fault, retained_reader_error

target, report = Path(target_text), Path(report_text)
def wants_path(path):
    if boundary == "candidate":
        return any(p.name.startswith(".chunkhound-pull-") for p in path.parents)
    return path.resolve() == target.resolve()
sys.argv = ["chunkhound", mode, project, "--verbose"]
original = None
outcome = None
held_workspaces = []
try:
    with reader_close_fault(pytest.MonkeyPatch(), wants_path) as fault_state:
        readers, held_workspaces = fault_state
        try:
            main()
        except SystemExit as error:
            outcome = error
        assert outcome is not None and outcome.code == 1
        original = retained_reader_error(outcome)
        held_target = readers[0][1]
        original.ownership.assert_covers(held_target)
        assert readers[0][0]().execute("SELECT 42").fetchone() == (42,)
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(held_target), "duckdb")
    original.retry_close()
    original.retry_close()
    native.acquire_install_lease(str(held_target), "duckdb").close()
    report.write_text(json.dumps({"actual_exit": outcome.code,
                                  "retained_then_retried": True}))
finally:
    if original is not None:
        original.retry_close()
    for workspace in held_workspaces:
        workspace.cleanup()
raise outcome
'''
    result = subprocess.run(
        ["uv", "run", "python", "-c", script, str(ROOT), boundary, str(target),
         str(report), mode, str(project)],
        cwd=project, env=cli_env(project), capture_output=True, text=True,
        timeout=180,
    )
    text = result.stdout + result.stderr
    assert report.exists(), text
    assert json.loads(report.read_text()) == {
        "actual_exit": 1, "retained_then_retried": True,
    }
    assert result.returncode == 1, text
    assert text.count("Cannot close hub database inspection reader") == 1
    for forbidden in (
        "REVIEW_SECRET", "raw-close-detail", "raw-reader-detail",
        "Traceback", "Full error details", "continuing local indexing",
        "identity is unreadable", "token=secret",
    ):
        assert forbidden not in text
    assert snapshot(target) == before[0]
    assert sidecar.read_text() == retained_root
    assert embedding_http.requests == []
    assert len(hub_http.requests) == (0 if boundary == "initial" else 2)
    search = cli(project, "search", "must_not_index", project, "--regex")
    assert search.returncode == 0, search.stdout + search.stderr
    assert "No results found" in search.stdout + search.stderr

import hashlib
import json
import subprocess

import pytest

import chunkhound_native as native
from chunkhound.hub.inspect import inspect_database
from tests.hub.activation_support import cli, cli_env, publish, write_config
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
    result = cli(project, "pull", project, "--version", rel, "--verbose")
    assert result.returncode == 0, result.stdout + result.stderr
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

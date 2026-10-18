"""Exercise pytest's public process contract under hostile developer state."""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _junit_counts(root: Path) -> tuple[int, int, int, int]:
    tree = ET.parse(root / "report.xml")
    cases = tree.findall(".//testcase")
    failures = tree.findall(".//failure") + tree.findall(".//error")
    skipped = tree.findall(".//skipped")
    xfailed = [item for item in skipped if item.get("type") == "pytest.xfail"]
    return len(cases), len(failures), len(skipped), len(xfailed)

PROBE = """
import os
import sys
from pathlib import Path
import pytest
from chunkhound.core.config.config import Config
from tests.helpers.isolation import run_control

# Collection, not just test setup, must already have a private environment.
assert "chunkhound_debug" not in os.environ
assert "openai_api_key" not in os.environ
assert "ANTHROPIC_API_KEY" not in os.environ
for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "FTP_PROXY"):
    assert proxy_name not in os.environ
assert os.environ["PATH"] == os.environ["EXPECTED_PATH"]
assert Path.home() != Path(os.environ["DEVELOPER_HOME"])
assert run_control("CHUNKHOUND_RUN_HEAVY_TESTS") == "1"
assert run_control("CHUNKHOUND_ENABLE_PERF_TESTS") == "1"
assert run_control("CH_SKIP_HCL_TESTS") == "1"
assert run_control("CHUNKHOUND_YAML_ENGINE") == "tree"
assert "CHUNKHOUND_RUN_HEAVY_TESTS" not in os.environ
assert "CHUNKHOUND_ENABLE_PERF_TESTS" not in os.environ
assert "CH_SKIP_HCL_TESTS" not in os.environ
assert "CHUNKHOUND_YAML_ENGINE" not in os.environ
assert Config().debug is False
assert Path.cwd() != Path(os.environ["DEVELOPER_CWD"])

@pytest.fixture
def deliberate_input(hermetic_environment, monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_DEBUG", "true")
    yield
    assert os.environ["CHUNKHOUND_DEBUG"] == "true"
    assert Path.home() == hermetic_environment
    extension = Path.home() / ".duckdb/extensions/version/platform/vss.duckdb_extension"
    assert extension.read_bytes() == b"installed-vss-artifact"
    runtime = Path(os.environ["CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR"])
    assert (runtime / "installed-artifact").read_text() == "private-watchman-write"

@pytest.mark.parametrize("case", range(4))
def test_contract(case, deliberate_input, hermetic_environment, tmp_path):
    assert Config().debug is True
    assert not (Path.home() / ".codex" / "auth.json").exists()
    assert Path(os.environ["CODEX_HOME"]).is_relative_to(Path.home())
    keys = ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR")
    for key in (*keys, "APPDATA", "LOCALAPPDATA"):
        assert Path(os.environ[key]).is_relative_to(hermetic_environment.parent)
    extension = Path.home() / ".duckdb/extensions/version/platform/vss.duckdb_extension"
    assert extension.read_bytes() == b"installed-vss-artifact"
    runtime = Path(os.environ["CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR"])
    assert (runtime / "installed-artifact").read_text() == "watchman"
    assert not (Path.home() / "leaked-from-previous-test").exists()
    (Path.home() / "leaked-from-previous-test").touch()
    (tmp_path / "retained-output").write_text("diagnostics remain available")
    # The VSS tree is one frozen, shared image linked into every sandbox.
    with pytest.raises(PermissionError):
        extension.write_bytes(b"private-vss-write")
    if sys.platform != "win32":
        # POSIX read-only directories also block new files. Windows ignores
        # the read-only attribute on directories for creation.
        with pytest.raises(PermissionError):
            (extension.parent / "leaked-install").touch()
    (runtime / "installed-artifact").write_text("private-watchman-write")
    if case == 3:
        pytest.xfail("Artifact cleanup also applies to failed test calls")
"""


PARSE_WORKER_PROBE = """
import shutil
from pathlib import Path
import pytest
from tests.contracts.pipeline_harness import index_with_rust

@pytest.mark.xdist_group("parse-worker-lifetime")
@pytest.mark.parametrize("kind", ["python", "pdf", "pdf"])
def test_parse_after_previous_sandbox_cleanup(tmp_path, kind):
    project = tmp_path / "project"
    project.mkdir()
    if kind == "python":
        (project / "source.py").write_text("def hello():\\n    return 42\\n")
    else:
        import tests
        source = Path(tests.__file__).parent / "fixtures/cast_research_paper.pdf"
        shutil.copy2(source, project / source.name)
    result = index_with_rust(project, tmp_path / "db", skip_embeddings=True)
    assert not result.errors, result.errors
    assert result.chunk_tuples
"""


NATIVE_RUNTIME_PROBE = """
import tempfile
from pathlib import Path
import duckdb
from tests.helpers.isolation import sandbox_context

source = (Path.home() / ".duckdb/extensions").resolve()
with tempfile.TemporaryDirectory() as scratch:
    root = Path(scratch)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    (artifacts / "vss").symlink_to(source, target_is_directory=True)
    # A fresh macOS process aborts after ~505 distinct VSS image loads.
    for case in range(550):
        with sandbox_context(root, artifacts):
            with duckdb.connect(config={"threads": 1}) as connection:
                connection.execute("INSTALL vss")
                connection.execute("LOAD vss")
                connection.execute("CREATE TABLE vectors (v FLOAT[3])")
                connection.execute("CREATE INDEX idx ON vectors USING HNSW (v)")
print("550 private sandboxes loaded VSS")
"""


def _developer_inputs(root: Path) -> tuple[Path, Path]:
    home, runtime = root / "developer", root / "installed-runtime"
    extension = home / ".duckdb/extensions/version/platform/vss.duckdb_extension"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"installed-vss-artifact")
    (home / ".codex").mkdir()
    (home / ".codex/auth.json").write_text('{"secret":"never-copy"}')
    (home / ".chunkhound.json").write_text('{"debug":true}')
    runtime.mkdir()
    (runtime / "installed-artifact").write_text("watchman")
    return home, runtime


def _hostile_environment(home: Path, runtime: Path) -> dict[str, str]:
    return dict(
        os.environ,
        **{
            "HOME": str(home),
            "USERPROFILE": str(home),
            "DEVELOPER_HOME": str(home),
            "DEVELOPER_CWD": str(home.parent),
            "chunkhound_debug": "true",
            "openai_api_key": "ambient-secret",
            "ANTHROPIC_API_KEY": "ambient-secret",
            "CHUNKHOUND_RUN_HEAVY_TESTS": "1",
            "CHUNKHOUND_ENABLE_PERF_TESTS": "1",
            "CH_SKIP_HCL_TESTS": "1",
            "CHUNKHOUND_YAML_ENGINE": "tree",
            "CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR": str(runtime),
            "HTTP_PROXY": "http://ambient-proxy:3128",
            "https_proxy": "http://ambient-proxy:3128",
            "ALL_PROXY": "socks5://ambient-proxy:1080",
            "NO_PROXY": "localhost",
            "FTP_PROXY": "http://ambient-proxy:21",
            "EXPECTED_PATH": os.environ["PATH"],
            "PYTHONPATH": str(ROOT),
        }
        | _hostile_paths(home),
    )


def _hostile_paths(home: Path) -> dict[str, str]:
    return {
        "CHUNKHOUND_GLOBAL_CONFIG_FILE": str(home / ".chunkhound.json"),
        "XDG_CONFIG_HOME": str(home / "ambient-config"),
        "XDG_CACHE_HOME": str(home / "ambient-cache"),
        "XDG_RUNTIME_DIR": str(home / "ambient-runtime"),
        "APPDATA": str(home / "ambient-appdata"),
        "LOCALAPPDATA": str(home / "ambient-localappdata"),
    }


def _tree_contents(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def _probe_command(probe: Path, workers: int, selection: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "tests.conftest",
        "-c",
        os.path.relpath(ROOT / "pyproject.toml", probe.parent),
        "--confcutdir",
        ".",
        "--basetemp",
        "retained",
        "--junitxml=report.xml",
        *([selection] if selection else []),
        "-q",
        "-n",
        str(workers),
    ]


def _run_probe(
    probe: Path, workers: int, selection: str, *extra: str
) -> subprocess.CompletedProcess[str]:
    root = probe.parent
    environment = _hostile_environment(root / "developer", root / "installed-runtime")
    return subprocess.run(
        [*_probe_command(probe, workers, selection), *extra],
        cwd=probe.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize(
    "selection", ["test_probe.py", "test_probe.py::test_contract", ".", ""]
)
def test_pytest_owns_state_before_collection_and_through_teardown(
    tmp_path: Path, workers: int, selection: str
) -> None:
    home, runtime = _developer_inputs(tmp_path)
    original = _tree_contents(home), _tree_contents(runtime)
    probe = tmp_path / "test_probe.py"
    probe.write_text(PROBE)
    (tmp_path / ".chunkhound.json").write_text('{"debug":true}')
    result = _run_probe(probe, workers, selection)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "report.xml").is_file()
    assert _junit_counts(tmp_path) == (4, 0, 1, 1)
    assert (_tree_contents(home), _tree_contents(runtime)) == original
    retained = tmp_path / "retained"
    assert len(list(retained.rglob("retained-output"))) == 4
    assert not list(retained.rglob(".sandbox-*"))
    assert not list(retained.rglob("vss.duckdb_extension"))
    assert not list(retained.rglob("installed-artifact"))


@pytest.mark.parametrize("workers", [0, 2])
def test_parse_workers_do_not_outlive_private_test_state(
    tmp_path: Path, workers: int
) -> None:
    _developer_inputs(tmp_path)
    probe = tmp_path / "test_probe.py"
    probe.write_text(PARSE_WORKER_PROBE)
    # One worker must consume successive cases to expose inherited stale cwd.
    result = _run_probe(probe, workers, "test_probe.py", "--dist=loadgroup")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _junit_counts(tmp_path) == (3, 0, 0, 0)
    assert not list((tmp_path / "retained").rglob(".sandbox-*"))


@pytest.mark.hnsw
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS dyld TLS limit")
def test_native_runtime_loads_survive_many_private_homes() -> None:
    result = subprocess.run(
        [sys.executable, "-c", NATIVE_RUNTIME_PROBE],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "550 private sandboxes loaded VSS" in result.stdout


@pytest.mark.parametrize("workers", [0, 2])
def test_collect_only_uses_private_cwd_and_relative_testpaths(
    tmp_path: Path, workers: int
) -> None:
    _developer_inputs(tmp_path)
    (tmp_path / ".chunkhound.json").write_text('{"debug":true}')
    probe = tmp_path / "test_probe.py"
    probe.write_text(PROBE)
    # A relative config/testpaths input must still resolve before cwd isolation.
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = .\n")
    result = _run_probe(probe, workers, "", "-c", "pytest.ini", "--collect-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("::") == 4
    assert (tmp_path / "report.xml").is_file()

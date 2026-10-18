"""Public pytest process contracts for offline defaults and deliberate live inputs."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _junit_counts(root: Path) -> tuple[int, int]:
    tree = ET.parse(root / "report.xml")
    failures = tree.findall(".//failure") + tree.findall(".//error")
    return len(tree.findall(".//testcase")), len(failures)


SESSION_RUNNER = """
import socket
import sys
import pytest

methods = ("connect", "connect_ex", "sendto", "sendmsg")
lookups = ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr",
           "getnameinfo")
originals = [(owner, name, getattr(owner, name))
             for owner, names in ((socket.socket, methods), (socket, lookups))
             for name in names if hasattr(owner, name)]
status = pytest.main(sys.argv[1:])
assert all(getattr(owner, name) is method for owner, name, method in originals), \\
       "Session cleanup did not restore socket APIs"
raise SystemExit(status)
"""

OFFLINE_PROBE = """
import socket
import pytest
from pytest_socket import SocketConnectBlockedError

def rejected():
    for kind, method, args in socket_attempts():
        with socket.socket(type=kind) as connection:
            connection.settimeout(0.5)
            with pytest.raises(SocketConnectBlockedError):
                getattr(connection, method)(*args)
    for method, args in lookup_attempts():
        with pytest.raises(SocketConnectBlockedError):
            getattr(socket, method)(*args)

def socket_attempts():
    for host in ("203.0.113.1", "2001:db8::1", "external.invalid"):
        yield socket.SOCK_STREAM, "connect", ((host, 443),)
        yield socket.SOCK_STREAM, "connect_ex", ((host, 443),)
        yield socket.SOCK_DGRAM, "sendto", (b"x", (host, 443))
        yield socket.SOCK_DGRAM, "sendto", (b"x", 0, (host, 443))
        if hasattr(socket.socket, "sendmsg"):
            yield socket.SOCK_DGRAM, "sendmsg", ([b"x"], [], 0, (host, 443))

def lookup_attempts():
    for host in ("203.0.113.1", "2001:db8::1", "external.invalid", b"external.invalid"):
        yield "getaddrinfo", (host, 443)
        yield "getnameinfo", ((host, 443), 0)
        for method in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
            yield method, (host,)

rejected()  # Collection must be offline too.

@pytest.fixture(scope="session", autouse=True)
def session_boundary():
    rejected()
    yield
    rejected()

@pytest.fixture(autouse=True)
def function_boundary(request):
    if request.node.get_closest_marker("live"):
        yield
        return
    rejected()
    yield
    rejected()

def test_offline():
    rejected()
"""
LOCAL_PROBE = """
import socket
import tempfile
import pytest

def bind_loopback(server, host):
    server.settimeout(0.5)
    try:
        server.bind((host, 0))
    except OSError:
        if server.family == socket.AF_INET6:
            pytest.skip("IPv6 loopback unavailable")
        raise

def exchange(server, client, payload):
    with server.accept()[0] as peer:
        peer.settimeout(0.5)
        client.sendall(payload)
        assert peer.recv(len(payload)) == payload

@pytest.mark.parametrize("family,host", [(socket.AF_INET, "127.0.0.1"),
                                         (socket.AF_INET, "localhost"),
                                         (socket.AF_INET6, "::1")])
@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_tcp(family, host, method):
    with socket.socket(family) as server, socket.socket(family) as client:
        bind_loopback(server, host)
        client.settimeout(0.5)
        server.listen()
        address = (host, server.getsockname()[1])
        assert getattr(client, method)(address) in (None, 0)
        exchange(server, client, b"tcp")

@pytest.mark.parametrize("method", ["sendto", "sendmsg"])
@pytest.mark.parametrize("family,host", [(socket.AF_INET, "127.0.0.1"),
                                         (socket.AF_INET6, "::1")])
def test_udp(method, family, host):
    if not hasattr(socket.socket, method):
        pytest.skip("Socket method unavailable")
    with (
        socket.socket(family, socket.SOCK_DGRAM) as server,
        socket.socket(family, socket.SOCK_DGRAM) as client,
    ):
        bind_loopback(server, host)
        client.settimeout(0.5)
        address = server.getsockname()
        args = (b"udp", address) if method == "sendto" else ([b"udp"], [], 0, address)
        assert getattr(client, method)(*args) == 3
        assert server.recv(3) == b"udp"

def test_dns():
    for host in ("localhost", "127.0.0.1", b"localhost"):
        assert socket.getaddrinfo(host, 80)
        assert socket.gethostbyname(host).startswith("127.")
        assert socket.gethostbyname_ex(host)[2]
    assert socket.getaddrinfo("::1", 80)
    assert socket.getaddrinfo(None, 0, flags=socket.AI_PASSIVE)
    assert socket.gethostbyaddr("127.0.0.1")[2]
    address = socket.getnameinfo(("127.0.0.1", 80), socket.NI_NUMERICHOST)
    assert address[0] == "127.0.0.1"

def test_unix():
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("Unix sockets unavailable")
    with (
        tempfile.TemporaryDirectory() as root,
        socket.socket(socket.AF_UNIX) as server,
        socket.socket(socket.AF_UNIX) as client,
    ):
        server.settimeout(0.5)
        client.settimeout(0.5)
        address = root + "/local.sock"
        server.bind(address)
        server.listen()
        client.connect(address)
        exchange(server, client, b"unix")
"""
RESTORE_PROBE = (
    OFFLINE_PROBE
    + """
import inspect

@pytest.mark.live
def test_live_restores_all_socket_apis():
    # Built-in equivalence avoids sending a packet to prove live opt-in.
    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        if hasattr(socket.socket, name):
            assert inspect.ismethoddescriptor(getattr(socket.socket, name))
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr",
                 "getnameinfo"):
        method = getattr(socket, name)
        if name == "getaddrinfo":
            assert method.__globals__["getaddrinfo"] is method
        else:
            assert inspect.isbuiltin(method)
    assert socket.getaddrinfo("0.0.0.0", 0, flags=socket.AI_NUMERICHOST)

def test_offline_after_live():
    rejected()
"""
)
LIVE_PROBE = """
import pytest
from tests.helpers.embedding_config import get_embedding_config_for_tests

@pytest.mark.live
def test_declared_live():
    data = get_embedding_config_for_tests()
    assert data["provider"] == "openai"
    assert data["api_key"] == "explicit-test-input"
"""


def _probe_command(probe: Path, options: tuple[str, ...]) -> list[str]:
    return [
        sys.executable,
        "-c",
        SESSION_RUNNER,
        "-p",
        "tests.conftest",
        "-c",
        str(ROOT / "pyproject.toml"),
        "--confcutdir",
        str(probe.parent),
        str(probe),
        "-q",
        "-n",
        "0",
        f"--junitxml={probe.parent / 'report.xml'}",
        *options,
    ]


def _run_probe(
    root: Path, source: str, *options: str
) -> subprocess.CompletedProcess[str]:
    probe = root / "test_probe.py"
    probe.write_text(source)
    return subprocess.run(
        _probe_command(probe, options),
        cwd=root,
        env=dict(
            os.environ, PYTHONPATH=str(ROOT), OPENAI_API_KEY="ambient-must-not-be-used"
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("options", [(), ("--run-live",)])
def test_default_rejects_outbound_before_collection_and_in_session(
    tmp_path: Path, options: tuple[str, ...]
) -> None:
    result = _run_probe(tmp_path, OFFLINE_PROBE, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _junit_counts(tmp_path) == (1, 0)


def test_default_excludes_real_provider_tests(tmp_path: Path) -> None:
    result = _run_probe(tmp_path, LIVE_PROBE)
    assert "deselected" in result.stdout, result.stdout + result.stderr
    assert "ambient-must-not-be-used" not in result.stdout


@pytest.mark.parametrize(
    "data", [None, {"embedding": {}}, {"embedding": {"provider": "openai"}}]
)
def test_requested_live_missing_prerequisites_fails_not_skips(
    tmp_path: Path, data: Any
) -> None:
    options = ["--run-live"]
    if data is not None:
        path = tmp_path / "live.json"
        path.write_text(json.dumps(data))
        options.extend(["--live-config", str(path)])
    result = _run_probe(tmp_path, LIVE_PROBE, *options)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Live prerequisite missing" in result.stdout
    assert "skipped" not in result.stdout


@pytest.mark.parametrize("workers", [0, 2])
def test_explicit_live_input_survives_worker_and_environment_isolation(
    tmp_path: Path, workers: int
) -> None:
    path = tmp_path / "live.json"
    path.write_text(
        json.dumps(
            {"embedding": {"provider": "openai", "api_key": "explicit-test-input"}}
        )
    )
    result = _run_probe(
        tmp_path,
        LIVE_PROBE,
        "--run-live",
        "--live-config",
        str(path),
        "-n",
        str(workers),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _junit_counts(tmp_path) == (1, 0)


def test_loopback_tcp_udp_dns_and_unix_remain_available(tmp_path: Path) -> None:
    result = _run_probe(tmp_path, LOCAL_PROBE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout


def test_live_restores_all_apis_then_reapplies_offline_guard(tmp_path: Path) -> None:
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"environment": {"OPENAI_API_KEY": "test-input"}}))
    result = _run_probe(
        tmp_path, RESTORE_PROBE, "--run-live", "--live-config", str(path)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _junit_counts(tmp_path) == (3, 0)


def test_unsupported_live_environment_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"environment": {"DASHSCOPE_API_KEY": "x"}}))
    result = _run_probe(
        tmp_path, LIVE_PROBE, "--run-live", "--live-config", str(path)
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Unsupported live environment input" in result.stdout


def test_live_input_requires_absolute_path(tmp_path: Path) -> None:
    result = _run_probe(
        tmp_path, LIVE_PROBE, "--run-live", "--live-config", "live.json"
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "absolute JSON path" in result.stderr

"""Unit tests for ``DaemonDiscovery._build_forwarded_args``.

Guards the argv reconstruction contract for the mcp → _daemon proxy.
Specifically pins the exclusion of remote-config flags: `_daemon` is in
``chunkhound.core.config.remote._SUBPROCESS_SKIP`` and never runs the
fetch, so forwarding those flags would leak the Authorization header
into ``/proc/<pid>/cmdline`` for zero functional benefit.
"""

from __future__ import annotations

import argparse

from chunkhound.api.cli.parsers.daemon_parser import add_daemon_subparser
from chunkhound.daemon.discovery import DaemonDiscovery


def _parse_daemon_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_daemon_subparser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["_daemon", *argv])


def test_daemon_argv_excludes_remote_config_auth_header() -> None:
    args = _parse_daemon_args(
        [
            "--project-dir",
            "/tmp/proj",
            "--socket-path",
            "/tmp/sock",
            "--remote-config-auth-header",
            "Bearer sekret",
        ]
    )
    forwarded = DaemonDiscovery._build_forwarded_args(args)
    assert "--remote-config-auth-header" not in forwarded
    assert "Bearer sekret" not in forwarded


def test_daemon_argv_excludes_remote_config_url() -> None:
    args = _parse_daemon_args(
        [
            "--project-dir",
            "/tmp/proj",
            "--socket-path",
            "/tmp/sock",
            "--remote-config-url",
            "https://example.com/config",
        ]
    )
    forwarded = DaemonDiscovery._build_forwarded_args(args)
    assert "--remote-config-url" not in forwarded
    assert "https://example.com/config" not in forwarded


def test_daemon_argv_still_forwards_other_common_flags() -> None:
    # Regression: the remote-config skips must not accidentally suppress
    # sibling common flags (--verbose, --debug, --config).
    args = _parse_daemon_args(
        [
            "--project-dir",
            "/tmp/proj",
            "--socket-path",
            "/tmp/sock",
            "--verbose",
            "--debug",
        ]
    )
    forwarded = DaemonDiscovery._build_forwarded_args(args)
    assert "--verbose" in forwarded
    assert "--debug" in forwarded

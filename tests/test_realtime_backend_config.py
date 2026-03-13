"""Focused tests for realtime backend config parsing and forwarding."""

import argparse
from pathlib import Path

from chunkhound.api.cli.parsers.daemon_parser import add_daemon_subparser
from chunkhound.api.cli.parsers.mcp_parser import add_mcp_subparser
from chunkhound.core.config.config import Config
from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.daemon.discovery import DaemonDiscovery
from chunkhound.watchman_runtime import loader as watchman_runtime_loader
from chunkhound.watchman_runtime.loader import (
    default_realtime_backend_for_current_install,
    default_realtime_backend_for_platform,
)


def _build_parser(add_subparser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_subparser(subparsers)
    return parser


def test_indexing_config_defaults_to_current_install_backend():
    config = IndexingConfig()
    assert config.realtime_backend == default_realtime_backend_for_current_install()


def test_default_realtime_backend_for_current_install_uses_watchdog_in_source_tree():
    assert default_realtime_backend_for_current_install() == "watchdog"


def test_default_realtime_backend_for_current_install_uses_watchman_when_payloads_ship(
    monkeypatch,
):
    monkeypatch.setattr(
        watchman_runtime_loader,
        "is_packaged_watchman_runtime_available",
        lambda **_: True,
    )

    assert default_realtime_backend_for_current_install() == "watchman"


def test_default_realtime_backend_for_supported_windows_host() -> None:
    assert (
        default_realtime_backend_for_platform(
            system_name="Windows",
            machine_name="AMD64",
        )
        == "watchman"
    )


def test_default_realtime_backend_for_unsupported_macos_host() -> None:
    assert (
        default_realtime_backend_for_platform(
            system_name="Darwin",
            machine_name="x86_64",
        )
        == "watchdog"
    )


def test_indexing_config_loads_realtime_backend_from_env(monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_INDEXING__REALTIME_BACKEND", "polling")
    config = IndexingConfig.load_from_env()
    assert config["realtime_backend"] == "polling"


def test_mcp_cli_realtime_backend_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CHUNKHOUND_INDEXING__REALTIME_BACKEND", "watchdog")
    parser = _build_parser(add_mcp_subparser)
    args = parser.parse_args(
        ["mcp", str(tmp_path), "--no-daemon", "--realtime-backend", "polling"]
    )

    config = Config(args=args)

    assert config.indexing.realtime_backend == "polling"


def test_daemon_parser_accepts_realtime_backend(tmp_path):
    parser = _build_parser(add_daemon_subparser)
    args = parser.parse_args(
        [
            "_daemon",
            "--project-dir",
            str(tmp_path),
            "--socket-path",
            "tcp:127.0.0.1:0",
            "--realtime-backend",
            "polling",
        ]
    )

    assert args.realtime_backend == "polling"


def test_daemon_forwarding_includes_realtime_backend(tmp_path):
    args = argparse.Namespace(realtime_backend="polling")
    forwarded = DaemonDiscovery(Path(tmp_path))._build_forwarded_args(args)
    assert "--realtime-backend" in forwarded
    assert "polling" in forwarded

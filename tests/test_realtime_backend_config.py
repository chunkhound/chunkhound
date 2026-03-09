"""Focused tests for realtime backend config parsing and forwarding."""

import argparse
from pathlib import Path

from chunkhound.api.cli.parsers.daemon_parser import add_daemon_subparser
from chunkhound.api.cli.parsers.mcp_parser import add_mcp_subparser
from chunkhound.core.config.config import Config
from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.daemon.discovery import DaemonDiscovery


def _build_parser(add_subparser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_subparser(subparsers)
    return parser


def test_indexing_config_defaults_to_watchdog():
    config = IndexingConfig()
    assert config.realtime_backend == "watchdog"


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

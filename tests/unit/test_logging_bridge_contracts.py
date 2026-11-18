"""Standard logging must reach application sinks without losing diagnostics."""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from loguru import logger

from chunkhound.utils.logging_bridge import bridge_root_logger, bridge_std_logger

if TYPE_CHECKING:
    from loguru import Message


@pytest.fixture
def bridged_logs() -> Iterator[tuple[logging.Logger, list[Message]]]:
    source = logging.getLogger(__name__)
    handlers, level, propagate = source.handlers[:], source.level, source.propagate
    messages: list[Message] = []
    sink = logger.add(
        messages.append, format="{message}", diagnose=False, backtrace=False
    )
    try:
        source.propagate = False
        bridge_std_logger(source.name)
        yield source, messages
    finally:
        logger.remove(sink)
        for handler in source.handlers[:]:
            if handler not in handlers:
                source.removeHandler(handler)
                handler.close()
        source.setLevel(level)
        source.propagate = propagate


@pytest.fixture
def root_bridged_logs() -> Iterator[list[Message]]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    messages: list[Message] = []
    sink = logger.add(
        messages.append, format="{message}", diagnose=False, backtrace=False
    )
    try:
        bridge_root_logger()
        yield messages
    finally:
        logger.remove(sink)
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)


def test_root_bridge_delivers_rust_and_plain_records(root_bridged_logs):
    messages = root_bridged_logs
    bridge_root_logger()  # repeated setup must not duplicate delivery
    logging.getLogger("chunkhound_native.pipeline").info("rust_marker")
    logging.getLogger("third.party").warning("plain_marker")
    assert [message.record["message"] for message in messages] == [
        "rust_marker",
        "plain_marker",
    ]
    rust, plain = messages
    assert rust.record["extra"]["rust_native"] is True
    assert plain.record["extra"]["rust_native"] is False
    assert rust.record["level"].name == "INFO"
    assert (
        plain.record["function"] == "test_root_bridge_delivers_rust_and_plain_records"
    )


@pytest.mark.parametrize(
    "level", [logging.WARNING, logging.ERROR, logging.CRITICAL, 45]
)
def test_standard_and_custom_severities_reach_application_logs(bridged_logs, level):
    source, messages = bridged_logs
    source.log(level, "delivery %s", "marker")
    (message,) = messages
    assert message.record["message"] == "delivery marker"
    assert message.record["level"].no == level
    if level != 45:
        assert message.record["level"].name == logging.getLevelName(level)
    assert (
        message.record["function"]
        == "test_standard_and_custom_severities_reach_application_logs"
    )
    assert message.record["file"].path == __file__


def test_exception_traceback_reaches_application_logs(bridged_logs):
    source, messages = bridged_logs
    try:
        raise ValueError("native failure detail")
    except ValueError:
        source.exception("native callback failed")
    (message,) = messages
    assert "native callback failed" in message
    assert "Traceback" in message
    assert "ValueError: native failure detail" in message
    assert (
        message.record["function"]
        == "test_exception_traceback_reaches_application_logs"
    )


def emit_worker_error(source: logging.Logger) -> None:
    source.error("worker failure")


def test_worker_thread_errors_reach_application_logs(bridged_logs):
    source, messages = bridged_logs
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(emit_worker_error, source).result(timeout=5)
    (message,) = messages
    assert message.record["message"] == "worker failure"
    assert message.record["function"] == "emit_worker_error"
    assert message.record["file"].path == __file__


def test_named_bridge_follows_root_verbosity():
    """--verbose must expose native watchdog INFO records; default stays quiet.

    Operators lose the native diagnostics if the named bridge pins a level,
    so the root logger threshold set from --verbose is the single source.
    """
    source = logging.getLogger("fsevents.verbosity.contract")
    root = logging.getLogger()
    handlers, level, propagate = source.handlers[:], source.level, source.propagate
    root_level = root.level
    messages: list[Message] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        root.setLevel(logging.INFO)
        bridge_std_logger(source.name)
        source.info("native_info_without_verbose")
        root.setLevel(logging.DEBUG)
        bridge_std_logger(source.name)  # verbosity changed after the first bridge
        source.info("native_info_with_verbose")
    finally:
        logger.remove(sink)
        for handler in source.handlers[:]:
            if handler not in handlers:
                source.removeHandler(handler)
                handler.close()
        source.setLevel(level)
        source.propagate = propagate
        root.setLevel(root_level)
    assert [message.record["message"] for message in messages] == [
        "native_info_with_verbose"
    ]


def test_combined_bridges_deliver_each_message_once(root_bridged_logs):
    """A named bridge must not hand the same record to the root bridge."""
    messages = root_bridged_logs
    source = logging.getLogger("combined.bridge.contract")
    handlers, level, propagate = source.handlers[:], source.level, source.propagate
    try:
        bridge_std_logger(source.name)
        source.warning("combined delivery")
        delivered = [message.record["message"] for message in messages]
        assert delivered == ["combined delivery"]
        assert source.propagate is False
    finally:
        for handler in source.handlers[:]:
            if handler not in handlers:
                source.removeHandler(handler)
                handler.close()
        source.setLevel(level)
        source.propagate = propagate


def test_repeated_bridge_setup_delivers_each_message_once(bridged_logs):
    source, messages = bridged_logs
    bridge_std_logger(source.name)
    bridge_std_logger(source.name)
    source.error("single delivery")
    assert [message.record["message"] for message in messages] == ["single delivery"]


def reject_message(message: Message) -> None:
    raise RuntimeError("sink unavailable")


def test_forwarding_failures_remain_visible(bridged_logs):
    source, _ = bridged_logs
    sink = logger.add(reject_message, catch=False)
    try:
        with pytest.raises(RuntimeError, match="sink unavailable"):
            source.error("cannot deliver")
    finally:
        logger.remove(sink)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_importing_mcp_stdio_keeps_std_logger_delivery():
    """Importing a transport must not disable process-global log delivery.

    Runs in a subprocess: the stdio module is imported during collection in
    the parent session, so only a fresh interpreter proves import purity.
    """
    script = """
import logging

from loguru import logger

messages = []
logger.add(messages.append, format="{message}")
import chunkhound.mcp_server.stdio  # noqa: F401
from chunkhound.utils.logging_bridge import bridge_std_logger

bridge_std_logger("fsevents")
logging.getLogger("fsevents").error("import_delivery_marker")
delivered = [str(message).strip() for message in messages]
assert delivered == ["import_delivery_marker"], delivered
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


def test_stdio_silencing_survives_only_inside_the_run():
    """Silence is scoped to the stdio run; sinks and disables outlive main().

    Subprocess isolation matters: main() mutates process-global loguru
    activation, so a fresh interpreter proves both the mute and the restore.
    """
    script = """
import asyncio

from loguru import logger

import chunkhound.mcp_server.stdio as stdio

messages = []
logger.add(messages.append, format="{message}")
logger.disable("embedder.quiet")


async def fake_serve(args=None, config=None):
    logger.error("during_run")
    assert messages == [], [str(message) for message in messages]


stdio._serve = fake_serve
asyncio.run(stdio.main())
logger.error("after_run")
embedder = {"__name__": "embedder.quiet", "logger": logger}
exec("def emit():\\n    logger.error('still_disabled')\\n", embedder)
embedder["emit"]()
delivered = [str(message).strip() for message in messages]
assert delivered == ["after_run"], delivered
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr

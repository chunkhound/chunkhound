from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil


def terminate_process_tree(pid: int, *, psutil_module: Any = psutil) -> None:
    try:
        root = psutil_module.Process(pid)
    except psutil_module.NoSuchProcess:
        return

    try:
        processes = root.children(recursive=True)
    except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
        processes = []
    processes.append(root)

    for process in processes:
        try:
            process.terminate()
        except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
            continue

    _, alive = psutil_module.wait_procs(processes, timeout=2.0)
    for process in alive:
        try:
            process.kill()
        except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
            continue
    psutil_module.wait_procs(alive, timeout=2.0)


def terminate_processes_using_root(
    root: Path,
    *,
    os_module: Any = os,
    psutil_module: Any = psutil,
    process_terminator: Callable[[int], None] = terminate_process_tree,
) -> None:
    root_str = str(root)
    current_pid = os_module.getpid()
    candidates: list[int] = []

    for process in psutil_module.process_iter(["pid", "cwd", "cmdline"]):
        pid = process.info.get("pid")
        if not isinstance(pid, int) or pid == current_pid:
            continue

        try:
            cwd = process.info.get("cwd")
            cmdline = process.info.get("cmdline") or []
        except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
            continue

        if isinstance(cwd, str) and cwd.startswith(root_str):
            candidates.append(pid)
            continue
        if any(isinstance(arg, str) and root_str in arg for arg in cmdline):
            candidates.append(pid)

    for pid in candidates:
        process_terminator(pid)


def remove_tree_with_retries(
    root: Path,
    *,
    attempts: int = 5,
    base_delay_seconds: float = 0.2,
    os_module: Any = os,
    shutil_module: Any = shutil,
    time_module: Any = time,
    process_root_terminator: Callable[[Path], None] = terminate_processes_using_root,
) -> None:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil_module.rmtree(root)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            last_error = error
            if os_module.name == "nt":
                process_root_terminator(root)
            if attempt == attempts - 1:
                raise
            time_module.sleep(base_delay_seconds * (attempt + 1))

    if last_error is not None:
        raise last_error

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoDocCLIExitError(Exception):
    exit_code: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    infos: tuple[str, ...] = ()

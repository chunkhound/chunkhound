from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path


def build_tree_manifest(
    root_dir: Path,
    *,
    ignore: Callable[[Path], bool] | None = None,
) -> dict[str, str]:
    """
    Return a stable mapping of relative POSIX paths -> sha256 hex digest.

    This is intended for byte-for-byte characterization tests of generated sites.
    """

    if not root_dir.exists():
        raise FileNotFoundError(str(root_dir))

    ignore_func = ignore or (lambda _path: False)

    manifest: dict[str, str] = {}
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if ignore_func(path):
            continue

        rel = path.relative_to(root_dir).as_posix()
        digest = sha256(path.read_bytes()).hexdigest()
        manifest[rel] = digest

    return manifest


from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_REQUIRED_WHEEL_PATHS: tuple[str, ...] = (
    "chunkhound/watchman_runtime/__init__.py",
    "chunkhound/watchman_runtime/README.md",
    "chunkhound/watchman_runtime/platforms/linux-x86_64/manifest.json",
    "chunkhound/watchman_runtime/platforms/linux-x86_64/bin/watchman",
    "chunkhound/watchman_runtime/platforms/macos-arm64/manifest.json",
    "chunkhound/watchman_runtime/platforms/macos-arm64/bin/watchman",
    "chunkhound/watchman_runtime/platforms/macos-x86_64/manifest.json",
    "chunkhound/watchman_runtime/platforms/macos-x86_64/bin/watchman",
    "chunkhound/watchman_runtime/platforms/windows-x86_64/manifest.json",
    "chunkhound/watchman_runtime/platforms/windows-x86_64/bin/watchman.cmd",
)


def _host_probe_command(*, binary_path: Path, probe_args: tuple[str, ...]) -> list[str]:
    if os.name == "nt":
        return ["cmd.exe", "/c", str(binary_path), *probe_args]
    return [str(binary_path), *probe_args]


def _verify_wheel_has_platform_only_tag(wheel_path: Path) -> None:
    if "-py3-none-" not in wheel_path.name or wheel_path.name.endswith("any.whl"):
        raise RuntimeError(
            "Watchman runtime wheels must use an explicit py3-none-platform tag: "
            f"{wheel_path}"
        )


def _verify_wheel_contents(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
    missing = [path for path in _REQUIRED_WHEEL_PATHS if path not in names]
    if missing:
        missing_rendered = "\n".join(f"- {item}" for item in missing)
        raise RuntimeError(
            "Wheel is missing required Watchman runtime resources: "
            f"{wheel_path}\n{missing_rendered}"
        )


def _verify_runtime_reads(*, wheel_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="chunkhound-watchman-wheel-verify-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel_path) as zf:
            zf.extractall(root)

        code = "\n".join(
            [
                "import os",
                "import subprocess",
                "import sys",
                "from pathlib import Path",
                "",
                (
                    "from chunkhound.watchman_runtime.loader import "
                    "materialize_watchman_binary, "
                    "resolve_packaged_watchman_runtime"
                ),
                "",
                "runtime = resolve_packaged_watchman_runtime()",
                (
                    "binary_path = materialize_watchman_binary("
                    "destination_root=Path('materialized'))"
                ),
                "assert binary_path.is_file()",
                "if os.name != 'nt':",
                "    assert os.access(binary_path, os.X_OK)",
                "if os.name == 'nt':",
                (
                    "    command = ['cmd.exe', '/c', str(binary_path), "
                    "*runtime.probe_args]"
                ),
                "else:",
                "    command = [str(binary_path), *runtime.probe_args]",
                (
                    "result = subprocess.run(command, check=True, capture_output=True, "
                    "text=True)"
                ),
                "assert 'watchman' in result.stdout.lower()",
                "assert 'placeholder' in result.stdout.lower()",
                "sys.exit(0)",
            ]
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Wheel runtime resource verification failed.\n"
                f"wheel={wheel_path}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Watchman runtime payloads exist in a built wheel and can be "
            "materialized into a runnable binary path."
        )
    )
    parser.add_argument(
        "wheels",
        nargs="+",
        type=Path,
        help="Path(s) to .whl file(s) to verify.",
    )
    args = parser.parse_args(argv)

    wheel_paths: list[Path] = []
    for raw in args.wheels:
        if raw.is_file() and raw.suffix == ".whl":
            wheel_paths.append(raw)
            continue
        raise FileNotFoundError(f"Wheel not found: {raw}")

    for wheel_path in wheel_paths:
        _verify_wheel_has_platform_only_tag(wheel_path)
        _verify_wheel_contents(wheel_path)
        _verify_runtime_reads(wheel_path=wheel_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

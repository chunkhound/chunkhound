from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_REQUIRED_WHEEL_PATHS: tuple[str, ...] = (
    # Prompt templates (filesystem reads via cleanup.py)
    "chunkhound/autodoc/prompts/cleanup_system_v2.txt",
    "chunkhound/autodoc/prompts/cleanup_user_v2.txt",
    "chunkhound/autodoc/prompts/cleanup_user_end_user_v1.txt",
    # Packaged Astro assets (read via importlib.resources)
    "chunkhound/autodoc/assets/__init__.py",
    "chunkhound/autodoc/assets/astro.config.mjs",
    "chunkhound/autodoc/assets/tsconfig.json",
    "chunkhound/autodoc/assets/README.md",
    "chunkhound/autodoc/assets/package.json",
    "chunkhound/autodoc/assets/src/layouts/DocLayout.astro",
    "chunkhound/autodoc/assets/src/styles/global.css",
    "chunkhound/autodoc/assets/public/favicon.ico",
)


def _verify_wheel_contents(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
    missing = [p for p in _REQUIRED_WHEEL_PATHS if p not in names]
    if missing:
        missing_rendered = "\n".join(f"- {item}" for item in missing)
        raise RuntimeError(
            "Wheel is missing required AutoDoc resources: "
            f"{wheel_path}\n{missing_rendered}"
        )


def _verify_runtime_reads(*, wheel_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="chunkhound-wheel-verify-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel_path) as zf:
            zf.extractall(root)

        body = "## Overview\\nB"
        code = "\n".join(
            [
                "from chunkhound.autodoc.cleanup import _build_cleanup_prompt",
                "from chunkhound.autodoc.template_loader import load_bytes, load_text",
                "",
                'assert "navData" in load_text("src/layouts/DocLayout.astro")',
                'assert load_text("src/styles/global.css").strip()',
                'assert load_bytes("public/favicon.ico")',
                "",
                f"_build_cleanup_prompt(title='T', body='{body}', audience='balanced')",
                f"_build_cleanup_prompt(title='T', body='{body}', audience='end-user')",
            ]
        )

        env = os.environ.copy()
        # Ensure we load from extracted wheel contents, not from the repo working tree.
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
            "Verify AutoDoc packaged assets + prompt templates exist in a built wheel."
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
        _verify_wheel_contents(wheel_path)
        _verify_runtime_reads(wheel_path=wheel_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Install the matching main/native wheels together and run the native smoke check.

Requires uv on PATH. From a checkout with downloaded artifacts, run:
    uv run --no-project python scripts/validate_release_wheels.py \
        --main-dist main-dist --native-dist native-dist

The wheel directories may also be absolute paths. Validation uses Python 3.12
in an isolated uv environment outside the checkout, never the local project
installation. The real smoke check is shared with native-wheel builds.
"""

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-dist", type=Path, default=Path("main-dist"))
    parser.add_argument("--native-dist", type=Path, default=Path("native-dist"))
    args = parser.parse_args()

    wheels: list[Path] = []
    for label, directory in (("main", args.main_dist), ("native", args.native_dist)):
        matches = sorted(
            path.resolve() for path in directory.glob("*.whl") if path.is_file()
        )
        if len(matches) != 1:
            raise SystemExit(
                f"Expected exactly one matching {label} wheel, found {len(matches)}"
            )
        wheels.append(matches[0])

    smoke_script = Path(__file__).resolve().with_name("smoke_test_native_wheel.py")
    with TemporaryDirectory(prefix="chunkhound-release-validation-") as temporary:
        validation_dir = Path(temporary).resolve()
        if validation_dir.is_relative_to(smoke_script.parent.parent):
            raise SystemExit("Set the system temporary directory outside the checkout")
        subprocess.run(
            ["uv", "python", "install", "3.12"], cwd=validation_dir, check=True
        )
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--isolated",
                "--python",
                "3.12",
                "--with",
                str(wheels[0]),
                "--with",
                str(wheels[1]),
                "python",
                str(smoke_script),
            ],
            cwd=validation_dir,
            check=True,
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc

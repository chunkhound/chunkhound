"""Verify every built wheel's ZIP contents match its RECORD before upload.

PyPI rejects newly uploaded wheels whose archive contents do not match the
included ``RECORD`` file (enforced since 2026-02-01; previously a warning).
The Linux native wheels are the ones at risk: ``auditwheel repair`` repacks
them with its own ``dir2zip`` helper, which writes explicit directory members
that maturin's macOS/Windows repair never emits. A rebuild step that derives
RECORD from ``ZipFile.infolist()`` therefore lists those directories while the
RECORD spec only ever describes files.

This mirrors warehouse's check
(``warehouse/forklift/legacy.py``: RECORD entries, backslash-normalized, must
equal the archive's non-directory members) so a bad wheel fails here instead
of at the PyPI upload step, which is the point of no return.

Usage:
    python scripts/validate_wheel_record.py dist-native [more-dirs ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path


def record_entries(archive: zipfile.ZipFile) -> set[str]:
    """Return the path names listed in the wheel's RECORD file."""
    records = [
        name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
    ]
    if len(records) != 1:
        raise ValueError(f"expected exactly one RECORD file, found {len(records)}")
    text = archive.read(records[0]).decode("utf-8")
    # RECORD files may use either separator (see the wheel specification).
    return {row[0].replace("\\", "/") for row in csv.reader(text.splitlines()) if row}


def zip_entries(archive: zipfile.ZipFile) -> set[str]:
    """Return the archive's file members, excluding directory entries."""
    return {info.filename for info in archive.infolist() if not info.is_dir()}


def check_wheel(wheel: Path) -> list[str]:
    """Return a list of problems with ``wheel``; empty means it is uploadable."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            listed = record_entries(archive)
            members = zip_entries(archive)
    except (OSError, UnicodeDecodeError, ValueError, zipfile.BadZipFile) as exc:
        return [f"{wheel.name}: cannot read wheel: {exc}"]

    problems = []
    for missing in sorted(members - listed):
        problems.append(
            f"{wheel.name}: present in archive but missing from RECORD: {missing}"
        )
    for extra in sorted(listed - members):
        problems.append(
            f"{wheel.name}: listed in RECORD but not an archive file member: {extra}"
        )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        type=Path,
        nargs="+",
        help="Directories to scan for *.whl files.",
    )
    args = parser.parse_args()

    wheels = sorted(
        wheel
        for directory in args.directories
        for wheel in directory.glob("*.whl")
        if wheel.is_file()
    )
    if not wheels:
        raise SystemExit("No wheels found to validate")

    problems = [problem for wheel in wheels for problem in check_wheel(wheel)]
    if problems:
        print("Wheels whose RECORD does not match their contents:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Validated {len(wheels)} wheel(s) against their RECORD.")


if __name__ == "__main__":
    main()

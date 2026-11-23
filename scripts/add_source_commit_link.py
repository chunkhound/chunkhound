"""Add the 'Source commit' Project-URL to a built wheel.

The main package gets this link from hatch_build's metadata hook. Native wheels
are built by maturin with pyproject.toml hidden, so that hook never runs for
them. This reuses hatch_build's helpers to write the same link into the wheel's
METADATA and refreshes its RECORD hashes.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hatch_build  # noqa: E402


def add_source_commit_link(wheel: Path, repo_root: Path) -> None:
    sha = hatch_build._source_commit(repo_root)
    if sha is None:
        raise SystemExit("Cannot add a source commit link outside a git checkout")
    line = (
        f"Project-URL: {hatch_build._SOURCE_COMMIT_URL_LABEL}, "
        f"{hatch_build._REPOSITORY_URL}/commit/{sha}\n"
    )

    with zipfile.ZipFile(wheel) as archive:
        entries = {
            info.filename: (info, archive.read(info)) for info in archive.infolist()
        }
    metadata = next(name for name in entries if name.endswith(".dist-info/METADATA"))
    record = next(name for name in entries if name.endswith(".dist-info/RECORD"))

    existing = entries[metadata][1].decode()
    if f"Project-URL: {hatch_build._SOURCE_COMMIT_URL_LABEL}, " in existing:
        return
    header, separator, body = existing.partition("\n\n")
    entries[metadata] = (
        entries[metadata][0],
        (header + "\n" + line + separator + body).encode(),
    )

    rows = []
    for name, (_, data) in entries.items():
        if name == record:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        rows.append((name, f"sha256={digest.decode()}", str(len(data))))
    rows.append((record, "", ""))
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    entries[record] = (entries[record][0], buffer.getvalue().encode())

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as rewritten:
        for name, (info, data) in entries.items():
            rewritten.writestr(info.filename, data, compress_type=zipfile.ZIP_DEFLATED)
    wheel.write_bytes(output.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    add_source_commit_link(args.wheel, REPO_ROOT)


if __name__ == "__main__":
    main()

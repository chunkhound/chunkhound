"""The RECORD gate must match the check PyPI runs on upload.

PyPI rejects newly uploaded wheels whose ZIP contents disagree with the
included RECORD file (enforced since 2026-02-01). The Linux native wheels hit
this because auditwheel's repair repacks them with directory members that
maturin's macOS/Windows repair never writes.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from scripts import validate_wheel_record


def _write_wheel(path: Path, files: dict[str, str], directories=()) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for directory in directories:
            archive.writestr(zipfile.ZipInfo(directory), b"")
        for name, content in files.items():
            archive.writestr(name, content)


def _record_rows(files: dict[str, str], directories=()) -> str:
    """Build a RECORD the way a buggy repack does: directories included."""
    rows = []
    for name in list(directories) + list(files):
        data = files.get(name, "")
        digest = base64.urlsafe_b64encode(hashlib.sha256(data.encode()).digest())
        rows.append((name, f"sha256={digest.rstrip(b'=').decode()}", str(len(data))))
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def _wheel_with_record(
    path: Path, record: str, directories=(), extra_files: dict[str, str] | None = None
) -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg-1.0.dist-info/METADATA": "Name: pkg\nVersion: 1.0\n",
        "pkg-1.0.dist-info/RECORD": record,
    }
    files.update(extra_files or {})
    _write_wheel(path, files, directories)


def test_accepts_wheel_whose_record_lists_only_files(tmp_path: Path) -> None:
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    files = {
        "pkg/__init__.py": "",
        "pkg-1.0.dist-info/METADATA": "Name: pkg\nVersion: 1.0\n",
    }
    _wheel_with_record(
        wheel,
        _record_rows(files) + "pkg-1.0.dist-info/RECORD,,\n",
    )

    assert validate_wheel_record.check_wheel(wheel) == []


def test_rejects_directory_members_leaked_into_record(tmp_path: Path) -> None:
    """The exact failure mode of the auditwheel-repaired Linux wheels."""
    wheel = tmp_path / "pkg-1.0-cp310-abi3-manylinux_2_31_x86_64.whl"
    directories = ("pkg/", "pkg.libs/", "pkg-1.0.dist-info/")
    files = {
        "pkg/__init__.py": "",
        "pkg.libs/libduckdb-deadbeef.so": "so",
        "pkg-1.0.dist-info/METADATA": "Name: pkg\nVersion: 1.0\n",
    }
    _wheel_with_record(
        wheel,
        _record_rows(files, directories) + "pkg-1.0.dist-info/RECORD,,\n",
        directories,
        extra_files={"pkg.libs/libduckdb-deadbeef.so": "so"},
    )

    problems = validate_wheel_record.check_wheel(wheel)

    assert problems
    assert all("not an archive file member" in problem for problem in problems)
    assert {problem.rsplit(": ", 1)[1] for problem in problems} == set(directories)


def test_rejects_file_missing_from_record(tmp_path: Path) -> None:
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    files = {"pkg/__init__.py": ""}
    _wheel_with_record(wheel, _record_rows(files) + "pkg-1.0.dist-info/RECORD,,\n")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("pkg/extra.py", "")

    problems = validate_wheel_record.check_wheel(wheel)

    assert problems
    assert any("missing from RECORD" in problem for problem in problems)


def test_rejects_wheel_without_record(tmp_path: Path) -> None:
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    _write_wheel(wheel, {"pkg/__init__.py": ""})

    problems = validate_wheel_record.check_wheel(wheel)

    assert problems
    assert "expected exactly one RECORD file" in problems[0]


def test_main_fails_when_no_wheels_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["validate_wheel_record.py", str(tmp_path)])

    with pytest.raises(SystemExit, match="No wheels found"):
        validate_wheel_record.main()

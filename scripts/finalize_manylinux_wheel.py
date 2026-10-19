"""Post-process a repaired Linux chunkhound_native wheel and enforce its glibc floor.

Run inside the manylinux_2_31 build container (needs `readelf`, `objdump` and
`patchelf` on PATH) right after `maturin build --auditwheel repair`:

    python scripts/finalize_manylinux_wheel.py <wheel> --max-glibc 2.31

1. Clears the glibc symbol version from Rust std's weak `statx`/`getrandom`
   references in the extension module. Rust references both as true ELF weak
   symbols so it can fall back to a raw syscall when the glibc wrapper is
   missing; the version stamped on them is a build-host artifact (`statx`
   debuted in glibc 2.28), not a real requirement. patchelf runs only when the
   versioned reference is actually present, and any patchelf failure is fatal.
2. Fails if any ELF file in the wheel (extension + bundled .libs/) still
   requires a GLIBC_x.y symbol version above --max-glibc. This is the check
   that keeps the wheel's manylinux tag honest.
3. Rewrites the wheel with RECORD hashes/sizes updated for the patched files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

WEAK_SYMBOLS_TO_UNVERSION = ("statx", "getrandom")
_GLIBC_VERSION_RE = re.compile(r"GLIBC_(\d+(?:\.\d+)+)")


def _parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _is_elf(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"\x7fELF"


def _run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def _has_versioned_undefined_ref(elf: Path, symbol: str) -> bool:
    pattern = re.compile(rf"\bUND\s+{re.escape(symbol)}@GLIBC_")
    return bool(pattern.search(_run("readelf", "--dyn-syms", "-W", str(elf))))


def _required_glibc_versions(elf: Path) -> set[tuple[int, ...]]:
    return {
        _parse_version(match)
        for match in _GLIBC_VERSION_RE.findall(_run("objdump", "-T", str(elf)))
    }


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _rewrite_wheel(wheel: Path, root: Path, patched: set[str]) -> None:
    with zipfile.ZipFile(wheel) as zin:
        entries = [(info, zin.read(info.filename)) for info in zin.infolist()]

    record_names = [
        info.filename
        for info, _ in entries
        if PurePosixPath(info.filename).parent.name.endswith(".dist-info")
        and PurePosixPath(info.filename).name == "RECORD"
    ]
    if len(record_names) != 1:
        raise SystemExit(f"Expected exactly one dist-info RECORD in {wheel}")
    record_name = record_names[0]

    original = {info.filename: data for info, data in entries}
    new_data = {name: (root / name).read_bytes() for name in patched}
    rows = list(csv.reader(io.StringIO(original[record_name].decode())))
    for row in rows:
        if row and row[0] in new_data:
            row[1] = _record_hash(new_data[row[0]])
            row[2] = str(len(new_data[row[0]]))
    record_buffer = io.StringIO()
    csv.writer(record_buffer, lineterminator="\n").writerows(rows)
    new_data[record_name] = record_buffer.getvalue().encode()

    tmp_wheel = wheel.with_suffix(".whl.tmp")
    with zipfile.ZipFile(tmp_wheel, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in entries:
            zout.writestr(info, new_data.get(info.filename, data))
    tmp_wheel.replace(wheel)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--max-glibc", required=True, help="e.g. 2.31")
    args = parser.parse_args()
    max_glibc = _parse_version(args.max_glibc)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.wheel) as zin:
            zin.extractall(root)

        elf_files = sorted(
            path for path in root.rglob("*.so*") if path.is_file() and _is_elf(path)
        )
        extensions = [path for path in elf_files if path.name.endswith(".abi3.so")]
        if len(extensions) != 1:
            raise SystemExit(
                f"Expected exactly one *.abi3.so extension in {args.wheel}, "
                f"found {[p.relative_to(root).as_posix() for p in extensions]}"
            )
        extension = extensions[0]

        patched: set[str] = set()
        for symbol in WEAK_SYMBOLS_TO_UNVERSION:
            if _has_versioned_undefined_ref(extension, symbol):
                _run("patchelf", "--clear-symbol-version", symbol, str(extension))
                patched.add(extension.relative_to(root).as_posix())
                print(f"cleared glibc version on weak symbol {symbol!r}")
                if _has_versioned_undefined_ref(extension, symbol):
                    raise SystemExit(f"patchelf did not clear {symbol!r} version")

        violations = []
        for elf in elf_files:
            versions = _required_glibc_versions(elf)
            highest = max(versions, default=())
            rel = elf.relative_to(root).as_posix()
            rendered = ".".join(map(str, highest)) if highest else "none"
            print(f"{rel}: highest required GLIBC_{rendered}")
            if highest > max_glibc:
                violations.append(f"{rel} requires GLIBC_{rendered}")
        if violations:
            raise SystemExit(
                f"Wheel exceeds glibc {args.max_glibc} floor: " + "; ".join(violations)
            )

        if patched:
            _rewrite_wheel(args.wheel, root, patched)
    return 0


if __name__ == "__main__":
    sys.exit(main())

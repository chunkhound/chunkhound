from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PRE_COMMIT_PACKAGE = "pre-commit==4.2.0"
# Keep in sync with .pre-commit-config.yaml ruff-pre-commit rev tag
RUFF_PACKAGE = "ruff==0.13.3"
GITHUB_EVENT_NAME_ENV = "CHUNKHOUND_GITHUB_EVENT_NAME"
GITHUB_BASE_SHA_ENV = "CHUNKHOUND_GITHUB_BASE_SHA"
GITHUB_HEAD_SHA_ENV = "CHUNKHOUND_GITHUB_HEAD_SHA"
GITHUB_DEFAULT_BRANCH_ENV = "CHUNKHOUND_GITHUB_DEFAULT_BRANCH"
HOOK_MARKER = "chunkhound-managed-pre-commit-hook"
# Prefix for temp dirs holding staged snapshots; shared with test fake script.
STAGED_SNAPSHOT_PREFIX = "chunkhound-staged-ruff-"


def _git_output(*args: str, check: bool = True, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
        input=stdin,
    )
    return result.stdout.strip()


def _git_output_bytes(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=False,
    )
    return result.stdout


def _uv_tool_command(package: str, executable: str, *args: str) -> list[str]:
    # Resolve to absolute path; respects PATHEXT on Windows for .cmd/.bat
    uv = shutil.which("uv") or "uv"
    return [uv, "tool", "run", "--from", package, executable, *args]


def _pre_commit_command(*args: str) -> list[str]:
    return _uv_tool_command(PRE_COMMIT_PACKAGE, "pre-commit", *args)


def _managed_hook_path() -> Path:
    return Path(_git_output("rev-parse", "--git-path", "hooks/pre-commit"))


def _managed_hook_contents() -> str:
    return "\n".join(
        [
            "#!/bin/sh",
            f"# {HOOK_MARKER}",
            "set -eu",
            "repo_root=$(git rev-parse --show-toplevel)",
            'cd "$repo_root"',
            "exec uv run python scripts/pre_commit.py run-staged",
            "",
        ]
    )


def _print_subprocess_failure(error: subprocess.CalledProcessError) -> int:
    if error.cmd:
        command = " ".join(str(part) for part in error.cmd)
        print(f"Command failed: {command}", file=sys.stderr)
    if error.stdout:
        print(error.stdout.rstrip(), file=sys.stderr)
    if error.stderr:
        print(error.stderr.rstrip(), file=sys.stderr)
    return error.returncode or 1


def _git_changed_python_files(*range_args: str) -> list[str]:
    output = subprocess.run(
        [
            "git",
            "diff",
            *range_args,
            "--diff-filter=d",
            "--name-only",
            "-z",
            "--",
            "*.py",
            "*.pyi",
        ],
        check=True,
        capture_output=True,
        text=False,
    ).stdout
    return [part.decode() for part in output.split(b"\0") if part]


def _install(overwrite_hook: bool) -> int:
    hook_path = _managed_hook_path()
    hook_contents = _managed_hook_contents()

    if hook_path.exists():
        existing = hook_path.read_text(encoding="utf-8")
        if HOOK_MARKER not in existing and not overwrite_hook:
            print(
                f"Refusing to overwrite existing Git hook: {hook_path}",
                file=sys.stderr,
            )
            print(
                "Chain `uv run python scripts/pre_commit.py run-staged` from that hook "
                "or rerun with --overwrite-hook.",
                file=sys.stderr,
            )
            return 1

    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(hook_contents, encoding="utf-8")
    hook_path.chmod(0o755)
    print(f"Installed managed pre-commit hook at {hook_path}")
    return 0


def _run_files(files: list[str]) -> int:
    if not files:
        print("No files provided for pre-commit run.", file=sys.stderr)
        return 1

    subprocess.run(_pre_commit_command("run", "--files", *files), check=True)
    return 0


def _git_ref_exists(ref: str) -> bool:
    try:
        _git_output("rev-parse", "--verify", ref)
    except subprocess.CalledProcessError:
        return False
    return True


def _merge_base(base: str, head: str) -> str:
    return _git_output("merge-base", base, head)


def _workflow_dispatch_diff_base(default_branch: str, head: str) -> str:
    default_branch_ref = f"refs/remotes/origin/{default_branch}"
    if not _git_ref_exists(default_branch_ref):
        raise RuntimeError(
            "workflow_dispatch requires a safe diff base; "
            f"missing remote default branch ref {default_branch_ref!r}"
        )
    return _merge_base(default_branch_ref, head)


def _diff_base(event_name: str, base: str, head: str) -> str:
    # PR base may diverge from target; merge-base isolates only the PR's changes.
    if event_name == "pull_request":
        return _merge_base(base, head)
    return base


def _changed_python_files(event_name: str, base: str, head: str) -> list[str]:
    return _git_changed_python_files(_diff_base(event_name, base, head), head)


def _github_actions_diff_range() -> tuple[str, str, str]:
    event_name = os.environ.get(GITHUB_EVENT_NAME_ENV, "workflow_dispatch").strip()
    base = os.environ.get(GITHUB_BASE_SHA_ENV, "").strip()
    head = os.environ.get(GITHUB_HEAD_SHA_ENV, "").strip() or _git_output(
        "rev-parse", "HEAD"
    )

    if base:
        return event_name, base, head
    if event_name == "workflow_dispatch":
        default_branch = os.environ.get(GITHUB_DEFAULT_BRANCH_ENV, "").strip()
        if not default_branch:
            raise RuntimeError(
                "workflow_dispatch requires an explicit safe diff base; "
                f"set {GITHUB_BASE_SHA_ENV} or {GITHUB_DEFAULT_BRANCH_ENV}"
            )
        return event_name, _workflow_dispatch_diff_base(default_branch, head), head

    raise RuntimeError(
        f"{GITHUB_BASE_SHA_ENV} must be set for GitHub event {event_name!r}"
    )


def _resolve_changed_files(args: argparse.Namespace, command: str) -> list[str]:
    if args.github_actions:
        event_name, base, head = _github_actions_diff_range()
    else:
        if args.event_name is None or args.base is None or args.head is None:
            raise RuntimeError(
                f"{command} requires --event-name, --base, and --head "
                "unless --github-actions is used"
            )
        event_name, base, head = args.event_name, args.base, args.head
    return _changed_python_files(event_name, base, head)


def _ruff_command(*args: str) -> list[str]:
    return _uv_tool_command(RUFF_PACKAGE, "ruff", *args)


def _run_ruff_command(*args: str, cwd: Path | None = None) -> int:
    result = subprocess.run(_ruff_command(*args), check=False, cwd=cwd)
    if result.returncode in (0, 1):
        return result.returncode
    raise subprocess.CalledProcessError(result.returncode, result.args)


def _file_digests(files: list[str]) -> dict[str, str | None]:
    digests: dict[str, str | None] = {}
    for file in files:
        path = Path(file)
        if not path.exists():
            digests[file] = None
            continue
        digests[file] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _rewritten_files(
    before: dict[str, str | None], after: dict[str, str | None]
) -> list[str]:
    return [file for file, digest in after.items() if digest != before.get(file)]


def _print_rewritten_files(message: str, files: list[str]) -> None:
    print(message, file=sys.stderr)
    for path in files:
        print(f"  {path}", file=sys.stderr)


def _lint_rewrite_message(ci_mode: bool) -> str:
    """Return the error message for CI vs local hook rewrite detection.

    ci_mode=True means rewrites are forbidden (CI validation path).
    ci_mode=False means rewrites are expected (local staged hook path).
    """
    if ci_mode:
        return (
            "Ruff rewrote changed Python files in CI. "
            "Run Ruff locally, commit the fixes, and push again:"
        )
    return "Ruff rewrote staged Python files. Review the staged diff and commit again:"


# Snapshot files live in a temp directory, so Ruff can't auto-discover
# pyproject.toml.  The CI path runs on worktree files and doesn't need this.
def _repo_ruff_args(command: str, *args: str) -> list[str]:
    pyproject = Path("pyproject.toml")
    if pyproject.is_file():
        return [command, "--config", str(pyproject.resolve()), *args]
    return [command, *args]


def _copy_tracked_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _tracked_python_modes(files: list[str]) -> dict[str, str]:
    output = subprocess.run(
        ["git", "ls-files", "--stage", "-z", "--", *files],
        check=True,
        capture_output=True,
        text=False,
    ).stdout
    entries = [part for part in output.split(b"\0") if part]
    modes: dict[str, str] = {}
    for entry in entries:
        meta, path_bytes = entry.split(b"\t", 1)
        mode, _object_id, stage = meta.decode().split()
        if stage != "0":
            raise RuntimeError(
                "run-staged does not support unmerged files. Resolve conflicts first."
            )
        path = path_bytes.decode()
        modes[path] = mode
    missing = [file for file in files if file not in modes]
    if missing:
        raise RuntimeError(
            "run-staged only supports tracked staged files; missing index entry for: "
            + ", ".join(sorted(missing))
        )
    return modes


def _unstaged_python_files(files: list[str]) -> set[str]:
    """Detect files with unstaged changes so we can preserve their worktree state."""
    output = subprocess.run(
        ["git", "diff", "--name-only", "-z", "--", *files],
        check=True,
        capture_output=True,
        text=False,
    ).stdout
    return {part.decode() for part in output.split(b"\0") if part}


def _write_index_snapshot(temp_root: Path, files: list[str]) -> dict[str, Path]:
    snapshots: dict[str, Path] = {}
    for file in files:
        snapshot_path = temp_root / file
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(_git_output_bytes("show", f":{file}"))
        snapshots[file] = snapshot_path
    return snapshots


def _update_index_from_snapshot(
    files: list[str], snapshots: dict[str, Path], modes: dict[str, str]
) -> None:
    # Compute all blob hashes first, then issue one update-index invocation
    # (batched via --index-info stdin). If hash-object fails mid-way, the
    # index is untouched.
    lines: list[str] = []
    for file in files:
        object_id = _git_output("hash-object", "-w", "--", str(snapshots[file]))
        lines.append(f"{modes[file]} {object_id} 0\t{file}")
    subprocess.run(
        ["git", "update-index", "--index-info"],
        # Use bytes mode (no text=True) to prevent Python from translating
        # \n to \r\n on Windows, which would corrupt --index-info records.
        input="\n".join(lines).encode("utf-8"),
        check=True,
    )


def _run_ruff_on_snapshot(
    files: list[str],
    snapshots: dict[str, Path],
    temp_root: Path,
) -> tuple[int, list[str]]:
    snapshot_paths = [str(snapshots[file]) for file in files]
    before = _file_digests(snapshot_paths)

    exit_code = 0
    # Ruff must see repo-relative paths so path-based settings like
    # per-file-ignores match the same files as normal repo runs.
    check_args = _repo_ruff_args("check", "--fix", "--", *files)
    if _run_ruff_command(*check_args, cwd=temp_root) != 0:
        exit_code = 1
    format_args = _repo_ruff_args("format", "--", *files)
    if _run_ruff_command(*format_args, cwd=temp_root) != 0:
        exit_code = 1

    changed_snapshot_paths = _rewritten_files(before, _file_digests(snapshot_paths))
    changed_files = [
        file
        for file, snapshot_path in snapshots.items()
        if str(snapshot_path) in changed_snapshot_paths
    ]
    return exit_code, changed_files


def _sync_clean_worktree_files(
    changed_files: list[str],
    snapshots: dict[str, Path],
    partially_staged_rewrites: set[str],
) -> None:
    # Sync fully-staged files so the worktree matches what will be committed;
    # skip partially-staged files to preserve the user's unstaged edits.
    for file in changed_files:
        if file in partially_staged_rewrites:
            continue
        _copy_tracked_file(snapshots[file], Path(file))


def _print_partial_staging_note(partially_staged: list[str]) -> None:
    if not partially_staged:
        return
    print(
        "Partially staged files were fixed in the index only; "
        "unstaged edits were left untouched:",
        file=sys.stderr,
    )
    for path in partially_staged:
        print(f"  {path}", file=sys.stderr)


def _run_staged() -> int:
    files = _git_changed_python_files("--cached")
    if not files:
        print("No staged Python files found.")
        return 0

    # Ruff must see the staged snapshot, not the live worktree, or partially
    # staged files would accidentally pull unrelated edits into the commit.
    tracked_modes = _tracked_python_modes(files)
    originally_partially_staged = _unstaged_python_files(files)
    with tempfile.TemporaryDirectory(prefix=STAGED_SNAPSHOT_PREFIX) as temp_dir:
        temp_root = Path(temp_dir)
        snapshots = _write_index_snapshot(temp_root, files)
        exit_code, changed_files = _run_ruff_on_snapshot(files, snapshots, temp_root)
        if not changed_files:
            return exit_code

        _update_index_from_snapshot(changed_files, snapshots, tracked_modes)
        partial_rewrites = originally_partially_staged.intersection(changed_files)
        _sync_clean_worktree_files(changed_files, snapshots, partial_rewrites)
        _print_rewritten_files(_lint_rewrite_message(ci_mode=False), changed_files)
        _print_partial_staging_note(sorted(partial_rewrites))
        return 1


def _run_ruff_format(files: list[str]) -> int:
    """Run ``ruff format --check`` (CI-only, no rewrites allowed)."""
    return _run_ruff_command("format", "--check", "--", *files)


def _run_ruff_lint_and_format(files: list[str]) -> int:
    """Run Ruff lint + format on the given files (CI path, no rewrites allowed).

    ``ruff check --fix`` rewrites are detected via SHA256 digests.
    ``ruff format`` runs with ``--check``.
    """
    # --fix must stay in sync with .pre-commit-config.yaml ruff-check args.
    if not files:
        return 0

    exit_code = 0
    before_fix = _file_digests(files)
    if _run_ruff_command("check", "--fix", "--", *files) != 0:
        exit_code = 1

    changed_files = _rewritten_files(before_fix, _file_digests(files))
    if changed_files:
        _print_rewritten_files(_lint_rewrite_message(ci_mode=True), changed_files)
        exit_code = 1

    if _run_ruff_format(files) != 0:
        exit_code = 1
    return exit_code


def _add_diff_range_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--event-name",
        choices=["pull_request", "merge_group", "push", "workflow_dispatch"],
    )
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--github-actions", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install")
    install.add_argument("--overwrite-hook", action="store_true")

    run_files = subparsers.add_parser("run-files")
    run_files.add_argument("files", nargs="+")

    subparsers.add_parser("run-staged")

    run_changed = subparsers.add_parser("run-changed")
    _add_diff_range_arguments(run_changed)

    run_ruff = subparsers.add_parser("run-ruff")
    _add_diff_range_arguments(run_ruff)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "install":
            return _install(args.overwrite_hook)
        if args.command == "run-files":
            return _run_files(args.files)
        if args.command == "run-staged":
            return _run_staged()
        if args.command == "run-changed":
            files = _resolve_changed_files(args, "run-changed")
            if not files:
                print("No changed Python files found.")
                return 0
            return _run_files(files)
        if args.command == "run-ruff":
            files = _resolve_changed_files(args, "run-ruff")
            if not files:
                print("No changed Python files found.")
                return 0
            return _run_ruff_lint_and_format(files)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        return _print_subprocess_failure(error)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

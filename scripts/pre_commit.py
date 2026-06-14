from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PRE_COMMIT_PACKAGE = "pre-commit==4.2.0"
GITHUB_EVENT_NAME_ENV = "CHUNKHOUND_GITHUB_EVENT_NAME"
GITHUB_BASE_SHA_ENV = "CHUNKHOUND_GITHUB_BASE_SHA"
GITHUB_HEAD_SHA_ENV = "CHUNKHOUND_GITHUB_HEAD_SHA"
GITHUB_DEFAULT_BRANCH_ENV = "CHUNKHOUND_GITHUB_DEFAULT_BRANCH"
HOOK_MARKER = "chunkhound-managed-pre-commit-hook"


def _git_output(*args: str, check: bool = True, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
        input=stdin,
    )
    return result.stdout.strip()


def _pre_commit_command(*args: str) -> list[str]:
    # Resolve to absolute path; respects PATHEXT on Windows for .cmd/.bat
    uv = shutil.which("uv") or "uv"
    return [uv, "tool", "run", "--from", PRE_COMMIT_PACKAGE, "pre-commit", *args]


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


def _run_staged() -> int:
    files = _git_changed_python_files("--cached")
    if not files:
        print("No staged Python files found.")
        return 0
    return _run_files(files)


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


def _run_changed(event_name: str, base: str, head: str) -> int:
    files = _changed_python_files(event_name, base, head)
    if not files:
        print("No changed Python files found.")
        return 0
    return _run_files(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install")
    install.add_argument("--overwrite-hook", action="store_true")

    run_files = subparsers.add_parser("run-files")
    run_files.add_argument("files", nargs="+")

    subparsers.add_parser("run-staged")

    run_changed = subparsers.add_parser("run-changed")
    run_changed.add_argument(
        "--event-name",
        choices=["pull_request", "merge_group", "push", "workflow_dispatch"],
    )
    run_changed.add_argument("--base")
    run_changed.add_argument("--head")
    run_changed.add_argument("--github-actions", action="store_true")

    args = parser.parse_args()

    try:
        if args.command == "install":
            return _install(args.overwrite_hook)
        if args.command == "run-files":
            return _run_files(args.files)
        if args.command == "run-staged":
            return _run_staged()
        if args.command == "run-changed":
            if args.github_actions:
                event_name, base, head = _github_actions_diff_range()
            else:
                if args.event_name is None or args.base is None or args.head is None:
                    raise RuntimeError(
                        "run-changed requires --event-name, --base, and --head "
                        "unless --github-actions is used"
                    )
                event_name, base, head = args.event_name, args.base, args.head
            return _run_changed(event_name, base, head)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        return _print_subprocess_failure(error)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

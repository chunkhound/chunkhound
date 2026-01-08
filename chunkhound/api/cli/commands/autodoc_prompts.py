from __future__ import annotations

import sys

__all__ = ["is_interactive", "prompt_yes_no", "prompt_text", "prompt_choice"]


def is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _read_input(prompt: str) -> str | None:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def prompt_yes_no(question: str, *, default: bool = False) -> bool:
    if not is_interactive():
        return False

    answers = {
        "y": True,
        "yes": True,
        "n": False,
        "no": False,
    }
    prompt = " [Y/n]: " if default else " [y/N]: "
    while True:
        raw = _read_input(question + prompt)
        if raw is None:
            return False
        answer = raw.strip().lower()
        if not answer:
            return default
        resolved = answers.get(answer)
        if resolved is not None:
            return resolved


def prompt_text(question: str, *, default: str | None = None) -> str | None:
    if not is_interactive():
        return default

    suffix = f" (default: {default})" if default else ""
    raw = _read_input(f"{question}{suffix}: ")
    if raw is None:
        return default
    answer = raw.strip()
    return answer if answer else default


def prompt_choice(
    question: str,
    *,
    choices: tuple[str, ...],
    default: str,
) -> str:
    if not is_interactive():
        return default

    choices_str = "/".join(choices)
    while True:
        answer = prompt_text(
            f"{question} ({choices_str})",
            default=default,
        )
        resolved = (answer or "").strip().lower()
        if resolved in choices:
            return resolved

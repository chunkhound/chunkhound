from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import TypeVar

from chunkhound.utils.text import slugify_kebab

T = TypeVar("T")


def _strip_markdown_for_search(markdown: str) -> str:
    text = re.sub(r"```.*?```", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_description(markdown: str, limit: int = 180) -> str:
    paragraph_lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph_lines:
                break
            continue
        if stripped.startswith("#"):
            continue
        if stripped.lower() in {"overview", "**overview**"}:
            continue
        if _is_list_item(stripped):
            paragraph_lines.append(_strip_list_marker(stripped))
            break
        paragraph_lines.append(stripped)
    paragraph = " ".join(paragraph_lines).strip()
    if not paragraph:
        return "Key workflows and responsibilities summarized for this topic."
    paragraph = re.sub(r"\s+", " ", paragraph)
    if len(paragraph) <= limit:
        return paragraph
    return paragraph[: limit - 3].rstrip() + "..."


def _strip_metadata_block(text: str) -> tuple[str | None, str]:
    stripped = text.lstrip()
    if not stripped.startswith("<!--"):
        return None, text
    start = text.find("<!--")
    end = text.find("-->", start + 4)
    if end == -1:
        return None, text
    metadata = text[start + 4 : end].strip()
    remainder = text[end + 3 :]
    return metadata, remainder.lstrip("\n")


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped
    return None


def _heading_text(heading: str) -> str:
    return heading.lstrip("# ").strip()


def _strip_first_heading(text: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("# "):
            return "\n".join(lines[idx + 1 :]).lstrip()
    return text


def _ensure_overview_heading(text: str) -> str:
    lines = text.splitlines()
    first_content_idx: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            return text
        if stripped.startswith("**Overview**"):
            remainder = stripped[len("**Overview**") :].strip()
            remainder = remainder.lstrip("-").lstrip(":").strip()
            new_lines = lines[:idx] + ["## Overview"]
            if remainder:
                new_lines.append(remainder)
            new_lines.extend(lines[idx + 1 :])
            return "\n".join(new_lines).lstrip()
        lowered = stripped.lower()
        if lowered.startswith("overview"):
            remainder = stripped[len("overview") :].strip()
            remainder = remainder.lstrip("-").lstrip(":").strip()
            new_lines = lines[:idx] + ["## Overview"]
            if remainder:
                new_lines.append(remainder)
            new_lines.extend(lines[idx + 1 :])
            return "\n".join(new_lines).lstrip()
        first_content_idx = idx
        break
    if first_content_idx is None:
        return text
    new_lines = (
        lines[:first_content_idx] + ["## Overview", ""] + lines[first_content_idx:]
    )
    return "\n".join(new_lines).lstrip()


def _is_list_item(line: str) -> bool:
    if line.startswith(("-", "*", "+")):
        return True
    return bool(re.match(r"\d+\.\s+", line))


def _strip_list_marker(line: str) -> str:
    if line.startswith(("-", "*", "+")):
        return line[1:].strip()
    return re.sub(r"^\d+\.\s+", "", line).strip()


def _remove_duplicate_title_line(text: str, title: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    removed = False
    for line in lines:
        stripped = line.strip()
        if not stripped and not cleaned:
            continue
        if not removed and stripped in {
            f"**{title}**",
            f"**{title.rstrip()}**",
        }:
            removed = True
            continue
        cleaned.append(line)
    return "\n".join(cleaned).lstrip()


def _scope_from_heading(heading: str) -> str:
    cleaned = _heading_text(heading)
    if " for " in cleaned:
        return cleaned.split(" for ", 1)[1].strip()
    return cleaned


def _default_site_title(scope_label: str) -> str:
    if scope_label and scope_label != "/":
        return f"AutoDoc - {scope_label}"
    return "AutoDoc Documentation"


def _slugify_title(title: str, order: int) -> str:
    slug = slugify_kebab(title, ascii_only=True)
    return f"{order:02d}-{slug}"


def _escape_yaml(value: str) -> str:
    escaped = json.dumps(value, ensure_ascii=True)
    return escaped[1:-1]


def _chunked(items: Iterable[T], size: int) -> list[list[T]]:
    if size <= 0:
        return [list(items)]
    batch: list[T] = []
    batches: list[list[T]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches

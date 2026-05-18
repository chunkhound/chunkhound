import re
from pathlib import Path

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.core.types.common import FileId, FilePath, LineNumber

_HUNK_HEADER = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@(.*)')


def parse_diff_to_chunks(raw_diff: str) -> list[Chunk]:
    chunks: list[Chunk] = []

    current_file: str | None = None
    hunk_start: int = 0
    hunk_end: int = 0
    hunk_lines: list[str] = []
    symbol: str = ""
    in_hunk: bool = False

    def flush_hunk() -> None:
        nonlocal in_hunk
        if not in_hunk or current_file is None:
            in_hunk = False
            return
        try:
            chunk = Chunk(
                symbol=symbol,
                start_line=LineNumber(max(1, hunk_start)),
                end_line=LineNumber(max(hunk_start, hunk_end)),
                code="".join(hunk_lines),
                chunk_type=ChunkType.BLOCK,
                file_id=FileId(0),
                language=Language.GIT_DIFF,
                file_path=FilePath(current_file),
            )
            chunks.append(chunk)
        except Exception:
            pass
        in_hunk = False

    for line in raw_diff.splitlines(keepends=True):
        stripped = line.rstrip('\n').rstrip('\r')

        if stripped.startswith('diff --git '):
            flush_hunk()
            current_file = None
            hunk_lines = []
            continue

        if stripped.startswith('+++ '):
            raw_path = stripped[4:]
            if raw_path == '/dev/null':
                current_file = None
            elif raw_path.startswith('b/'):
                current_file = raw_path[2:]
            else:
                current_file = raw_path
            continue

        m = _HUNK_HEADER.match(stripped)
        if m:
            flush_hunk()
            new_start = int(m.group(1))
            count_str = m.group(2)
            new_count = int(count_str) if count_str is not None else 1
            context_text = m.group(3).strip()

            if new_count == 0:
                in_hunk = False
                hunk_lines = []
                continue

            hunk_start = new_start
            hunk_end = new_start
            if context_text:
                symbol = context_text
            elif current_file is not None:
                symbol = f"{Path(current_file).name}:{new_start}"
            else:
                symbol = f":{new_start}"
            hunk_lines = [line]
            in_hunk = True
            continue

        if in_hunk:
            hunk_lines.append(line)
            # track end line: count added lines (lines starting with '+' but not '+++')
            if stripped.startswith('+') and not stripped.startswith('+++'):
                hunk_end += 1

    flush_hunk()
    return chunks

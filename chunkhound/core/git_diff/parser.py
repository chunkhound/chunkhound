import re
from pathlib import Path

from loguru import logger

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import (
    ChunkType,
    FileId,
    FilePath,
    Language,
    LineNumber,
)

_HUNK_HEADER = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@(.*)')


def parse_diff_to_chunks(raw_diff: str, max_chunk_chars: int = 10_000) -> list[Chunk]:
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
            code = "".join(hunk_lines)
            if len(code) <= max_chunk_chars:
                chunks.append(Chunk(
                    symbol=symbol,
                    start_line=LineNumber(max(1, hunk_start)),
                    end_line=LineNumber(max(hunk_start, hunk_end)),
                    code=code,
                    chunk_type=ChunkType.BLOCK,
                    file_id=FileId(0),
                    language=Language.GIT_DIFF,
                    file_path=FilePath(current_file),
                ))
            else:
                # Split large hunks at line boundaries to avoid token limit errors.
                # JSON/HTML diffs can be nearly 1:1 chars-to-tokens, so 10k chars is
                # the safe upper bound for a 16384-token embedding model.
                #
                # Track the cumulative new-file line offset so each fragment gets
                # the real start_line it represents, not a fabricated ordinal.
                parts: list[str] = []
                part_line_starts: list[int] = []
                running_line: int = 0      # new-file lines consumed so far
                part_line_offset: int = 0  # new-file offset at start of current part
                current_part: list[str] = []
                current_len = 0
                for ln in hunk_lines:
                    # '+' (addition) and ' ' (context) lines advance the new-file
                    # line counter; '-' (deletion) and '@@' headers do not.
                    advances = bool(ln) and ln[0] in ("+", " ")
                    if len(ln) > max_chunk_chars:
                        # Single line exceeds limit (e.g. minified SVG/JS): flush
                        # current part, then split the line at char boundaries.
                        if current_part:
                            part_line_starts.append(part_line_offset)
                            parts.append("".join(current_part))
                            part_line_offset = running_line
                            current_part = []
                            current_len = 0
                        for i in range(0, len(ln), max_chunk_chars):
                            part_line_starts.append(running_line)
                            parts.append(ln[i : i + max_chunk_chars])
                        if advances:
                            running_line += 1
                        continue
                    if current_len + len(ln) > max_chunk_chars and current_part:
                        part_line_starts.append(part_line_offset)
                        parts.append("".join(current_part))
                        part_line_offset = running_line
                        current_part = []
                        current_len = 0
                    current_part.append(ln)
                    current_len += len(ln)
                    if advances:
                        running_line += 1
                if current_part:
                    part_line_starts.append(part_line_offset)
                    parts.append("".join(current_part))
                # end offset for part i = start of part i+1 minus 1; last part ends
                # at running_line (total new-file lines consumed in this hunk).
                part_line_end_offsets = part_line_starts[1:] + [running_line]
                for n, (part, line_start, line_end_off) in enumerate(
                    zip(parts, part_line_starts, part_line_end_offsets), 1
                ):
                    part_symbol = f"{symbol} (part {n})" if len(parts) > 1 else symbol
                    part_start = max(1, hunk_start + line_start)
                    raw_end = hunk_start + line_end_off - 1
                    part_end = min(hunk_end, max(part_start, raw_end))
                    chunks.append(Chunk(
                        symbol=part_symbol,
                        start_line=LineNumber(part_start),
                        end_line=LineNumber(part_end),
                        code=part,
                        chunk_type=ChunkType.BLOCK,
                        file_id=FileId(0),
                        language=Language.GIT_DIFF,
                        file_path=FilePath(current_file),
                    ))
        except Exception as exc:
            logger.warning(
                "parse_diff_to_chunks: skipping malformed hunk in {!r}: {}",
                current_file,
                exc,
            )
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
            hunk_end = new_start + new_count - 1
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

    flush_hunk()
    return chunks

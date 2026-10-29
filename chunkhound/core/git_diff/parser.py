import re
from collections.abc import AsyncIterator
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

_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@(.*)")
_MAX_METADATA_LINE_CHARS = 1024 * 1024


class StreamingDiffParser:
    """Incrementally turn unified-diff text into bounded searchable chunks."""

    def __init__(self, max_chunk_chars: int = 10_000) -> None:
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")
        self._max_chunk_chars = max_chunk_chars
        self._line_buffer = ""
        self._streaming_line_advances: bool | None = None

        self._current_file: str | None = None
        self._hunk_start = 0
        self._hunk_end = 0
        self._symbol = ""
        self._in_hunk = False
        self._running_line = 0

        self._draft_text = ""
        self._draft_start = 0
        self._pending: tuple[str, int] | None = None
        self._draft_count = 0
        self._part_number = 0

    def feed(self, text: str) -> list[Chunk]:
        """Consume an arbitrary text fragment and return completed chunks."""
        output: list[Chunk] = []
        if not text:
            return output
        self._line_buffer += text

        while self._line_buffer:
            if self._streaming_line_advances is not None:
                newline = self._line_buffer.find("\n")
                if newline < 0:
                    self._append_hunk_text(
                        self._line_buffer, self._running_line, output
                    )
                    self._line_buffer = ""
                    break
                piece = self._line_buffer[: newline + 1]
                self._line_buffer = self._line_buffer[newline + 1 :]
                self._append_hunk_text(piece, self._running_line, output)
                if self._streaming_line_advances:
                    self._running_line += 1
                self._streaming_line_advances = None
                continue

            newline = self._line_buffer.find("\n")
            if newline >= 0:
                line = self._line_buffer[: newline + 1]
                self._line_buffer = self._line_buffer[newline + 1 :]
                self._consume_line(line, output)
                continue

            if self._can_stream_partial_body(self._line_buffer):
                advances = self._line_buffer[0] in ("+", " ")
                self._append_hunk_text(self._line_buffer, self._running_line, output)
                self._line_buffer = ""
                self._streaming_line_advances = advances
                break

            if len(self._line_buffer) > _MAX_METADATA_LINE_CHARS:
                raise ValueError(
                    "git diff metadata line exceeds the 1 MiB safety limit"
                )
            break

        return output

    def finish(self) -> list[Chunk]:
        """Flush an unterminated final line and the active hunk."""
        output: list[Chunk] = []
        if self._streaming_line_advances is not None:
            if self._line_buffer:
                self._append_hunk_text(self._line_buffer, self._running_line, output)
            if self._streaming_line_advances:
                self._running_line += 1
            self._streaming_line_advances = None
            self._line_buffer = ""
        elif self._line_buffer:
            self._consume_line(self._line_buffer, output)
            self._line_buffer = ""
        self._flush_hunk(output)
        return output

    def _can_stream_partial_body(self, line: str) -> bool:
        if not self._in_hunk or not line:
            return False
        # Inside a hunk, "+++ ..." is a valid added source line. A real file
        # header follows a new "diff --git" boundary, which flushes the hunk.
        metadata_prefixes = ("diff --git ", "@@")
        if any(prefix.startswith(line) for prefix in metadata_prefixes):
            return False
        if any(line.startswith(prefix) for prefix in metadata_prefixes):
            return False
        return True

    def _consume_line(self, line: str, output: list[Chunk]) -> None:
        stripped = line.rstrip("\n").rstrip("\r")

        if stripped.startswith("diff --git "):
            self._flush_hunk(output)
            self._current_file = None
            return

        if not self._in_hunk and stripped.startswith("+++ "):
            raw_path = stripped[4:]
            if raw_path == "/dev/null":
                self._current_file = None
            elif raw_path.startswith("b/"):
                self._current_file = raw_path[2:]
            else:
                self._current_file = raw_path
            return

        match = _HUNK_HEADER.match(stripped)
        if match:
            self._flush_hunk(output)
            new_start = int(match.group(1))
            count = match.group(2)
            new_count = int(count) if count is not None else 1
            if new_count == 0:
                return

            self._hunk_start = new_start
            self._hunk_end = new_start + new_count - 1
            context = match.group(3).strip()
            if context:
                self._symbol = context
            elif self._current_file is not None:
                self._symbol = f"{Path(self._current_file).name}:{new_start}"
            else:
                self._symbol = f":{new_start}"
            self._in_hunk = True
            self._running_line = 0
            self._append_hunk_text(line, 0, output)
            return

        if self._in_hunk:
            self._append_complete_hunk_line(line, output)

    def _append_complete_hunk_line(self, line: str, output: list[Chunk]) -> None:
        advances = bool(line) and line[0] in ("+", " ")
        if len(line) > self._max_chunk_chars:
            self._append_hunk_text(line, self._running_line, output)
        else:
            would_overflow = (
                self._draft_text
                and len(self._draft_text) + len(line) > self._max_chunk_chars
            )
            draft_has_new_line = self._running_line > self._draft_start
            if would_overflow:
                if draft_has_new_line:
                    self._rotate_draft(output)
                else:
                    # A hunk header is useful context, but never let it push the
                    # first source line beyond the hard embedding-size bound.
                    self._draft_text = ""
                    self._draft_start = self._running_line
            if not self._draft_text:
                self._draft_start = self._running_line
            self._draft_text += line
        if advances:
            self._running_line += 1

    def _append_hunk_text(
        self, text: str, line_offset: int, output: list[Chunk]
    ) -> None:
        while text:
            if not self._draft_text:
                self._draft_start = line_offset
            available = self._max_chunk_chars - len(self._draft_text)
            self._draft_text += text[:available]
            text = text[available:]
            if len(self._draft_text) == self._max_chunk_chars:
                self._rotate_draft(output)

    def _rotate_draft(self, output: list[Chunk]) -> None:
        if not self._draft_text:
            return
        current = (self._draft_text, self._draft_start)
        self._draft_count += 1
        if self._pending is not None:
            self._emit_pending(current[1], output)
        self._pending = current
        self._draft_text = ""

    def _emit_pending(self, next_offset: int, output: list[Chunk]) -> None:
        if self._pending is None or self._current_file is None:
            self._pending = None
            return
        text, line_offset = self._pending
        self._part_number += 1
        part_start = max(1, self._hunk_start + line_offset)
        raw_end = self._hunk_start + next_offset - 1
        part_end = min(self._hunk_end, max(part_start, raw_end))
        symbol = (
            f"{self._symbol} (part {self._part_number})"
            if self._draft_count > 1
            else self._symbol
        )
        output.append(
            Chunk(
                symbol=symbol,
                start_line=LineNumber(part_start),
                end_line=LineNumber(part_end),
                code=text,
                chunk_type=ChunkType.BLOCK,
                file_id=FileId(0),
                language=Language.GIT_DIFF,
                file_path=FilePath(self._current_file),
            )
        )
        self._pending = None

    def _flush_hunk(self, output: list[Chunk]) -> None:
        if not self._in_hunk:
            self._reset_hunk()
            return
        try:
            self._rotate_draft(output)
            self._emit_pending(self._running_line, output)
        except Exception as exc:
            logger.warning(
                "StreamingDiffParser: skipping malformed hunk in {!r}: {}",
                self._current_file,
                exc,
            )
        self._reset_hunk()

    def _reset_hunk(self) -> None:
        self._in_hunk = False
        self._running_line = 0
        self._draft_text = ""
        self._draft_start = 0
        self._pending = None
        self._draft_count = 0
        self._part_number = 0


async def stream_diff_to_chunks(
    fragments: AsyncIterator[str], max_chunk_chars: int = 10_000
) -> AsyncIterator[Chunk]:
    """Parse bounded unified-diff fragments without materializing files."""
    parser = StreamingDiffParser(max_chunk_chars=max_chunk_chars)
    try:
        async for fragment in fragments:
            for chunk in parser.feed(fragment):
                yield chunk
        for chunk in parser.finish():
            yield chunk
    finally:
        close = getattr(fragments, "aclose", None)
        if callable(close):
            await close()


def parse_diff_to_chunks(raw_diff: str, max_chunk_chars: int = 10_000) -> list[Chunk]:
    """Parse an in-memory diff through the same state machine as streaming input."""
    parser = StreamingDiffParser(max_chunk_chars=max_chunk_chars)
    return [*parser.feed(raw_diff), *parser.finish()]

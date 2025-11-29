"""RapidYAML-backed YAML parser."""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from contextlib import contextmanager
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
from bisect import bisect_left
from time import perf_counter

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.interfaces.language_parser import LanguageParser, ParseResult
from chunkhound.parsers.universal_parser import UniversalParser
from chunkhound.parsers.yaml_template_sanitizer import sanitize_helm_templates
from chunkhound.utils.chunk_deduplication import deduplicate_chunks

logger = logging.getLogger(__name__)


def _env_wants_tree_sitter() -> bool:
    """Return True when RapidYAML should be disabled via env."""
    value = os.environ.get("CHUNKHOUND_YAML_ENGINE", "").strip().lower()
    if not value:
        value = os.environ.get("CHUNKHOUND_DISABLE_RAPIDYAML", "").strip().lower()
        if value in {"1", "true", "yes"}:
            return True
        return False
    return value in {"tree", "treesitter", "tree_sitter", "ts"}


class RapidYamlParser(LanguageParser):
    """LanguageParser implementation that prefers RapidYAML, with fallback."""

    _KEY_NODE_TYPES = {"KEYVAL", "KEYMAP", "KEYSEQ"}

    def __init__(self, fallback: UniversalParser) -> None:
        self._fallback = fallback
        self._enabled = not _env_wants_tree_sitter()
        self._ryml = None
        self._tree = None
        # Memoize paths that should not be parsed with RapidYAML again this process
        self._denylist_paths: set[str] = set()
        # Counters for one-line summary logging
        self._count_sanitized = 0
        self._count_pre_skip = 0
        self._count_complex_skip = 0
        self._count_ryml_ok = 0
        self._count_ryml_fail = 0
        self._count_fallback_ts = 0
        # Perf counters
        self._t_sanitize = 0.0
        self._t_parse_in_place = 0.0
        self._t_emit_yaml = 0.0
        self._t_locate = 0.0
        self._emit_calls = 0
        self._locate_calls = 0
        self._rewrite_counts: Counter[str] = Counter()
        self._pre_skip_reasons: Counter[str] = Counter()

        if self._enabled:
            try:
                import ryml  # type: ignore[import-not-found]

                self._ryml = ryml
                self._tree = ryml.Tree()
            except Exception as exc:  # pragma: no cover - import-time guard
                self._enabled = False
                logger.info(
                    "RapidYAML disabled (import failure): %s. Falling back to tree-sitter.",
                    exc,
                )

    # ------------------------------------------------------------------#
    # LanguageParser interface (delegating most behavior to fallback)
    # ------------------------------------------------------------------#
    @property
    def language(self) -> Language:
        return self._fallback.language

    @property
    def supported_extensions(self) -> set[str]:
        return self._fallback.supported_extensions

    @property
    def supported_chunk_types(self) -> set[ChunkType]:
        return self._fallback.supported_chunk_types

    @property
    def is_initialized(self) -> bool:
        return self._fallback.is_initialized

    @property
    def config(self):
        return self._fallback.config

    def parse_file(self, file_path: Path, file_id: FileId) -> list[Chunk]:
        if not self._can_use_rapid():
            return self._fallback.parse_file(file_path, file_id)

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Attempt to decode using fallback parser
            return self._fallback.parse_file(file_path, file_id)

        return self.parse_content(content, file_path, file_id)

    def parse_content(
        self,
        content: str,
        file_path: Path | None = None,
        file_id: FileId | None = None,
    ) -> list[Chunk]:
        # Denylist: skip ryml attempts for known-bad paths (no tree-sitter fallback)
        if file_path is not None and str(file_path) in getattr(self, "_denylist_paths", set()):
            self._count_fallback_ts += 1
            return []

        if not self._can_use_rapid():
            return self._fallback.parse_content(content, file_path, file_id)

        t0 = perf_counter()
        sanitized = sanitize_helm_templates(content)
        self._t_sanitize += perf_counter() - t0
        effective_content = sanitized.text
        fallback_source = effective_content if sanitized.changed else content
        if sanitized.changed:
            self._count_sanitized += 1
            summary = _summarize_rewrites(sanitized.rewrites)
            path_str = str(file_path) if file_path else "<memory>"
            logger.debug(
                "Sanitized templated YAML %s (%s)",
                path_str,
                summary,
            )
            # Aggregate rewrite kinds for summary
            try:
                self._rewrite_counts.update(r.kind for r in sanitized.rewrites)
            except Exception:
                pass

        # If sanitizer advises pre-skip (eg, non-YAML fragments), avoid ryml churn
        if getattr(sanitized, "pre_skip", False):
            # Pre-skip files: attempt one last parse via tree-sitter fallback only
            self._count_pre_skip += 1
            try:
                reason = getattr(sanitized, "pre_skip_reason", None) or "unknown"
                self._pre_skip_reasons.update([str(reason)])
            except Exception:
                pass
            self._count_fallback_ts += 1
            try:
                return self._fallback.parse_content(content, file_path, file_id)
            except Exception:
                # If fallback also fails, treat as empty
                return []

        if _has_complex_keys(effective_content):
            path_str = str(file_path) if file_path else "<memory>"
            logger.debug(
                "RapidYAML skipped %s: detected complex YAML keys. Falling back to tree-sitter.",
                path_str,
            )
            self._count_complex_skip += 1
            self._count_fallback_ts += 1
            return []

        if not effective_content.strip():
            return []

        try:
            perf = _RymlPerf()
            builder = _RapidYamlChunkBuilder(
                self._ryml,
                self._tree,
                effective_content,
                file_id or FileId(0),
                perf=perf,
            )
            chunks = builder.build_chunks()
            # Accumulate perf
            self._t_parse_in_place += perf.parse_in_place
            self._t_emit_yaml += perf.emit_yaml
            self._t_locate += perf.locate
            self._emit_calls += perf.emit_calls
            self._locate_calls += perf.locate_calls
            self._count_ryml_ok += 1
            return chunks
        except Exception as exc:
            logger.debug(
                "RapidYAML parser failed (%s). Falling back to tree-sitter.", exc
            )
            # Add to denylist to avoid repeated attempts
            if file_path is not None:
                self._denylist_paths.add(str(file_path))
            self._count_ryml_fail += 1
            self._count_fallback_ts += 1
            return []

    def parse_with_result(self, file_path: Path, file_id: FileId) -> ParseResult:
        if not self._can_use_rapid():
            return self._fallback.parse_with_result(file_path, file_id)
        # Reuse fallback's ParseResult structure but supply rapid chunks.
        chunks = self.parse_file(file_path, file_id)
        return ParseResult(
            chunks=[chunk.to_dict() for chunk in chunks],
            language=Language.YAML,
            total_chunks=len(chunks),
            parse_time=0.0,
            errors=[],
            warnings=[],
            metadata={"parser_type": "rapid_yaml"},
        )

    def supports_incremental_parsing(self) -> bool:
        return False

    def parse_incremental(
        self, file_path: Path, previous_chunks: list[dict[str, object]] | None = None
    ) -> list[Chunk]:
        return self.parse_file(file_path, FileId(0))

    def get_parse_tree(self, content: str):
        return self._fallback.get_parse_tree(content)

    def setup(self) -> None:
        self._fallback.setup()

    def cleanup(self) -> None:
        # Emit one-line summary for this parser instance
        top_rewrites = ", ".join(
            f"{k}={v}" for k, v in self._rewrite_counts.most_common(6)
        ) or "-"
        logger.info(
            (
                "RapidYAML summary: sanitized=%d pre_skip=%d complex_skip=%d "
                "ryml_ok=%d ryml_fail=%d fallback_ts=%d | "
                "t_sanitize=%.2fs t_parse=%.2fs t_emit=%.2fs t_locate=%.2fs | "
                "emits=%d locates=%d | top_rewrites=[%s] | pre_skip_reasons=%s"
            ),
            self._count_sanitized,
            self._count_pre_skip,
            self._count_complex_skip,
            self._count_ryml_ok,
            self._count_ryml_fail,
            self._count_fallback_ts,
            self._t_sanitize,
            self._t_parse_in_place,
            self._t_emit_yaml,
            self._t_locate,
            self._emit_calls,
            self._locate_calls,
            top_rewrites,
            dict(self._pre_skip_reasons),
        )
        # Delegate cleanup if supported
        if hasattr(self._fallback, "cleanup"):
            try:
                self._fallback.cleanup()  # type: ignore[call-arg]
            except Exception:
                pass

    def reset(self) -> None:
        self._fallback.reset()

    def can_parse_file(self, file_path: Path) -> bool:
        return self._fallback.can_parse_file(file_path)

    def detect_language(self, file_path: Path) -> Language | None:
        return self._fallback.detect_language(file_path)

    def validate_syntax(self, content: str) -> list[str]:
        return self._fallback.validate_syntax(content)

    # ------------------------------------------------------------------#
    def _can_use_rapid(self) -> bool:
        return self._enabled and self._ryml is not None and self._tree is not None


@dataclass
class _RymlPerf:
    parse_in_place: float = 0.0
    emit_yaml: float = 0.0
    locate: float = 0.0
    emit_calls: int = 0
    locate_calls: int = 0


@dataclass
class _LineLocator:
    """Utility to approximate line ranges for emitted YAML blocks."""

    lines: Sequence[str]
    depth_positions: List[int]
    fallback_line: int = 0

    perf: _RymlPerf | None = None

    def __init__(self, content: str, perf: _RymlPerf | None = None) -> None:
        self.lines = content.splitlines()
        self.depth_positions = [0]
        self.fallback_line = 0
        self.perf = perf
        # Precompute stripped lines to avoid repeated .strip()
        self._stripped_lines: List[str] = [ln.strip() for ln in self.lines]
        # Build an index map for exact-match lookups: stripped_line -> sorted list of indices
        self._index_map: dict[str, List[int]] = {}
        for idx, s in enumerate(self._stripped_lines):
            if not s:
                continue
            bucket = self._index_map.get(s)
            if bucket is None:
                self._index_map[s] = [idx]
            else:
                bucket.append(idx)

        # Multi-line context index for more accurate disambiguation
        # Maps (line1, line2, line3) tuples to list of starting indices
        self._context_index: dict[tuple[str, ...], List[int]] = {}
        for idx in range(len(self._stripped_lines)):
            # Build context from up to 3 lines
            context = tuple(self._stripped_lines[idx : idx + 3])
            if context not in self._context_index:
                self._context_index[context] = [idx]
            else:
                self._context_index[context].append(idx)

    def truncate(self, depth: int) -> None:
        if depth + 1 < len(self.depth_positions):
            self.depth_positions = self.depth_positions[: depth + 1]

    def locate(
        self,
        first_line: str,
        depth: int,
        node_type: str = "KEYVAL",
        parent_key: str | None = None,
        emitted_yaml: str | None = None,
    ) -> tuple[int, int]:
        """Locate source lines for a YAML node using progressive fallback strategy.

        Args:
            first_line: First line of emitted YAML
            depth: Nesting depth of node
            node_type: Type of node (KEYVAL, KEYSEQ, KEYMAP)
            parent_key: Optional parent key name for scoped search
            emitted_yaml: Full emitted YAML for multi-line context
        """
        _t0 = perf_counter()
        if not self.lines:
            self.fallback_line += 1
            start, end = self.fallback_line, self.fallback_line
            if self.perf is not None:
                self.perf.locate += perf_counter() - _t0
                self.perf.locate_calls += 1
            return start, end

        # Ensure depth_positions is large enough
        while len(self.depth_positions) <= depth:
            self.depth_positions.append(self.depth_positions[-1])

        start_search = self.depth_positions[depth]
        idx = None

        # Level 1: Multi-line context match (most accurate)
        if emitted_yaml:
            context_lines = self._extract_context_signature(emitted_yaml, max_lines=3)
            if len(context_lines) >= 2:  # Need at least 2 lines for good context
                idx = self._find_by_context(context_lines, start_search)
                if idx is None and start_search > 0:
                    # Retry from beginning if not found
                    idx = self._find_by_context(context_lines, 0)

        # Level 2: Single-line with parent scope awareness
        if idx is None and parent_key:
            idx = self._find_with_parent_scope(first_line, start_search, parent_key)

        # Level 3: Node-type specific search (KEYSEQ special handling)
        if idx is None:
            idx = self._find_from(first_line, start_search, node_type)
            if idx is None and start_search > 0:
                idx = self._find_from(first_line, 0, node_type)

        # Level 4: Fallback counter
        if idx is None:
            self.fallback_line += 1
            return self.fallback_line, self.fallback_line

        # Update position tracking
        self.depth_positions[depth] = idx + 1
        for i in range(depth + 1, len(self.depth_positions)):
            self.depth_positions[i] = max(self.depth_positions[i], idx + 1)

        # Compute block end
        end_idx = self._compute_block_end(idx, node_type)

        start, end = idx + 1, end_idx + 1
        if self.perf is not None:
            self.perf.locate += perf_counter() - _t0
            self.perf.locate_calls += 1
        return start, end

    def _find_from(self, target_line: str, start: int, node_type: str = "KEYVAL") -> int | None:
        """Find target line with node-type specific logic."""
        stripped_target = target_line.strip()
        if not stripped_target:
            return None

        # KEYSEQ nodes need special handling
        if node_type == "KEYSEQ":
            # Block-style arrays: "- item"
            if stripped_target.startswith("- "):
                return self._find_keyseq_item(stripped_target, start)
            # Inline arrays: "key: [...]" - use fuzzy matching for JSON normalization
            # RapidYAML may normalize spaces in JSON arrays
            if "[" in stripped_target:
                # Extract key part before the array
                key_part = stripped_target.split("[")[0].strip()
                for idx in range(max(0, start), len(self._stripped_lines)):
                    cand = self._stripped_lines[idx]
                    if cand.startswith(key_part) and "[" in cand:
                        return idx
                # Retry from beginning if not found
                if start > 0:
                    for idx in range(0, len(self._stripped_lines)):
                        cand = self._stripped_lines[idx]
                        if cand.startswith(key_part) and "[" in cand:
                            return idx
                return None

        # KEYVAL and KEYMAP use standard index-based search
        positions = self._index_map.get(stripped_target)
        if positions:
            pos = bisect_left(positions, max(0, start))
            if pos < len(positions):
                return positions[pos]

        # Fallback: prefix match scan from start
        for idx in range(max(0, start), len(self._stripped_lines)):
            cand = self._stripped_lines[idx]
            if not cand:
                continue
            if cand.startswith(stripped_target):
                return idx
        return None

    def _compute_block_end(self, start_idx: int, node_type: str = "KEYVAL") -> int:
        """Find end of block with node-type specific logic.

        Args:
            start_idx: Starting line index (0-based)
            node_type: Type of node to help with boundary detection
        """
        if start_idx >= len(self.lines):
            return start_idx

        base_indent = self._indent_level(self.lines[start_idx])
        end_idx = start_idx

        for idx in range(start_idx + 1, len(self.lines)):
            line = self.lines[idx]
            stripped = line.strip()

            if not stripped:  # Blank lines belong to block
                end_idx = idx
                continue

            indent = self._indent_level(line)

            # For KEYSEQ, be inclusive of all list items at same/higher indent
            if node_type == "KEYSEQ" and stripped.startswith("- "):
                if indent >= base_indent:
                    end_idx = idx
                    continue

            # Stop at same or less indent (unless continuing array)
            if indent <= base_indent and not stripped.startswith("- "):
                break

            end_idx = idx

        return end_idx

    @staticmethod
    def _indent_level(line: str) -> int:
        idx = 0
        for ch in line:
            if ch in (" ", "\t"):
                idx += 1
            else:
                break
        return idx

    def _extract_context_signature(self, text: str, max_lines: int = 3) -> list[str]:
        """Extract up to max_lines of non-blank text as context signature.

        Multi-line context is more unique than single lines and helps
        disambiguate identical first lines in different locations.
        Skips leading blank lines to maximize context quality.
        """
        lines = text.splitlines()
        # Filter blank lines first, then take up to max_lines
        non_blank = [line.strip() for line in lines if line.strip()]
        return non_blank[:max_lines]

    def _find_by_context(self, context_lines: list[str], start: int) -> int | None:
        """Find location using multi-line context signature.

        More accurate than single-line matching for disambiguating
        duplicate content in different parts of file.
        """
        if not context_lines:
            return None

        # Convert to tuple for indexing
        context_tuple = tuple(context_lines)

        # Fast path: exact context match via index
        positions = self._context_index.get(context_tuple)
        if positions:
            pos = bisect_left(positions, max(0, start))
            if pos < len(positions):
                return positions[pos]

        # Fallback: sequential search for context match
        for idx in range(max(0, start), len(self._stripped_lines)):
            # Check if context starting at idx matches
            window = tuple(self._stripped_lines[idx : idx + len(context_lines)])
            if window == context_tuple:
                return idx

        return None

    def _find_with_parent_scope(self, target_line: str, start: int, parent_key: str) -> int | None:
        """Find target_line within parent key's scope.

        Strategy: Search for parent_key, then look for target_line after it
        within the parent's indentation block.
        """
        stripped_target = target_line.strip()
        parent_stripped = f"{parent_key}:"

        # Find parent key starting from start position
        parent_idx = None
        for idx in range(max(0, start), len(self._stripped_lines)):
            if self._stripped_lines[idx] == parent_stripped:
                parent_idx = idx
                break

        if parent_idx is None:
            return None

        # Get parent indentation
        parent_indent = self._indent_level(self.lines[parent_idx])

        # Search for target_line within parent's block (higher indent)
        for idx in range(parent_idx + 1, len(self.lines)):
            line = self.lines[idx]
            stripped = line.strip()

            if not stripped:  # Skip blank lines
                continue

            indent = self._indent_level(line)
            if indent <= parent_indent:  # Exited parent block
                break

            if stripped == stripped_target or stripped.startswith(stripped_target):
                return idx

        return None

    def _find_keyseq_item(self, target_line: str, start: int) -> int | None:
        """Find KEYSEQ list item starting from position.

        KEYSEQ nodes only return first array element from emit_yaml(),
        so we search from start position (after parent key) for the
        first occurrence to avoid matching unrelated earlier arrays.
        """
        stripped_target = target_line.strip()

        # Search from start for list item
        for idx in range(max(0, start), len(self._stripped_lines)):
            cand = self._stripped_lines[idx]
            if not cand:
                continue
            # Exact match preferred for list markers
            if cand == stripped_target:
                return idx
            # Fallback to prefix match
            if cand.startswith(stripped_target):
                return idx

        return None


class _RapidYamlChunkBuilder:
    """Walks a RapidYAML tree and produces Chunk objects."""

    def __init__(self, ryml_module, tree, content: str, file_id: FileId, perf: _RymlPerf | None = None) -> None:
        self.ryml = ryml_module
        self.tree = tree
        self.file_id = file_id
        self.content = content
        self._buffer = bytearray(content.encode("utf-8"))
        self.lines = content.splitlines()
        self.perf = perf
        self.locator = _LineLocator(content, perf=self.perf)
        self._decoder = ryml_module.u

    def build_chunks(self) -> list[Chunk]:
        self.tree.clear()
        self.tree.clear_arena()
        with _suppress_c_output():
            _t0 = perf_counter()
            self.ryml.parse_in_place(self._buffer, tree=self.tree)
            if self.perf is not None:
                self.perf.parse_in_place += perf_counter() - _t0

        root = self.tree.root_id()
        chunks: list[Chunk] = []
        path: list[str] = []

        for node, depth in self.ryml.walk(self.tree, root):
            ancestor_len = max(depth - 1, 0)
            if len(path) != ancestor_len:
                path = path[:ancestor_len]
            self.locator.truncate(depth)

            key_text = self._key_text(node)
            if key_text:
                if len(path) == ancestor_len:
                    path.append(key_text)
                else:
                    path = path[:ancestor_len] + [key_text]

            node_type = self.tree.type_str(node)
            if node_type not in RapidYamlParser._KEY_NODE_TYPES:
                continue

            if path:
                symbol = ".".join(path)
            elif key_text:
                symbol = key_text
            else:
                symbol = f"yaml_node_{node}"

            # Get parent key from path for context-aware line location
            # path[-1] is the current node, path[-2] is the actual parent
            parent_key = path[-2] if len(path) >= 2 else None

            chunk = self._create_chunk(node, node_type, symbol, depth, parent_key)
            if chunk:
                chunks.append(chunk)

        # Deduplicate chunks to prevent duplicate chunk IDs
        # (e.g., YAML files with repeated config values like "name: example-config")
        chunks = deduplicate_chunks(chunks, Language.YAML)

        return chunks

    def _create_chunk(
        self, node: int, node_type: str, symbol: str, depth: int, parent_key: str | None = None
    ) -> Chunk | None:
        with _suppress_c_output():
            _t0 = perf_counter()
            emitted = self.ryml.emit_yaml(self.tree, node)
            if self.perf is not None:
                self.perf.emit_yaml += perf_counter() - _t0
                self.perf.emit_calls += 1
        normalized = emitted.lstrip("\n")
        first_line = normalized.splitlines()[0] if normalized else symbol
        start_line, end_line = self.locator.locate(
            first_line=first_line,
            depth=depth,
            node_type=node_type,
            parent_key=parent_key,
            emitted_yaml=normalized,
        )

        file_snippet = self._slice_content(start_line, end_line)
        if not file_snippet.strip():
            file_snippet = normalized or first_line

        chunk_type = self._chunk_type_for(node_type)
        metadata = {
            "parser": "rapid_yaml",
            "node_type": node_type.lower(),
            "key_path": symbol,
            "value_kind": self._value_kind(node_type),
        }

        value = self.tree.val(node)
        if value is not None:
            metadata["scalar_value"] = self._decoder(value)

        return Chunk(
            symbol=symbol,
            start_line=LineNumber(start_line),
            end_line=LineNumber(max(start_line, end_line)),
            code=file_snippet,
            chunk_type=chunk_type,
            file_id=self.file_id,
            language=Language.YAML,
            metadata=metadata,
        )

    def _slice_content(self, start_line: int, end_line: int) -> str:
        if not self.content:
            return ""
        start_idx = max(0, start_line - 1)
        end_idx = min(len(self.lines), end_line)
        if start_idx >= len(self.lines):
            return ""
        return "\n".join(self.lines[start_idx:end_idx])

    def _key_text(self, node: int) -> str | None:
        key = self.tree.key(node)
        if key is None:
            return None
        text = self._decoder(key).strip()
        return text or None

    @staticmethod
    def _chunk_type_for(node_type: str) -> ChunkType:
        if node_type == "KEYVAL":
            return ChunkType.KEY_VALUE
        if node_type == "KEYSEQ":
            return ChunkType.ARRAY
        return ChunkType.BLOCK

    @staticmethod
    def _value_kind(node_type: str) -> str:
        if node_type == "KEYVAL":
            return "scalar"
        if node_type == "KEYSEQ":
            return "sequence"
        return "mapping"


_COMPLEX_KEY_RE = re.compile(r"^\s*\?\s*(?:\{|\[|$)")


def _has_complex_keys(content: str) -> bool:
    for line in content.splitlines():
        if _COMPLEX_KEY_RE.match(line):
            return True
    return False


def _summarize_rewrites(rewrites) -> str:
    if not rewrites:
        return "no rewrites"
    counts = Counter(rewrite.kind for rewrite in rewrites)
    parts = [f"{kind}={counts[kind]}" for kind in sorted(counts)]
    return ", ".join(parts)


@contextmanager
def _suppress_c_output():
    """Temporarily redirect C-level stdout/stderr to os.devnull during ryml calls."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            os.close(devnull)
    except Exception:
        # Fail open: if redirection fails, proceed without silencing
        yield

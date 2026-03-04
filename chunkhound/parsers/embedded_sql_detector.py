"""Embedded SQL detection for string literals across all languages.

This module provides functionality to detect and parse SQL code embedded
in string literals within source code files of any programming language.

Common patterns:
- Python: cursor.execute("SELECT * FROM users WHERE id = %s")
- Java: Statement stmt = conn.createStatement("SELECT * FROM...")
- JavaScript: db.query(`SELECT * FROM users`)
- C#: var cmd = new SqlCommand("SELECT * FROM...")
"""

import re
from dataclasses import dataclass
from typing import Any

from tree_sitter import Node as TSNode

from chunkhound.core.types.common import Language
from chunkhound.parsers.universal_engine import UniversalChunk, UniversalConcept


@dataclass(frozen=True)
class EmbeddedSqlMatch:
    """Represents a detected SQL string in source code."""

    sql_content: str  # The extracted SQL code
    start_line: int  # Line where string starts
    end_line: int  # Line where string ends
    host_context: str  # Surrounding code context (e.g., function name)
    confidence: float  # 0-1 score of SQL detection confidence


# Pre-compiled regex patterns for SQL confidence scoring
_PRIMARY_KEYWORD_PATTERNS = {
    keyword: re.compile(r"\b" + keyword + r"\b")
    for keyword in [
        "SELECT", "INSERT", "UPDATE", "DELETE",
        "CREATE", "ALTER", "DROP",
    ]
}

# DDL keywords that indicate a DEFINITION concept
_DDL_KEYWORDS = {"CREATE", "ALTER", "DROP"}

_SECONDARY_KEYWORD_PATTERNS = {
    keyword: re.compile(r"\b" + keyword + r"\b")
    for keyword in [
        "HAVING", "LIMIT", "OFFSET",
    ]
}

_SQL_PATTERNS: list[tuple[re.Pattern, int, bool]] = [
    (re.compile(r"\bFROM\s+\w+"), 25, True),   # weak — matches English prose
    (re.compile(r"\bWHERE\s+\w+"), 20, True),   # weak — matches English prose
    (re.compile(r"\bJOIN\s+\w+"), 20, False),
    (re.compile(r"\bSET\s+\w+\s*="), 20, False),
    (re.compile(r"\bORDER\s+BY\b"), 20, False),
    (re.compile(r"\bGROUP\s+BY\b"), 20, False),
    (re.compile(r"\bINTO\s+\w+"), 20, False),
    (re.compile(r"\bVALUES\s*\("), 10, False),
    (re.compile(r"\bTABLE\s+"), 20, False),
    (re.compile(r"\bPRIMARY\s+KEY\b"), 10, False),
    (re.compile(r"\bFOREIGN\s+KEY\b"), 10, False),
]

_SELECT_PATTERN = re.compile(r"\bSELECT\b")
_FROM_PATTERN = re.compile(r"\bFROM\b")
_DDL_COMPOUND_PATTERN = re.compile(
    r"\b(CREATE|ALTER|DROP)\s+(TABLE|VIEW|INDEX|FUNCTION|TRIGGER|PROCEDURE|SCHEMA|SEQUENCE)\b"
)

# Matches Python string prefixes (r, b, f, u, rb, br, rf, fr, uppercase variants),
# C# verbatim prefix (@), and C# interpolated/verbatim combos ($, @$, $@).
_STRING_PREFIX_RE = re.compile(r"^(?:[brufBRUF]{1,2}|[@$]{1,2})")

_DML_COMPOUND_PATTERNS = [
    re.compile(r"\bDELETE\s+FROM\b"),
    re.compile(r"\bUPDATE\s+\S+\s+SET\b"),
    re.compile(r"\bINSERT\s+INTO\b"),
]

_MIN_CONFIDENCE = 0.6

# Regex for non-alpha, non-space chars (structural SQL evidence in SELECT…FROM gap)
_NON_ALPHA_SPACE_RE = re.compile(r"[^A-Z\s]")


def _select_from_combo_fires(content_upper: str) -> tuple[bool, bool]:
    """Return (fires, is_structural) for the SELECT…FROM combination.

    fires: True if the gap suggests real SQL (not English prose).
    is_structural: True only when the gap contains non-alpha chars (*, ,, digits)
        which is hard structural evidence. False for the short-word-count path,
        which is weak evidence that should not suppress weak-signal halving.

    Requires the gap to contain non-alphabetic chars (e.g. *, ,, digits, .)
    OR be a very short identifier list (<=2 words). An empty gap or pure English
    prose (>2 all-alpha words) returns (False, False).
    """
    m_select = _SELECT_PATTERN.search(content_upper)
    m_from = _FROM_PATTERN.search(content_upper)
    if not (m_select and m_from):
        return False, False
    if m_from.start() <= m_select.end():
        return False, False  # FROM appears before or adjacent to SELECT
    gap = content_upper[m_select.end():m_from.start()].strip()
    if not gap:
        return False, False  # adjacent keywords — English phrase pattern
    if _NON_ALPHA_SPACE_RE.search(gap):
        return True, True  # structural evidence: *, ,, (, digits, etc.
    # Short identifier lists (1-2 words like "id" or "user_name") pass;
    # 3+ all-alpha words are likely English prose, not SQL column names.
    return len(gap.split()) <= 2, False


# Common string node types across languages.
# Includes both parent types (e.g. string, string_literal) and child types
# (e.g. string_content, string_fragment). The `continue` in _visit_node
# skips children of matched nodes to prevent double-processing.
_STRING_NODE_TYPES = {
    "string",
    "string_literal",
    "string_fragment",   # defensive: parent `string`/`template_string` matched first;
    "string_content",    # fires only if a grammar exposes these as top-level nodes
    "template_string",  # JavaScript/TypeScript template strings
    "encapsed_string",  # PHP strings
    "raw_string_literal",
    "interpreted_string_literal",
    "verbatim_string_literal",  # C# verbatim strings (@"...")
    "string_value",
    "quoted_string",
    "multiline_string_literal",  # Kotlin multiline strings
}


class EmbeddedSqlDetector:
    """Detects and extracts SQL code embedded in string literals.

    This detector works across all programming languages by:
    1. Finding string literal nodes in the AST
    2. Applying heuristics to identify SQL content
    3. Extracting and cleaning the SQL code
    4. Returning matches with metadata
    """

    def __init__(self, host_language: Language):
        """Initialize detector for a specific host language.

        Args:
            host_language: The programming language being parsed
        """
        self.host_language = host_language

    def detect_in_tree(self, root_node: TSNode) -> list[EmbeddedSqlMatch]:
        """Detect embedded SQL in a parsed AST tree.

        Args:
            root_node: Root node of the tree-sitter AST

        Returns:
            List of detected SQL matches with metadata
        """
        matches: list[EmbeddedSqlMatch] = []
        self._visit_node(root_node, matches)
        return matches

    def _visit_node(
        self,
        node: TSNode,
        matches: list[EmbeddedSqlMatch]
    ) -> None:
        """Iteratively visit nodes to find string literals (avoids recursion limit)."""
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type in _STRING_NODE_TYPES:
                self._check_string_node(current, matches)
                continue  # skip children of string nodes to avoid duplicates
            stack.extend(reversed(current.children))

    def _check_string_node(
        self,
        node: TSNode,
        matches: list[EmbeddedSqlMatch]
    ) -> None:
        """Check if a string node contains SQL.

        Args:
            node: String literal node
            matches: List to append matches to
        """
        # Extract string content
        string_content = node.text.decode("utf-8", errors="replace") if node.text else ""

        # Remove string delimiters (quotes)
        cleaned_content = self._clean_string_content(string_content)

        # Skip if too short to be meaningful SQL
        if len(cleaned_content.strip()) < 15:
            return

        # Check if content looks like SQL
        confidence = self._calculate_sql_confidence(cleaned_content)

        if confidence < _MIN_CONFIDENCE:
            return

        # Get context (parent function/method name if available)
        context = self._get_context(node)

        # Create match
        match = EmbeddedSqlMatch(
            sql_content=cleaned_content,
            start_line=node.start_point[0] + 1,  # Convert to 1-indexed
            end_line=node.end_point[0] + 1,
            host_context=context,
            confidence=confidence
        )

        matches.append(match)

    def _clean_string_content(self, raw_string: str) -> str:
        """Remove string delimiters and unescape content.

        Args:
            raw_string: Raw string with quotes

        Returns:
            Cleaned string content
        """
        # Strip language-specific prefixes (r"", b"", f"", @"", etc.) once,
        # then try quote delimiters longest-first so '"""' beats '"'.
        stripped = _STRING_PREFIX_RE.sub("", raw_string)
        for quote in ['"""', "'''", '"', "'", "`"]:
            if stripped.startswith(quote) and stripped.endswith(quote):
                content = stripped[len(quote):-len(quote)]
                break
        else:
            content = stripped

        # Handle common escape sequences
        content = content.replace("\\n", "\n")
        content = content.replace("\\t", "\t")
        content = content.replace('\\"', '"')
        content = content.replace("\\'", "'")

        return content

    def _calculate_sql_confidence(self, content: str) -> float:
        """Calculate confidence that content is SQL.

        Scoring model (points, normalised to 0-1 via /100):
          Primary keyword (SELECT/INSERT/UPDATE/DELETE/CREATE/ALTER/DROP):
            +30 (required gate — if absent, returns 0.0 immediately)
          Secondary keywords (HAVING/LIMIT/OFFSET): +10 each
          Clause patterns (FROM/WHERE/JOIN/SET/ORDER BY/GROUP BY): +10-25 each
          DDL compound (CREATE TABLE, ALTER INDEX, ...): +20
          SELECT+FROM combination: +10
          DML compound (DELETE FROM, UPDATE...SET, INSERT INTO): +10

        Threshold: 0.6 (i.e. 60 raw points).

        Args:
            content: String content to analyze

        Returns:
            Confidence score between 0 and 1
        """
        content_upper = content.upper()
        score = 0.0

        # Check for primary SQL keywords (30 points, only count one)
        has_primary = False
        for pattern in _PRIMARY_KEYWORD_PATTERNS.values():
            if pattern.search(content_upper):
                score += 30
                has_primary = True
                break

        if not has_primary:
            return 0.0

        # Check for secondary SQL keywords (10 points each)
        for pattern in _SECONDARY_KEYWORD_PATTERNS.values():
            if pattern.search(content_upper):
                score += 10

        # Check for SQL patterns; track weak vs strong contributions
        weak_score = 0.0
        has_strong_pattern = False
        for pattern, points, is_weak in _SQL_PATTERNS:
            if pattern.search(content_upper):
                score += points
                if is_weak:
                    weak_score += points
                else:
                    has_strong_pattern = True

        # Bonus: DDL verb immediately followed by DDL object type
        if _DDL_COMPOUND_PATTERN.search(content_upper):
            score += 20
            has_strong_pattern = True

        # Bonus: SELECT + FROM combination. Structural gap (*, ,, digits) is strong
        # evidence; a short word-count gap is weak and counted in weak_score.
        combo_fires, combo_structural = _select_from_combo_fires(content_upper)
        if combo_fires:
            score += 10
            if combo_structural:
                has_strong_pattern = True
            else:
                weak_score += 10

        # Bonus: compound DML patterns (DELETE FROM, UPDATE...SET, INSERT INTO)
        for pattern in _DML_COMPOUND_PATTERNS:
            if pattern.search(content_upper):
                score += 10
                has_strong_pattern = True
                break

        # Penalise strings with only weak signals — halve their contribution
        if not has_strong_pattern and weak_score > 0:
            score -= weak_score // 2

        # Convert to 0-1 scale
        normalized = min(score / 100.0, 1.0)

        return normalized

    def _get_context(self, node: TSNode) -> str:
        """Get contextual information about where the string appears.

        Tries to find the containing function/method/class name.

        Args:
            node: String node

        Returns:
            Context description (e.g., "function:get_users")
        """
        # Walk up the tree to find a function/method/class
        current = node.parent

        while current:
            node_type = current.type

            # Check for function-like nodes (exclude call sites)
            if ("function" in node_type or "method" in node_type) and "call" not in node_type:
                # Tree-sitter places identifier/name children before parameter
                # lists and bodies across all supported languages.
                for child in current.children:
                    if "identifier" in child.type or "name" in child.type:
                        name = child.text.decode("utf-8", errors="replace") if child.text else "unknown"
                        return f"function:{name}"

            # Check for class nodes
            if "class" in node_type:
                for child in current.children:
                    if "identifier" in child.type or "name" in child.type:
                        name = child.text.decode("utf-8", errors="replace") if child.text else "unknown"
                        return f"class:{name}"

            current = current.parent

        return "global"

    @staticmethod
    def _classify_sql_concept(sql_content: str) -> UniversalConcept:
        """Classify SQL content into the appropriate UniversalConcept.

        DDL statements (CREATE, ALTER, DROP) are DEFINITION.
        DML statements (SELECT, INSERT, UPDATE, DELETE) are BLOCK.
        """
        content_upper = sql_content.strip().upper()
        for keyword in _DDL_KEYWORDS:
            if _PRIMARY_KEYWORD_PATTERNS[keyword].search(content_upper):
                return UniversalConcept.DEFINITION
        return UniversalConcept.BLOCK

    def create_embedded_sql_chunks(
        self,
        matches: list[EmbeddedSqlMatch],
    ) -> list[UniversalChunk]:
        """Convert detected SQL matches into UniversalChunk objects.

        Args:
            matches: Detected SQL matches

        Returns:
            List of UniversalChunk objects for embedded SQL
        """
        chunks: list[UniversalChunk] = []

        for match in matches:
            concept = self._classify_sql_concept(match.sql_content)
            metadata: dict[str, Any] = {
                "embedded": True,
                "host_language": self.host_language.value,
                "host_context": match.host_context,
                "sql_confidence": match.confidence,
                "detected_language": "sql",
                "chunk_type_hint": "embedded_sql",
            }
            if concept == UniversalConcept.DEFINITION:
                metadata["kind"] = "embedded_sql_ddl"
            else:
                metadata["kind"] = "embedded_sql_dml"
            chunk = UniversalChunk(
                concept=concept,
                name=f"embedded_sql_line_{match.start_line}",
                content=match.sql_content,
                start_line=match.start_line,
                end_line=match.end_line,
                metadata=metadata,
                language_node_type="embedded_sql",
            )
            chunks.append(chunk)

        return chunks

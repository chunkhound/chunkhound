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

from chunkhound.core.types.common import Language
from chunkhound.parsers.universal_engine import UniversalChunk, UniversalConcept

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


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
        "WHERE", "JOIN", "ORDER", "GROUP",
        "HAVING", "LIMIT", "OFFSET", "SET",
    ]
}

_SQL_PATTERNS = [
    (re.compile(r"\bFROM\s+\w+"), 25),
    (re.compile(r"\bWHERE\s+\w+"), 10),
    (re.compile(r"\bJOIN\s+\w+"), 10),
    (re.compile(r"\bSET\s+\w+\s*="), 10),
    (re.compile(r"\bORDER\s+BY\b"), 10),
    (re.compile(r"\bGROUP\s+BY\b"), 10),
    (re.compile(r"\bINTO\s+\w+"), 20),
    (re.compile(r"\bVALUES\s*\("), 10),
    (re.compile(r"\bTABLE\s+"), 20),
    (re.compile(r"\bPRIMARY\s+KEY\b"), 10),
    (re.compile(r"\bFOREIGN\s+KEY\b"), 10),
]

_SELECT_PATTERN = re.compile(r"\bSELECT\b")
_FROM_PATTERN = re.compile(r"\bFROM\b")


class EmbeddedSqlDetector:
    """Detects and extracts SQL code embedded in string literals.

    This detector works across all programming languages by:
    1. Finding string literal nodes in the AST
    2. Applying heuristics to identify SQL content
    3. Extracting and cleaning the SQL code
    4. Returning matches with metadata
    """

    # Common string node types across languages
    STRING_NODE_TYPES = {
        "string",
        "string_literal",
        "string_fragment",
        "string_content",  # Python strings, PHP string content
        "template_string",  # JavaScript/TypeScript template strings
        "encapsed_string",  # PHP strings
        "raw_string_literal",
        "interpreted_string_literal",
        "verbatim_string_literal",  # C# verbatim strings (@"...")
        "string_value",
        "quoted_string",
    }

    def __init__(self, host_language: Language):
        """Initialize detector for a specific host language.

        Args:
            host_language: The programming language being parsed
        """
        self.host_language = host_language

    def detect_in_tree(
        self, root_node: TSNode, source_bytes: bytes
    ) -> list[EmbeddedSqlMatch]:
        """Detect embedded SQL in a parsed AST tree.

        Args:
            root_node: Root node of the tree-sitter AST
            source_bytes: Original source code as bytes

        Returns:
            List of detected SQL matches with metadata
        """
        matches: list[EmbeddedSqlMatch] = []

        # Recursively visit all nodes
        self._visit_node(root_node, source_bytes, matches)

        return matches

    def _visit_node(
        self,
        node: TSNode,
        source_bytes: bytes,
        matches: list[EmbeddedSqlMatch]
    ) -> None:
        """Recursively visit nodes to find string literals.

        Args:
            node: Current AST node
            source_bytes: Source code bytes
            matches: List to append matches to
        """
        # Check if this node is a string literal
        if node.type in self.STRING_NODE_TYPES:
            self._check_string_node(node, source_bytes, matches)
            # Don't visit children of string nodes to avoid duplicates
            # (e.g., template_string contains string_fragment children)
            return

        # Recursively visit children
        for child in node.children:
            self._visit_node(child, source_bytes, matches)

    def _check_string_node(
        self,
        node: TSNode,
        source_bytes: bytes,
        matches: list[EmbeddedSqlMatch]
    ) -> None:
        """Check if a string node contains SQL.

        Args:
            node: String literal node
            source_bytes: Source code bytes
            matches: List to append matches to
        """
        # Extract string content
        string_content = node.text.decode("utf-8") if node.text else ""

        # Remove string delimiters (quotes)
        cleaned_content = self._clean_string_content(string_content)

        # Skip if too short to be meaningful SQL
        if len(cleaned_content.strip()) < 15:
            return

        # Check if content looks like SQL
        confidence = self._calculate_sql_confidence(cleaned_content)

        # Require at least 60% confidence
        if confidence < 0.6:
            return

        # Get context (parent function/method name if available)
        context = self._get_context(node, source_bytes)

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
        # Remove common quote types
        for quote in ['"""', "'''", '"', "'", "`"]:
            if raw_string.startswith(quote) and raw_string.endswith(quote):
                raw_string = raw_string[len(quote):-len(quote)]
                break

        # Handle common escape sequences
        raw_string = raw_string.replace("\\n", "\n")
        raw_string = raw_string.replace("\\t", "\t")
        raw_string = raw_string.replace('\\"', '"')
        raw_string = raw_string.replace("\\'", "'")

        return raw_string

    def _calculate_sql_confidence(self, content: str) -> float:
        """Calculate confidence that content is SQL.

        Uses pre-compiled regex patterns for efficient matching:
        - Presence of SQL keywords (weighted)
        - SQL-like syntax patterns
        - Structure indicators (FROM, WHERE, JOIN)

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

        # Check for SQL patterns (additional points)
        for pattern, points in _SQL_PATTERNS:
            if pattern.search(content_upper):
                score += points

        # Bonus: SELECT + FROM combination is very strong indicator
        if (
            _SELECT_PATTERN.search(content_upper)
            and _FROM_PATTERN.search(content_upper)
        ):
            score += 10

        # Convert to 0-1 scale
        normalized = min(score / 100.0, 1.0)

        return normalized

    def _get_context(self, node: TSNode, source_bytes: bytes) -> str:
        """Get contextual information about where the string appears.

        Tries to find the containing function/method/class name.

        Args:
            node: String node
            source_bytes: Source code bytes

        Returns:
            Context description (e.g., "function:get_users")
        """
        # Walk up the tree to find a function/method/class
        current = node.parent

        while current:
            node_type = current.type

            # Check for function-like nodes
            if "function" in node_type or "method" in node_type:
                # Try to find the function name
                for child in current.children:
                    if "identifier" in child.type or "name" in child.type:
                        name = child.text.decode("utf-8") if child.text else "unknown"
                        return f"function:{name}"

            # Check for class nodes
            if "class" in node_type:
                for child in current.children:
                    if "identifier" in child.type or "name" in child.type:
                        name = child.text.decode("utf-8") if child.text else "unknown"
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
            chunk = UniversalChunk(
                concept=concept,
                name=f"embedded_sql_line_{match.start_line}",
                content=match.sql_content,
                start_line=match.start_line,
                end_line=match.end_line,
                metadata={
                    "embedded": True,
                    "host_language": self.host_language.value,
                    "host_context": match.host_context,
                    "sql_confidence": match.confidence,
                    "detected_language": "sql",
                },
                language_node_type="embedded_sql_string",
            )
            chunks.append(chunk)

        return chunks

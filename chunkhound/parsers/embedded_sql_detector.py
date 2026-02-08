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

from tree_sitter import Node

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


class EmbeddedSqlDetector:
    """Detects and extracts SQL code embedded in string literals.

    This detector works across all programming languages by:
    1. Finding string literal nodes in the AST
    2. Applying heuristics to identify SQL content
    3. Extracting and cleaning the SQL code
    4. Returning matches with metadata
    """

    # SQL keywords that indicate SQL content (case-insensitive)
    SQL_KEYWORDS = {
        # DML (Data Manipulation Language)
        "SELECT", "INSERT", "UPDATE", "DELETE", "MERGE",
        # DDL (Data Definition Language)
        "CREATE", "ALTER", "DROP", "TRUNCATE",
        # DCL (Data Control Language)
        "GRANT", "REVOKE",
        # TCL (Transaction Control Language)
        "COMMIT", "ROLLBACK", "SAVEPOINT",
        # Other common SQL
        "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER",
        "ON", "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET",
        "UNION", "INTERSECT", "EXCEPT",
        # T-SQL specific
        "EXEC", "EXECUTE", "BEGIN", "END", "DECLARE",
        "WITH", "AS", "TABLE", "VIEW", "INDEX", "PROCEDURE", "FUNCTION",
    }

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
        "string_value",
        "quoted_string",
    }

    def __init__(self, host_language: Language):
        """Initialize detector for a specific host language.

        Args:
            host_language: The programming language being parsed
        """
        self.host_language = host_language

    def detect_in_tree(self, root_node: Node, source_bytes: bytes) -> list[EmbeddedSqlMatch]:
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
        node: Node,
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
        node: Node,
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

        Uses heuristics:
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

        # Check for primary SQL keywords (30 points each, but only count one)
        primary_keywords = {"SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"}
        has_primary = False
        for keyword in primary_keywords:
            if re.search(r'\b' + keyword + r'\b', content_upper):
                score += 30
                has_primary = True
                break  # Only count one primary keyword

        # If no primary keyword, very unlikely to be SQL
        if not has_primary:
            return 0.0

        # Check for secondary SQL keywords (10 points each)
        secondary_keywords = {"FROM", "WHERE", "JOIN", "ORDER", "GROUP", "HAVING", "LIMIT", "OFFSET", "INTO", "SET"}
        for keyword in secondary_keywords:
            if re.search(r'\b' + keyword + r'\b', content_upper):
                score += 10

        # Check for SQL patterns (additional points)
        patterns = [
            (r'\bFROM\s+\w+', 15),  # FROM table - increased weight
            (r'\bWHERE\s+\w+', 10),  # WHERE condition
            (r'\bJOIN\s+\w+', 10),  # JOIN table
            (r'\bSET\s+\w+\s*=', 10),  # SET col = val
            (r'\bORDER\s+BY\b', 10),  # ORDER BY
            (r'\bGROUP\s+BY\b', 10),  # GROUP BY
            (r'\bINTO\s+\w+', 10),  # INSERT INTO / DELETE INTO
            (r'\bVALUES\s*\(', 10),  # VALUES (...)
            (r'\bTABLE\s+', 20),  # TABLE keyword (CREATE/ALTER/DROP TABLE)
            (r'\bPRIMARY\s+KEY\b', 10),  # PRIMARY KEY
            (r'\bFOREIGN\s+KEY\b', 10),  # FOREIGN KEY
        ]

        for pattern, points in patterns:
            if re.search(pattern, content_upper):
                score += points

        # Bonus: SELECT + FROM combination is very strong indicator
        if re.search(r'\bSELECT\b', content_upper) and re.search(r'\bFROM\b', content_upper):
            score += 10

        # Convert to 0-1 scale
        # A simple query like "SELECT * FROM users" should get ~0.65-0.7
        # A complex query should get close to 1.0
        normalized = min(score / 100.0, 1.0)

        return normalized

    def _get_context(self, node: Node, source_bytes: bytes) -> str:
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

    def create_embedded_sql_chunks(
        self,
        matches: list[EmbeddedSqlMatch],
        sql_parser: Any | None = None
    ) -> list[UniversalChunk]:
        """Convert detected SQL matches into UniversalChunk objects.

        Args:
            matches: Detected SQL matches
            sql_parser: Optional SQL parser to parse the SQL content

        Returns:
            List of UniversalChunk objects for embedded SQL
        """
        chunks: list[UniversalChunk] = []

        for i, match in enumerate(matches):
            # Create a chunk for the embedded SQL
            chunk = UniversalChunk(
                concept=UniversalConcept.DEFINITION,  # SQL is a definition
                name=f"embedded_sql_{i+1}",
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
                language_node_type="embedded_sql_string"
            )
            chunks.append(chunk)

        return chunks

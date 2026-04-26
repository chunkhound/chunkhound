"""Tests for embedded SQL detection in string literals."""

import pytest

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.parsers.parser_factory import ParserFactory


class TestEmbeddedSqlDetection:
    """Test embedded SQL detection across different languages."""

    def test_python_embedded_sql(self):
        """Test SQL detection in Python strings."""
        python_code = '''
def get_user(user_id):
    query = """
    SELECT id, name, email
    FROM users
    WHERE id = %s
    ORDER BY name
    """
    cursor.execute(query, (user_id,))
    return cursor.fetchone()

def insert_user(name, email):
    sql = "INSERT INTO users (name, email) VALUES (%s, %s)"
    cursor.execute(sql, (name, email))
'''

        # Create parser with embedded SQL detection enabled
        factory = ParserFactory()
        parser = factory.create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(python_code)

        # Should have regular chunks + embedded SQL chunks
        assert len(chunks) > 0

        # Find embedded SQL chunks
        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) >= 2, "Should detect at least 2 SQL queries"

        # Check first embedded SQL chunk
        select_chunk = next((c for c in embedded_chunks if "SELECT" in c.code.upper()), None)
        assert select_chunk is not None
        assert "users" in select_chunk.code.lower()
        assert select_chunk.metadata.get("host_language") == "python"
        assert select_chunk.metadata.get("sql_confidence", 0) >= 0.6

        # Check second embedded SQL chunk
        insert_chunk = next((c for c in embedded_chunks if "INSERT" in c.code.upper()), None)
        assert insert_chunk is not None
        assert insert_chunk.metadata.get("host_context", "").startswith("function:")

        # All embedded SQL chunks must have the EMBEDDED_SQL chunk type
        for chunk in embedded_chunks:
            assert chunk.chunk_type == ChunkType.EMBEDDED_SQL

    def test_javascript_embedded_sql(self):
        """Test SQL detection in JavaScript template strings."""
        js_code = '''
async function getUsers() {
    const query = `
        SELECT u.id, u.name, u.email, p.phone
        FROM users u
        LEFT JOIN phones p ON u.id = p.user_id
        WHERE u.active = true
        ORDER BY u.name
    `;
    return await db.query(query);
}

function createTable() {
    return db.execute(`
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price DECIMAL(10, 2)
        )
    `);
}
'''

        parser = ParserFactory().create_parser(Language.JAVASCRIPT, detect_embedded_sql=True)
        chunks = parser.parse_content(js_code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) >= 2

        # Verify SELECT query
        select_chunk = next((c for c in embedded_chunks if "SELECT" in c.code.upper()), None)
        assert select_chunk is not None
        assert "JOIN" in select_chunk.code.upper()
        assert select_chunk.metadata.get("host_language") == "javascript"

        # Verify CREATE TABLE
        create_chunk = next((c for c in embedded_chunks if "CREATE TABLE" in c.code.upper()), None)
        assert create_chunk is not None
        assert "products" in create_chunk.code.lower()

    def test_java_embedded_sql(self):
        """Test SQL detection in Java strings."""
        java_code = '''
public class UserDao {
    public User findById(int id) {
        String sql = "SELECT * FROM users WHERE id = ?";
        return jdbcTemplate.queryForObject(sql, new UserMapper(), id);
    }

    public void updateUser(User user) {
        String updateSql = "UPDATE users SET name = ?, email = ? WHERE id = ?";
        jdbcTemplate.update(updateSql, user.getName(), user.getEmail(), user.getId());
    }

    public void deleteUser(int id) {
        jdbcTemplate.update("DELETE FROM users WHERE id = ?", id);
    }
}
'''

        parser = ParserFactory().create_parser(Language.JAVA, detect_embedded_sql=True)
        chunks = parser.parse_content(java_code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) >= 3

        # Check for different SQL operations
        operations = set()
        for chunk in embedded_chunks:
            content_upper = chunk.code.upper()
            if "SELECT" in content_upper:
                operations.add("SELECT")
            if "UPDATE" in content_upper:
                operations.add("UPDATE")
            if "DELETE" in content_upper:
                operations.add("DELETE")

        assert len(operations) == 3, "Should detect SELECT, UPDATE, and DELETE"

    def test_no_false_positives(self):
        """Test that non-SQL strings are not detected as SQL."""
        python_code = '''
def greet(name):
    message = "Hello, welcome to our application!"
    log = f"User {name} logged in from the main page"
    return message

def format_text():
    text = """
    This is a long string that mentions SELECT
    but is not actually SQL code. It's just
    documentation or comments.
    """
    return text
'''

        parser = ParserFactory().create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(python_code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]

        # SELECT alone scores 0.30, below the 0.60 threshold — no false positives
        assert len(embedded_chunks) == 0, "SELECT without SQL structure should not be detected"

    def test_disabled_when_explicitly_off(self):
        """Test that embedded SQL detection can be disabled explicitly."""
        python_code = '''
def query_db():
    sql = "SELECT * FROM users"
    return execute(sql)
'''

        # Create parser with embedded SQL detection explicitly disabled
        parser = ParserFactory().create_parser(Language.PYTHON, detect_embedded_sql=False)
        chunks = parser.parse_content(python_code)

        # Should NOT have embedded SQL chunks
        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) == 0, "Should not detect SQL when disabled"

    def test_sql_confidence_scoring(self):
        """Test that confidence scoring works correctly."""
        python_code = '''
def test():
    # High confidence - clear SQL with multiple keywords
    high = "SELECT id, name FROM users WHERE active = 1 ORDER BY name"

    # Medium confidence - SQL but simpler
    medium = "SELECT * FROM products"

    # Low confidence - just one keyword
    low = "This text has SELECT in it but is not SQL"
'''

        parser = ParserFactory().create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(python_code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]

        # Should detect high (1.0) and medium (0.65) but not low (0.30)
        assert len(embedded_chunks) == 2

        # Check confidence scores are reasonable
        for chunk in embedded_chunks:
            confidence = chunk.metadata.get("sql_confidence", 0)
            assert 0 <= confidence <= 1.0
            assert confidence >= 0.6, "Detected SQL should have sufficient confidence"

    def test_multiline_sql(self):
        """Test detection of multi-line SQL queries."""
        python_code = '''
def complex_query():
    query = """
        SELECT
            u.id,
            u.name,
            u.email,
            COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at > '2024-01-01'
        GROUP BY u.id, u.name, u.email
        HAVING COUNT(o.id) > 5
        ORDER BY order_count DESC
        LIMIT 100
    """
    return db.execute(query)
'''

        parser = ParserFactory().create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(python_code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) >= 1

        # Find the complex query
        complex_chunk = embedded_chunks[0]
        assert "SELECT" in complex_chunk.code.upper()
        assert "GROUP BY" in complex_chunk.code.upper()
        assert "HAVING" in complex_chunk.code.upper()
        assert complex_chunk.metadata.get("sql_confidence", 0) > 0.8

    def test_ddl_confidence_scores(self):
        """DDL statements without PRIMARY KEY should score >= 0.6."""
        from chunkhound.parsers.embedded_sql_detector import EmbeddedSqlDetector
        from chunkhound.core.types.common import Language

        detector = EmbeddedSqlDetector(Language.PYTHON)
        ddl_statements = [
            "CREATE TABLE users (id INT, name TEXT)",
            "ALTER TABLE orders ADD COLUMN total DECIMAL(10,2)",
            "DROP TABLE temp_data",
        ]
        for stmt in ddl_statements:
            confidence = detector._calculate_sql_confidence(stmt)
            assert confidence >= 0.6, f"Expected >= 0.6 for {stmt!r}, got {confidence}"

    def test_ddl_detection_without_primary_key(self):
        """Integration test: DDL strings without PRIMARY KEY are detected."""
        python_code = '''
def setup_db():
    conn.execute("CREATE TABLE users (id INT, name TEXT)")
    conn.execute("ALTER TABLE orders ADD COLUMN total DECIMAL(10,2)")
    conn.execute("DROP TABLE temp_data")
'''
        parser = ParserFactory().create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(python_code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) == 3, (
            f"Expected 3 DDL chunks, got {len(embedded_chunks)}: "
            f"{[c.code for c in embedded_chunks]}"
        )

    def test_clean_string_content_strips_prefixes(self):
        """_clean_string_content must strip language-specific string prefixes."""
        from chunkhound.parsers.embedded_sql_detector import EmbeddedSqlDetector

        detector = EmbeddedSqlDetector(Language.PYTHON)
        sql = "SELECT * FROM users WHERE id = 1"

        cases = [
            (f'r"{sql}"', sql),
            (f'b"{sql}"', sql),
            (f'f"{sql}"', sql),
            (f'rb"{sql}"', sql),
            (f'R"{sql}"', sql),
            (f'@"{sql}"', sql),    # C# verbatim prefix
            (f'$"{sql}"', sql),    # C# interpolated prefix
            (f'@$"{sql}"', sql),   # C# verbatim+interpolated
            (f'$@"{sql}"', sql),   # C# interpolated+verbatim
            (f'"{sql}"', sql),     # plain — should still work
        ]
        for raw, expected in cases:
            result = detector._clean_string_content(raw)
            assert result == expected, (
                f"_clean_string_content({raw!r}) → {result!r}, expected {expected!r}"
            )

    def test_python_raw_string_sql_detected(self):
        """Integration: r-prefixed Python SQL strings must produce clean embedded chunks."""
        python_code = '''
def fetch_users():
    query = r"SELECT * FROM users WHERE active = 1"
    return db.execute(query)
'''
        parser = ParserFactory().create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(python_code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) >= 1, "Should detect SQL in r-prefixed string"

        chunk = embedded_chunks[0]
        # The stored sql_content must NOT start with the raw prefix or quotes
        assert not chunk.code.startswith('r"'), (
            f"sql_content should not contain raw prefix, got: {chunk.code!r}"
        )
        assert "SELECT" in chunk.code.upper()
        assert "users" in chunk.code.lower()

    def test_bare_dml_confidence_scores(self):
        """Bare DML statements must score >= 0.6 to cross the detection threshold."""
        from chunkhound.parsers.embedded_sql_detector import EmbeddedSqlDetector

        detector = EmbeddedSqlDetector(Language.PYTHON)
        cases = [
            "DELETE FROM users WHERE id = 1",
            "UPDATE users SET name = 'Alice' WHERE id = 1",
            "INSERT INTO users (name) VALUES ('Alice')",
        ]
        for sql in cases:
            score = detector._calculate_sql_confidence(sql)
            assert score >= 0.6, (
                f"Expected confidence >= 0.6 for {sql!r}, got {score}"
            )

    def test_english_prose_with_sql_keywords_not_detected(self):
        """English sentences with SQL keywords must not be detected as SQL."""
        python_code = '''
def describe():
    msg = "SELECT from the list WHERE possible"
    return msg
'''
        parser = ParserFactory().create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(python_code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) == 0, (
            f"English prose should not be detected as SQL, got: {[c.code for c in embedded_chunks]}"
        )

    def test_select_from_gap_threshold(self):
        """_select_from_combo_fires must reject 3+ pure-alpha words between SELECT and FROM."""
        from chunkhound.parsers.embedded_sql_detector import _select_from_combo_fires

        # 3 pure-alpha words — English prose, must be rejected
        fires, structural = _select_from_combo_fires("SELECT NAME AGE STATUS FROM USERS")
        assert not fires
        # 2 pure-alpha words — plausible column list, must pass but is NOT structural
        fires, structural = _select_from_combo_fires("SELECT ID NAME FROM USERS")
        assert fires
        assert not structural
        # 1 word — clearly a column or table name, must pass but is NOT structural
        fires, structural = _select_from_combo_fires("SELECT ID FROM USERS")
        assert fires
        assert not structural
        # structural chars (comma) — always passes and IS structural
        fires, structural = _select_from_combo_fires("SELECT ID, NAME, AGE FROM USERS")
        assert fires
        assert structural

    def test_english_prose_select_from_scores_below_threshold(self):
        """Regression: English phrases like SELECT name FROM the_list must score < 0.6."""
        from chunkhound.parsers.embedded_sql_detector import EmbeddedSqlDetector
        from chunkhound.core.types.common import Language

        detector = EmbeddedSqlDetector(Language.PYTHON)
        score = detector._calculate_sql_confidence("SELECT name FROM the_list WHERE possible")
        assert score < 0.6, f"English prose scored {score:.2f}, expected < 0.6"

    def test_sql_language_no_self_referential_detection(self):
        """detect_embedded_sql=True on a SQL file must not produce embedded chunks."""
        sql_code = """
CREATE TABLE users (id INT PRIMARY KEY, name TEXT);
SELECT * FROM users WHERE id = 1;
"""
        parser = ParserFactory().create_parser(Language.SQL, detect_embedded_sql=True)
        chunks = parser.parse_content(sql_code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) == 0, (
            f"SQL files should not self-detect embedded SQL, got: {[c.code for c in embedded_chunks]}"
        )

    @pytest.mark.parametrize("language", [
        Language.PYTHON,
        Language.JAVASCRIPT,
        Language.TYPESCRIPT,
        Language.JAVA,
        Language.CSHARP,
        Language.GO,
        Language.RUST,
        Language.PHP,
    ])
    def test_cross_language_detection(self, language):
        """Test that detection works across different languages."""
        # Simple SQL that should work in any language
        code_templates = {
            Language.PYTHON: 'query = "SELECT * FROM users WHERE id = 1"',
            Language.JAVASCRIPT: 'const query = "SELECT * FROM users WHERE id = 1";',
            Language.TYPESCRIPT: 'const query: string = "SELECT * FROM users WHERE id = 1";',
            Language.JAVA: 'String query = "SELECT * FROM users WHERE id = 1";',
            Language.CSHARP: 'string query = "SELECT * FROM users WHERE id = 1";',
            Language.GO: 'query := "SELECT * FROM users WHERE id = 1"',
            Language.RUST: 'let query = "SELECT * FROM users WHERE id = 1";',
            Language.PHP: '<?php $query = "SELECT * FROM users WHERE id = 1"; ?>',
        }

        if language not in code_templates:
            pytest.skip(f"No template for {language}")

        code = code_templates[language]

        if not ParserFactory().is_language_available(language):
            pytest.skip(f"{language} parser not available")

        parser = ParserFactory().create_parser(language, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded_chunks = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded_chunks) >= 1, f"Should detect SQL in {language.value}"
        assert embedded_chunks[0].metadata.get("host_language") == language.value

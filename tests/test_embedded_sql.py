"""Tests for embedded SQL detection in string literals."""

import pytest

from chunkhound.core.types.common import Language
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

        # Should have very few or no false positives
        # The second string mentions SELECT but lacks other SQL structure
        assert len(embedded_chunks) <= 1, "Should not have many false positives"

    def test_disabled_by_default(self):
        """Test that embedded SQL detection is disabled by default."""
        python_code = '''
def query_db():
    sql = "SELECT * FROM users"
    return execute(sql)
'''

        # Create parser WITHOUT embedded SQL detection
        parser = ParserFactory().create_parser(Language.PYTHON)
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

        # Should detect the clear SQL statements
        assert len(embedded_chunks) >= 1

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

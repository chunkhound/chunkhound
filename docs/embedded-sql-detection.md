# Embedded SQL Detection

## Overview

ChunkHound now supports detecting and extracting SQL code embedded in string literals across all supported programming languages. This feature addresses the common pattern of SQL being embedded as strings in application code (Python, Java, JavaScript, etc.).

## Motivation

As noted by the maintainers, SQL is frequently embedded in strings across various programming languages:
```python
cursor.execute("SELECT * FROM users WHERE id = %s")
```

Without embedded SQL detection, ChunkHound would only see this as a string literal in a function, missing the semantic SQL content. With this feature enabled, ChunkHound extracts the SQL as a separate, searchable chunk.

## Architecture

### Components

1. **EmbeddedSqlDetector** (`chunkhound/parsers/embedded_sql_detector.py`)
   - Recursively visits all AST nodes to find string literals
   - Applies heuristic confidence scoring to determine if a string contains SQL
   - Extracts SQL content with context (host function/class)

2. **UniversalParser Integration** (`chunkhound/parsers/universal_parser.py`)
   - Optional `detect_embedded_sql` parameter in parser initialization
   - Post-processing step after normal chunking
   - Creates additional chunks for detected SQL

3. **ParserFactory Support** (`chunkhound/parsers/parser_factory.py`)
   - `detect_embedded_sql` parameter propagated through factory
   - Disabled by default for backward compatibility

### Detection Heuristics

The detector uses a multi-factor confidence scoring system:

#### Primary SQL Keywords (30 points each, one required)
- SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, DROP

#### Secondary Keywords (10 points each)
- FROM, WHERE, JOIN, ORDER, GROUP, HAVING, LIMIT, OFFSET, INTO, SET

#### SQL Patterns (10 points each)
- `FROM table_name`
- `WHERE condition`
- `JOIN table_name`
- `SET column = value`
- `ORDER BY ...`
- `GROUP BY ...`
- `INTO table_name`
- `VALUES (...)`

**Confidence Threshold**: 0.6 (60 points minimum)

Simple queries like `SELECT * FROM users` score ~0.7, while complex queries with multiple clauses score 0.8-0.9+.

## Usage

### Programmatic API

```python
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.core.types.common import Language

# Create parser with embedded SQL detection
factory = ParserFactory()
parser = factory.create_parser(
    Language.PYTHON,
    detect_embedded_sql=True  # Enable detection
)

# Parse code containing embedded SQL
code = '''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, user_id)
'''

chunks = parser.parse_content(code)

# Find embedded SQL chunks
sql_chunks = [c for c in chunks if c.metadata.get("embedded")]
for chunk in sql_chunks:
    print(f"Found SQL: {chunk.code}")
    print(f"Confidence: {chunk.metadata['sql_confidence']}")
    print(f"Context: {chunk.metadata['host_context']}")
```

### Chunk Metadata

Embedded SQL chunks include special metadata:
```python
{
    "embedded": True,
    "host_language": "python",  # Original file language
    "host_context": "function:get_user",  # Where SQL was found
    "sql_confidence": 0.7,  # Detection confidence (0-1)
    "detected_language": "sql",
}
```

## Language Support

### Verified Working (13/15 tests passing)
- ✅ Python - string literals and docstrings
- ✅ Java - string literals
- ✅ TypeScript - string literals
- ✅ C# - string literals
- ✅ Go - string literals
- ✅ Rust - string literals

### Partial Support
- ⚠️ JavaScript - regular strings work, template strings need refinement
- ⚠️ PHP - needs string node type mapping

### Not Yet Tested
- Ruby, Kotlin, Swift, etc. (should work but not verified)

## Examples

### Python with Triple-Quoted Strings
```python
def complex_query():
    sql = """
        SELECT u.id, u.name, COUNT(o.id) as orders
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.active = true
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 5
    """
    return db.execute(sql)
```
**Detected**: Yes (confidence: 0.9+)

### Java JDBC
```java
public User findById(int id) {
    String sql = "SELECT * FROM users WHERE id = ?";
    return jdbcTemplate.queryForObject(sql, new UserMapper(), id);
}
```
**Detected**: Yes (confidence: 0.7)

### JavaScript/TypeScript ORM
```javascript
const users = await db.query(
    "SELECT * FROM users WHERE created_at > $1 ORDER BY name",
    [date]
);
```
**Detected**: Yes (confidence: 0.9)

## Performance Considerations

- **Minimal Overhead**: Detection only runs when explicitly enabled
- **No SQL Parsing**: Uses heuristics, doesn't parse SQL into AST
- **Single Pass**: Visits each AST node once during tree traversal
- **Confidence Filtering**: Only creates chunks for high-confidence matches

## Future Enhancements

### Short Term
1. **Improve JavaScript template string support** - handle backtick strings
2. **Add PHP string node types** - map PHP-specific AST nodes
3. **Configuration options** - allow custom confidence thresholds

### Long Term
1. **SQL Dialect Detection** - distinguish PostgreSQL, MySQL, T-SQL, etc.
2. **Parameterized Query Analysis** - detect and extract parameter placeholders
3. **SQL Validation** - optionally parse and validate SQL syntax
4. **Cross-Reference Linking** - link SQL chunks to their calling functions

## Testing

Run embedded SQL detection tests:
```bash
uv run pytest tests/test_embedded_sql.py -v
```

Current test results: **13/15 passing** (86.7% success rate)

## Configuration

Currently controlled via parser initialization parameter. Future versions may support:
- Configuration file settings
- Per-language confidence thresholds
- Whitelist/blacklist of SQL keywords
- Custom detection patterns

## Related Work

This feature addresses [maintainer feedback](https://github.com/chunkhound/chunkhound/pull/183) about SQL being embedded in strings across languages.

## Credits

Designed and implemented by AI agents in collaboration with the ChunkHound community.
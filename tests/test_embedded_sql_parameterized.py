"""Tests for embedded SQL detection with parameterized queries from various frameworks."""

import pytest

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import ParserFactory


class TestParameterizedQueries:
    """Test detection of parameterized queries across different database frameworks."""

    def test_python_mysqldb_percent_s(self):
        """Test Python MySQLdb/MySQL Connector style (%s placeholders)."""
        code = '''
import mysql.connector

def get_user(user_id):
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    return cursor.fetchone()

def insert_user(name, email):
    sql = "INSERT INTO users (name, email) VALUES (%s, %s)"
    cursor.execute(sql, (name, email))
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 2

        # Verify SELECT with %s
        select_chunk = next(c for c in embedded if "SELECT" in c.code.upper())
        assert "%s" in select_chunk.code
        assert "WHERE id = %s" in select_chunk.code

        # Verify INSERT with %s
        insert_chunk = next(c for c in embedded if "INSERT" in c.code.upper())
        assert "%s" in insert_chunk.code
        assert "VALUES (%s, %s)" in insert_chunk.code

    def test_python_psycopg2_percent_s(self):
        """Test Python psycopg2 (PostgreSQL) style (%s and %(name)s)."""
        code = '''
import psycopg2

def get_users_by_status(status, limit):
    query = "SELECT * FROM users WHERE status = %s LIMIT %s"
    cursor.execute(query, (status, limit))
    return cursor.fetchall()

def update_user_email(user_id, email):
    query = "UPDATE users SET email = %(email)s WHERE id = %(user_id)s"
    cursor.execute(query, {"email": email, "user_id": user_id})
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 2

        # Verify SELECT with positional %s
        select_chunk = next(c for c in embedded if "SELECT" in c.code.upper())
        assert "%s" in select_chunk.code

        # Verify UPDATE with named %(name)s
        update_chunk = next(c for c in embedded if "UPDATE" in c.code.upper())
        assert "%(email)s" in update_chunk.code
        assert "%(user_id)s" in update_chunk.code

    def test_python_sqlite3_question_mark(self):
        """Test Python sqlite3 style (? placeholders)."""
        code = '''
import sqlite3

def find_by_id(conn, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()

def batch_insert(conn, users):
    sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)"
    cursor = conn.cursor()
    cursor.executemany(sql, users)
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 2

        # Both should have ? placeholders
        for chunk in embedded:
            assert "?" in chunk.code

    def test_java_jdbc_question_mark(self):
        """Test Java JDBC style (? placeholders)."""
        code = '''
public class UserDao {
    public User findById(Connection conn, int id) throws SQLException {
        String sql = "SELECT * FROM users WHERE id = ?";
        PreparedStatement stmt = conn.prepareStatement(sql);
        stmt.setInt(1, id);
        return stmt.executeQuery();
    }

    public void updateUser(Connection conn, String name, String email, int id) {
        String sql = "UPDATE users SET name = ?, email = ? WHERE id = ?";
        PreparedStatement stmt = conn.prepareStatement(sql);
        stmt.setString(1, name);
        stmt.setString(2, email);
        stmt.setInt(3, id);
        stmt.executeUpdate();
    }

    public void deleteOldUsers(Connection conn, Date cutoffDate) {
        String sql = "DELETE FROM users WHERE created_at < ? AND active = ?";
        PreparedStatement stmt = conn.prepareStatement(sql);
        stmt.setDate(1, cutoffDate);
        stmt.setBoolean(2, false);
        stmt.executeUpdate();
    }
}
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.JAVA, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 3

        # All should contain ? placeholders
        for chunk in embedded:
            assert "?" in chunk.code

    def test_javascript_pg_dollar_placeholders(self):
        """Test JavaScript node-postgres style ($1, $2, etc.)."""
        code = '''
const { Pool } = require('pg');
const pool = new Pool();

async function getUserById(userId) {
    const result = await pool.query(
        "SELECT * FROM users WHERE id = $1",
        [userId]
    );
    return result.rows[0];
}

async function insertUser(name, email, age) {
    const query = "INSERT INTO users (name, email, age) VALUES ($1, $2, $3) RETURNING id";
    const result = await pool.query(query, [name, email, age]);
    return result.rows[0].id;
}

async function searchUsers(searchTerm, limit, offset) {
    const sql = `
        SELECT * FROM users
        WHERE name ILIKE $1 OR email ILIKE $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
    `;
    const result = await pool.query(sql, [`%${searchTerm}%`, limit, offset]);
    return result.rows;
}
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.JAVASCRIPT, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 3

        # Check for numbered placeholders
        select_chunk = next(c for c in embedded if "WHERE id = $1" in c.code)
        assert "$1" in select_chunk.code

        insert_chunk = next(c for c in embedded if "RETURNING" in c.code.upper())
        assert all(f"${i}" in insert_chunk.code for i in [1, 2, 3])

        search_chunk = next(c for c in embedded if "ILIKE" in c.code.upper())
        assert "$1" in search_chunk.code and "$2" in search_chunk.code and "$3" in search_chunk.code

    def test_javascript_mysql2_question_mark(self):
        """Test JavaScript mysql2 style (? placeholders)."""
        code = '''
const mysql = require('mysql2/promise');

async function getUser(connection, userId) {
    const [rows] = await connection.execute(
        'SELECT * FROM users WHERE id = ?',
        [userId]
    );
    return rows[0];
}

async function updateUserStatus(connection, userId, status) {
    const sql = 'UPDATE users SET status = ?, updated_at = NOW() WHERE id = ?';
    await connection.execute(sql, [status, userId]);
}
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.JAVASCRIPT, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 2

        # Both should have ? placeholders
        for chunk in embedded:
            assert "?" in chunk.code

    def test_csharp_sqlcommand_at_params(self):
        """Test C# SqlCommand style (@param placeholders)."""
        code = '''
using System.Data.SqlClient;

public class UserRepository
{
    public User GetById(SqlConnection conn, int userId)
    {
        var sql = "SELECT * FROM Users WHERE Id = @userId";
        using (var cmd = new SqlCommand(sql, conn))
        {
            cmd.Parameters.AddWithValue("@userId", userId);
            return cmd.ExecuteReader();
        }
    }

    public void UpdateUser(SqlConnection conn, int id, string name, string email)
    {
        var sql = "UPDATE Users SET Name = @name, Email = @email WHERE Id = @id";
        using (var cmd = new SqlCommand(sql, conn))
        {
            cmd.Parameters.AddWithValue("@name", name);
            cmd.Parameters.AddWithValue("@email", email);
            cmd.Parameters.AddWithValue("@id", id);
            cmd.ExecuteNonQuery();
        }
    }

    public List<User> SearchUsers(SqlConnection conn, string searchTerm, int maxResults)
    {
        var query = @"
            SELECT TOP (@maxResults) *
            FROM Users
            WHERE Name LIKE @searchTerm OR Email LIKE @searchTerm
            ORDER BY CreatedAt DESC
        ";
        using (var cmd = new SqlCommand(query, conn))
        {
            cmd.Parameters.AddWithValue("@searchTerm", "%" + searchTerm + "%");
            cmd.Parameters.AddWithValue("@maxResults", maxResults);
            return cmd.ExecuteReader();
        }
    }
}
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.CSHARP, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 3

        # Check for @ placeholders
        select_chunk = next(c for c in embedded if "SELECT" in c.code.upper() and "@userId" in c.code)
        assert "@userId" in select_chunk.code

        update_chunk = next(c for c in embedded if "UPDATE" in c.code.upper())
        assert "@name" in update_chunk.code and "@email" in update_chunk.code and "@id" in update_chunk.code

        search_chunk = next(c for c in embedded if "LIKE @searchTerm" in c.code)
        assert "@searchTerm" in search_chunk.code and "@maxResults" in search_chunk.code

    def test_go_pq_dollar_placeholders(self):
        """Test Go pq (PostgreSQL) style ($1, $2, etc.)."""
        code = '''
package main

import (
    "database/sql"
    _ "github.com/lib/pq"
)

func getUserByID(db *sql.DB, userID int) (*User, error) {
    query := "SELECT id, name, email FROM users WHERE id = $1"
    row := db.QueryRow(query, userID)

    var user User
    err := row.Scan(&user.ID, &user.Name, &user.Email)
    return &user, err
}

func insertUser(db *sql.DB, name, email string) (int64, error) {
    query := "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id"
    var id int64
    err := db.QueryRow(query, name, email).Scan(&id)
    return id, err
}

func searchUsers(db *sql.DB, searchTerm string, limit, offset int) ([]User, error) {
    query := `
        SELECT id, name, email
        FROM users
        WHERE name ILIKE $1 OR email ILIKE $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
    `
    rows, err := db.Query(query, "%"+searchTerm+"%", limit, offset)
    // ... scan rows
}
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.GO, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 3

        # Check for numbered placeholders
        for chunk in embedded:
            assert any(f"${i}" in chunk.code for i in range(1, 4))

    def test_php_pdo_named_placeholders(self):
        """Test PHP PDO style (:param placeholders)."""
        code = '''<?php
class UserRepository {
    private $pdo;

    public function getUserById($userId) {
        $sql = "SELECT * FROM users WHERE id = :user_id";
        $stmt = $this->pdo->prepare($sql);
        $stmt->execute(['user_id' => $userId]);
        return $stmt->fetch();
    }

    public function updateUser($id, $name, $email) {
        $sql = "UPDATE users SET name = :name, email = :email WHERE id = :id";
        $stmt = $this->pdo->prepare($sql);
        $stmt->execute([
            'name' => $name,
            'email' => $email,
            'id' => $id
        ]);
    }

    public function searchUsers($searchTerm, $limit, $offset) {
        $sql = "SELECT * FROM users
                WHERE name LIKE :search OR email LIKE :search
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset";
        $stmt = $this->pdo->prepare($sql);
        $stmt->execute([
            'search' => "%{$searchTerm}%",
            'limit' => $limit,
            'offset' => $offset
        ]);
        return $stmt->fetchAll();
    }
}
?>'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.PHP, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 3

        # Check for named placeholders with colons
        select_chunk = next(c for c in embedded if "SELECT" in c.code.upper() and ":user_id" in c.code)
        assert ":user_id" in select_chunk.code

        update_chunk = next(c for c in embedded if "UPDATE" in c.code.upper())
        assert ":name" in update_chunk.code and ":email" in update_chunk.code and ":id" in update_chunk.code

        search_chunk = next(c for c in embedded if "LIKE :search" in c.code)
        assert ":search" in search_chunk.code and ":limit" in search_chunk.code

    def test_php_mysqli_question_mark(self):
        """Test PHP MySQLi style (? placeholders)."""
        code = '''<?php
class UserDao {
    private $mysqli;

    public function findById($userId) {
        $stmt = $this->mysqli->prepare("SELECT * FROM users WHERE id = ?");
        $stmt->bind_param("i", $userId);
        $stmt->execute();
        return $stmt->get_result()->fetch_assoc();
    }

    public function insertUser($name, $email, $age) {
        $sql = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";
        $stmt = $this->mysqli->prepare($sql);
        $stmt->bind_param("ssi", $name, $email, $age);
        $stmt->execute();
        return $stmt->insert_id;
    }
}
?>'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.PHP, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 2

        # Both should have ? placeholders
        for chunk in embedded:
            assert "?" in chunk.code

    def test_rust_tokio_postgres_dollar_placeholders(self):
        """Test Rust tokio-postgres style ($1, $2, etc.)."""
        code = '''
use tokio_postgres::{Client, Error};

async fn get_user_by_id(client: &Client, user_id: i32) -> Result<User, Error> {
    let row = client
        .query_one("SELECT id, name, email FROM users WHERE id = $1", &[&user_id])
        .await?;

    Ok(User {
        id: row.get(0),
        name: row.get(1),
        email: row.get(2),
    })
}

async fn insert_user(client: &Client, name: &str, email: &str) -> Result<i64, Error> {
    let row = client
        .query_one(
            "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
            &[&name, &email]
        )
        .await?;
    Ok(row.get(0))
}

async fn update_user_email(client: &Client, user_id: i32, new_email: &str) -> Result<u64, Error> {
    let updated = client
        .execute(
            "UPDATE users SET email = $1, updated_at = NOW() WHERE id = $2",
            &[&new_email, &user_id]
        )
        .await?;
    Ok(updated)
}
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.RUST, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 3

        # Check for numbered placeholders
        select_chunk = next(c for c in embedded if "WHERE id = $1" in c.code)
        assert "$1" in select_chunk.code

        insert_chunk = next(c for c in embedded if "RETURNING" in c.code.upper())
        assert "$1" in insert_chunk.code and "$2" in insert_chunk.code

        update_chunk = next(c for c in embedded if "UPDATE" in c.code.upper())
        assert "$1" in update_chunk.code and "$2" in update_chunk.code

    def test_typescript_typeorm_style(self):
        """Test TypeScript TypeORM query builder (not detected - uses builder pattern)."""
        code = '''
import { getRepository } from 'typeorm';

async function getUserById(userId: number) {
    // Raw query - should be detected
    const rawQuery = "SELECT * FROM users WHERE id = $1";
    const result = await getRepository(User).query(rawQuery, [userId]);
    return result[0];
}

async function complexQuery(status: string, minAge: number) {
    const sql = `
        SELECT u.id, u.name, COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.status = $1 AND u.age >= $2
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 0
        ORDER BY order_count DESC
    `;
    return await getRepository(User).query(sql, [status, minAge]);
}
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.TYPESCRIPT, detect_embedded_sql=True)
        chunks = parser.parse_content(code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]

        # Should detect the raw SQL queries
        assert len(embedded) == 2

        # Verify both have parameterized placeholders
        assert any("$1" in c.code for c in embedded)
        assert any("$1" in c.code and "$2" in c.code for c in embedded)

    def test_mixed_placeholder_styles(self):
        """Test that different placeholder styles don't interfere."""
        python_code = '''
# MySQL style
query1 = "SELECT * FROM users WHERE id = %s AND status = %s"

# PostgreSQL style
query2 = "SELECT * FROM users WHERE id = %(user_id)s"

# SQLite style
query3 = "SELECT * FROM users WHERE id = ?"
'''

        factory = ParserFactory()
        parser = factory.create_parser(Language.PYTHON, detect_embedded_sql=True)
        chunks = parser.parse_content(python_code)

        embedded = [c for c in chunks if c.metadata.get("embedded")]
        assert len(embedded) == 3

        # Each should maintain its placeholder style
        assert any("%s" in c.code and "status" in c.code for c in embedded)
        assert any("%(user_id)s" in c.code for c in embedded)
        assert any("?" in c.code and "%s" not in c.code and "$" not in c.code for c in embedded)

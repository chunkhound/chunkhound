"""CLI commands for migrating databases to global mode."""

import sys
from pathlib import Path

from loguru import logger

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.providers.database.duckdb_provider import DuckDBProvider

try:
    import duckdb
except ImportError:
    duckdb = None


async def discover_command(args, formatter: RichOutputFormatter) -> None:
    """Discover existing per-repo databases.

    Args:
        args: Parsed command-line arguments (search_path, max_depth)
        formatter: Output formatter for displaying results
    """
    if duckdb is None:
        formatter.error("DuckDB not available. Install with: uv pip install duckdb")
        sys.exit(1)

    search_path = args.search_path
    max_depth = args.max_depth

    formatter.section_header(f"Discovering ChunkHound databases in: {search_path}")
    formatter.info(f"Max depth: {max_depth}\n")

    discovered = []

    for db_path in search_path.rglob(".chunkhound/db/chunks.db"):
        # Check depth
        try:
            relative = db_path.relative_to(search_path)
            depth = len(relative.parents)
            if depth > max_depth:
                continue
        except ValueError:
            continue

        # Validate it's a DuckDB file
        try:
            conn = duckdb.connect(str(db_path), read_only=True)

            # Check for required tables
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = {t[0] for t in tables}

            has_files = "files" in table_names
            has_chunks = "chunks" in table_names

            if not (has_files and has_chunks):
                formatter.warning(f"Skipping (missing tables): {db_path}")
                conn.close()
                continue

            # Get statistics
            file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

            # Get embedding status (check for any embeddings* table)
            embedding_tables = [t for t in table_names if t.startswith("embeddings")]
            embedding_count = 0
            for et in embedding_tables:
                try:
                    # Validate table name to prevent SQL injection (defense in depth)
                    if not et.replace("_", "").replace("-", "").isalnum():
                        formatter.warning(f"Skipping table with invalid name: {et}")
                        continue
                    count = conn.execute(f"SELECT COUNT(*) FROM {et}").fetchone()[0]
                    embedding_count += count
                except Exception:
                    pass

            conn.close()

            # Derive base_directory from db path
            base_dir = db_path.parent.parent.parent

            discovered.append(
                {
                    "db_path": db_path,
                    "base_directory": base_dir,
                    "file_count": file_count,
                    "chunk_count": chunk_count,
                    "embedding_count": embedding_count,
                    "has_embeddings": embedding_count > 0,
                }
            )

            formatter.success(f"Found: {base_dir}")
            formatter.verbose_info(
                f"  Files: {file_count}, Chunks: {chunk_count}, Embeddings: {embedding_count}"
            )

        except Exception as e:
            formatter.warning(f"Skipping (not valid DuckDB): {db_path} ({e})")
            continue

    formatter.section_header(
        f"\nDiscovery complete: {len(discovered)} database(s) found"
    )

    if discovered:
        formatter.info("\nTo migrate a database, run:")
        formatter.info("  chunkhound migrate to-global --source <db_path>")
    else:
        formatter.info("No databases found. Try adjusting --search-path or --max-depth")


def get_common_columns(conn, table1: str, table2: str) -> list[str]:
    """Get intersection of columns between two tables."""
    try:
        cols1 = set(row[0] for row in conn.execute(f"DESCRIBE {table1}").fetchall())
        cols2 = set(row[0] for row in conn.execute(f"DESCRIBE {table2}").fetchall())
        return list(cols1 & cols2)
    except Exception:
        return []


async def migrate_command(args, formatter: RichOutputFormatter) -> None:
    """Migrate a per-repo database to global mode.

    Args:
        args: Parsed command-line arguments (source, global_db, base_dir, dry_run)
        formatter: Output formatter for displaying results
    """
    if duckdb is None:
        formatter.error("DuckDB not available. Install with: uv pip install duckdb")
        sys.exit(1)

    source_db = args.source
    global_db = args.global_db
    base_directory = args.base_dir
    dry_run = args.dry_run

    if base_directory is None:
        # Auto-detect: /project/.chunkhound/db/chunks.db -> /project
        base_directory = source_db.parent.parent.parent

    formatter.section_header(f"{'[DRY RUN] ' if dry_run else ''}Migrating database")
    formatter.info(f"Source: {source_db}")
    formatter.info(f"Target: {global_db}")
    formatter.info(f"Base directory: {base_directory}\n")

    if dry_run:
        formatter.info("(No changes will be made)\n")

    # Initialize variables for cleanup
    source_conn = None
    target_provider = None
    target_conn = None

    try:
        # Step 1: Validate source
        formatter.info("[1/4] Validating source database...")
        source_conn = duckdb.connect(str(source_db), read_only=True)

        # Check tables exist
        tables = source_conn.execute("SHOW TABLES").fetchall()
        source_tables = {t[0] for t in tables}

        required_tables = {"files", "chunks"}
        missing = required_tables - source_tables
        if missing:
            formatter.error(f"Missing required tables in source: {missing}")
            source_conn.close()
            sys.exit(1)

        # Get row counts
        file_count = source_conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunk_count = source_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        # Identify embedding tables
        embedding_tables = [t for t in source_tables if t.startswith("embeddings")]
        embedding_count = 0
        for et in embedding_tables:
            # Validate table name to prevent SQL injection (defense in depth)
            if not et.replace("_", "").replace("-", "").isalnum():
                formatter.warning(f"Skipping table with invalid name: {et}")
                continue
            embedding_count += source_conn.execute(
                f"SELECT COUNT(*) FROM {et}"
            ).fetchone()[0]

        formatter.success(
            f"Source valid: {file_count} files, {chunk_count} chunks, {embedding_count} embeddings"
        )

        if dry_run:
            source_conn.close()
            formatter.success("\n[OK] DRY RUN complete - migration would succeed")
            return

        # Close read-only source connection to avoid lock issues
        source_conn.close()

        # Step 2: Initialize global database using Provider (ensures correct schema)
        formatter.info("[2/4] Initializing global database...")
        global_db.parent.mkdir(parents=True, exist_ok=True)

        # Use DuckDBProvider to initialize schema correctly
        target_provider = DuckDBProvider(db_path=global_db, base_directory=Path.home())
        target_provider.connect()
        target_provider.disconnect()  # Close to release lock for migration connection

        # Now open our own connection for migration
        target_conn = duckdb.connect(str(global_db))

        # Enable VSS extension just in case
        try:
            target_conn.execute("INSTALL vss; LOAD vss;")
        except Exception:
            pass

        formatter.success("Global database ready")

        # Steps 3-4: Atomic transaction (registration + data copy)
        formatter.info("[3/4] Preparing atomic migration (registration + data copy)...")

        # SECURITY: Path sanitization for ATTACH command
        resolved_path = source_db.resolve()

        if len(str(resolved_path)) > 4096:
            formatter.error("Source path too long")
            sys.exit(1)
        if "\x00" in str(resolved_path):
            formatter.error("Invalid null byte in source path")
            sys.exit(1)
        if not resolved_path.is_absolute():
            formatter.error("Source must be an absolute path")
            sys.exit(1)
        if not resolved_path.exists() or not resolved_path.is_file():
            formatter.error(f"Source database not found: {resolved_path}")
            sys.exit(1)
        if resolved_path == global_db.resolve():
            formatter.error("Source and target databases are the same file!")
            sys.exit(1)

        safe_source_path = str(resolved_path).replace("'", "''")
        attach_sql = f"ATTACH '{safe_source_path}' AS source_db (READ_ONLY)"

        # Start SINGLE transaction
        target_conn.execute("BEGIN TRANSACTION")
        try:
            # Attach source database
            target_conn.execute(attach_sql)

            # Step 3: Register base_directory
            project_name = base_directory.name
            target_conn.execute(
                """
                INSERT OR REPLACE INTO indexed_roots 
                (base_directory, project_name, indexed_at, updated_at, file_count)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """,
                [str(base_directory), project_name, file_count],
            )
            formatter.verbose_info(f"Registered: {project_name}")

            # Step 4: Copy data with safe column mapping

            # Copy FILES
            common_files_cols = get_common_columns(
                target_conn, "files", "source_db.files"
            )
            if not common_files_cols:
                raise RuntimeError("No common columns found for 'files' table")

            cols_str = ", ".join(common_files_cols)
            formatter.verbose_info(
                f"Copying files ({len(common_files_cols)} columns)..."
            )
            target_conn.execute(f"""
                INSERT OR IGNORE INTO files ({cols_str})
                SELECT {cols_str} FROM source_db.files
            """)

            # Copy CHUNKS
            common_chunks_cols = get_common_columns(
                target_conn, "chunks", "source_db.chunks"
            )
            if not common_chunks_cols:
                raise RuntimeError("No common columns found for 'chunks' table")

            cols_str = ", ".join(common_chunks_cols)
            formatter.verbose_info(
                f"Copying chunks ({len(common_chunks_cols)} columns)..."
            )
            target_conn.execute(f"""
                INSERT OR IGNORE INTO chunks ({cols_str})
                SELECT {cols_str} FROM source_db.chunks
            """)

            # Copy EMBEDDINGS (handle multiple tables)
            for et in embedding_tables:
                # Validate table name to prevent SQL injection (defense in depth)
                if not et.replace("_", "").replace("-", "").isalnum():
                    formatter.warning(f"Skipping table with invalid name: {et}")
                    continue

                if et == "embeddings":
                    # Legacy table: Copy to temp table for later migration
                    formatter.verbose_info(
                        "Detected legacy embeddings table, queuing for migration..."
                    )

                    # Create legacy table in target if not exists
                    target_conn.execute("""
                        CREATE TABLE IF NOT EXISTS embeddings (
                            id INTEGER, chunk_id INTEGER, provider TEXT, 
                            model TEXT, embedding FLOAT[], dims INTEGER, 
                            created_at TIMESTAMP
                        )
                    """)

                    # Copy data
                    common_cols = get_common_columns(
                        target_conn, "embeddings", "source_db.embeddings"
                    )
                    if common_cols:
                        cols_str = ", ".join(common_cols)
                        target_conn.execute(f"""
                            INSERT INTO embeddings ({cols_str})
                            SELECT {cols_str} FROM source_db.embeddings
                        """)

                else:
                    # Modern table (e.g. embeddings_1536)
                    # Ensure target table exists
                    target_conn.execute(
                        f"CREATE TABLE IF NOT EXISTS {et} AS SELECT * FROM source_db.{et} WHERE 1=0"
                    )

                    common_cols = get_common_columns(target_conn, et, f"source_db.{et}")
                    if common_cols:
                        cols_str = ", ".join(common_cols)
                        formatter.verbose_info(f"Copying {et}...")
                        target_conn.execute(f"""
                            INSERT INTO {et} ({cols_str})
                            SELECT {cols_str} FROM source_db.{et}
                        """)

            # Commit transaction
            target_conn.execute("COMMIT")
            formatter.success("Registration + data copy complete (atomic)")

        except Exception as e:
            target_conn.execute("ROLLBACK")
            formatter.error(f"Migration failed, rolled back: {e}")
            try:
                target_conn.execute("DETACH source_db")
            except Exception:
                pass
            target_conn.close()
            sys.exit(1)
        finally:
            try:
                target_conn.execute("DETACH source_db")
            except Exception:
                pass

        # Step 5: Validate
        formatter.info("[4/4] Validating migration...")

        # Simple validation of indexed_roots
        saved_root = target_conn.execute(
            "SELECT * FROM indexed_roots WHERE base_directory = ?",
            [str(base_directory)],
        ).fetchone()

        if not saved_root:
            formatter.error("Migration validation FAILED - root not found!")
            sys.exit(1)

        formatter.success(f"Validation complete - {project_name} registered")

        formatter.section_header("\nMigration successful!")
        formatter.success(f"{file_count} files, {chunk_count} chunks processed")

    except Exception as e:
        formatter.error(f"\nMigration failed: {e}")
        logger.exception("Full error details:")
        sys.exit(1)
    finally:
        # Cleanup connections
        if source_conn is not None:
            try:
                source_conn.close()
            except Exception:
                pass
        if target_conn is not None:
            try:
                target_conn.close()
            except Exception:
                pass

use duckdb::Connection;

use super::DuckDbHnswBackend;
use crate::error::DbError;

impl DuckDbHnswBackend {
    /// Column DDL for the `files` table. Shared by `setup_schema` (initial DB
    /// creation) and `run_attach_copy_compaction` (rebuilding the same table
    /// from scratch during compaction) so the two can't silently drift apart —
    /// previously each hardcoded its own copy of this list.
    pub(super) const FILES_COLUMNS_DDL: &str = "\
        id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),
        path TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        extension TEXT,
        size INTEGER,
        modified_time TIMESTAMP,
        content_hash TEXT,
        language TEXT,
        skip_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP";

    /// Column DDL for the `chunks` table — see `FILES_COLUMNS_DDL`.
    pub(super) const CHUNKS_COLUMNS_DDL: &str = "\
        id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
        file_id INTEGER REFERENCES files(id),
        chunk_type TEXT NOT NULL,
        symbol TEXT,
        code TEXT NOT NULL,
        start_line INTEGER,
        end_line INTEGER,
        start_byte INTEGER,
        end_byte INTEGER,
        language TEXT,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP";

    /// Column DDL for the `schema_version` table — see `FILES_COLUMNS_DDL`.
    pub(super) const SCHEMA_VERSION_COLUMNS_DDL: &str = "\
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        description TEXT";

    /// Column DDL for an `embeddings_{dims}` table. Shared by
    /// `ensure_embedding_table_dims` (initial creation) and
    /// `run_attach_copy_compaction` (rebuilding during compaction) — see
    /// `FILES_COLUMNS_DDL`.
    pub(super) fn embedding_columns_ddl(dims: u32) -> String {
        format!(
            "id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
            chunk_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            embedding FLOAT[{dims}],
            dims INTEGER NOT NULL DEFAULT {dims},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        )
    }

    // SCHEMA PARITY: This DDL must stay in sync with the Python canonical source at
    // chunkhound/providers/database/duckdb/schema_constants.py (_FILES_TABLE_COLUMNS,
    // _CHUNKS_TABLE_COLUMNS, _SCHEMA_VERSION_TABLE_COLUMNS).  The cross-check test
    // tests/contracts/test_schema_parity.py::TestSchemaParity catches column-level drift at CI time.
    // When adding or renaming columns, update schema_constants.py FIRST, then mirror here.
    pub(super) fn setup_schema(conn: &Connection) -> Result<(), DbError> {
        conn.execute_batch(&format!(
            "
            CREATE SEQUENCE IF NOT EXISTS files_id_seq START 1;
            CREATE TABLE IF NOT EXISTS files ({files});
            ALTER TABLE files ADD COLUMN IF NOT EXISTS skip_reason TEXT;
            CREATE SEQUENCE IF NOT EXISTS chunks_id_seq START 1;
            CREATE TABLE IF NOT EXISTS chunks ({chunks});
            CREATE SEQUENCE IF NOT EXISTS embeddings_id_seq START 1;
            CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
            CREATE INDEX IF NOT EXISTS idx_files_language ON files(language);
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol);
            CREATE TABLE IF NOT EXISTS schema_version ({schema_version});
            INSERT INTO schema_version (version, description)
                SELECT 1, 'Initial schema'
                WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 1);
        ",
            files = Self::FILES_COLUMNS_DDL,
            chunks = Self::CHUNKS_COLUMNS_DDL,
            schema_version = Self::SCHEMA_VERSION_COLUMNS_DDL,
        ))?;
        Ok(())
    }

    pub(super) fn try_load_vss(conn: &Connection) -> bool {
        // INSTALL may fail in air-gapped environments or when the extension is already present
        // at a different version — that is non-fatal.  Mirror Python's connection_manager.py
        // which uses separate execute() calls so that LOAD is always attempted regardless of
        // INSTALL's outcome.
        let _ = conn.execute_batch("INSTALL vss");
        if let Err(e) = conn.execute_batch("LOAD vss") {
            log::warn!("VSS extension unavailable (vector search disabled): {e}");
            return false;
        }
        if let Err(e) = conn.execute_batch("SET hnsw_enable_experimental_persistence = true") {
            log::warn!(
                "HNSW persistence unavailable \
                 (hnsw_enable_experimental_persistence not supported by this DuckDB build): {e}"
            );
            return false;
        }
        true
    }

    /// Ensure VSS is loaded, lazily — only on first write with embeddings.
    pub(super) fn ensure_vss(&mut self) -> Result<bool, DbError> {
        if self.has_vss {
            return Ok(true);
        }
        let conn = self.conn_or_err()?;
        let ok = Self::try_load_vss(conn);
        self.has_vss = ok;
        Ok(ok)
    }

    pub(super) fn ensure_embedding_table_dims(conn: &Connection, dims: u32) -> Result<(), DbError> {
        let table = format!("embeddings_{dims}");
        // Legacy compat: the 1536-dimension chunk_id index was originally named
        // idx_embeddings_1536_chunk_id; newer dimensions use idx_{dims}_chunk_id.
        // Mirrors schema_constants._embedding_chunk_id_index_name in Python.
        let chunk_id_idx = if dims == 1536 {
            format!("idx_embeddings_{dims}_chunk_id")
        } else {
            format!("idx_{dims}_chunk_id")
        };
        let cols = Self::embedding_columns_ddl(dims);
        conn.execute_batch(&format!(
            "
            CREATE TABLE IF NOT EXISTS \"{table}\" ({cols});
            CREATE INDEX IF NOT EXISTS {chunk_id_idx}
            ON \"{table}\" (chunk_id);
            CREATE INDEX IF NOT EXISTS idx_{dims}_provider_model
            ON \"{table}\" (provider, model);
        "
        ))?;
        // Created separately, best-effort: unlike the two plain indexes above,
        // this can legitimately fail with a constraint violation if the table
        // already has duplicate (chunk_id, provider, model) rows — e.g. a
        // table rebuilt by compaction whose data predates this index ever
        // being enforced. Python's post-reconnect
        // _executor_ensure_embedding_upsert_contract dedupes and creates it
        // as a fallback when this doesn't succeed.
        if let Err(e) = conn.execute_batch(&format!(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_{dims}_chunk_provider_model_unique
             ON \"{table}\" (chunk_id, provider, model)"
        )) {
            log::warn!(
                "Could not create unique upsert-contract index on {table} \
                 (likely duplicate chunk_id/provider/model rows); Python's \
                 reconnect will repair this: {e}"
            );
        }
        Ok(())
    }

    pub(super) fn discover_embedding_tables(
        conn: &Connection,
    ) -> Result<Vec<(String, u32)>, DbError> {
        let mut stmt = conn.prepare(
            "SELECT table_name FROM information_schema.tables
             WHERE table_schema = 'main'
             AND table_name SIMILAR TO 'embeddings_[0-9]+'",
        )?;
        let names: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();
        let result = names
            .into_iter()
            .filter_map(|name| {
                let dims: u32 = name.strip_prefix("embeddings_")?.parse().ok()?;
                Some((name, dims))
            })
            .collect();
        Ok(result)
    }
}

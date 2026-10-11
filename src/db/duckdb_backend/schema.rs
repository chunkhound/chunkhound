use std::path::{Path, PathBuf};

use duckdb::core::{LogicalTypeHandle, LogicalTypeId};
use duckdb::{Config, Connection};

use super::DuckDbHnswBackend;
use crate::error::DbError;

// Kept equal to chunkhound/core/index_format.py by native/Python contract tests.
const INDEX_FORMAT_VERSION: i64 = 1;
const LEGACY_INDEX_FORMAT_VERSION: i64 = 1;

const INSPECTION_FAILED: &str = "Cannot safely inspect index; format compatibility is unknown, \
    so automatic recovery is blocked. Some healthy WALs require writes to replay and cannot \
    be inspected read-only. Preserve the database and any WAL/recovery files. Restore a \
    known-good complete index or recover a copy using compatible DuckDB/VSS tooling, then \
    retry. Reopening writable does not bypass this check.";
const WAL_VSS_UNAVAILABLE: &str = "Cannot safely inspect index: WAL is present but local DuckDB \
    VSS support could not be loaded or configured; format compatibility is unknown, so \
    automatic recovery is blocked. Preserve the database and any WAL/recovery files. Make \
    a compatible VSS extension available using an explicit setup operation, then retry. \
    Reopening writable does not bypass this check.";

impl DuckDbHnswBackend {
    // Python decodes UNION to its active member, including nested UNIONs.
    // Use logical metadata rather than Rust Value: its Arrow representation
    // conflates scale-zero DECIMAL with HUGEINT, and cannot preserve UHUGEINT.
    fn decoded_type_sql(kind: &LogicalTypeHandle, expression: &str) -> String {
        if kind.id() != LogicalTypeId::Union {
            return format!("typeof({expression})");
        }
        let mut sql = format!("CASE union_tag({expression})");
        for index in 0..kind.num_children() {
            let member = kind.child_name(index).replace('\'', "''");
            let extracted = format!("union_extract({expression}, '{member}')");
            let member_type = Self::decoded_type_sql(&kind.child(index), &extracted);
            sql.push_str(&format!(" WHEN '{member}' THEN {member_type}"));
        }
        sql.push_str(" ELSE NULL END");
        sql
    }

    pub(super) fn check_index_format(conn: &Connection) -> Result<(), DbError> {
        Self::check_catalog_index_format(conn, false)
    }

    fn check_catalog_index_format(conn: &Connection, inspected: bool) -> Result<(), DbError> {
        // These are constant identifiers, never supplied by a path or caller.
        let (catalog, table) = if inspected {
            ("'inspected'", "inspected.main.schema_version")
        } else {
            ("current_database()", "main.schema_version")
        };
        let inspect = || -> Result<(), DbError> {
            let present: i64 = conn.query_row(
                &format!(
                    "SELECT count(*) FROM information_schema.tables \
                    WHERE table_catalog={catalog} AND table_schema='main' \
                    AND table_name='schema_version'"
                ),
                [],
                |row| row.get(0),
            )?;
            let mut version = LEGACY_INDEX_FORMAT_VERSION as u128;
            if present != 0 {
                // Preserve integer type semantics: DuckDB/Rust numeric coercions
                // would otherwise accept booleans, decimals or numeric strings.
                // Text transport also avoids narrowing HUGEINT/UHUGEINT values.
                let metadata = conn.prepare(&format!("SELECT version FROM {table}"))?;
                let kind = Self::decoded_type_sql(&metadata.column_logical_type(0), "version");
                let mut stmt = conn.prepare(&format!(
                    "SELECT {kind}, CAST(version AS VARCHAR) FROM {table}"
                ))?;
                let mut rows = stmt.query([])?;
                while let Some(row) = rows.next()? {
                    let kind: Option<String> = row.get(0)?;
                    let stored: Option<String> = row.get(1)?;
                    let value = stored.and_then(|value| value.parse::<u128>().ok());
                    if !matches!(
                        kind.as_deref(),
                        Some(
                            "TINYINT"
                                | "SMALLINT"
                                | "INTEGER"
                                | "BIGINT"
                                | "HUGEINT"
                                | "UTINYINT"
                                | "USMALLINT"
                                | "UINTEGER"
                                | "UBIGINT"
                                | "UHUGEINT"
                        )
                    ) || value.is_none_or(|value| value < 1)
                    {
                        return Err(DbError::Other("Invalid stored index format".into()));
                    }
                    version = version.max(value.expect("validated positive integer"));
                }
            }
            if version != INDEX_FORMAT_VERSION as u128 {
                return Err(DbError::Other(
                    "Unsupported index format; upgrade ChunkHound or rebuild".into(),
                ));
            }
            Ok(())
        };
        inspect().map_err(|error| match error {
            DbError::DuckDb(_) => DbError::Other(INSPECTION_FAILED.into()),
            other => other,
        })
    }

    /// Inspect every recovery candidate before any rename, removal or writable
    /// replay. This is an inspection checkpoint, not lifetime/race protection.
    pub(super) fn preflight_index_formats(live: &Path) -> Result<(), DbError> {
        if live == Path::new(":memory:") {
            return Ok(());
        }
        for suffix in ["", ".compact_backup", ".compact_new", ".old", ".compact"] {
            let mut name = live.as_os_str().to_os_string();
            name.push(suffix);
            let path = PathBuf::from(name);
            match path.metadata() {
                Ok(meta) if meta.is_file() && meta.len() > 0 => {}
                Ok(_) => continue,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(_) => return Err(DbError::Other(INSPECTION_FAILED.into())),
            }
            let inspect = || -> Result<(), DbError> {
                let flags = Config::default()
                    .with("autoinstall_known_extensions", "false")?
                    .with("autoload_known_extensions", "false")?
                    .with("temp_directory", "")?;
                let conn = Connection::open_in_memory_with_flags(flags)?;
                let mut wal = path.as_os_str().to_os_string();
                wal.push(".wal");
                if PathBuf::from(wal)
                    .try_exists()
                    .map_err(|_| DbError::Other(INSPECTION_FAILED.into()))?
                {
                    // Never call try_load_vss here: it INSTALLs and swallows
                    // failures. Arbitrary WAL must not attach after failed LOAD.
                    conn.execute_batch("LOAD vss; SET hnsw_enable_experimental_persistence=true")
                        .map_err(|_| DbError::Other(WAL_VSS_UNAVAILABLE.into()))?;
                }
                let escaped = path
                    .to_str()
                    .ok_or_else(|| DbError::Other(INSPECTION_FAILED.into()))?
                    .replace('\'', "''");
                conn.execute_batch(&format!("ATTACH '{escaped}' AS inspected (READ_ONLY)"))?;
                Self::check_catalog_index_format(&conn, true)
            };
            inspect().map_err(|error| match error {
                DbError::DuckDb(_) => DbError::Other(INSPECTION_FAILED.into()),
                other => other,
            })?;
        }
        Ok(())
    }

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
        Self::check_index_format(conn)?;
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
                SELECT 1, 'ChunkHound index format'
                WHERE NOT EXISTS (SELECT 1 FROM schema_version);
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

#[cfg(test)]
mod format_tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::SystemTime;

    use duckdb::Config;

    use super::*;
    use crate::db::duckdb_backend::test_support::config;
    use crate::db::DbBackend;

    fn sidecar(path: &Path, suffix: &str) -> PathBuf {
        let mut name = path.as_os_str().to_os_string();
        name.push(suffix);
        PathBuf::from(name)
    }

    fn snapshot(root: &Path) -> BTreeMap<PathBuf, (Vec<u8>, SystemTime)> {
        let mut result = BTreeMap::new();
        for entry in std::fs::read_dir(root).expect("snapshot directory") {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                result.extend(snapshot(&path));
            } else {
                result.insert(
                    path.clone(),
                    (
                        std::fs::read(&path).expect("snapshot bytes"),
                        path.metadata()
                            .expect("metadata")
                            .modified()
                            .expect("mtime"),
                    ),
                );
            }
        }
        result
    }

    fn fixture_connection(path: &Path) -> Connection {
        let flags = Config::default()
            .with("autoinstall_known_extensions", "false")
            .expect("no install")
            .with("autoload_known_extensions", "false")
            .expect("no autoload")
            .with("temp_directory", "")
            .expect("no spill");
        Connection::open_with_flags(path, flags).expect("fixture connection")
    }

    fn checkpointed(path: &Path, version: i64) {
        let conn = fixture_connection(path);
        conn.execute_batch("CREATE TABLE schema_version(version BIGINT, description VARCHAR)")
            .expect("metadata table");
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", [version])
            .expect("metadata row");
    }

    fn assert_refused_unchanged(root: &Path, live: &Path, expected: &str) {
        let before = snapshot(root);
        let mut backend = DuckDbHnswBackend::new(config(live.to_string_lossy().into_owned()));
        let message = backend
            .open()
            .expect_err("format must block open")
            .to_string();
        assert!(message.contains(expected), "{message}");
        assert!(
            !message.contains(&live.to_string_lossy().to_string()),
            "unsanitized: {message}"
        );
        assert_eq!(snapshot(root), before, "preflight changed original files");
    }

    #[test]
    fn rejects_future_candidates_before_recovery_changes_files() {
        for suffix in ["", ".compact_backup", ".compact_new", ".old", ".compact"] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let live = tmp.path().join("quoted' chunks.db");
            if !suffix.is_empty() {
                checkpointed(&live, 1);
            }
            checkpointed(&sidecar(&live, suffix), 2);
            std::fs::write(sidecar(&live, ".swap_intent"), "phase1").expect("intent");
            assert_refused_unchanged(tmp.path(), &live, "Unsupported index format");
        }
    }

    #[test]
    fn file_state_read_rejects_future_candidate_before_recovery() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let live = tmp.path().join("chunks.db");
        {
            let conn = fixture_connection(&live);
            DuckDbHnswBackend::setup_schema(&conn).expect("live schema");
        }
        checkpointed(&sidecar(&live, ".old"), 2);
        std::fs::write(sidecar(&live, ".swap_intent"), "phase2").expect("intent");
        let before = snapshot(tmp.path());
        let backend = DuckDbHnswBackend::new(config(live.to_string_lossy().into_owned()));
        let result = backend.read_file_states();
        assert!(
            snapshot(tmp.path()) == before,
            "diff read recovered before format preflight"
        );
        assert!(result.is_err(), "diff read accepted future candidate");
    }

    #[test]
    fn setup_schema_rejects_bad_metadata_before_ddl() {
        for metadata in [
            "SELECT 2 AS version",
            "SELECT 0 AS version",
            "SELECT -1 AS version",
            "SELECT NULL::INTEGER AS version",
            "SELECT '1' AS version",
            "SELECT true AS version",
            "SELECT 1.0 AS version",
            "SELECT 1.5 AS version",
            "SELECT 1 AS missing_version",
            "SELECT unnest([1, 2]) AS version",
        ] {
            let conn = Connection::open_in_memory().expect("memory");
            conn.execute_batch(&format!(
                "CREATE TABLE schema_version AS SELECT *, 'retained' AS description FROM ({metadata})"
            )).expect("metadata");
            assert!(
                DuckDbHnswBackend::setup_schema(&conn).is_err(),
                "accepted {metadata}"
            );
            let tables: i64 = conn
                .query_row(
                    "SELECT count(*) FROM information_schema.tables",
                    [],
                    |row| row.get(0),
                )
                .expect("tables");
            assert_eq!(tables, 1, "DDL ran before rejection: {metadata}");
        }
    }

    // Producer and missing-extension checks run in an isolated native process:
    // no Python DuckDB WAL/version mismatch and no process-global HOME races.
    fn native_process(mode: &str, path: &Path, kind: &str, future: bool, home: Option<&Path>) {
        let mut command = Command::new(std::env::current_exe().expect("test executable"));
        command
            .args([
                "--exact",
                "db::duckdb_backend::schema::format_tests::native_fixture_process",
                "--nocapture",
            ])
            .env("CH_FORMAT_MODE", mode)
            .env("CH_FORMAT_PATH", path)
            .env("CH_FORMAT_KIND", kind)
            .env("CH_FORMAT_FUTURE", if future { "1" } else { "0" });
        if let Some(home) = home {
            command.env("HOME", home);
        }
        let output = command.output().expect("native fixture subprocess");
        assert!(
            output.status.success(),
            "native subprocess failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn native_fixture_process() {
        let Ok(mode) = std::env::var("CH_FORMAT_MODE") else {
            return;
        };
        let path = PathBuf::from(std::env::var_os("CH_FORMAT_PATH").expect("path"));
        let kind = std::env::var("CH_FORMAT_KIND").expect("kind");
        if mode == "missing_vss" {
            let before = snapshot(path.parent().expect("parent"));
            let conn = fixture_connection(Path::new(":memory:"));
            assert!(
                conn.execute_batch("LOAD vss").is_err(),
                "VSS unexpectedly available"
            );
            if kind.ends_with("checkpoint") {
                DuckDbHnswBackend::preflight_index_formats(&path).expect("checkpoint needs no VSS");
            } else {
                let message = DuckDbHnswBackend::preflight_index_formats(&path)
                    .expect_err("WAL requires local VSS")
                    .to_string();
                assert!(
                    message.contains("WAL") && message.contains("VSS"),
                    "{message}"
                );
                assert!(message.contains("explicit setup operation"), "{message}");
                assert!(
                    message.contains("format compatibility is unknown"),
                    "{message}"
                );
                assert!(
                    !message.contains(&path.to_string_lossy().to_string()),
                    "{message}"
                );
            }
            assert_eq!(snapshot(path.parent().expect("parent")), before);
            assert!(
                !PathBuf::from(std::env::var_os("HOME").expect("isolated home"))
                    .join(".duckdb")
                    .exists()
            );
            return;
        }
        assert_eq!(mode, "produce");
        let conn = fixture_connection(&path);
        conn.execute_batch(
            "SET wal_autocheckpoint='1GB';
            CREATE TABLE schema_version(version INTEGER);
            INSERT INTO schema_version VALUES (1);
            CREATE TABLE ordinary(id INTEGER)",
        )
        .expect("base");
        if kind.starts_with("hnsw") {
            conn.execute_batch(
                "LOAD vss; SET hnsw_enable_experimental_persistence=true;
                CREATE TABLE vectors(id INTEGER, vec FLOAT[3]);
                INSERT INTO vectors VALUES (0, [0, 0, 0])",
            )
            .expect("local VSS fixture");
            if kind != "hnsw_create" {
                conn.execute_batch("CREATE INDEX hnsw_idx ON vectors USING HNSW(vec)")
                    .expect("checkpointed HNSW");
            }
        }
        conn.execute_batch("CHECKPOINT").expect("checkpoint");
        if kind.ends_with("checkpoint") {
            return;
        }
        conn.execute_batch("INSERT INTO ordinary VALUES (1)")
            .expect("WAL insert");
        if kind.starts_with("hnsw") {
            if kind == "hnsw_create" {
                conn.execute_batch("CREATE INDEX hnsw_idx ON vectors USING HNSW(vec)")
                    .expect("HNSW creation WAL");
            }
            conn.execute_batch("INSERT INTO vectors VALUES (1, [1, 1, 1])")
                .expect("vector WAL");
        }
        if std::env::var("CH_FORMAT_FUTURE").expect("future") == "1" {
            conn.execute_batch("INSERT INTO schema_version VALUES (2)")
                .expect("future WAL");
        }
        // Deliberately bypass DuckDB drop/checkpoint to retain committed native WAL.
        std::process::exit(0);
    }

    #[test]
    fn supported_wal_inspection_preserves_originals() {
        for kind in ["ordinary", "hnsw_insert"] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let live = tmp.path().join("quoted' chunks.db");
            native_process("produce", &live, kind, false, None);
            let before = snapshot(tmp.path());
            DuckDbHnswBackend::preflight_index_formats(&live).expect("inspect supported WAL");
            assert_eq!(snapshot(tmp.path()), before);
        }
    }

    #[test]
    fn missing_local_vss_blocks_only_wal_before_attachment() {
        for kind in [
            "ordinary_checkpoint",
            "hnsw_checkpoint",
            "ordinary",
            "unreadable",
        ] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let live = tmp.path().join("chunks.db");
            native_process("produce", &live, kind, false, None);
            if kind == "unreadable" {
                // With WAL present, LOAD must fail before even parsing this DB.
                std::fs::write(&live, b"unreadable database with a WAL").expect("unreadable DB");
            }
            let home = tmp.path().join("isolated-home");
            std::fs::create_dir(&home).expect("isolated home");
            let before = snapshot(tmp.path());
            native_process("missing_vss", &live, kind, false, Some(&home));
            assert_eq!(snapshot(tmp.path()), before);
        }
    }

    #[test]
    fn healthy_hnsw_creation_wal_is_refused_without_mutating_originals() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let originals = tmp.path().join("originals");
        std::fs::create_dir(&originals).expect("original directory");
        let live = originals.join("chunks.db");
        checkpointed(&live, 1);
        let candidate = sidecar(&live, ".compact");
        native_process("produce", &candidate, "hnsw_create", true, None);
        std::fs::write(sidecar(&live, ".swap_intent"), "phase2").expect("intent");
        let before = snapshot(&originals);
        // Disposable copies are ONLY a fixture-health proof, never production recovery.
        let copy = tmp.path().join("disposable.db");
        std::fs::copy(&candidate, &copy).expect("DB copy");
        std::fs::copy(sidecar(&candidate, ".wal"), sidecar(&copy, ".wal")).expect("WAL copy");
        {
            let conn = fixture_connection(Path::new(":memory:"));
            conn.execute_batch("LOAD vss; SET hnsw_enable_experimental_persistence=true")
                .expect("local VSS");
            let escaped = copy.to_str().expect("path").replace('\'', "''");
            conn.execute_batch(&format!("ATTACH '{escaped}' AS healthy"))
                .expect("healthy copy replay");
            let version: i64 = conn
                .query_row(
                    "SELECT MAX(version) FROM healthy.main.schema_version",
                    [],
                    |row| row.get(0),
                )
                .expect("copy version");
            let vectors: i64 = conn
                .query_row("SELECT count(*) FROM healthy.main.vectors", [], |row| {
                    row.get(0)
                })
                .expect("copy vectors");
            let indexes: i64 = conn.query_row("SELECT count(*) FROM duckdb_indexes() WHERE database_name='healthy' AND index_name='hnsw_idx'",
                [], |row| row.get(0)).expect("copy HNSW");
            assert_eq!((version, vectors, indexes), (2, 2, 1));
        }
        assert_refused_unchanged(&originals, &live, "format compatibility is unknown");
        let error = DuckDbHnswBackend::preflight_index_formats(&live)
            .expect_err("read-only refusal")
            .to_string();
        assert!(
            error.contains("Preserve the database") && error.contains("recover a copy"),
            "{error}"
        );
        assert!(
            !error.contains("corrupt") && !error.contains("Unsupported index format"),
            "{error}"
        );
        assert_eq!(snapshot(&originals), before);
    }

    #[test]
    fn unreadable_candidate_is_not_treated_as_legacy() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let live = tmp.path().join("chunks.db");
        checkpointed(&live, 1);
        std::fs::write(sidecar(&live, ".compact"), b"incomplete compaction")
            .expect("bad candidate");
        std::fs::write(sidecar(&live, ".swap_intent"), "phase1").expect("intent");
        assert_refused_unchanged(tmp.path(), &live, "format compatibility is unknown");
    }

    #[test]
    fn future_candidate_wal_blocks_recovery() {
        for kind in ["ordinary", "hnsw_insert"] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let live = tmp.path().join("chunks.db");
            checkpointed(&live, 1);
            let candidate = sidecar(&live, ".compact_backup");
            native_process("produce", &candidate, kind, true, None);
            assert!(sidecar(&candidate, ".wal").metadata().expect("WAL").len() > 0);
            std::fs::write(sidecar(&live, ".swap_intent"), "phase1").expect("intent");
            assert_refused_unchanged(tmp.path(), &live, "Unsupported index format");
        }
    }
}

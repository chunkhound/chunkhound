use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub chunk_type: String,
    pub symbol: Option<String>,
    pub code: String,
    pub start_line: Option<i64>,
    pub end_line: Option<i64>,
    pub start_byte: Option<i64>,
    pub end_byte: Option<i64>,
    pub language: Option<String>,
    pub metadata: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub provider: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub existing_file_id: Option<i64>,
    pub path: String,
    pub mtime: Option<f64>,
    pub size_bytes: Option<i64>,
    pub content_hash: Option<String>,
    pub language: Option<String>,
    pub chunks: Vec<ChunkRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbWriterBatch {
    pub files: Vec<FileRecord>,
    pub delete_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub file_ids: Vec<i64>,
    pub chunks_written: u64,
    pub embeddings_written: u64,
}

/// Snapshot of a single `files` row, as read from the DB by
/// `DbBackend::read_file_states` (implemented by `DuckDbHnswBackend`) and
/// consumed by the pipeline's diff phase (`pipeline::differ::compute_diff`).
/// Shared between the two modules so the `files` table's column shape has
/// exactly one reader.
#[derive(Debug, Clone)]
pub struct DbFileEntry {
    pub id: i64,
    pub path: String,
    pub mtime: f64, // Unix timestamp
    pub content_hash: Option<String>,
}

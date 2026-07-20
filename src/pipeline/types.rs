//! Pipeline internal types — never exposed to Python directly.
//!
//! These types flow through the parse → embed → store pipeline inside Rust.

use std::path::PathBuf;

/// A single file after being parsed by the Python callback.
#[derive(Debug, Clone)]
pub(crate) struct ParsedFile {
    pub path: PathBuf,
    pub language: Option<String>,
    pub file_size: u64,
    pub mtime: f64,
    pub content_hash: String,
    pub chunks: Vec<NewChunk>,
    pub error: Option<String>,
}

/// A chunk from the Python parse callback, before embedding.
#[derive(Debug, Clone)]
pub(crate) struct NewChunk {
    pub chunk_type: String,
    pub symbol: Option<String>,
    pub code: String,
    pub start_line: Option<i64>,
    pub end_line: Option<i64>,
    pub start_byte: Option<i64>,
    pub end_byte: Option<i64>,
    pub language: Option<String>,
    pub metadata: Option<String>,
    pub embed_text: Option<String>,

    // Filled by the embed thread in Phase 2+
    pub embedding: Option<Vec<f32>>,
    pub provider: Option<String>,
    pub model: Option<String>,
}

/// Produced by the parse thread, consumed by the embed thread.
#[derive(Debug, Clone)]
pub(crate) struct DiffResult {
    pub file_id: i64,
    pub path: String,
    pub name: String,
    pub extension: Option<String>,
    pub language: Option<String>,
    pub file_size: u64,
    pub mtime: f64,
    pub content_hash: String,
    pub is_new_file: bool,
    pub chunks_to_insert: Vec<NewChunk>,
}

/// Produced by the embed thread, consumed by the store thread.
#[derive(Debug, Clone)]
pub(crate) struct EmbeddedBatch {
    pub diffs: Vec<DiffResult>,
}

/// A single row from the `files` table, loaded once at startup.
#[derive(Debug, Clone)]
pub(crate) struct DbFileState {
    pub file_id: i64,
    pub size_bytes: Option<i64>,
    pub mtime: Option<f64>,
    pub content_hash: Option<String>,
}

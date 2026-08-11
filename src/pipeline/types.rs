//! Pipeline internal types — never exposed to Python directly.
//!
//! These types flow through the parse → embed → store pipeline inside Rust.

use std::path::PathBuf;

/// A single file after being parsed by the Python callback.
#[derive(Debug, Clone)]
pub(crate) struct ParsedFile {
    pub path: PathBuf,
    /// Project-relative DB key, computed by Python's `get_relative_path_safe()`
    /// and supplied via `IndexingPipeline::run()`'s `rel_keys` map — the
    /// single symlink-aware source of truth. Consumed verbatim by
    /// `build_db_batch` instead of being re-derived from `path`.
    pub rel_path: String,
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

    // Filled by the embed thread
    pub embedding: Option<Vec<f32>>,
    pub provider: Option<String>,
    pub model: Option<String>,
}

//! Pipeline state — internal counters and error accumulator.

/// Tracks progress and errors during a pipeline run.
#[derive(Debug, Clone)]
pub(crate) struct PipelineState {
    pub files_processed: u64,
    pub files_skipped: u64,
    pub chunks_written: u64,
    pub embeddings_generated: u64,
    pub errors: Vec<String>,
    pub total_files: u64,
}

impl PipelineState {
    pub fn new(total_files: u64) -> Self {
        Self {
            files_processed: 0,
            files_skipped: 0,
            chunks_written: 0,
            embeddings_generated: 0,
            errors: Vec::new(),
            total_files,
        }
    }

    pub fn record_parsed(&mut self, count: u64) {
        self.files_processed += count;
    }

    pub fn record_skipped(&mut self, reason: &str, file_count: u64) {
        self.files_skipped += file_count;
        self.errors.push(reason.to_string());
    }

    pub fn into_report(self, elapsed_secs: f64) -> super::report::PipelineReport {
        super::report::PipelineReport {
            files_processed: self.files_processed,
            files_skipped: self.files_skipped,
            chunks_written: self.chunks_written,
            embeddings_generated: self.embeddings_generated,
            elapsed_secs,
            errors: self.errors,
            peak_rss_mb: None,
        }
    }
}

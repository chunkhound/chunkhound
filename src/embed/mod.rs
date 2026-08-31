//! Embedding adapters used by the native indexing pipeline.

mod callback;
mod factory;
mod openai;
mod retry;
mod token;
mod voyageai;

pub(crate) use callback::PythonEmbedCallback;
pub(crate) use factory::{create_embed_fn, EmbedConfig};
pub(crate) use openai::OpenAiProvider;
pub(crate) use voyageai::VoyageAiProvider;

#[derive(Debug, Default)]
pub(crate) struct EmbedBatchResult {
    pub vectors: Vec<Option<Vec<f32>>>,
    pub errors: Vec<String>,
    pub stats: BatchCallStats,
}

impl EmbedBatchResult {
    pub fn empty(len: usize) -> Self {
        Self {
            vectors: vec![None; len],
            errors: Vec::new(),
            stats: BatchCallStats::default(),
        }
    }
}

pub(crate) trait EmbedBatchFn: Send + Sync {
    /// Return exactly one slot, in input order, for each supplied text.
    fn embed_batch(&self, texts: &[String]) -> Result<EmbedBatchResult, String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct BatchCallStats {
    pub requests: usize,
    pub retries: usize,
}

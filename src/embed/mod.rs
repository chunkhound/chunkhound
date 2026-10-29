//! Embedding adapters used by the native indexing pipeline.

mod callback;
mod common;
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
    fn embed_batch_cancellable(
        &self,
        texts: &[String],
        stop: &crate::cancellation::StopToken,
    ) -> Result<EmbedBatchResult, crate::error::PipelineError> {
        stop.check()?;
        let result = self
            .embed_batch(texts)
            .map_err(crate::error::PipelineError::ProviderError);
        stop.check()?;
        result
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct BatchCallStats {
    pub requests: usize,
    pub retries: usize,
}

use super::{EmbedBatchFn, EmbedBatchResult};
use crate::error::PipelineError;
use pyo3::prelude::*;

const MAX_ERROR_LEN: usize = 500;

pub(crate) struct PythonEmbedCallback {
    callback: Py<PyAny>,
}

impl PythonEmbedCallback {
    pub fn new(callback: Py<PyAny>) -> Self {
        Self { callback }
    }
}

impl EmbedBatchFn for PythonEmbedCallback {
    fn embed_batch(&self, texts: &[String]) -> Result<EmbedBatchResult, String> {
        Python::with_gil(|py| {
            let result = self
                .callback
                .bind(py)
                .call1((texts.to_vec(),))
                .map_err(|error| classify_python_embed_error(error.to_string()))?;
            extract_vectors_from_python(result, texts.len())
        })
    }
}

pub(crate) fn extract_vectors_from_python(
    value: Bound<'_, PyAny>,
    expected: usize,
) -> Result<EmbedBatchResult, String> {
    let vectors: Vec<Vec<f64>> = value
        .extract()
        .map_err(|error| classify_python_embed_error(error.to_string()))?;
    let mut result = EmbedBatchResult::empty(expected);
    for (index, vector) in vectors.into_iter().take(expected).enumerate() {
        if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
            return Err(PipelineError::ResponseFormat(
                "Python callback returned an invalid vector".to_string(),
            )
            .to_string());
        }
        result.vectors[index] = Some(vector.into_iter().map(|value| value as f32).collect());
    }
    if result.vectors.iter().any(Option::is_none) {
        result.errors.push(format!(
            "provider returned {} vector(s) for {} requested",
            result
                .vectors
                .iter()
                .filter(|vector| vector.is_some())
                .count(),
            expected
        ));
    }
    Ok(result)
}

pub(crate) fn classify_python_embed_error(error: String) -> String {
    let sanitized = error
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    let bounded = sanitized.trim();
    if bounded.len() <= MAX_ERROR_LEN {
        bounded.to_string()
    } else {
        format!(
            "{}...",
            bounded.chars().take(MAX_ERROR_LEN).collect::<String>()
        )
    }
}

//! Small, provider-independent token-aware batching utilities.

/// The approximation used by the Python providers and the shared token
/// utility. Keeping this at three characters per token is important for
/// parity with the callback path.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.len().div_ceil(3).max(1)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    pub max_tokens: usize,
    pub max_items: usize,
}

#[derive(Debug, Clone)]
pub struct BatchChunk {
    pub indices: Vec<usize>,
    pub texts: Vec<String>,
    pub tokens: usize,
}

pub struct BatchBuilder {
    config: BatchConfig,
    indices: Vec<usize>,
    texts: Vec<String>,
    tokens: usize,
}

impl BatchBuilder {
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config: BatchConfig {
                max_tokens: config.max_tokens.max(1),
                max_items: config.max_items.max(1),
            },
            indices: Vec::new(),
            texts: Vec::new(),
            tokens: 0,
        }
    }

    pub fn push(&mut self, index: usize, text: String) -> Option<BatchChunk> {
        let tokens = estimate_tokens(&text);
        let full = !self.texts.is_empty()
            && (self.texts.len() >= self.config.max_items
                || self.tokens + tokens > self.config.max_tokens);
        if full {
            let batch = self.take();
            self.indices.push(index);
            self.texts.push(text);
            self.tokens = tokens;
            Some(batch)
        } else {
            self.indices.push(index);
            self.texts.push(text);
            self.tokens += tokens;
            None
        }
    }

    pub fn finish(&mut self) -> Option<BatchChunk> {
        if self.texts.is_empty() {
            None
        } else {
            Some(self.take())
        }
    }

    fn take(&mut self) -> BatchChunk {
        BatchChunk {
            indices: std::mem::take(&mut self.indices),
            texts: std::mem::take(&mut self.texts),
            tokens: std::mem::take(&mut self.tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_tokens_using_three_char_ratio() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 1);
        assert_eq!(estimate_tokens("abcd"), 2);
    }

    #[test]
    fn splits_by_tokens_and_item_count() {
        let mut builder = BatchBuilder::new(BatchConfig {
            max_tokens: 1,
            max_items: 10,
        });
        assert!(builder.push(0, "abc".into()).is_none());
        let first = builder.push(1, "abc".into()).expect("first batch");
        assert_eq!(first.indices, vec![0]);
        assert_eq!(builder.finish().expect("second batch").indices, vec![1]);
    }
}

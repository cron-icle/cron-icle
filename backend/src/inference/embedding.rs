//! Embedding-provider and vector-index contracts.
//!
//! Nomic generates vectors; sqlite-vec stores/searches them. Keeping these
//! responsibilities separate allows CPU fallback and model replacement.

pub trait TextEmbedder: Send + Sync {
    fn dimensions(&self) -> usize;
    fn embed(&self, input: &str) -> Result<Vec<f32>, String>;
}
pub trait VectorIndex: Send + Sync {
    fn insert(&self, semantic_event_id: &str, embedding: &[f32]) -> Result<(), String>;
    fn search(&self, embedding: &[f32], limit: usize) -> Result<Vec<(String, f32)>, String>;
}

pub fn validate_embedding(embedding: &[f32], expected_dimensions: usize) -> Result<(), String> {
    if embedding.len() != expected_dimensions {
        return Err(format!(
            "embedding dimension {} does not match expected {}",
            embedding.len(),
            expected_dimensions
        ));
    }
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err("embedding contains a non-finite value".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/embedding_provider_tests.rs"]
mod tests;

//! Local LLM inference: model download/setup, the bundled llama.cpp engine
//! lifecycle, and the embedding provider used for semantic search.

pub mod embedding;
pub mod model_provider;
pub mod native;
pub mod setup;

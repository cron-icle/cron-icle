//! Local semantic-processing adapter contracts.
//!
//! Model runtimes are optional and machine-specific. This module validates the
//! stable JSON shape before it reaches `semantic_events`, keeping capture and
//! persistence independent from Gemma or any future local model.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticModelOutput {
    pub category: String,
    pub summary: String,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub relationships: Vec<String>,
    pub confidence: f32,
}

pub trait LocalSemanticAnalyzer: Send + Sync {
    fn analyze_text(&self, input: &str) -> Result<SemanticModelOutput, String>;
    fn analyze_image(&self, bytes: &[u8]) -> Result<SemanticModelOutput, String>;
}

pub fn parse_and_validate_model_json(json: &str) -> Result<SemanticModelOutput, String> {
    if json.len() > 64 * 1024 {
        return Err("semantic model output exceeds 64 KiB".into());
    }
    let output: SemanticModelOutput = serde_json::from_str(json)
        .map_err(|error| format!("invalid semantic model JSON: {error}"))?;
    validate_model_output(output)
}

pub fn validate_image_input(bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() {
        return Err("image input is empty".into());
    }
    if bytes.len() > 10 * 1024 * 1024 {
        return Err("image input exceeds 10 MiB".into());
    }
    let png = bytes.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]);
    let jpeg = bytes.starts_with(&[255, 216, 255]);
    if !png && !jpeg {
        return Err("image input must be PNG or JPEG".into());
    }
    Ok(())
}

pub fn validate_model_output(output: SemanticModelOutput) -> Result<SemanticModelOutput, String> {
    if output.category.trim().is_empty() {
        return Err("semantic category is empty".into());
    }
    if output.summary.trim().is_empty() {
        return Err("semantic summary is empty".into());
    }
    if !(0.0..=1.0).contains(&output.confidence) {
        return Err("semantic confidence must be between 0 and 1".into());
    }
    Ok(output)
}

#[cfg(test)]
#[path = "../tests/local_semantic_processing_tests.rs"]
mod tests;

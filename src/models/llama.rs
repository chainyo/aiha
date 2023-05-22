//! Module for the Llama model
use super::base::{ ModelConfig, ModelLibraries };

/// A struct representing a Llama model configuration
pub struct LlamaModelConfig {
    /// Llama model hidden_size
    hidden_size: i32,
    /// Llama model intermediate_size
    intermediate_size: i32,
    /// Llama model max_position_embeddings
    max_sequence_length: i32,
    /// Llama model num_attention_heads
    num_attention_heads: i32,
    /// Llama model num_hidden_layers
    num_hidden_layers: i32,
    /// Llama model Hugging Face repository name
    repo_name: String,
    /// Llama model type
    model_type: String,
    /// Llama model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// Llama model implementation
impl LlamaModelConfig {
    /// Build a new `LlamaModelConfig` struct based on the provided parameters
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        max_sequence_length: i32,
        num_attention_heads: i32,
        num_hidden_layers: i32,
        repo_name: String,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> LlamaModelConfig {
        LlamaModelConfig {
            hidden_size,
            intermediate_size,
            max_sequence_length,
            num_attention_heads,
            num_hidden_layers,
            repo_name,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `LlamaModelConfig`
impl ModelConfig for LlamaModelConfig {
    fn hidden_size(&self) -> &i32 {
        &self.hidden_size
    }

    fn intermediate_size(&self) -> &i32 {
        &self.intermediate_size
    }

    fn max_position_embeddings(&self) -> &i32 {
        &self.max_sequence_length
    }

    fn num_attention_heads(&self) -> &i32 {
        &self.num_attention_heads
    }

    fn num_hidden_layers(&self) -> &i32 {
        &self.num_hidden_layers
    }

    fn repo_name(&self) -> &str {
        &self.repo_name
    }

    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn available_libraries(&self) -> &[ModelLibraries] {
        &self.available_libraries
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_new() {
        let llama_model_config = LlamaModelConfig::new(
            1024,
            4096,
            1024,
            16,
            24,
            "meta/llama".to_string(),
            "llama".to_string(),
            vec![ModelLibraries::PyTorch],
        );

        assert_eq!(*llama_model_config.hidden_size(), 1024);
        assert_eq!(*llama_model_config.intermediate_size(), 4096);
        assert_eq!(*llama_model_config.max_position_embeddings(), 1024);
        assert_eq!(*llama_model_config.num_attention_heads(), 16);
        assert_eq!(*llama_model_config.num_hidden_layers(), 24);
        assert_eq!(llama_model_config.repo_name(), "meta/llama");
        assert_eq!(llama_model_config.model_type(), "llama");
        assert_eq!(llama_model_config.available_libraries(), vec![ModelLibraries::PyTorch]);
    }
}
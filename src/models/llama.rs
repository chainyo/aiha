//! Module for the Llama model
use serde::Deserialize;

use crate::models::{ ModelConfig, ModelLibraries };

/// A struct representing the Llama architecture parameters
#[derive(Clone, Debug, Deserialize)]
pub struct LlamaParams {
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
}

/// Llama model parameters implementation
impl LlamaParams {
    /// Build a new `LlamaParams` struct based on the provided parameters
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        max_sequence_length: i32,
        num_attention_heads: i32,
        num_hidden_layers: i32,
    ) -> LlamaParams {
        LlamaParams {
            hidden_size,
            intermediate_size,
            max_sequence_length,
            num_attention_heads,
            num_hidden_layers,
        }
    }
}

/// A struct representing a Llama model configuration
#[derive(Clone, Debug, Deserialize)]
pub struct LlamaModelConfig {
    /// Llama model parameters
    params: LlamaParams,
    /// Llama model type
    model_type: String,
    /// Llama model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// Llama model implementation
impl LlamaModelConfig {
    /// Build a new `LlamaModelConfig` struct based on the provided parameters
    pub fn new(
        params: LlamaParams,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> LlamaModelConfig {
        LlamaModelConfig {
            params,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `LlamaModelConfig`
impl ModelConfig for LlamaModelConfig {
    fn hidden_size(&self) -> &i32 {
        &self.params.hidden_size
    }

    fn intermediate_size(&self) -> &i32 {
        &self.params.intermediate_size
    }

    fn max_position_embeddings(&self) -> &i32 {
        &self.params.max_sequence_length
    }

    fn num_attention_heads(&self) -> &i32 {
        &self.params.num_attention_heads
    }

    fn num_hidden_layers(&self) -> &i32 {
        &self.params.num_hidden_layers
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
    fn test_llama_model_params() {
        let llama_params = LlamaParams {
            hidden_size: 768,
            intermediate_size: 3072,
            max_sequence_length: 1024,
            num_attention_heads: 12,
            num_hidden_layers: 12,
        };

        assert_eq!(llama_params.hidden_size, 768);
        assert_eq!(llama_params.intermediate_size, 3072);
        assert_eq!(llama_params.max_sequence_length, 1024);
        assert_eq!(llama_params.num_attention_heads, 12);
        assert_eq!(llama_params.num_hidden_layers, 12);
    }

    #[test]
    fn test_llama_model_config() {
        let llama_params = LlamaParams {
            hidden_size: 768,
            intermediate_size: 3072,
            max_sequence_length: 1024,
            num_attention_heads: 12,
            num_hidden_layers: 12,
        };

        let llama_model_config = LlamaModelConfig {
            params: llama_params,
            model_type: "llama".to_string(),
            available_libraries: vec![ModelLibraries::PyTorch],
        };

        assert_eq!(llama_model_config.params.hidden_size, 768);
        assert_eq!(llama_model_config.params.intermediate_size, 3072);
        assert_eq!(llama_model_config.params.max_sequence_length, 1024);
        assert_eq!(llama_model_config.params.num_attention_heads, 12);
        assert_eq!(llama_model_config.params.num_hidden_layers, 12);
        assert_eq!(llama_model_config.model_type, "llama");
        assert_eq!(llama_model_config.available_libraries, vec![ModelLibraries::PyTorch]);
    }

    #[test]
    fn test_llama_model_trait_implementation() {
        let llama_params = LlamaParams {
            hidden_size: 768,
            intermediate_size: 3072,
            max_sequence_length: 1024,
            num_attention_heads: 12,
            num_hidden_layers: 12,
        };

        let llama_model_config = LlamaModelConfig {
            params: llama_params,
            model_type: "llama".to_string(),
            available_libraries: vec![ModelLibraries::PyTorch],
        };

        assert_eq!(*llama_model_config.hidden_size(), 768);
        assert_eq!(*llama_model_config.intermediate_size(), 3072);
        assert_eq!(*llama_model_config.max_position_embeddings(), 1024);
        assert_eq!(*llama_model_config.num_attention_heads(), 12);
        assert_eq!(*llama_model_config.num_hidden_layers(), 12);
        assert_eq!(llama_model_config.model_type(), "llama");
        assert_eq!(llama_model_config.available_libraries(), vec![ModelLibraries::PyTorch]);
    }
}
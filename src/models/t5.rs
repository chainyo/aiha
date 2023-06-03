//! Module for the T5 model
use serde::Deserialize;

use crate::models::{ ModelConfig, ModelLibraries };

/// A struct representing the T5 architecture parameters
#[derive(Clone, Debug, Deserialize)]
pub struct T5Params {
    /// T5 model hidden_size
    d_model: i32,
    /// T5 model intermediate_size
    d_ff: i32,
    /// T5 model max_position_embeddings
    n_positions: i32,
    /// T5 model num_attention_heads
    n_heads: i32,
    /// T5 model num_hidden_layers
    n_layers: i32,
}

/// T5 model parameters implementation
impl T5Params {
    /// Build a new `T5Params` struct based on the provided parameters
    pub fn new(
        d_model: i32,
        d_ff: i32,
        n_positions: i32,
        n_heads: i32,
        n_layers: i32,
    ) -> T5Params {
        T5Params {
            d_model,
            d_ff,
            n_positions,
            n_heads,
            n_layers,
        }
    }
}

/// A struct representing a T5 model configuration
#[derive(Clone, Debug, Deserialize)]
pub struct T5ModelConfig {
    /// T5 model parameters
    params: T5Params,
    /// T5 model type
    model_type: String,
    /// T5 model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// T5 model implementation
impl T5ModelConfig {
    /// Build a new `T5ModelConfig` struct based on the provided parameters
    pub fn new(
        params: T5Params,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> T5ModelConfig {
        T5ModelConfig {
            params,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `T5ModelConfig`
impl ModelConfig for T5ModelConfig {
    fn hidden_size(&self) -> &i32 {
        &self.params.d_model
    }

    fn intermediate_size(&self) -> &i32 {
        &self.params.d_ff
    }

    fn max_position_embeddings(&self) -> &i32 {
        &self.params.n_positions
    }

    fn num_attention_heads(&self) -> &i32 {
        &self.params.n_heads
    }

    fn num_hidden_layers(&self) -> &i32 {
        &self.params.n_layers
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
    use super::*;

    #[test]
    fn test_t5_model_params() {
        let t5_params = T5Params::new(768, 3072, 512, 12, 12);
        assert_eq!(t5_params.d_model, 768);
        assert_eq!(t5_params.d_ff, 3072);
        assert_eq!(t5_params.n_positions, 512);
        assert_eq!(t5_params.n_heads, 12);
        assert_eq!(t5_params.n_layers, 12);
    }

    #[test]
    fn test_t5_model_config() {
        let t5_params = T5Params::new(768, 3072, 512, 12, 12);
        let t5_model_config = T5ModelConfig::new(
            t5_params,
            "t5".to_string(),
            vec![ModelLibraries::TensorFlow],
        );
        assert_eq!(t5_model_config.params.d_model, 768);
        assert_eq!(t5_model_config.params.d_ff, 3072);
        assert_eq!(t5_model_config.params.n_positions, 512);
        assert_eq!(t5_model_config.params.n_heads, 12);
        assert_eq!(t5_model_config.params.n_layers, 12);
        assert_eq!(t5_model_config.model_type, "t5");
        assert_eq!(t5_model_config.available_libraries, vec![ModelLibraries::TensorFlow]);
    }

    #[test]
    fn test_t5_model_trait_implementation() {
        let t5_params = T5Params::new(768, 3072, 512, 12, 12);
        let t5_model_config = T5ModelConfig::new(
            t5_params,
            "t5".to_string(),
            vec![ModelLibraries::TensorFlow],
        );
        assert_eq!(t5_model_config.hidden_size(), &768);
        assert_eq!(t5_model_config.intermediate_size(), &3072);
        assert_eq!(t5_model_config.max_position_embeddings(), &512);
        assert_eq!(t5_model_config.num_attention_heads(), &12);
        assert_eq!(t5_model_config.num_hidden_layers(), &12);
        assert_eq!(t5_model_config.model_type(), "t5");
        assert_eq!(t5_model_config.available_libraries(), vec![ModelLibraries::TensorFlow]);
    }
}
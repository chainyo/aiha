//! Module for the T5 model
use super::base::{ ModelConfig, ModelLibraries };

/// A struct representing a T5 model configuration
pub struct T5ModelConfig {
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
    /// T5 model Hugging Face repository name
    repo_name: String,
    /// T5 model type
    model_type: String,
    /// T5 model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// T5 model implementation
impl T5ModelConfig {
    /// Build a new `T5ModelConfig` struct based on the provided parameters
    pub fn new(
        d_model: i32,
        d_ff: i32,
        n_positions: i32,
        n_heads: i32,
        n_layers: i32,
        repo_name: String,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> T5ModelConfig {
        T5ModelConfig {
            d_model,
            d_ff,
            n_positions,
            n_heads,
            n_layers,
            repo_name,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `T5ModelConfig`
impl ModelConfig for T5ModelConfig {
    fn hidden_size(&self) -> &i32 {
        &self.d_model
    }

    fn intermediate_size(&self) -> &i32 {
        &self.d_ff
    }

    fn max_position_embeddings(&self) -> &i32 {
        &self.n_positions
    }

    fn num_attention_heads(&self) -> &i32 {
        &self.n_heads
    }

    fn num_hidden_layers(&self) -> &i32 {
        &self.n_layers
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
    use super::*;

    #[test]
    fn test_t5_model_config() {
        let t5_model_config = T5ModelConfig::new(
            1024,
            4096,
            512,
            16,
            12,
            "t5-base".to_string(),
            "t5".to_string(),
            vec![ModelLibraries::PyTorch],
        );

        assert_eq!(*t5_model_config.hidden_size(), 1024);
        assert_eq!(*t5_model_config.intermediate_size(), 4096);
        assert_eq!(*t5_model_config.max_position_embeddings(), 512);
        assert_eq!(*t5_model_config.num_attention_heads(), 16);
        assert_eq!(*t5_model_config.num_hidden_layers(), 12);
        assert_eq!(t5_model_config.repo_name(), "t5-base");
        assert_eq!(t5_model_config.model_type(), "t5");
        assert_eq!(
            t5_model_config.available_libraries(),
            vec![ModelLibraries::PyTorch]
        );
    }
}
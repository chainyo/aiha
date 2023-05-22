//! Module for the GPT2 model
use super::base::{ ModelConfig, ModelLibraries };

/// A struct representing a GPT2 model configuration
pub struct GPT2ModelConfig {
    /// GPT2 model hidden_size
    n_embd: i32,
    /// GPT2 model intermediate_size
    n_inner: i32,
    /// GPT2 model max_position_embeddings
    n_positions: i32,
    /// GPT2 model num_attention_heads
    n_head: i32,
    /// GPT2 model num_hidden_layers
    n_layer: i32,
    /// GPT2 model Hugging Face repository name
    repo_name: String,
    /// GPT2 model type
    model_type: String,
    /// GPT2 model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// GPT2 model implementation
impl GPT2ModelConfig {
    /// Build a new `GPT2ModelConfig` struct based on the provided parameters
    pub fn new(
        n_embd: i32,
        n_inner: Option<i32>,
        n_positions: i32,
        n_head: i32,
        n_layer: i32,
        repo_name: String,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> GPT2ModelConfig {
        let n_inner = n_inner.unwrap_or(4 * n_embd);
        GPT2ModelConfig {
            n_embd,
            n_inner,
            n_positions,
            n_head,
            n_layer,
            repo_name,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `GPT2ModelConfig`
impl ModelConfig for GPT2ModelConfig {
    fn hidden_size(&self) -> &i32 {
        &self.n_embd
    }

    fn intermediate_size(&self) -> &i32 {
        &self.n_inner
    }

    fn max_position_embeddings(&self) -> &i32 {
        &self.n_positions
    }

    fn num_attention_heads(&self) -> &i32 {
        &self.n_head
    }

    fn num_hidden_layers(&self) -> &i32 {
        &self.n_layer
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
    fn test_gpt2_model_config_new() {
        let gpt2_model_config = GPT2ModelConfig::new(
            768,
            None,
            1024,
            12,
            12,
            "gpt2".to_string(),
            "gpt2".to_string(),
            vec![ModelLibraries::PyTorch],
        );
        assert_eq!(*gpt2_model_config.hidden_size(), 768);
        assert_eq!(*gpt2_model_config.intermediate_size(), 3072);
        assert_eq!(*gpt2_model_config.max_position_embeddings(), 1024);
        assert_eq!(*gpt2_model_config.num_attention_heads(), 12);
        assert_eq!(*gpt2_model_config.num_hidden_layers(), 12);
        assert_eq!(gpt2_model_config.repo_name(), "gpt2".to_string());
        assert_eq!(gpt2_model_config.model_type(), "gpt2".to_string());
        assert_eq!(gpt2_model_config.available_libraries(), vec![ModelLibraries::PyTorch]);
    }
}
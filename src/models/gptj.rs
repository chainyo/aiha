//! Module for the GPT-J model
use super::base::{ ModelConfig, ModelLibraries };

/// A struct representing the GPT-J architecture parameters
pub struct GPTJParams {
    /// GPT-J model hidden_size
    n_embd: i32,
    /// GPT-J model intermediate_size
    n_inner: i32,
    /// GPT-J model max_position_embeddings
    n_positions: i32,
    /// GPT-J model num_attention_heads
    n_head: i32,
    /// GPT-J model num_hidden_layers
    n_layer: i32,
}

/// GPT-J model implementation
impl GPTJParams {
    /// Build a new `GPTJParams` struct based on the provided parameters
    pub fn new(
        n_embd: i32,
        n_inner: Option<i32>,
        n_positions: i32,
        n_head: i32,
        n_layer: i32,
    ) -> GPTJParams {
        let n_inner = n_inner.unwrap_or(4 * n_embd);
        GPTJParams {
            n_embd,
            n_inner,
            n_positions,
            n_head,
            n_layer,
        }
    }
}

/// A struct representing a GPT-J model configuration
pub struct GPTJModelConfig {
    /// GPT-J model parameters
    params: GPTJParams,
    /// GPT-J model Hugging Face repository name
    repo_name: String,
    /// GPT-J model type
    model_type: String,
    /// GPT-J model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// GPT-J model implementation
impl GPTJModelConfig {
    /// Build a new `GPTJModelConfig` struct based on the provided parameters
    pub fn new(
        params: GPTJParams,
        repo_name: String,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> GPTJModelConfig {
        GPTJModelConfig {
            params,
            repo_name,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `GPTJModelConfig`
impl ModelConfig for GPTJModelConfig {
    fn hidden_size(&self) -> &i32 {
        &self.params.n_embd
    }

    fn intermediate_size(&self) -> &i32 {
        &self.params.n_inner
    }

    fn max_position_embeddings(&self) -> &i32 {
        &self.params.n_positions
    }

    fn num_attention_heads(&self) -> &i32 {
        &self.params.n_head
    }

    fn num_hidden_layers(&self) -> &i32 {
        &self.params.n_layer
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
    fn test_gpt_j_params() {
        let params = GPTJParams::new(
            1024,
            None,
            1024,
            16,
            28,
        );
        assert_eq!(params.n_embd, 1024);
        assert_eq!(params.n_inner, 4096);
        assert_eq!(params.n_positions, 1024);
        assert_eq!(params.n_head, 16);
        assert_eq!(params.n_layer, 28);
    }

    #[test]
    fn test_gpt_j_model_config() {
        let params = GPTJParams::new(
            1024,
            None,
            1024,
            16,
            28,
        );
        let model_config = GPTJModelConfig::new(params, "EleutherAI/gpt-j-6B".to_string(), "gpt-j".to_string(), vec![ModelLibraries::PyTorch]);
        assert_eq!(model_config.params.n_embd, 1024);
        assert_eq!(model_config.params.n_inner, 4096);
        assert_eq!(model_config.params.n_positions, 1024);
        assert_eq!(model_config.params.n_head, 16);
        assert_eq!(model_config.params.n_layer, 28);
        assert_eq!(model_config.repo_name, "EleutherAI/gpt-j-6B");
        assert_eq!(model_config.model_type, "gpt-j");
        assert_eq!(model_config.available_libraries, vec![ModelLibraries::PyTorch]);
    }

    #[test]
    fn test_gpt_j_model_trait_implementation() {
        let params = GPTJParams::new(1024, Some(2048), 1024, 16, 28);
        let model_config = GPTJModelConfig::new(params, "EleutherAI/gpt-j-6B".to_string(), "gpt-j".to_string(), vec![ModelLibraries::PyTorch]);
        assert_eq!(*model_config.hidden_size(), 1024);
        assert_eq!(*model_config.intermediate_size(), 2048);
        assert_eq!(*model_config.max_position_embeddings(), 1024);
        assert_eq!(*model_config.num_attention_heads(), 16);
        assert_eq!(*model_config.num_hidden_layers(), 28);
        assert_eq!(model_config.repo_name, "EleutherAI/gpt-j-6B");
        assert_eq!(model_config.model_type, "gpt-j");
        assert_eq!(model_config.available_libraries, vec![ModelLibraries::PyTorch]);
    }
}
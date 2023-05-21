//! Module for the GPT-J model
use super::base::{ ModelConfig, ModelLibraries };

/// A struct representing a GPT-J model configuration
pub struct GPTJModelConfig {
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
        n_embd: i32,
        n_inner: Option<i32>,
        n_positions: i32,
        n_head: i32,
        n_layer: i32,
        repo_name: String,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> GPTJModelConfig {
        let n_inner = n_inner.unwrap_or(4 * n_embd);
        GPTJModelConfig {
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

/// Implementation of the `ModelConfig` trait for `GPTJModelConfig`
impl ModelConfig for GPTJModelConfig {
    fn hidden_size(&self) -> i32 {
        self.n_embd
    }

    fn intermediate_size(&self) -> i32 {
        self.n_inner
    }

    fn max_position_embeddings(&self) -> i32 {
        self.n_positions
    }

    fn num_attention_heads(&self) -> i32 {
        self.n_head
    }

    fn num_hidden_layers(&self) -> i32 {
        self.n_layer
    }

    fn repo_name(&self) -> String {
        self.repo_name.clone()
    }

    fn model_type(&self) -> String {
        self.model_type.clone()
    }

    fn available_libraries(&self) -> Vec<ModelLibraries> {
        self.available_libraries.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gptj_model_config_new() {
        let gptj_model_config = GPTJModelConfig::new(
            1024,
            None,
            1024,
            16,
            28,
            "EleutherAI/gpt-j-6B".to_string(),
            "gptj".to_string(),
            vec![ModelLibraries::PyTorch, ModelLibraries::TensorFlow],
        );
        assert_eq!(gptj_model_config.n_embd, 1024);
        assert_eq!(gptj_model_config.n_inner, 4096);
        assert_eq!(gptj_model_config.n_positions, 1024);
        assert_eq!(gptj_model_config.n_head, 16);
        assert_eq!(gptj_model_config.n_layer, 28);
        assert_eq!(gptj_model_config.repo_name, "EleutherAI/gpt-j-6B".to_string());
        assert_eq!(gptj_model_config.model_type, "gptj".to_string());
        assert_eq!(
            gptj_model_config.available_libraries,
            vec![ModelLibraries::PyTorch, ModelLibraries::TensorFlow]
        );
    }
}
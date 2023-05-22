//! Module for the GPT-Neo model
use super::base::{ ModelConfig, ModelLibraries };

/// A struct representing a GPT-Neo model configuration
pub struct GPTNeoModelConfig {
    /// GPT-Neo model hidden_size
    hidden_size: i32,
    /// GPT-Neo model intermediate_size
    intermediate_size: i32,
    /// GPT-Neo model max_position_embeddings
    max_position_embeddings: i32,
    /// GPT-Neo model num_attention_heads
    num_attention_heads: i32,
    /// GPT-Neo model num_hidden_layers
    num_hidden_layers: i32,
    /// GPT-Neo model Hugging Face repository name
    repo_name: String,
    /// GPT-Neo model type
    model_type: String,
    /// GPT-Neo model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// GPT-Neo model implementation
impl GPTNeoModelConfig {
    /// Build a new `GPTNeoModelConfig` struct based on the provided parameters
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        max_position_embeddings: i32,
        num_attention_heads: i32,
        num_hidden_layers: i32,
        repo_name: String,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> GPTNeoModelConfig {
        GPTNeoModelConfig {
            hidden_size,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
            repo_name,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `GPTNeoModelConfig`
impl ModelConfig for GPTNeoModelConfig {
    fn hidden_size(&self) -> &i32 {
        &self.hidden_size
    }

    fn intermediate_size(&self) -> &i32 {
        &self.intermediate_size
    }

    fn max_position_embeddings(&self) -> &i32 {
        &self.max_position_embeddings
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
    use super::*;

    #[test]
    fn test_gptneo_model_config_new() {
        let gptneo_model_config = GPTNeoModelConfig::new(
            768,
            3072,
            2048,
            12,
            12,
            "EleutherAI/gpt-neo-20b".to_string(),
            "gpt-neo".to_string(),
            vec![ModelLibraries::TensorFlow, ModelLibraries::PyTorch],
        );

        assert_eq!(*gptneo_model_config.hidden_size(), 768);
        assert_eq!(*gptneo_model_config.intermediate_size(), 3072);
        assert_eq!(*gptneo_model_config.max_position_embeddings(), 2048);
        assert_eq!(*gptneo_model_config.num_attention_heads(), 12);
        assert_eq!(*gptneo_model_config.num_hidden_layers(), 12);
        assert_eq!(gptneo_model_config.repo_name(), "EleutherAI/gpt-neo-20b");
        assert_eq!(gptneo_model_config.model_type(), "gpt-neo");
        assert_eq!(gptneo_model_config.available_libraries(), vec![ModelLibraries::TensorFlow, ModelLibraries::PyTorch]);
    }
}
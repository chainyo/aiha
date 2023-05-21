//! Module for the Bert model
use super::base::{ ModelConfig, ModelLibraries };

/// A struct representing a Bert model configuration
pub struct BertModelConfig {
    /// Bert model hidden_size
    hidden_size: i32,
    /// Bert model intermediate_size
    intermediate_size: i32,
    /// Bert model max_position_embeddings
    max_position_embeddings: i32,
    /// Bert model num_attention_heads
    num_attention_heads: i32,
    /// Bert model num_hidden_layers
    num_hidden_layers: i32,
    /// Bert model Hugging Face repository name
    repo_name: String,
    /// Bert model type
    model_type: String,
    /// Bert model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// Bert model implementation
impl BertModelConfig {
    /// Build a new `BertModelConfig` struct based on the provided parameters
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        max_position_embeddings: i32,
        num_attention_heads: i32,
        num_hidden_layers: i32,
        repo_name: String,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> BertModelConfig {
        BertModelConfig {
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

/// Implementation of the `ModelConfig` trait for `BertModelConfig`
impl ModelConfig for BertModelConfig {
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }

    fn max_position_embeddings(&self) -> i32 {
        self.max_position_embeddings
    }

    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }

    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
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
    fn test_bert_model_config() {
        let bert_model_config = BertModelConfig::new(
            768,
            3072,
            512,
            12,
            12,
            "bert-base-uncased".to_string(),
            "bert".to_string(),
            vec![ModelLibraries::PyTorch],
        );

        assert_eq!(bert_model_config.hidden_size(), 768);
        assert_eq!(bert_model_config.intermediate_size(), 3072);
        assert_eq!(bert_model_config.max_position_embeddings(), 512);
        assert_eq!(bert_model_config.num_attention_heads(), 12);
        assert_eq!(bert_model_config.num_hidden_layers(), 12);
        assert_eq!(bert_model_config.repo_name(), "bert-base-uncased");
        assert_eq!(bert_model_config.model_type(), "bert");
        assert_eq!(bert_model_config.available_libraries(), vec![ModelLibraries::PyTorch]);
    }
}
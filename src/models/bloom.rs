//! Module for the BLOOM model
use super::base::{ ModelConfig, ModelLibraries };

/// A struct representing the BLOOM architecture parameters
pub struct BloomParams {
    /// BLOOM model hidden_size
    n_embd: i32,
    /// BLOOM model intermediate_size
    n_inner: i32,
    /// BLOOM model doesn't have max_position_embeddings (apparently)
    /// BLOOM model num_attention_heads
    num_attention_heads: i32,
    /// BLOOM model num_hidden_layers
    n_layer: i32,
}

/// A struct representing a BLOOM model configuration
pub struct BloomModelConfig {
    /// BLOOM model parameters
    params: BloomParams,
    /// BLOOM model Hugging Face repository name
    repo_name: String,
    /// BLOOM model type
    model_type: String,
    /// BLOOM model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// BLOOM model implementation
impl BloomModelConfig {
    /// Build a new `BloomModelConfig` struct based on the provided parameters
    pub fn new(
        params: BloomParams,
        repo_name: String,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> BloomModelConfig {
        BloomModelConfig {
            params,
            repo_name,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `BloomModelConfig`
impl ModelConfig for BloomModelConfig {
    fn hidden_size(&self) -> &i32 {
        &self.params.n_embd
    }

    fn intermediate_size(&self) -> &i32 {
        &self.params.n_inner
    }

    fn max_position_embeddings(&self) -> &i32 {
        // self.max_position_embeddings
        &0
    }

    fn num_attention_heads(&self) -> &i32 {
        &self.params.num_attention_heads
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
    fn test_bloom_model_params() {
        let bloom_params = BloomParams {
            n_embd: 768,
            n_inner: 3072,
            num_attention_heads: 12,
            n_layer: 12,
        };

        assert_eq!(bloom_params.n_embd, 768);
        assert_eq!(bloom_params.n_inner, 3072);
        assert_eq!(bloom_params.num_attention_heads, 12);
        assert_eq!(bloom_params.n_layer, 12);
    }

    #[test]
    fn test_bloom_model_config() {
        let bloom_params = BloomParams {
            n_embd: 768,
            n_inner: 3072,
            num_attention_heads: 12,
            n_layer: 12,
        };

        let bloom_model_config = BloomModelConfig {
            params: bloom_params,
            repo_name: "bigscience/bloomz-3b".to_string(),
            model_type: "bloom".to_string(),
            available_libraries: vec![ModelLibraries::PyTorch],
        };

        assert_eq!(bloom_model_config.params.n_embd, 768);
        assert_eq!(bloom_model_config.params.n_inner, 3072);
        assert_eq!(bloom_model_config.params.num_attention_heads, 12);
        assert_eq!(bloom_model_config.params.n_layer, 12);
        assert_eq!(bloom_model_config.repo_name, "bigscience/bloomz-3b");
        assert_eq!(bloom_model_config.model_type, "bloom");
        assert_eq!(bloom_model_config.available_libraries, vec![ModelLibraries::PyTorch]);
    }

    #[test]
    fn test_bloom_model_trait_implementation() {
        let bloom_params = BloomParams {
            n_embd: 768,
            n_inner: 3072,
            num_attention_heads: 12,
            n_layer: 12,
        };

        let bloom_model_config = BloomModelConfig {
            params: bloom_params,
            repo_name: "bigscience/bloomz-3b".to_string(),
            model_type: "bloom".to_string(),
            available_libraries: vec![ModelLibraries::PyTorch],
        };

        assert_eq!(*bloom_model_config.hidden_size(), 768);
        assert_eq!(*bloom_model_config.intermediate_size(), 3072);
        assert_eq!(*bloom_model_config.max_position_embeddings(), 0);
        assert_eq!(*bloom_model_config.num_attention_heads(), 12);
        assert_eq!(*bloom_model_config.num_hidden_layers(), 12);
        assert_eq!(bloom_model_config.repo_name(), "bigscience/bloomz-3b");
        assert_eq!(bloom_model_config.model_type(), "bloom");
        assert_eq!(bloom_model_config.available_libraries(), vec![ModelLibraries::PyTorch]);
    }
}
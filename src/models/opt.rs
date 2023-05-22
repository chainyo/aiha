//! Module for the OPT model
use super::base::{ ModelConfig, ModelLibraries };

/// A struct representing the OPT architecture parameters
pub struct OPTParams {
    /// OPT model hidden_size
    hidden_size: i32,
    /// OPT model intermediate_size
    ffn_dim: i32,
    /// OPT model max_position_embeddings
    max_position_embeddings: i32,
    /// OPT model num_attention_heads
    num_attention_heads: i32,
    /// OPT model num_hidden_layers
    num_hidden_layers: i32,
}

/// OPT model parameters implementation
impl OPTParams {
    /// Build a new `OPTParams` struct based on the provided parameters
    pub fn new(
        hidden_size: i32,
        ffn_dim: i32,
        max_position_embeddings: i32,
        num_attention_heads: i32,
        num_hidden_layers: i32,
    ) -> OPTParams {
        OPTParams {
            hidden_size,
            ffn_dim,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
        }
    }
}

/// A struct representing a OPT model configuration
pub struct OPTModelConfig {
    /// OPT model parameters
    params: OPTParams,
    /// OPT model Hugging Face repository name
    repo_name: String,
    /// OPT model type
    model_type: String,
    /// OPT model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// OPT model implementation
impl OPTModelConfig {
    /// Build a new `OPTModelConfig` struct based on the provided parameters
    pub fn new(
        params: OPTParams,
        repo_name: String,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> OPTModelConfig {
        OPTModelConfig {
            params,
            repo_name,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `OPTModelConfig`
impl ModelConfig for OPTModelConfig {
    fn hidden_size(&self) -> &i32 {
        &self.params.hidden_size
    }

    fn intermediate_size(&self) -> &i32 {
        &self.params.ffn_dim
    }

    fn max_position_embeddings(&self) -> &i32 {
        &self.params.max_position_embeddings
    }

    fn num_attention_heads(&self) -> &i32 {
        &self.params.num_attention_heads
    }

    fn num_hidden_layers(&self) -> &i32 {
        &self.params.num_hidden_layers
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
    fn test_opt_model_params() {
        let opt_params = OPTParams::new(
            768,
            3072,
            512,
            12,
            12,
        );

        assert_eq!(opt_params.hidden_size, 768);
        assert_eq!(opt_params.ffn_dim, 3072);
        assert_eq!(opt_params.max_position_embeddings, 512);
        assert_eq!(opt_params.num_attention_heads, 12);
        assert_eq!(opt_params.num_hidden_layers, 12);
    }

    #[test]
    fn test_opt_model_config() {
        let opt_params = OPTParams::new(
            768,
            3072,
            512,
            12,
            12,
        );

        let opt_model_config = OPTModelConfig::new(
            opt_params,
            String::from("facebook/opt-1.3b"),
            String::from("opt"),
            vec![ModelLibraries::TensorFlow, ModelLibraries::PyTorch],
        );

        assert_eq!(opt_model_config.params.hidden_size, 768);
        assert_eq!(opt_model_config.params.ffn_dim, 3072);
        assert_eq!(opt_model_config.params.max_position_embeddings, 512);
        assert_eq!(opt_model_config.params.num_attention_heads, 12);
        assert_eq!(opt_model_config.params.num_hidden_layers, 12);
        assert_eq!(opt_model_config.repo_name, "facebook/opt-1.3b");
        assert_eq!(opt_model_config.model_type, "opt");
        assert_eq!(opt_model_config.available_libraries, vec![ModelLibraries::TensorFlow, ModelLibraries::PyTorch]);
    }

    #[test]
    fn test_opt_model_trait_implementation() {
        let opt_params = OPTParams::new(
            768,
            3072,
            512,
            12,
            12,
        );

        let opt_model_config = OPTModelConfig::new(
            opt_params,
            String::from("facebook/opt-1.3b"),
            String::from("opt"),
            vec![ModelLibraries::TensorFlow, ModelLibraries::PyTorch],
        );

        assert_eq!(opt_model_config.hidden_size(), &768);
        assert_eq!(opt_model_config.intermediate_size(), &3072);
        assert_eq!(opt_model_config.max_position_embeddings(), &512);
        assert_eq!(opt_model_config.num_attention_heads(), &12);
        assert_eq!(opt_model_config.num_hidden_layers(), &12);
        assert_eq!(opt_model_config.repo_name(), "facebook/opt-1.3b");
        assert_eq!(opt_model_config.model_type(), "opt");
        assert_eq!(opt_model_config.available_libraries(), vec![ModelLibraries::TensorFlow, ModelLibraries::PyTorch]);
    }
}
//! Module for the GPT2 model
use serde::Deserialize;
use serde_json::Value;

use crate::models::{ ModelConfigTrait, ModelError, ModelLibraries };

/// A struct representing the GPT2 architecture parameters
#[derive(Clone, Debug, Deserialize)]
pub struct GPT2Params {
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
}

/// GPT2 model implementation
impl GPT2Params {
    /// Build a new `GPT2Params` struct based on the provided parameters
    pub fn new(
        n_embd: i32,
        n_inner: Option<i32>,
        n_positions: i32,
        n_head: i32,
        n_layer: i32,
    ) -> GPT2Params {
        let n_inner = n_inner.unwrap_or(4 * n_embd);
        GPT2Params {
            n_embd,
            n_inner,
            n_positions,
            n_head,
            n_layer,
        }
    }
    /// Build from a JSON value
    pub fn from_json(value: Value) -> Result<GPT2Params, ModelError> {
        let n_embd = value["n_embd"]
            .as_i64()
            .ok_or(ModelError::MissingField("n_embd".to_string()))? as i32;

        let n_inner = value["n_inner"]
            .as_i64()
            .map(|val| val as i32); // map the i64 to i32 if it exists

        let n_positions = value["n_positions"]
            .as_i64()
            .ok_or(ModelError::MissingField("n_positions".to_string()))? as i32;

        let n_head = value["n_head"]
            .as_i64()
            .ok_or(ModelError::MissingField("n_head".to_string()))? as i32;

        let n_layer = value["n_layer"]
            .as_i64()
            .ok_or(ModelError::MissingField("n_layer".to_string()))? as i32;

        Ok(GPT2Params::new(n_embd, n_inner, n_positions, n_head, n_layer))
    }
}

/// A struct representing a GPT2 model configuration
#[derive(Clone, Debug, Deserialize)]
pub struct GPT2ModelConfig {
    /// GPT2 model parameters
    params: GPT2Params,
    /// GPT2 model type
    model_type: String,
    /// GPT2 model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// GPT2 model implementation
impl GPT2ModelConfig {
    /// Build a new `GPT2ModelConfig` struct based on the provided parameters
    pub fn new(
        params: GPT2Params,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> GPT2ModelConfig {
        GPT2ModelConfig {
            params,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `GPT2ModelConfig`
impl ModelConfigTrait for GPT2ModelConfig {
    fn hidden_size(&self) -> i32 {
        self.params.n_embd
    }

    fn intermediate_size(&self) -> i32 {
        self.params.n_inner
    }

    fn max_position_embeddings(&self) -> i32 {
        self.params.n_positions
    }

    fn num_attention_heads(&self) -> i32 {
        self.params.n_head
    }

    fn num_hidden_layers(&self) -> i32 {
        self.params.n_layer
    }

    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn available_libraries(&self) -> &[ModelLibraries] {
        &self.available_libraries
    }

    fn from_json(value: Value) -> Result<Self, ModelError> where Self: Sized {
        let params = GPT2Params::from_json(value.clone())?;
    
        let model_type = match value["model_type"].as_str() {
            Some(model_type) => model_type.to_string(),
            None => return Err(ModelError::MissingField("model_type".to_string())),
        };

        // TODO: Implement this
        let available_libraries = vec![ModelLibraries::PyTorch];
        // let available_libraries = match value["available_libraries"].as_array() {
        //     Some(al) => al.iter().map(|v| ModelLibraries::from_str(v.as_str().unwrap()).unwrap()).collect(),
        //     None => return Err(ModelError::MissingField("available_libraries".to_string())),
        // };

        Ok(GPT2ModelConfig::new(params, model_type, available_libraries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_params() {
        let params = GPT2Params::new(768, None, 1024, 12, 12);
        assert_eq!(params.n_embd, 768);
        assert_eq!(params.n_inner, 3072);
        assert_eq!(params.n_positions, 1024);
        assert_eq!(params.n_head, 12);
        assert_eq!(params.n_layer, 12);
    }

    #[test]
    fn test_gpt2_model_config() {
        let params = GPT2Params::new(768, None, 1024, 12, 12);
        let model_config = GPT2ModelConfig::new(params, "gpt2".to_string(), vec![ModelLibraries::Transformers]);
        assert_eq!(model_config.params.n_embd, 768);
        assert_eq!(model_config.params.n_inner, 3072);
        assert_eq!(model_config.params.n_positions, 1024);
        assert_eq!(model_config.params.n_head, 12);
        assert_eq!(model_config.params.n_layer, 12);
        assert_eq!(model_config.model_type, "gpt2");
        assert_eq!(model_config.available_libraries, vec![ModelLibraries::Transformers]);
    }

    #[test]
    fn test_gpt2_model_trait_implementation() {
        let params = GPT2Params::new(768, None, 1024, 12, 12);
        let model_config = GPT2ModelConfig::new(params, "gpt2".to_string(), vec![ModelLibraries::Transformers]);
        assert_eq!(model_config.hidden_size(), 768);
        assert_eq!(model_config.intermediate_size(), 3072);
        assert_eq!(model_config.max_position_embeddings(), 1024);
        assert_eq!(model_config.num_attention_heads(), 12);
        assert_eq!(model_config.num_hidden_layers(), 12);
        assert_eq!(model_config.model_type(), "gpt2");
        assert_eq!(model_config.available_libraries(), vec![ModelLibraries::Transformers]);
    }
}
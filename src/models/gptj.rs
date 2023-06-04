//! Module for the GPT-J model
use serde::Deserialize;
use serde_json::Value;

use crate::models::{ ModelConfigTrait, ModelError, ModelLibraries };

/// A struct representing the GPT-J architecture parameters
#[derive(Clone, Debug, Deserialize)]
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
    /// Build from a JSON value
    pub fn from_json(value: Value) -> Result<GPTJParams, ModelError> {
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
    
        Ok(GPTJParams::new(n_embd, n_inner, n_positions, n_head, n_layer))
    }
}

/// A struct representing a GPT-J model configuration
#[derive(Clone, Debug, Deserialize)]
pub struct GPTJModelConfig {
    /// GPT-J model parameters
    params: GPTJParams,
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
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> GPTJModelConfig {
        GPTJModelConfig {
            params,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfig` trait for `GPTJModelConfig`
impl ModelConfigTrait for GPTJModelConfig {
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

    fn from_json(value: Value) -> Result<Self, ModelError> {
        let params = GPTJParams::from_json(value.clone())?;

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

        Ok(GPTJModelConfig::new(params, model_type, available_libraries))
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
        let model_config = GPTJModelConfig::new(params, "gpt-j".to_string(), vec![ModelLibraries::PyTorch]);
        assert_eq!(model_config.params.n_embd, 1024);
        assert_eq!(model_config.params.n_inner, 4096);
        assert_eq!(model_config.params.n_positions, 1024);
        assert_eq!(model_config.params.n_head, 16);
        assert_eq!(model_config.params.n_layer, 28);
        assert_eq!(model_config.model_type, "gpt-j");
        assert_eq!(model_config.available_libraries, vec![ModelLibraries::PyTorch]);
    }

    #[test]
    fn test_gpt_j_model_trait_implementation() {
        let params = GPTJParams::new(1024, Some(2048), 1024, 16, 28);
        let model_config = GPTJModelConfig::new(params, "gpt-j".to_string(), vec![ModelLibraries::PyTorch]);
        assert_eq!(model_config.hidden_size(), 1024);
        assert_eq!(model_config.intermediate_size(), 2048);
        assert_eq!(model_config.max_position_embeddings(), 1024);
        assert_eq!(model_config.num_attention_heads(), 16);
        assert_eq!(model_config.num_hidden_layers(), 28);
        assert_eq!(model_config.model_type, "gpt-j");
        assert_eq!(model_config.available_libraries, vec![ModelLibraries::PyTorch]);
    }
}
//! Module for the GPT-Neo model
use serde::Deserialize;
use serde_json::Value;

use crate::models::{ ModelConfigTrait, ModelError, ModelLibraries };

/// A struct representing the GPT-Neo architecture parameters
#[derive(Clone, Debug, Deserialize)]
pub struct GPTNeoParams {
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
}

/// GPT-Neo model parameters implementation
impl GPTNeoParams {
    /// Build a new `GPTNeoParams` struct based on the provided parameters
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        max_position_embeddings: i32,
        num_attention_heads: i32,
        num_hidden_layers: i32,
    ) -> GPTNeoParams {
        GPTNeoParams {
            hidden_size,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
        }
    }
    /// Build from a JSON value
    pub fn from_json(value: Value) -> Result<GPTNeoParams, ModelError> {
        let hidden_size = value["hidden_size"]
            .as_i64()
            .ok_or(ModelError::MissingField("hidden_size".to_string()))? as i32;

        let intermediate_size = value["intermediate_size"]
            .as_i64()
            .ok_or(ModelError::MissingField("intermediate_size".to_string()))? as i32;

        let max_position_embeddings = value["max_position_embeddings"]
            .as_i64()
            .ok_or(ModelError::MissingField("max_position_embeddings".to_string()))? as i32;

        let num_attention_heads = value["num_attention_heads"]
            .as_i64()
            .ok_or(ModelError::MissingField("num_attention_heads".to_string()))? as i32;

        let num_hidden_layers = value["num_hidden_layers"]
            .as_i64()
            .ok_or(ModelError::MissingField("num_hidden_layers".to_string()))? as i32;

        Ok(GPTNeoParams::new(
            hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers
        ))
    }
}

/// A struct representing a GPT-Neo model configuration
#[derive(Clone, Debug, Deserialize)]
pub struct GPTNeoModelConfig {
    /// GPT-Neo model parameters
    params: GPTNeoParams,
    /// GPT-Neo model type
    model_type: String,
    /// GPT-Neo model available libraries
    available_libraries: Vec<ModelLibraries>,
}

/// GPT-Neo model implementation
impl GPTNeoModelConfig {
    /// Build a new `GPTNeoModelConfig` struct based on the provided parameters
    pub fn new(
        params: GPTNeoParams,
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> GPTNeoModelConfig {
        GPTNeoModelConfig {
            params,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfigTrait` trait for `GPTNeoModelConfig`
impl ModelConfigTrait for GPTNeoModelConfig {
    fn hidden_size(&self) -> i32 {
        self.params.hidden_size
    }

    fn intermediate_size(&self) -> i32 {
        self.params.intermediate_size
    }

    fn max_position_embeddings(&self) -> i32 {
        self.params.max_position_embeddings
    }

    fn num_attention_heads(&self) -> i32 {
        self.params.num_attention_heads
    }

    fn num_hidden_layers(&self) -> i32 {
        self.params.num_hidden_layers
    }

    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn available_libraries(&self) -> &[ModelLibraries] {
        &self.available_libraries
    }

    fn from_json(value: Value) -> Result<Self, ModelError> {
        let params = GPTNeoParams::from_json(value["params"].clone())?;
    
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

        Ok(GPTNeoModelConfig::new(params, model_type, available_libraries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_neo_params() {
        let params = GPTNeoParams::new(768,3072,1024,12,12);
        assert_eq!(params.hidden_size, 768);
        assert_eq!(params.intermediate_size, 3072);
        assert_eq!(params.max_position_embeddings, 1024);
        assert_eq!(params.num_attention_heads, 12);
        assert_eq!(params.num_hidden_layers, 12);
    }

    #[test]
    fn test_gpt_neo_model_config() {
        let params = GPTNeoParams::new(768,3072,1024,12,12);
        let model_config = GPTNeoModelConfig::new(params, "gpt_neo".to_string(), vec![ModelLibraries::PyTorch]);
        assert_eq!(model_config.params.hidden_size, 768);
        assert_eq!(model_config.params.intermediate_size, 3072);
        assert_eq!(model_config.params.max_position_embeddings, 1024);
        assert_eq!(model_config.params.num_attention_heads, 12);
        assert_eq!(model_config.params.num_hidden_layers, 12);
        assert_eq!(model_config.model_type, "gpt_neo");
        assert_eq!(model_config.available_libraries, vec![ModelLibraries::PyTorch]);
    }

    #[test]
    fn test_gpt_neo_model_trait_implementation() {
        let params = GPTNeoParams::new(768,3072,1024,12,12);
        let model_config = GPTNeoModelConfig::new(params, "gpt_neo".to_string(), vec![ModelLibraries::PyTorch]);
        assert_eq!(model_config.hidden_size(), 768);
        assert_eq!(model_config.intermediate_size(), 3072);
        assert_eq!(model_config.max_position_embeddings(), 1024);
        assert_eq!(model_config.num_attention_heads(), 12);
        assert_eq!(model_config.num_hidden_layers(), 12);
        assert_eq!(model_config.model_type(), "gpt_neo");
        assert_eq!(model_config.available_libraries(), vec![ModelLibraries::PyTorch]);
    }
}
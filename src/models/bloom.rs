//! Module for the BLOOM model
use serde::Deserialize;
use serde_json::Value;

use crate::models::{ ModelConfigTrait, ModelError, ModelLibraries };

/// A struct representing the BLOOM architecture parameters
#[derive(Clone, Debug, Deserialize)]
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

/// BLOOM model parameters implementation
impl BloomParams {
    /// Build a new `BloomParams` struct based on the provided parameters
    pub fn new(
        n_embd: i32,
        n_inner: i32,
        num_attention_heads: i32,
        n_layer: i32,
    ) -> BloomParams {
        BloomParams {
            n_embd,
            n_inner,
            num_attention_heads,
            n_layer,
        }
    }
    /// Build from a JSON value
    pub fn from_json(value: Value) -> Result<BloomParams, ModelError> {
        let n_embd = value["n_embd"]
            .as_i64()
            .ok_or(ModelError::MissingField("n_embd".to_string()))? as i32;

        let n_inner = value["n_inner"]
            .as_i64()
            .ok_or(ModelError::MissingField("n_inner".to_string()))? as i32;

        let num_attention_heads = value["num_attention_heads"]
            .as_i64()
            .ok_or(ModelError::MissingField("num_attention_heads".to_string()))? as i32;

        let n_layer = value["n_layer"]
            .as_i64()
            .ok_or(ModelError::MissingField("n_layer".to_string()))? as i32;

        Ok(BloomParams::new(
            n_embd,
            n_inner,
            num_attention_heads,
            n_layer,
        ))
    }
}

/// A struct representing a BLOOM model configuration
#[derive(Clone, Debug, Deserialize)]
pub struct BloomModelConfig {
    /// BLOOM model parameters
    params: BloomParams,
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
        model_type: String,
        available_libraries: Vec<ModelLibraries>,
    ) -> BloomModelConfig {
        BloomModelConfig {
            params,
            model_type,
            available_libraries,
        }
    }
}

/// Implementation of the `ModelConfigTrait` trait for `BloomModelConfig`
impl ModelConfigTrait for BloomModelConfig {
    fn hidden_size(&self) -> i32 {
        self.params.n_embd
    }

    fn intermediate_size(&self) -> i32 {
        self.params.n_inner
    }

    fn max_position_embeddings(&self) -> i32 {
        // self.max_position_embeddings
        0
    }

    fn num_attention_heads(&self) -> i32 {
        self.params.num_attention_heads
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
        let params = BloomParams::from_json(value.clone())?;

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

        Ok(BloomModelConfig::new(
            params,
            model_type,
            available_libraries,
        ))
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
            model_type: "bloom".to_string(),
            available_libraries: vec![ModelLibraries::PyTorch],
        };

        assert_eq!(bloom_model_config.params.n_embd, 768);
        assert_eq!(bloom_model_config.params.n_inner, 3072);
        assert_eq!(bloom_model_config.params.num_attention_heads, 12);
        assert_eq!(bloom_model_config.params.n_layer, 12);
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
            model_type: "bloom".to_string(),
            available_libraries: vec![ModelLibraries::PyTorch],
        };

        assert_eq!(bloom_model_config.hidden_size(), 768);
        assert_eq!(bloom_model_config.intermediate_size(), 3072);
        assert_eq!(bloom_model_config.max_position_embeddings(), 0);
        assert_eq!(bloom_model_config.num_attention_heads(), 12);
        assert_eq!(bloom_model_config.num_hidden_layers(), 12);
        assert_eq!(bloom_model_config.model_type(), "bloom");
        assert_eq!(bloom_model_config.available_libraries(), vec![ModelLibraries::PyTorch]);
    }
}
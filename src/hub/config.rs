//! Config metadata struct
use serde::Deserialize;

use crate::models::{
    BertModelConfig,
    // BloomModelConfig,
    // GPT2ModelConfig,
    GPTJModelConfig,
    // GPTNeoModelConfig,
    // LlamaModelConfig,
    ModelConfigTrait,
    ModelError,
    // OPTModelConfig,
    // T5ModelConfig,
};

/// Enum all the possible model types
#[derive(Clone, Debug, Deserialize)]
pub enum ModelConfig {
    /// Bert model config
    Bert(BertModelConfig),
    /// Bloom model config
    // Bloom(BloomModelConfig),
    // /// GPT2 model config
    // Gpt2(GPT2ModelConfig),
    /// GPTJ model config
    GptJ(GPTJModelConfig),
    // /// GPTNeo model config
    // GPTNeo(GPTNeoModelConfig),
    // /// Llama model config
    // Llama(LlamaModelConfig),
    // /// OPT model config
    // Opt(OPTModelConfig),
    // /// T5 model config
    // T5(T5ModelConfig),
}

/// Model config implementation
impl ModelConfigTrait for ModelConfig {
    fn hidden_size(&self) -> i32 {
        match self {
            ModelConfig::Bert(config) => config.hidden_size(),
            // ModelConfig::Bloom(config) => config.hidden_size(),
            // ModelConfig::Gpt2(config) => config.hidden_size(),
            ModelConfig::GptJ(config) => config.hidden_size(),
            // ModelConfig::GPTNeo(config) => config.hidden_size(),
            // ModelConfig::Llama(config) => config.hidden_size(),
            // ModelConfig::Opt(config) => config.hidden_size(),
            // ModelConfig::T5(config) => config.hidden_size(),
        }
    }
    fn intermediate_size(&self) -> i32 {
        match self {
            ModelConfig::Bert(config) => config.intermediate_size(),
            // ModelConfig::Bloom(config) => config.intermediate_size(),
            // ModelConfig::Gpt2(config) => config.intermediate_size(),
            ModelConfig::GptJ(config) => config.intermediate_size(),
            // ModelConfig::GPTNeo(config) => config.intermediate_size(),
            // ModelConfig::Llama(config) => config.intermediate_size(),
            // ModelConfig::Opt(config) => config.intermediate_size(),
            // ModelConfig::T5(config) => config.intermediate_size(),
        }
    }
    fn max_position_embeddings(&self) -> i32 {
        match self {
            ModelConfig::Bert(config) => config.max_position_embeddings(),
            // ModelConfig::Bloom(config) => config.max_position_embeddings(),
            // ModelConfig::Gpt2(config) => config.max_position_embeddings(),
            ModelConfig::GptJ(config) => config.max_position_embeddings(),
            // ModelConfig::GPTNeo(config) => config.max_position_embeddings(),
            // ModelConfig::Llama(config) => config.max_position_embeddings(),
            // ModelConfig::Opt(config) => config.max_position_embeddings(),
            // ModelConfig::T5(config) => config.max_position_embeddings(),
        }
    }
    fn num_attention_heads(&self) -> i32 {
        match self {
            ModelConfig::Bert(config) => config.num_attention_heads(),
            // ModelConfig::Bloom(config) => config.num_attention_heads(),
            // ModelConfig::Gpt2(config) => config.num_attention_heads(),
            ModelConfig::GptJ(config) => config.num_attention_heads(),
            // ModelConfig::GPTNeo(config) => config.num_attention_heads(),
            // ModelConfig::Llama(config) => config.num_attention_heads(),
            // ModelConfig::Opt(config) => config.num_attention_heads(),
            // ModelConfig::T5(config) => config.num_attention_heads(),
        }
    }
    fn num_hidden_layers(&self) -> i32 {
        match self {
            ModelConfig::Bert(config) => config.num_hidden_layers(),
            // ModelConfig::Bloom(config) => config.num_hidden_layers(),
            // ModelConfig::Gpt2(config) => config.num_hidden_layers(),
            ModelConfig::GptJ(config) => config.num_hidden_layers(),
            // ModelConfig::GPTNeo(config) => config.num_hidden_layers(),
            // ModelConfig::Llama(config) => config.num_hidden_layers(),
            // ModelConfig::Opt(config) => config.num_hidden_layers(),
            // ModelConfig::T5(config) => config.num_hidden_layers(),
        }
    }
    fn model_type(&self) -> &str {
        match self {
            ModelConfig::Bert(config) => config.model_type(),
            // ModelConfig::Bloom(config) => config.model_type(),
            // ModelConfig::Gpt2(config) => config.model_type(),
            ModelConfig::GptJ(config) => config.model_type(),
            // ModelConfig::GPTNeo(config) => config.model_type(),
            // ModelConfig::Llama(config) => config.model_type(),
            // ModelConfig::Opt(config) => config.model_type(),
            // ModelConfig::T5(config) => config.model_type(),
        }
    }
    fn available_libraries(&self) -> &[crate::ModelLibraries] {
        match self {
            ModelConfig::Bert(config) => config.available_libraries(),
            // ModelConfig::Bloom(config) => config.available_libraries(),
            // ModelConfig::Gpt2(config) => config.available_libraries(),
            ModelConfig::GptJ(config) => config.available_libraries(),
            // ModelConfig::GPTNeo(config) => config.available_libraries(),
            // ModelConfig::Llama(config) => config.available_libraries(),
            // ModelConfig::Opt(config) => config.available_libraries(),
            // ModelConfig::T5(config) => config.available_libraries(),
        }
    }
    fn from_json(value: serde_json::Value) -> Result<Self, ModelError> where Self: Sized {
        let model_type = value["model_type"].as_str().ok_or(ModelError::MissingField("model_type".to_string()))?;
        match model_type {
            "bert" => Ok(ModelConfig::Bert(BertModelConfig::from_json(value)?)),
            // "bloom" => Ok(ModelConfig::Bloom(BloomModelConfig::from_json(value)?)),
            // "gpt2" => Ok(ModelConfig::Gpt2(GPT2ModelConfig::from_json(value)?)),
            "gptj" => Ok(ModelConfig::GptJ(GPTJModelConfig::from_json(value)?)),
            // "gpt_neo" => Ok(ModelConfig::GPTNeo(GPTNeoModelConfig::from_json(value)?)),
            // "gpt_neox" => Ok(ModelConfig::GPTNeo(GPTNeoModelConfig::from_json(value)?)),
            // "llama" => Ok(ModelConfig::Llama(LlamaModelConfig::from_json(value)?)),
            // "opt" => Ok(ModelConfig::Opt(OPTModelConfig::from_json(value)?)),
            // "t5" => Ok(ModelConfig::T5(T5ModelConfig::from_json(value)?)),
            _ => Err(ModelError::ModelNotImplemented(model_type.to_string())),
        }
    }
}


#[cfg(test)]
mod tests {
}

//! Config metadata struct
use serde::Deserialize;

use crate::models::{
    BertModelConfig,
    BloomModelConfig,
    GPT2ModelConfig,
    GPTJModelConfig,
    GPTNeoModelConfig,
    LlamaModelConfig,
    OPTModelConfig,
    T5ModelConfig,
};

/// Enum all the possible model types
#[derive(Clone, Debug, Deserialize)]
pub enum ModelConfig {
    /// Bert model config
    Bert(BertModelConfig),
    /// Bloom model config
    Bloom(BloomModelConfig),
    /// GPT2 model config
    Gpt2(GPT2ModelConfig),
    /// GPTJ model config
    GptJ(GPTJModelConfig),
    /// GPTNeo model config
    GPTNeo(GPTNeoModelConfig),
    /// Llama model config
    Llama(LlamaModelConfig),
    /// OPT model config
    Opt(OPTModelConfig),
    /// T5 model config
    T5(T5ModelConfig),
}

#[cfg(test)]
mod tests {
}

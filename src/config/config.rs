use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    #[serde(default)]
    dim_ff: i32,
    #[serde(default)]
    hidden_size: i32,
    #[serde(default)]
    intermediate_size: i32,
    #[serde(default)]
    max_position_embeddings: i32,
    #[serde(default)]
    n_head: i32,
    #[serde(default)]
    n_layer: i32,
    #[serde(default)]
    num_attention_heads: i32,
    #[serde(default)]
    num_hidden_layers: i32,
}

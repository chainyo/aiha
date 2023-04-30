use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::error::ConfigError;

/// A struct representing a model config URL.
#[pyclass]
#[derive(Debug)]
pub struct ConfigUrl {
    path: String,
}

#[pymethods]
impl ConfigUrl {
    #[new]
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_owned(),
        }
    }

    /// Returns a string representing the model config URL.
    /// 
    /// # Example
    /// 
    /// ```
    /// use aiha::config::ConfigUrl;
    /// 
    /// let model_path = "EleutherAI/gpt-neox-20b";
    /// let config_url = ConfigUrl::new(model_path);
    /// 
    /// assert_eq!(config_url.get_url(), "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json");
    #[pyo3(text_signature = "($self)")]
    pub fn get_url(&self) -> String {
        let base_url = "https://huggingface.co/";
        format!("{}{}/resolve/main/config.json", base_url, self.path)
    }
}

/// A struct representing a generic model config.
#[pyclass]
#[derive(Debug)]
pub struct ModelConfig {
    hidden_size: Option<i32>,
    intermediate_size: Option<i32>,
    max_position_embeddings: Option<i32>,
    num_attention_heads: Option<i32>,
    num_hidden_layers: Option<i32>,
}

#[pymethods]
impl ModelConfig {
    fn check_fields(&self) -> Result<(), ConfigError> {
        if self.hidden_size.is_none() {
            return Err(ConfigError("hidden_size".to_string()));
        }
        if self.intermediate_size.is_none() {
            return Err(ConfigError("intermediate_size".to_string()));
        }
        if self.max_position_embeddings.is_none() {
            return Err(ConfigError("max_position_embeddings".to_string()));
        }
        if self.num_attention_heads.is_none() {
            return Err(ConfigError("num_attention_heads".to_string()));
        }
        if self.num_hidden_layers.is_none() {
            return Err(ConfigError("num_hidden_layers".to_string()));
        }
        Ok(())
    }

    #[new]
    pub fn new(
        hidden_size: Option<i32>,
        intermediate_size: Option<i32>,
        max_position_embeddings: Option<i32>,
        num_attention_heads: Option<i32>,
        num_hidden_layers: Option<i32>,
    ) -> PyResult<Self> {
        let config = Self {
            hidden_size,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
        };
        match config.check_fields() {
            Ok(_) => Ok(config),
            Err(e) => Err(PyErr::new::<PyValueError, _>(format!("ConfigError: {}", e.0))),
        }
    }

    /// Returns a string representing the model config.
    /// 
    /// # Example
    /// 
    /// ```
    /// use aiha::config::ModelConfig;
    #[pyo3(text_signature = "($self)")]
    pub fn get_config(&self) -> String {
        format!("{{\"hidden_size\":{},\"intermediate_size\":{},\"max_position_embeddings\":{},\"num_attention_heads\":{},\"num_hidden_layers\":{}}}", self.hidden_size.unwrap(), self.intermediate_size.unwrap(), self.max_position_embeddings.unwrap(), self.num_attention_heads.unwrap(), self.num_hidden_layers.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test for the ConfigUrl struct
    #[test]
    fn test_config_url() {
        let model_path = "EleutherAI/gpt-neox-20b";
        let config_url = ConfigUrl::new(model_path);
        assert_eq!(config_url.get_url(), "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json");
    }

    // Test for the ModelConfig struct
    #[test]
    fn test_model_config() {
        let config = ModelConfig::new(Some(1024), Some(4096), Some(2048), Some(16), Some(24)).unwrap();
        assert_eq!(config.get_config(), "{\"hidden_size\":1024,\"intermediate_size\":4096,\"max_position_embeddings\":2048,\"num_attention_heads\":16,\"num_hidden_layers\":24}");
    }

    // Test for the ConfigError struct
    #[test]
    fn test_config_error() {
        let config = ModelConfig::new(None, None, None, None, None);
        assert_eq!(config.is_err(), true);
    }
}

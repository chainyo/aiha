use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyFileNotFoundError};
use pyo3::types::PyDict;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use crate::scan::Scan;
use crate::utils::{download_model_config, is_file_local};

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

    /// Download the model config file from the model config URL.
    /// 
    /// # Example
    /// 
    /// ```
    /// use aiha::config::ConfigUrl;
    /// 
    /// let model_path = "EleutherAI/gpt-neox-20b";
    /// let config_url = ConfigUrl::new(model_path);
    /// 
    /// config_url.download_config();
    /// ```
    #[pyo3(text_signature = "($self)")]
    pub fn download_config(&self) -> Result<(), pyo3::PyErr> {
        let url = self.get_url();
        let model_cache_folder = Scan::get_model_cache_folder(&self.path)
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Failed to get the model cache folder")
            })?;
        let config_path = model_cache_folder.join("config.json");
        if !config_path.exists() {
            download_model_config(&url, &config_path)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(())
    }
}

/// A struct representing a generic model config.
pub struct ModelConfig {
    hidden_size: i32,
    intermediate_size: i32,
    max_position_embeddings: i32,
    num_attention_heads: i32,
    num_hidden_layers: i32,
}

impl ModelConfig {
    fn new(
        hidden_size: i32,
        intermediate_size: i32,
        max_position_embeddings: i32,
        num_attention_heads: i32,
        num_hidden_layers: i32,
    ) -> Result<Self, PyErr> {
        let config = Self {
            hidden_size,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
        };
        Ok(config)
    }

    pub fn from_file(path: &str) -> Result<Self, PyErr>{
        let config_path: PathBuf;
        if is_file_local(path) {
            config_path = Path::new(path).to_path_buf();
        } else {
            let config_url = ConfigUrl::new(path);
            config_url.download_config()?;
            let model_cache_folder = Scan::get_model_cache_folder(&config_url.path)
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Failed to get the model cache folder")
                })?;
            config_path = model_cache_folder.join("config.json");
    
        }
        // Load the config file.
        let file = File::open(config_path).map_err(|e| {
            PyErr::new::<PyFileNotFoundError, _>(format!("Failed to open config file: {}", e))
        })?;
        let reader = BufReader::new(file);
        let config_json: serde_json::Value =
            serde_json::from_reader(reader).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Failed to parse config file: {}", e))
            })?;

        // Get the config values.
        let hidden_size = config_json["hidden_size"]
            .as_i64()
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "Failed to get the hidden size from the config file",
                )
            })? as i32;
        let intermediate_size = config_json["intermediate_size"]
            .as_i64()
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "Failed to get the intermediate size from the config file",
                )
            })? as i32;
        let max_position_embeddings = config_json["max_position_embeddings"]
            .as_i64()
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "Failed to get the max position embeddings from the config file",
                )
            })? as i32;
        let num_attention_heads = config_json["num_attention_heads"]
            .as_i64()
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "Failed to get the number of attention heads from the config file",
                )
            })? as i32;
        let num_hidden_layers = config_json["num_hidden_layers"]
            .as_i64()
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "Failed to get the number of hidden layers from the config file",
                )
            })? as i32;
    
        let config = Self {
            hidden_size,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
        };
        Ok(config)
    }

    /// Returns a python dictionary containing the model config.
    #[pyo3(text_signature = "($self)")]
    pub fn to_dict(&self) -> Result<Py<PyDict>, pyo3::PyErr> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("hidden_size", self.hidden_size)?;
            dict.set_item("intermediate_size", self.intermediate_size)?;
            dict.set_item("max_position_embeddings", self.max_position_embeddings)?;
            dict.set_item("num_attention_heads", self.num_attention_heads)?;
            dict.set_item("num_hidden_layers", self.num_hidden_layers)?;
            Ok(dict.into_py(py))
        })
    }
}

#[pyclass(dict, module = "config", name = "ModelConfig")]
#[pyo3(text_signature = "(self, model)")]
#[derive(Clone)]
pub struct PyModelConfig {
    config: ModelConfig,
}

impl PyModelConfig {
    fn new(config: ModelConfig) -> Self {
        PyModelConfig { config }
    }

    fn from_file(model: PyModel) -> Self {
        PyModelConfig::new(ModelConfigImpl::new(model))
    }
}

#[pymethods]
impl PyModelConfig {
    #[new]
    fn __new__(model: PyRef<PyModel>) -> Self {
        PyModelConfig::from_file(model.clone())
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
}

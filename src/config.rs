use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use crate::error::ConfigError;
use crate::scan::Scan;
use crate::utils::download_model_config;

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

    /// Downloads the model config file from the HuggingFace Hub.
    #[pyo3(text_signature = "($self)")]
    fn download_config(&self) -> Result<(), pyo3::PyErr> {
        let url = format!("{}/config.json", self.get_url());
        let save_path = format!("{}/config.json", ".cache/aiha");
        download_model_config(&url, &save_path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        Ok(())
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

    #[getter]
    pub fn hidden_size(&self) -> PyResult<Option<i32>> {
        Ok(self.hidden_size)
    }

    #[setter]
    pub fn set_hidden_size(&mut self, value: Option<i32>) -> PyResult<()> {
        self.hidden_size = value;
        Ok(())
    }

    #[getter]
    pub fn intermediate_size(&self) -> PyResult<Option<i32>> {
        Ok(self.intermediate_size)
    }

    #[setter]
    pub fn set_intermediate_size(&mut self, value: Option<i32>) -> PyResult<()> {
        self.intermediate_size = value;
        Ok(())
    }

    #[getter]
    pub fn max_position_embeddings(&self) -> PyResult<Option<i32>> {
        Ok(self.max_position_embeddings)
    }

    #[setter]
    pub fn set_max_position_embeddings(&mut self, value: Option<i32>) -> PyResult<()> {
        self.max_position_embeddings = value;
        Ok(())
    }

    #[getter]
    pub fn num_attention_heads(&self) -> PyResult<Option<i32>> {
        Ok(self.num_attention_heads)
    }

    #[setter]
    pub fn set_num_attention_heads(&mut self, value: Option<i32>) -> PyResult<()> {
        self.num_attention_heads = value;
        Ok(())
    }

    #[getter]
    pub fn num_hidden_layers(&self) -> PyResult<Option<i32>> {
        Ok(self.num_hidden_layers)
    }

    #[setter]
    pub fn set_num_hidden_layers(&mut self, value: Option<i32>) -> PyResult<()> {
        self.num_hidden_layers = value;
        Ok(())
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

    // Test for the ConfigError struct
    #[test]
    fn test_config_error() {
        let config = ModelConfig::new(None, None, None, None, None);
        assert_eq!(config.is_err(), true);
    }

    // Test for the ModelConfig struct
    #[test]
    fn test_model_config() {
        let config = ModelConfig::new(Some(1024), Some(4096), Some(2048), Some(16), Some(32));
        assert_eq!(config.is_ok(), true);
    }

    // Test for the ModelConfig struct
    #[test]
    fn test_to_dict() {
        pyo3::prepare_freethreaded_python();
        let config = ModelConfig::new(Some(1024), Some(4096), Some(2048), Some(16), Some(32)).unwrap();
        let dict_result = config.to_dict();
        assert!(dict_result.is_ok());
        let dict = dict_result.unwrap();

        Python::with_gil(|py| {
            let dict_ref = dict.as_ref(py);
            assert_eq!(dict_ref.get_item("hidden_size").unwrap().extract::<i32>().unwrap(), 1024);
            assert_eq!(dict_ref.get_item("intermediate_size").unwrap().extract::<i32>().unwrap(), 4096);
            assert_eq!(dict_ref.get_item("max_position_embeddings").unwrap().extract::<i32>().unwrap(), 2048);
            assert_eq!(dict_ref.get_item("num_attention_heads").unwrap().extract::<i32>().unwrap(), 16);
            assert_eq!(dict_ref.get_item("num_hidden_layers").unwrap().extract::<i32>().unwrap(), 32);
        });
    }
}

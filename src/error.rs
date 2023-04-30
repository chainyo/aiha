use pyo3::{exceptions, PyErr};

/// Errors that can occur when creating a new model config.
#[derive(Debug)]
pub struct ConfigError(pub String);
impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Missing field: {}", self.0)
    }
}
impl std::error::Error for ConfigError {}

impl From<ConfigError> for PyErr {
    fn from(err: ConfigError) -> PyErr {
        PyErr::new::<exceptions::PyValueError, _>(err.to_string())
    }
}

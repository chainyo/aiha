//! Config metadata struct
use serde::Deserialize;

/// Struct for storing the config metadata
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    /// The list of architectures
    pub architectures: Vec<String>,
    /// The model type
    pub model_type: String,
}

/// Implement the `Config` struct
impl Config {
    /// Create a new Config struct
    pub fn new(architectures: Vec<String>, model_type: String) -> Self {
        Self {
            architectures,
            model_type,
        }
    }
    /// Get the list of architectures as a vector of strings
    pub fn get_architectures(&self) -> Vec<&'_ String> {
        self.architectures.iter().collect()
    }
    /// Get the model_type as a string
    pub fn get_model_type(&self) -> &String {
        &self.model_type
    }
}

/// Implement the partial equality for the `Config` struct
impl PartialEq for Config {
    fn eq(&self, other: &Self) -> bool {
        self.architectures == other.architectures && self.model_type == other.model_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_config_new() {
        let architectures = vec!["armv7".to_string(), "armv8".to_string()];
        let model_type = "classification".to_string();
        let config = Config::new(
            vec!["armv7".to_string(), "armv8".to_string()],
            "classification".to_string()
        );
        assert_eq!(config.architectures, architectures);
        assert_eq!(config.model_type, model_type);
    }

    #[test]
    fn test_config_get_architectures() {
        let config = Config::new(
            vec!["armv7".to_string(), "armv8".to_string()],
            "classification".to_string()
        );
        let architectures = config.get_architectures();
        assert_eq!(architectures, vec!["armv7", "armv8"]);
    }

    #[test]
    fn test_config_get_model_type() {
        let config = Config::new(
            vec!["armv7".to_string(), "armv8".to_string()],
            "classification".to_string()
        );
        let model_type = config.get_model_type();
        assert_eq!(model_type, "classification");
    }

    #[test]
    fn test_config_partial_eq() {
        let architectures = vec!["armv7".to_string(), "armv8".to_string()];
        let model_type = "classification".to_string();
        let config = Config::new(architectures.clone(), model_type.clone());
        let config2 = Config::new(architectures, model_type);
        assert_eq!(config, config2);
    }
}

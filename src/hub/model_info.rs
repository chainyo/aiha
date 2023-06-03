//! Model Info metadata struct
use std::collections::HashMap;
use std::fmt;
use std::ops::Not;

use serde_json::Value;
use serde::Deserialize;

use crate::hub::Config;
use crate::hub::ModelFile;
use crate::hub::Siblings;

/// Struct for storing the model metadata
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    /// The model ID of the repository (e.g. `username/repo_name`)
    pub model_id: Option<String>,
    /// The associated tags of the repository
    pub tags: Option<Vec<String>>,
    /// The pipeline tag of the repository
    pub pipeline_tag: Option<String>,
    /// The siblings of the repository
    pub siblings: Option<Siblings>,
    /// The config file associated with the repository
    pub config: Option<Config>,
    /// The security status (e.g. `{"containsInfected": False}`)
    pub security_status: Option<HashMap<String, Value>>,
}

/// Implement the `ModelInfo` struct
impl ModelInfo {
    /// Create a new ModelInfo struct
    pub fn new(
        model_id: Option<String>,
        tags: Option<Vec<String>>,
        pipeline_tag: Option<String>,
        siblings: Option<Siblings>,
        config: Option<Config>,
        security_status: Option<HashMap<String, Value>>,
    ) -> Self {
        Self {
            model_id,
            tags,
            pipeline_tag,
            siblings,
            config,
            security_status,
        }
    }
    /// Get the siblings of the repository
    pub fn get_siblings(&self) -> Option<&'_ Siblings> {
        self.siblings.as_ref()
    }
    /// Get the config model_type of the repository
    pub fn get_model_type(&self) -> Option<&'_ String> {
        self.config.as_ref().map(|config| config.get_model_type())
    }
    /// Get the config architectures of the repository
    pub fn get_architectures(&self) -> Option<Vec<&'_ String>> {
        self.config.as_ref().map(|config| config.get_architectures())
    }
    /// Check for security vulnerabilities
    pub fn has_vulnerabilities(&self) -> bool {
        if let Some(security_status) = &self.security_status {
            if let Some(true) = security_status.get("hasUnsafeFile").and_then(|v| v.as_bool()) {
                return true;
            }
            if let Some(value) = security_status
                .get("scansDone").map(|v| !v.is_null().not()) {
                    if !value {
                        return true;
                    }
                }
            if let Some(value) = security_status
                .get("clamAVInfectedFiles").map(|v| !v.is_null().not()) {
                    if !value {
                        return true;
                    }
                }
            if let Some(value) = security_status
                .get("dangerousPickles").map(|v| !v.is_null().not()) {
                    if !value {
                        return true;
                    }
                }
        }
        false
    }
}

/// Implement the display of the ModelInfo struct
impl fmt::Display for ModelInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model Name: {:?}", self.model_id)?;
        if let Some(tags) = &self.tags {
            write!(f, ", Tags: {:?}", tags)?;
        }
        if let Some(pipeline_tag) = &self.pipeline_tag {
            write!(f, ", Task: {:?}", pipeline_tag)?;
        }
        Ok(())
    }
}

/// Implement the creation of the ModelInfo struct from a serde_json::Value
impl From<serde_json::Value> for ModelInfo {
    fn from(response_json: serde_json::Value) -> Self {
        let _siblings: Vec<serde_json::Value> = serde_json::from_value(response_json["siblings"].clone()).unwrap_or_default();
        let siblings = Siblings::new(
            _siblings
                .iter()
                .map(|sibling| ModelFile::from(sibling.clone()))
                .collect(),
        );
        let _config: HashMap<String, serde_json::Value> = serde_json::from_value(response_json["config"].clone()).unwrap_or_default();
        let config = Config::new(
            serde_json::from_value(_config["architectures"].clone()).unwrap_or_default(),
            serde_json::from_value(_config["model_type"].clone()).unwrap_or_default()
        );
        ModelInfo::new(
            response_json["id"].as_str().map(|s| s.to_string()),
            response_json["tags"].as_array().map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect()),
            response_json["pipeline_tag"].as_str().map(|s| s.to_string()),
            Some(siblings),
            Some(config),
            serde_json::from_value(response_json["securityStatus"].clone()).unwrap_or_default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::{from_value, json};
    use crate::hub::{LfsInfo, ModelFile, Siblings};

    fn create_model_files() -> Vec<ModelFile> {
        vec![
            ModelFile {
                rfilename: String::from("file1"),
                size: Some(100),
                blob_id: Some(String::from("blob1")),
                lfs: Some(LfsInfo {
                    size: 100,
                    sha256: String::from("abc123"),
                    pointer_size: Some(50),
                }),
            },
            ModelFile {
                rfilename: String::from("file2"),
                size: Some(200),
                blob_id: Some(String::from("blob2")),
                lfs: Some(LfsInfo {
                    size: 200,
                    sha256: String::from("def456"),
                    pointer_size: Some(100),
                }),
            },
        ]
    }

    fn create_sample_siblings() -> Siblings {
        Siblings { siblings: create_model_files() }
    }

    fn create_model_info(vulnerabilities: bool) -> ModelInfo {
        let siblings = create_sample_siblings();
        let config = Config::new(vec!["GPTJForCausalLM".to_string()], "gptj".to_string());
        let security_status = if vulnerabilities {
            Some(from_value(json!({
                "scansDone": null,
                "dangerousPickles": null,
                "hasUnsafeFile": true,
                "repositoryId": "models/EleutherAI/gpt-j-6b",
                "revision": "f98c709453c9402b1309b032f40df1c10ad481a2",
                "clamAVInfectedFiles": vec![
                    "pytorch_model.bin".to_string(),
                    "config.json".to_string(),
                    "vocab.txt".to_string()]
            })).unwrap())
        } else {
            Some(from_value(json!({
                "scansDone": null,
                "dangerousPickles": null,
                "hasUnsafeFile": false,
                "repositoryId": "models/EleutherAI/gpt-j-6b",
                "revision": "f98c709453c9402b1309b032f40df1c10ad481a2",
                "clamAVInfectedFiles": null,
            })).unwrap())
        };
        ModelInfo::new(
            Some("EleutherAI/gpt-j-6b".to_string()),
            Some(vec!["causal-lm".to_string(), "pytorch".to_string()]),
            Some("text-generation".to_string()),
            Some(siblings),
            Some(config),
            security_status,
        )
    }

    #[test]
    fn test_new_model_info() {
        let model_id = Some("username/repo_name".to_string());
        let tags = Some(vec!["tag1".to_string(), "tag2".to_string()]);
        let pipeline_tag = Some("pipeline-tag".to_string());
        let siblings = Some(create_sample_siblings());
        let architectures = vec!["x86_64".to_string(), "armv7".to_string()];
        let model_type = "image_classification".to_string();
        let config = Some(Config::new(architectures, model_type));
        let security_status = Some(HashMap::new());

        let model_info = ModelInfo::new(
            model_id.clone(),
            tags.clone(),
            pipeline_tag.clone(),
            siblings.clone(),
            config.clone(),
            security_status.clone(),
        );

        assert_eq!(model_info.model_id, model_id);
        assert_eq!(model_info.tags, tags);
        assert_eq!(model_info.pipeline_tag, pipeline_tag);
        assert_eq!(model_info.siblings, siblings);
        assert_eq!(model_info.config, config);
        assert_eq!(model_info.security_status, security_status);
    }

    #[test]
    fn test_model_info_get_siblings() {
        let model_info = create_model_info(false);
        assert_eq!(
            model_info.get_siblings(),
            Some(&create_sample_siblings())
        );
    }

    #[test]
    fn test_model_info_get_model_type() {
        let model_info = create_model_info(false);
        assert_eq!(model_info.get_model_type(), Some(&"gptj".to_string()));
    }

    #[test]
    fn test_model_info_get_architectures() {
        let model_info = create_model_info(false);
        assert_eq!(
            model_info.get_architectures(),
            Some(vec![&"GPTJForCausalLM".to_string()])
        );
    }

    #[test]
    fn test_model_info_check_security() {
        let model_info = create_model_info(false);
        assert!(!model_info.has_vulnerabilities());
        let model_info = create_model_info(true);
        assert!(model_info.has_vulnerabilities());
    }

    #[test]
    fn test_model_info_to_string() {
        let model_id = Some("username/repo_name".to_string());
        let tags = Some(vec!["tag1".to_string(), "tag2".to_string()]);
        let pipeline_tag = Some("task1".to_string());
        let model_info = ModelInfo {
            model_id,
            tags,
            pipeline_tag,
            siblings: None,
            config: None,
            security_status: None,
        };
        assert_eq!(
            model_info.to_string(),
            "Model Name: Some(\"username/repo_name\"), Tags: [\"tag1\", \"tag2\"], Task: \"task1\""
        );
    }
}
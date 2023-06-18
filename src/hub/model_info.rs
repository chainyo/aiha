//! Model Info metadata struct
use std::collections::HashMap;
use std::fmt;
use std::ops::Not;

use serde::Deserialize;
use serde_json::Value;

use crate::hub::{ModelConfig, ModelFile, Siblings};
use crate::models::{ModelConfigTrait, ModelLibraries};

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
    pub config: Option<ModelConfig>,
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
        config: Option<ModelConfig>,
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
    pub fn get_model_type(&self) -> Option<String> {
        self.config
            .as_ref()
            .map(|config| config.model_type().to_string())
    }
    /// Get the config available libraries of the repository
    pub fn get_available_libraries(&self) -> Option<Vec<ModelLibraries>> {
        self.config
            .as_ref()
            .map(|config| config.available_libraries().to_vec())
    }
    /// Check for security vulnerabilities
    pub fn has_vulnerabilities(&self) -> bool {
        if let Some(security_status) = &self.security_status {
            if let Some(true) = security_status
                .get("hasUnsafeFile")
                .and_then(|v| v.as_bool())
            {
                return true;
            }
            if let Some(value) = security_status.get("scansDone").map(|v| !v.is_null().not()) {
                if !value {
                    return true;
                }
            }
            if let Some(value) = security_status
                .get("clamAVInfectedFiles")
                .map(|v| !v.is_null().not())
            {
                if !value {
                    return true;
                }
            }
            if let Some(value) = security_status
                .get("dangerousPickles")
                .map(|v| !v.is_null().not())
            {
                if !value {
                    return true;
                }
            }
        }
        false
    }
    /// Create a new ModelInfo struct from a serde_json::Value
    pub fn from_json(value: serde_json::Value) -> Self {
        let _siblings: Vec<serde_json::Value> =
            serde_json::from_value(value["siblings"].clone()).unwrap_or_default();
        let siblings = Siblings::new(
            _siblings
                .iter()
                .map(|sibling| ModelFile::from(sibling.clone()))
                .collect(),
        );
        ModelInfo::new(
            value["id"].as_str().map(|s| s.to_string()),
            value["tags"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect()),
            value["pipeline_tag"].as_str().map(|s| s.to_string()),
            Some(siblings),
            None,
            serde_json::from_value(value["securityStatus"].clone()).unwrap_or_default(),
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::{ModelFile, Siblings};
    use pretty_assertions::assert_eq;
    use serde_json::{from_value, json};

    fn create_model_files() -> Vec<ModelFile> {
        vec![
            ModelFile {
                rfilename: String::from("file1"),
                size: Some(100),
                oid: Some(String::from("oid1")),
            },
            ModelFile {
                rfilename: String::from("file2"),
                size: Some(200),
                oid: Some(String::from("oid2")),
            },
        ]
    }

    fn create_sample_siblings() -> Siblings {
        Siblings {
            siblings: create_model_files(),
        }
    }

    fn create_model_info(vulnerabilities: bool) -> ModelInfo {
        let siblings = create_sample_siblings();
        let security_status = if vulnerabilities {
            Some(
                from_value(json!({
                    "scansDone": null,
                    "dangerousPickles": null,
                    "hasUnsafeFile": true,
                    "repositoryId": "models/EleutherAI/gpt-j-6b",
                    "revision": "f98c709453c9402b1309b032f40df1c10ad481a2",
                    "clamAVInfectedFiles": vec![
                        "pytorch_model.bin".to_string(),
                        "config.json".to_string(),
                        "vocab.txt".to_string()]
                }))
                .unwrap(),
            )
        } else {
            Some(
                from_value(json!({
                    "scansDone": null,
                    "dangerousPickles": null,
                    "hasUnsafeFile": false,
                    "repositoryId": "models/EleutherAI/gpt-j-6b",
                    "revision": "f98c709453c9402b1309b032f40df1c10ad481a2",
                    "clamAVInfectedFiles": null,
                }))
                .unwrap(),
            )
        };
        ModelInfo::new(
            Some("EleutherAI/gpt-j-6b".to_string()),
            Some(vec!["causal-lm".to_string(), "pytorch".to_string()]),
            Some("text-generation".to_string()),
            Some(siblings),
            None,
            security_status,
        )
    }

    #[test]
    fn test_new_model_info() {
        let model_id = Some("username/repo_name".to_string());
        let tags = Some(vec!["tag1".to_string(), "tag2".to_string()]);
        let pipeline_tag = Some("pipeline-tag".to_string());
        let siblings = Some(create_sample_siblings());
        let security_status = Some(HashMap::new());

        let model_info = ModelInfo::new(
            model_id.clone(),
            tags.clone(),
            pipeline_tag.clone(),
            siblings.clone(),
            None,
            security_status.clone(),
        );

        assert_eq!(model_info.model_id, model_id);
        assert_eq!(model_info.tags, tags);
        assert_eq!(model_info.pipeline_tag, pipeline_tag);
        assert_eq!(model_info.siblings, siblings);
        assert_eq!(model_info.security_status, security_status);
    }

    #[test]
    fn test_model_info_get_siblings() {
        let model_info = create_model_info(false);
        assert_eq!(model_info.get_siblings(), Some(&create_sample_siblings()));
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

//! Module for interacting with Hugging Face Hub.
use std::collections::{HashSet, HashMap};
use std::error::Error;
use std::fmt;
use std::ops::Not;

use percent_encoding::{ utf8_percent_encode, AsciiSet, CONTROLS };
use reqwest::{Client, header::HeaderMap};
use serde::Deserialize;
use serde_json::Value;
use tokio::time::Duration;


const CUSTOM_ENCODE_SET: &AsciiSet = &CONTROLS.add(b' ').add(b'/').add(b':').add(b'@');
const HUB_ENDPOINT: &str = "https://huggingface.co";

/// Struct for storing the files metadata
pub struct HubFile {
    /// The name of the file
    pub name: String,
    /// The size of the file in bytes
    pub size: u64,
    /// The extension of the file
    pub extension: String,
    /// The URL of the file
    pub url: String,
}

/// Implement the `HubFile` struct
impl HubFile {
    /// Create a new HubFile
    pub fn new(name: String, size: u64, extension: String, url: String) -> Self {
        Self {
            name,
            size,
            extension,
            url,
        }
    }
}

/// Struct that represent a list of siblings of a model
#[derive(Clone, Debug, Deserialize)]
pub struct Siblings {
    siblings: Vec<String>,
}

/// Implement the `Siblings` struct
impl Siblings {
    /// Create a new Siblings struct
    pub fn new(siblings: Vec<String>) -> Self {
        Self {
            siblings,
        }
    }
    /// Get the list of siblings as a vector of strings
    pub fn get_siblings(&self) -> Vec<&'_ String> {
        self.siblings.iter().collect()
    }
}

/// Implement the partial equality for the `Siblings` struct
impl PartialEq for Siblings {
    fn eq(&self, other: &Self) -> bool {
        self.siblings == other.siblings
    }
}

/// Struct for storing the config metadata
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    architectures: Vec<String>,
    model_type: String,
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

/// Struct for storing the model metadata
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    /// The model ID of the repository (e.g. `username/repo_name`)
    model_id: Option<String>,
    /// The associated tags of the repository
    tags: Option<Vec<String>>,
    /// The pipeline tag of the repository
    pipeline_tag: Option<String>,
    /// The siblings of the repository
    siblings: Option<Siblings>,
    /// The config file associated with the repository
    config: Option<Config>,
    /// The security status (e.g. `{"containsInfected": False}`)
    security_status: Option<HashMap<String, Value>>,
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
        let _siblings: Vec<HashMap<String, String>> = serde_json::from_value(response_json["siblings"].clone()).unwrap_or_default();
        let siblings = Siblings::new(
            _siblings
                .iter()
                .map(|sibling| sibling["rfilename"].clone())
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

/// Format the user agent string
fn http_user_agent(library_name: Option<&str>, library_version: Option<&str>, user_agent: Option<&str>) -> String {
    let mut parts = vec![];
    if let Some(name) = library_name {
        parts.push(format!("{}-rust", name));
    }
    if let Some(version) = library_version {
        parts.push(version.to_string());
    }
    if let Some(agent) = user_agent {
        parts.push(agent.to_string());
    }
    parts.join("; ")
}

/// Deduplicate the user agent string
fn deduplicate_user_agent(user_agent: &str) -> String {
    let keys: Vec<_> = user_agent.split(';').map(|s| s.trim()).collect();
    let mut deduplicated = Vec::new();
    let mut seen = HashSet::new();
    for key in keys {
        if !seen.contains(key) {
            deduplicated.push(key);
            seen.insert(key.to_owned());
        }
    }
    deduplicated.join("; ")
}

/// Make a request to the Hugging Face Hub API to retrieve the model info
pub async fn retrieve_model_info(
    repo_id: &str,
    revision: Option<&str>,
    timeout: Option<f32>,
    files_metadata: Option<bool>,
    token: Option<impl ToString>,
) -> Result<ModelInfo, Box<dyn Error>> {
    let path = if let Some(rev) = revision.as_ref() {
        let encoded_revision = utf8_percent_encode(rev, CUSTOM_ENCODE_SET).to_string();
        format!("{}/api/models/{}/revision/{}", HUB_ENDPOINT, repo_id, encoded_revision)
    } else {
        format!("{}/api/models/{}", HUB_ENDPOINT, repo_id)
    };

    let mut params = HashMap::new();
    params.insert("securityStatus", "true");
    if files_metadata.unwrap_or(false) {
        params.insert("blobs", "true");
    }

    let mut headers = HeaderMap::new();
    let _user_agent = deduplicate_user_agent(
        http_user_agent(
            Some(env!("CARGO_PKG_NAME")),
            Some(env!("CARGO_PKG_VERSION")),
            None
        ).as_str()
    );
    headers.insert("user-agent", _user_agent.parse().unwrap());
    if let Some(token) = token {
        headers.insert("authorization", format!("Bearer {}", token.to_string()).parse().unwrap());
    } else {
        return Err("No token provided".into());
    }

    // check if timeout is not none and create a Duration object with the timeout value in seconds
    let _timeout = if let Some(timeout) = timeout {
        Some(Duration::from_secs_f32(timeout))
    } else {
        Some(Duration::from_secs_f32(30.0))
    };

    let client = Client::new();
    let response = client.get(path)
        .headers(headers)
        .timeout(_timeout.unwrap())
        .query(&params)
        .send()
        .await?;

    let response_json = response.json::<serde_json::Value>().await?;
    let model_info = ModelInfo::from(response_json);
    Ok(model_info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{from_value, json};

    fn create_model_info(vulnerabilities: bool) -> ModelInfo {
        let siblings = Siblings::new(
            vec!["pytorch_model.bin".to_string(), "config.json".to_string(), "vocab.txt".to_string()]
        );
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
    fn test_custom_encode_set() {
        let encoded = percent_encoding::utf8_percent_encode("abc:/@", CUSTOM_ENCODE_SET);
        assert_eq!(encoded.to_string(), "abc%3A%2F%40");
    }

    #[test]
    fn test_hub_endpoint_constant() {
        assert_eq!(HUB_ENDPOINT, "https://huggingface.co");
    }

    #[test]
    fn test_new() {
        let siblings = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];
        let s = Siblings::new(siblings.clone());
        assert_eq!(s.siblings, siblings);
    }

    #[test]
    fn test_get_siblings() {
        let siblings = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];
        let s = Siblings::new(siblings.clone());
        let siblings_refs = s.get_siblings();
        assert_eq!(siblings_refs.len(), siblings.len());
        for i in 0..siblings.len() {
            assert_eq!(siblings_refs[i], &siblings[i]);
        }
    }

    #[test]
    fn test_hub_file_new() {
        let name = "test.txt".to_string();
        let size = 1024;
        let extension = "txt".to_string();
        let url = "https://example.com/test.txt".to_string();
        let hub_file = HubFile::new(name.clone(), size, extension.clone(), url.clone());
        assert_eq!(hub_file.name, name);
        assert_eq!(hub_file.size, size);
        assert_eq!(hub_file.extension, extension);
        assert_eq!(hub_file.url, url);
    }

    #[test]
    fn test_new_model_info() {
        let model_id = Some("username/repo_name".to_string());
        let tags = Some(vec!["tag1".to_string(), "tag2".to_string()]);
        let pipeline_tag = Some("pipeline-tag".to_string());
        let siblings = Some(Siblings::new(vec!["sibling1".to_string(), "sibling2".to_string()]));
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
            Some(&Siblings {
                siblings: vec![
                    "pytorch_model.bin".to_string(), "config.json".to_string(), "vocab.txt".to_string()
                ]
            })
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

    #[test]
    fn test_http_user_agent() {
        let library_name = Some("aiha");
        let library_version = Some("1.0.0");
        let user_agent = Some("my-user-agent");
        let result = http_user_agent(library_name, library_version, user_agent);
        assert_eq!(result, "aiha-rust; 1.0.0; my-user-agent");
    }

    #[test]
    fn test_http_user_agent_no_library_name() {
        let library_name = None;
        let library_version = Some("1.0.0");
        let user_agent = Some("my-user-agent");
        let result = http_user_agent(library_name, library_version, user_agent);
        assert_eq!(result, "1.0.0; my-user-agent");
    }

    #[test]
    fn test_http_user_agent_no_library_version() {
        let library_name = Some("aiha");
        let library_version = None;
        let user_agent = Some("my-user-agent");
        let result = http_user_agent(library_name, library_version, user_agent);
        assert_eq!(result, "aiha-rust; my-user-agent");
    }

    #[test]
    fn test_http_user_agent_no_user_agent() {
        let library_name = Some("aiha");
        let library_version = Some("1.0.0");
        let user_agent = None;
        let result = http_user_agent(library_name, library_version, user_agent);
        assert_eq!(result, "aiha-rust; 1.0.0");
    }

    #[test]
    fn test_deduplicate_user_agent() {
        let user_agent = "aiha-rust; 1.0.0; my-user-agent; aiha-rust";
        let result = deduplicate_user_agent(user_agent);
        assert_eq!(result, "aiha-rust; 1.0.0; my-user-agent");
    }

    #[test]
    fn test_deduplicate_user_agent_no_duplicates() {
        let user_agent = "aiha-rust; 1.0.0; my-user-agent";
        let result = deduplicate_user_agent(user_agent);
        assert_eq!(result, "aiha-rust; 1.0.0; my-user-agent");
    }

    #[tokio::test]
    async fn test_retrieve_model_info() {
        let repo_id = "EleutherAI/gpt-j-6b";
        let revision = Some("main");
        let timeout = Some(30.0);
        let files_metadata = Some(false);
        let token = Some("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");

        let result = retrieve_model_info(repo_id, revision, timeout, files_metadata, token).await;
        assert!(result.is_ok());

        let model_info = result.unwrap();
        assert_eq!(model_info.model_id, Some("EleutherAI/gpt-j-6b".to_string()));
        assert_eq!(model_info.tags, Some(
            vec![
                "pytorch".to_string(),
                "tf".to_string(),
                "jax".to_string(),
                "gptj".to_string(),
                "text-generation".to_string(),
                "en".to_string(),
                "dataset:the_pile".to_string(),
                "arxiv:2104.09864".to_string(),
                "arxiv:2101.00027".to_string(),
                "transformers".to_string(),
                "causal-lm".to_string(),
                "license:apache-2.0".to_string(),
                "has_space".to_string(),
            ]
        ));
        assert_eq!(model_info.pipeline_tag, Some("text-generation".to_string()));
        assert_eq!(
            model_info.siblings.unwrap().get_siblings(),
            vec![
                ".gitattributes",
                "README.md",
                "added_tokens.json",
                "config.json",
                "flax_model.msgpack",
                "merges.txt",
                "pytorch_model.bin",
                "special_tokens_map.json",
                "tf_model.h5",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json"
            ]
        );
        assert_eq!(model_info.config.as_ref().unwrap().architectures, vec!["GPTJForCausalLM".to_string()]);
        assert_eq!(model_info.config.as_ref().unwrap().model_type, "gptj".to_string());
        assert_eq!(
            model_info.security_status,
            Some(from_value(json!({
                "scansDone": null,
                "dangerousPickles": null,
                "hasUnsafeFile": false,
                "repositoryId": "models/EleutherAI/gpt-j-6b",
                "revision": "f98c709453c9402b1309b032f40df1c10ad481a2",
                "clamAVInfectedFiles": null,
            })).unwrap())
        );
    }
}
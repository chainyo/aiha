//! Module for interacting with Hugging Face Hub.
use std::collections::HashMap;
use percent_encoding::{ utf8_percent_encode, AsciiSet, CONTROLS };

/// The set of characters that are percent-encoded in the path segment of a URI.
const CUSTOM_ENCODE_SET: &AsciiSet = &CONTROLS.add(b' ').add(b'/').add(b':').add(b'@');
/// The Hugging Face Hub API endpoint
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

/// Struct for storing the model metadata
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

/// Struct for storing the model metadata
#[derive(Debug)]
pub struct ModelInfo {
    /// The model ID of the repository (e.g. `username/repo_name`)
    model_id: Option<String>,
    /// The repository sha at this revision
    sha: Option<String>,
    /// The last modified date of the repository
    last_modified: Option<String>,
    /// The associated tags of the repository
    tags: Option<Vec<String>>,
    /// The pipeline tag of the repository
    pipeline_tag: Option<String>,
    /// Whether the repository is private or not
    private: bool,
    /// The name of the author of the repository
    author: Option<String>,
    /// The config file associated with the repository
    config: Option<HashMap<String, String>>,
    /// The security status (e.g. `{"containsInfected": False}`)
    security_status: Option<HashMap<String, bool>>,
}

/// Implement the `ModelInfo` struct
impl ModelInfo {
    /// Create a new ModelInfo struct
    pub fn new(
        model_id: Option<String>,
        sha: Option<String>,
        last_modified: Option<String>,
        tags: Option<Vec<String>>,
        pipeline_tag: Option<String>,
        private: bool,
        author: Option<String>,
        config: Option<HashMap<String, String>>,
        security_status: Option<HashMap<String, bool>>,
    ) -> Self {
        Self {
            model_id,
            sha,
            last_modified,
            tags,
            pipeline_tag,
            private,
            author,
            config,
            security_status,
        }
    }

    /// Return a string representation of the ModelInfo struct
    fn to_string(&self) -> String {
        let mut result = format!("Model Name: {:?}", self.model_id);
        if let Some(tags) = &self.tags {
            result += &format!(", Tags: {:?}", tags);
        }
        if let Some(pipeline_tag) = &self.pipeline_tag {
            result += &format!(", Task: {:?}", pipeline_tag);
        }
        result
    }
}


/// Make a request to the Hugging Face Hub API to retrieve the model info
fn retrieve_model_info(
    repo_id: &str,
    revision: Option<&str>,
    timeout: Option<f32>,
    files_metadata: Option<bool>,
    token: Option<impl ToString>,
) -> ModelInfo {
    let path = if let Some(rev) = revision.as_ref() {
        let encoded_revision = utf8_percent_encode(rev, &CUSTOM_ENCODE_SET).to_string();
        format!("{}/api/models/{}/revision/{}", HUB_ENDPOINT, repo_id, encoded_revision)
    } else {
        format!("{}/api/models/{}", HUB_ENDPOINT, repo_id)
    };
    let mut params = HashMap::new();
    params.insert("securityStatus", "true");
    if files_metadata.unwrap_or(false) {
        params.insert("blobs", "true");
    }
    let mut headers = HashMap::new();
    

    // r = get_session().get(path, headers=headers, timeout=timeout, params=params)
    // hf_raise_for_status(r)
    // d = r.json()
    // return ModelInfo(**d)
    ModelInfo::new(None, None, None, None, None, false, None, None, None)
}

#[cfg(test)]
mod tests {
}
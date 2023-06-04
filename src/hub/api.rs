//! Module for interacting with Hugging Face Hub.
use std::collections::HashMap;
use std::error::Error;

use percent_encoding::utf8_percent_encode;
use reqwest::Client;
use serde_json::json;
use tokio::time::Duration;

use crate::hub::{
    ModelConfig,
    ModelInfo,
    Siblings,
    CUSTOM_ENCODE_SET,
    HUB_ENDPOINT,
    build_headers,
};
use crate::models::ModelConfigTrait;

/// Make a request to the Hugging Face Hub API to retrieve the model info
pub async fn retrieve_model_info(
    repo_id: &str,
    revision: Option<&str>,
    timeout: Option<f32>,
    files_metadata: Option<bool>,
    token: Option<&str>,
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

    let headers = build_headers(token)?;

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
    let model_info = ModelInfo::from_json(response_json);
    Ok(model_info)
}

/// Make a request to the Hugging Face Hub API to retrieve specific files info for a model
pub async fn list_files_info(
    repo_id: &str,
    revision: Option<&str>,
    siblings: &mut Siblings,
    token: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let path = if let Some(rev) = revision.as_ref() {
        let encoded_revision = utf8_percent_encode(rev, CUSTOM_ENCODE_SET).to_string();
        format!("{}/api/models/{}/paths-info/{}", HUB_ENDPOINT, repo_id, encoded_revision)
    } else {
        format!("{}/api/models/{}/paths-info/main", HUB_ENDPOINT, repo_id)
    };
    let headers = build_headers(token)?;
    let data = json!({
        "paths": siblings.get_sibling_names(),
        "expand": true
    });

    let client = Client::new();
    let response = client.post(path)
        .headers(headers)
        .json(&data)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    if let Some(response_files) = response.as_array() {
        for item in response_files.iter() {
            if let Some(
                existing_model_file
            ) = siblings.siblings
                    .iter_mut()
                    .find(|file| file.get_rfilename() == item["path"].as_str().unwrap()) {
                        existing_model_file.size = item["size"].as_i64();
                        existing_model_file.oid = item["oid"].as_str().map(|s| s.to_string());
                    } else {
                        continue;
                    }
        }
    }
    Ok(())
}

/// Get the model config file from the Hugging Face Hub API and store it in the ModelInfo struct
pub async fn get_model_config(
    repo_id: &str,
    revision: Option<&str>,
    model_config: &mut Option<ModelConfig>,
    token: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let path = if let Some(rev) = revision.as_ref() {
        let encoded_revision = utf8_percent_encode(rev, CUSTOM_ENCODE_SET).to_string();
        format!("{}/{}/raw/{}/config.json", HUB_ENDPOINT, repo_id, encoded_revision)
    } else {
        format!("{}/{}/raw/main/config.json", HUB_ENDPOINT, repo_id)
    };
    let headers = build_headers(token)?;

    let client = Client::new();
    let response = client.get(path)
        .headers(headers)
        .send()
        .await?;

    let response_json = response.json::<serde_json::Value>().await?;
    let _config = ModelConfig::from_json(response_json);
    if let Ok(config) = _config {
        *model_config = Some(config);
    } else {
        // handle error, set `model_info.config` to `None`
        *model_config = None;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use pretty_assertions::assert_eq;
    use serde_json::from_value;

    use crate::hub::{ModelFile, Siblings};

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
            model_info.siblings.as_ref().unwrap(),
            &Siblings::new(vec![
                ModelFile::new(".gitattributes".to_string(), None, None),
                ModelFile::new("README.md".to_string(), None, None),
                ModelFile::new("added_tokens.json".to_string(), None, None),
                ModelFile::new("config.json".to_string(), None, None),
                ModelFile::new("flax_model.msgpack".to_string(), None, None),
                ModelFile::new("merges.txt".to_string(), None, None),
                ModelFile::new("pytorch_model.bin".to_string(), None, None),
                ModelFile::new("special_tokens_map.json".to_string(), None, None),
                ModelFile::new("tf_model.h5".to_string(), None, None),
                ModelFile::new("tokenizer.json".to_string(), None, None),
                ModelFile::new("tokenizer_config.json".to_string(), None, None),
                ModelFile::new("vocab.json".to_string(), None, None),
            ])
        );
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

    #[tokio::test]
    async fn test_list_files_info() {
        let repo_id = "EleutherAI/gpt-j-6b";
        let revision = Some("main");
        let timeout = Some(30.0);
        let files_metadata = Some(false);
        let token = Some("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");

        let result = retrieve_model_info(
            repo_id, revision, timeout, files_metadata, token
        ).await;
        assert!(result.is_ok());
        let mut model_info = result.unwrap();

        let result_list = list_files_info(
            model_info.model_id.as_ref().unwrap(),
            revision,
            model_info.siblings.as_mut().unwrap(),
            token
        ).await;
        assert!(result_list.is_ok());

        // Check that for each file, the size and oid are set
        for file in model_info.siblings.as_ref().unwrap().siblings.iter() {
            assert!(file.size.is_some());
            assert!(file.oid.is_some());
        }
    }

    #[tokio::test]
    async fn test_get_model_config() {
        let repo_id = "EleutherAI/gpt-j-6b";
        let revision = Some("main");
        let timeout = Some(30.0);
        let files_metadata = Some(false);
        let token = Some("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");

        let result = retrieve_model_info(
            repo_id, revision, timeout, files_metadata, token
        ).await;
        assert!(result.is_ok());
        let mut model_info = result.unwrap();

        let get_config = get_model_config(
            model_info.model_id.as_ref().unwrap(),
            revision,
            &mut model_info.config,
            token
        ).await;
        assert!(get_config.is_ok());

        // Check that the config is set
        assert!(model_info.config.is_some());
    }
}
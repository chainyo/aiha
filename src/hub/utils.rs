//! Utils for Hub interactions
use std::collections::HashSet;
use std::error::Error;

use percent_encoding::{AsciiSet, CONTROLS};
use reqwest::header::HeaderMap;

/// This set is used to encode the path of the model id
pub const CUSTOM_ENCODE_SET: &AsciiSet = &CONTROLS.add(b' ').add(b'/').add(b':').add(b'@');
/// The default endpoint for the Hugging Face Hub
pub const HUB_ENDPOINT: &str = "https://huggingface.co";

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

/// Build the headers for the request
pub fn build_headers(token: Option<&str>) -> Result<HeaderMap, Box<dyn Error>> {
    let mut headers = HeaderMap::new();
    let _user_agent = deduplicate_user_agent(
        http_user_agent(
            Some(env!("CARGO_PKG_NAME")),
            Some(env!("CARGO_PKG_VERSION")),
            None
        ).as_str()
    );
    headers.insert("user-agent", _user_agent.parse()?);
    match token {
        Some(t) => {
            headers.insert("authorization", format!("Bearer {}", t).parse()?);
            Ok(headers)
        },
        None => Err("No token provided".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

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
}
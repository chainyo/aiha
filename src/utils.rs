use reqwest::Url;
use std::fs::File;
use std::io::copy;

/// Given a model path, download the config file from the HuggingFace Hub.
/// 
/// # Arguments
///
/// * `model_path` - A string slice that holds the model path.
/// * `save_path` - A string slice that holds the save path.
/// 
/// # Example
/// 
/// ```
/// use aiha::config::utils::download_model_config;
/// 
/// let model_path = "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json";
/// let save_path = "config.json";
/// let result = download_model_config(model_path, save_path);
/// assert!(result.is_ok());
/// ```
/// 
/// # Errors
/// 
/// * `Box<dyn std::error::Error>` - If the model path is invalid.
/// * `Box<dyn std::error::Error>` - If the save path is invalid.
/// * `Box<dyn std::error::Error>` - If the model config file cannot be downloaded.
/// 
pub fn download_model_config(model_path: &str, save_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url = Url::parse(model_path)?;
    let mut response = reqwest::blocking::get(url)?;
    let mut out = File::create(save_path)?;
    copy(&mut response, &mut out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test for the download_model_config() function
    #[test]
    fn test_download_model_config() {
        let model_path = "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json";
        let result = download_model_config(model_path, "config.json");
        assert!(result.is_ok());
    }
}
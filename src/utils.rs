use reqwest::Url;
use std::error::Error;
use std::fs::File;
use std::io::copy;
use std::path::PathBuf;

/// Given a model path, download the config file from the HuggingFace Hub.
pub fn download_model_config(model_path: &str, save_path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let url = Url::parse(model_path)?;
    let mut response = reqwest::blocking::get(url)?;
    let mut out = File::create(save_path)?;
    copy(&mut response, &mut out)?;
    Ok(())
}

/// Given a file path, check if the file is local.
pub fn is_file_local(file_path: &str) -> bool {
    std::path::Path::new(file_path).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // Test for the download_model_config() function
    #[test]
    fn test_download_model_config() {
        let model_path = "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json";
        let temp_dir = tempdir().unwrap();
        let save_path = temp_dir.path().join("config.json");
        
        let result = download_model_config(model_path, &save_path);
        assert!(result.is_ok());
    }

    // Test for the is_file_local() function
    #[test]
    fn test_is_file_local() {
        let file_path = "tests/data/config.json";
        let result = is_file_local(file_path);
        assert_eq!(result, true);
    }
}
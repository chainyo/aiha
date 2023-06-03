//! Model File metadata struct
use std::fmt;

use serde::Deserialize;

/// Struct for storing the model file metadata
#[derive(Clone, Debug, Deserialize)]
pub struct ModelFile {
    /// The filename of the model file
    pub rfilename: String,
    /// The size of the model file
    pub size: Option<i32>,
    /// The file git OID
    pub oid: Option<String>,
}

/// Implement the `ModelFile` struct
impl ModelFile {
    /// Create a new ModelFile struct
    pub fn new(rfilename: String, size: Option<i32>, oid: Option<String>) -> Self {
        Self {
            rfilename,
            size,
            oid,
        }
    }
    /// Retrieve the filename of the model file
    pub fn get_rfilename(&self) -> &'_ String {
        &self.rfilename
    }
    /// Retrieve the size of the model file
    pub fn get_size(&self) -> Option<i32> {
        self.size
    }
    /// Retrieve the file git OID
    pub fn get_oid(&self) -> Option<&'_ String> {
        self.oid.as_ref()
    }
}

/// Implement partial equality for the ModelFile struct
impl PartialEq for ModelFile {
    fn eq(&self, other: &Self) -> bool {
        self.rfilename == other.rfilename
            && self.size == other.size
            && self.oid == other.oid
    }
}

/// Implement the display of the ModelFile struct
impl fmt::Display for ModelFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model File: {:?}", self.rfilename)?;
        if let Some(size) = &self.size {
            write!(f, ", Size: {:?}", size)?;
        }
        if let Some(blob_id) = &self.oid {
            write!(f, ", Blob ID: {:?}", blob_id)?;
        }
        Ok(())
    }
}

/// Implement the creation of the ModelFile struct from a serde_json::Value
impl From<serde_json::Value> for ModelFile {
    fn from(response_json: serde_json::Value) -> Self {
        ModelFile::new(
            response_json["rfilename"].as_str().map(|s| s.to_string()).unwrap_or_default(),
            serde_json::from_value(response_json["size"].clone()).unwrap_or_default(),
            serde_json::from_value(response_json["oid"].clone()).unwrap_or_default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn test_modelfile_impl() {
        let rfilename = "rfilename".to_string();
        let size = Some(1);
        let oid = Some("oid".to_string());
        let modelfile = ModelFile::new(rfilename.clone(), size, oid.clone());
        assert_eq!(modelfile.rfilename, rfilename);
        assert_eq!(modelfile.size, size);
        assert_eq!(modelfile.oid, oid);

        assert_eq!(modelfile.get_rfilename(), &rfilename);
        assert_eq!(modelfile.get_size(), size);
    }

    #[test]
    fn test_modelfile_partial_eq() {
        let rfilename = "rfilename".to_string();
        let size = Some(1);
        let oid = Some("oid".to_string());
        let modelfile = ModelFile::new(rfilename.clone(), size, oid.clone());
        let modelfile2 = ModelFile::new(rfilename, size, oid);
        assert_eq!(modelfile, modelfile2);
    }

    #[test]
    fn test_modelfile_display() {
        let rfilename = "rfilename".to_string();
        let size = Some(1);
        let oid = Some("oid".to_string());
        let modelfile = ModelFile::new(rfilename, size, oid);
        assert_eq!(
            format!(
                "Model File: {:?}, Size: {:?}, Blob ID: {:?}",
                "rfilename", 1, "blob_id"
            ),
            format!("{}", modelfile)
        );
    }

    #[test]
    fn test_modelfile_from_value() {
        let rfilename = "rfilename".to_string();
        let size = Some(1);
        let oid = Some("oid".to_string());
        let response_json = json!({
            "name": rfilename,
            "size": size.clone(),
            "oid": oid,
        });
        let modelfile = ModelFile::from(response_json);
        assert_eq!(modelfile.rfilename, rfilename);
        assert_eq!(modelfile.size, size);
        assert_eq!(modelfile.oid, oid);
    }
}

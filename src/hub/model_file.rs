//! Model File metadata struct
use std::collections::HashMap;
use std::fmt;

use serde::Deserialize;

use crate::hub::LfsInfo;

/// Struct for storing the model file metadata
#[derive(Clone, Debug, Deserialize)]
pub struct ModelFile {
    /// The filename of the model file
    pub rfilename: String,
    /// The size of the model file
    pub size: Option<i32>,
    /// The file git OID
    pub blob_id: Option<String>,
    /// The LFS metadata
    pub lfs: Option<LfsInfo>,
}

/// Implement the `ModelFile` struct
impl ModelFile {
    /// Create a new ModelFile struct
    pub fn new(rfilename: String, size: Option<i32>, blob_id: Option<String>, lfs: Option<LfsInfo>) -> Self {
        Self {
            rfilename,
            size,
            blob_id,
            lfs,
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
}

/// Implement partial equality for the ModelFile struct
impl PartialEq for ModelFile {
    fn eq(&self, other: &Self) -> bool {
        self.rfilename == other.rfilename
            && self.size == other.size
            && self.blob_id == other.blob_id
            && self.lfs == other.lfs
    }
}

/// Implement the display of the ModelFile struct
impl fmt::Display for ModelFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model File: {:?}", self.rfilename)?;
        if let Some(size) = &self.size {
            write!(f, ", Size: {:?}", size)?;
        }
        if let Some(blob_id) = &self.blob_id {
            write!(f, ", Blob ID: {:?}", blob_id)?;
        }
        if let Some(lfs) = &self.lfs {
            write!(f, ", LFS: {:?}", lfs)?;
        }
        Ok(())
    }
}

/// Implement the creation of the ModelFile struct from a serde_json::Value
impl From<serde_json::Value> for ModelFile {
    fn from(response_json: serde_json::Value) -> Self {
        let lfs = if response_json["lfs"].is_object() {
            let _lfs: HashMap<String, serde_json::Value> = serde_json::from_value(response_json["lfs"].clone()).unwrap_or_default();
            let pointer_size = if _lfs.contains_key("pointer_size") {
                serde_json::from_value(_lfs["pointer_size"].clone()).unwrap_or_default()
            } else {
                None
            };
            Some(LfsInfo::new(
                serde_json::from_value(_lfs["size"].clone()).unwrap_or_default(),
                serde_json::from_value(_lfs["oid"].clone()).unwrap_or_default(),
                pointer_size,
            ))
        } else {
            None
        };
        ModelFile::new(
            response_json["name"].as_str().map(|s| s.to_string()).unwrap_or_default(),
            serde_json::from_value(response_json["size"].clone()).unwrap_or_default(),
            serde_json::from_value(response_json["blob_id"].clone()).unwrap_or_default(),
            lfs,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_modelfile_impl() {
        let rfilename = "rfilename".to_string();
        let size = Some(1);
        let blob_id = Some("blob_id".to_string());
        let lfs = Some(LfsInfo::new(1, "sha256".to_string(), Some(2)));
        let modelfile = ModelFile::new(rfilename.clone(), size.clone(), blob_id.clone(), lfs.clone());
        assert_eq!(modelfile.rfilename, rfilename);
        assert_eq!(modelfile.size, size);
        assert_eq!(modelfile.blob_id, blob_id);
        assert_eq!(modelfile.lfs, lfs);

        assert_eq!(modelfile.get_rfilename(), &rfilename);
        assert_eq!(modelfile.get_size(), size);
    }

    #[test]
    fn test_modelfile_partial_eq() {
        let rfilename = "rfilename".to_string();
        let size = Some(1);
        let blob_id = Some("blob_id".to_string());
        let lfs = Some(LfsInfo::new(1, "sha256".to_string(), Some(2)));
        let modelfile = ModelFile::new(rfilename.clone(), size.clone(), blob_id.clone(), lfs.clone());
        let modelfile2 = ModelFile::new(rfilename.clone(), size.clone(), blob_id.clone(), lfs.clone());
        assert_eq!(modelfile, modelfile2);
    }

    #[test]
    fn test_modelfile_display() {
        let rfilename = "rfilename".to_string();
        let size = Some(1);
        let blob_id = Some("blob_id".to_string());
        let lfs = Some(LfsInfo::new(1, "sha256".to_string(), Some(2)));
        let modelfile = ModelFile::new(rfilename.clone(), size.clone(), blob_id.clone(), lfs.clone());
        assert_eq!(
            format!(
                "Model File: {:?}, Size: {:?}, Blob ID: {:?}, LFS: {:?}",
                "rfilename", 1, "blob_id", lfs.unwrap()
            ),
            format!("{}", modelfile)
        );
    }

    #[test]
    fn test_modelfile_from_value() {
        let rfilename = "rfilename".to_string();
        let size = Some(1);
        let blob_id = Some("blob_id".to_string());
        let lfs = Some(LfsInfo::new(1, "sha256".to_string(), Some(2)));
        let response_json = json!({
            "name": rfilename.clone(),
            "size": size.clone(),
            "blob_id": blob_id.clone(),
            "lfs": {
                "size": 1,
                "oid": "sha256",
                "pointer_size": 2
            }
        });
        let modelfile = ModelFile::from(response_json);
        assert_eq!(modelfile.rfilename, rfilename);
        assert_eq!(modelfile.size, size);
        assert_eq!(modelfile.blob_id, blob_id);
        assert_eq!(modelfile.lfs, lfs);
    }
}

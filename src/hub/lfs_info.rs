//! LFS metadata struct
use serde::Deserialize;

/// Struct for storing the LFS metadata
#[derive(Clone, Debug, Deserialize)]
pub struct LfsInfo {
    /// The size of the LFS file
    pub size: i32,
    /// The SHA256 hash of the LFS file
    pub sha256: String,
    /// The pointer size of the LFS file
    pub pointer_size: Option<i32>,
}

/// Implement the `LfsInfo` struct
impl LfsInfo {
    /// Create a new LfsInfo struct
    pub fn new(size: i32, sha256: String, pointer_size: Option<i32>) -> Self {
        Self {
            size,
            sha256,
            pointer_size,
        }
    }
}

/// Implement partial equality for the LfsInfo struct
impl PartialEq for LfsInfo {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self.sha256 == other.sha256
            && self.pointer_size == other.pointer_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_lfsinfo_new() {
        let size = 1;
        let sha256 = "sha256".to_string();
        let pointer_size = Some(2);
        let lfsinfo = LfsInfo::new(size, sha256.clone(), pointer_size);
        assert_eq!(lfsinfo.size, size);
        assert_eq!(lfsinfo.sha256, sha256);
        assert_eq!(lfsinfo.pointer_size, pointer_size);
    }

    #[test]
    fn test_lfsinfo_partial_eq() {
        let size = 1;
        let sha256 = "sha256".to_string();
        let pointer_size = Some(2);
        let lfsinfo = LfsInfo::new(size, sha256.clone(), pointer_size);
        let lfsinfo2 = LfsInfo::new(size, sha256, pointer_size);
        assert_eq!(lfsinfo, lfsinfo2);
    }
}
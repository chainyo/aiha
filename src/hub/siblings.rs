//! Siblings metadata struct
use serde::Deserialize;

use crate::hub::ModelFile;

/// Struct that represent a list of siblings of a model
#[derive(Clone, Debug, Deserialize)]
pub struct Siblings {
    /// The list of siblings of a model
    pub siblings: Vec<ModelFile>,
}

/// Implement the `Siblings` struct
impl Siblings {
    /// Create a new Siblings struct
    pub fn new(siblings: Vec<ModelFile>) -> Self {
        Self { siblings }
    }
    /// Get the list of siblings as a vector of strings
    pub fn get_sibling_names(&self) -> Vec<&'_ String> {
        self.siblings.iter().map(|s| s.get_rfilename()).collect()
    }
}

/// Implement the partial equality for the `Siblings` struct
impl PartialEq for Siblings {
    fn eq(&self, other: &Self) -> bool {
        self.siblings == other.siblings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_siblings_new() {
        let siblings = vec![ModelFile::new(
            "model.json".to_string(),
            Some(1),
            Some("123".to_string()),
        )];
        let siblings = Siblings::new(siblings);
        assert_eq!(siblings.siblings.len(), 1);
    }

    #[test]
    fn test_siblings_get_sibling_names() {
        let siblings = vec![
            ModelFile::new("model.json".to_string(), Some(1), Some("123".to_string())),
            ModelFile::new("model2.json".to_string(), Some(1), Some("123".to_string())),
            ModelFile::new("model3.json".to_string(), Some(1), Some("123".to_string())),
        ];
        let siblings = Siblings::new(siblings);
        let sibling_names = siblings.get_sibling_names();
        assert_eq!(sibling_names.len(), 3);
        assert_eq!(sibling_names[0], "model.json");
        assert_eq!(sibling_names[1], "model2.json");
        assert_eq!(sibling_names[2], "model3.json");
    }

    #[test]
    fn test_siblings_partial_eq() {
        let s1 = vec![ModelFile::new(
            "model.json".to_string(),
            Some(1),
            Some("123".to_string()),
        )];
        let siblings = Siblings::new(s1);
        let s2 = vec![ModelFile::new(
            "model.json".to_string(),
            Some(1),
            Some("123".to_string()),
        )];
        let siblings2 = Siblings::new(s2);
        assert_eq!(siblings, siblings2);
    }
}

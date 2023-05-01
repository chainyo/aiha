use pyo3::prelude::*;
use std::env::consts::{ARCH, OS};
use std::fs;
use std::path::PathBuf;

#[pyclass]
pub struct Scan {}

#[pymethods]
impl Scan {
    /// Returns a string representing the operating system the code is running on.
    /// 
    /// # Example
    /// 
    /// ```
    /// use aiha::scan::Scan;
    /// 
    /// let os = Scan::os_scan();
    /// println!("{}", os);
    /// ```
    #[staticmethod]
    pub fn os_scan() -> PyResult<String> {
        Ok(OS.to_string())
    }

    /// Returns a string representing the CPU architecture the code is running on.
    /// 
    /// # Example
    /// 
    /// ```
    /// use aiha::scan::Scan;
    /// 
    /// let arch = Scan::arch_scan();
    /// println!("{}", arch);
    /// ```
    #[staticmethod]
    pub fn arch_scan() -> PyResult<String> {
        Ok(ARCH.to_string())
    }

    /// Returns a string representing the user's home directory.
    /// Unix systems use the $HOME environment variable,
    /// while Windows systems use the $USERPROFILE environment variable.
    #[staticmethod]
    fn get_home_dir() -> Option<PathBuf> {
        #[cfg(target_family = "unix")]
        {
            std::env::var("HOME").ok().map(PathBuf::from)
        }

        #[cfg(target_family = "windows")]
        {
            std::env::var("USERPROFILE").ok().map(PathBuf::from)
        }
    }

    /// Returns a string representing the user's home directory.
    #[staticmethod]
    pub fn home_dir() -> PyResult<Option<String>> {
        match Scan::get_home_dir() {
            Some(path) => Ok(Some(path.to_str().unwrap().to_string())),
            None => Ok(None),
        }
    }

    /// Create the AIHA cache folder in the user's home directory if it doesn't exist.
    /// 
    /// # Example
    /// 
    /// ```
    /// use aiha::scan::Scan;
    /// 
    /// Scan::create_aiha_cache_folder();
    /// ```
    #[staticmethod]
    fn create_aiha_cache_folder() -> std::io::Result<()> {
        // Create the AIHA cache folder in the user's home directory if it doesn't exist
        if let Some(mut cache_dir) = Scan::get_home_dir() {
            cache_dir.push(".cache");
            cache_dir.push("aiha");

            if !cache_dir.exists() {
                fs::create_dir_all(&cache_dir)?;
            }
        } else {
            eprintln!("Failed to get home directory");
        }
        Ok(())
    }

    /// Return the path to the AIHA cache folder in the user's home directory.
    /// If the folder doesn't exist, create it.
    /// 
    /// # Example
    /// 
    /// ```
    /// use aiha::scan::Scan;
    /// 
    /// let cache_dir = Scan::get_aiha_cache_folder();
    /// println!("{}", cache_dir);
    /// ```
    #[staticmethod]
    fn get_aiha_cache_folder() -> Option<PathBuf> {
        if let Err(e) = Scan::create_aiha_cache_folder() {
            eprintln!("Failed to create AIHA cache folder: {}", e);
        }

        if let Some(mut cache_dir) = Scan::get_home_dir() {
            cache_dir.push(".cache");
            cache_dir.push("aiha");
            Some(cache_dir)
        } else {
            None
        }
    }

    /// Create the folders for one model in the AIHA cache folder.
    #[staticmethod]
    pub fn create_model_cache_folder(model_name: &str) -> std::io::Result<()> {
        // If model_name contains a / then we need to create an organization folder
        // and a model folder, otherwise we just need to create a model folder
        if model_name.contains("/") {
            let mut model_path = Scan::get_aiha_cache_folder().unwrap();
            let model_name_split: Vec<&str> = model_name.split("/").collect();
            let org_name = model_name_split[0];
            let model_name = model_name_split[1];

            model_path.push(org_name);
            model_path.push(model_name);

            if !model_path.exists() {
                fs::create_dir_all(&model_path)?;
            }
        } else {
            let mut model_path = Scan::get_aiha_cache_folder().unwrap();
            model_path.push(model_name);

            if !model_path.exists() {
                fs::create_dir_all(&model_path)?;
            }
        }
        Ok(())
    }

    /// Clean the AIHA cache folder.
    #[staticmethod]
    pub fn clean_cache() -> std::io::Result<()> {
        if let Some(cache_dir) = Scan::get_aiha_cache_folder() {
            fs::remove_dir_all(&cache_dir)?;
        } else {
            eprintln!("Failed to get AIHA cache folder");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test for the os_scan() method of the Scan struct
    #[test]
    fn test_os_scan() {
        let os = Scan::os_scan();
        assert!(os.is_ok());
    }

    // Test for the arch_scan() method of the Scan struct
    #[test]
    fn test_arch_scan() {
        let arch = Scan::arch_scan();
        assert!(arch.is_ok());
    }

    // Test for the home_dir() method of the Scan struct
    #[test]
    fn test_home_dir() {
        let home_dir = Scan::home_dir();
        assert!(home_dir.is_ok());
    }

    // Test for the get_aiha_cache_folder() method of the Scan struct
    #[test]
    fn test_get_aiha_cache_folder() {
        let cache_dir = Scan::get_aiha_cache_folder();
        assert!(cache_dir.is_some());
    }

    // Test for the create_aiha_cache_folder() method of the Scan struct
    #[test]
    fn test_create_aiha_cache_folder() {
        let cache_dir = Scan::create_aiha_cache_folder();
        assert!(cache_dir.is_ok());
    }

    // Test for the get_home_dir() method of the Scan struct
    #[test]
    fn test_get_home_dir() {
        let home_dir = Scan::get_home_dir();
        assert!(home_dir.is_some());
    }

    // Test for the create_model_cache_folder() method of the Scan struct
    #[test]
    fn test_create_model_cache_folder_no_org() {
        let model_name = "test_model";
        let model_cache_folder = Scan::create_model_cache_folder(model_name);
        assert!(model_cache_folder.is_ok());
    }

    // Test for the create_model_cache_folder() method of the Scan struct
    #[test]
    fn test_create_model_cache_folder_with_org() {
        let model_name = "test_org/test_model";
        let model_cache_folder = Scan::create_model_cache_folder(model_name);
        assert!(model_cache_folder.is_ok());
    }

    // Test for the clean_cache() method of the Scan struct
    #[test]
    fn test_clean_cache() {
        let clean_cache = Scan::clean_cache();
        assert!(clean_cache.is_ok());
    }
}


use pyo3::prelude::*;
use std::env::consts::{ARCH, OS};
use std::path::PathBuf;

#[pyclass]
pub struct Scan {}

#[pymethods]
impl Scan {
    #[new]
    pub fn new() -> Self {
        Scan {}
    }
    /// Returns a string representing the operating system the code is running on.
    ///
    /// # Examples
    ///
    /// ```
    /// let scanner = Scanner;
    /// let os = scanner.os_scan();
    /// println!("Operating system: {}", os);
    /// ```
    #[pyo3(text_signature = "($self)")]
    pub fn os_scan(&self) -> PyResult<String> {
        Ok(OS.to_string())
    }

    /// Returns a string representing the CPU architecture the code is running on.
    ///
    /// # Examples
    ///
    /// ```
    /// let scanner = Scanner;
    /// let arch = scanner.arch_scan();
    /// println!("CPU architecture: {}", arch);
    /// ```
    #[pyo3(text_signature = "($self)")]
    pub fn arch_scan(&self) -> PyResult<String> {
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
    #[pyo3(text_signature = "($self)")]
    pub fn home_dir(&self) -> PyResult<Option<String>> {
        match Scan::get_home_dir() {
            Some(path) => Ok(Some(path.to_str().unwrap().to_string())),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test for the new() method of the Scan struct
    #[test]
    fn test_scan_new() {
        let scan = Scan::new();
        assert!(scan.os_scan().is_ok());
        assert!(scan.arch_scan().is_ok());
    }

    // Test for the os_scan() method of the Scan struct
    #[test]
    fn test_scan_os_scan() {
        let scan = Scan::new();
        let os_pyresult = scan.os_scan();
        assert!(os_pyresult.is_ok());
        let os_string = os_pyresult.unwrap();
        assert_eq!(os_string, OS);
    }

    // Test for the arch_scan() method of the Scan struct
    #[test]
    fn test_scan_arch_scan() {
        let scan = Scan::new();
        let arch_pyresult = scan.arch_scan();
        assert!(arch_pyresult.is_ok());
        let arch_string = arch_pyresult.unwrap();
        assert_eq!(arch_string, ARCH);
    }

    // Test for the get_home_dir() method of the Scan struct
    #[test]
    fn test_scan_get_home_dir() {
        let home_dir = Scan::get_home_dir();
        assert!(home_dir.is_some());
    }
}


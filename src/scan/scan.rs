use std::env::consts::OS;
use std::env::consts::ARCH;

/// A struct that provides scanning functionality for the user hardware.
pub struct Scan;

impl Scan {
    /// Returns a string representing the operating system the code is running on.
    ///
    /// # Examples
    ///
    /// ```
    /// let scanner = Scanner;
    /// let os = scanner.os_scan();
    /// println!("Operating system: {}", os);
    /// ```
    pub fn os_scan(&self) -> String {
        OS.to_string()
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
    pub fn arch_scan(&self) -> String {
        ARCH.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_os_scan() {
        let scan = Scan;
        let os = scan.os_scan();
        assert_eq!(os, OS.to_string());
    }

    #[test]
    fn test_arch_scan() {
        let scan = Scan;
        let arch = scan.arch_scan();
        assert_eq!(arch, ARCH.to_string());
    }
}
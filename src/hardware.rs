//! Module for analyzing the hardware of the running system.
use num_cpus;
use rust_gpu_tools::Device;
use sys_info;

/// Returns the operating system of the running system.
pub fn scan_arch() -> String {
    let os = sys_info::os_type().unwrap();
    os
}

/// Returns the architecture of the running system.
pub fn scan_release() -> String {
    let release = sys_info::os_release().unwrap();
    release
}

/// Returns the number of physical cores of the running system.
pub fn scan_cpu_cores() -> u16 {
    let cores = num_cpus::get_physical();
    cores as u16
}

/// Returns the number of logical cores of the running system.
pub fn scan_cpu_threads() -> u16 {
    let threads = num_cpus::get();
    threads as u16
}

/// Returns the number of GPUs of the running system.
pub fn scan_gpu_count() -> u16 {
    //  
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_arch() {
        let arch = scan_arch();
        assert_eq!(arch, sys_info::os_type().unwrap());
    }

    #[test]
    fn test_scan_release() {
        let release = scan_release();
        assert_eq!(release, sys_info::os_release().unwrap());
    }

    #[test]
    fn test_scan_cpu_cores() {
        let cores = scan_cpu_cores();
        assert_eq!(cores, num_cpus::get_physical() as u16);
    }

    #[test]
    fn test_scan_cpu_threads() {
        let threads = scan_cpu_threads();
        assert_eq!(threads, num_cpus::get() as u16);
    }

    #[test]
    fn test_scan_gpu_count() {
        let _gpu_count = scan_gpu_count();
    }
}
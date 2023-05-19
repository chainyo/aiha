//! Module for analyzing the hardware of the running system.
use num_cpus;
use nvml_wrapper::Nvml;


/// Returns the operating system of the running system.
pub fn scan_os() -> String {
    let os = std::env::consts::OS.to_string();
    os
}

/// Returns the architecture of the running system.
pub fn scan_arch() -> String {
    let arch = std::env::consts::ARCH.to_string();
    arch
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

/// Returns the number of available GPUs of the running system.
pub fn scan_gpu_count(os: &str, arch: &str) -> Result<u32, String> {
    match (os, arch) {
        ("linux", _) => _scan_gpu_count(),
        ("windows", _) => _scan_gpu_count(),
        ("macos", arch) if arch != "aarch64" => _scan_gpu_count(),
        _ => Err("GPU scan is only supported on Linux, Windows, and macOS (excluding Apple Silicon).".to_string())
    }
}

/// Returns the number of available GPUs of the running system for NVIDIA GPUs.
fn _scan_gpu_count() -> Result<u32, String> {
    let nvml = Nvml::init().map_err(|e| e.to_string())?;
    let devices = nvml.device_count().map_err(|e| e.to_string())?;
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_os() {
        let arch = scan_os();
        assert_eq!(arch, std::env::consts::OS.to_string());
    }

    #[test]
    fn test_scan_arch() {
        let arch = scan_arch();
        assert_eq!(arch, std::env::consts::ARCH.to_string());
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
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let gpu_count = scan_gpu_count(&os, &arch);
        if os == "linux" || os == "windows" || (os == "macos" && arch != "aarch64") {
            assert!(gpu_count.is_ok());
        } else {
            assert!(gpu_count.is_err());
        }
    }
}
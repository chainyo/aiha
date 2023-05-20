//! Module for analyzing the hardware of the running system.
use num_cpus;
use nvml_wrapper::Nvml;
use nvml_wrapper::enums::device::DeviceArchitecture;
use nvml_wrapper::enum_wrappers::device::Brand;
use nvml_wrapper::structs::device::CudaComputeCapability;


/// Struct for storing the hardware information of the running system.
#[derive(Debug)]
pub struct Hardware {
    pub os: String,
    pub arch: String,
    pub cpu_cores: u16,
    pub cpu_threads: u16,
    pub gpu_count: u32,
    pub nvidia_gpus: Vec<NvidiaDevice>,
}

/// Trait for GPU devices that provides a method to obtain all information as a string.
pub trait GPUDevice {
    // Return a string with all information of the GPU device.
    fn get_info_string(&self) -> String;
    // Returns the memory_info of the GPU device.
    fn get_memory_info(&self) -> u64;
    // Returns the memory_info of the GPU device formatted as a string.
    fn get_memory_info_formatted(&self) -> String;
    // Returns the compute_capability of the GPU device formatted as a string.
    fn get_compute_capability_formatted(&self) -> String;
}

/// Struct for storing the GPU information of the running system.
#[derive(Debug)]
pub struct NvidiaDevice {
    architecture: DeviceArchitecture,
    brand: Brand,
    cuda_compute_capability: CudaComputeCapability,
    memory_info: u64,
    name: String,
    num_cores: u32,
    uuid: String,
}

/// Implementation of GPUDevice for NvidiaDevice.
impl GPUDevice for NvidiaDevice {
    // Returns the memory_info of the GPU device.
    fn get_info_string(&self) -> String {
        format!(
            "uuid: {}\nname: NVIDIA {}\narchitecture: {:?}\nbrand: {:?}\nmemory: {}\ncompute capability: {}\ncores: {}",
            self.uuid,
            self.name,
            self.architecture,
            self.brand,
            self.get_memory_info_formatted(),
            self.get_compute_capability_formatted(),
            self.num_cores,
        )
    }
    // Returns the memory_info of the GPU device.
    fn get_memory_info(&self) -> u64 {
        self.memory_info
    }
    // Returns the memory_info of the GPU device formatted as a string.
    fn get_memory_info_formatted(&self) -> String {
        let memory_info = self.memory_info;
        let memory_info = memory_info as f64;
        let memory_info = memory_info / 1024.0 / 1024.0 / 1024.0;
        format!("{:.2} GB", memory_info)
    }
    // Returns the compute_capability of the GPU device formatted as a string.
    fn get_compute_capability_formatted(&self) -> String {
        let compute_capability = &self.cuda_compute_capability;
        format!("{}.{}", compute_capability.major, compute_capability.minor)
    }
}

/// Scan the hardware of the running system and return a Hardware struct.
// TODO: Add support for AMD GPUs.
// TODO: Add support for Apple Silicon.
pub fn scan_hardware() -> Result<Hardware, String> {
    // Get the operating system, architecture, and CPU information.
    let os = scan_os();
    let arch = scan_arch();
    let cpu_cores = scan_cpu_cores();
    let cpu_threads = scan_cpu_threads();
    // Get the number of available GPUs or return an error.
    let nvml = Nvml::init().map_err(|e| e.to_string())?;
    let gpu_count = scan_gpu_count(&os, &arch, &nvml)?;
    // If gpu_count is 0, then the system does not have any NVIDIA GPUs,
    // so we can return the Hardware struct. Otherwise, we need to get
    // the information for each GPU.
    let nvidia_gpus = if gpu_count > 0 {
        (0..gpu_count).map(|i| {
            // Get the information for the GPU at index i.
            let device = nvml.device_by_index(i).map_err(|e| e.to_string())?;
            let architecture = device.architecture().map_err(|e| e.to_string())?;
            let brand = device.brand().map_err(|e| e.to_string())?;
            let cuda_compute_capability = device.cuda_compute_capability().map_err(|e| e.to_string())?;
            let memory_info = device.memory_info().map_err(|e| e.to_string())?.total;
            let name = device.name().map_err(|e| e.to_string())?;
            let num_cores = device.num_cores().map_err(|e| e.to_string())?;
            let uuid = device.uuid().map_err(|e| e.to_string())?;
            // Return the NvidiaDevice struct.
            Ok(NvidiaDevice {
                architecture,
                brand,
                cuda_compute_capability,
                memory_info,
                name,
                num_cores,
                uuid,
            })
        }).collect::<Result<Vec<NvidiaDevice>, String>>()?
    } else {
        Vec::new()
    };
    // Add the NVIDIA GPUs to the Hardware struct.
    Ok(Hardware {
        os,
        arch,
        cpu_cores,
        cpu_threads,
        gpu_count,
        nvidia_gpus,
    })
}

/// Returns the operating system of the running system.
pub fn scan_os() -> String {
    std::env::consts::OS.to_string()
}

/// Returns the architecture of the running system.
pub fn scan_arch() -> String {
    std::env::consts::ARCH.to_string()
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
pub fn scan_gpu_count(os: &str, arch: &str, nvml: &Nvml) -> Result<u32, String> {
    match (os, arch) {
        ("linux", _) => _scan_gpu_count(nvml),
        ("windows", _) => _scan_gpu_count(nvml),
        ("macos", arch) if arch != "aarch64" => _scan_gpu_count(nvml),
        _ => Err("GPU scan is only supported on Linux, Windows, and macOS (excluding Apple Silicon).".to_string())
    }
}

/// Returns the number of available GPUs of the running system for NVIDIA GPUs.
fn _scan_gpu_count(nvml: &Nvml) -> Result<u32, String> {
    let devices = nvml.device_count().map_err(|e| e.to_string())?;
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Setup a Hardware struct for testing.
    fn setup_nvidia_device() -> NvidiaDevice {
        NvidiaDevice {
            architecture: DeviceArchitecture::Kepler,
            brand: Brand::Tesla,
            cuda_compute_capability: CudaComputeCapability { major: 3, minor: 7 },
            memory_info: 4096 * 1024 * 1024,
            name: "Tesla K80".to_string(),
            num_cores: 2496,
            uuid: "GPU-4c2b7f7c-0b7e-0e1a-1e1f-2f3e4d5e6f7g".to_string(),
        }
    }

    #[test]
    fn test_struct_hardware() {
        let hardware = Hardware {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            cpu_cores: 8,
            cpu_threads: 16,
            gpu_count: 1,
            nvidia_gpus: vec![
                setup_nvidia_device(),
            ],
        };
    
        assert_eq!(hardware.os, "linux".to_string());
        assert_eq!(hardware.arch, "x86_64".to_string());
        assert_eq!(hardware.cpu_cores, 8);
        assert_eq!(hardware.cpu_threads, 16);
        assert_eq!(hardware.gpu_count, 1);
        assert_eq!(hardware.nvidia_gpus.len(), 1);
    
        let nvidia_gpu = &hardware.nvidia_gpus[0];
        assert_eq!(nvidia_gpu.architecture, DeviceArchitecture::Kepler);
        assert_eq!(nvidia_gpu.brand, Brand::Tesla);
        assert_eq!(nvidia_gpu.cuda_compute_capability, CudaComputeCapability { major: 3, minor: 7 });
        assert_eq!(nvidia_gpu.memory_info, 4294967296);
        assert_eq!(nvidia_gpu.name, "Tesla K80".to_string());
        assert_eq!(nvidia_gpu.num_cores, 2496);
        assert_eq!(nvidia_gpu.uuid, "GPU-4c2b7f7c-0b7e-0e1a-1e1f-2f3e4d5e6f7g".to_string());
    }

    #[test]
    fn test_struct_nvidia_device() {
        let nvidia_device = setup_nvidia_device();
        assert_eq!(nvidia_device.architecture, DeviceArchitecture::Kepler);
        assert_eq!(nvidia_device.brand, Brand::Tesla);
        assert_eq!(nvidia_device.cuda_compute_capability, CudaComputeCapability { major: 3, minor: 7 });
        assert_eq!(nvidia_device.memory_info, 4294967296);
        assert_eq!(nvidia_device.name, "Tesla K80".to_string());
        assert_eq!(nvidia_device.num_cores, 2496);
        assert_eq!(nvidia_device.uuid, "GPU-4c2b7f7c-0b7e-0e1a-1e1f-2f3e4d5e6f7g".to_string());
    }

    #[test]
    fn test_nvidia_device_get_info_string() {
        let device = setup_nvidia_device();
        let expected_info_string = "uuid: GPU-4c2b7f7c-0b7e-0e1a-1e1f-2f3e4d5e6f7g\nname: NVIDIA Tesla K80\narchitecture: Kepler\nbrand: Tesla\nmemory: 4.00 GB\ncompute capability: 3.7\ncores: 2496";
        assert_eq!(device.get_info_string(), expected_info_string);
    }

    #[test]
    fn test_nvidia_device_get_memory_info() {
        let device = setup_nvidia_device();
        let expected_info = 4294967296;
        assert_eq!(device.get_memory_info(), expected_info);
    }

    #[test]
    fn test_nvidia_device_get_memory_info_string() {
        let device = setup_nvidia_device();
        let expected_info_string = "4.00 GB".to_string();
        assert_eq!(device.get_memory_info_formatted(), expected_info_string);
    }

    #[test]
    fn test_scan_hardware() {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let hardware = scan_hardware();
        if os == "linux" || os == "windows" || (os == "macos" && arch != "aarch64") {
            assert!(hardware.is_ok());
        } else {
            assert!(hardware.is_err());
        }
    }

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
        let nvml = Nvml::init();
        if nvml.is_err() {
            println!("Skipping test: NVML initialization failed.");
            return;
        }
        let gpu_count = scan_gpu_count(&os, &arch, &nvml.unwrap());
        if os == "linux" || os == "windows" || (os == "macos" && arch != "aarch64") {
            assert!(gpu_count.is_ok());
        } else {
            assert!(gpu_count.is_err());
        }
    }
}